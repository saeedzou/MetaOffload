import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math

torch.manual_seed(1)

def linear_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)
    return module

def recurrent_init(module):
    if isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
            if 'bias_ih' in name:
                param.requires_grad = False
    return module

class LuongAttention(nn.Module):
    def __init__(self, hidden_dim, bias=True):
        super(LuongAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = linear_init(nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias))

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch_size, 1, hidden_dim]
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        # compute attention score
        attn_score = torch.bmm(decoder_hidden, self.attention(encoder_outputs).transpose(1, 2))
        # compute attention weights
        attn_weights = F.softmax(attn_score, dim=-1)
        # compute attention context
        attn_context = torch.bmm(attn_weights, encoder_outputs)
        return attn_context, attn_weights

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # Add a LayerNorm layer
        self.layer_norm = nn.LayerNorm(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = x @ self.weight
        output = adj @ support
        if self.bias is not None:
            output += self.bias
        # Apply layer norm
        output = self.layer_norm(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid//2)
        self.gc2 = GraphConvolution(nhid//2, nhid)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x

class EncoderNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(EncoderNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = recurrent_init(nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True))

    def forward(self, embedded):
        output, hidden = self.lstm(embedded)
        return output, hidden

class BaseDecoderNetwork(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, device='cpu', is_attention=False):
        super(BaseDecoderNetwork, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.is_attention = is_attention
        # Use uniformly initialized embedding layer
        self.embedding = nn.Parameter(torch.FloatTensor(self.output_dim, self.hidden_dim).uniform_(-1.0, 1.0))
        self.lstm = recurrent_init(nn.LSTM(self.hidden_dim * 2 if is_attention else self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True))
        # output projection layer
        self.output_layer = linear_init(nn.Linear(self.hidden_dim, self.output_dim, bias=False))
        self.critic_head = linear_init(nn.Linear(self.output_dim, self.output_dim))
        # categorical distribution for pi
        self.categorical = Categorical
        if is_attention:
            self.attention = LuongAttention(self.hidden_dim, bias=False)
            self.concat = nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False)

    def forward(self, encoder_outputs, encoder_hidden, actions=None):
        batch_size = encoder_outputs.size(0)
        decoding_len = encoder_outputs.size(1)
        # initialize the input of the decoder with zeros
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        decoder_hidden = encoder_hidden

        logits = torch.zeros(batch_size, decoding_len, self.output_dim, device=self.device)
        qvalues = torch.zeros(batch_size, decoding_len, self.output_dim, device=self.device)
        decoder_action = torch.zeros(batch_size, decoding_len, device=self.device)
        context = torch.zeros(batch_size, 1, self.hidden_dim, device=self.device)
        # loop over the sequence length
        for t in range(decoding_len):
            # get action distribution and Q value
            logit, Q, decoder_hidden, context = self.forward_step(decoder_input, decoder_hidden, encoder_outputs, context)
            # sample an action from the action distribution
            if actions is None:
                action = self.categorical(logit).sample()
            else:
                action = actions[:, t].unsqueeze(1).long()
            # update the decoder input
            decoder_input = action
            # update the decoder outputs
            logits[:, t, :] = logit.squeeze(1)
            qvalues[:, t, :] = Q.squeeze(1)
            decoder_action[:, t] = action.squeeze(1)
        # V = sum over last dimension of Q * pi
        values = (qvalues * logits).sum(-1)
        return decoder_action, logits, values


    def forward_step(self, x, hidden, encoder_outputs, context):
        embedded = self.embedding[x]
        if self.is_attention:
            attn_context, _ = self.attention(context, encoder_outputs)
            embedded = torch.cat((embedded, attn_context), dim=-1)
        output, hidden = self.lstm(embedded, hidden)
        if self.is_attention:
            context, _ = self.attention(output, encoder_outputs)
            output = self.concat(torch.cat((output, context), dim=-1))
            output = nn.Tanh()(output)
        output = self.output_layer(output)
        pi = F.softmax(output, dim=-1)
        Q = self.critic_head(output)
        return pi, Q, hidden, context


class DecoderNetwork(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, device='cpu', is_attention=False):
        super(DecoderNetwork, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.is_attention = is_attention
        # Use uniformly initialized embedding layer
        self.embedding = nn.Parameter(torch.FloatTensor(self.output_dim, self.hidden_dim).uniform_(-1.0, 1.0))
        self.embedding_norm = nn.LayerNorm(self.hidden_dim)
        self.lstm = recurrent_init(nn.LSTM(self.hidden_dim * 2 if is_attention else self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True))
        self.lstm_norm = nn.LayerNorm(self.hidden_dim)
        self.actor_head = linear_init(nn.Linear(self.hidden_dim, self.output_dim))
        self.critic_head = linear_init(nn.Linear(self.hidden_dim, self.output_dim))
        # categorical distribution for pi
        self.categorical = Categorical
        if is_attention:
            self.attention = LuongAttention(self.hidden_dim)
            self.concat = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def forward(self, encoder_outputs, encoder_hidden, actions=None):
        batch_size = encoder_outputs.size(0)
        decoding_len = encoder_outputs.size(1)

        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        decoder_hidden = encoder_hidden

        logits = torch.zeros(batch_size, decoding_len, self.output_dim, device=self.device)
        qvalues = torch.zeros(batch_size, decoding_len, self.output_dim, device=self.device)
        decoder_action = torch.zeros(batch_size, decoding_len, device=self.device)
        context = torch.zeros(batch_size, 1, self.hidden_dim, device=self.device)
        # loop over the sequence length
        for t in range(decoding_len):
            # get action distribution and Q value
            logit, Q, decoder_hidden, context = self.forward_step(decoder_input, decoder_hidden, encoder_outputs, context)
            # sample an action from the action distribution
            if actions is None:
                action = self.categorical(logit).sample()
            else:
                action = actions[:, t].unsqueeze(1).long()
            decoder_input = action
            logits[:, t, :] = logit.squeeze(1)
            qvalues[:, t, :] = Q.squeeze(1)
            decoder_action[:, t] = action.squeeze(1)
        # V = sum over last dimension of Q * pi
        values = (qvalues * logits).sum(-1)
        return decoder_action, logits, values


    def forward_step(self, x, hidden, encoder_outputs, context):
        embedded = self.embedding[x]
        embedded = self.embedding_norm(embedded)
        if self.is_attention:
            attn_context, _ = self.attention(context, encoder_outputs)
            embedded = torch.cat((embedded, attn_context), dim=-1)
        output, hidden = self.lstm(embedded, hidden)
        output = self.lstm_norm(output)
        if self.is_attention:
            context, _ = self.attention(output, encoder_outputs)
            output = self.concat(torch.cat((output, context), dim=-1))
            output = nn.Tanh()(output)
        pi = F.softmax(self.actor_head(output), dim=-1)
        Q = self.critic_head(output)
        return pi, Q, hidden, context


class BaselineSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device='cuda', is_attention=False):
        super(BaselineSeq2Seq, self).__init__()
        self.embedding = linear_init(nn.Linear(input_dim, hidden_dim))
        self.encoder = EncoderNetwork(input_dim, hidden_dim, num_layers)
        self.decoder = BaseDecoderNetwork(output_dim, hidden_dim, num_layers, device, is_attention)

    def forward(self, x, decoder_inputs=None):
        x = self.embedding(x)
        encoder_outputs, encoder_hidden = self.encoder(x)
        actions, logits, values = self.decoder(encoder_outputs, encoder_hidden, decoder_inputs)
        return actions, logits, values
    
    def evaluate_actions(self, x, decoder_inputs=None):
        x = self.embedding(x)
        encoder_outputs, encoder_hidden = self.encoder(x)
        _, logits, values = self.decoder(encoder_outputs, encoder_hidden, decoder_inputs)
        return values, logits

class GraphSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device='cuda', is_attention=False):
        super(GraphSeq2Seq, self).__init__()
        self.graph_embedding = GCN(input_dim, hidden_dim)
        self.point_embedding = linear_init(nn.Linear(input_dim, hidden_dim))
        self.point_embedding_norm = nn.LayerNorm(hidden_dim)
        self.aggregation = linear_init(nn.Linear(hidden_dim * 2, hidden_dim))
        self.aggregation_norm = nn.LayerNorm(hidden_dim)
        self.encoder = EncoderNetwork(input_dim, hidden_dim, num_layers)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = DecoderNetwork(output_dim, hidden_dim, num_layers, device, is_attention)

    def forward(self, x, adj, decoder_inputs=None):
        x_g = self.graph_embedding(x, adj)
        x_p = self.point_embedding(x)
        x_p = self.point_embedding_norm(x_p)
        x = torch.cat([x_g, x_p], dim=-1)
        x = self.aggregation(x)
        x = self.aggregation_norm(x)
        encoder_outputs, encoder_hidden = self.encoder(x)
        encoder_outputs = self.encoder_norm(encoder_outputs)
        actions, logits, values = self.decoder(encoder_outputs, encoder_hidden, decoder_inputs)
        return actions, logits, values
    
    def evaluate_actions(self, x, adj, decoder_inputs=None):
        x_g = self.graph_embedding(x, adj)
        x_p = self.point_embedding(x)
        x_p = self.point_embedding_norm(x_p)
        x = torch.cat([x_g, x_p], dim=-1)
        x = self.aggregation(x)
        x = self.aggregation_norm(x)
        encoder_outputs, encoder_hidden = self.encoder(x)
        encoder_outputs = self.encoder_norm(encoder_outputs)
        _, logits, values = self.decoder(encoder_outputs, encoder_hidden, decoder_inputs)
        return values, logits