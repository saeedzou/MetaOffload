import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import linear_init, recurrent_init
from attentions import LuongAttention
from graph_embeddings import GATV2, GCN, CustomGraphLayer, GAT

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
        self.lstm = nn.LSTM(self.hidden_dim * 2 if is_attention else self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)
        # output projection layer
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        self.critic_head = nn.Linear(self.output_dim, self.output_dim)
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
        context = encoder_outputs.mean(dim=1, keepdim=True)
        decoder_hidden = encoder_hidden

        logits = torch.zeros(batch_size, decoding_len, self.output_dim, device=self.device)
        qvalues = torch.zeros(batch_size, decoding_len, self.output_dim, device=self.device)
        decoder_action = torch.zeros(batch_size, decoding_len, device=self.device)
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
            embedded = torch.cat((embedded, context), dim=-1)
        output, hidden = self.lstm(embedded, hidden)
        if self.is_attention:
            context, _ = self.attention(output, encoder_outputs)
            output = self.concat(torch.cat((output, context), dim=-1))
            context = nn.Tanh()(output)
        output = self.output_layer(context)
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
        self.lstm = nn.LSTM(self.hidden_dim * 2 if is_attention else self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.actor_head = nn.Linear(self.hidden_dim, self.output_dim)
        self.critic_head = nn.Linear(self.hidden_dim, 1)
        # categorical distribution for pi
        self.categorical = Categorical
        if is_attention:
            self.attention = LuongAttention(self.hidden_dim, bias=False)
            self.concat = nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False)

    def forward(self, encoder_outputs, encoder_hidden, actions=None):
        batch_size = encoder_outputs.size(0)
        decoding_len = encoder_outputs.size(1)

        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        context = encoder_outputs.mean(dim=1, keepdim=True)
        decoder_hidden = encoder_hidden

        logits = torch.zeros(batch_size, decoding_len, self.output_dim, device=self.device)
        values = torch.zeros(batch_size, decoding_len, 1, device=self.device)
        decoder_action = torch.zeros(batch_size, decoding_len, device=self.device)
        # loop over the sequence length
        for t in range(decoding_len):
            # get action distribution and Q value
            logit, value, decoder_hidden, context = self.forward_step(decoder_input, decoder_hidden, encoder_outputs, context)
            # sample an action from the action distribution
            if actions is None:
                action = self.categorical(logit).sample()
            else:
                action = actions[:, t].unsqueeze(1).long()
            decoder_input = action
            logits[:, t, :] = logit.squeeze(1)
            values[:, t, :] = value.squeeze(1)
            decoder_action[:, t] = action.squeeze(1)
        values = values.squeeze(-1)
        return decoder_action, logits, values

    def forward_step(self, x, hidden, encoder_outputs, context):
        embedded = self.embedding[x] # [batch_size, 1, hidden_dim]
        if self.is_attention:
            embedded = torch.cat((embedded, context), dim=-1) # [batch_size, 1, hidden_dim * 2]
        output, hidden = self.lstm(embedded, hidden)
        if self.is_attention:
            context, _ = self.attention(output, encoder_outputs)
            output = self.concat(torch.cat((output, context), dim=-1))
            context = nn.Tanh()(output)
        output = self.actor_head(context)
        pi = F.softmax(output, dim=-1)
        value = self.critic_head(context)
        return pi, value, hidden, context

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
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device='cuda', is_attention=False, graph='gatv2'):
        super(GraphSeq2Seq, self).__init__()
        if graph == 'gcn':
            self.graph_embedding = GCN(input_dim, hidden_dim)
        elif graph == 'gat':
            self.graph_embedding = GAT(input_dim, hidden_dim)
        elif graph == 'gatv2':
            self.graph_embedding = GATV2(input_dim, hidden_dim)
        elif graph == 'custom':
            self.graph_embedding = CustomGraphLayer(input_dim, hidden_dim)
        else:
            raise NotImplementedError(f'Graph embedding {graph} not implemented.')
        self.encoder = EncoderNetwork(input_dim, hidden_dim, num_layers)
        self.decoder = BaseDecoderNetwork(output_dim, hidden_dim, num_layers, device, is_attention)

    def forward(self, x, adj, decoder_inputs=None):
        x = self.graph_embedding(x, adj)
        encoder_outputs, encoder_hidden = self.encoder(x)
        actions, logits, values = self.decoder(encoder_outputs, encoder_hidden, decoder_inputs)
        return actions, logits, values
    
    def evaluate_actions(self, x, adj, decoder_inputs=None):
        x = self.graph_embedding(x, adj)
        encoder_outputs, encoder_hidden = self.encoder(x)
        _, logits, values = self.decoder(encoder_outputs, encoder_hidden, decoder_inputs)
        return values, logits

class Graph2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device='cuda', is_attention=False):
        super(Graph2Seq, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.node_embedding = CustomGraphLayer(input_dim, hidden_dim)
        self.graph_embedding = nn.Linear(hidden_dim, hidden_dim * num_layers * 2, bias=False)
        self.decoder = DecoderNetwork(output_dim, hidden_dim, num_layers, device, is_attention)

    def forward(self, x, adj, decoder_inputs=None):
        # Node embedding
        x = self.node_embedding(x, adj)  # [N, nodes, hidden_dim]
        # Graph embedding
        h = self.graph_embedding(x)  # [N, nodes, hidden_dim * num_layers]
        # Max pooling over nodes dimension
        h = h.transpose(1, 2)  # Transpose to get [N, hidden_dim * num_layers, nodes]
        h = F.max_pool1d(h, kernel_size=h.shape[-1])  # Max pooling over nodes
        h = h.squeeze(-1)  # Remove the last dimension, get [N, hidden_dim * num_layers * 2]
        # Reshape to a 2 element tuple of (num_layers, N, hidden_dim)
        h1 = h[:, :self.hidden_dim*self.num_layers].reshape(self.num_layers, -1, self.hidden_dim)
        h2 = h[:, self.hidden_dim*self.num_layers:].reshape(self.num_layers, -1, self.hidden_dim)
        h = (h1, h2)
        # Decoder
        actions, logits, values = self.decoder(x, h, decoder_inputs)
        return actions, logits, values
    
    def forward(self, x, adj, decoder_inputs=None):
        # Node embedding
        x = self.node_embedding(x, adj)  # [N, nodes, hidden_dim]
        # Graph embedding
        h = self.graph_embedding(x)  # [N, nodes, hidden_dim * num_layers]
        # Max pooling over nodes dimension
        h = h.transpose(1, 2)  # Transpose to get [N, hidden_dim * num_layers, nodes]
        h = F.max_pool1d(h, kernel_size=h.shape[-1])  # Max pooling over nodes
        h = h.squeeze(-1)  # Remove the last dimension, get [N, hidden_dim * num_layers * 2]
        # Reshape to a 2 element tuple of (num_layers, N, hidden_dim)
        h1 = h[:, :self.hidden_dim*self.num_layers].reshape(self.num_layers, -1, self.hidden_dim)
        h2 = h[:, self.hidden_dim*self.num_layers:].reshape(self.num_layers, -1, self.hidden_dim)
        h = (h1, h2)
        # Decoder
        _, logits, values = self.decoder(x, h, decoder_inputs)
        return values, logits

