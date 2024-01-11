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
    return module

class LuongAttention(nn.Module):
    def __init__(self, hidden_dim, bias=False):
        super(LuongAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = linear_init(nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias))

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch_size, 1, hidden_dim]
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        # compute attention score
        attn_score = torch.bmm(decoder_hidden, self.attention(encoder_outputs).transpose(1, 2)) # [batch_size, 1, seq_len]
        # compute attention weights
        attn_weights = F.softmax(attn_score, dim=-1) # [batch_size, 1, seq_len]
        # compute attention context
        attn_context = torch.bmm(attn_weights, encoder_outputs) # [batch_size, 1, hidden_dim]
        return attn_context, attn_weights

class GraphNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(GraphNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.scale = nn.Parameter(torch.FloatTensor(num_features))
            self.shift = nn.Parameter(torch.FloatTensor(num_features))
        else:
            self.register_parameter('scale', None)
            self.register_parameter('shift', None)
        self.alpha = nn.Parameter(torch.FloatTensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.scale)
            nn.init.zeros_(self.shift)
        nn.init.ones_(self.alpha)

    def forward(self, x):
        x_shape = x.shape
        batch_size = x_shape[0]
        x = x.view(batch_size, self.num_features, -1)
        mean = x.mean(dim=-1, keepdim=True)
        mean_x2 = (x ** 2).mean(dim=-1, keepdim=True)
        var = mean_x2 - mean ** 2
        x_norm = (x - self.alpha.view(1, -1, 1) * mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(batch_size, self.num_features, -1)
        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)
        return x_norm.view(x_shape)

class CustomGraphLayer(nn.Module):
    def __init__(self, in_features, out_features, k_hops=3, activation=nn.ReLU()):
        super(CustomGraphLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k_hops = k_hops
        self.activation = activation

        self.embedding = nn.Linear(in_features, out_features, bias=False)
        # K linear layers
        self.layers = nn.ModuleList([nn.Linear(out_features * 2, out_features, bias=False) for _ in range(k_hops+1)])


    def forward(self, x, adj):
        # convert adj to binary matrix
        adj = torch.where(adj > 0, torch.ones_like(adj), torch.zeros_like(adj))
        x = self.embedding(x) # [batch_size, nodes, out_features]
        x_f = x
        x_b = x
        for k in range(self.k_hops):
            x_f_prime = (adj @ x_f) / (adj.sum(dim=-1, keepdim=True)) # [batch_size, nodes, out_features]
            x_f_prime = torch.concat((x_b, x_f_prime), dim=-1) # [batch_size, nodes, out_features * 2]
            x_f = self.activation(self.layers[k](x_f_prime)) # [batch_size, nodes, out_features]

            x_b_prime = (adj.transpose(1, 2) @ x_b) / (adj.transpose(1, 2).sum(dim=-1, keepdim=True)) # [batch_size, nodes, out_features]
            x_b_prime = torch.concat((x_b, x_b_prime), dim=-1) # [batch_size, nodes, out_features * 2]
            x_b = self.activation(self.layers[k](x_b_prime)) # [batch_size, nodes, out_features]
        
        x = torch.concat((x_f, x_b), dim=-1) # [batch_size, nodes, out_features * 2]
        x = self.layers[-1](x) # [batch_size, nodes, out_features]
        return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads, concat=True, leaky_relu_slope=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.leaky_relu_slope = leaky_relu_slope
        self.n_heads = n_heads
        if self.concat:
            assert self.out_features % n_heads == 0
            self.n_hidden = self.out_features // n_heads
        else:
            self.n_hidden = self.out_features

        self.W = nn.Parameter(torch.FloatTensor(size=(in_features, self.n_hidden * n_heads)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.FloatTensor(size=(2*self.n_hidden, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leaky_relu = nn.LeakyReLU(self.leaky_relu_slope)

    def forward(self, h, adj):
        # h.shape (B, N, in_feature)
        # adj.shape (B, N, N)
        # W.shape (in_feature, out_feature)
        Wh = (h @ self.W)
        # Wh.shape (B, N, out_feature)
        Wh = Wh.view(-1, self.n_heads, h.shape[1], self.n_hidden)
        # Wh.shape (B, n_heads, N, n_hidden)
        e = self._prepare_attentional_mechanism_input(Wh)
        # e.shape (B, n_heads, N, N)
        zero_vec = -9e15*torch.ones_like(e)
        # adj.shape (B, N, N)
        attention = torch.where(adj.unsqueeze(1) > 0, e, zero_vec)
        # attention.shape (B, n_heads, N, N)
        attention = F.softmax(attention, dim=-1)
        # attention.shape (B, n_heads, N, N)
        h_prime = torch.matmul(attention, Wh)
        # h_prime.shape (B, n_heads, N, n_hidden)
        if self.concat:
            # h_prime.shape (B, N, n_hidden * n_heads)
            h_prime = h_prime.view(-1, h.shape[1], self.out_features)
            return F.elu(h_prime)
        else:
            # take the average of the heads
            # h_prime.shape (B, N, n_hidden)
            h_prime = h_prime.mean(dim=1)
            return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (B, n_heads, N, n_hidden)
        # self.a.shape (2 * n_hidden, 1)
        # Wh1&2.shape (B, n_heads, N, 1)
        # e.shape (B, n_heads, N, N)
        Wh1 = Wh @ self.a[:self.n_hidden, :]
        Wh2 = Wh @ self.a[self.n_hidden:, :]
        e = Wh1 + Wh2.transpose(2, 3)
        return self.leaky_relu(e)
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nheads=8):
        super(GAT, self).__init__()
        self.gat_layer = GraphAttentionLayer(nfeat, nhid, nheads, concat=True)
        self.gn1 = GraphNorm(nfeat, affine=True)
        self.output_layer = GraphAttentionLayer(nhid, nhid, 1, concat=False)
        self.gn2 = GraphNorm(nhid, affine=True)

    def forward(self, x, adj):
        x = self.gn1(x)
        x = self.gat_layer(x, adj)
        x = self.gn2(x)
        x = self.output_layer(x, adj)
        return x

class GraphAttentionLayerV2(nn.Module):
    def __init__(self, in_features, out_features, n_heads, concat=True, leaky_relu_slope=0.2, share_weights=False):
        super(GraphAttentionLayerV2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.leaky_relu_slope = leaky_relu_slope
        self.n_heads = n_heads
        if self.concat:
            assert self.out_features % n_heads == 0
            self.n_hidden = self.out_features // n_heads
        else:
            self.n_hidden = self.out_features

        self.W_l = nn.Parameter(torch.FloatTensor(size=(in_features, self.n_hidden * n_heads)))
        nn.init.xavier_uniform_(self.W_l.data, gain=1.414)

        if share_weights:
            self.W_r = self.W_l
        else:
            self.W_r = nn.Parameter(torch.FloatTensor(size=(in_features, self.n_hidden * n_heads)))
            nn.init.xavier_uniform_(self.W_r.data, gain=1.414)
        
        self.a = nn.Parameter(torch.FloatTensor(size=(self.n_hidden, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.leaky_relu_slope)

    def forward(self, h, adj):
        # h.shape (B, N, in_feature)
        # adj.shape (B, N, N)
        # W.shape (in_feature, out_feature)
        Wh_l = (h @ self.W_l)
        Wh_r = (h @ self.W_r)
        # Wh.shape (B, N, out_feature)
        Wh_l = Wh_l.view(-1, self.n_heads, h.shape[1], self.n_hidden)
        Wh_r = Wh_r.view(-1, self.n_heads, h.shape[1], self.n_hidden)
        # Wh.shape (B, n_heads, N, n_hidden)
        # apply leaky relu before attention
        Wh_l = self.leaky_relu(Wh_l)
        Wh_r = self.leaky_relu(Wh_r)
        e = (Wh_l @ self.a) + (Wh_r @ self.a).transpose(2, 3)
        # e.shape (B, n_heads, N, N)
        zero_vec = -9e15*torch.ones_like(e)
        # adj.shape (B, N, N)
        attention = torch.where(adj.unsqueeze(1) > 0, e, zero_vec)
        # attention.shape (B, n_heads, N, N)
        attention = F.softmax(attention, dim=-1)
        # attention.shape (B, n_heads, N, N)
        h_prime = torch.matmul(attention, Wh_r)
        # h_prime.shape (B, n_heads, N, n_hidden)
        if self.concat:
            # h_prime.shape (B, N, n_hidden * n_heads)
            h_prime = h_prime.view(-1, h.shape[1], self.out_features)
            return F.elu(h_prime)
        else:
            # take the average of the heads
            # h_prime.shape (B, N, n_hidden)
            h_prime = h_prime.mean(dim=1)
            return F.elu(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GATV2(nn.Module):
    def __init__(self, nfeat, nhid, nheads=8):
        super(GATV2, self).__init__()
        self.gat_layer = GraphAttentionLayerV2(nfeat, nhid//2, nheads, concat=True)
        self.gn1 = GraphNorm(nfeat, affine=True)
        self.output_layer = GraphAttentionLayerV2(nhid//2, nhid, 1, concat=False)
        self.gn2 = GraphNorm(nhid//2, affine=True)

    def forward(self, x, adj):
        x = self.gn1(x)
        x = self.gat_layer(x, adj)
        x = self.gn2(x)
        x = self.output_layer(x, adj)
        return x

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
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid//2)
        self.gn1 = GraphNorm(nfeat)
        self.gc2 = GraphConvolution(nhid//2, nhid)
        self.gn2 = GraphNorm(nhid//2)

    def forward(self, x, adj):
        x = self.gn1(x)
        x = F.relu(self.gc1(x, adj))
        x = self.gn2(x)
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
        self.embedding_norm = nn.LayerNorm(self.hidden_dim)
        self.lstm = recurrent_init(nn.LSTM(self.hidden_dim * 2 if is_attention else self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True))
        self.lstm_norm = nn.LayerNorm(self.hidden_dim)
        self.actor_head = linear_init(nn.Linear(self.hidden_dim, self.output_dim))
        self.critic_head = linear_init(nn.Linear(self.hidden_dim, 1))
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
        embedded = self.embedding_norm(embedded) # [batch_size, 1, hidden_dim]
        if self.is_attention:
            embedded = torch.cat((embedded, context), dim=-1) # [batch_size, 1, hidden_dim * 2]
        output, hidden = self.lstm(embedded, hidden)
        output = self.lstm_norm(output)
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
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device='cuda', is_attention=False):
        super(GraphSeq2Seq, self).__init__()
        self.gat = GATV2(input_dim, hidden_dim)
        self.gcn = GCN(input_dim, hidden_dim)
        self.encoder = EncoderNetwork(input_dim, hidden_dim, num_layers)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = DecoderNetwork(output_dim, hidden_dim, num_layers, device, is_attention)

    def forward(self, x, adj, decoder_inputs=None):
        x1 = self.gat(x, adj)
        x2 = self.gcn(x, adj)
        x = x1 + x2
        encoder_outputs, encoder_hidden = self.encoder(x)
        encoder_outputs = self.encoder_norm(encoder_outputs)
        actions, logits, values = self.decoder(encoder_outputs, encoder_hidden, decoder_inputs)
        return actions, logits, values
    
    def evaluate_actions(self, x, adj, decoder_inputs=None):
        x1 = self.gat(x, adj)
        x2 = self.gcn(x, adj)
        x = x1 + x2
        encoder_outputs, encoder_hidden = self.encoder(x)
        encoder_outputs = self.encoder_norm(encoder_outputs)
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

