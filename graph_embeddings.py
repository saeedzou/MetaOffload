import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self, in_features, out_features, n_heads, concat=True, leaky_relu_slope=0.2, directed=False):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.leaky_relu_slope = leaky_relu_slope
        self.n_heads = n_heads
        self.directed = directed
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
        if not self.directed:
            adj = adj + adj.transpose(1, 2)
            adj = torch.where(adj > 0, torch.ones_like(adj), torch.zeros_like(adj))
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
    def __init__(self, in_features, out_features, n_heads, concat=True, leaky_relu_slope=0.2, share_weights=False, directed=True):
        super(GraphAttentionLayerV2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.leaky_relu_slope = leaky_relu_slope
        self.n_heads = n_heads
        self.directed = directed
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
        if not self.directed:
            adj = adj + adj.transpose(1, 2)
            adj = torch.where(adj > 0, torch.ones_like(adj), torch.zeros_like(adj))
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

    def forward(self, x, adj):
        x = self.gat_layer(x, adj)
        return x

class GraphConvolutionLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, directed=False):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.directed = directed
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
        if not self.directed:
            # make adj symmetric
            adj = adj + adj.transpose(1, 2)
            adj = torch.where(adj > 0, torch.ones_like(adj), torch.zeros_like(adj))
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
        self.gc1 = GraphConvolutionLayer(nfeat, nhid//2)
        self.gn1 = GraphNorm(nfeat)
        self.gc2 = GraphConvolutionLayer(nhid//2, nhid)
        self.gn2 = GraphNorm(nhid//2)

    def forward(self, x, adj):
        x = self.gn1(x)
        x = F.relu(self.gc1(x, adj))
        x = self.gn2(x)
        x = self.gc2(x, adj)
        return x
