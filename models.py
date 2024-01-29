import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from attentions import LuongAttention
from graph_embeddings import GATV2, GCN, CustomGraphLayer, GAT

class EncoderNetwork(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(EncoderNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)

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
    def __init__(self, output_dim, hidden_dim, num_layers, device='cpu', is_attention=False, arch='policy'):
        super(DecoderNetwork, self).__init__()
        assert arch in ['policy', 'value']
        self.arch = arch
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.is_attention = is_attention
        # Use uniformly initialized embedding layer
        self.embedding = nn.Parameter(torch.FloatTensor(self.output_dim, self.hidden_dim).uniform_(-1.0, 1.0))
        self.lstm = nn.LSTM(self.hidden_dim * 2 if is_attention else self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)
        if arch == 'policy':
            self.actor_head = nn.Linear(self.hidden_dim, self.output_dim)
        self.critic_head = nn.Linear(self.hidden_dim, 1)
        # categorical distribution for pi
        self.categorical = Categorical
        if is_attention:
            self.attention = LuongAttention(self.hidden_dim, bias=False)
            self.concat = nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False)

    def forward(self, encoder_outputs, encoder_hidden, actions=None):
        assert actions is not None or self.arch == 'policy'
        batch_size = encoder_outputs.size(0)
        decoding_len = encoder_outputs.size(1)

        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        context = encoder_outputs.mean(dim=1, keepdim=True)
        decoder_hidden = encoder_hidden

        logits = torch.zeros(batch_size, decoding_len, self.output_dim, device=self.device) if self.arch == 'policy' else None
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
            if self.arch == 'policy':
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
        value = self.critic_head(context)
        if self.arch == 'policy':
            output = self.actor_head(context)
            pi = F.softmax(output, dim=-1)
            return pi, value, hidden, context
        return None, value, hidden, context

class BaselineSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device='cuda', is_attention=False):
        super(BaselineSeq2Seq, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim, bias=False)
        self.encoder = EncoderNetwork(hidden_dim, num_layers)
        self.decoder = BaseDecoderNetwork(output_dim, hidden_dim, num_layers, device, is_attention)

    def forward(self, x, decoder_inputs=None):
        x = self.embedding(x)
        encoder_outputs, encoder_hidden = self.encoder(x)
        actions, logits, values = self.decoder(encoder_outputs, encoder_hidden, decoder_inputs)
        return actions, logits, values

class BaselineSeq2SeqDual(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device='cuda', is_attention=False, arch='policy'):
        super(BaselineSeq2SeqDual, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim, bias=False)
        self.encoder = EncoderNetwork(hidden_dim, num_layers)
        self.decoder = DecoderNetwork(output_dim, hidden_dim, num_layers, device, is_attention, arch)

    def forward(self, x, decoder_inputs=None):
        x = self.embedding(x)
        encoder_outputs, encoder_hidden = self.encoder(x)
        actions, logits, values = self.decoder(encoder_outputs, encoder_hidden, decoder_inputs)
        return actions, logits, values

class GraphSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device='cuda', is_attention=False, graph='gatv2'):
        super(GraphSeq2Seq, self).__init__()
        self.graph = graph
        if graph == 'gcn':
            self.graph_embedding = GCN(6, hidden_dim)
        elif graph == 'gat':
            self.graph_embedding = GAT(6, hidden_dim)
        elif graph == 'gatv2':
            self.graph_embedding = GATV2(6, hidden_dim)
        elif graph == 'custom':
            self.graph_embedding = CustomGraphLayer(6, hidden_dim)
        else:
            raise NotImplementedError(f'Graph embedding {graph} not implemented.')
        self.point_embedding = nn.Linear(12, hidden_dim, bias=False)
        self.embedding = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        if graph == 'custom':
            self.full_graph = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.encoder = EncoderNetwork(hidden_dim, num_layers)
        self.decoder = BaseDecoderNetwork(output_dim, hidden_dim, num_layers, device, is_attention)

    def forward(self, x, adj, decoder_inputs=None):
        x_g = x[:, :, :6]
        x_p = x[:, :, 6:]
        x_g = self.graph_embedding(x_g, adj)
        x_p = self.point_embedding(x_p)
        x = torch.cat((x_g, x_p), dim=-1)
        x = self.embedding(x)
        encoder_outputs, encoder_hidden = self.encoder(x)
        if self.graph == 'custom':
            x_fg = self.full_graph(x_g)
            x_fg = x_fg.permute(0, 2, 1)
            x_fg = F.max_pool1d(x_fg, kernel_size=x_fg.shape[-1])
            x_fg = x_fg.permute(0, 2, 1)
            encoder_outputs = encoder_outputs + x_fg
        actions, logits, values = self.decoder(encoder_outputs, encoder_hidden, decoder_inputs)
        return actions, logits, values
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def freeze_embedding(self):
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.graph_embedding.parameters():
            param.requires_grad = False
        for param in self.point_embedding.parameters():
            param.requires_grad = False
        for param in self.full_graph.parameters():
            param.requires_grad = False
    
    def freeze_decoder(self, part='lstm'):
        if part == 'lstm':
            for param in self.decoder.lstm.parameters():
                param.requires_grad = False
        elif part == 'attention':
            for param in self.decoder.attention.parameters():
                param.requires_grad = False
            for param in self.decoder.concat.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError(f'Part {part} not implemented.')

class GraphSeq2SeqDual(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device='cuda', is_attention=False, graph='gatv2', arch='policy'):
        super(GraphSeq2SeqDual, self).__init__()
        self.graph = graph
        if graph == 'gcn':
            self.graph_embedding = GCN(6, hidden_dim)
        elif graph == 'gat':
            self.graph_embedding = GAT(6, hidden_dim)
        elif graph == 'gatv2':
            self.graph_embedding = GATV2(6, hidden_dim)
        elif graph == 'custom':
            self.graph_embedding = CustomGraphLayer(6, hidden_dim)
        else:
            raise NotImplementedError(f'Graph embedding {graph} not implemented.')
        self.point_embedding = nn.Linear(12, hidden_dim, bias=False)
        self.embedding = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        if graph == 'custom':
            self.full_graph = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.encoder = EncoderNetwork(hidden_dim, num_layers)
        self.decoder = DecoderNetwork(output_dim, hidden_dim, num_layers, device, is_attention, arch)

    def forward(self, x, adj, decoder_inputs=None):
        x_g = x[:, :, :6]
        x_p = x[:, :, 6:]
        x_g = self.graph_embedding(x_g, adj)
        x_p = self.point_embedding(x_p)
        x = torch.cat((x_g, x_p), dim=-1)
        x = self.embedding(x)
        encoder_outputs, encoder_hidden = self.encoder(x)
        if self.graph == 'custom':
            x_fg = self.full_graph(x_g)
            x_fg = x_fg.permute(0, 2, 1)
            x_fg = F.max_pool1d(x_fg, kernel_size=x_fg.shape[-1])
            x_fg = x_fg.permute(0, 2, 1)
            encoder_outputs = encoder_outputs + x_fg
        actions, logits, values = self.decoder(encoder_outputs, encoder_hidden, decoder_inputs)
        return actions, logits, values


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


