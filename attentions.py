from utils import linear_init, recurrent_init
import torch
import torch.nn as nn
import torch.nn.functional as F

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