# Description: This class implements the scaled dot product attention in the way outlined 
# in the paper "Attention is all you need" by Vaswani et al.

# **THIS FILE IS MOSTLY EDUCATIONAL**, as the PyTorch library already has an implementation
# of this attention mechanism (torch.nn.MultiheadAttention) likely with better performance
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        # Reshape to perform multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.depth).float()
        )

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attention, V)

        # Reshape to concatenate multi-head attention output
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.depth)
        )

        # Linear projection
        output = self.output_linear(context)

        return output
