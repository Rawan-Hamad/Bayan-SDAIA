"""Lab 2 starter: scaled dot-product attention and multi-head attention."""
import math
import torch.nn as nn


def attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = scores.softmax(dim=-1)
    output = weights @ v
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=786, num_heads=6):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        # Linear projections
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        # Split into heads
        batch_size = q.size(0)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)

        # Move heads dimension
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        x = attention(q, k, v, mask)

        # Combine heads
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, -1, self.d_model)

        # Final linear layer
        x = self.out_linear(x)

        return x
