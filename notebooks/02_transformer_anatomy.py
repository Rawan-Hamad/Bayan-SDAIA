"""Lab 2 starter notebook-as-script.
Complete the marked sections, verify numerical equivalence, inspect parameter
accounting, causal masking, attention heads and pad-attention leakage.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1)

    scores = scores / math.sqrt(q.size(-1))

    if mask is not None:
        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

    weights = F.softmax(scores, dim=-1)

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
        batch_size = q.size(0)

        # Linear projections
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        # Split into heads
        q = q.view(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim
        )

        k = k.view(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim
        )

        v = v.view(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim
        )

        # Move heads dimension
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        x = attention(q, k, v, mask)

        # Combine heads
        x = x.transpose(1, 2)
        x = x.contiguous().view(
            batch_size,
            -1,
            self.d_model
        )

        # Final projection
        x = self.out_linear(x)

        return x


def main():
    torch.manual_seed(0)

    # =========================================================
    # 1. Numerical equivalence
    # =========================================================

    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)

    ours = attention(q, k, v)

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.size(-1))

    reference_weights = F.softmax(scores, dim=-1)
    reference = reference_weights @ v

    print("Numerical equivalence:")
    print(torch.allclose(
        ours,
        reference,
        atol=1e-6
    ))

    # =========================================================
    # 2. Inspect attention weight matrix
    # =========================================================

    print("\nAttention weights:")
    print(reference_weights)

    print("\nRow sums:")
    print(reference_weights.sum(dim=-1))

    # =========================================================
    # 3. Exercise Multi-Head Attention
    # =========================================================

    mha = MultiHeadAttention(
        d_model=786,
        num_heads=6
    )

    q = torch.randn(1, 4, 786)
    k = torch.randn(1, 4, 786)
    v = torch.randn(1, 4, 786)

    output = mha(q, k, v)

    print("\nMulti-Head Attention output shape:")
    print(output.shape)

    # =========================================================
    # 4. Causal mask
    # =========================================================

    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)

    causal_mask = torch.tril(
        torch.ones(4, 4)
    )

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.size(-1))

    masked_scores = scores.masked_fill(
        causal_mask == 0,
        float("-inf")
    )

    masked_weights = F.softmax(
        masked_scores,
        dim=-1
    )

    print("\nCausal mask:")
    print(causal_mask)

    print("\nMasked attention weights:")
    print(masked_weights)

    future_attention = masked_weights[0].triu(
        diagonal=1
    )

    print("\nFuture positions:")
    print(future_attention)

    print("\nFuture attention is zero:")
    print(torch.allclose(
        future_attention,
        torch.zeros_like(future_attention),
        atol=1e-6
    ))

    # =========================================================
    # 5. Attention-map diagnostics + pad leak
    # =========================================================

    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)

    tokens = [
        "[CLS]",
        "token1",
        "token2",
        "[PAD]"
    ]

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.size(-1))

    # ---------------------------------------------------------
    # Without padding mask
    # ---------------------------------------------------------

    weights_no_mask = F.softmax(
        scores,
        dim=-1
    )

    pad_mass_no_mask = weights_no_mask[..., 3].sum()

    print("\nTokens:")
    print(tokens)

    print("\nAttention without padding mask:")
    print(weights_no_mask)

    print("\nPad attention mass without mask:")
    print(pad_mass_no_mask.item())

    # ---------------------------------------------------------
    # With correct padding mask
    # ---------------------------------------------------------

    pad_mask = torch.tensor([
        [1, 1, 1, 0]
    ])

    masked_scores = scores.masked_fill(
        pad_mask[:, None, :] == 0,
        float("-inf")
    )

    weights_with_mask = F.softmax(
        masked_scores,
        dim=-1
    )

    pad_mass_with_mask = weights_with_mask[..., 3].sum()

    print("\nAttention with padding mask:")
    print(weights_with_mask)

    print("\nPad attention mass with mask:")
    print(pad_mass_with_mask.item())

    # ---------------------------------------------------------
    # Compare pad leakage
    # ---------------------------------------------------------

    print("\nPad mass comparison:")
    print("Without mask:", pad_mass_no_mask.item())
    print("With mask:", pad_mass_with_mask.item())

    print("\nPad leak removed:")
    print(torch.allclose(
        pad_mass_with_mask,
        torch.tensor(0.0),
        atol=1e-6
    ))


if __name__ == "__main__":
    main()