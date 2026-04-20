"""HAN-style AttentionWithContext (Yang et al., 2016) — PyTorch port."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionWithContext(nn.Module):
    """
    Self-attention with a learned context vector u.

    Input:  (batch, seq_len, hidden_dim)
    Output: (batch, hidden_dim)

    Computation:
        u_it = tanh(h_it @ W + b)      # project
        α_it = softmax(u_it @ u)        # score vs context vector
        s    = Σ α_it * h_it            # weighted sum
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.u = nn.Parameter(torch.empty(hidden_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.u.unsqueeze(0))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, H)
        uit = torch.tanh(self.W(x))           # (B, T, H)
        ait = (uit * self.u).sum(dim=-1)       # (B, T)

        if mask is not None:                   # mask: (B, T), True = padding
            ait = ait.masked_fill(mask, float("-inf"))

        alpha = F.softmax(ait, dim=-1).unsqueeze(-1)   # (B, T, 1)
        return (x * alpha).sum(dim=1)                  # (B, H)
