import torch
import torch.nn as nn
import torch.nn.functional as F
from src.ternary_packed import TernaryPackedLinear
from utils import moe_load_balancing_loss, TopExpertsRouter

class TernarySwiGLU(nn.Module):
    def __init__(self, d_in, d_hidden, depth):
        super().__init__()
        self.w1 = TernaryPackedLinear(d_in, d_hidden)
        self.w2 = TernaryPackedLinear(d_in, d_hidden)

        layers = []
        for _ in range(depth - 1):
            layers.append(TernaryPackedLinear(d_hidden, d_hidden))
            layers.append(nn.GELU())
        self.hidden_layers = nn.Sequential(*layers)

        self.w3 = TernaryPackedLinear(d_hidden, d_in)

    def forward(self, x):
        h = F.silu(self.w1(x)) * self.w2(x)
        h = self.hidden_layers(h)
        return self.w3(h)


class TernaryMoEFeedForward(nn.Module):
    def __init__(self, d_model, ffn_mult, n_experts, top_experts, dropout):
        super().__init__()
        d_hidden = ffn_mult * d_model
        self.n_experts = n_experts
        self.top_experts = top_experts
        self.router = TopExpertsRouter(d_model, n_experts, top_experts=top_experts)
        self.experts = nn.ModuleList(
            [TernarySwiGLU(d_model, d_hidden, depth=1) for _ in range(n_experts)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)

        top_experts_idx, weights, probs = self.router(x_flat)
        top1_idx = top_experts_idx[:, 0]

        out = torch.zeros_like(x_flat)
        for k in range(self.top_experts):
            idx_k = top_experts_idx[:, k]
            w_k = weights[:, k].unsqueeze(-1)
            for e in range(self.n_experts):
                mask = (idx_k == e)
                if mask.any():
                    y = self.experts[e](x_flat[mask])
                    out[mask] += w_k[mask] * y

        out = out.view(B, T, D)
        out = self.dropout(out)
        aux_loss = moe_load_balancing_loss(probs, top1_idx, self.n_experts)
        return out, aux_loss
