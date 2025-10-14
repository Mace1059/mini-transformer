import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import moe_load_balancing_loss, TopExpertsRouter

# ========================================
#               Normal MOE
# ========================================


class SwiGLU(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden)
        self.w2 = nn.Linear(d_in, d_hidden)
        self.w3 = nn.Linear(d_hidden, d_in)

    def forward(self, x):
        # Project input x (d_in) onto d_hidden twice
        # Apply SiLU activation on one branch
        # Multiply the two elementwise (gating)
        # Project back onto the original dimension with w3

        return self.w3(F.silu(self.w1(x)) * self.w2(x))
    
class MoEFeedForward(nn.Module):
    def __init__(self, d_model, ffn_mult, n_experts, top_experts, dropout):
        super().__init__()

        # Expand the space for feed forward
        d_hidden = ffn_mult * d_model
        self.n_experts = n_experts
        self.top_experts = top_experts
        self.router = TopExpertsRouter(d_model, n_experts, top_experts=top_experts)
        # Each expert is a SwiGLU layer
        self.experts = nn.ModuleList(
            [SwiGLU(d_model, d_hidden) for _ in range(n_experts)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,D = x.shape
        N = B*T
        x_flat = x.reshape(N, D)

        top_experts_idx, weights, probs = self.router(x_flat) # (N,k), (N,k), (N,E)
        # Get top expert across all tokens as proxy for calculating auxilary loss
        top1_idx = top_experts_idx[:,0] # (N,)

        # Output buffer
        out = torch.zeros_like(x_flat)

        # For each of the top k experts chosen for each token
        for k in range(self.top_experts):
            idx_k = top_experts_idx[:, k] # (N,)
            w_k = weights[:, k].unsqueeze(-1) # (N,1)
            # Iterate through each expert
            for e in range(self.n_experts):
                # Create a mask for when an expert appears in top_k experts
                mask = (idx_k == e)
                # If any tokens use expert k...
                if mask.any():
                    # Pass the information to that expert (SwiGLU layer)
                    # Select only the rows where X-flat[mask] is true
                    y = self.experts[e](x_flat[mask])

                    # accumulate output across experts weighted by the expert
                    out[mask] += w_k[mask] * y

        out = out.reshape(B,T,D)
        out = self.dropout(out)

        aux_loss = moe_load_balancing_loss(probs, top1_idx, self.n_experts)
        return out, aux_loss