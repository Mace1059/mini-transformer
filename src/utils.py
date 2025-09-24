import torch
import torch.nn as nn
import torch.nn.functional as F
from src.ternary_packed import TernaryPackedLinear

# ========================================
#         Ternary MOE Experiment
# ========================================

class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, ternary_threshold=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ternary_threshold = ternary_threshold
        # master float params (trainable)
        self.weight_fp = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight_fp)

    def reset_parameters(self):
        # Random ternary initialization
        probs = torch.randint(0, 3, self.weight_fp.shape)  # values 0,1,2
        ternary = probs - 1  # shift to -1,0,1
        self.weight_fp.data = ternary.to(torch.int8)
    
    def forward(self, x):
        # Cast weights and bias to match the input dtype
        weight = self.weight_fp.to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


class TernarySwiGLU(nn.Module):
    def __init__(self, d_in, d_hidden, depth):
        super().__init__()
        self.depth = depth

        self.w1 = TernaryPackedLinear(d_in, d_hidden)
        self.w2 = TernaryPackedLinear(d_in, d_hidden)

        self.hidden_layers = nn.ModuleList()
        for _ in range(depth - 1):
            self.hidden_layers.append(TernaryPackedLinear(d_hidden, d_hidden))
            self.hidden_layers.append(nn.GELU())

        self.w3 = TernaryPackedLinear(d_hidden, d_in)

    def forward(self, x):
        h = F.silu(self.w1(x)) * self.w2(x)
        for layer in self.hidden_layers:
            h = layer(h)

        return self.w3(h)


class TernaryMoEFeedForward(nn.Module):
    def __init__(self, d_model, ffn_mult, n_experts, top_experts, dropout):
        super().__init__()

        # Expand the space for feed forward
        d_hidden = ffn_mult * d_model
        self.n_experts = n_experts
        self.top_experts = top_experts
        self.router = TopExpertsRouter(d_model, n_experts, top_experts=top_experts)
        # Each expert is a SwiGLU layer
        self.experts = nn.ModuleList(
            [TernarySwiGLU(d_model, d_hidden, depth=1) for _ in range(n_experts)]
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


class DeepTernaryFFN(nn.Module):
    def __init__(self, d_in, d_hidden, depth=2):
        super().__init__()
        layers = []
        for i in range(depth):
            in_dim = d_in if i == 0 else d_hidden
            out_dim = d_hidden
            layers.append(TernaryLinear(in_dim, out_dim))
            layers.append(nn.GELU())   # or SwiGLU-style activation
        # final projection back to input dim
        layers.append(TernaryLinear(d_hidden, d_in))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

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
    
# The router decides which k of the E experts should process each token
class TopExpertsRouter(nn.Module):

    def __init__(self, d_model, n_experts, top_experts):
        super().__init__()
        self.top_experts = top_experts
        self.w = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        # x: (B, T, D) or (N, D)
        if x.dim() == 3:
            N = x.shape[0] * x.shape[1]
            x = x.reshape(N, x.shape[-1])

        # Score every expert across the linear layer
        logits = self.w(x)
        
        # Softmax over to create probability distribution 
        probs = F.softmax(logits, dim=-1)

        # Gets top k experts to use and saves their value/id within E
        top_experts_val, top_experts_idx = probs.topk(k=self.top_experts, dim=-1)

        # Renormalize probabilities to sum to 1
        weights = top_experts_val / (top_experts_val.sum(dim=-1, keepdim=True) + 1e-9)
        # Save probs for moe load balancing loss
        return top_experts_idx, weights, probs # (N, k), (N, k), (N,E)
    
# Auxilary loss
def moe_load_balancing_loss(router_probs, top1_idx, n_experts):

    # Router_probs is the probability distribution over experts
    N = router_probs.shape[0]

    # Get the fraction of tokens routed to each expert (look at top1)
    with torch.no_grad():
        counts = torch.bincount(top1_idx, minlength=n_experts).float() / max(N, 1)
    
    # We need to know, on average, how much probability mass does the router give to each expert
    prob_mean = router_probs.mean(dim=0)

    # Calculate auxillary loss by n_expoerts * sum ( fraction of tokens routed to ith expert * average probability router of ith expert)
    aux = n_experts * torch.sum(counts * prob_mean)
    return aux
    
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

