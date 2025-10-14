import torch
import torch.nn as nn
import torch.nn.functional as F

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
    