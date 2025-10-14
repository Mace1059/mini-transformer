import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.attention import QLoraFusedMHA
from src.layers.moe import MoEFeedForward
from src.config import n_embed, moe_weight, hyperparameters, early_layers, middle_layers, late_layers 

# ----------------------------
# Transformer Block
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, ffn_mult, dropout):
        super().__init__()
        self.norm1 = nn.RMSNorm(n_embed)  
        self.attn = QLoraFusedMHA()
        self.norm2 = nn.RMSNorm(n_embed)  
        self.moe = MoEFeedForward(d_model=n_embed, ffn_mult=ffn_mult, dropout=dropout)

    def forward(self, x):
        # Residual network
        attn_out, _ = self.attn(self.norm1(x), None, use_cache=False) 
        x = x + attn_out
        ff_out, aux = self.moe(self.norm2(x))
        return x + ff_out, aux

# ----------------------------
# Model
# ----------------------------
class LLaMAHybridQLoRA(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList(
            [TransformerBlock(
                ffn_mult=hyperparameters['early']['ffn_mult'],
                dropout=hyperparameters['early']['dropout']) for _ in range(early_layers)]
            +
            [TransformerBlock(
                ffn_mult=hyperparameters['middle']['ffn_mult'],
                dropout=hyperparameters['middle']['dropout']) for _ in range(middle_layers)]
            +
            [TransformerBlock(
                ffn_mult=hyperparameters['late']['ffn_mult'],
                dropout=hyperparameters['late']['dropout']) for _ in range(late_layers)]
        )
        self.norm_final = nn.RMSNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
    
    def forward(self, idx, targets=None):
        x = self.token_embedding(idx)
        moe_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x)
            moe_loss += aux_loss

        x = self.norm_final(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) + moe_weight * moe_loss
        return logits, loss
    
    @torch.no_grad()

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
    
    def generate(    self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        device = input_ids.device
        B = input_ids.size(0)

        past_kvs = [None] * len(self.blocks)
        idx = input_ids
        for _ in range(max_new_tokens):
            if past_kvs[0] is not None:
                x = idx[:, -1:]
            # Use the full context for the first forward if no kv is cached
            else:
                x = idx
            
            h = self.token_embedding(x)
            new_past_kvs = []
            for block_idx, block in enumerate(self.blocks):
                h_norm = block.norm1(h)
                attn_out, new_kv = block.attn(
                    h_norm,
                    past_kv=past_kvs[block_idx],
                    use_cache=True
                )
                h = h + attn_out
                ff_out, _ = block.moe(block.norm2(h))
                h = h + ff_out
                new_past_kvs.append(new_kv)

            past_kvs = new_past_kvs

            # Final norm + logits
            h = self.final_norm(h)
            logits = self.lm_head(h[:, -1, :])  # (B, vocab)

            # Sampling using temperature
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat([idx, next_id], dim=1)

        return idx