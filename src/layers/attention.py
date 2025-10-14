import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, LlamaConfig
from src.layers.linear import QLoraLinear
from src.config import n_heads, n_embed, lora_r, lora_alpha, lora_dropout, block_size

class QLoraFusedMHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = n_heads
        self.d_hidden = n_embed // n_heads
        self.w_q = QLoraLinear(n_embed, n_embed, lora_r, lora_alpha, lora_dropout) # ()
        self.w_k = QLoraLinear(n_embed, self.d_hidden, lora_r, lora_alpha, lora_dropout)
        self.w_v = QLoraLinear(n_embed, self.d_hidden, lora_r, lora_alpha, lora_dropout)
        self.out_proj = QLoraLinear(n_embed, n_embed, lora_r, lora_alpha, lora_dropout) # ()

        # Set up rotary embedding
        config = LlamaConfig(
            hidden_size=n_embed,
            num_attention_heads=n_heads,
            max_position_embeddings=block_size,
        )
        self.rotary_emb = LlamaRotaryEmbedding(config)
    
    def forward(self, x, past_kv=None, use_cache=False):
        B,T,C = x.shape
        
        # Use a different q for each head
        q = self.w_q(x)
        q = q.view(B,T, self.n_heads, self.d_hidden).transpose(1, 2) # (B, h, T, d)

        # Use the same k, v across heads
        k = self.w_k(x).unsqueeze(1) # (B, 1, T, d)
        v = self.w_v(x).unsqueeze(1) # (B, 1, T, d)


        
        cos, sin = self.rotary_emb(q, seq_len=T)

        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV Caching for generation
        if past_kv is not None:
            past_k, past_v = past_kv
            # Add recent token to 
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        attn_out = F.scaled_dot_product_attention(q,k,v, is_causal=True)
        out = attn_out.transpose(1,2).contiguous().view(B,T,C)
        out = self.out_proj(out)

        # If using the cache, return k and v for next token
        if use_cache:
            return out, (k,v)
        else:
            return out, None


