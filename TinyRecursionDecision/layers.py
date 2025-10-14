import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
  def __init__(self, dim, eps=1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))
  
  def forward(self, x):
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
    x = x / rms
    return x * self.weight
  
class SwiGLU(nn.Module):
  def __init__(self, dim, mlp_mult, dropout):
    super().__init__()
    hidden_dim = dim * mlp_mult
    self.fc1 = nn.Linear(dim, 2 * hidden_dim, bias=False)
    self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    A_B = self.fc1(x)             # (B,L,2H)
    A, B = A_B.chunk(2, dim=-1)   # (B,L,H) (B,L,H)
    x = F.silu(A) * B
    x = self.fc2(x)
    return self.dropout(x)
    
class TokenMixingSwiGLU(nn.Module):
  def __init__(self, dim, mlp_mult, dropout):
    super().__init__()
    hidden_dim = dim * mlp_mult
    self.fc1 = nn.Linear(dim, 2 * hidden_dim, bias=False)
    self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, L, D = x.shape
    # Transpose so sequence axis becomes “channel”
    x = x.transpose(1, 2).contiguous().view(B * D, L)  # (B*D, L)
    A_B = self.fc1(x)             # (B,L,2H)
    A, B = A_B.chunk(2, dim=-1)   # (B,L,H) (B,L,H)
    x = F.silu(A) * B
    x = self.fc2(x)
    x = self.dropout(x)
    x = x.view(B,D,L).transpose(1, 2).contiguous()
    return x

def apply_rope(q, k):
  B, H, L, Dh = q.shape
  device = q.device
  theta = 10000.0
  inv_freq = 1.0 / (theta ** (torch.arange(0, Dh, 2, device=device).float() / Dh))
  
  t = torch.arange(L, device=device).float()
  freqs = torch.einsum('l,d->ld', t, inv_freq)
  cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
  sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)

  def _rope(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    return torch.stack([xr1, xr2], dim=-1).flatten(-2)

  return _rope(q), _rope(k)

class SelfAttention(nn.Module):
  def __init__(self, dim, heads, rope=True, dropout=0):
    super().__init__()
    assert dim % heads == 0
    self.heads = heads
    self.head_dim = dim // heads
    self.rope = rope

    self.qkv = nn.Linear(dim, dim*3, bias=False)
    self.proj = nn.Linear(dim, dim, bias=False)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, L, D = x.shape
    
    # Build q, k and v
    qkv = self.qkv(x)
    q, k, v = qkv.chunk(3, dim=-1)

    def reshape(x):
      return x.view(B, L, self.heads, self.head_dim).transpose(1, 2)

    q, k, v = map(reshape, (q, k, v))

    # Apply rotary embeddings
    if self.rope:
      q, k = apply_rope(q, k)

    # Calculate attention
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    attn = attn.softmax(dim=-1)
    attn = self.dropout(attn)
    out = torch.matmul(attn, v)
    out = out.transpose(1, 2).contiguous().view(B, L, D)
    out = self.proj(out)
    return out

# Tiny Recursive Model Block
class TRMBlock(nn.Module):
  """
  One TRM layer:
  - If attention is on:
      x -> Norm -> Attention -> residual
  - Else:
      x -> Norm -> TokenMixing -> residual
  Then:
      x -> Norm -> SwiGLU FFN -> residual
  """
  def __init__(self, dim, seq_len, use_attn, heads, mlp_mult, token_mix_mult, rope=True, dropout=0):
    super().__init__()
    self.use_attn = use_attn
    if use_attn:
      self.norm_attn = RMSNorm(dim)
      self.attn = SelfAttention(dim, heads, rope, dropout)
    else:
      self.norm_mix = RMSNorm(dim)
      self.tmix = TokenMixingSwiGLU(seq_len, token_mix_mult, dropout)
    
    self.norm_ffn = RMSNorm(dim)
    self.ffn = SwiGLU(dim, mlp_mult, dropout)

  def forward(self, x):
    if self.use_attn:
      x = x + self.attn(self.norm_attn(x))
    else:
      x = x + self.tmix(self.norm_mix(x))
    
    x = x + self.ffn(self.norm_ffn(x))
    return x