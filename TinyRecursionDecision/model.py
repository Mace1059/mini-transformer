from typing import Tuple, Optional
import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import TRMBlock, RMSNorm

# TRM applied to classification/decision within a node
class TRMClassifier(nn.Module):
  def __init__(self, cfg: ModelConfig, num_classes):
    super().__init__()
    # Store config 
    self.dim = cfg.embed_dim
    self.z_len = cfg.z_len
    self.max_x_len = cfg.max_x_len
    self.num_classes = cfg.num_classes
    self.n_latent_steps = cfg.n_latent_steps
    self.n_outer_loops = cfg.n_outer_loops

    # Role embeddings 
    # 0 = latent update (x + y + z)
    # 1 = answer update (y + z)
    self.role_embed = nn.Embedding(2, self.dim)

    # Learnable initial states
    self.y_init = nn.Parameter(torch.zeros(1, 1, self.dim))
    self.z_init = nn.Parameter(torch.zeros(1, self.z_len, self.dim))

    # TRM Trunk Network
    self.latent_seq_len = 1 + 1 + self.z_len + self.max_x_len  # [role, y, z..., x...]
    self.answer_seq_len = 1 + 1 + self.z_len

    self.blocks_latent = nn.ModuleList([
      TRMBlock(self.dim, self.latent_seq_len, use_attn=cfg.use_attn, heads=cfg.attn_heads, rope=cfg.rope, mlp_mult=cfg.mlp_mult, token_mix_mult=cfg.token_mix_mult, dropout=cfg.dropout)
      for _ in range(cfg.num_layers)
    ])

    self.blocks_answer= nn.ModuleList([
      TRMBlock(self.dim, self.answer_seq_len, use_attn=cfg.use_attn, heads=cfg.attn_heads, rope=cfg.rope, mlp_mult=cfg.mlp_mult, token_mix_mult=cfg.token_mix_mult, dropout=cfg.dropout)
      for _ in range(cfg.num_layers)
    ])

    self.norm_y = RMSNorm(self.dim)
    self.class_head = nn.Linear(self.dim, self.num_classes, bias=False)
    self.halt_head = nn.Linear(self.dim, 1, bias=False)


  # Packing functions to define what goes into trunk
  # Latent: x + y + z
  def _pack_latent(self, x_tokens, y_tok, z_tokens):
    B = x_tokens.size(0)
    role = self.role_embed.weight[0].unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
    return torch.cat([role, y_tok, z_tokens, x_tokens], dim=1)

  def _unpack_latent(self, seq):
    y_tok = seq[:, 1:2, :]
    z_tokens = seq[:, 2:2 + self.z_len, :]
    return y_tok, z_tokens

  # Answer: y + z 
  def _pack_answer(self, y_tok, z_tokens):
    B = y_tok.size(0)
    role = self.role_embed.weight[1].unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
    return torch.cat([role, y_tok, z_tokens], dim=1)

  def _unpack_answer(self, seq):
    y_tok = seq[:, 1:2, :]
    z_tokens = seq[:, 2:2 + self.z_len, :]
    return y_tok, z_tokens


  # Latent recursion updates z and partially y
  def _trunk_latent(self, x_tokens, y_tok, z_tokens):
    h = self._pack_latent(x_tokens, y_tok, z_tokens)
    for blk in self.blocks_latent:
      h = blk(h)
    return self._unpack_latent(h)

  # Answer recursion updates y
  def _trunk_answer(self, y_tok, z_tokens):
    h = self._pack_answer(y_tok, z_tokens)
    for blk in self.blocks_answer:
      h = blk(h)
    return self._unpack_answer(h)  


  @torch.no_grad()
  def init_state(self, batch_size, device=None):
    device = device or self.y_init.device
    y0 = self.y_init.expand(batch_size, 1, self.dim).to(device)
    z0 = self.z_init.expand(batch_size, self.z_len, self.dim).to(device)
    return y0.clone(), z0.clone()
  



  def forward(self, x_tokens, y=None, z=None):
    B = x_tokens.size(0)
    device = x_tokens.device

    if y is None or z is None:
      y0, z0 = self.init_state(B, device)
      if y is None: y = y0
      if z is None: z = z0

    with torch.no_grad():
      # T-1 outer loops, disable gradient computation, still refine reasoning state
      for _ in range(self.n_outer_loops - 1):
        # Latent recursion loop
        for _ in range(self.n_latent_steps):
          y, z = self._trunk_latent(x_tokens, y, z)
        # Answer refinement
        y, z = self._trunk_answer(y, z)

    # T-th outer loop where gradient flows
    for _ in range(self.n_latent_steps):
      y, z = self._trunk_latent(x_tokens, y, z)
    y, z = self._trunk_answer(y, z)

    # now, y has the answer
    y_norm = self.norm_y(y).squeeze(1)
    class_logits = self.class_head(y_norm)
    halt_logits = self.halt_head(y_norm)

    return (y.detach(), z.detach()), class_logits, halt_logits