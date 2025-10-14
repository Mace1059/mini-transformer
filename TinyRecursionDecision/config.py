from dataclasses import dataclass

@dataclass
class ModelConfig:
  embed_dim = 256
  num_layers = 2    # number of TRMBlock Layers
  z_len = 16        # number of latent tokens
  use_attn = True
  attn_heads = 4
  rope = True
  max_x_len = 256   # size of data for training
  num_classes = 4

  # Recursive hyperparameters
  n_latent_steps = 6    # refine within recursion
  n_outer_loops = 3     # number of times to rerun recursion (deep supervision steps), all but last are no_grad() to save memory

  # FFN/Mixer sizes to expand in SwiGLU FFL
  mlp_mult = 2
  token_mix_mult = 2
  dropout = 0

@dataclass
class DataConfig:
  vocab_size = 4096
  pad_token_id = 0
  batch_size = 32
  num_workers = 2

@dataclass
class TrainingConfig:
  epochs = 20
  lr = 3e-4
  w_decay = 0.01 #L2 regularization term
  grad_clip = 1
  warmup_steps = 100
  max_steps = 10000
  halt_loss_weight = 0.1    # Encourages model to halt when answer is found

@dataclass
class MiscConfig:
  seed = 42
  device = 'cuda'
  log_every = 50
  save_dir = './checkpoints'