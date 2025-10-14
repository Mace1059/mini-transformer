import torch

# LLaMA-Compatible Hyperparameters
block_size = 2048
n_embed = 2048
n_heads = 32


# Training Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
max_iters = 1000
eval_print_interval = 100
eval_iters = 5
learning_rate = 1e-3


# Transformer Block Hyperparameters
early_layers = 2
middle_layers = 2
late_layers = 2
hyperparameters = {
    'early': {
        'ffn_mult': 2,
        'dropout': 0.3,
        'lr_scale': 1.2,
    },
    'middle': {
        'ffn_mult': 4,
        'dropout': 0.4,
        'lr_scale': 1,
    },
    'late': {
        'ffn_mult': 8,
        'dropout': 0.5,
        'lr_scale': 0.8,
    }
}


# MoE Hyperparameters
n_experts = 8
top_k_experts = 2
moe_weight = 1e-2


# QLora Config
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
