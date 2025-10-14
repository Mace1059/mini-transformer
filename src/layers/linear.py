from src.config import n_embed, lora_r, lora_alpha, lora_dropout 
import torch
import torch.nn as nn
import bitsandbytes as bnb
import math


# Create a quantized linear layer like QLora
class QLoraLinear(nn.Module):
    def __init__(self, in_features, out_features, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout):
        super().__init__()
        self.linear = bnb.nn.Linear4bit(in_features, out_features, bias=False, quant_type='nf4', compress_statistics=True, compute_dtype=torch.bfloat16)
        # Freeze base weights
        self.linear.requires_grad_(False)

        self.r = r
        if r > 0:
            # Define Lora parameters for each Linear layer
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = lora_alpha / r
            self.dropout = nn.Dropout(lora_dropout)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = self.lora_B = None

    def forward(self, x):
        logits = self.linear(x)
        if self.r > 0:
            # Perform the QLora matrix multiplication of A and B
            # Apply dropout
            # Scale
            lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
            return logits + lora_out
        return logits
        
