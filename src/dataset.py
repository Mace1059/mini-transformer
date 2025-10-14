import torch
from transformers import AutoTokenizer
from src.config import block_size, batch_size, device


# ----------------------------
# Initialize Tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
tokenizer.pad_token = tokenizer.eos_token

# ----------------------------
# Data Loading
# ----------------------------
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]

# ----------------------------
# Training/Test Split
# ----------------------------
split = int(len(tokens) * 0.9)
train_data = tokens[:split]
val_data = tokens[split:]


def get_batch(split):
    """Returns a batch of training or validation data."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
