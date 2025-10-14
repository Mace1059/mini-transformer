# main.py
from src.train import train_model
from src.model import LLaMAHybridQLoRA
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
vocab_size = tokenizer.vocab_size

model = LLaMAHybridQLoRA(vocab_size)
train_model(model)