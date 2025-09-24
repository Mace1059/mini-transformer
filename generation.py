import torch
from src.improvements import BigramLanguageModel, decode, device

# Recreate model
model = BigramLanguageModel().to(device)

# Load trained weights
state_dict = torch.load("checkpoints/final_model.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Generate text
context = torch.zeros((1,1), dtype=torch.long, device=device)
output = model.generate(context, max_new_tokens=500)
print(decode(output[0].tolist()))
