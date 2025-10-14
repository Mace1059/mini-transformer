import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from src.dataset import get_batch
from src.config import eval_iters, learning_rate, device, max_iters, eval_print_interval, moe_weight

@torch.no_grad()
def estimate_loss(model):
    """Estimates train and validation loss."""
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss, _ = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model(model, optimizer=None, save_path="checkpoints/final_model.pt"):
    model = model.to(device)
    model.train()

    if optimizer is None:
        # Only train LoRA adapter weights
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("Training initiated")
    for step in tqdm(range(max_iters), desc="Training"):
        xb, yb = get_batch('train')
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=torch.float16):
            _, loss = model(xb, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()



        if iter % eval_print_interval == 0:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    # ----------------------------
    # Save Model & Loss
    # ----------------------------
    torch.save(model.state_dict(), save_path)
    final_losses = estimate_loss(model)

    print("Final model and losses saved.")
    print(f"Final losses → train: {final_losses['train']:.4f}, val: {final_losses['val']:.4f}")
