import torch
import torch.nn as nn
import torch.nn.functional as F
from src.moe_layers import MoEFeedForward  # assumes your MoE implementation is correct

# ----------------------------
# Training Hyperparameters
# ----------------------------
batch_size = 1024
block_size = 64
max_iters = 1000
eval_print_interval = 100
eval_iters = 5
learning_rate = 2e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 128
early_layers = 1
middle_layers = 1
late_layers = 1
moe_weight = 1e-2


hyperparameters = {
    'early': {
        'ffn_mult': 2,
        'dropout': 0.3,
        'lr_scale': 1.2,
        'n_heads': 8,
    },
    'middle': {
        'ffn_mult': 4,
        'dropout': 0.4,
        'lr_scale': 1,
        'n_heads': 4,
    },
    'late': {
        'ffn_mult': 8,
        'dropout': 0.5,
        'lr_scale': 0.8,
        'n_heads': 4
    }
}


torch.manual_seed(1337)

# ----------------------------
# Data Loading
# ----------------------------
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

train_data = train_data.pin_memory()
val_data   = val_data.pin_memory()

def get_batch(split):
    """Returns a batch of training or validation data."""
    src = train_data if split == 'train' else val_data
    i = torch.randint(0, len(src) - block_size - 1, (batch_size,))
    idx = i.unsqueeze(1) + torch.arange(block_size)
    x = src[idx].pin_memory().to(device, non_blocking=True)
    y = src[idx + 1].pin_memory().to(device, non_blocking=True)
    return x, y


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

# ----------------------------
# Model Components
# ----------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # FlashAttention (PyTorch 2.0+)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out


class FusedMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads
        self.d = embed_dim // num_heads
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):                       # x: [B, T, C]
        B, T, C = x.shape
        qkv = self.in_proj(x)                   # [B, T, 3C]
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape to [B, h, T, d]
        q = q.view(B, T, self.h, self.d).transpose(1, 2).contiguous()
        k = k.view(B, T, self.h, self.d).transpose(1, 2).contiguous()
        v = v.view(B, T, self.h, self.d).transpose(1, 2).contiguous()
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # one fast call
        y = y.transpose(1, 2).contiguous().view(B, T, C)             # [B, T, C]
        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head, ffn_mult=4, dropout=0.1, use_moe=True, n_experts=8, top_experts=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = FusedMHA(n_embed, n_head)
        self.ln2 = nn.LayerNorm(n_embed)
        self.use_moe = use_moe

        if use_moe:
            self.ff = MoEFeedForward(n_embed, ffn_mult, n_experts, top_experts, dropout)
        else:
            self.ff = FeedForward(n_embed, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        if self.use_moe:
            ff_out, aux_loss = self.ff(self.ln2(x))
            x = x + ff_out
            return x, aux_loss
        else:
            x = x + self.ff(self.ln2(x))
            return x, torch.tensor(0.0, device=x.device)


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList(
            [Block(n_embed,
                n_head=hyperparameters['early']['n_heads'],
                ffn_mult=hyperparameters['early']['ffn_mult'],
                dropout=hyperparameters['early']['dropout']) for _ in range(early_layers)]
            +
            [Block(n_embed,
                n_head=hyperparameters['middle']['n_heads'],
                ffn_mult=hyperparameters['middle']['ffn_mult'],
                dropout=hyperparameters['middle']['dropout']) for _ in range(middle_layers)]
            +
            [Block(n_embed,
                n_head=hyperparameters['late']['n_heads'],
                ffn_mult=hyperparameters['late']['ffn_mult'],
                dropout=hyperparameters['late']['dropout']) for _ in range(late_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        moe_aux_total = 0.0
        for block in self.blocks:
            x, moe_aux = block(x)
            moe_aux_total += moe_aux

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(B * T, vocab_size)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss, moe_aux_total

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

# ----------------------------
# Training Loop
# ----------------------------

# Enable fast Flash / memory-efficient attention kernels
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

# Allow TF32 math (faster matmuls on modern GPUs)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ----------------------------
# Layer-wise LR Optimizer
# ----------------------------
def build_optimizer(model, base_lr, layer_decay):
    param_groups = []

    # Embeddings (lowest LR)
    param_groups.append({
        "params": list(model.token_embedding_table.parameters()) +
                  list(model.position_embedding_table.parameters()),
        "lr": base_lr * (layer_decay ** len(model.blocks))
    })

    # Blocks (decayed per layer + scaled per stage)
    n_blocks = len(model.blocks)
    layer_idx = 0

    # Early layers
    for _ in range(early_layers):
        lr = base_lr * (layer_decay ** (n_blocks - layer_idx - 1)) * hyperparameters['early']['lr_scale']
        param_groups.append({
            "params": model.blocks[layer_idx].parameters(),
            "lr": lr
        })
        layer_idx += 1

    # Middle layers
    for _ in range(middle_layers):
        lr = base_lr * (layer_decay ** (n_blocks - layer_idx - 1)) * hyperparameters['middle']['lr_scale']
        param_groups.append({
            "params": model.blocks[layer_idx].parameters(),
            "lr": lr
        })
        layer_idx += 1

    # Late layers
    for _ in range(late_layers):
        lr = base_lr * (layer_decay ** (n_blocks - layer_idx - 1)) * hyperparameters['late']['lr_scale']
        param_groups.append({
            "params": model.blocks[layer_idx].parameters(),
            "lr": lr
        })
        layer_idx += 1

    # Final LayerNorm + LM head
    param_groups.append({
        "params": list(model.ln_f.parameters()) + list(model.lm_head.parameters()),
        "lr": base_lr
    })

    return torch.optim.AdamW(param_groups, weight_decay=0.01)






if __name__ == "__main__":

    model = BigramLanguageModel().to(device)
    optimizer = build_optimizer(model, base_lr=learning_rate, layer_decay=0.9)
    scaler = torch.cuda.amp.GradScaler()

    print("Training initiated")
    for iter in range(max_iters):
        if iter % eval_print_interval == 0:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits, loss, moe_aux = model(xb, yb)
            total_loss = loss + moe_weight * moe_aux

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # ----------------------------
    # Save Model & Loss
    # ----------------------------
    torch.save(model.state_dict(), "checkpoints/final_model.pt")
    final_losses = estimate_loss(model)
    torch.save(final_losses, "checkpoints/final_losses.pt")

    print("Final model and losses saved.")
    print(f"Final losses → train: {final_losses['train']:.4f}, val: {final_losses['val']:.4f}")

    # Optional text generation test
    # context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
else:
    print('improvements_v2 imported')