import torch
import torch.nn as nn
import torch.nn.functional as F
from src.moe_layers import MoEFeedForward  # assumes your MoE implementation is correct

# ----------------------------
# Training Hyperparameters
# ----------------------------
batch_size = 128
block_size = 128
max_iters = 5000
eval_print_interval = 500
eval_iters = 5
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 128
n_layers = 6
moe_weight = 1e-2


hyperparameters = {
    'early': {
        'ffn_mult': 2,
        'dropout': 0.05,
        'lr_scale': 1.2,
        'n_heads': 8,
    },
    'middle': {
        'ffn_mult': 4,
        'dropout': 0.1,
        'lr_scale': 1,
        'n_heads': 6,
    },
    'late': {
        'ffn_mult': 8,
        'drouput': 0.3,
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


def get_batch(split):
    """Returns a batch of training or validation data."""
    src = train_data if split == 'train' else val_data
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i:i+block_size] for i in ix])
    y = torch.stack([src[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


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


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


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
        head_size = n_embed // n_head
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_head, head_size)
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
        self.blocks = nn.ModuleList([
            Block(n_embed, n_head=4, ffn_mult=4, dropout=0.1)
            for _ in range(n_layers)
        ])
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
model = BigramLanguageModel().to(device)
# model = torch.compile(model)  # 🚀 compile for speed

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

print("🚀 Training initiated")
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

print("✅ Final model and losses saved.")
print(f"Final losses → train: {final_losses['train']:.4f}, val: {final_losses['val']:.4f}")

# Optional text generation test
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
