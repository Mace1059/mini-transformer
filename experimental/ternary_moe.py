import torch
import torch.nn as nn
from torch.nn import functional as F
from src.ternary_moe_layers import TernaryMoEFeedForward

# Training-Persistent Hyperparameters
batch_size = 128
block_size = 128
max_iters = 5000
eval_print_interval = 500
eval_iters = 5
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 128
n_layers = 3
moe_weight = 1e-2


torch.manual_seed(1337)

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get vocab size and create encoder/decoder
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s] # string to list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # list of integers to string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    # Generate batch of data inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Use no_grad because we don't call backward on this function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out    


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # Create key, query, and value 
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril((torch.ones(block_size, block_size))))
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
 
    def forward(self, x):
        # Concatenate on the C dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), 
            nn.GELU(), 
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout) 
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head, ffn_mult, dropout, use_moe=True, n_experts=8, top_experts=2):
        super().__init__()
        head_size = n_embed // n_head
        self.attn = MultiHeadAttention(n_head, head_size)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.use_moe = use_moe
        if self.use_moe:
            self.ff = TernaryMoEFeedForward(n_embed, ffn_mult, n_experts, top_experts, dropout)
        else:
            self.ff = FeedForward(n_embed, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        if self.use_moe:
            ff_out, aux = self.ff(self.ln2(x))
            x = x + ff_out
            # Put auxillary loss on module for caller to use 
            self._moe_aux = aux
        else:
            x = x + self.ff(self.ln2(x))
            self._moe_aux = torch.tensor(0.0, device=x.device)
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # (B,T,C)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList(
            [Block(n_embed, n_head=4, ffn_mult=8, dropout=0.5) 
             for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    # idx and targets are tensors of size (B,T)
    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx) # (B,T,C)

        # Introduce positional embedding
        positional_embedding = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embeddings + positional_embedding

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)


        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond) # Calls forward, (B,T,C)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=1) # (B, C)
            next_idx = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, next_idx), dim=1) # (B, T+1)
        
        return idx


model = BigramLanguageModel()
m = model.to(device)


from torch.cuda.amp import autocast, GradScaler

if __name__ == "__main__":

    # Create torch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # AMP scaler for mixed precision training
    scaler = GradScaler()

    print("training initiated")
    for iter in range(max_iters):
        if iter % eval_print_interval == 0:
            losses = estimate_loss()
            print(f"(ternary) step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with autocast(dtype=torch.float16):
            logits, loss = model(xb, yb)
            moe_aux = sum(getattr(b, "_moe_aux", 0) for b in model.blocks if hasattr(b, "_moe_aux"))
            loss = loss + moe_weight * moe_aux

        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.save(model.state_dict(), "checkpoints/final_model_ternary.pt")
    print("Final model weights saved!")

    final_losses = estimate_loss()
    print(f"Final losses → train: {final_losses['train']:.4f}, val: {final_losses['val']:.4f}")

    torch.save(final_losses, "checkpoints/final_losses_ternary.pt")
    print("Final losses saved!")


# context = torch.zeros((1,1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))