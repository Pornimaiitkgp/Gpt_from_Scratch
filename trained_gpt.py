import torch
import torch.nn as nn
import torch.nn.functional as F


# LOAD DATA
text = open("input.txt", "r", encoding="utf-8").read()
print("Data length in characters:", len(text))


# Character-level tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Unique characters:", vocab_size)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


# PREPARE DATA
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
block_size = 128   # context length
batch_size = 64
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x, y


# MODEL DEFINITION
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
    def forward(self, x):
        B, T, C = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        att = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.masked_fill(torch.tril(torch.ones(T, T)) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ V
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    def forward(self, x):
        return self.net(x)
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim):
        super().__init__()
        self.attn = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = FeedForward(embedding_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_heads=4, hidden_dim=256, num_layers=4, block_size=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embedding_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


# TRAINING
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MiniGPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
max_iters = 5000
eval_interval = 500
for step in range(max_iters):
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % eval_interval == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")


# GENERATE TEXT
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\nGenerated text:\n")
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))