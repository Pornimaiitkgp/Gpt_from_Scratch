# A tiny GPT-style model (character-level) 

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)
random.seed(42)



# 1) Data / Tokenizer
text = "hello world"  
# build vocabulary (character-level)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])
# encode entire text as integers
data = torch.tensor(encode(text), dtype=torch.long)  # shape (N,)



# 2) Hyperparameters
device = "cpu"  # switch to "cuda" if available and desired
batch_size = 4
block_size = 8        # context length (how many previous chars the model sees)
max_iters = 200       # training iterations (keep small for toy data)
eval_interval = 50
learning_rate = 1e-2
embedding_dim = 32    # model width
num_heads = 4
num_layers = 2
ffn_hidden = embedding_dim * 4
dropout = 0.0



# 3) Data utilities
# prepare minibatch: randomly sample sequences of length block_size
def get_batch(batch_size):
    inputs = torch.zeros((batch_size, block_size), dtype=torch.long)
    targets = torch.zeros((batch_size, block_size), dtype=torch.long)
    N = len(data)
    for i in range(batch_size):
        start = random.randint(0, max(0, N - block_size - 1))
        chunk = data[start:start + block_size + 1]
        inputs[i] = chunk[:block_size]
        targets[i] = chunk[1:block_size + 1]
    return inputs.to(device), targets.to(device)



# 4) Model components
# Causal mask helper (so tokens cannot attend to future positions)
def causal_mask(seq_len, device):
    # mask shape: (1, 1, seq_len, seq_len) or (seq_len, seq_len)
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    return mask  # True where allowed
class MultiHeadSelfAttentionCausal(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.to_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        qkv = self.to_qkv(x)  # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)  # (B, T, 3, H, D)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # each: (B, H, T, D)
        # compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        # apply causal mask (set -inf where future tokens)
        mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
        att = F.softmax(scores, dim=-1)  # (B, H, T, T)
        att = self.dropout(att)
        out = torch.matmul(att, V)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.proj(out)
        return out
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttentionCausal(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_hidden, dropout)
    def forward(self, x):
        # x: (B, T, C)
        x = x + self.attn(self.ln1(x))    # residual around attention
        x = x + self.ffn(self.ln2(x))     # residual around ffn
        return x
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim, num_heads, num_layers, ffn_hidden, dropout=0.0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ffn_hidden, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.block_size = block_size
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx):
        # idx: (B, T) integer token IDs
        B, T = idx.shape
        assert T <= self.block_size, "Sequence too long for model block_size"
        tok = self.token_emb(idx)             # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))[None, :, :]  # (1, T, C)
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits
    # convenience: generate next tokens autoregressively
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx: (B, T) starting context
        for _ in range(max_new_tokens):
            B, T = idx.shape
            # crop to block_size
            idx_cond = idx[:, -self.block_size:]
            logits = self.forward(idx_cond)  # (B, T_cond, V)
            logits_last = logits[:, -1, :]   # (B, V)
            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx



# 5) Instantiate model
model = MiniGPT(vocab_size=vocab_size,
                block_size=block_size,
                embed_dim=embedding_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                ffn_hidden=ffn_hidden,
                dropout=dropout).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)



# 6) Training loop (toy)
model.train()
for it in range(max_iters):
    xb, yb = get_batch(batch_size)
    logits = model(xb)                     # (B, T, V)
    # compute cross-entropy loss for all tokens
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B*T, V), yb.view(B*T))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (it + 1) % eval_interval == 0 or it == 0:
        print(f"iter {it+1:4d}, loss {loss.item():.4f}")



# 7) Generate text
model.eval()
# start with a random token or the first char
start = torch.tensor([[encode("h")[0]]], dtype=torch.long).to(device)  # batch=1
gen_idx = model.generate(start, max_new_tokens=30)[0].tolist()
print("Generated sequence (indices):", gen_idx)
print("Generated text:", decode(gen_idx))