import torch
import torch.nn as nn
from torch.nn import functional as F

context_window = 64 # Changed to 64 words (much larger effective context)

class BabyTransformerLayer(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key   = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        
        # We need device dynamically
        device = x.device
        mask = torch.tril(torch.ones(T, T)).to(device)
        wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        out_attention = wei @ v
        x = x + out_attention 
        
        out_ffn = self.ffn(x)
        x = x + out_ffn
        return x

class BabyGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128): # Increased dimension for more capacity
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_window, embedding_dim)
        
        # 4 layers for more capacity
        self.layers = nn.Sequential(
            BabyTransformerLayer(embedding_dim),
            BabyTransformerLayer(embedding_dim),
            BabyTransformerLayer(embedding_dim),
            BabyTransformerLayer(embedding_dim)
        )
        
        self.linear_head = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        
        x = self.layers(x)
        logits = self.linear_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_window:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
