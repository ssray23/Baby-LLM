import torch
import torch.nn as nn
from torch.nn import functional as F

context_window = 64 # Changed to 64 words (much larger effective context)

class BabyTransformerLayer(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key   = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

    def forward(self, x):
        B, T, C = x.shape

        # Pre-norm attention: normalize BEFORE attention/FFN, not after.
        # Without this, residual activations compound unchecked across all 4
        # stacked layers, which destabilizes training and is the single
        # biggest cause of garbled/random-looking generations.
        x_norm = self.ln1(x)
        q = self.query(x_norm)
        k = self.key(x_norm)
        v = self.value(x_norm)

        wei = q @ k.transpose(-2, -1) * (C ** -0.5)

        # We need device dynamically
        device = x.device
        mask = torch.tril(torch.ones(T, T)).to(device)
        wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        out_attention = wei @ v
        x = x + out_attention

        out_ffn = self.ffn(self.ln2(x))
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

        # Final LayerNorm before the output head. Standard in every GPT-style
        # model (GPT-2 calls this ln_f) so the head always sees a
        # consistently-scaled input regardless of how deep the stack is.
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.linear_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        x = self.layers(x)
        x = self.ln_f(x)
        logits = self.linear_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=10, stop_token_ids=None):
        """
        temperature: lower (e.g. 0.5-0.8) makes the distribution more peaked
            around the model's actual favorite word instead of sampling
            uniformly from the full softmax, which is what let obviously
            wrong words get picked before.
        top_k: only sample from the k highest-probability tokens each step,
            so the model can no longer draw a near-zero-probability word.
        stop_token_ids: optional set of token ids. Generation stops as soon
            as one is produced (after at least 1 new token), so the model
            doesn't ramble on for the full max_new_tokens once it has
            finished its answer.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_window:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if stop_token_ids is not None and idx_next.item() in stop_token_ids:
                break

            idx = torch.cat((idx, idx_next), dim=1)
        return idx
