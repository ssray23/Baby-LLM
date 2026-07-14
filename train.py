import torch
import os
import re
from model import BabyGPT, context_window

batch_size = 32
max_iters = 5000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def train_model():
    print(f"Using device: {device}")
    
    # Read all data files
    data_dir = 'data'
    training_text = ""
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                training_text += f.read() + "\n"
                
    # Lowercase text consistently for tokenizing
    training_text = training_text.lower()
    
    # Vocabulary (Word-Level)
    # This splits text by words and keeps punctuation as separate tokens
    tokens = re.findall(r'\w+|[^\w\s]', training_text)
    vocab = sorted(list(set(tokens)))
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")
    
    word_to_int = { w:i for i,w in enumerate(vocab) }
    int_to_word = { i:w for i,w in enumerate(vocab) }
    encode = lambda s: [word_to_int[w] for w in re.findall(r'\w+|[^\w\s]', s)]
    
    data = torch.tensor(encode(training_text), dtype=torch.long)
    
    # Train/val split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    def get_batch(split):
        d = train_data if split == 'train' else val_data
        if len(d) <= context_window:
            d = train_data # fallback if val data is too small
        ix = torch.randint(len(d) - context_window, (batch_size,))
        x = torch.stack([d[i:i+context_window] for i in ix])
        y = torch.stack([d[i+1:i+context_window+1] for i in ix])
        return x.to(device), y.to(device)
    
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(10)
            for k in range(10):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    model = BabyGPT(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("Training the Baby LLM...")
    for steps in range(max_iters):
        if steps % 500 == 0:
            losses = estimate_loss()
            print(f"Step {steps}: Train Loss = {losses['train']:.4f}, Val Loss = {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
        optimizer.step()
    
    losses = estimate_loss()
    print(f"Final Loss: Train={losses['train']:.4f}, Val={losses['val']:.4f}")
    loss = losses['train'] # Keep variable assignment for return value below
    
    # Save the model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab, # Saved word-level vocabulary
        'vocab_size': vocab_size
    }
    torch.save(checkpoint, 'baby_llm.pth')
    print("Model saved to baby_llm.pth")
    return {"status": "success", "loss": float(loss.item()), "vocab_size": vocab_size}

if __name__ == "__main__":
    train_model()
