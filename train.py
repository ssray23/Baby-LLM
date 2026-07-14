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
    
    def get_batch():
        ix = torch.randint(len(data) - context_window, (batch_size,))
        x = torch.stack([data[i:i+context_window] for i in ix])
        y = torch.stack([data[i+1:i+context_window+1] for i in ix])
        return x.to(device), y.to(device)
    
    model = BabyGPT(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("Training the Baby LLM...")
    for steps in range(max_iters):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if steps % 500 == 0:
            print(f"Step {steps}: Current Loss = {loss.item():.4f}")
    
    print(f"Final Loss: {loss.item():.4f}")
    
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
