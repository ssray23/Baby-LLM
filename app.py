from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import re
from model import BabyGPT
import os
from train import train_model

app = FastAPI()

# Global state for model
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model = None
encode = None
decode = None
vocab = []

def clean_decoded_text(text):
    # Remove weird spaces around punctuation
    return text.replace(' .', '.').replace(' ?', '?').replace(' ,', ',').replace('[ Q ] :', '[Q]:').replace('[ A ] :', '[A]:').replace('\n ', '\n').strip()

def load_model():
    global model, encode, decode, vocab
    checkpoint_path = 'baby_llm.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Check if this is the new word-level model
        if 'vocab' not in checkpoint:
            print("WARNING: Found old character-level checkpoint. Please run /train to generate a word-level checkpoint.")
            model = None
            return
            
        vocab = checkpoint['vocab']
        vocab_size = checkpoint['vocab_size']
        model = BabyGPT(vocab_size).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        word_to_int = { w:i for i,w in enumerate(vocab) }
        int_to_word = { i:w for i,w in enumerate(vocab) }
        
        encode = lambda s: [word_to_int.get(w, 0) for w in re.findall(r'\w+|[^\w\s]', s)]
        decode = lambda l: clean_decoded_text(' '.join([int_to_word.get(i, '') for i in l]))
        print("Model loaded successfully.")
    else:
        model = None
        print("WARNING: Model checkpoint not found. Please train first.")

# Load initially
load_model()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50 # Reduced because 1 token = 1 word now, 50 words is plenty

@app.post("/generate")
def generate_text(req: GenerateRequest):
    if not model:
        return JSONResponse(status_code=500, content={"error": "Model not trained yet or using old weights. Please hit Train."})
    
    # 0. Prompt Normalization Hack
    raw_prompt = req.prompt.strip()
    if raw_prompt:
        # Force first letter to be uppercase
        raw_prompt = raw_prompt[0].upper() + raw_prompt[1:]
        # Ensure it ends with a question mark
        if not raw_prompt.endswith('?'):
            raw_prompt += '?'
        
        # Hack to fix specific entity casing that the toy model is fragile to
        raw_prompt = raw_prompt.replace('albert', 'Albert').replace('einstein', 'Einstein')
    
    # 1. Prompt Injection
    formatted_prompt = f"[Q]: {raw_prompt}\n[A]:"
    
    # 2. Vocabulary Check (Word-Level)
    prompt_tokens = re.findall(r'\w+|[^\w\s]', formatted_prompt)
    for token in prompt_tokens:
        if token not in vocab:
            return {"response": "Sorry, I am not trained on this information."}

    context = torch.tensor([encode(formatted_prompt)], dtype=torch.long, device=device)
    
    if context.numel() == 0:
         return {"response": "Sorry, I am not trained on this information."}
         
    generated_tokens = model.generate(context, max_new_tokens=req.max_tokens)[0].tolist()
    generated_text = decode(generated_tokens)
    
    # 3. Response Parsing & Fallback Logic
    # We want to extract just the answer part generated after our prompt
    # Because decoding might slightly alter spacing, we search for [A]: 
    if "[A]:" in generated_text:
        parts = generated_text.split("[A]:")
        generated_only = parts[-1]
    else:
        generated_only = generated_text
    
    # Crop at the next question block if the model rambles
    if "[Q]:" in generated_only:
        generated_only = generated_only.split("[Q]:")[0]
        
    clean_answer = generated_only.strip()
    
    if not clean_answer:
        return {"response": "Sorry, I am not trained on this information."}
        
    return {"response": clean_answer}

@app.post("/train")
def train_endpoint():
    try:
        result = train_model()
        load_model() # Reload with new weights
        return {"response": f"Training completed successfully! Final Loss: {result['loss']:.4f}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Serve the frontend statically
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
