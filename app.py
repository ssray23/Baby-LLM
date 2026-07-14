from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from model import BabyGPT
from train import train_model

app = FastAPI()

# Global state for model
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model = None
encode = None
decode = None
vocab = []
stop_token_ids = None

# Global state for the retrieval index (exact/near-exact question lookup
# against data/*.txt). This runs BEFORE the generative model and answers
# directly whenever a close match exists, since a from-scratch model this
# small cannot reliably generalize facts, it can only approximate what it
# memorized. RETRIEVAL_THRESHOLD was tuned by testing paraphrases against
# the real data: matches score 0.4-0.8, unrelated questions score ~0.0.
RETRIEVAL_THRESHOLD = 0.35
qa_pairs = []
qa_vectorizer = None
qa_vectors = None


def clean_decoded_text(text):
    return text.replace(' .', '.').replace(' ?', '?').replace(' ,', ',').replace('[ Q ] :', '[Q]:').replace('[ A ] :', '[A]:').replace('\n ', '\n').strip()


def load_qa_pairs(data_dir='data'):
    """Parse [Q]/[A] pairs straight out of the training text files."""
    pairs = []
    pattern = re.compile(r'\[Q\]:\s*(.+?)\s*\n\[A\]:\s*(.+?)(?=\n\s*\n\[Q\]:|\Z)', re.DOTALL)
    if not os.path.isdir(data_dir):
        return pairs
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            for q, a in pattern.findall(text):
                pairs.append((q.strip(), a.strip()))
    return pairs


def build_retrieval_index():
    global qa_pairs, qa_vectorizer, qa_vectors
    qa_pairs = load_qa_pairs()
    if not qa_pairs:
        qa_vectorizer, qa_vectors = None, None
        print("WARNING: No Q/A pairs found for retrieval index.")
        return
    questions = [q for q, a in qa_pairs]
    qa_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    qa_vectors = qa_vectorizer.fit_transform(questions)
    print(f"Retrieval index built: {len(qa_pairs)} Q/A pairs.")


def retrieve_answer(user_prompt):
    """Returns (answer, score) for the closest known question, or (None, 0.0)."""
    if qa_vectorizer is None:
        return None, 0.0
    v = qa_vectorizer.transform([user_prompt])
    sims = cosine_similarity(v, qa_vectors)[0]
    best_idx = sims.argmax()
    best_score = sims[best_idx]
    if best_score >= RETRIEVAL_THRESHOLD:
        return qa_pairs[best_idx][1], float(best_score)
    return None, float(best_score)


def load_model():
    global model, encode, decode, vocab, stop_token_ids
    checkpoint_path = 'baby_llm.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
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

        # "[" only ever starts a new [Q]:/[A]: block in this dataset, so we
        # treat it as an end-of-answer signal. This stops generation once
        # the model starts drifting into the next block instead of always
        # burning through the full max_tokens budget.
        stop_id = word_to_int.get('[')
        stop_token_ids = {stop_id} if stop_id is not None else None

        print("Model loaded successfully.")
    else:
        model = None
        print("WARNING: Model checkpoint not found. Please train first.")


# Load initially
load_model()
build_retrieval_index()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50 # Reduced because 1 token = 1 word now, 50 words is plenty

@app.post("/generate")
def generate_text(req: GenerateRequest):
    raw_prompt = req.prompt.strip()
    if not raw_prompt:
        return {"response": "Sorry, I am not trained on this information."}

    # 1. Retrieval first: if this question (or a close paraphrase of it)
    # exists in the training data, answer directly. This is accurate by
    # construction, since it returns text that was actually written by a
    # human, not generated token-by-token by a 4-layer toy model.
    retrieved_answer, score = retrieve_answer(raw_prompt)
    if retrieved_answer is not None:
        return {"response": retrieved_answer}

    # 2. No confident match: fall back to the generative model.
    if not model:
        return JSONResponse(status_code=500, content={"error": "Model not trained yet or using old weights. Please hit Train."})

    prompt_for_model = raw_prompt[0].upper() + raw_prompt[1:]
    if not prompt_for_model.endswith('?'):
        prompt_for_model += '?'
    prompt_for_model = prompt_for_model.replace('albert', 'Albert').replace('einstein', 'Einstein')

    formatted_prompt = f"[Q]: {prompt_for_model}\n[A]:"

    prompt_tokens = re.findall(r'\w+|[^\w\s]', formatted_prompt)
    for token in prompt_tokens:
        if token not in vocab:
            return {"response": "Sorry, I am not trained on this information."}

    context = torch.tensor([encode(formatted_prompt)], dtype=torch.long, device=device)

    if context.numel() == 0:
         return {"response": "Sorry, I am not trained on this information."}

    generated_tokens = model.generate(
        context,
        max_new_tokens=req.max_tokens,
        temperature=0.8,
        top_k=10,
        stop_token_ids=stop_token_ids,
    )[0].tolist()
    generated_text = decode(generated_tokens)

    if "[A]:" in generated_text:
        parts = generated_text.split("[A]:")
        generated_only = parts[-1]
    else:
        generated_only = generated_text

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
        load_model()
        build_retrieval_index()
        return {"response": f"Training completed successfully! Final Loss: {result['loss']:.4f}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Serve the frontend statically
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
