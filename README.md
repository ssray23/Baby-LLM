# Baby LLM 👶🤖

Baby LLM is a lightweight, educational implementation of a **decoder-only autoregressive Transformer**—the exact same architecture that powers state-of-the-art Generative Pre-trained Transformers (like GPT-3, GPT-4, and Gemini). 

Designed to be simple, transparent, and easy to run locally, Baby LLM helps bridge the gap between high-level theory and concrete PyTorch code.

---

## 🏗️ Architecture: How GPT Works Under the Hood

The architecture of Baby LLM ([model.py](file:///Users/suddharay/Library/Mobile%20Documents/com~apple~CloudDocs/Mac%20Projects/Baby%20LLM/model.py)) mirrors the core components of modern GPT models:

```
[ Input Text ]
      │
      ▼
[ Word Tokenizer ] ──────► Splitting text into individual words & punctuation
      │
      ▼
[ Token Embeddings ] ────► Maps words to continuous vector spaces (d = 128)
      │
      ▼
[ Positional Embeddings ] ─► Adds sequence order info (context window = 64)
      │
      ▼
┌────────────────────────┐
│  Transformer Block 1   │ 
├────────────────────────┤
│  Transformer Block 2   │   Stacked Layers (4 total) for deep pattern representation
├────────────────────────┤
│  Transformer Block 3   │ 
├────────────────────────┤
│  Transformer Block 4   │ 
└──────────┬─────────────┘
           │
           ▼
[ Output Linear Head ] ──► Projects vectors back to vocabulary dimension
           │
           ▼
[ Softmax Sampling ] ────► Computes probability of the next word & generates!
```

### 1. Tokenization & Embeddings (The Language of Vectors)
A neural network cannot process text directly; it only understands numbers.
* **Tokenization**: We split incoming text into words and punctuation (word-level tokenization). For example: 
  `"How many continents?"` $\rightarrow$ `["How", "many", "continents", "]` $\rightarrow$ `[142, 85, 33, 9]`.
* **Token Embeddings (`self.token_embedding`)**: Translates high-dimensional discrete tokens into dense vector representations of size `embedding_dim = 128`. Words with similar meanings will eventually occupy close physical coordinates in this space.
* **Positional Embeddings (`self.position_embedding`)**: Attention mechanisms process all tokens in parallel and do not inherently know the order of words. We add a positional vector corresponding to each word's index (from `0` to `63` in the context window) so the model knows that *"dog bites man"* is different from *"man bites dog"*.

---

### 2. Causal Self-Attention (The Core Engine)
At the heart of the `BabyTransformerLayer` is Self-Attention. It allows the model to look back at previous words in the sentence to build context.

* **Queries, Keys, and Values (Q, K, V)**:
  * **Query (Q)**: What a word is currently searching/asking for.
  * **Key (K)**: What a word contains or can offer.
  * **Value (V)**: The actual informational content of the word.
  
  We multiply the Query of token $A$ with the Keys of all prior tokens to produce an **attention score**. This score determines how much token $A$ should "focus" on each of the other words. We then multiply these scores by the **Values (V)** to aggregate a new, context-rich representation.

* **Causality / The Triangular Mask**:
  Because GPT is an autoregressive model (predicting the *next* word from left to right), tokens should not be allowed to look ahead into future words.
  We enforce this by applying a lower-triangular mask (`torch.tril`):
  $$
  \text{Mask} = \begin{pmatrix}
  1 & 0 & 0 \\
  1 & 1 & 0 \\
  1 & 1 & 1
  \end{pmatrix}
  $$
  Any value in the attention matrix corresponding to a `0` is filled with $-\infty$ before calculating the Softmax. This forces their attention weights to `0`, ensuring the model only learns to predict using the past.

---

### 3. Residual Connections & Layer Normalization
* **Residual Connections (`x = x + attention_out`)**: High-speed highways that bypass the transformer blocks. They add the block's input back to its output. This prevents gradients from shrinking (vanishing) during backpropagation, allowing us to build deeper networks.
* **Feed-Forward Networks (FFN)**: After aggregating context from other tokens via attention, each token vector is passed through a simple multi-layer perceptron (Linear $\rightarrow$ ReLU $\rightarrow$ Linear) in isolation. This allows the model to perform computations and process the gathered context.

---

### 4. Next-Word Prediction & Generation
During generation, the model:
1. Takes a sequence of up to 64 tokens.
2. Performs a forward pass to produce a vocabulary probability distribution (logits) for the next token.
3. Takes the logit at the very last token position and applies **Softmax** to convert it into a probability distribution.
4. Uses **Multinomial Sampling** (`torch.multinomial`) to randomly pick the next word weighted by its probability, avoiding repetitive outputs.
5. Appends the new word to the prompt and repeats (autoregressive generation).

---

## 📂 Project Structure

```
├── model.py            # Neural Network architecture (BabyGPT & BabyTransformerLayer)
├── train.py            # Word-level vocabulary creation, data batch loader, & PyTorch training loop
├── app.py              # FastAPI backend serving the generation & training endpoints
├── baby_llm.pth        # Saved model weights & vocabulary (generated after training)
├── data/               # Knowledge text files containing question-answer pairs
│   ├── knowledge_1.txt
│   ├── knowledge_2.txt
│   └── ...
└── frontend/           # Static web files for interactive play
    ├── index.html
    ├── style.css
    └── app.js
```

---

## 🚀 Running Baby LLM Locally

### 1. Installation
Ensure you have Python 3.8+ installed, then install the dependencies:
```bash
pip install torch fastapi uvicorn pydantic
```

### 2. Training the Model
To train the model on the custom text files in the `data/` directory, run:
```bash
python train.py
```
This reads the text, creates a word-level vocabulary, trains the neural network over `5000` steps using the `AdamW` optimizer, and saves the weights to `baby_llm.pth`.

### 3. Running the Server & Frontend
Start the FastAPI server:
```bash
uvicorn app:app --reload --port 8000
```

Once started:
* **Interactive Chat UI**: Open `http://localhost:8000` in your web browser.
* **Train API**: `POST http://localhost:8000/train` triggers a retraining loop.
* **Generate API**: `POST http://localhost:8000/generate` (takes `prompt` and `max_tokens` JSON inputs).
