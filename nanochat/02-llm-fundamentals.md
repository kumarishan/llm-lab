# Chapter 2: LLM Fundamentals — Tokens, Embeddings, and Attention

## What you'll learn

- How a language model works mathematically: it predicts the probability of the next token given all previous tokens
- What tokens are, how text becomes integers, and why the choice of tokenization unit matters
- How token embeddings map integers to dense vectors in a semantic space
- How the self-attention mechanism lets tokens "look at" each other and what the Q, K, V matrices actually compute
- How transformer blocks stack attention and feed-forward layers with residual connections to build up a full language model

## Prerequisites

- You have completed Chapter 1 (environment set up, nanochat running)
- You are comfortable with Python and NumPy
- You know what a vector and a matrix are, and what matrix multiplication means
- You know basic probability: what $P(A|B)$ means

> **What we will NOT cover here.** nanochat uses several advanced techniques — rotary positional embeddings (RoPE), Group-Query Attention (GQA), sliding window attention, and QK normalization. These are optimisations layered on top of the core transformer. We will cover them in Chapter 4. This chapter is about the foundations that everything else is built on.

---

## 1. What is a Language Model?

A language model has one job: **given a sequence of tokens, predict what token comes next.**

More precisely, it models the probability:

$$P(\text{token}_t \mid \text{token}_1, \text{token}_2, \ldots, \text{token}_{t-1})$$

Read this as: "what is the probability of seeing token $t$, given that we have already seen tokens 1 through $t-1$?"

That is a surprisingly powerful thing to learn. If you can predict the next token well, you must have learned a lot about syntax, facts, reasoning patterns, and style — because all of those things constrain what word comes next.

### Autoregressive generation

When a model generates text, it applies this probability rule repeatedly:

1. Start with a prompt (some tokens)
2. Sample the next token from $P(\text{token}_t \mid \text{token}_1, \ldots, \text{token}_{t-1})$
3. Append that token to the sequence
4. Go to step 2

This is called **autoregressive generation**: each new token is conditioned on all the tokens generated before it.

The joint probability of an entire sequence $(\text{token}_1, \ldots, \text{token}_T)$ factors into a chain of conditionals:

$$P(\text{token}_1, \ldots, \text{token}_T) = \prod_{t=1}^{T} P(\text{token}_t \mid \text{token}_1, \ldots, \text{token}_{t-1})$$

This is the **chain rule of probability**, and it is exact — no approximation involved. Autoregressive models are directly trained to maximise this quantity.

### A simple Python illustration

Let's make this concrete with a tiny Python example. Imagine a three-word vocabulary and a made-up probability table.

✍️
```python
import numpy as np

# A tiny vocabulary: 3 tokens
vocab = ["the", "cat", "sat"]
vocab_size = 3

# A made-up conditional probability table.
# P[context_token][next_token] = probability
# (In a real model, context can be arbitrarily long; here we simplify to single-token context.)
P = {
    "the": {"the": 0.05, "cat": 0.80, "sat": 0.15},
    "cat": {"the": 0.10, "cat": 0.05, "sat": 0.85},
    "sat": {"the": 0.70, "cat": 0.20, "sat": 0.10},
}

def sequence_probability(tokens):
    """Compute the joint probability of a token sequence."""
    log_prob = 0.0
    for i in range(1, len(tokens)):
        context = tokens[i - 1]
        next_tok = tokens[i]
        log_prob += np.log(P[context][next_tok])
    return np.exp(log_prob)

sequence = ["the", "cat", "sat"]
print(f"P({sequence}) = {sequence_probability(sequence):.4f}")

# Greedy generation from a starting token
def greedy_generate(start_token, n_steps):
    tokens = [start_token]
    for _ in range(n_steps):
        context = tokens[-1]
        probs = P[context]
        next_tok = max(probs, key=probs.get)  # pick highest probability
        tokens.append(next_tok)
    return tokens

print("Greedy generation:", greedy_generate("the", 4))
```

> **What's happening.** `sequence_probability` multiplies together the conditional probabilities at each step — that's the chain rule. `greedy_generate` picks the highest-probability token at each step; a real model would sample from the distribution instead, giving more varied output.

---

## 2. Tokens and Vocabulary

### Text is not numbers — tokenisation bridges the gap

Neural networks operate on numbers. Text is characters. A **tokeniser** converts text into a sequence of integers, each integer identifying a token in a fixed vocabulary.

```
"Hello, world!" → [15339, 11, 1917, 0]
```

Each integer is a **token ID**, indexing into a vocabulary of size $V$. nanochat uses $V = 32{,}768$.

### Why not characters or words?

**Character-level models** use a vocabulary of about 100 characters. Sequences become very long (every character is a step), gradients must travel further, and the model has to learn to spell from scratch. Character models exist but they are inefficient.

**Word-level models** split on whitespace. The vocabulary explodes ("run", "runs", "running", "ran" are separate entries), rare words get no representation ("photosynthesis" might never appear in training), and you cannot handle typos or new words.

**Subword tokenisation** (which almost all modern LLMs use, including nanochat) is the sweet spot:

- Common words ("the", "and") get their own single token
- Rare words are split into smaller pieces ("photosynthesis" → "photo", "syn", "thesis")
- Vocabulary stays manageable (~32K–128K)
- No unknown tokens: any string can be encoded

The most common algorithm is **Byte-Pair Encoding (BPE)**, which iteratively merges the most frequent pairs of existing tokens until the vocabulary reaches the target size.

### Seeing tokenisation in action

nanochat ships with a tokeniser. Let's use it:

✍️
```python
# From the nanochat project root
import tiktoken  # nanochat uses tiktoken under the hood

# GPT-4o tokenizer (cl100k or o200k base — nanochat configures this)
enc = tiktoken.get_encoding("cl100k_base")

text = "Hello, I am a language model."
token_ids = enc.encode(text)
print("Token IDs:", token_ids)

# Decode each token individually to see what pieces it found
tokens_as_text = [enc.decode([t]) for t in token_ids]
print("Tokens:   ", tokens_as_text)
print(f"Characters: {len(text)}, Tokens: {len(token_ids)}")
```

You should see output like:
```
Token IDs: [9906, 11, 358, 1097, 264, 4221, 1646, 13]
Tokens:    ['Hello', ',', ' I', ' am', ' a', ' language', ' model', '.']
Characters: 30, Tokens: 8
```

Notice that spaces are often part of the token (` am`, not `am`). This is an implementation detail of GPT tokenisers.

### Vocabulary size matters

The vocabulary size $V$ creates a direct tradeoff:

| Larger $V$ | Smaller $V$ |
|---|---|
| Fewer tokens per sentence (cheaper computation) | More tokens per sentence (more steps) |
| Each token carries more meaning | Each token carries less meaning |
| Embedding table is larger in memory | Embedding table is smaller |
| Rare concepts get their own token | More splitting of rare words |

nanochat's $V = 32{,}768$ is a common choice for models of its size. GPT-4 uses 100,277. Llama 3 uses 128,256.

---

## 3. Token Embeddings

### From integers to vectors

A token ID like `15339` is just a number. It has no geometric meaning — is 15339 "larger" or "closer" to 15340 than to 1? Not in any useful sense.

We want to map each token ID to a **dense vector** in a high-dimensional space, where similar tokens land close together. This mapping is the **embedding layer**, and it is just a lookup table:

```
Embedding matrix  E  ∈  R^(V × d)

E[token_id]  =  a vector of dimension d
```

For nanochat, $V = 32{,}768$ and $d = 768$ (the `n_embd` config parameter). So the embedding matrix is shape `[32768, 768]`.

Looking up an embedding is equivalent to multiplying a one-hot vector by $E$, but in practice we just index into the matrix directly.

### What does an embedding "mean"?

During training, the model adjusts the embedding vectors so that tokens that appear in similar contexts end up in similar locations in embedding space. This is learned purely from data — no human labels.

The famous result from Word2Vec (2013) holds approximately for transformer embeddings too:

```
embedding("king") - embedding("man") + embedding("woman")  ≈  embedding("queen")
```

Directions in embedding space correspond to semantic concepts. The model learns these geometric relationships automatically because they help it predict the next token.

### A NumPy illustration

✍️
```python
import numpy as np

# Tiny example: vocab of 5 tokens, embedding dimension 4
V = 5
d = 4
rng = np.random.default_rng(42)

# The embedding matrix: each row is the embedding for one token
E = rng.normal(0, 1, size=(V, d))
print("Embedding matrix shape:", E.shape)

# Look up embeddings for a sequence of tokens
token_ids = [2, 0, 4, 1]
embeddings = E[token_ids]  # shape: (4, 4) — 4 tokens, each with 4-dim vector
print("Sequence embeddings shape:", embeddings.shape)
print("Embedding for token 2:", embeddings[0])

# Cosine similarity between two token embeddings
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim = cosine_similarity(E[0], E[1])
print(f"Cosine similarity between token 0 and token 1: {sim:.3f}")
```

> **What's happening.** `E[token_ids]` is just array indexing — we grab the rows of the matrix corresponding to our token IDs. In PyTorch this is `nn.Embedding`, which is exactly what `self.transformer.wte` does in nanochat (wte = "word token embedding").

### The embedding matrix in nanochat

In `gpt.py`, the embedding layer is:

```python
# From nanochat/gpt.py
"wte": nn.Embedding(padded_vocab_size, config.n_embd),
```

After embedding, nanochat immediately applies layer normalisation:

```python
x = self.transformer.wte(idx)   # shape: (batch, seq_len, n_embd)
x = norm(x)
```

This is "norm after embedding" — a stabilisation technique that ensures the embedding vectors enter the transformer with controlled magnitude regardless of how the embedding weights were initialised.

---

## 4. The Transformer's Job

We now have a sequence of vectors — one per token, each of dimension $d$. Call this sequence:

$$\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T \quad \text{where each } \mathbf{x}_i \in \mathbb{R}^d$$

The transformer takes this sequence of vectors **in**, and outputs a new sequence of vectors **out** — same shape, but now each vector "knows about" the other tokens in the sequence:

```
Input:   x_1  x_2  x_3  ...  x_T        (each is a d-dim vector)
            \    |   /          |
         [ Transformer layers ]
            /    |   \          |
Output:  h_1  h_2  h_3  ...  h_T        (each is a d-dim vector)
```

The output vectors $\mathbf{h}_t$ are called **hidden states** or **contextualised representations**. The last hidden state in the sequence, $\mathbf{h}_T$, is used to predict the next token.

### The residual stream

A crucial design choice of transformers is the **residual connection**: instead of replacing $\mathbf{x}$ at each layer, each layer *adds* its output to the running value:

$$\mathbf{x} \leftarrow \mathbf{x} + \text{layer}(\mathbf{x})$$

Think of it as a **residual stream** — a highway of information that flows from the input through all the layers to the output. Each layer "writes" small updates to this stream. Nothing is ever erased, only added to.

```
x ──────────────────────────────────────────────────────► output
 \                  |                    |
  └─► Attn ────► (+)  ──► MLP ────────► (+)
           Layer 1                 Layer 1
```

This design has two practical benefits:
1. Gradients flow easily from output back to input during training (no vanishing gradient problem)
2. Early layers can pass information directly to late layers without going through every transformation

### Depth: stacking N blocks

The same pattern (attention + MLP + residual) is repeated $N$ times. nanochat's default is `n_layer = 12`. Each layer has its own separate parameters and learns to extract different features.

A common mental model: early layers handle low-level patterns (syntax, morphology), later layers handle high-level patterns (semantics, reasoning). This is an approximation — the reality is more distributed — but it is a useful intuition.

---

## 5. Self-Attention: The Core Mechanism

Self-attention is the part of the transformer that allows tokens to *communicate with each other*. Without it, each position would be processed independently and the model could not relate "it" to "the cat" earlier in the sentence.

### The intuition

Imagine you are trying to understand the word "bank" in the sentence "I deposited money at the bank." You look at the surrounding words — "deposited", "money" — and conclude this is a financial bank, not a river bank. Self-attention formalises this: each token queries all other tokens and gathers relevant information.

More precisely, for each position $t$, the attention mechanism asks:
> "Which other positions in the sequence are relevant to understanding position $t$, and how relevant are they?"

The answer is expressed as a weighted average over all other positions.

### Queries, Keys, and Values

The attention mechanism uses three linear projections of the input, applied to every position:

| Name | Symbol | Role |
|------|--------|------|
| Query | $Q$ | "What am I looking for?" |
| Key | $K$ | "What do I contain?" |
| Value | $V$ | "What information will I contribute?" |

For a sequence of $T$ tokens with embedding dimension $d$, and a chosen attention head dimension $d_k$:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

where $X \in \mathbb{R}^{T \times d}$ is the matrix of all token embeddings, and $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ are learned weight matrices.

So $Q, K, V$ each have shape $\mathbb{R}^{T \times d_k}$.

### The attention score

The relevance of position $j$ to position $i$ is measured by the **dot product** of position $i$'s query with position $j$'s key:

$$\text{score}(i, j) = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}$$

The $\sqrt{d_k}$ scaling factor prevents the dot products from becoming so large that the softmax saturates. When $d_k$ is large, dot products tend to grow in magnitude, pushing the softmax into regions where gradients are tiny.

For all positions at once, this is a matrix multiplication:

$$\text{Scores} = \frac{Q K^\top}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}$$

Entry $(i, j)$ of this matrix is how much position $i$ attends to position $j$.

### Softmax to get attention weights

We apply softmax row-wise to turn raw scores into a probability distribution (attention weights that sum to 1):

$$A = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{T \times T}$$

Row $i$ of $A$ is a probability distribution over all $T$ positions. $A_{ij}$ is "how much does position $i$ attend to position $j$?"

### Output: weighted sum of Values

The attention output for each position is a weighted average of the Value vectors, using the attention weights:

$$\text{Output} = A V \in \mathbb{R}^{T \times d_k}$$

**The complete scaled dot-product attention formula:**

$$\boxed{\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V}$$

This is the most important equation in the transformer. Read it carefully: every output position is a soft, weighted combination of all value vectors, where the weights are determined by query-key compatibility.

### A complete NumPy implementation

✍️
```python
import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (T, d_k)  — queries
    K: (T, d_k)  — keys
    V: (T, d_v)  — values
    mask: (T, T) boolean mask, True = masked out (set to -inf)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)      # (T, T)
    if mask is not None:
        scores[mask] = -1e9               # large negative => near-zero after softmax
    weights = softmax(scores, axis=-1)    # (T, T), rows sum to 1
    output = weights @ V                  # (T, d_v)
    return output, weights

# Example: sequence of 4 tokens, embedding dim 6, head dim 4
np.random.seed(0)
T, d, d_k = 4, 6, 4

# Toy weight matrices
W_Q = np.random.randn(d, d_k) * 0.1
W_K = np.random.randn(d, d_k) * 0.1
W_V = np.random.randn(d, d_k) * 0.1

# Input: 4 token embeddings, each dimension 6
X = np.random.randn(T, d)

Q = X @ W_Q    # (4, 4)
K = X @ W_K    # (4, 4)
V = X @ W_V    # (4, 4)

output, weights = scaled_dot_product_attention(Q, K, V)
print("Attention output shape:", output.shape)
print("Attention weights (each row sums to 1):")
print(np.round(weights, 3))
print("Row sums:", weights.sum(axis=-1))
```

> **What's happening.** `Q @ K.T` computes all pairwise scores at once. After softmax, row $i$ of `weights` tells us how much position $i$ borrowed from each other position. `weights @ V` then blends the values accordingly.

### ASCII diagram: attention as information mixing

```
Positions:    1       2       3       4

Queries:     q_1     q_2     q_3     q_4

Keys:        k_1     k_2     k_3     k_4

Scores:     q_1·k_1  q_1·k_2  q_1·k_3  q_1·k_4   ← position 1 scoring all others
            q_2·k_1  q_2·k_2  q_2·k_3  q_2·k_4   ← position 2 scoring all others
            ...

Softmax rows → attention weights A (T×T matrix)

Values:      v_1     v_2     v_3     v_4

Output[1] = A[1,1]*v_1 + A[1,2]*v_2 + A[1,3]*v_3 + A[1,4]*v_4
```

### Causal masking

A language model cannot be allowed to "see the future." When predicting token $t$, only tokens $1, \ldots, t-1$ should be visible. We enforce this with a **causal mask**: we set scores $(i, j)$ to $-\infty$ when $j > i$, which means position $i$ cannot attend to any position after it.

After softmax, $e^{-\infty} = 0$, so those positions contribute nothing.

✍️
```python
def causal_mask(T):
    """Return a (T, T) boolean mask. True = should be masked out (future positions)."""
    # Upper triangle excluding diagonal
    return np.triu(np.ones((T, T), dtype=bool), k=1)

T = 4
mask = causal_mask(T)
print("Causal mask (True = blocked):")
print(mask.astype(int))
```

```
Causal mask:
[[0 1 1 1]
 [0 0 1 1]
 [0 0 0 1]
 [0 0 0 0]]
```

Position 0 can only attend to itself. Position 1 can attend to positions 0 and 1. Position 3 (the last) can attend to all positions. This is the autoregressive property enforced in the attention layer.

✍️
```python
# Full causal attention
output_causal, weights_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask(T))
print("Causal attention weights:")
print(np.round(weights_causal, 3))
# Upper triangle should be ~0
```

### Multi-head attention

Using a single attention operation means the model has only one way to combine information. **Multi-head attention** runs $H$ independent attention operations in parallel, each with its own $W_Q^h, W_K^h, W_V^h$ matrices and a smaller head dimension $d_k = d / H$.

$$\text{head}_h = \text{Attention}(X W_Q^h, \, X W_K^h, \, X W_V^h)$$

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \, W_O$$

where $W_O \in \mathbb{R}^{(H \cdot d_k) \times d}$ projects the concatenated heads back to the residual stream dimension $d$.

Different heads learn to attend to different kinds of relationships: one head might track syntactic dependencies, another might track coreference, and so on. This emerges from training — it is not programmed in.

nanochat uses `n_head = 6`, so with `n_embd = 768`, each head has dimension $768 / 6 = 128$.

✍️
```python
def multi_head_attention(X, W_Qs, W_Ks, W_Vs, W_O, mask=None):
    """
    X:    (T, d)
    W_Qs: list of H matrices, each (d, d_k)
    W_Ks: list of H matrices, each (d, d_k)
    W_Vs: list of H matrices, each (d, d_v)
    W_O:  (H * d_v, d)
    """
    heads = []
    for W_Q, W_K, W_V in zip(W_Qs, W_Ks, W_Vs):
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V
        head_out, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
        heads.append(head_out)

    concatenated = np.concatenate(heads, axis=-1)   # (T, H * d_v)
    output = concatenated @ W_O                      # (T, d)
    return output

# Example with H=2 heads, d=6, d_k=3
H = 2
d, d_k = 6, 3
np.random.seed(1)
X = np.random.randn(4, d)
W_Qs = [np.random.randn(d, d_k) * 0.1 for _ in range(H)]
W_Ks = [np.random.randn(d, d_k) * 0.1 for _ in range(H)]
W_Vs = [np.random.randn(d, d_k) * 0.1 for _ in range(H)]
W_O  = np.random.randn(H * d_k, d) * 0.1

out = multi_head_attention(X, W_Qs, W_Ks, W_Vs, W_O, mask=causal_mask(4))
print("Multi-head attention output shape:", out.shape)  # Should be (4, 6)
```

---

## 6. The MLP (Feed-Forward) Block

After attention allows tokens to communicate, each position passes through a **Multi-Layer Perceptron (MLP)** independently. There is no information mixing across positions here — the MLP at position $t$ only sees $\mathbf{x}_t$.

The structure is:

$$\text{MLP}(\mathbf{x}) = W_2 \cdot \text{activation}(W_1 \mathbf{x})$$

where $W_1 \in \mathbb{R}^{d \times 4d}$ expands to a larger hidden dimension (by convention 4x), the activation introduces non-linearity, and $W_2 \in \mathbb{R}^{4d \times d}$ projects back down.

### Why does the MLP exist?

Attention is linear (softmax aside): it computes weighted averages of values. The MLP is where the network can apply non-linear transformations. Research suggests that much of the model's **factual knowledge** is stored in the MLP weights (they act like a key-value memory).

If attention is "routing information from relevant positions," then the MLP is "processing that information to produce new representations."

### ReLU squared: what nanochat uses

nanochat uses a novel activation: **ReLU squared**, written $\text{ReLU}(x)^2$.

Standard ReLU: $\text{ReLU}(x) = \max(0, x)$

Squared ReLU: $\text{ReLU}^2(x) = \max(0, x)^2$

✍️
```python
import numpy as np
import matplotlib

# Show the shape of ReLU vs ReLU^2
x = np.linspace(-2, 2, 200)
relu    = np.maximum(0, x)
relu2   = np.maximum(0, x) ** 2

print("ReLU(-1.0)   =", np.maximum(0, -1.0))        # 0.0
print("ReLU(0.5)    =", np.maximum(0, 0.5))          # 0.5
print("ReLU^2(-1.0) =", np.maximum(0, -1.0)**2)      # 0.0
print("ReLU^2(0.5)  =", np.maximum(0, 0.5)**2)       # 0.25
print("ReLU^2(2.0)  =", np.maximum(0, 2.0)**2)       # 4.0
```

Properties of ReLU squared:
- Still zero for negative inputs (same sparsity as ReLU — many neurons are exactly zero)
- Smooth derivative at $x=0$ (unlike ReLU which has a kink), which can help training
- Grows faster than ReLU for positive inputs (stronger signal amplification)
- No learned parameters — simpler than GeLU or Swish

In nanochat's `MLP.forward`:

```python
# From nanochat/gpt.py
def forward(self, x):
    x = self.c_fc(x)
    x = F.relu(x).square()   # ReLU^2 activation
    x = self.c_proj(x)
    return x
```

The `.square()` call is equivalent to `** 2`. The result is squared ReLU applied elementwise.

### NumPy MLP illustration

✍️
```python
def mlp(x, W1, W2):
    """
    x:  (d,)     — single token embedding
    W1: (4d, d)  — first projection (expand)
    W2: (d, 4d)  — second projection (contract)
    """
    hidden = W1 @ x                    # (4d,)
    hidden = np.maximum(0, hidden)**2  # ReLU^2 activation
    out = W2 @ hidden                  # (d,)
    return out

d = 6
np.random.seed(2)
x = np.random.randn(d)
W1 = np.random.randn(4*d, d) * 0.1
W2 = np.random.randn(d, 4*d) * 0.1

out = mlp(x, W1, W2)
print("MLP input shape:", x.shape)
print("MLP output shape:", out.shape)
print("Sparsity (fraction of hidden units that are zero):",
      (np.maximum(0, W1 @ x) == 0).mean())
```

---

## 7. Layer Normalisation

### Why we need normalisation

During training, the values flowing through a deep network can grow or shrink wildly. A single large activation can cascade through all subsequent layers, making it impossible for the optimiser to make consistent progress. **Layer normalisation** rescales the activations at each step to have a stable magnitude.

Without it, training deep transformers (12+ layers) is extremely difficult. Normalisation is one of the key ingredients that makes them trainable at all.

### RMSNorm

nanochat uses **RMSNorm** (Root Mean Square Norm), a simplified variant of the original Layer Normalisation.

Standard Layer Norm normalises using both mean and variance:
$$\text{LayerNorm}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x} - \mu}{\sigma} + \beta$$

RMSNorm skips the mean subtraction (assuming the mean is already near zero) and drops the learned shift parameter $\beta$:
$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \qquad \text{where } \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$$

In the original paper, RMSNorm also has a learned scale $\gamma$, but nanochat removes even that — making it **parameter-free**:

```python
# From nanochat/gpt.py
def norm(x):
    return F.rms_norm(x, (x.size(-1),))  # no learnable parameters
```

This simplification works well in practice and reduces the parameter count slightly.

### Pre-norm vs post-norm

In the original "Attention is All You Need" paper (2017), normalisation was applied *after* each sub-layer (post-norm). Modern transformers almost universally use **pre-norm**: apply normalisation *before* the sub-layer, then add to the residual stream.

```
Post-norm (original):   x → sublayer → (x + output) → norm
Pre-norm (modern):      x → norm → sublayer → (x + output)
```

Pre-norm is more stable during training because the residual stream $\mathbf{x}$ is never normalised before being added to the output — it remains in its natural scale, making it easier for the optimiser.

nanochat uses pre-norm throughout. Look at `Block.forward`:

```python
# From nanochat/gpt.py
def forward(self, x, ...):
    x = x + self.attn(norm(x), ...)   # pre-norm: norm(x) before attention
    x = x + self.mlp(norm(x))         # pre-norm: norm(x) before MLP
    return x
```

✍️
```python
import numpy as np

def rms_norm(x):
    """Parameter-free RMS normalisation along the last dimension."""
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-8)
    return x / rms

# Example
x = np.array([3.0, -1.0, 0.5, 2.0])
normed = rms_norm(x)
print("Original:", x)
print("RMS normed:", normed)
print("RMS of original:", np.sqrt(np.mean(x**2)))
print("RMS of normed (should be ~1.0):", np.sqrt(np.mean(normed**2)))
```

---

## 8. The Full Transformer Block

Now we can assemble everything. A single **transformer block** applies:

1. RMSNorm + Self-Attention (with residual connection)
2. RMSNorm + MLP (with residual connection)

Mathematically:

$$\mathbf{x} \leftarrow \mathbf{x} + \text{Attention}(\text{Norm}(\mathbf{x}))$$

$$\mathbf{x} \leftarrow \mathbf{x} + \text{MLP}(\text{Norm}(\mathbf{x}))$$

```
┌─────────────────────────────────────┐
│  Transformer Block                  │
│                                     │
│  x ──┬──────────────────────────►   │
│      │                              │
│      └─► RMSNorm ─► Attention ─►(+)─┤
│                                  │  │
│  x ──┬──────────────────────────►   │  (after attn residual)
│      │                              │
│      └─► RMSNorm ─► MLP ──────►(+)─┤
│                                     │
└─────────────────────────────────────┘
```

A **full transformer** is just $N$ of these blocks stacked sequentially, where the output of block $k$ becomes the input of block $k+1$:

```
Embeddings
    │
    ▼
 Block 0
    │
    ▼
 Block 1
    │
    ▼
   ...
    │
    ▼
 Block N-1
    │
    ▼
 RMSNorm
    │
    ▼
 LM Head
```

### NumPy transformer block

✍️
```python
def transformer_block(x, W_Qs, W_Ks, W_Vs, W_O, W_mlp1, W_mlp2, mask=None):
    """
    A single transformer block (simplified, single sequence no batch).
    x: (T, d)
    """
    # Pre-norm + attention + residual
    x_normed = np.array([rms_norm(x[t]) for t in range(len(x))])
    attn_out = multi_head_attention(x_normed, W_Qs, W_Ks, W_Vs, W_O, mask=mask)
    x = x + attn_out

    # Pre-norm + MLP + residual (applied independently to each position)
    x_normed2 = np.array([rms_norm(x[t]) for t in range(len(x))])
    mlp_out = np.array([mlp(x_normed2[t], W_mlp1, W_mlp2) for t in range(len(x))])
    x = x + mlp_out

    return x

# Wire it up
T, d, d_k, H = 4, 6, 3, 2
np.random.seed(3)
X = np.random.randn(T, d)
W_Qs = [np.random.randn(d, d_k) * 0.1 for _ in range(H)]
W_Ks = [np.random.randn(d, d_k) * 0.1 for _ in range(H)]
W_Vs = [np.random.randn(d, d_k) * 0.1 for _ in range(H)]
W_O  = np.random.randn(H * d_k, d) * 0.1
W_mlp1 = np.random.randn(4*d, d) * 0.1
W_mlp2 = np.random.randn(d, 4*d) * 0.1

output = transformer_block(X, W_Qs, W_Ks, W_Vs, W_O, W_mlp1, W_mlp2, mask=causal_mask(T))
print("Block output shape:", output.shape)  # (4, 6) — same shape as input
```

---

## 9. The Output Head

After all $N$ transformer blocks, we have a sequence of hidden states $\mathbf{h}_1, \ldots, \mathbf{h}_T$. Each $\mathbf{h}_t \in \mathbb{R}^d$ is a contextualised representation of position $t$, informed by all previous positions.

To predict the next token after position $t$, we use $\mathbf{h}_t$ as input to a linear **language model head**:

$$\text{logits}_t = \mathbf{h}_t W_{\text{lm\_head}} \in \mathbb{R}^V$$

where $W_{\text{lm\_head}} \in \mathbb{R}^{d \times V}$ is a learned projection matrix (the "unembedding" matrix). $V = 32{,}768$ is the vocabulary size.

### Softmax: from logits to probabilities

The raw logits are not probabilities — they can be any real number. We apply softmax to convert them:

$$P(\text{token}_j \mid \mathbf{x}_{1:t}) = \frac{e^{\text{logit}_j}}{\sum_{k=1}^{V} e^{\text{logit}_k}}$$

The result is a valid probability distribution over the entire vocabulary (all values in $[0,1]$, summing to 1).

### Cross-entropy loss during training

During training, we know what token *should* come next (it's in the training data). We measure how well the model predicts it using **cross-entropy loss**:

$$\mathcal{L} = -\log P(\text{correct token} \mid \mathbf{x}_{1:t})$$

If the model assigns probability 1.0 to the correct token, the loss is 0. If it assigns probability 0.001, the loss is $-\log(0.001) \approx 6.9$. Minimising this loss across all positions and all training examples is the entire training objective.

✍️
```python
import numpy as np

def cross_entropy_loss(logits, target_token_id):
    """
    logits:          (V,) raw scores over vocabulary
    target_token_id: int, the correct next token
    """
    # Numerically stable log-softmax
    log_probs = logits - np.log(np.sum(np.exp(logits - logits.max()))) - logits.max()
    return -log_probs[target_token_id]

V = 5  # tiny vocab
np.random.seed(0)
logits = np.random.randn(V)
target = 2

loss = cross_entropy_loss(logits, target)
probs = np.exp(logits - logits.max())
probs /= probs.sum()
print(f"Logits: {logits.round(3)}")
print(f"Probs:  {probs.round(3)}")
print(f"Probability of correct token (id={target}): {probs[target]:.3f}")
print(f"Cross-entropy loss: {loss:.3f}")
print(f"-log({probs[target]:.3f}) = {-np.log(probs[target]):.3f}")  # same thing
```

### Temperature and sampling during inference

During inference there is no "correct" token — we have to choose one. Different strategies exist:

**Greedy decoding:** always pick $\arg\max(\text{logits})$. Deterministic, but repetitive and boring.

**Temperature sampling:** divide logits by a temperature $\tau$ before softmax:
$$P_\tau(\text{token}_j) = \text{softmax}(\text{logits} / \tau)$$

- $\tau = 1.0$: sample from the model's true distribution
- $\tau < 1.0$ (e.g. 0.7): distribution becomes sharper, model is more confident
- $\tau > 1.0$ (e.g. 1.5): distribution becomes flatter, model is more "creative" (random)
- $\tau \to 0$: degenerates to greedy decoding

**Top-k sampling:** before softmax, zero out all logits except the top $k$ (by magnitude), then sample from the remaining $k$.

nanochat's `generate` method supports both:

```python
# From nanochat/gpt.py
if top_k is not None and top_k > 0:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')
if temperature > 0:
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
```

### Logit softcapping

nanochat also applies a softcap to the logits before loss computation:

```python
# From nanochat/gpt.py
softcap = 15
logits = softcap * torch.tanh(logits / softcap)
```

This squashes logits smoothly into the range $(-15, +15)$ using $\tanh$. It prevents any single token from having an astronomically high logit relative to the rest, which can destabilise training. The function $15 \cdot \tanh(x/15)$ is approximately linear near $x=0$ and saturates at $\pm 15$.

✍️
```python
import numpy as np

def softcap(logits, cap=15.0):
    return cap * np.tanh(logits / cap)

x = np.array([-100, -20, -5, 0, 5, 20, 100])
print("Raw logits: ", x)
print("Softcapped: ", softcap(x).round(2))
# Notice: values close to 0 are nearly unchanged; extreme values are clamped near ±15
```

---

## 10. Connecting to nanochat's GPT Class

Let's now map everything we have learned onto the actual nanochat code. Open `nanochat/gpt.py` and look at the key structures.

### GPTConfig

```python
# From nanochat/gpt.py
@dataclass
class GPTConfig:
    sequence_len: int = 2048   # maximum context length (T)
    vocab_size: int = 32768    # vocabulary size (V)
    n_layer: int = 12          # number of transformer blocks (N)
    n_head: int = 6            # number of attention heads (H)
    n_kv_head: int = 6         # key/value heads for GQA (= n_head for standard MHA)
    n_embd: int = 768          # embedding / residual stream dimension (d)
    window_pattern: str = "SSSL"  # sliding window pattern (advanced, Chapter 4)
```

Every parameter maps directly to what you've learned:

| Config field | What it means |
|---|---|
| `sequence_len` | Maximum $T$ — how many tokens of context |
| `vocab_size` | $V$ — size of the vocabulary lookup table |
| `n_layer` | $N$ — how many `Block` objects are stacked |
| `n_head` | $H$ — how many attention heads per layer |
| `n_embd` | $d$ — the dimension of every embedding and hidden state |

The per-head dimension is derived: `head_dim = n_embd // n_head = 768 // 6 = 128`.

### GPT.__init__: the model structure

```python
# From nanochat/gpt.py (simplified view)
self.transformer = nn.ModuleDict({
    "wte": nn.Embedding(vocab_size, n_embd),          # token embedding table
    "h": nn.ModuleList([Block(config, i)               # N transformer blocks
                        for i in range(n_layer)]),
})
self.lm_head = Linear(n_embd, vocab_size, bias=False) # output projection
```

This is exactly our architecture:
1. `wte` = the embedding matrix $E \in \mathbb{R}^{V \times d}$
2. `h` = the list of transformer blocks
3. `lm_head` = the unembedding matrix $W_{\text{lm\_head}} \in \mathbb{R}^{d \times V}$

Note that nanochat uses **untied weights**: `wte` and `lm_head` are *separate* matrices. Some models tie them (the input embedding is reused as the output projection transposed) to save parameters. nanochat explicitly does not do this.

### GPT.forward: the data flow

```python
# From nanochat/gpt.py (simplified view, ignoring advanced features)
def forward(self, idx, targets=None):
    # 1. Embed tokens
    x = self.transformer.wte(idx)   # (B, T, d)
    x = norm(x)                      # RMSNorm after embedding

    # 2. Pass through all transformer blocks
    for block in self.transformer.h:
        x = block(x, ...)            # (B, T, d)

    # 3. Final norm
    x = norm(x)

    # 4. Project to vocabulary logits
    logits = self.lm_head(x)         # (B, T, V)
    logits = softcap * torch.tanh(logits / softcap)

    # 5. Compute loss if training
    if targets is not None:
        loss = F.cross_entropy(logits, targets)
        return loss
    else:
        return logits
```

Compare this to our mathematical description:

| Code | Math |
|------|------|
| `wte(idx)` | Look up $E[\text{token}_t]$ for each $t$ |
| `norm(x)` | Apply RMSNorm |
| `block(x, ...)` | $x \leftarrow x + \text{Attn}(\text{Norm}(x))$, then $x \leftarrow x + \text{MLP}(\text{Norm}(x))$ |
| `lm_head(x)` | $\text{logits} = x W_{\text{lm\_head}}$ |
| `F.cross_entropy(...)` | $\mathcal{L} = -\log P(\text{correct token})$ |

### Block and the residual stream

```python
# From nanochat/gpt.py
class Block(nn.Module):
    def forward(self, x, ...):
        x = x + self.attn(norm(x), ...)   # attention sub-layer, residual add
        x = x + self.mlp(norm(x))          # MLP sub-layer, residual add
        return x
```

Two lines. That is all a transformer block is. Pre-norm, sublayer, residual add. Twice.

### Parameter counts at default config

Let's compute approximate parameter counts to build intuition:

✍️
```python
# nanochat default config: n_embd=768, n_head=6, n_layer=12, vocab_size=32768

n_embd   = 768
n_head   = 6
n_layer  = 12
vocab    = 32768
head_dim = n_embd // n_head  # 128

# Token embedding matrix
wte_params = vocab * n_embd
print(f"Token embeddings (wte):  {wte_params:>12,}  ({wte_params/1e6:.1f}M)")

# LM head (unembedding)
lm_head_params = n_embd * vocab
print(f"LM head:                 {lm_head_params:>12,}  ({lm_head_params/1e6:.1f}M)")

# Per transformer block:
# Attention: W_Q, W_K, W_V each (d, d), W_O (d, d)
attn_per_block = 4 * n_embd * n_embd
# MLP: W_fc (d, 4d) + W_proj (4d, d)
mlp_per_block = 2 * n_embd * 4 * n_embd
block_params = attn_per_block + mlp_per_block
print(f"Per block (attn+mlp):    {block_params:>12,}  ({block_params/1e6:.1f}M)")
print(f"All blocks (x{n_layer}):      {block_params*n_layer:>12,}  ({block_params*n_layer/1e6:.1f}M)")

total = wte_params + lm_head_params + block_params * n_layer
print(f"\nTotal (approx):          {total:>12,}  ({total/1e6:.1f}M params)")
```

```
Token embeddings (wte):    25,165,824  (25.2M)
LM head:                   25,165,824  (25.2M)
Per block (attn+mlp):       4,718,592  (4.7M)
All blocks (x12):          56,623,104  (56.6M)

Total (approx):           106,954,752  (107.0M params)
```

A 107M parameter model is a "small" language model by modern standards (GPT-3 is 175B, GPT-4 is estimated at ~1.8T). But it is large enough to learn real language patterns, and small enough to train and run on a single GPU.

### What is not covered yet

nanochat's actual implementation has several additions beyond this baseline:

- **Rotary positional embeddings (RoPE):** How the model knows the *position* of each token. Chapter 4.
- **Group-Query Attention (GQA):** Sharing K and V heads across Q heads to reduce memory. Chapter 4.
- **Sliding window attention:** Some layers only attend to a local context window. Chapter 4.
- **QK normalisation:** Normalising Q and K before the dot product. Chapter 4.
- **Value embeddings (ResFormer-style):** An additional residual path through the V computation. Chapter 5.
- **Smear gate / backout:** Experimental features for blending information. Chapter 5.

These are refinements and optimisations. The foundations you have learned here are the core of every transformer, including nanochat.

---

## Check Your Understanding

**Question 1.** The attention formula is $\text{Attention}(Q, K, V) = \text{softmax}(QK^\top / \sqrt{d_k}) \, V$.

- What would happen if we removed the $\sqrt{d_k}$ scaling factor? (Hint: think about what large dot products do to softmax.)
- Why do we apply softmax row-wise rather than to the whole matrix?
- What does the causal mask do to the upper triangle of the score matrix, and why?

**Question 2.** Consider the residual connection $\mathbf{x} \leftarrow \mathbf{x} + \text{sublayer}(\mathbf{x})$.

- If the sublayer's parameters are initialised to zero (so its output starts as zero), what does the model look like at the beginning of training?
- Why is this a good property? (Hint: look at how nanochat initialises `c_proj` in `init_weights`.)

**Question 3.** nanochat uses a vocabulary size of 32,768 and `n_embd = 768`.

- What is the shape of the embedding matrix `wte`?
- What is the shape of the output produced by `lm_head` for a batch of 2 sequences each with 512 tokens?
- Given that nanochat uses untied weights, how many total parameters do `wte` and `lm_head` together contain?

---

## What's Next

You now understand the complete mathematical story of a language model from raw text all the way to a probability distribution over next tokens. You have implemented each piece from scratch in NumPy.

In **Chapter 3**, we will look at the training loop: how gradient descent adjusts the model's parameters to minimise cross-entropy loss, what a training batch looks like, and how to run nanochat's training script on a small dataset.

In **Chapter 4**, we will return to the advanced features of nanochat's attention mechanism — RoPE, GQA, sliding window attention, and QK normalisation — now that you have the foundation to appreciate what each one is solving.

---

*Chapter 2 complete. The transformer is not magic — it is matrix multiplications, softmax, and residual additions, stacked twelve times.*
