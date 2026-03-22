# Chapter 5: Multi-Head Self-Attention

## What you'll learn

- Why tokens need to "look at" each other and how dot-product attention makes that possible
- How Q, K, and V projections divide responsibility for attending
- Why we scale by `sqrt(head_dim)` and what happens if we don't
- How splitting into multiple heads lets the model attend to different aspects simultaneously
- What the KV cache is, why it exists, and how the sequential calling pattern in `microgpt.py` enforces causal attention without an explicit mask
- How the residual connection around the attention block enables gradient flow through deep networks

---

## Prerequisites

- You have read chapters 1-4 (setup, tokenization, autograd, model foundations)
- You can read Python list comprehensions fluently
- You are comfortable with the dot product of two vectors (multiply element-wise, sum the results)
- You do not need to know any linear algebra beyond that

---

## The big picture: why do tokens need to attend to each other?

Consider generating the next character after the sequence `"ann"`. The model has seen three tokens. To make a good prediction it needs to know not just what the current token is, but also what came before it — is this the end of a short name, or the middle of a longer one?

Self-attention is the mechanism that lets every token gather information from every other token in the context. The result for each token is a weighted blend of all the value vectors it has seen, where the weights are determined by how relevant each past token is to the current one.

---

## ASCII diagram: scaled dot-product attention (one head)

```
Current token x  (n_embd = 16 dims)
       |
  _____|___________
 |     |           |
 Wq    Wk          Wv
 |     |           |
 q     k           v        <- all 16-dim vectors
       |           |
       +-- append to KV cache --+
                                |
For each past token t:          |
  keys[li]   = [k0, k1, ... kt] |   (t+1 entries, each 16 dims)
  values[li] = [v0, v1, ... vt] |   (t+1 entries, each 16 dims)
                                |
Split into n_head=4 heads (head_dim=4 each):
                                |
  head 0: q[0:4]  k_h[t][0:4]  v_h[t][0:4]
  head 1: q[4:8]  k_h[t][4:8]  v_h[t][4:8]
  head 2: q[8:12] k_h[t][8:12] v_h[t][8:12]
  head 3: q[12:]  k_h[t][12:]  v_h[t][12:]
                                |
For each head h:
  attn_logits[t] = dot(q_h, k_h[t]) / sqrt(head_dim)
  attn_weights   = softmax(attn_logits)   <- sums to 1
  head_out       = sum_t( attn_weights[t] * v_h[t] )
                                |
  Concatenate head outputs -> x_attn  (16 dims)
       |
  Output projection Wo  (16x16 linear)
       |
  + x_residual   (residual connection)
       |
  x   (16 dims, ready for MLP block)
```

---

## Step-by-step walkthrough

The attention block lives inside the `gpt()` function at
[`microgpt.py` lines 108-134](../microgpt.py#L108).

### Step 1 — Token and position embeddings (lines 109-112)

```python
tok_emb = state_dict['wte'][token_id]
pos_emb = state_dict['wpe'][pos_id]
x = [t + p for t, p in zip(tok_emb, pos_emb)]
x = rmsnorm(x)
```

`wte` is the **word token embedding** table, shape `(vocab_size, 16)`. Row `token_id` is a learned 16-dimensional vector that encodes the identity of the token.

`wpe` is the **word position embedding** table, shape `(block_size, 16)`. Row `pos_id` encodes where in the sequence this token sits.

We add them element-wise. The result `x` is a single 16-dim vector that carries both *what* the token is and *where* it appears.

> **Why not concatenate?** Addition keeps the vector the same size and lets the model learn how to mix identity and position information through training. Concatenation would double the width and increase parameter count for no observed benefit in practice.

The initial `rmsnorm` normalises `x` before it enters the residual stream. This is unusual (GPT-2 puts the norm inside the residual branch, not before it) but the comment in the code notes it is not redundant because of how gradients flow through the residual connection during the backward pass.

---

### Step 2 — Q, K, V projections (lines 118-120)

```python
q = linear(x, state_dict[f'layer{li}.attn_wq'])
k = linear(x, state_dict[f'layer{li}.attn_wk'])
v = linear(x, state_dict[f'layer{li}.attn_wv'])
```

Each weight matrix is `(16, 16)`. `linear(x, W)` computes `W @ x`, producing a 16-dim output vector.

Think of these three projections as assigning three different roles to the current token:

| Vector | Role | Intuition |
|--------|------|-----------|
| `q` (query) | "What am I looking for?" | What information would help me predict the next token? |
| `k` (key)   | "What do I advertise?"   | What information can I provide to other tokens? |
| `v` (value) | "What do I give?"        | The actual content to share when I am attended to. |

In other words: **`q`** is what this token is *looking for* when it queries the context; **`k`** is what it *advertises* about itself — future positions will compare their queries against keys stored in the cache; **`v`** is the *content* it will contribute if another token attends to it. The next step appends `k` and `v` so later tokens can retrieve that content.

All three start from the same `x`, but the three weight matrices are trained independently, so the model learns to project `x` into three distinct spaces.

---

### Step 3 — The KV cache (lines 121-122)

```python
keys[li].append(k)
values[li].append(v)
```

`keys` and `values` are passed into `gpt()` from the outside. Look at the training loop ([lines 161-165](../microgpt.py#L161)):

```python
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
for pos_id in range(n):
    token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
    logits = gpt(token_id, pos_id, keys, values)
```

Each call to `gpt()` processes exactly one token at one position. Before the call, `keys[li]` contains the `k` vectors from all previous positions. After `append(k)`, it contains all previous plus the current one.

This is the **KV cache**: once a token's `k` and `v` are computed, they are stored and reused for every future token in the same sequence. We never recompute them.

> **What's happening:** At position 0 (first token), `keys[0]` starts empty, grows to 1 entry after the append. At position 4, `keys[0]` has 5 entries. When computing attention for position 4, the query `q` from position 4 can look at all 5 keys — itself and all four predecessors. This is causal attention, achieved not by masking but by the sequential calling pattern.

---

### Step 4 — Causal attention without a mask

In the original "Attention Is All You Need" paper, and in standard implementations of GPT that process a full sequence in a single forward pass, a **causal mask** (an upper-triangular matrix of `-inf` values) is applied to the attention logits before softmax. This prevents token at position `t` from attending to tokens at positions `t+1, t+2, ...`.

`microgpt.py` does not need this mask. Because `gpt()` is called one token at a time and `keys[li]` only holds keys from positions `0` through `current`, it is structurally impossible for the current query to see future keys — they have not been appended yet.

This is a deliberate simplification. It is correct, but it means the code as written cannot be parallelised across positions during training in the way a masked-attention implementation can. For a model this small (16-dim embeddings, 27-character vocabulary), the sequential approach is fine.

---

### Step 5 — Splitting into heads (lines 124-128)

```python
for h in range(n_head):
    hs = h * head_dim
    q_h = q[hs:hs+head_dim]
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
    v_h = [vi[hs:hs+head_dim] for vi in values[li]]
```

With `n_head=4` and `head_dim=4`, the 16-dim vectors are partitioned into four non-overlapping 4-dim slices:

```
Index:  0  1  2  3  |  4  5  6  7  |  8  9 10 11  | 12 13 14 15
        head 0       |   head 1     |    head 2    |    head 3
```

`hs = h * head_dim` is just the starting index of head `h`'s slice. `q[hs:hs+head_dim]` extracts 4 elements.

For `k_h`, we need the same slice from every key vector stored in `keys[li]`. The list comprehension `[ki[hs:hs+head_dim] for ki in keys[li]]` does that: it iterates over all cached keys and extracts the relevant 4 dims from each.

For a fixed head `h`, **`q_h`** is a single vector of length `head_dim` — the query for the **current** position only. **`k_h`** and **`v_h`** are parallel lists: one `head_dim` slice per **cached** position, so each has length `len(keys[li])` (the same as the number of tokens processed so far in this forward pass). If you are on the third token, each list has three entries (positions 0, 1, and 2). Step 6 will compute one attention logit per entry — three logits in that case.

Picture three words arriving in order: **"the"**, then **"cat"**, then **"sat"**. While processing **"the"**, the cache has one key/value pair — for that head, `k_h` and `v_h` each have length 1. While processing **"cat"**, they have length 2 ( **"the"** and **"cat"** ). While processing **"sat"**, they have length 3. The model never needs a causal mask here because the cache only ever holds past and current tokens (Step 4).

> **Why multiple heads?** A single head computes one weighted blend of values. Different heads attend to different subspaces of the embedding and can therefore learn to track different kinds of relationships simultaneously — one head might track whether characters are repeated, another might track position within a name. The outputs are concatenated, so the final result carries information from all of them.

---

### Step 6 — Scaled dot-product attention (lines 129-131)

```python
attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
               for t in range(len(k_h))]
attn_weights = softmax(attn_logits)
head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
            for j in range(head_dim)]
```

This step implements **scaled dot-product attention** for that head: **logits** (match scores) → **softmax** (weights) → **`head_out`** (weighted mix of value slices). The loop over `h` repeats this for every head.

**Logits — how much does each past token match?**

```python
attn_logits = [
    sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
    for t in range(len(k_h))
]
```

This yields one score per past index `t`: the dot product of the current query with that token's key, scaled by `√head_dim`. With `head_dim = 4` and three cached tokens, `head_dim**0.5` is `2.0`, so:

```
attn_logits[0] = (q_h · k_h[0]) / 2.0   # match to token at position 0
attn_logits[1] = (q_h · k_h[1]) / 2.0   # match to token at position 1
attn_logits[2] = (q_h · k_h[2]) / 2.0   # match to token at position 2 (can be "self")
```

Higher means the current query aligns more strongly with that key. The dot product is a similarity measure: same direction → large positive; orthogonal → near zero.


**Why divide by `sqrt(head_dim)`?** The core question is: why does the dot product's variance grow with the head dimension, and why does dividing by `sqrt(head_dim)` fix it?

**Why does variance grow with `d`?** Write `d = head_dim`. Suppose each element of `q` and `k` is drawn from `N(0, 1)` (mean 0, variance 1). The dot product is

```
q · k = q[0]*k[0] + q[1]*k[1] + ... + q[d-1]*k[d-1]
```

For each `j`, the term `q[j]*k[j]` is a product of two independent standard normals. For independent random variables `X` and `Y`,

```
Var(X·Y) = Var(X)·Var(Y) + Var(X)·E[Y]² + Var(Y)·E[X]²
```

With `E[X] = E[Y] = 0` and `Var(X) = Var(Y) = 1`, this gives `Var(q[j]*k[j]) = 1`. The terms for different `j` are independent, so by the variance addition rule,

```
Var(q · k) = 1 + 1 + ... + 1  (d terms)  =  d
```

The standard deviation of `q · k` is therefore `√d`.

**Why is that a problem?** When `d` is small, logits stay moderate and softmax spreads attention across tokens. When `d` is large, the raw dot products spread out: some logits are large positive, some large negative. Softmax maps large gaps to weights that are nearly 1 on one token and nearly 0 elsewhere — a near-one-hot distribution. Gradients through a saturated softmax are tiny almost everywhere, which stalls learning.

**The fix is exact, not approximate.** For any constant `c`, `Var(X / c) = Var(X) / c²`. If `Var(q · k) = d`, then

```
Var((q · k) / √d) = d / (√d)² = d/d = 1
```

So dividing by `√d` brings the variance of the logits back to 1 regardless of `d`, keeping softmax in a regime where gradients remain useful. By contrast, dividing by `d` would give variance `1/d`, which shrinks toward 0 as `d` grows and can starve gradients for a different reason. A fixed divisor (like 2) only matches one particular `d`. `√d` is the unique scale that exactly cancels variance growth proportional to `d`.

**Softmax — scores become a probability vector.**

```python
attn_weights = softmax(attn_logits)
```

Softmax turns raw scores into non-negative weights that sum to 1.

**Head output — weighted blend of value vectors.**

```python
head_out = [
    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
    for j in range(head_dim)
]
```

`v_h` has shape `[n_tokens, head_dim]`: one value vector per past position. For each output dimension `j`, the code forms a **weighted sum** of the `j`-th components across those vectors — the same weights `attn_weights[t]` for every `j`.

So `head_out` is one `head_dim`-dimensional vector: a soft mixture of all past value vectors, with mixture weights given by attention. If `attn_weights` were one-hot — e.g. `[0, 1, 0]` — `head_out` would equal that past token's value vector exactly (hard retrieval). Softmax usually produces soft weights, so the result is a blend.

The length of `attn_logits` (and thus `attn_weights`) is `len(k_h)`, which grows with sequence position; how far back the current token can look is exactly what Steps 3–5 set up.


---

### Step 7 — Output projection and residual connection (lines 133-134)

```python
x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
x = [a + b for a, b in zip(x, x_residual)]
```

`x_attn` is the concatenation of all four head outputs: 4 heads × 4 dims = 16 dims. The output projection `attn_wo` is a `(16, 16)` matrix that mixes information across heads and projects the concatenated result back into the residual stream.

The residual connection adds the pre-attention `x_residual` to the projected output. This has two important effects:

1. **Gradient flow.** During the backward pass, the gradient of the loss flows back through both paths: through the attention transformation and directly through the addition. The direct path means that early layers receive a clean gradient signal even when the network is deep.

2. **Identity initialisation.** At the start of training, when weights are near zero, the attention block's contribution is near zero and `x ≈ x_residual`. This means the network starts close to an identity function and learns incrementally, which is more stable than starting from a random transformation.

---

## Hands-on exercises

### Exercise 1 — Trace attention for a 3-token sequence

Open a Python REPL and paste the following. It reproduces the attention calculation for one head using plain floats (no `Value` objects).

```python
import math, random

random.seed(0)
head_dim = 4
n_embd   = 16

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def softmax(logits):
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]

# Simulate three tokens arriving one at a time.
# We'll use random vectors in place of real Q/K/V projections.
keys_cache   = []
values_cache = []

for pos in range(3):
    q = [random.gauss(0, 1) for _ in range(head_dim)]
    k = [random.gauss(0, 1) for _ in range(head_dim)]
    v = [random.gauss(0, 1) for _ in range(head_dim)]

    keys_cache.append(k)
    values_cache.append(v)

    # Scaled dot-product attention for this head
    attn_logits  = [dot(q, keys_cache[t]) / math.sqrt(head_dim)
                    for t in range(len(keys_cache))]
    attn_weights = softmax(attn_logits)
    head_out     = [sum(attn_weights[t] * values_cache[t][j]
                        for t in range(len(values_cache)))
                    for j in range(head_dim)]

    print(f"pos={pos}")
    print(f"  attn_weights = {[round(w, 4) for w in attn_weights]}")
    print(f"  sum of weights = {sum(attn_weights):.6f}")
    print(f"  head_out = {[round(o, 4) for o in head_out]}")
    
```

Expected output shape (values will differ due to random seeds):

```
pos=0
  attn_weights = [1.0]
  sum of weights = 1.000000
  head_out = [...]
pos=1
  attn_weights = [0.xxxx, 0.xxxx]
  sum of weights = 1.000000
  head_out = [...]
pos=2
  attn_weights = [0.xxxx, 0.xxxx, 0.xxxx]
  sum of weights = 1.000000
  head_out = [...]
```

At position 0 the query can only attend to itself, so the single weight is 1.0. At position 2 you have three weights that sum to 1.0. Verify the sum is exactly 1.0 for every position.

---

### Exercise 2 — Verify attn_weights sum to 1.0

Extend the script above to assert the sum:

```python
assert abs(sum(attn_weights) - 1.0) < 1e-9, \
    f"Weights don't sum to 1: {sum(attn_weights)}"
print("All weight sums check out.")
```

This is a fast sanity check you can add to your own experiments any time you implement attention from scratch.

---

### Exercise 3 — Inspect the KV cache after 3 tokens

Add this at the end of the loop from exercise 1:

```python
print("\n--- KV cache state after 3 tokens ---")
print(f"Number of entries in keys_cache:   {len(keys_cache)}")
print(f"Number of entries in values_cache: {len(values_cache)}")
print(f"Shape of each key:   {len(keys_cache[0])} dims")
print(f"Shape of each value: {len(values_cache[0])} dims")
print("\nAll cached keys:")
for i, k in enumerate(keys_cache):
    print(f"  pos {i}: {[round(x, 4) for x in k]}")
```

You should see three key vectors, one per position. In `microgpt.py`, `keys[li]` (for layer index `li`) plays exactly this role. After processing a name like `"ann"` (3 characters, plus the BOS token = 4 tokens), `keys[0]` has 4 entries.

---

## Check your understanding

1. **Q, K, V roles.** If you had to describe in one sentence each what `q`, `k`, and `v` represent for a given token, what would you say? Why do we need all three rather than just using `x` directly for the dot product?

2. **The scaling factor.** Change `head_dim**0.5` to `1` in the attention logit computation from exercise 1 and re-run. What happens to the attention weight distribution at positions 1 and 2? Is the distribution more or less concentrated?

3. **KV cache growth.** If you generate a sequence of length `block_size = 16`, how many key vectors will be stored in `keys[0]` by the time the last token is processed? How many total float values does that represent, given `n_embd = 16`?

4. **Causal masking.** Explain in one or two sentences why `microgpt.py` does not need an attention mask even though it implements causal (autoregressive) attention. What would break if you collected all keys for an entire sequence upfront and then ran attention?

5. **Residual connection.** The line `x = [a + b for a, b in zip(x, x_residual)]` is a residual connection. What would happen to the gradient of the loss with respect to the embeddings `wte` if this line were removed and replaced with just `x = x_attn_out`?

6. **Multi-head intuition.** With `n_head=4` and `head_dim=4`, each head attends over a 4-dim subspace. Could you get the same expressiveness with a single head operating over all 16 dims? What is the argument for preferring multiple smaller heads?

---

## What's next

Chapter 5 completes the attention block. After the residual addition on line 134, control passes to the **MLP block** (lines 136-141):

```python
x_residual = x
x = rmsnorm(x)
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
x = [xi.relu() for xi in x]
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
x = [a + b for a, b in zip(x, x_residual)]
```

The MLP applies two linear projections with a ReLU non-linearity between them. It is often described as the part of the transformer that "stores facts" — while attention routes information between tokens, the MLP transforms the resulting representation at each position independently.

Chapter 6 will cover the MLP block, why its hidden layer is 4× wider than `n_embd`, and how ReLU enables the network to approximate arbitrary functions.
