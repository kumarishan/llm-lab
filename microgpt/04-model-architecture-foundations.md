# Chapter 4: Model Architecture Foundations

## What you'll learn

- What hyperparameters control the size and shape of the model, and what each one means geometrically
- How model parameters are allocated and why they are `Value` objects
- How to count parameters by hand and cross-check against the code
- How the three building-block functions — `linear`, `softmax`, and `rmsnorm` — work, including the numerical stability decisions baked into each one

---

## Prerequisites

- **Chapter 1** — project setup and running the script
- **Chapter 2** — how the tokenizer maps characters to integers and how `vocab_size` is computed
- **Chapter 3** — the `Value` class, the computation graph, and how `.backward()` propagates gradients

If you have not read those chapters, `state_dict`, `params`, and the three helper functions in this chapter will feel unmotivated. Read them first.

---

## Source location

All code in this chapter lives in `microgpt.py` at lines 74–106.

---

## The four hyperparameters

```python
# microgpt.py lines 75-79
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd  = 16    # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window
n_head  = 4     # number of attention heads
head_dim = n_embd // n_head  # derived: 16 // 4 = 4
```

These five values are the entire geometry of the model. Before any parameter is created, these numbers determine exactly how much memory the model needs and what shapes every matrix will have. Understanding them concretely makes the rest of the code legible.

### `n_embd = 16` — the embedding dimension

Every token, at every position in the sequence, is represented as a vector of `n_embd` numbers. This vector is the model's internal "thought" about that token in context. Real models use 768 (GPT-2 small) or 4096 (LLaMA-3 8B). Here it is 16, small enough to fit in your head and run in pure Python.

```
token "a"  →  [0.12, -0.03, 0.87, ..., 0.44]  (16 numbers)
```

### `n_layer = 1` — depth

Each transformer layer reads the current vector for every position and refines it. Stacking layers lets the model combine evidence from different distances and abstraction levels. With one layer the model is shallow but fully functional and much easier to trace.

### `block_size = 16` — context window

The model can attend to at most `block_size` previous tokens. The names dataset never exceeds 15 characters, so 16 is sufficient without wasting memory on position embeddings that will never fire.

### `n_head = 4` and `head_dim = 4`

Multi-head attention splits the 16-dimensional vector into `n_head` independent slices of `head_dim` numbers each. Each head learns to attend based on different criteria (e.g., one head for recency, another for vowel patterns). Splitting and recombining costs nothing extra in parameters because `n_head * head_dim == n_embd`.

```
n_embd = 16 = 4 heads × 4 dimensions per head
```

---

## Parameter initialization

### The `matrix` factory

```python
# microgpt.py line 80
matrix = lambda nout, nin, std=0.08: \
    [[Value(random.gauss(0, std)) for _ in range(nin)]
     for _ in range(nout)]
```

`matrix(nout, nin)` returns a list of `nout` rows, each containing `nin` `Value` objects. The result is a 2D Python list — no NumPy, no tensors.

Every weight is sampled from a Gaussian distribution with mean 0 and standard deviation 0.08.

**Why `Value` objects?**
Parameters must flow through the computation graph so that `.backward()` can compute `dLoss/dWeight` for every single weight. A plain `float` has no `grad` attribute and no way to record what operations it participated in. Wrapping each scalar in `Value` solves both problems at the cost of speed, which is acceptable for a pedagogical model.

**Why `std=0.08`?**
If weights start too large, the outputs of early layers are large, activation functions saturate, and gradients vanish immediately. If weights start too small, the signal shrinks to zero as it passes through layers. `std=0.08` is a reasonable small value that keeps activations in a sane range at initialization. For reference, GPT-2 uses `std=0.02` scaled by `1/sqrt(2*n_layer)`.

### The `state_dict`

```python
# microgpt.py lines 81-88
state_dict = {
    'wte':     matrix(vocab_size, n_embd),   # token embeddings
    'wpe':     matrix(block_size, n_embd),   # position embeddings
    'lm_head': matrix(vocab_size, n_embd),   # output projection
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
```

A `dict` keyed by human-readable strings is the natural way to name model weights. The name `state_dict` deliberately mirrors the PyTorch convention (`model.state_dict()`). If you have ever fine-tuned a Hugging Face model, you have already worked with objects that behave exactly like this dict.

**What each matrix stores:**

| Key | Shape | Role |
|-----|-------|------|
| `wte` | `vocab_size × n_embd` | One embedding vector per token in the vocabulary |
| `wpe` | `block_size × n_embd` | One embedding vector per position (0–15) |
| `lm_head` | `vocab_size × n_embd` | Projects the final hidden state back to a score over all vocabulary tokens |
| `layer0.attn_wq/k/v/o` | `n_embd × n_embd` | Query, Key, Value, and output projection matrices for attention |
| `layer0.mlp_fc1` | `4*n_embd × n_embd` | First MLP layer, expands dimension by 4× |
| `layer0.mlp_fc2` | `n_embd × 4*n_embd` | Second MLP layer, contracts back to `n_embd` |

> **The 4× expansion in the MLP** is a design choice from the original Transformer paper. The intuition is that the model benefits from having a wider "working space" for reasoning before projecting back down. Nearly every transformer architecture since 2017 has kept this ratio.

### Flattening into `params`

```python
# microgpt.py line 89
params = [p for mat in state_dict.values() for row in mat for p in row]
```

This triple-nested list comprehension unrolls every matrix in the dict into a single flat list of `Value` objects. The optimizer in a later chapter iterates over this list to update every weight in one loop. Think of it as the model's complete parameter roster.

---

## Counting parameters by hand

The `state_dict` was built with `vocab_size = 28` (26 lowercase letters plus one BOS token, confirmed by running the script on the names dataset). With `n_embd = 16`, `block_size = 16`, and `n_layer = 1`:

**Global tables:**

```
wte:     28  ×  16  =   448
wpe:     16  ×  16  =   256
lm_head: 28  ×  16  =   448
                       -----
                        1152
```

**Per-layer matrices (layer 0 only, since n_layer = 1):**

```
attn_wq:  16 × 16 =  256
attn_wk:  16 × 16 =  256
attn_wv:  16 × 16 =  256
attn_wo:  16 × 16 =  256
          subtotal:  1024

mlp_fc1:  64 × 16 = 1024
mlp_fc2:  16 × 64 = 1024
          subtotal: 2048
```

**Grand total:**

```
1152 + 1024 + 2048 = 4224
```

> **Note on the comment in the source file:** Line 90 reads `# 21532`. That number is from an earlier version of the model with different hyperparameters. The code that follows that comment is correct; only the inline remark is stale. Running the script will print the actual parameter count for the hyperparameters currently in the file. The arithmetic above matches the current configuration.

For context: GPT-2 small has 124 million parameters. This model has roughly 4,000 — about 30,000 times smaller. It can learn to generate plausible names but nothing more, which is exactly what it was designed for.

---

## The three building-block functions

Every computation in the transformer model — embeddings, attention, the MLP — is assembled from three functions. Understanding these three functions means understanding 80% of the forward pass.

### 1. `linear(x, w)` — matrix-vector multiplication

```python
# microgpt.py lines 94-95
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

**What it does:** multiplies a weight matrix `w` by a vector `x` and returns a new vector.

The outer list comprehension iterates over rows `wo` of `w`. For each row it computes the dot product with `x`. The result is a list of `len(w)` scalars — one per output neuron.

**Spelled out for a 2×3 example:**

```
x  = [x0, x1, x2]          (input, 3 numbers)
w  = [[w00, w01, w02],      (weight matrix, 2 rows × 3 cols)
      [w10, w11, w12]]

linear(x, w) = [
    w00*x0 + w01*x1 + w02*x2,   # dot product of row 0 with x
    w10*x0 + w11*x1 + w12*x2,   # dot product of row 1 with x
]
→ output has 2 numbers
```

**Why it works for autograd:** `wi * xi` calls `Value.__mul__` and `sum(...)` chains `Value.__add__` calls. Every intermediate `Value` node records its children. When `.backward()` runs later, gradients flow back through every multiplication and addition, all the way to the weights.

**Shape rule:** if `x` has `nin` elements and `w` has shape `(nout, nin)`, the output has `nout` elements. This is the standard linear layer, identical to `torch.nn.Linear` with no bias.

---

### 2. `softmax(logits)` — probability distribution over vocabulary

```python
# microgpt.py lines 97-101
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

**What it does:** converts a list of raw scores (logits) into a probability distribution — a list of non-negative numbers that sum to 1.

The mathematical definition is:

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

**The numerical stability trick — why `max_val` is subtracted:**

`exp(z)` grows extremely fast. With logits like `[100, 101, 102]`, naive `exp(102)` overflows to `inf` in floating point, making the division undefined. Subtracting the maximum value before exponentiating does not change the result mathematically:

```
exp(z_i - max) / sum_j exp(z_j - max)
= exp(z_i) * exp(-max) / (sum_j exp(z_j) * exp(-max))
= exp(z_i) / sum_j exp(z_j)
```

The `exp(-max)` factors cancel exactly. After the shift, the largest logit becomes `exp(0) = 1` and all others are `exp(negative)`, safely in `(0, 1]`. No overflow.

**Why `max_val` uses `.data` and not a `Value`:**

```python
max_val = max(val.data for val in logits)
```

`max_val` is extracted as a plain Python float, not a `Value`. This is intentional. The `max` operation has a non-differentiable kink (its gradient with respect to non-maximum inputs is zero, and the gradient at the maximum is technically undefined). By extracting a scalar constant we avoid trying to differentiate through `max`. The shift is just a numerical trick; the gradient computation proceeds correctly through the `exp` and division that follow.

---

### 3. `rmsnorm(x)` — Root Mean Square normalization

```python
# microgpt.py lines 103-106
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

**What it does:** rescales the vector `x` so its root mean square is close to 1.

The formula is:

```
RMSNorm(x)_i = x_i / sqrt( mean(x^2) + epsilon )
```

**Step by step:**

1. `ms` — mean of the squared elements: `(x0² + x1² + ... + xn²) / n`
2. `scale = (ms + 1e-5) ** -0.5` — this is `1 / sqrt(ms + epsilon)`, computed using `Value.__pow__` with exponent `-0.5`
3. Every element of `x` is multiplied by this single scalar

**The `1e-5` epsilon:** prevents division by zero when `x` is the zero vector. In practice the RMS is never exactly zero after a few training steps, but defensive code avoids the edge case from the start.

**How RMSNorm differs from LayerNorm:**

| | LayerNorm | RMSNorm |
|---|---|---|
| Subtracts mean | Yes | No |
| Divides by std | Yes | Divides by RMS (no centering) |
| Learned scale γ | Yes (one per dimension) | Not in this implementation |
| Learned bias β | Yes | Not in this implementation |

LayerNorm adds learned parameters `γ` and `β` that the model can use to re-scale and re-center after normalization. This implementation strips both. It also skips mean subtraction. The result is simpler, fewer parameters, and slightly faster — with negligible quality loss at this scale. Modern architectures (LLaMA, Mistral, Gemma) have converged on RMSNorm for the same reasons.

**Why normalize at all?** Without normalization, activations can grow or shrink exponentially across layers. Training becomes unstable and gradients explode or vanish. Normalization keeps every vector in a consistent magnitude range regardless of what the model has learned so far, making optimization much more reliable.

---

## Hands-on exercises

Work through these in a Python REPL with `microgpt.py` in your path. After the script runs once, you can import from it or just copy the relevant code.

### Step 1: Verify the parameter count

Copy this snippet into a fresh Python file or REPL:

```python
import random
random.seed(42)

# Paste the Value class from microgpt.py here, or import it.
# Then set up the same hyperparameters:

vocab_size = 28   # adjust if your run prints a different vocab size
n_embd     = 16
block_size = 16
n_head     = 4
n_layer    = 1

# Tally by component:
counts = {
    'wte':      vocab_size * n_embd,
    'wpe':      block_size * n_embd,
    'lm_head':  vocab_size * n_embd,
    'attn_wq':  n_embd * n_embd,
    'attn_wk':  n_embd * n_embd,
    'attn_wv':  n_embd * n_embd,
    'attn_wo':  n_embd * n_embd,
    'mlp_fc1':  4 * n_embd * n_embd,
    'mlp_fc2':  n_embd * 4 * n_embd,
}
for name, count in counts.items():
    print(f"{name:12s}: {count:5d}")
print(f"{'TOTAL':12s}: {sum(counts.values()):5d}")
```

Does the total match what the script prints when you run it? (Remember to use the `vocab_size` your script actually prints, not the hardcoded 28 above.)

### Step 2: Call `linear` manually

```python
from microgpt import Value  # or paste the class inline

# A 2-output, 3-input linear layer
x = [Value(1.0), Value(2.0), Value(3.0)]
w = [
    [Value(1.0), Value(0.0), Value(-1.0)],   # row 0
    [Value(0.0), Value(1.0),  Value(2.0)],   # row 1
]

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

out = linear(x, w)
print([o.data for o in out])
# Row 0: 1*1 + 0*2 + (-1)*3 = -2
# Row 1: 0*1 + 1*2 +   2*3 =  8
# Expected: [-2.0, 8.0]
```

Now call `.backward()` on one output and inspect `x[0].grad`. Can you predict the gradient before running it?

> **Hint:** `out[0] = x[0]*w[0][0] + x[1]*w[0][1] + x[2]*w[0][2]`. The gradient of `out[0]` with respect to `x[0]` is `w[0][0].data`.

### Step 3: Verify that `softmax` outputs sum to 1

```python
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

logits = [Value(1.0), Value(2.0), Value(3.0)]
probs = softmax(logits)
print([round(p.data, 6) for p in probs])
# Should be approximately [0.090031, 0.244728, 0.665241]

total = sum(p.data for p in probs)
print(f"sum of probs: {total:.10f}")
# Should be 1.0000000000 (or extremely close)
```

Now try with extreme values to see the stability trick in action:

```python
logits_extreme = [Value(1000.0), Value(1001.0), Value(1002.0)]
probs_extreme = softmax(logits_extreme)
print([round(p.data, 6) for p in probs_extreme])
# Should give the same proportions as [1, 2, 3] — softmax is shift-invariant
```

Without the `max_val` subtraction, `Value(1000.0).exp()` would produce `inf`. With it, the largest logit becomes `exp(0) = 1` and all is well.

### Step 4: Verify `rmsnorm` output scale

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

x = [Value(1.0), Value(2.0), Value(3.0)]
xn = rmsnorm(x)
values = [v.data for v in xn]
print(values)

# Verify: RMS of the output should be ~1
rms_out = (sum(v**2 for v in values) / len(values)) ** 0.5
print(f"RMS of output: {rms_out:.6f}")
# Should be approximately 1.0
```

Compare the direction of the vector (relative proportions) before and after normalization. RMSNorm rescales magnitude but preserves direction.

---

## What's happening under the hood

**`state_dict` is just a dict.** There is no magic class, no `nn.Module`, no automatic registration. Every weight is manually created and manually collected into `params`. This transparency is the whole point — you can see every parameter and trace exactly how it flows to the loss.

**`linear` is a pure Python implementation of `torch.nn.Linear(bias=False)`.** The shapes, naming, and behavior are identical. The only difference is that PyTorch runs this in C++ on GPU; here it runs in a Python loop on your CPU. The algorithm is the same.

**`softmax` is called twice per token in training** — once inside the attention mechanism (to compute attention weights over previous positions) and once after `lm_head` (to convert logits into a probability distribution for the loss). Both calls use the same function.

**`rmsnorm` is called three times per layer** — once before the attention block, once before the MLP block, and once on the initial embedding. In later chapters you will see exactly where each call sits in the forward pass.

---

## Check your understanding

1. If `n_embd` were doubled to 32 with everything else unchanged, how many parameters would the model have? Identify which matrices grow and by how much.

2. The `matrix` function uses `random.gauss(0, std)`. What would happen during training if `std` were set to `10.0` instead of `0.08`? Think about what `rmsnorm` receives on the first forward pass.

3. `linear(x, w)` returns `len(w)` values. Given the shapes in the table above, what is the input and output length of `linear(x, state_dict['mlp_fc1'])` at layer 0?

4. In `softmax`, why is `max_val` extracted with `val.data` rather than keeping it as a `Value`? What would break if you tried to use a `Value` as the argument to Python's built-in `max()`?

5. `rmsnorm` has no learned parameters. Look at where `rmsnorm` is used in lines 112 and 117-118 (the `gpt` function). After training, the model cannot adjust the normalization behavior. What can it adjust to compensate?

<details>
<summary>Hints (expand after attempting)</summary>

1. Every matrix dimension involving `n_embd` doubles in one direction; matrices that are `n_embd × n_embd` grow by 4×. Work through the table row by row.

2. With large initial weights, the input to `rmsnorm` has a huge RMS, so `scale` is tiny. After normalization the vector is small but the direction is preserved. The attention logits computed from these embeddings will be near zero, softmax will produce a near-uniform distribution, and early training steps will produce near-random predictions — similar to random initialization but with noisier gradients.

3. `mlp_fc1` has shape `(4*n_embd, n_embd) = (64, 16)`. Input to `linear`: length 16. Output: length 64.

4. `Value` objects do not define `__lt__` or `__gt__`, so Python's `max()` cannot compare them by value. Even if it could, we want the constant float, not a node in the computation graph — the maximum is just a shift constant, not something we differentiate through.

5. The model can adjust the weights that feed into and out of the `rmsnorm` call — specifically the attention and MLP projection matrices. By scaling those up or down, the model effectively controls how much the normalized representation matters.

</details>

---

## What's next

You now have all the tools to read the full forward pass. In **Chapter 5: Attention Mechanism**, you will see how `linear`, `softmax`, and the `state_dict` matrices combine to implement multi-head causal self-attention — the core innovation of the transformer. The key insight coming up: attention is just a weighted sum of value vectors, where the weights are computed by comparing query and key vectors with a dot product followed by `softmax`.

The `head_dim` and `n_head` values you met in this chapter become central: the model splits each 16-dimensional vector into 4 groups of 4 and runs independent attention in each group, then concatenates the results.
