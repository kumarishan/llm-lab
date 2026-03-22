# Chapter 11: Inference — KV Cache, Flash Attention, and Sampling

## What you'll learn

- Why naive autoregressive generation is O(T²) in computation and how the KV cache reduces it to O(T)
- How nanochat's `KVCache` class works in detail: its tensor layout, `get_layer_cache()`, `advance()`, and the prefill-then-decode two-phase pattern
- How `flash_attention.py` chooses between Flash Attention 3 and PyTorch SDPA, and what sliding window attention does to the cache
- How temperature and top-k sampling turn raw logits into the next token
- How `Engine.generate()` ties everything together into a streaming, multi-turn conversation loop

## Prerequisites

- You have completed Chapters 1–4 (environment set up, GPT architecture, attention mechanism)
- You understand what Q, K, V projections are and what the attention dot product computes
- You are comfortable with PyTorch tensors and shapes

---

## 1. Inference vs Training

During **training** the model sees a full sequence, computes a forward pass to get logits at every position, compares them to the ground-truth next tokens, and calls `.backward()` to accumulate gradients. Then an optimizer updates every weight.

During **inference** none of that happens. There are no ground-truth targets, no gradients, and no weight updates. The model just runs a forward pass and you read off the logit vector for the last position.

```python
# Training: gradients flow, weights change
loss = model(input_ids, targets=targets)
loss.backward()
optimizer.step()

# Inference: no gradient tracking at all
with torch.no_grad():
    logits = model(input_ids)
```

`torch.no_grad()` — or equivalently the `@torch.inference_mode()` decorator used throughout `engine.py` — tells PyTorch not to build the autograd graph. This saves both memory (no activation storage for backprop) and a small amount of compute. For large models the memory saving is significant because you don't need to keep the intermediate activations that backpropagation requires.

Inference is not free, though. The bottleneck shifts from gradient computation to **attention over long contexts**, and that is where the KV cache comes in.

---

## 2. The Autoregressive Generation Problem

A language model generates text one token at a time. To generate token $t$, it must condition on all tokens $1, \ldots, t-1$. That means you need a forward pass for every new token.

The naive approach: at each step, pass the entire sequence seen so far through the model.

```
Step 1:  forward([token_1])                            -> logits -> sample token_2
Step 2:  forward([token_1, token_2])                   -> logits -> sample token_3
Step 3:  forward([token_1, token_2, token_3])          -> logits -> sample token_4
...
Step T:  forward([token_1, ..., token_{T-1}])          -> logits -> sample token_T
```

Each attention layer must compute:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

where $Q$, $K$, $V$ all have shape $(T, d)$ at step $T$. The $QK^\top$ matrix multiply is $O(T^2 \cdot d)$.

Summing over all $T$ steps:

$$\sum_{t=1}^{T} O(t^2 \cdot d) = O(T^3 \cdot d)$$

For a 1000-token conversation with 12 attention layers and $d=64$, this is enormous — and mostly wasted work, because you are recomputing attention for the same past tokens over and over.

---

## 3. The KV Cache: The Central Optimization

### The key insight

Look at what changes between step $t$ and step $t+1$:

- The query $Q_t$ for token $t$: **changes** (new token, new embedding)
- The keys $K_1, \ldots, K_{t-1}$ for all past tokens: **do not change** (same tokens, same weights)
- The values $V_1, \ldots, V_{t-1}$ for all past tokens: **do not change**

The weights $W_K$ and $W_V$ are fixed after training. The embeddings of past tokens don't change. Therefore the $K$ and $V$ projections for past tokens are identical across steps. If you save them, you never have to recompute them.

**The KV cache stores the K and V tensors for every past token. On each new step, you only compute Q, K, V for the single new token, then attend to the full cached K and V.**

### Complexity comparison

| Approach        | Per-step attention cost | Total cost (T steps) |
|----------------|------------------------|----------------------|
| Naive           | O(t^2 * d)             | O(T^3 * d)           |
| KV Cache        | O(t * d)               | O(T^2 * d)           |

The KV cache eliminates one power of T. For long sequences this is the difference between seconds and minutes.

### Memory cost

Nothing is free. The cache must hold:

```
num_layers x batch_size x max_seq_len x num_kv_heads x head_dim x 2 (K and V)
```

In float16 (2 bytes per element), a 12-layer model with 6 KV heads, head dimension 128, and a 2048-token context at batch size 1 uses:

```
12 x 1 x 2048 x 6 x 128 x 2 x 2 bytes = 75,497,472 bytes ~= 72 MB
```

That is manageable for a small model. For large production models (hundreds of layers, thousands of heads, tens of thousands of context length) the KV cache becomes the dominant memory consumer. This is why researchers study quantized KV caches, paged attention (vLLM), and other techniques — but those are beyond our scope here.

### Diagram: what the KV cache does

```
NAIVE (no cache)
================

Step 2:  embed [t1, t2]  --> Q2, K2, V2  (2 tokens, T=2)
         attention: Q2 @ K2^T            (2x2 matrix)

Step 3:  embed [t1, t2, t3]  --> Q3, K3, V3  (3 tokens, T=3)
         attention: Q3 @ K3^T                (3x3 matrix)
         ^ K for t1 and t2 were ALREADY computed at step 2 -- wasted work

Step 4:  embed [t1, t2, t3, t4]  --> Q4, K4, V4  (4 tokens)
         attention: Q4 @ K4^T                (4x4 matrix)
         ^ K, V for t1, t2, t3 all recomputed -- wasted work again


WITH KV CACHE
=============

Initial state:  cache_k = [], cache_v = []

Step 2:  embed t2  --> q_new, k_new, v_new  (just 1 new token)
         store:  cache_k = [k1, k2]
                 cache_v = [v1, v2]
         attend: q_new @ [k1, k2]^T         (1x2 -- tiny)

Step 3:  embed t3  --> q_new, k_new, v_new
         store:  cache_k = [k1, k2, k3]
                 cache_v = [v1, v2, v3]
         attend: q_new @ [k1, k2, k3]^T     (1x3)

Step 4:  embed t4  --> q_new, k_new, v_new
         store:  cache_k = [k1, k2, k3, k4]
         attend: q_new @ [k1, k2, k3, k4]^T (1x4)

k1, k2, k3 are READ from cache -- not recomputed.
```

The query matrix shrinks from T rows (one per token in the full sequence) to just 1 row (the new token). The dominant O(T^2) cost of recomputing all K/V projections disappears.

---

## 4. KV Cache With Sliding Window Attention

nanochat does not use full attention in every layer. As covered in Chapter 4, it uses a **sliding window pattern** configured by `window_pattern` in `GPTConfig`.

The default pattern is `"SSSL"`:

```python
# nanochat/gpt.py, GPTConfig
window_pattern: str = "SSSL"
```

This pattern is tiled across all layers and the final layer always gets full context (L). For a 12-layer model the resulting window assignments are:

```
Layer:    0  1  2  3  4  5  6  7  8  9  10  11
Pattern:  S  S  S  L  S  S  S  L  S  S   S   L
```

Where:
- **L (Long)**: attend to the full context — window size = `sequence_len`
- **S (Short)**: attend to the last ~quarter of the context — window size = `ceil(sequence_len / 4)`, rounded up to the nearest 128 (FA3 tile alignment)

This has two consequences for the KV cache:

1. **L-layers** need to cache every token the model has ever seen, up to `max_seq_len`.
2. **S-layers** can discard tokens that fall outside the window — but nanochat's current implementation pre-allocates the full `max_seq_len` for all layers and applies the window via a mask at attention time rather than truncating the cache. This keeps the implementation simple and uniform.

```
KV Cache tensor layout (nanochat):

k_cache: (num_layers, batch_size, max_seq_len, num_kv_heads, head_dim)
v_cache: same shape

For a 12-layer model, batch=1, seq=2048, 6 KV heads, head_dim=128:
k_cache: (12, 1, 2048, 6, 128)

Dimension meanings:
  dim 0: which transformer layer (0..11)
  dim 1: which batch element (0..B-1)
  dim 2: which sequence position (0..T-1)
  dim 3: which KV head (0..H_kv-1)
  dim 4: the head_dim-length vector
```

The window mask is applied inside `flash_attention.py` at attention time: only the last `window` positions are attended to, even though they are all in the cache.

---

## 5. nanochat's `KVCache` Class

File: `nanochat/engine.py`, class `KVCache`

### Pre-allocation and shape

```python
class KVCache:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        # Pre-allocate: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim,
                                   device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim,
                                   device=device, dtype=dtype)
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Previous token's normalized embedding for "smearing"
        self.prev_embedding = None
```

Two important details:

**Tensor order is (B, T, H, D), not (B, H, T, D).** FA3 requires this layout. PyTorch's own attention expects (B, H, T, D). The `flash_attention.py` SDPA fallback transposes before and after calling `F.scaled_dot_product_attention`.

**Pre-allocation.** The entire cache is allocated up front with `torch.zeros`. There is no dynamic resizing. This means you must decide `seq_len` (the maximum number of tokens this generation session will ever produce) before you start. The `Engine.generate()` method computes a `kv_length_hint` for this:

```python
kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
```

### `cache_seqlens`: tracking position

`cache_seqlens` is a 1-D int32 tensor of shape `(batch_size,)`. Entry `i` holds the number of tokens currently stored in the cache for batch element `i`. This is the **write pointer**: when you add new K/V entries, they go at positions `cache_seqlens[i]` through `cache_seqlens[i] + T_new - 1`.

FA3's `flash_attn_with_kvcache` API requires this tensor to know where to write and how many cached entries to attend to.

```python
def get_pos(self):
    """Get current position (assumes all batch elements at same position)."""
    return self.cache_seqlens[0].item()
```

### `get_layer_cache()`: per-layer views

Each transformer layer has its own K/V slice:

```python
def get_layer_cache(self, layer_idx):
    """Return (k_cache, v_cache) views for a specific layer."""
    return self.k_cache[layer_idx], self.v_cache[layer_idx]
```

This returns a **view** (not a copy), so modifications to the returned tensors update the cache in-place. FA3 writes into these views directly during `flash_attn_with_kvcache`.

### `advance()`: incrementing the write pointer

```python
def advance(self, num_tokens):
    """Advance the cache position by num_tokens."""
    self.cache_seqlens += num_tokens
```

This must be called after processing each new set of tokens. In `gpt.py`, the attention block calls `advance` for you via the last layer:

```python
if self.layer_idx == kv_cache.n_layers - 1:
    kv_cache.advance(T)
```

The position is only advanced once, by the last layer, rather than once per layer.

### `prev_embedding`: the "smear" feature

nanochat includes an architecture feature called **smearing**: a learned gating mechanism that mixes the previous token's normalized embedding into the current token's representation. This is a cheap way to give the model bigram-level context without extra attention steps.

For inference, the cache must carry this embedding across decoding steps:

```python
self.prev_embedding = None  # set to None initially
```

After each forward pass, `gpt.py` writes the current token's embedding into `kv_cache.prev_embedding`. On the next decode step, the model reads it back to apply the smear gate. The `KVCache` is thus not just a K/V store — it also carries this single extra activation between steps.

### `reset()` and `prefill()`

```python
def reset(self):
    """Reset cache to empty state."""
    self.cache_seqlens.zero_()
    self.prev_embedding = None
```

`reset()` clears the position counter. It does not zero the cached tensors (unnecessary — new writes will overwrite old values in order).

```python
def prefill(self, other):
    """Copy cached KV from another cache into this one."""
    assert self.get_pos() == 0
    other_pos = other.get_pos()
    self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
    self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
    self.cache_seqlens.fill_(other_pos)
    if other.prev_embedding is not None:
        self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()
```

`prefill()` is used when you want to generate multiple independent samples from the same prompt. Rather than running the full prompt through the model `num_samples` times, you run it once with `batch_size=1`, then clone the cache `num_samples` times. The `expand()` + `clone()` on `prev_embedding` broadcasts the single-batch embedding to all `num_samples` rows.

---

## 6. The Two-Phase Generation Loop

### Phase 1: Prefill

The prompt (all tokens the user wrote) is processed **all at once** in a single forward pass. This is called the **prefill phase**.

```
Prompt: [<bos>, <|user_start|>, "What", "is", "2+2", "?", <|user_end|>, <|assistant_start|>]
        --> single forward pass, T=8
        --> K and V for all 8 tokens written into cache
        --> logits[:, -1, :] gives the distribution over the first assistant token
```

Prefill is fast because the GPU can parallelize attention across all T positions simultaneously. The KV cache is filled in one shot.

### Phase 2: Decode

After prefill, you enter the **decode loop**: one new token per iteration.

```
Iteration 1:
  input:  [sampled_token_1]   (just 1 token)
  cache:  K/V for positions 0..7 already there
  writes: K/V for position 8 into cache
  reads:  attend to cache positions 0..8
  output: logits --> sample token_2

Iteration 2:
  input:  [sampled_token_2]
  writes: K/V for position 9
  reads:  attend to cache positions 0..9
  output: logits --> sample token_3
...
```

Each decode step processes exactly 1 token, so the attention query is shape `(B, 1, H, D)` instead of `(B, T, H, D)`. The heavy computation collapses from a T×T matrix to a 1×T vector dot product.

### The full loop in `Engine.generate()`

File: `nanochat/engine.py`, `Engine.generate()`

```python
@torch.inference_mode()
def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):

    # Phase 1: prefill with a batch-size-1 cache
    kv_cache_prefill = KVCache(batch_size=1, seq_len=len(tokens), ...)
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
    logits = logits[:, -1, :].expand(num_samples, -1)  # shape: (num_samples, vocab_size)

    # Clone the prefill cache for each sample
    kv_cache_decode = KVCache(batch_size=num_samples, seq_len=kv_length_hint, ...)
    kv_cache_decode.prefill(kv_cache_prefill)
    del kv_cache_prefill

    # Phase 2: decode loop
    while True:
        # Stop if reached max_tokens or all rows finished
        if max_tokens is not None and num_generated >= max_tokens:
            break
        if all(state.completed for state in row_states):
            break

        # Sample the next token from logits
        next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)

        # ... handle tool use, forced tokens, EOS detection ...

        yield token_column, token_masks  # stream tokens to caller

        # One forward pass with the single new token
        ids = torch.tensor(token_column, ...).unsqueeze(1)  # shape (B, 1)
        logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]
```

The key line is the last one: `ids` has shape `(B, 1)` — just one token per batch element. The model runs the full transformer, but because each attention block only queries one new token against the cached K/V, it is O(T) per layer instead of O(T^2).

### Stopping conditions

The loop ends when any of these is true:

- `num_generated >= max_tokens` (hard length limit)
- All batch elements have generated an `<|assistant_end|>` token or a `<|bos|>` token (both signal end-of-turn)

The `RowState.completed` flag tracks whether each batch row has hit a stop condition.

---

## 7. Flash Attention 3

### Why attention needs its own kernel

Naive attention computes:

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(d)  # (B, H, T, T) -- may be huge
probs  = scores.softmax(dim=-1)
out    = probs @ v
```

For T=4096, H=32, the `scores` tensor alone is 4096x4096x32 = 536 million floats — over 1 GB in fp32 just for one batch element. This tensor is written to GPU global memory and then read back for the softmax and the final matmul.

**Flash Attention** avoids materializing the full scores matrix by processing the sequence in tiles that fit in fast SRAM (the GPU's on-chip shared memory). The softmax normalization is maintained in a running fashion across tiles using the log-sum-exp trick. The output is computed tile by tile and accumulated. Global memory never holds the full T×T matrix.

FA2 and FA3 are CUDA kernels that implement this fused operation. They are not pure Python — they are compiled C++/CUDA.

### FA3 vs SDPA

nanochat's `flash_attention.py` picks one of two implementations at import time:

```python
def _load_flash_attention_3():
    if not torch.cuda.is_available():
        return None
    major, _ = torch.cuda.get_device_capability()
    # FA3 kernels are compiled for Hopper (sm90) only
    if major != 9:
        return None
    # ...
    return get_kernel('varunneal/flash-attention-3').flash_attn_interface
```

| Condition                          | Implementation used |
|-----------------------------------|---------------------|
| Hopper GPU (H100, sm90) + bf16    | FA3                 |
| Anything else                     | PyTorch SDPA        |

The selection happens at module load time and is stored in the module-level boolean `USE_FA3`. Every call to `flash_attn_func` or `flash_attn_with_kvcache` checks this flag and dispatches accordingly.

Why bf16 only for FA3? The FA3 Hopper kernels are compiled specifically for bfloat16 (and fp8). If you are on a Hopper GPU but your `COMPUTE_DTYPE` is fp16 or fp32, the code falls back to SDPA rather than crashing.

### The public API

The module exports a `flash_attn` namespace that matches the FA3 interface exactly, so the rest of the codebase can write:

```python
from nanochat.flash_attention import flash_attn

# Training
y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

# Inference with KV cache
y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v,
                                        cache_seqlens=cache_seqlens,
                                        causal=True, window_size=window_size)
```

Both functions accept `window_size=(left, right)` where `left=-1` means unlimited (full context) and `left=N` means attend to at most the last N tokens.

---

## 8. SDPA: The Universal Fallback

`torch.nn.functional.scaled_dot_product_attention` (SDPA) is PyTorch's built-in fused attention. It automatically picks the best available kernel (FlashAttention2 on CUDA, or a standard implementation on CPU/MPS). It works on every device nanochat supports.

### Sliding window in SDPA

SDPA does not natively support sliding window masks, so `_sdpa_attention` builds one manually when needed:

```python
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length (training)
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation (most common decode step)
    if Tq == 1:
        if window >= 0 and window < Tk:
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Chunk inference: build explicit causal + sliding window mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx                              # causal
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)     # sliding window
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)
```

Three cases:

1. **Training / full prefill (`Tq == Tk`, no window):** use `is_causal=True` — SDPA builds the causal mask internally, which is faster than passing an explicit mask.

2. **Single-token decode (`Tq == 1`):** there is no causal masking needed (a single query position is never "ahead" of itself). If a sliding window applies, we slice the K/V cache to include only the last `window+1` positions before calling SDPA.

3. **Chunk inference (`Tq > 1` but `Tq != Tk`):** this occurs during prefill when there are already cached tokens. An explicit boolean mask combining causality and sliding window is constructed and passed to SDPA.

### SDPA's tensor layout vs FA3's

Note the transposes in the SDPA path:

```python
# FA3 wants (B, T, H, D)
# SDPA wants (B, H, T, D)

# flash_attn_func SDPA path:
q = q.transpose(1, 2)   # (B, T, H, D) -> (B, H, T, D)
k = k.transpose(1, 2)
v = v.transpose(1, 2)
y = _sdpa_attention(q, k, v, ...)
return y.transpose(1, 2)   # back to (B, T, H, D)
```

The rest of the model uses FA3-style layout `(B, T, H, D)` throughout. The transpositions in the SDPA fallback are a seam that adapts between the two conventions.

---

## 9. Sampling Strategies

After the forward pass, the model produces **logits**: a vector of shape `(vocab_size,)` with one real-valued score for each token in the vocabulary. Higher score = model thinks this token is more likely next.

File: `nanochat/engine.py`, `sample_next_token()`

```python
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token. Returns (B, 1)."""
    assert temperature >= 0.0
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)
```

### Greedy decoding (temperature=0)

Always pick the token with the highest logit. Deterministic, reproducible, but tends to produce repetitive text because the model gets "stuck" in the highest-probability groove.

### Temperature scaling

Temperature $\tau$ divides the logits before softmax:

$$P_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

- **$\tau \to 0$:** logits are amplified infinitely; softmax collapses to argmax (greedy)
- **$\tau = 1.0$:** raw model probabilities (no modification)
- **$\tau > 1$:** logits shrink; probabilities become more uniform (more random)

✍️
```python
import torch
import torch.nn.functional as F

# Suppose the model gives these logits for 5 vocabulary items
logits = torch.tensor([3.0, 1.0, 0.5, -0.5, -2.0])

for tau in [0.5, 1.0, 1.5, 2.0]:
    probs = F.softmax(logits / tau, dim=-1)
    print(f"tau={tau:.1f}: {[f'{p:.3f}' for p in probs.tolist()]}")
```

Expected output (approximately):
```
tau=0.5: ['0.872', '0.106', '0.019', '0.003', '0.000']
tau=1.0: ['0.656', '0.182', '0.110', '0.040', '0.011']
tau=1.5: ['0.536', '0.198', '0.149', '0.081', '0.037']
tau=2.0: ['0.460', '0.206', '0.171', '0.103', '0.059']
```

Lower temperature = more probability mass on the top token; higher = more spread across alternatives.

### Top-K sampling

After applying temperature, zero out all but the top K logits, then sample from the remainder.

```
logits (before): [3.0, 1.0, 0.5, -0.5, -2.0]
top-3 mask:      [3.0, 1.0, 0.5,  -inf,  -inf]
softmax:         [0.738, 0.205, 0.057, 0.0, 0.0]
multinomial:     sample from this trimmed distribution
```

Top-K prevents the model from accidentally sampling a very unlikely token (which without any constraint happens at any temperature > 0). The default in `chat_cli.py` is `top_k=50`.

Note the order of operations in nanochat's implementation: when both temperature and top-k are active, the code takes the top-k **before** dividing by temperature. The same top-K candidates are selected either way; only the relative probabilities among those candidates are then rescaled by temperature.

### Top-P (nucleus sampling) — for reference

nanochat does not currently implement top-P, but it is worth knowing conceptually. Instead of a fixed K, you keep the smallest set of tokens whose cumulative probability exceeds P (e.g., P=0.9). This adapts to the sharpness of the distribution: when the model is confident, fewer tokens are included; when uncertain, more are.

---

## 10. Streaming Generation

`Engine.generate()` is a Python **generator** — it uses `yield` rather than returning a final list. This means tokens are available to the caller immediately as they are produced.

```python
for token_column, token_masks in engine.generate(conversation_tokens, **kwargs):
    token = token_column[0]          # pick batch element 0
    text = tokenizer.decode([token])
    print(text, end="", flush=True)  # print immediately
```

`flush=True` flushes stdout after every token so the output appears character by character rather than buffered. This is what makes the CLI feel responsive.

`token_masks` is a parallel list indicating, for each batch element, whether the token was **sampled** (1) or **forced** (0). Forced tokens come from the tool-use state machine (e.g., the calculator output is injected as forced tokens after a `<|python_end|>` marker). For normal conversation, all masks are 1.

> **What's happening.** `yield token_column, token_masks` suspends `generate()` and hands control back to the caller. The caller processes the token (prints it, sends it to a browser, etc.), then resumes the generator on the next loop iteration. No separate thread is needed — Python's generator protocol handles the interleaving.

For a web server, the same streaming interface works over Server-Sent Events: the server iterates the generator and pushes each decoded token fragment to the browser as it arrives.

---

## 11. Multi-Turn Conversation State

File: `scripts/chat_cli.py`

```python
conversation_tokens = [bos]   # start with just the BOS token

while True:
    user_input = input("\nUser: ").strip()
    # ...

    # Append this turn's user message
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)
    conversation_tokens.append(assistant_start)

    # Generate the assistant's response
    response_tokens = []
    for token_column, token_masks in engine.generate(conversation_tokens, **kwargs):
        token = token_column[0]
        response_tokens.append(token)
        print(tokenizer.decode([token]), end="", flush=True)

    # Append the completed response to conversation history
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)
```

### What `conversation_tokens` contains after two turns

```
[<bos>
 <|user_start|> "Hello" <|user_end|>
 <|assistant_start|> "Hi there!" <|assistant_end|>
 <|user_start|> "What is 2+2?" <|user_end|>
 <|assistant_start|>]
```

The entire conversation history is passed to `engine.generate()` on every turn. The prefill phase re-processes the whole history to fill the KV cache, then decode continues from the new `<|assistant_start|>`.

This is simpler than maintaining the KV cache across turns (which would require resetting and re-prefilling only the delta), and for the context lengths nanochat targets it is fast enough.

The `clear` command simply resets `conversation_tokens` to `[bos]`, discarding the entire history:

```python
if user_input.lower() == 'clear':
    conversation_tokens = [bos]
    print("Conversation cleared.")
    continue
```

> **What's happening.** When you type `clear`, `conversation_tokens` is reassigned to a fresh single-element list. The old list (and its many tokens) is garbage collected. The next call to `engine.generate()` sees only `[bos]` and starts a brand-new prefill with an empty context.

---

## 12. Hands-on: Running chat_cli.py

✍️
```bash
# From the nanochat project root
python -m scripts.chat_cli --model-tag=<your-model-tag>
```

Replace `<your-model-tag>` with the tag from your trained checkpoint. If you don't have a tag, you can omit it and the script will try to load the latest checkpoint.

Available flags:

| Flag                | Default | Meaning                                              |
|--------------------|---------|------------------------------------------------------|
| `--model-tag TAG`  | None    | Load checkpoint with this tag                        |
| `--step N`         | None    | Load checkpoint at step N (latest if omitted)        |
| `--temperature T`  | 0.6     | Sampling temperature (0.0 = greedy, ~0.6 = balanced) |
| `--top-k K`        | 50      | Top-K sampling; 0 disables it                        |
| `--device-type`    | auto    | Force cuda / cpu / mps                               |
| `-p PROMPT`        | ""      | Non-interactive: get one response and exit           |

### Example session

The following shows what a typical session looks like. Each user turn is processed through
the full prefill + decode cycle described above.

```
NanoChat Interactive Mode
--------------------------------------------------
Type 'quit' or 'exit' to end the conversation
Type 'clear' to start a new conversation
--------------------------------------------------

User: What is the capital of France?
Assistant: The capital of France is Paris.

User: clear
Conversation cleared.

User: What is 17 * 23?
Assistant: 17 * 23 = 391.
```

> **What's happening.** Each call to `engine.generate()` triggers a full prefill of `conversation_tokens` followed by the token-by-token decode loop. The printed tokens appear in real time as the generator yields them. The `clear` command resets `conversation_tokens = [bos]`, so the next turn starts with no history.

For a single non-interactive query (e.g., in a script), use `-p`:

✍️
```bash
python -m scripts.chat_cli -p "What is the capital of France?" --temperature=0.0
```

Setting `--temperature=0.0` gives greedy decoding: deterministic, always the same answer.

---

## Check Your Understanding

**Question 1.** During the decode phase, `Engine.generate()` passes a tensor of shape `(B, 1)` to `model.forward()` on each iteration. Why 1 and not T (the full sequence length)?

**Question 2.** The `KVCache` pre-allocates tensors of shape `(num_layers, B, T_max, H, D)` at initialization. What would go wrong if you pre-allocated `T_max` to be too small?

**Question 3.** Suppose you set `temperature=0.0`. What does `sample_next_token` do differently, and why might this produce worse conversation quality compared to `temperature=0.6` with `top_k=50`?

<details>
<summary>Answers</summary>

**A1.** The KV cache stores K and V for all previous tokens. The model only needs to compute Q, K, V for the **new** token and attend to the full cached K/V. There is no reason to re-embed the previous tokens — they are already represented in the cache. Passing the full sequence would require recomputing K/V for every past token, defeating the cache entirely.

**A2.** Pre-allocating `T_max` too small means `cache_seqlens` will eventually point past the end of the allocated tensor. When `advance()` increments past `T_max`, subsequent writes by FA3 or the SDPA fallback would either write out of bounds (undefined behavior in CUDA) or produce wrong results. In practice you would see garbage tokens or a crash. The `kv_length_hint` in `Engine.generate()` is sized to `len(tokens) + max_tokens` specifically to prevent this.

**A3.** With `temperature=0.0`, `sample_next_token` returns `torch.argmax(logits)` — always the single highest-probability token. This is deterministic but tends toward repetitive, formulaic text because the model's distribution has long tails of interesting alternatives that are never explored. Temperature 0.6 with top-k 50 keeps the top 50 candidates and flattens their probabilities slightly, allowing the model to occasionally choose second- or third-best tokens that produce more varied, natural-sounding text.

</details>

---

## What's Next

Chapter 12 covers **evaluation and benchmarking**: how to measure whether your trained model is actually good, what standard benchmarks exist, and how nanochat's `core_eval.py` runs them.

You now have a complete picture of the inference stack: from the mathematical justification of the KV cache, through nanochat's concrete `KVCache` implementation, to the FA3/SDPA dispatch, the sampling functions, and the streaming multi-turn conversation loop that ties it all together.