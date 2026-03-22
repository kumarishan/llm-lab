# Chapter 4: The GPT Architecture — Attention, RoPE, GQA, and Modern Innovations

## What you'll learn

- How `GPTConfig` encodes an entire model as a small dataclass, and how a single depth parameter derives all other hyperparameters
- Why Rotary Position Embeddings (RoPE) encode relative positions in the attention dot product — and the rotation matrix math behind them
- How Grouped Query Attention (GQA) shrinks the KV cache by 4× with almost no quality loss
- What half a dozen recent innovations (QK norm, sliding windows, value embeddings, smearing, backout, logit softcap, residual scalars) each solve and why nanochat uses them
- How every piece connects in a single end-to-end forward pass

---

## Prerequisites

- Chapters 1–3: you have nanochat installed and you understand the basic attention mechanism
  (queries, keys, values; $\text{softmax}(QK^T/\sqrt{d}) \cdot V$)
- Comfort reading Python dataclasses and PyTorch `nn.Module` code
- No prior knowledge of RoPE, GQA, Flash Attention, or any other modern architectural innovation is assumed

---

## Architecture overview

Before diving into details, here is the complete data flow through `GPT.forward()`. Each box in the diagram corresponds to a section of this chapter.

```
Input token ids  (B, T)
        │
        ▼
┌─────────────────────┐
│  Token Embedding     │  wte: lookup table (vocab_size, n_embd)
│  + RMSNorm           │  normalise right after embedding
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Smearing            │  mix prev-token embedding into current position
└────────┬────────────┘
         │  x0 = x  (save for x0 blending)
         │
         ▼  ┌──────────────────────────────────────────────────────────────┐
         │  │  For each layer i of n_layer:                                │
         │  │                                                              │
         │  │   x = resid_lambdas[i] * x + x0_lambdas[i] * x0            │
         │  │                                                              │
         │  │   ┌──────────────────────────────────────────────────┐      │
         │  │   │  CausalSelfAttention                             │      │
         │  │   │   RoPE  →  Q, K rotation                        │      │
         │  │   │   GQA   →  n_head queries / n_kv_head KV        │      │
         │  │   │   QK norm → stabilise attention scores          │      │
         │  │   │   Value embed (alternating layers)              │      │
         │  │   │   Flash Attn 3 / SDPA (sliding window)         │      │
         │  │   └──────────────────────────────────────────────────┘      │
         │  │                                                              │
         │  │   x = x + attn(norm(x))    ← residual                      │
         │  │   x = x + mlp(norm(x))     ← residual  (ReLU²)             │
         │  │                                                              │
         │  │   cache x at layer n_layer//2  →  x_backout                 │
         │  └──────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Backout             │  x = x - backout_lambda * x_backout
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Final RMSNorm       │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  lm_head (Linear)    │  project to vocab_size logits
│  Logit softcap ±15   │  tanh clamp for training stability
└────────┬────────────┘
         ▼
    Cross-entropy loss  (training) / raw logits (inference)
```

Now let's build up each piece from scratch.

---

## Section 1 — GPTConfig: the configuration dataclass

**File:** `nanochat/gpt.py`, class `GPTConfig`

### What it contains

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6        # number of query heads
    n_kv_head: int = 6     # number of key/value heads (GQA)
    n_embd: int = 768
    window_pattern: str = "SSSL"
```

Every piece of the model's shape is in this one dataclass. Let's walk through each field:

| Field | Typical value | What it means |
|---|---|---|
| `sequence_len` | 2048 | Maximum number of tokens the model can process at once |
| `vocab_size` | 32768 | Number of distinct tokens the tokenizer can produce |
| `n_layer` | 12 | Number of transformer blocks stacked on top of each other |
| `n_head` | 6 | Number of query heads in multi-head attention |
| `n_kv_head` | 6 (or `n_head//4`) | Number of key/value heads (< `n_head` enables GQA) |
| `n_embd` | 768 | Embedding dimension — the "width" of the model |
| `window_pattern` | `"SSSL"` | Per-layer sliding window pattern (explained in Section 6) |

The embedding dimension `n_embd` and the head dimension `head_dim` are related by:

$$\text{head\_dim} = \frac{\text{n\_embd}}{\text{n\_head}}$$

For the default config: $768 / 6 = 128$ dimensions per head.

### The depth dial

The codebase derives all hyperparameters from a single `--depth` integer in the training script. When you run:

```bash
python scripts/base_train.py --depth 4
```

the script calls a helper that sets `n_layer`, `n_head`, `n_embd`, `n_kv_head`, and `sequence_len` in a compute-optimal way. The exact formula follows standard scaling-law guidance (Chinchilla): as depth grows, width grows proportionally so that the compute budget stays balanced between depth and width.

The key relationship is:

$$\text{n\_kv\_head} = \frac{\text{n\_head}}{4}$$

This is the 4:1 GQA grouping ratio, which reduces the KV cache by 4× at almost no cost in quality. We'll explain exactly why in Section 5.

> **What's happening:** A single integer controls the entire model. This design choice means you can explore different sizes without remembering a long list of consistent parameter values. At depth 1 the model is tiny and trains on a laptop; at full scale it's GPT-2 class running on 8×H100.

---

## Section 2 — The custom Linear layer

**File:** `nanochat/gpt.py`, class `Linear`

```python
class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))
```

### The problem: precision vs. speed

Neural network training involves two competing requirements:

1. **Optimizer precision.** The Adam optimizer accumulates small gradient updates into weights. If the weights are stored in 16-bit float (bf16), updates smaller than ~$6 \times 10^{-5}$ simply vanish into rounding error. Fp32 has a much smaller unit of least precision.

2. **Compute speed.** Modern tensor cores (NVIDIA A100/H100) perform bf16 matrix multiplications 4–8× faster than fp32. All the heavy matrix multiplications in transformers — the Q, K, V projections, the MLP — should run in bf16.

The standard solution is `torch.amp.autocast`, which wraps forward passes and automatically downcasts certain operations. nanochat instead uses a cleaner approach: **weights live in fp32 permanently, and each Linear layer casts its weight to match whatever dtype the input activations carry**.

The activations flow in bf16 (set by `COMPUTE_DTYPE` from `nanochat/common.py`). When the forward pass calls `self.weight.to(dtype=x.dtype)`, a bf16 view of the fp32 weight is created on the fly, the matmul runs in bf16, and the original fp32 master weight is untouched. The optimizer later updates the fp32 master weight with full precision.

```
Stored weight: fp32  ──────────────────────── optimizer reads fp32, updates fp32
                           │
                           ▼ .to(bf16) — view, no copy
                        bf16 weight
                           │
Input (bf16) ──────────────┤ F.linear()
                           │
Output (bf16) ─────────────┘
```

> **What's happening:** The `.to(dtype=x.dtype)` call does not allocate new memory for a full copy of the weight matrix. PyTorch creates a reinterpreted view. The matmul kernel sees bf16 operands and runs on tensor cores; the optimizer later sees fp32 parameters and applies updates without rounding loss.

---

## Section 3 — Rotary Position Embeddings (RoPE)

**File:** `nanochat/gpt.py`, function `apply_rotary_emb` and `_precompute_rotary_embeddings`

### The problem: transformers are position-blind

The raw attention mechanism treats the sequence as a *set*, not a list. Given tokens at positions 1, 2, 3, the model computes exactly the same attention scores regardless of order — a dog biting a man looks identical to a man biting a dog if you scramble the words.

**Early fix: learned positional embeddings (GPT-2 style).**
Add a learned vector $p_t$ to token $t$'s embedding before the attention layers:

$$x_t \leftarrow x_t + p_t$$

This works, but has problems: the model sees each absolute position as a distinct learned feature, so it cannot generalise well to sequences longer than its training window, and it carries no structural relationship between nearby positions.

### RoPE: encode position through rotation

RoPE (Su et al., 2021: [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)) takes a different approach. Instead of adding a positional offset before attention, it *rotates* the query and key vectors at each position so that the angle between position $m$'s query and position $n$'s key encodes only the *relative gap* $m - n$.

**The rotation matrix.**
For a vector $\mathbf{x} \in \mathbb{R}^d$, split it into $d/2$ pairs $(x_{2i}, x_{2i+1})$. Apply a 2D rotation by angle $m\theta_i$ to each pair:

$$\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

where the rotation frequency for dimension pair $i$ is:

$$\theta_i = \frac{1}{\text{base}^{2i/d}}$$

The `base` is called `rope_base` in nanochat (currently hardcoded to 100,000 in `_precompute_rotary_embeddings`). Larger base values mean slower-rotating dimensions and better extrapolation to longer sequences — this is why recent models use 100K or even 500K instead of the original paper's 10K.

**The key property: relative positions.**
After rotating both query $q_m$ at position $m$ and key $k_n$ at position $n$, the dot product becomes:

$$q_m \cdot k_n = (R_m \, q) \cdot (R_n \, k) = q^T R_m^T R_n \, k = q^T R_{n-m} \, k$$

Because rotations are orthogonal ($R_m^T = R_{-m}$), and two rotations compose by adding angles ($R_{-m} R_n = R_{n-m}$), the result depends *only on the relative position* $n - m$. This means:

- The model learns "two tokens apart" as a single concept, regardless of absolute position
- Positions beyond the training window can be extrapolated (with some caveats)
- No extra parameters are needed — the positional signal is entirely in the rotation angles

### Precomputing the rotation tables

`_precompute_rotary_embeddings` builds two buffers, `cos` and `sin`, of shape `(1, seq_len, 1, head_dim/2)`. For each position $t$ and dimension pair $i$:

```python
# From gpt.py: GPT._precompute_rotary_embeddings()
channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
inv_freq = 1.0 / (base ** (channel_range / head_dim))
t = torch.arange(seq_len, dtype=torch.float32, device=device)
freqs = torch.outer(t, inv_freq)          # shape: (seq_len, head_dim/2)
cos, sin = freqs.cos(), freqs.sin()
cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # add batch and head dims
```

`torch.outer(t, inv_freq)` computes every $(t, i)$ combination: element $(t, i)$ is $t \cdot \theta_i$. That gives the full table of angles.

### Applying the rotations

```python
# From gpt.py: apply_rotary_emb()
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]   # split last dim into two halves
    y1 = x1 * cos + x2 * sin           # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

The shape of `x` here is `(B, T, H, head_dim)`. Splitting at `head_dim // 2` gives two halves `x1` and `x2`, corresponding to the "real" and "imaginary" parts of each 2D rotation. The broadcasted multiply with `cos` and `sin` (shape `(1, T, 1, head_dim/2)`) applies the correct angle $t \cdot \theta_i$ to every head and batch element simultaneously.

In the attention forward pass:

```python
# From gpt.py: CausalSelfAttention.forward()
cos, sin = cos_sin
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
```

Values are *not* rotated — they carry content, not position.

> **What's happening:** Because the frequencies $\theta_i$ decrease geometrically (low $i$ = fast rotation, high $i$ = slow rotation), different dimension pairs encode position at different timescales. The first pair completes a full rotation in just a few tokens; the last pair takes tens of thousands of tokens to complete one cycle. Together they form a unique "fingerprint" for each relative offset.

---

## Section 4 — QK Normalization

**File:** `nanochat/gpt.py`, `CausalSelfAttention.forward()`

### The problem: attention score explosion

After applying RoPE, queries and keys go into the attention dot product:

$$\text{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}$$

The $\sqrt{d}$ scaling factor was introduced in the original Transformer paper to prevent the dot products from growing proportionally with dimension size, which pushes the softmax into a region of very small gradients. However, as models grow larger and deeper:

1. The norm of Q and K vectors can grow during training, causing scores to exceed the range where the $\sqrt{d}$ scaling helps
2. Sudden spikes in attention scores cause loss spikes and training instability

### The solution: normalise Q and K directly

nanochat applies RMSNorm to both queries and keys *before* computing the attention scores:

```python
# From gpt.py: CausalSelfAttention.forward()
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
q, k = norm(q), norm(k)  # QK norm
q = q * 1.2  # sharper attention (split scale between Q and K)
k = k * 1.2
```

where `norm` is:

```python
# From gpt.py (top-level)
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

RMSNorm normalises a vector to unit scale:

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}}$$

After QK norm, every query and key vector has magnitude approximately 1. The attention scores become:

$$\text{score}_{ij} = q_i \cdot k_j \approx \cos(\angle(q_i, k_j))$$

This is bounded in $[-1, 1]$ before the $1.2$ rescaling, giving a predictable score range regardless of model depth or dimension.

The `1.2` multiplier is applied *after* normalisation to sharpen the attention distribution slightly — the combined effect is that scores are bounded in approximately $[-1.44, 1.44]$, still controlled but not flat.

> **What's happening:** nanochat's RMSNorm has no learnable scale or bias parameters (see Section 13). This is intentional: the attention pre-head scaling factor `1.2` plays the role of the scale. It is simpler, and empirically just as effective.

QK norm was popularised in ViT-22B (Zhai et al., 2022) and is now standard in large models including Gemma and recent Llama variants.

---

## Section 5 — Grouped Query Attention (GQA)

**File:** `nanochat/gpt.py`, class `CausalSelfAttention`

### Three variants of multi-head attention

| Name | Query heads | Key/Value heads | KV cache size |
|---|---|---|---|
| MHA (Multi-Head) | $H$ | $H$ | $O(H)$ per layer per token |
| MQA (Multi-Query) | $H$ | $1$ | $O(1)$ per layer per token |
| GQA (Grouped-Query) | $H$ | $G < H$ | $O(G)$ per layer per token |

**Standard Multi-Head Attention (MHA)** computes $H$ independent attention heads. Each head has its own Q, K, V projections. During inference, the KV cache must store one key and one value tensor per head per token, which quickly becomes the memory bottleneck for long sequences.

**Multi-Query Attention (MQA)** uses a single K and V shared across all query heads. Memory drops by $H\times$, but quality can degrade because all heads share the same key/value "view" of the context.

**Grouped Query Attention (GQA)** (Ainslie et al., 2023: [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)) is the middle ground: use $G$ key/value heads, each shared by a *group* of $H/G$ query heads. nanochat uses $G = H/4$:

```python
# From gpt.py: CausalSelfAttention.__init__()
self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
```

For the default config with `n_head=6` and `n_kv_head=6`, the model is standard MHA. In practice, larger models set `n_kv_head = n_head // 4`, so 4 query heads share 1 K/V head.

### How expansion works

After projection, the key and value tensors have shape `(B, T, n_kv_head, head_dim)`. Before attention, Flash Attention automatically handles GQA by repeating the KV heads to match the query head count. Conceptually:

$$K_{\text{expanded}} = \text{repeat}(K, \text{repeats}=H/G, \text{dim}=\text{head\_dim})$$

The SDPA fallback in `flash_attention.py` uses PyTorch's native `enable_gqa=True` flag:

```python
# From flash_attention.py: _sdpa_attention()
enable_gqa = q.size(1) != k.size(1)
return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
```

### Memory savings in practice

With 32 query heads and 8 KV heads (a typical Llama-2 70B configuration), the KV cache shrinks by 4×. For a 2048-token sequence with `head_dim=128` and `n_layer=32`, in fp16:

$$\text{MHA KV cache} = 2 \times 32 \times 128 \times 2048 \times 32 \times 2 \approx 1.07 \text{ GB}$$

$$\text{GQA KV cache (8 heads)} = 2 \times 8 \times 128 \times 2048 \times 32 \times 2 \approx 0.27 \text{ GB}$$

The 4× reduction directly translates to either 4× longer sequences or 4× larger batch sizes at inference time.

---

## Section 6 — Sliding Window Attention

**File:** `nanochat/gpt.py`, `_compute_window_sizes()` and `window_pattern`

### The quadratic memory problem

Vanilla attention has time and memory complexity $O(T^2)$ in the sequence length $T$. For $T = 2048$ this is manageable. For $T = 100\,000$ (modern long-context models), the $10^{10}$ attention matrix becomes prohibitive.

Even at modest $T = 2048$, tokens early in the sequence attending to all other tokens is wasteful: most of what token 500 needs to know is in the nearby context, not in the first few tokens.

### Sliding window: local attention

Sliding window attention (Beltagy et al., 2020: [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)) restricts each token to attend to only the $W$ most recent tokens. This reduces the attention complexity from $O(T^2)$ to $O(T \cdot W)$.

In nanochat, the window size is specified per layer through the `window_pattern` string. The `_compute_window_sizes` method maps this string to actual sizes:

```python
# From gpt.py: GPT._compute_window_sizes()
long_window = config.sequence_len
short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size
char_to_window = {
    "L": (long_window, 0),
    "S": (short_window, 0),
}
# Tile pattern across layers
for layer_idx in range(config.n_layer):
    char = pattern[layer_idx % len(pattern)]
    window_sizes.append(char_to_window[char])
# Final layer always gets full context
window_sizes[-1] = (long_window, 0)
```

The `window_size` is passed to Flash Attention as a `(left, right)` tuple:
- `left`: how many past tokens to attend to. $-1$ means unlimited (full context).
- `right`: how many future tokens to attend to. Always $0$ for causal language models.

For `window_pattern = "SSSL"` (the default) with 12 layers:

```
Layer  0: S  (short window = 512 tokens)
Layer  1: S
Layer  2: S
Layer  3: L  (full context = 2048 tokens)
Layer  4: S
Layer  5: S
Layer  6: S
Layer  7: L
Layer  8: S
Layer  9: S
Layer 10: S
Layer 11: L  (always L — final layer must see full context)
```

### Why alternating local and global layers?

The intuition is a **hierarchy of information gathering**:

- **Short (S) layers** compute local syntactic and semantic composition: subject-verb agreement, phrase boundaries, nearby co-reference. They are cheap and handle the dense local signal.
- **Long (L) layers** integrate information from anywhere in the document: topic tracking, long-range dependencies, reasoning chains. They are expensive but only run one-third of the time.

This is similar in spirit to the Longformer and BigBird architectures, but implemented at the layer level rather than the head level.

The `short_window` formula:

```python
short_window = -(-long_window // 4 // 128) * 128
```

deserves unpacking. Starting from `long_window = 2048`:
1. Divide by 4: quarter-context = 512
2. Divide by 128 and ceiling: align to Flash Attention 3's internal tile size for best kernel efficiency
3. Multiply back by 128: 512 → 512 (already aligned in this case)

For `long_window = 1024`, step 1 gives 256, which is already 128-aligned, so `short_window = 256`.

> **What's happening:** The `(-(-x // n)) * n` pattern is a one-liner ceiling division in Python. `-(-512 // 128) * 128 = -(−5) * 128 = 5 * 128 = 640`. Wait — let us recheck: `-(-512//4//128)*128`. `512//4=128`, `128//128=1`, `-(-1)*128=128`. So `short_window=128` for `sequence_len=512`. The key point is alignment to FA3 tile boundaries for performance.

---

## Section 7 — Flash Attention 3 (FA3) vs SDPA

**File:** `nanochat/flash_attention.py`

### The memory bottleneck of naive attention

Standard attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right) V$$

The matrix $QK^T$ has shape $(T, T)$. For $T = 2048$ and fp16, that is $2048^2 \times 2 \approx 8$ MB *per head per batch element*. This matrix must be fully materialised in GPU HBM (high-bandwidth memory) before the softmax and the multiplication with $V$ can proceed. At scale, this is the dominant memory bottleneck.

### Flash Attention: never materialise the full matrix

Flash Attention (Dao et al., 2022: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)) rewrites the computation as a tiled online algorithm. It processes blocks of $Q$ against blocks of $K, V$ in SRAM (fast on-chip memory), maintains a running softmax numerator and denominator, and produces the output without ever writing the full $T \times T$ attention matrix to HBM.

**Flash Attention 3** (Shah et al., 2024) extends this to use Hopper GPU hardware features: asynchronous data movement between HBM and SRAM, warp specialisation for overlap, and FP8 support. It achieves up to 75% of the H100's theoretical peak throughput.

### How nanochat chooses between FA3 and SDPA

```python
# From flash_attention.py: _load_flash_attention_3()
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

The compute capability check (`major != 9`) restricts FA3 to Hopper GPUs (H100 family, SM 90). Ada Lovelace (RTX 4090, SM 89), Blackwell (SM 100), and all Ampere cards fall through to the SDPA fallback.

```python
# From flash_attention.py: _resolve_use_fa3()
def _resolve_use_fa3():
    # ...
    if HAS_FA3:
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return True
        return False  # FA3 Hopper kernels only support bf16 and fp8
    return False
```

Even on H100, FA3 is only enabled for bf16 — FA3 Hopper kernels do not support fp16 or fp32.

The unified API mirrors FA3's interface exactly:

```python
# Training (no KV cache)
y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

# Inference (with KV cache)
y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
```

On non-Hopper hardware, these calls route to PyTorch's `F.scaled_dot_product_attention` (SDPA), which itself dispatches to optimised fused kernels (Flash Attention 2 backend on supported hardware, or a memory-efficient fallback elsewhere).

> **What's happening:** You do not need an H100 to train nanochat. On an A100, RTX 4090, or Apple Silicon, the SDPA fallback runs transparently with the same interface. The only difference is kernel efficiency: FA3 is ~2× faster than SDPA on H100 due to hardware-specific optimisations.

---

## Section 8 — Value Embeddings (ResFormer-style)

**File:** `nanochat/gpt.py`, `has_ve()`, `value_embeds`, and `CausalSelfAttention.forward()`

### Standard attention values

In standard attention, the value tensor is a linear projection of the input:

$$V = X W_V$$

The values carry the *content* that attention aggregates. But they are derived purely from the current hidden state $X$, which at early layers is close to the raw token embedding. The token's identity is only implicitly present through this projection.

### Value residual: inject token identity directly

nanochat adds a learnable token-specific embedding to the values on alternating layers:

$$V = X W_V + \text{gate}(X) \odot \text{VE}[\text{token\_ids}]$$

where $\text{VE}$ is a lookup table of shape `(vocab_size, kv_dim)` — one learned vector per token in the vocabulary. The gate is a small sigmoid-based scalar per KV head, making the blend adaptive.

This idea comes from the ResFormer paper (Dong et al., 2023). The intuition: by injecting the raw token identity into the value vectors, each layer gets a direct "hint" about *what word* is being processed, not just the current residual stream state. This can help the model retrieve word-specific information more easily.

```python
# From gpt.py: GPT.__init__()
head_dim = config.n_embd // config.n_head
kv_dim = config.n_kv_head * head_dim
self.value_embeds = nn.ModuleDict({
    str(i): nn.Embedding(padded_vocab_size, kv_dim)
    for i in range(config.n_layer) if has_ve(i, config.n_layer)
})
```

The `has_ve` function determines which layers get value embeddings:

```python
# From gpt.py
def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2
```

This ensures: (a) layers alternate (even or odd parity depending on total depth), and (b) the last layer always has a value embedding. For `n_layer = 12`, layers 1, 3, 5, 7, 9, 11 have value embeddings.

The gate mechanism in the attention forward pass:

```python
# From gpt.py: CausalSelfAttention.forward()
if ve is not None:
    ve = ve.view(B, T, self.n_kv_head, self.head_dim)
    gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    # gate shape: (B, T, n_kv_head), range (0, 3)
    v = v + gate.unsqueeze(-1) * ve
```

The gate input is only the first 12 channels of the hidden state (`ve_gate_channels = 12`). This is intentionally cheap — the gate computation is a $12 \times \text{n\_kv\_head}$ matrix multiply, negligible compared to the full projections.

The range `(0, 3)` comes from `3 * sigmoid(...)`. A gate of 0 means "ignore value embedding"; a gate of 3 means "add 3× the value embedding". The small uniform initialisation (`0.0` to `0.02`) means gates start near 0 and gradually learn their contribution.

Value embeddings are stored in `COMPUTE_DTYPE` rather than fp32, because the optimizer can tolerate reduced precision for embeddings:

```python
# From gpt.py: init_weights()
if COMPUTE_DTYPE != torch.float16:
    for ve in self.value_embeds.values():
        ve.to(dtype=COMPUTE_DTYPE)
```

---

## Section 9 — Smearing

**File:** `nanochat/gpt.py`, `GPT.forward()`

### The problem: each token starts isolated

After the initial token embedding lookup and RMSNorm, each token position contains information about only *that* token. The transformer layers must then propagate context through attention, which takes many layers to build up even simple bigram statistics.

### Smearing: cheap bigram information

Smearing mixes the previous token's embedding into the current token's representation before the transformer layers run:

$$x_t \leftarrow x_t + \text{gate}(x_t) \cdot x_{t-1}$$

where the gate is a learned scalar per position (based on the current token's first 24 channels):

```python
# From gpt.py: GPT.forward()
gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
```

The first token position is left unchanged (it has no predecessor). For all other positions, a learned gate weight — scaled by the global `smear_lambda` scalar — determines how much of the previous token bleeds into the current one.

The gate input is only `x[..., :24]` (the first 24 embedding channels). This keeps the smear gate computation tiny: `smear_gate` is a `Linear(24, 1, bias=False)` layer — just 24 parameters.

**Why it helps:** Bigram statistics are among the most powerful priors in language modelling. "New York" is more likely than "New Banana". Without smearing, the model must learn these correlations from scratch through attention. With smearing, the token embedding layer already sees its left neighbor, giving the transformer a head start.

**Smearing during inference with KV cache:**

```python
# From gpt.py: GPT.forward() — KV cache path
x_pre_smear = kv_cache.prev_embedding
kv_cache.prev_embedding = x[:, -1:, :]
if T > 1:
    # Prefill: same as training
    gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
    x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
elif x_pre_smear is not None:
    # Decode: single token, use cached prev embedding
    gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
    x = x + gate * x_pre_smear
```

During token-by-token generation, the KV cache stores the previous token's embedding so it is available when the next single token is processed.

---

## Section 10 — Per-layer residual scalars: `resid_lambdas` and `x0_lambdas`

**File:** `nanochat/gpt.py`, `GPT.__init__()`, `init_weights()`, and `forward()`

### The standard residual connection

Every transformer block wraps its attention and MLP sublayers with a residual connection:

$$x \leftarrow x + f(x)$$

This ensures that gradients can flow from the output back to the input without vanishing — even if $f$ is a very weak transformation, the identity path keeps gradients alive.

### Per-layer scaling: `resid_lambdas`

nanochat adds a learnable scalar per layer that scales the incoming residual stream:

$$x \leftarrow \lambda_{\text{resid},i} \cdot x + f(x)$$

This is inspired by work in modded-nanogpt and related projects. The initialisation is:

```python
# From gpt.py: init_weights()
for i in range(n_layer):
    self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
```

For a 12-layer model, `resid_lambdas` initialises as a decreasing sequence from approximately $1.15$ (layer 0) to $1.05$ (layer 11). Early layers receive a slightly amplified residual; later layers receive less amplification. The model can adjust these scalars during training.

### Initial embedding blending: `x0_lambdas`

Additionally, the original token embedding is blended back in at every layer:

$$x \leftarrow \lambda_{\text{resid},i} \cdot x + \lambda_{x_0,i} \cdot x_0 + f(x)$$

where $x_0$ is the normalised token embedding saved before the first transformer block.

```python
# From gpt.py: GPT.forward()
x0 = x  # save initial normalized embedding for x0 residual
for i, block in enumerate(self.transformer.h):
    x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    # ...
```

The `x0_lambdas` initialise with an early-layer bias:

```python
# From gpt.py: init_weights()
for i in range(n_layer):
    self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))
```

Layer 0 gets $\lambda_{x_0} \approx 0.20$ (a meaningful amount of raw embedding); layer 11 gets $\approx 0.05$ (very little). The intuition is: early layers are mostly transforming the raw token embeddings, so mixing the embedding back in is helpful. Later layers have built up rich abstract representations and need less grounding in the raw input.

These two scalars appear as separate `nn.Parameter` vectors and receive different optimizer hyperparameters:

```python
# From gpt.py: setup_optimizer()
dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, ...),  # very small LR
dict(kind='adamw', params=x0_params, lr=scalar_lr, ...),            # larger LR, higher beta1
```

The small learning rate for `resid_lambdas` reflects that these scalars control training stability and should change slowly.

---

## Section 11 — Backout

**File:** `nanochat/gpt.py`, `GPT.forward()`

### The residual stream accumulates redundancy

The residual stream $x$ carries information from every layer. By the final layer, it contains:
- Low-level lexical features (from early layers)
- Mid-level syntactic features
- High-level semantic features (from late layers)

The final norm and `lm_head` projection must project all of this down to vocabulary logits. However, the low-level features that were useful for early layers may add *noise* to the final logit projection rather than signal.

### Backout: subtract the mid-layer state

nanochat caches the residual stream at the midpoint of the transformer and subtracts a fraction of it from the final state before the logit projection:

```python
# From gpt.py: GPT.forward()
x0 = x
backout_layer = n_layer // 2    # cache at halfway point
x_backout = None
for i, block in enumerate(self.transformer.h):
    x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    # ...
    x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
    if i == backout_layer:
        x_backout = x

# After all layers:
if x_backout is not None:
    x = x - self.backout_lambda.to(x.dtype) * x_backout
x = norm(x)
```

The learnable scalar `backout_lambda` initialises to $0.2$:

```python
self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
```

At initialisation, the final representation is:

$$x_{\text{final}} = x_{\text{last}} - 0.2 \cdot x_{\text{mid}}$$

The model can increase or decrease `backout_lambda` during training. If the mid-layer features are genuinely redundant with the final features, subtracting them "cleans up" the residual stream before the logit projection. If the mid-layer features are informative for token prediction, the model will learn `backout_lambda ≈ 0`.

This is a relatively experimental technique. The implementation is straightforward: no extra parameters beyond the single scalar, no change to the layer computation itself.

---

## Section 12 — Logit softcap

**File:** `nanochat/gpt.py`, `GPT.forward()`

### The problem: exploding logits

The `lm_head` projects the final hidden state from `n_embd` to `vocab_size` (32768) dimensions. With fp32 arithmetic and a model that has not yet learned to regulate its weights, these logits can grow very large. A logit of $10^5$ causes:

1. `torch.exp(1e5)` → `inf` in the cross-entropy loss numerator
2. `NaN` loss → `NaN` gradients → training crash

The $1/\sqrt{d}$ scaling in attention helps, but the logit explosion can still occur through the MLP pathway or early in training before the learning rate warmup has finished.

### Solution: tanh softcap

nanochat uses a smooth saturation function:

$$\hat{l}_i = \text{cap} \cdot \tanh\!\left(\frac{l_i}{\text{cap}}\right)$$

with `cap = 15`.

```python
# From gpt.py: GPT.forward()
softcap = 15
logits = self.lm_head(x)
logits = logits[..., :self.config.vocab_size]   # slice padding
logits = logits.float()                          # switch to fp32
logits = softcap * torch.tanh(logits / softcap)  # squash to [-15, 15]
```

Properties of this softcap:
- For small $|l_i| \ll \text{cap}$: $\tanh(l/\text{cap}) \approx l/\text{cap}$, so $\hat{l}_i \approx l_i$ (identity, no distortion)
- For large $|l_i| \gg \text{cap}$: $\tanh(l/\text{cap}) \to \pm 1$, so $\hat{l}_i \to \pm \text{cap}$ (saturation)
- The derivative $\frac{d\hat{l}}{dl} = 1 - \tanh^2(l/\text{cap})$ is always positive (smooth gradient flow)

The choice `cap = 15` means: logits rarely saturate during normal training (typical logits for the correct token are 3–8), but any pathological spike is prevented.

This technique was popularised by Gemma 2 (Google DeepMind, 2024). It is a pure stability measure — it does not change the model's expressivity for typical inputs.

> **What's happening:** The `logits.float()` call before the softcap and cross-entropy is important. The logit tensor coming out of `lm_head` is in bf16. Computing `cross_entropy` in bf16 loses too much precision (bf16 has only 7 mantissa bits). Casting to fp32 first ensures the loss and its gradient are accurate.

---

## Section 13 — No bias, no tied embeddings, RMSNorm

**File:** `nanochat/gpt.py`, `Linear`, `norm()`, `GPT.__init__()`

### No bias in linear layers

```python
self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
```

Every `Linear` in nanochat passes `bias=False`. This is a finding from scaling law research: at training scales where modern language models operate, bias terms contribute negligible expressivity while adding parameter count and optimizer state. Removing them simplifies weight initialisation (no need to initialise biases separately) and reduces memory usage.

### Untied (separate) token embedding and output embedding

```python
# From gpt.py: GPT.__init__()
self.transformer = nn.ModuleDict({
    "wte": nn.Embedding(padded_vocab_size, config.n_embd),  # input
    # ...
})
self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)  # output
```

GPT-2 used *tied* embeddings: the output matrix `lm_head.weight` was shared with the input embedding `wte.weight`. The argument was parameter efficiency: the same matrix encodes "what tokens mean" (input) and "which token to predict" (output).

nanochat uses separate matrices with different initialisations:

```python
# From gpt.py: init_weights()
torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
```

The input embedding initialises with std 0.8 (large, to spread tokens across the embedding space). The output head initialises with std 0.001 (near-zero, so training starts from approximately uniform predictions). Tied embeddings cannot have different initialisations, so untying is necessary here.

### RMSNorm with no learnable parameters

```python
# From gpt.py
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

Standard Layer Normalisation (Ba et al., 2016) computes:

$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sigma} + \beta$$

with learnable gain $\gamma$ and bias $\beta$ per dimension. RMSNorm (Zhang and Sennrich, 2019) simplifies this by:
1. Removing the mean subtraction (the $-\mu$ term) — empirically this has little effect
2. Removing the learnable parameters $\gamma$ and $\beta$

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}$$

nanochat's `norm()` calls `F.rms_norm(x, (x.size(-1),))` without providing any weight or bias argument, making it purely a scale normalisation with zero learnable parameters.

This is possible because the model has other mechanisms to control the effective scale at each normalisation point: the projection weights that follow each `norm()` call learn the appropriate scale. Removing $\gamma$ and $\beta$ reduces the parameter count and eliminates one potential source of training pathologies.

---

## Section 14 — Full GPT forward pass walkthrough

**File:** `nanochat/gpt.py`, `GPT.forward()`

Let's trace a single training step through the entire forward pass with shape annotations. Assume `B=4` (batch size), `T=2048` (sequence length), `n_embd=768`, `n_head=6`, `head_dim=128`, `n_layer=12`.

### Step 1: Token embedding and normalisation

```python
x = self.transformer.wte(idx)  # (4, 2048) → (4, 2048, 768)
x = x.to(COMPUTE_DTYPE)        # cast to bf16
x = norm(x)                    # RMSNorm: (4, 2048, 768)
```

Each integer token id is looked up in the embedding table. The result is a dense vector of size 768. RMSNorm is applied immediately after — this is not standard (GPT-2 does not normalise the embedding), but empirically it stabilises training.

### Step 2: Rotary embedding setup

```python
T0 = 0 if kv_cache is None else kv_cache.get_pos()
cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
# cos, sin: (1, 2048, 1, 64) each
```

The precomputed cosine and sine tables are sliced to the current sequence length (and offset for inference with KV cache).

### Step 3: Smearing

```python
gate = self.smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
# gate: (4, 2047, 1)
x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
# x: (4, 2048, 768) — token i now carries info from token i-1
```

### Step 4: Save x0 and enter layer loop

```python
x0 = x  # (4, 2048, 768)
x_backout = None
backout_layer = 6  # n_layer // 2
```

### Step 5: For each of 12 transformer blocks

**5a. Residual scaling and x0 blending:**

```python
x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
```

At layer 0: `x = 1.15 * x + 0.20 * x0` — a strong blend of initial embedding.
At layer 11: `x ≈ 1.05 * x + 0.05 * x0` — mostly residual, small initial embedding contribution.

**5b. Value embedding lookup (alternating layers):**

```python
ve = self.value_embeds[str(i)](idx).to(x.dtype)
# ve: (4, 2048, kv_dim)
```

**5c. Attention sublayer:**

Inside `Block.forward()`:
```python
x = x + self.attn(norm(x), ve, cos_sin, self.window_sizes[i], kv_cache)
```

Inside `CausalSelfAttention.forward()`:

1. Project: `q = c_q(x)` → `(4, 2048, 6, 128)`, `k = c_k(x)` → `(4, 2048, n_kv_head, 128)`
2. Value embed: `v = v + gate * ve` (on VE layers)
3. RoPE: rotate Q and K with cos/sin tables
4. QK norm: `q, k = norm(q), norm(k)`, then scale by 1.2
5. Flash Attention: `y = flash_attn(q, k, v, window_size=window_sizes[i])`
6. Reshape: `y = y.view(4, 2048, 768)`
7. Output projection: `y = c_proj(y)` → `(4, 2048, 768)`

**5d. MLP sublayer:**

```python
x = x + self.mlp(norm(x))
```

Inside `MLP.forward()`:
1. `x = c_fc(x)` → `(4, 2048, 3072)` (4× expansion)
2. `x = F.relu(x).square()` → squared ReLU activation
3. `x = c_proj(x)` → `(4, 2048, 768)` (project back)

**5e. Backout cache:**

```python
if i == 6:
    x_backout = x
```

### Step 6: Backout and final norm

```python
x = x - self.backout_lambda * x_backout   # subtract mid-layer state
x = norm(x)                                # (4, 2048, 768)
```

### Step 7: Logit projection and softcap

```python
logits = self.lm_head(x)                       # (4, 2048, padded_vocab_size)
logits = logits[..., :32768]                   # slice padding
logits = logits.float()                        # cast to fp32
logits = 15 * torch.tanh(logits / 15)          # softcap
```

### Step 8: Loss computation (training) or return logits (inference)

```python
loss = F.cross_entropy(
    logits.view(-1, 32768),
    targets.view(-1),
    ignore_index=-1,
    reduction='mean'
)
```

The cross-entropy loss is the negative log probability of the correct next token:

$$\mathcal{L} = -\frac{1}{BT} \sum_{b,t} \log p_\theta(x_{b,t+1} \mid x_{b,1:t})$$

This loss signal backpropagates through every operation described above, adjusting all parameters to make the model more likely to predict the correct next token.

---

## A note on the MLP activation: squared ReLU

The MLP uses:

```python
# From gpt.py: MLP.forward()
x = F.relu(x).square()
```

This is $\text{ReLU}^2(x) = \max(0, x)^2$, sometimes written $\text{ReGLU}$ variants or $\text{ReReLU}$.

Compare to standard activations:
- **ReLU**: $\max(0, x)$ — simple, but creates dead neurons when $x < 0$ throughout training
- **GELU**: $x \Phi(x)$ — smooth, widely used in GPT-2/3/4; computationally more expensive
- **SiLU/Swish**: $x \sigma(x)$ — similar to GELU; used in Llama

Squared ReLU has several appealing properties:
1. **Sparsity:** For large negative inputs, output is exactly 0. For positive inputs, larger values are amplified quadratically. This creates natural sparse activation patterns.
2. **Simplicity:** No approximation needed (unlike GELU) — just ReLU followed by element-wise squaring.
3. **Empirical performance:** Found to match or exceed GELU in some controlled studies at the compute scales relevant to nanochat.

---

## Check your understanding

**Question 1.** Two tokens are at positions $m = 100$ and $n = 103$ in a sequence. After applying RoPE, the attention score between their query and key depends on the relative gap $n - m = 3$, not on the absolute positions. Explain why: which mathematical property of rotation matrices makes this true? What would change if you used learned absolute positional embeddings instead?

**Question 2.** Suppose `n_head = 8` and `n_kv_head = 2`. Draw out which query heads share which KV heads. How many times is each key/value head "expanded" to serve its group of query heads? What is the reduction factor in KV cache memory compared to standard MHA?

**Question 3.** Consider a 12-layer model with `window_pattern = "SL"`. Write out the window assignment for all 12 layers, remembering the rule that the last layer always gets `L`. How many `S` layers and how many `L` layers are there? Why must the final layer always be `L`?

---

## What's next

You now understand every architectural decision in `gpt.py`. The next chapter covers **training dynamics** — how the weight initialisation scheme, the Muon/AdamW optimizer split, learning rate schedules, and the FLOPs budget calculation all work together to make training stable and efficient. We will trace a complete training step from data batch to parameter update.

Chapter 5 will reference many of the concepts introduced here — particularly the precision setup (fp32 master weights, bf16 activations) and the parameter groupings in `setup_optimizer()` — so re-reading Sections 2 and 10 before continuing is worthwhile.
