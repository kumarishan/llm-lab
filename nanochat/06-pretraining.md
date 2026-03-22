# Chapter 6: Pretraining — The Training Loop, Optimizers, and Mixed Precision

## What you'll learn

- How the full pretraining loop is structured: forward pass, loss, backward pass, optimizer step, repeat
- How cross-entropy loss is computed and why Bits Per Byte (BPB) is a better evaluation metric
- The mathematics of AdamW: momentum, variance, bias correction, and decoupled weight decay
- How the Muon optimizer improves on Adam for weight matrices by orthogonalizing gradient updates via the Polar Express iteration
- How mixed-precision training (bf16, fp16, fp8) saves memory and speeds up training without sacrificing model quality

## Prerequisites

- Chapter 4: GPT Architecture — you know what the model looks like and what a forward pass produces
- Chapter 5: Data Pipeline — you know how batches of token IDs are loaded and what `x` and `y` mean
- Basic Python familiarity; no prior ML training experience required

---

## 6.1 The Training Loop at a Glance

At its core, pretraining is a loop that repeats the same five actions thousands or millions of times:

```
1. Get a batch of tokens  (x, y)
2. Forward pass           logits = model(x)
3. Compute loss           loss = cross_entropy(logits, y)
4. Backward pass          loss.backward()  — fills .grad on every parameter
5. Optimizer step         optimizer.step() — nudges every parameter toward lower loss
```

Each full trip through steps 1–5 is one **training step** (also called an iteration). The total number of tokens seen during training is:

```
total_tokens = batch_size_in_tokens × num_steps
```

For nanochat, a typical run at `--depth 20` trains on roughly 20 billion tokens over ~38,000 steps with a batch size of 524,288 tokens per step.

Here is a minimal skeleton of a training loop, stripped of all the production details, that captures the essential structure:

```python
# Minimal training loop skeleton (conceptual — not from the repo)
model = GPT(config)
optimizer = build_optimizer(model.parameters())

for step in range(num_steps):
    x, y = get_batch()                    # x: (B, T) input tokens
                                           # y: (B, T) target tokens (x shifted left by 1)
    loss = model(x, y)                    # forward pass + loss computation
    loss.backward()                        # backward pass: fill .grad on all parameters
    optimizer.step()                       # update parameters
    model.zero_grad(set_to_none=True)     # reset gradients for next step
```

Everything in this chapter is an elaboration of one of these five lines.

---

## 6.2 Cross-Entropy Loss: Learning to Predict the Next Token

### What the model outputs

After a forward pass, the model returns a tensor of **logits** with shape `(B, T, V)`, where:
- `B` is the batch size (number of sequences)
- `T` is the sequence length
- `V` is the vocabulary size (number of possible tokens)

Each row `logits[b, t, :]` is a vector of `V` raw scores — one per token — representing how confident the model is that token `v` comes after the first `t` tokens of sequence `b`. These scores have not been normalized yet.

### Converting logits to probabilities

To compare against a true next token, we convert the logit vector to a probability distribution using the **softmax** function:

```
p(v | context) = exp(z_v) / sum_w exp(z_w)
```

where `z_v` is the logit for token `v`.

### The cross-entropy formula

If the true next token is `y`, the loss for this single prediction is:

```
L = -log p(y | context) = -log [ exp(z_y) / sum_w exp(z_w) ]
```

This is the negative log-probability of the correct token. When the model is very confident and correct, `p(y)` is close to 1 and the loss approaches 0. When the model is wrong or uncertain, `p(y)` is small and the loss is large.

In practice we never call softmax explicitly — PyTorch's `F.cross_entropy` fuses the softmax and the log for numerical stability.

### Averaging over tokens and batches

The model processes `B × T` predictions in a single forward pass. The loss for the whole batch is the mean:

```
L_batch = (1 / (B × T)) × sum over all (b, t) of -log p(y[b,t] | context[b,t])
```

Why average rather than sum? Averaging makes the loss independent of batch size and sequence length, so the same learning rate works regardless of those hyperparameters.

### The `ignore_index` for masked positions

Not every position in `y` should contribute to the loss. Special tokens like `<|bos|>` placed at the beginning of a document should not be predicted — they are padding artifacts, not real text. Positions marked with `y = -1` (the `ignore_index`) are excluded from the average automatically by `F.cross_entropy`.

In `loss_eval.py`, the evaluation function handles this explicitly:

```python
# from nanochat/loss_eval.py
if (y.int() < 0).any():
    valid = y >= 0
    y_safe = torch.where(valid, y, torch.zeros_like(y))
    num_bytes2d = torch.where(
        valid,
        token_bytes[y_safe],
        torch.zeros_like(y, dtype=token_bytes.dtype)
    )
    total_nats += (loss2d * (num_bytes2d > 0)).sum()
    total_bytes += num_bytes2d.sum()
```

The `num_bytes2d > 0` mask ensures that both ignored positions (index -1) and special tokens (byte length 0) are excluded.

---

## 6.3 Bits Per Byte (BPB): A Tokenizer-Agnostic Metric

### Why not just report the loss?

Cross-entropy loss is measured in **nats** (natural logarithm base). Its numerical value depends on the vocabulary size: a model with a 50,000-token vocabulary operating at 3.2 nats is not directly comparable to one with a 100,000-token vocabulary also at 3.2 nats, because each token encodes a different number of bits of information.

nanochat reports **Bits Per Byte (BPB)** instead, which normalizes by the number of UTF-8 bytes the predicted tokens represent, making comparisons across tokenizers meaningful.

### Converting nats to bits per byte

The conversion has two steps:

1. **Nats to bits**: divide by `log(2)` (since 1 nat = 1/log(2) bits ≈ 1.443 bits)
2. **Bits to bits-per-byte**: divide by the number of bytes the target tokens represent

Combining:

```
BPB = total_nats / (log(2) × total_bytes)
```

A BPB of 1.0 would mean the model perfectly predicts every byte. In practice, a well-trained model achieves BPB around 0.8–0.9 on English text. Lower is better.

### Code walkthrough: `evaluate_bpb` in `nanochat/loss_eval.py`

```python
# from nanochat/loss_eval.py
@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none')  # (B, T) — per-token losses
        loss2d = loss2d.view(-1)
        y = y.view(-1)
        # fast path: no ignored positions
        num_bytes2d = token_bytes[y]
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()
    # ...
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
```

Key points:
- `loss_reduction='none'` asks the model to return per-token losses instead of the batch mean
- `token_bytes` is a 1-D tensor of shape `(vocab_size,)` that maps each token ID to the number of UTF-8 bytes it represents; special tokens that should not be counted (like `<|bos|>`) have byte length 0
- The multiplication `loss2d * (num_bytes2d > 0)` zeros out any loss for special tokens before summing
- Across multiple GPUs, `dist.all_reduce` sums the totals before computing the ratio

---

## 6.4 Gradient Descent: How Parameters Learn

### The basic idea

Every parameter `θ` in the model has an associated gradient `∂L/∂θ` — the direction in which increasing `θ` increases the loss. The simplest update rule is plain gradient descent:

```
θ ← θ - lr × ∂L/∂θ
```

where `lr` is the **learning rate** (a small positive number like 0.001).

### How PyTorch computes gradients: autograd

When you call `loss.backward()`, PyTorch traverses the computation graph built during the forward pass in reverse, applying the chain rule at each operation. Every parameter tensor that has `requires_grad=True` accumulates its gradient into a `.grad` attribute.

The key point: gradients accumulate by addition. If you call `.backward()` twice without zeroing gradients in between, the second call adds to the first. That is why `model.zero_grad(set_to_none=True)` must be called at the end of each step — or, in gradient accumulation, deliberately not called until all micro-batches are processed.

### Gradient clipping

With deep networks, gradients can sometimes become very large (a phenomenon called **exploding gradients**), causing the parameters to take a huge step and destabilizing training. The standard remedy is to clip the global gradient norm to a maximum value:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This rescales all gradients proportionally so that their combined L2 norm does not exceed `max_norm`. nanochat relies on the Muon optimizer's built-in orthogonalization (Section 6.6) to keep update norms bounded, which reduces the need for explicit clipping, but it is a standard tool to know about.

---

## 6.5 AdamW: The Workhorse Optimizer

Plain gradient descent has a critical weakness: it uses the same learning rate for every parameter, regardless of how frequently or how noisily that parameter's gradient fluctuates. **Adam** (Adaptive Moment Estimation) fixes this by tracking a running estimate of the gradient's mean and variance and using them to scale each parameter's update individually.

### First moment: momentum

Instead of using the raw gradient `g_t` directly, Adam maintains an **exponential moving average** of past gradients:

```
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
```

`m_t` is called the **first moment** (it is an estimate of E[g]). The default `β₁ = 0.9` means the average is weighted heavily toward recent gradients but has some memory of earlier ones. This smooths out noisy gradient estimates and gives the update "momentum" — the parameter keeps moving in roughly the same direction even if a single gradient is misleading.

### Second moment: adaptive learning rates

Adam also maintains an exponential moving average of squared gradients:

```
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
```

`v_t` is the **second moment** (an estimate of E[g²]). Its square root `√v_t` approximates the standard deviation of the gradient. Dividing the update by `√v_t` gives parameters with noisy (high variance) gradients a smaller effective learning rate, and parameters with consistent (low variance) gradients a larger one.

### Bias correction

There is a subtle problem with starting both `m_0 = 0` and `v_0 = 0`: in the early steps, the estimates are biased toward zero because they have not had time to build up. Adam corrects for this by dividing each moment by `(1 - β^t)`:

```
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
```

At step `t = 1`, the correction is `1 / (1 - 0.9) = 10×`, which scales the moment up to compensate for the cold start. As `t` grows large, `β^t → 0` and the correction approaches 1 (no effect).

### Weight decay: decoupled L2 regularization

Ordinary L2 regularization adds a penalty `λ/2 × ||θ||²` to the loss, which results in a gradient term `λ × θ` that is added to `g_t` before the update. When this is done inside Adam, the weight decay gets divided by `√v_t` just like the gradient, which means parameters with large gradients receive proportionally less regularization. This is undesirable.

**AdamW** ("W" for weight decay) decouples the weight decay from the gradient: the decay is applied directly to the parameter, independently of the moment estimates:

```
θ ← θ × (1 - lr × λ)   [weight decay step]
θ ← θ - lr × m̂_t / (√v̂_t + ε)   [adam step]
```

This ensures every parameter decays toward zero at a consistent rate, regardless of gradient noise.

### The full AdamW update

Putting it all together, the complete update at step `t` is:

```
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

θ_t = θ_{t-1} × (1 - lr × λ)  −  lr × m̂_t / (√v̂_t + ε)
```

Typical hyperparameters: `β₁ = 0.9`, `β₂ = 0.95`, `ε = 1e-8`, `λ = 0.1`.

### nanochat's fused AdamW implementation

nanochat implements this in `nanochat/optim.py` as a single compiled kernel decorated with `@torch.compile`:

```python
# from nanochat/optim.py — adamw_step_fused
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t,
                     beta1_t, beta2_t, eps_t, wd_t) -> None:
    # Weight decay (decoupled, applied before the update)
    p.mul_(1 - lr_t * wd_t)
    # Update running averages (lerp_ is cleaner and fuses well)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    # Bias corrections
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # Compute update and apply
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)
```

A few implementation details worth understanding:

**`lerp_`** (`linear interpolation`): `x.lerp_(y, w)` computes `x = x * (1-w) + y * w`, which is mathematically equivalent to the exponential moving average update `exp_avg = β₁ * exp_avg + (1-β₁) * grad` but written in a form that fuses more cleanly in the compiled graph.

**`@torch.compile(dynamic=False, fullgraph=True)`**: This decorator instructs PyTorch to compile the entire function into a single optimized CUDA kernel using the Inductor backend. `dynamic=False` tells the compiler that tensor shapes will not change, enabling more aggressive optimization. `fullgraph=True` requires the entire function to be captured as one graph (no Python fallback for parts of it). The result is that all seven operations — weight decay, two moment updates, two bias corrections, division, and parameter update — are fused into a single GPU kernel with no Python overhead between them.

**0-D CPU tensors for hyperparameters**: The function takes hyperparameters as 0-D tensors (`step_t`, `lr_t`, etc.) rather than Python scalars. This prevents `torch.compile` from recompiling the kernel every time the learning rate changes (as it would if scalars were traced as constants).

### Which parameters use AdamW?

AdamW is used for parameters that are not full weight matrices: embeddings (the token embedding lookup table), the unembedding (the final projection to vocabulary logits), and scalar parameters like learned residual scaling factors. These parameters either have no obvious "row/column" structure to exploit or are too small for Muon's machinery to add value.

---

## 6.6 The Muon Optimizer: Orthogonal Updates for Weight Matrices

### The problem with Adam for weight matrices

In a transformer, most of the learnable parameters live in large 2-D weight matrices (query/key/value projections, feedforward layers, etc.). When Adam updates one of these matrices, it scales each individual element's update by the inverse of that element's gradient standard deviation. The resulting update has no geometric structure — elements are updated independently with no awareness of how the matrix operates as a linear map.

A more principled approach asks: given a gradient matrix `G`, what is the best step we can take in the space of weight matrices? If we define "best" as maximizing the decrease in loss for a unit-sized step (measured by the spectral norm or Frobenius norm), the answer is to replace `G` with its **nearest orthogonal matrix** — the matrix `Q` such that `Q^T Q = I` (for wide matrices) or `Q Q^T = I` (for tall matrices) that is closest to `G` in Frobenius norm.

This nearest orthogonal matrix is `U V^T` from the SVD `G = U Σ V^T`. It has the property that all singular values are exactly 1, meaning every "direction" in weight space gets an equal-sized step, rather than large directions dominating the update.

### Muon: SGD momentum followed by orthogonalization

**Muon** (Momentum Orthogonalized by Newton-Schulz) combines two ideas:

1. Apply Nesterov momentum to the raw gradient to smooth out noise
2. Orthogonalize the smoothed gradient before applying it as an update

The update rule is:

```
g̃_t = β × g̃_{t-1} + (1 - β) × g_t          [Nesterov momentum]
G    = nesterov_combination(g_t, g̃_t)
Q    = orthogonalize(G)                       [Polar Express iteration]
θ_t  = θ_{t-1} - lr × Q                      [parameter update]
```

The orthogonalized matrix `Q` has unit spectral norm, which means the optimizer takes a step of well-defined size in weight space regardless of the gradient's scale.

### The Polar Express iteration: fast approximate orthogonalization

Computing the exact SVD at every step would be too slow. Instead, nanochat uses the **Polar Express** algorithm (arXiv:2505.16932), a Newton-Schulz iteration that computes a matrix polynomial approximation of the orthogonal factor.

The iteration starts by normalizing the gradient matrix:

```
X₀ = G / (||G||_F × 1.01 + ε)
```

Then it applies 5 iterations of a quintic polynomial:

For a **wide** matrix (`rows ≤ cols`):
```
A = X @ X^T
B = b * A + c * (A @ A)
X = a * X + B @ X
```

For a **tall** matrix (`rows > cols`):
```
A = X^T @ X
B = b * A + c * (A @ A)
X = a * X + X @ B
```

The constants `(a, b, c)` for each of the 5 iterations are precomputed to maximize the convergence rate. nanochat defines them in `optim.py`:

```python
# from nanochat/optim.py
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]
```

After 5 iterations, `X` approximates `U S' V^T` where `S'` is diagonal with entries approximately uniform on `[0.5, 1.5]` — close to the identity but not exactly. The comment in the code explains that this imperfect orthogonalization does not hurt model performance in practice and has better convergence properties than the original Newton-Schulz iteration used in the earlier modded-nanogpt implementation.

The polar express computation runs in bfloat16, which is fast and sufficient for this approximate orthogonalization.

### Factored second moments: per-row and per-column variance

After orthogonalization, all rows of the update matrix nominally have unit norm. But in practice, different rows of the gradient can have different scales even after orthogonalization, because the polar express iteration is only approximate. To compensate, nanochat applies a **variance reduction** step that rescales each row (or column) of the update.

Instead of maintaining a full `(rows, cols)` variance buffer (which would double the memory cost), nanochat uses a **factored second moment**: a single vector of length `rows` (for tall matrices) or `cols` (for wide matrices). This is the same trick used in Adafactor.

The `red_dim` variable in `muon_step_fused` controls which dimension is reduced:

```python
# from nanochat/optim.py — inside _step_muon
red_dim = -1 if shape[-2] >= shape[-1] else -2
```

For a tall matrix (more rows than columns), the second moment is shaped `(num_params, rows, 1)` — one variance estimate per row. For a wide matrix, it is `(num_params, 1, cols)` — one per column.

### Cautious weight decay

Standard weight decay applies uniformly: `θ ← θ × (1 - lr × λ)`. Muon uses a **cautious** variant that only decays a weight when the gradient and the weight have the same sign — that is, when the parameter is moving away from zero on its own, decay pushes it back. When the gradient already points toward zero (signs differ), no extra decay is applied.

```python
# from nanochat/optim.py — inside muon_step_fused
mask = (g * stacked_params) >= 0
stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
```

The intuition: weight decay is a regularizer designed to prevent weights from growing too large. If the gradient is already reducing a weight's magnitude, adding more decay is redundant and might over-penalize useful structure. The cautious mask makes decay purely additive: it only acts when it would not conflict with the gradient direction.

### The full Muon step

```python
# from nanochat/optim.py — muon_step_fused (annotated)
@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer,
                    second_momentum_buffer, momentum_t, lr_t, wd_t, beta2_t,
                    ns_steps, red_dim) -> None:

    # Step 1: Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Step 2: Polar Express orthogonalization (5 iterations)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):  # tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:                        # wide matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # Step 3: Factored variance reduction (per-row or per-column scaling)
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Step 4: Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
```

### Stacking for efficiency

Muon processes all parameters with the same shape together in a single batched kernel call. In `_step_muon`, individual parameter tensors are stacked into a single `(num_params, rows, cols)` tensor before calling `muon_step_fused`. This means the polar express iteration runs on many matrices in parallel, using the GPU efficiently.

```python
# from nanochat/optim.py — inside _step_muon
stacked_grads = torch.stack([p.grad for p in params])
stacked_params = torch.stack(params)
# ... single kernel call for all params ...
torch._foreach_copy_(params, list(stacked_params.unbind(0)))
```

### Learning rate scaling for non-square matrices

One subtlety: the learning rate for Muon is scaled by `sqrt(max(1, rows/cols))`:

```python
# from nanochat/optim.py — inside _step_muon
self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
```

This compensates for the fact that tall matrices (more rows than columns) have more "room to move" after orthogonalization and would otherwise take effectively smaller steps than square matrices.

### Why Muon only for weight matrices?

Muon requires the parameter to be a 2-D matrix so that the polar express iteration makes geometric sense. It should not be used for:
- The **embedding table** (shape `(vocab_size, d_model)`): it is accessed sparsely — only a few rows see gradient updates per batch — making orthogonalization of the full matrix inappropriate
- The **unembedding layer** (the final `lm_head`): it is tied to or closely related to the embedding and benefits from AdamW's independent element-wise scaling
- **Scalar parameters** (`0-D` or `1-D`): no meaningful matrix geometry to exploit

---

## 6.7 Mixed Precision Training

### Why not train in float32?

Float32 is the "safe" format: 8 exponent bits, 23 mantissa bits, no surprises. But it has a cost:
- **Memory**: a 1-billion-parameter model stored in fp32 requires 4 GB just for the weights, plus another 4–8 GB for optimizer states
- **Speed**: GPU tensor cores are optimized for lower-precision arithmetic; float32 matmuls run at roughly half the throughput of bfloat16

### The three formats

| Format | Exponent bits | Mantissa bits | Range | Notes |
|--------|--------------|----------------|-------|-------|
| float32 | 8 | 23 | ±3.4 × 10³⁸ | Safe default |
| bfloat16 | 8 | 7 | ±3.4 × 10³⁸ | Same range as fp32, less precision |
| float16 | 5 | 10 | ±6.5 × 10⁴ | Narrow range, needs GradScaler |

**bfloat16** is the preferred format for modern hardware (Ampere A100 and later, identified as CUDA SM ≥ 8.0). It has the same exponent range as float32, so gradients and activations are unlikely to overflow or underflow. The reduced mantissa (7 bits instead of 23) means less numerical precision, but language model training is surprisingly tolerant of this.

**float16** has a much narrower dynamic range. Gradients that are very small (near zero) or very large can overflow or underflow, causing NaN values to propagate through the network. Training in fp16 requires a **GradScaler**: the loss is multiplied by a large constant before backward, the gradients are divided by the same constant before the optimizer step, and the scale factor is adjusted dynamically to stay within fp16 range.

### nanochat's approach: explicit casting, not autocast

PyTorch's standard mixed precision API is `torch.autocast`, which automatically casts operations to the lower-precision type as they are encountered. nanochat takes a different approach: explicit casting inside the model.

The custom `Linear` class in `nanochat/gpt.py` stores its weights in float32 but casts them to `COMPUTE_DTYPE` at the start of each forward pass. This gives the optimizer high-precision weights to work with (important for AdamW's momentum and variance estimates) while running the actual matrix multiplications in bfloat16.

### The `COMPUTE_DTYPE` detection logic

`COMPUTE_DTYPE` is determined once at module import time in `nanochat/common.py`:

```python
# from nanochat/common.py
def _detect_compute_dtype():
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        # fp16 training requires GradScaler (not yet implemented), so fall back to fp32.
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"
COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()
```

The logic:
- If the environment variable `NANOCHAT_DTYPE` is set, use that (useful for experiments)
- On GPUs with SM ≥ 8.0 (A100, A10, H100, RTX 3090+), use bfloat16
- On older GPUs (V100, T4), fall back to float32 (fp16 would need a GradScaler, which adds complexity)
- On CPU or Apple Silicon (MPS), use float32

At startup, `base_train.py` prints the detected dtype and the reason:
```
COMPUTE_DTYPE: torch.bfloat16 (auto-detected: CUDA SM 80 (bf16 supported))
```

### GradScaler for fp16

If `COMPUTE_DTYPE` is `torch.float16` (forcing fp16 on an older GPU), nanochat initializes a GradScaler:

```python
# from scripts/base_train.py
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
```

The training loop then wraps `.backward()` and `optimizer.step()` with the scaler:

```python
# from scripts/base_train.py — inside the training loop
if scaler is not None:
    scaler.scale(loss).backward()
else:
    loss.backward()

# ... then, before optimizer.step():
if scaler is not None:
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
else:
    optimizer.step()
```

The GradScaler multiplies the loss by a large factor (typically 2¹⁶) before calling backward. This shifts the gradient magnitudes up into the representable range of float16. Before the optimizer step, `scaler.unscale_` divides them back. If any gradient is inf or NaN, the step is skipped and the scale factor is halved.

---

## 6.8 FP8 Training (Optional, H100+)

On NVIDIA Hopper GPUs (H100, H800), PyTorch exposes an even lower-precision format: **float8**. FP8 training can achieve roughly 2x the throughput of bfloat16 because cuBLAS can execute 8-bit matrix multiplications at double the rate of 16-bit.

There are two float8 formats, used for different roles:
- **`float8_e4m3fn`**: 4 exponent bits, 3 mantissa bits, range ±448. Higher precision; used for input activations and weights in the forward pass
- **`float8_e5m2`**: 5 exponent bits, 2 mantissa bits, range ±57344. Wider range; used for gradients in the backward pass, which can be larger in magnitude

### Dynamic scaling to prevent underflow/overflow

Float8 has a tiny representable range. Without intervention, most values would round to zero (underflow) or infinity (overflow). Dynamic scaling solves this:

1. Compute `amax = max(|tensor|)` across the entire tensor
2. Compute `scale = FP8_MAX / amax` — maps the largest value to the edge of the FP8 range
3. Multiply the tensor by `scale`, cast to float8, and store `1/scale` for dequantization
4. Pass the quantized tensor and its inverse scale to `torch._scaled_mm`, which handles dequantization internally

### nanochat's minimal implementation

`nanochat/fp8.py` implements this in roughly 150 lines as a drop-in replacement for PyTorch's `nn.Linear`:

```python
# from nanochat/fp8.py — Float8Linear.forward (simplified)
class Float8Linear(nn.Linear):
    def forward(self, input):
        input = input.to(COMPUTE_DTYPE)      # cast to bf16
        input_2d = input.reshape(-1, orig_shape[-1])
        output = _Float8Matmul.apply(input_2d, self.weight)  # FP8 matmul
        # ...
```

The `_Float8Matmul` autograd function quantizes both input and weight to `e4m3fn`, calls `torch._scaled_mm`, and saves the quantized tensors for the backward pass (which uses `e5m2` for gradient quantization).

To enable FP8 training, pass `--fp8` to `base_train.py`. The script converts all eligible `nn.Linear` layers (those with dimensions divisible by 16 and at least 128 units wide) to `Float8Linear`:

```python
# from scripts/base_train.py
if args.fp8:
    fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
    convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
```

Evaluation is always done in bfloat16 (with FP8 temporarily disabled via a context manager) to avoid the precision loss affecting metrics.

---

## 6.9 Gradient Accumulation: Simulating Large Batches

### The problem

Larger batch sizes generally lead to better, more stable training — gradients averaged over more samples have lower variance. But GPU memory is finite. A batch size of 524,288 tokens at sequence length 2048 would require `524288 / 2048 = 256` sequences processed simultaneously, which far exceeds the memory of most GPUs.

### The solution: accumulate gradients over micro-batches

Instead of processing all 256 sequences at once, we process them in smaller **micro-batches** (e.g., 32 sequences at a time) and accumulate the gradients across multiple forward-backward passes before calling the optimizer.

Since `.backward()` adds to `.grad` rather than overwriting it, we get the same total gradient as if we had processed all sequences at once — but using only `1/N` of the memory.

```python
# Gradient accumulation pattern (conceptual)
optimizer.zero_grad()

for micro_step in range(grad_accum_steps):
    x, y = next(micro_batch_loader)
    loss = model(x, y)
    loss = loss / grad_accum_steps  # IMPORTANT: normalize so total loss is correct
    loss.backward()                  # gradients accumulate in .grad

optimizer.step()
```

The critical step is dividing the loss by `grad_accum_steps` before calling `.backward()`. Without this division, the gradients would be `grad_accum_steps` times larger than expected, effectively multiplying the learning rate by `grad_accum_steps` and causing instability.

### How nanochat computes gradient accumulation steps

```python
# from scripts/base_train.py
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
```

If `total_batch_size = 524288`, `device_batch_size = 32`, `max_seq_len = 2048`, and there is 1 GPU:
```
tokens_per_fwdbwd = 32 × 2048 = 65,536
grad_accum_steps = 524,288 / 65,536 = 8
```

Eight micro-batches are processed per optimizer step. The training loop in `base_train.py` reflects this:

```python
# from scripts/base_train.py — training step
for micro_step in range(grad_accum_steps):
    loss = model(x, y)
    train_loss = loss.detach()  # for logging
    loss = loss / grad_accum_steps
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    x, y, dataloader_state_dict = next(train_loader)  # prefetch next batch
```

Note that the next batch is prefetched during the GPU backward pass, overlapping CPU data loading with GPU computation.

---

## 6.10 Learning Rate Scheduling

The learning rate is not fixed throughout training. A carefully shaped schedule improves both the speed of learning and the final model quality.

### The warmup phase

At the very start of training, the optimizer's moment estimates (`m_t`, `v_t`) are initialized to zero. They have not yet built up meaningful estimates of the gradient statistics. Using the full learning rate immediately would cause large, poorly directed parameter updates that can destabilize training. **Warmup** addresses this by ramping the learning rate from zero to its target value over the first few dozen steps.

### The constant phase

Once the moments are established, training proceeds at the full learning rate. This phase accounts for most of the training run.

### The warmdown phase

Near the end of training, reducing the learning rate allows the optimizer to fine-tune the parameters into a sharper, lower-loss minimum. The shape of the decay (linear, cosine, exponential) is largely a matter of empirical preference; nanochat uses a **linear warmdown**.

### nanochat's schedule

```python
# from scripts/base_train.py
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps          # default: 40 steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)  # default: 65% of total
    if it < warmup_iters:
        return (it + 1) / warmup_iters        # linear ramp: 0 → 1
    elif it <= num_iterations - warmdown_iters:
        return 1.0                            # constant at maximum LR
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac  # linear decay to final_lr_frac
```

The function returns a multiplier in `[final_lr_frac, 1.0]`. The actual learning rate for each parameter group is `initial_lr × multiplier`. With `--warmdown-ratio 0.65` and `--final-lr-frac 0.05`:
- Steps 0–39: LR ramps from 0 to `initial_lr`
- Steps 40 to 35% of total: constant at `initial_lr`
- Final 65% of training: linear decay from `initial_lr` to `0.05 × initial_lr`

This schedule is unusual in that the warmdown occupies the majority of training — most of the training budget is spent carefully converging into a good minimum, not exploring at high learning rate.

The multiplier is applied at each step inside the training loop:

```python
# from scripts/base_train.py — inside training loop
lrm = get_lr_multiplier(step)
for group in optimizer.param_groups:
    group["lr"] = group["initial_lr"] * lrm
    if group['kind'] == 'muon':
        group["momentum"] = get_muon_momentum(step)
        group["weight_decay"] = get_weight_decay(step)
```

### Muon momentum schedule

Muon's momentum coefficient also varies over training:

```python
# from scripts/base_train.py
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97  # ramp from 0.85 to 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress  # decay to 0.90 during warmdown
    else:
        return 0.97
```

At the very start of training, momentum is 0.85 (shorter memory). It quickly ramps to 0.97 (longer memory), giving updates more inertia for the stable middle of training. During the warmdown, it is reduced to 0.90, allowing the optimizer to react more quickly to the changing loss landscape.

### Weight decay schedule

Weight decay for Muon follows a cosine decay to zero:

```python
# from scripts/base_train.py
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))
```

This starts at `weight_decay_scaled` and decays smoothly to 0 over the course of training. The rationale: regularization is most important early in training when the model might overfit to the first documents it sees; near convergence, full regularization can actually pull parameters away from a good minimum.

---

## 6.11 Hands-On: Run Pretraining on CPU

You now know enough to run a small pretraining experiment. The following command trains a tiny model on CPU, which completes in a few minutes and demonstrates the full training pipeline without requiring a GPU.

> ✍️ Run this from the root of the nanochat repository:

```bash
python -m scripts.base_train \
    --depth=4 \
    --max-seq-len=512 \
    --device-batch-size=1 \
    --eval-tokens=512 \
    --core-metric-every=-1 \
    --total-batch-size=512 \
    --num-iterations=20
```

**What each flag does:**

| Flag | Value | Meaning |
|------|-------|---------|
| `--depth` | `4` | Model size dial: 4 transformer layers, `4 × 64 = 256`-dim model |
| `--max-seq-len` | `512` | Context window: sequences of up to 512 tokens |
| `--device-batch-size` | `1` | One sequence per micro-batch (fits in CPU memory) |
| `--eval-tokens` | `512` | Only evaluate on 512 tokens (fast validation) |
| `--core-metric-every` | `-1` | Disable the slower CORE benchmark evaluation |
| `--total-batch-size` | `512` | Full batch = 512 tokens (one micro-batch, no accumulation) |
| `--num-iterations` | `20` | Train for exactly 20 optimizer steps |

**Expected output:**

```
step 00000/00020 (0.00%) | loss: 10.693452 | lrm: 0.03 | ...
step 00001/00020 (5.00%) | loss: 10.686433 | lrm: 0.05 | ...
...
step 00010/00020 (50.00%) | loss: 9.932514 | lrm: 1.00 | ...
...
step 00020/00020 (100.00%) | loss: 9.103827 | lrm: 0.05 | ...
```

The loss should decrease from roughly 10.7 (near `log(vocab_size)` — i.e., all tokens equally likely) toward 9–10 over 20 steps. With only 20 steps on a tiny model, you will not see meaningful learning; this run is just verifying that the full pipeline executes correctly.

> **What's happening inside those 20 steps:**
> 1. The model starts with randomly initialized weights. A random model assigns roughly equal probability to all `~50,000` tokens, giving loss `≈ log(50000) ≈ 10.8`
> 2. After each step, AdamW (for embeddings) and Muon (for weight matrices) update the parameters
> 3. The learning rate starts near zero (warmup), then peaks, then decays (warmdown covers 65% of 20 steps, so it starts at step ~7)
> 4. The loss printed is an EMA-smoothed training loss, not the BPB metric — BPB is only computed when `--eval-every` triggers

### GPU training command

For a meaningful pretraining run on a GPU, the defaults are already tuned for good results:

> ✍️ On a single A100 or similar:

```bash
python -m scripts.base_train \
    --depth=20 \
    --run=my-run-name
```

This trains a ~400M parameter model with compute-optimal data allocation (~4B tokens at depth 20 with default ratio of 10.5). The defaults choose the batch size, learning rates, and training horizon automatically.

For multi-GPU training with 8 GPUs:

```bash
torchrun --nproc_per_node=8 -m scripts.base_train \
    --depth=20 \
    --run=my-run-name
```

For FP8 training on H100+:

```bash
python -m scripts.base_train \
    --depth=20 \
    --fp8 \
    --run=my-fp8-run
```

---

## 6.12 Monitoring with Weights and Biases

When `--run` is set to anything other than `"dummy"`, nanochat logs metrics to [Weights and Biases](https://wandb.ai). The script calls `wandb.init(project="nanochat", name=args.run, ...)` and logs to it throughout training.

### What gets logged

**Every 100 training steps:**
- `train/loss`: smoothed training loss (EMA with β=0.9)
- `train/lrm`: current learning rate multiplier
- `train/dt`: wall-clock time per step in seconds
- `train/tok_per_sec`: training throughput in tokens per second
- `train/mfu`: Model FLOP Utilization — the fraction of peak GPU FLOPs being used (e.g., 40% is good for a single A100 with bf16)

**Every 250 steps (default `--eval-every`):**
- `val/bpb`: Bits Per Byte on the validation set — this is the primary quality metric

**Every 2000 steps (default `--core-metric-every`):**
- `core_metric`: performance on the CORE benchmark (Chapter 7 covers evaluation in detail)

### Reading loss curves

A healthy training run has the following shape when you plot `train/loss` against step:
1. **Steep initial drop**: the model quickly moves away from the random initialization (high loss) in the first few hundred steps
2. **Gradual smooth decline**: loss decreases steadily throughout training
3. **Faster final drop**: during the warmdown phase (last 65% of steps), the lower learning rate typically causes the loss to decrease more steeply for a period as the model converges

Signs of problems:
- **Loss plateau early**: learning rate may be too low, or there is a bug in the data pipeline
- **Loss spike or NaN**: learning rate too high, or the model encountered a bad batch; if it recovers, it is a one-off; if it does not, restart with lower LR
- **Loss increasing**: something is very wrong — possibly incorrect gradient accumulation normalization or a bug in the optimizer
- **`val/bpb` significantly higher than `train/loss`** (in BPB terms): possible overfitting (unlikely at scale) or a mismatch between training and validation data distribution

### Model FLOP Utilization (MFU)

MFU is computed as:

```
MFU = (actual_flops_per_second) / (peak_flops_per_second)
```

where `actual_flops = num_flops_per_token × batch_size / step_time`. A value of 30–50% MFU is typical for a well-optimized single-GPU run; multi-GPU runs with efficient communication can reach higher. If MFU is below 20%, look for bottlenecks in data loading, gradient accumulation overhead, or memory bandwidth.

---

## Check Your Understanding

**Question 1.** The training loop divides the loss by `grad_accum_steps` before calling `.backward()`. What would happen if this division were omitted? How would it affect the effective learning rate?

**Question 2.** AdamW applies weight decay as `θ ← θ × (1 - lr × λ)`, while the original Adam applies it as an additive term inside the gradient update. Why does the original formulation interact badly with Adam's variance estimate? Specifically, what happens to the effective weight decay for parameters that have high-variance gradients?

**Question 3.** After the Polar Express iteration, the update matrix `X` has rows with approximately unit norm. But Muon still applies a variance reduction step. Looking at the code in `muon_step_fused`, what problem would occur if variance reduction were skipped entirely — that is, if the update `g = X` were applied directly with no per-row scaling?

---

## What's Next

Chapter 7 covers **evaluation**: how to measure what a pretrained model has actually learned beyond the BPB number. You will run the CORE benchmark, examine generation quality, and understand what "good" looks like for a base language model before any instruction tuning.
