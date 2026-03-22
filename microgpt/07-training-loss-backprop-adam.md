# Chapter 7: Training — Loss, Backprop, and Adam

Everything before this chapter was preparation: loading data, building a tokenizer, wiring up autograd, and constructing a forward pass. This chapter is where the model actually learns. You will trace through the training loop line by line, understand why cross-entropy loss is the right objective for next-token prediction, watch `loss.backward()` propagate gradients through the entire computation graph, and see exactly how the Adam optimizer uses those gradients to nudge parameters toward better predictions.

---

## What you'll learn

- How cross-entropy loss measures how wrong a language model's predictions are, and what the loss value tells you numerically
- Why averaging loss over a sequence is preferable to summing it
- How `loss.backward()` traverses the computation graph built by `gpt()` and deposits gradients on every parameter
- Why `p.grad = 0` at the end of each step is not optional
- How Adam maintains two running statistics — a gradient mean and a gradient variance — and why that makes it more reliable than plain gradient descent
- What bias correction does and why it matters most in the early steps
- How linear learning rate decay shapes the optimization trajectory

---

## Prerequisites

- Chapter 3 completed: you understand the `Value` class, the computation graph, and how `backward()` propagates gradients via the chain rule
- Chapter 6 completed: you can trace a token through `gpt()` and know what `logits` contains when it returns
- Familiarity with Python's `enumerate` and list comprehensions

---

## 7.1 The training loop at a glance

The full training loop spans lines 146–184 of `microgpt.py`. Before diving into each section, read it once as a whole:

```python
# microgpt.py  lines 146–184

# Adam optimizer buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)  # first moment buffer
v = [0.0] * len(params)  # second moment buffer

num_steps = 1000
for step in range(num_steps):
    # Tokenize document
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward pass
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    # Backward pass
    loss.backward()

    # Adam update
    lr_t = learning_rate * (1 - step / num_steps)  # linear LR decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')
```

Each iteration of the outer `for step` loop is one **training step**. A single step processes one document, runs a complete forward pass over its tokens, computes a scalar loss, propagates gradients backward, and adjusts every parameter once. The loop repeats 1 000 times, cycling through the shuffled document list with `step % len(docs)`.

---

## 7.2 Cross-entropy loss

### From logits to probabilities

After `gpt()` returns, `logits` is a Python list of `vocab_size` (28) `Value` objects. Each value is a raw, unbounded score. Before you can interpret them as probabilities they must be passed through `softmax`:

```python
probs = softmax(logits)
```

`softmax` (lines 97–101) subtracts the maximum logit for numerical stability, exponentiates each element, and divides by the total. The result is a probability distribution: all 28 values are positive and they sum to 1.0. `probs[i]` is the model's current estimate of "how likely is token `i` to come next?"

### Negative log-likelihood

With probabilities in hand, the loss for a single position is:

```python
loss_t = -probs[target_id].log()
```

This is the **negative log-likelihood** of the correct token under the model's predicted distribution — also called **cross-entropy loss** when applied to a one-hot target. Mathematically:

```
L_t = -log( p_θ(target | context) )
```

Two boundary cases make this formula intuitive:

| Model state | probs[target] | loss_t |
|---|---|---|
| Perfect prediction | 1.0 | −log(1.0) = **0.0** |
| Uniform (random model, 28 tokens) | 1/28 ≈ 0.036 | −log(0.036) ≈ **3.32** |
| Confident wrong prediction | ≈ 0.0 | −log(≈0) → **very large** |

The loss pushes the model to assign higher probability to the correct token and lower probability to everything else — without ever being given a direct "this is wrong" signal for incorrect tokens. The math handles it: raising the correct token's probability necessarily lowers all others because they must sum to 1.

> **What's happening — the baseline loss of 3.3**
>
> When you first run `microgpt.py`, the parameters are random Gaussian samples (standard deviation 0.08, line 80). The model has no knowledge yet, so its predictions are close to uniform over 28 tokens. You should therefore see the initial loss hover near `−log(1/28) = log(28) ≈ 3.3`. If you see something wildly different, check that the random seed is 42 and the parameters have not been accidentally modified before training.

### Averaging over the sequence

```python
loss = (1 / n) * sum(losses)
```

`n` is the number of positions processed in this document (capped at `block_size = 16`). Dividing by `n` gives the **mean** cross-entropy over the sequence rather than the sum.

Why average rather than sum? Documents have different lengths. A document with 12 characters produces 13 positions; one with 4 characters produces 5. Summing would make longer documents contribute larger raw loss values purely because of their length, and the gradient update would be larger after long documents than short ones. Averaging normalises for length so each step has roughly the same gradient magnitude regardless of which document was drawn.

---

## 7.3 The backward pass

### The computation graph grows during the forward pass

Every arithmetic operation on a `Value` node creates a new `Value` node and records the relationship between them. A single call to `gpt()` for one position triggers:

- Two embedding lookups (token + position) — each is a list read, no graph nodes yet
- An element-wise addition to combine embeddings — `n_embd = 16` add nodes
- `rmsnorm` — multiplications, a sum, a `**-0.5`, and 16 multiplications
- One transformer layer: Q/K/V projections (`linear`), softmax over attention logits, weighted sum of values, output projection, residual addition
- MLP block: two `linear` calls, 64 `relu` nodes, residual addition
- `lm_head` projection — 28 × 16 multiplications and additions

For a document of length `n`, `gpt()` is called `n` times, each time extending the same `keys` and `values` lists (the KV cache). The entire forward pass for one training step produces a graph with tens of thousands of `Value` nodes all connected up to the final scalar `loss`.

### loss.backward()

```python
loss.backward()
```

`backward()` (lines 59–72 of `microgpt.py`) does two things:

1. **Topological sort** — it walks the graph from `loss` to the leaf parameters, building a list where every node appears after all nodes that depend on it.
2. **Reverse accumulation** — it iterates that list in reverse, applying the chain rule at each node:

```python
child.grad += local_grad * v.grad
```

`local_grad` is the derivative of that node with respect to its child (stored at construction time), and `v.grad` is the gradient flowing back from above. The `+=` means gradients from multiple paths through the graph accumulate correctly.

After `loss.backward()` completes, every `Value` node in the graph — including every element of every weight matrix in `state_dict` — has a `.grad` that is `∂loss/∂parameter`.

### Why p.grad = 0 is not optional

```python
p.grad = 0
```

This single line at the end of the Adam update is one of the most common sources of bugs in hand-rolled training loops. If you omit it, the `+=` accumulation in `backward()` adds the new gradient *on top of* whatever was left from the previous step. After two steps, every `p.grad` would be the sum of two steps' gradients; after ten steps, ten steps' worth. The optimizer would see inflated gradients and take wildly oversized steps. Loss would diverge rather than decrease.

> **What's happening — stale gradients**
>
> To feel the effect, comment out `p.grad = 0` and run for 20 steps. You will likely see the loss spike or go to `nan` within a handful of steps. Then restore the line and watch loss decrease smoothly. The contrast makes the purpose of gradient zeroing concrete.

### The KV cache is reset per step

```python
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
```

This line appears inside the training loop, before the inner `for pos_id` loop. It creates fresh empty lists for each document. The attention mechanism in `gpt()` appends to these lists at each position, building up past keys and values so that position `t` can attend to positions 0 through `t`. Resetting them at the start of each document ensures that one document's context does not leak into the next.

---

## 7.4 The Adam optimizer

Plain gradient descent updates parameters as `p.data -= lr * p.grad`. That works, but it treats every parameter identically regardless of the history of its gradients, and a fixed learning rate that is large enough to be useful early on often becomes too large later and causes oscillation. Adam (Adaptive Moment Estimation, Kingma & Ba 2014) solves both problems by maintaining a running mean and running variance of each parameter's gradient history.

### First moment: gradient mean

```python
m[i] = beta1 * m[i] + (1 - beta1) * p.grad
```

`m[i]` is an **exponential moving average** of the gradient for parameter `i`. At each step, 85 % of the old average is retained (`beta1 = 0.85`) and 15 % of the new gradient is mixed in. This produces momentum: if the gradient has been consistently positive for many steps, `m[i]` will be a stable positive number rather than fluctuating with each noisy gradient sample.

### Second moment: gradient variance

```python
v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
```

`v[i]` tracks the exponential moving average of the **squared** gradient. A parameter whose gradient fluctuates wildly (high variance) will accumulate a large `v[i]`; a parameter whose gradient is consistently small will have a small `v[i]`.

### Bias correction

Both `m` and `v` are initialised to 0. In the very first step, `m[i]` is only `(1 - 0.85) * grad = 0.15 * grad` — a substantial underestimate of the true gradient mean. The bias correction terms fix this:

```python
m_hat = m[i] / (1 - beta1 ** (step + 1))
v_hat = v[i] / (1 - beta2 ** (step + 1))
```

At `step = 0` (first step): `1 - 0.85^1 = 0.15`, so `m_hat = m[i] / 0.15 = grad`. The denominator exactly undoes the initial underestimation. By step 20 or so, `beta1^21 ≈ 0.004`, the denominator is nearly 1, and bias correction becomes negligible.

The math behind this: if the true mean is `μ`, then `E[m[i]] = μ * (1 - beta1^t)` after `t` steps starting from 0. Dividing by `(1 - beta1^t)` recovers an unbiased estimate of `μ`.

### The update rule

```python
p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
```

Written as a formula:

```
θ_t = θ_{t-1} - α_t · m̂_t / (√v̂_t + ε)
```

The key insight is in the denominator. `√v̂_t` is the root mean square of recent gradients — a measure of how large this parameter's gradients have been. Dividing by it **normalises the step size per parameter**:

- A parameter with consistently large gradients gets a smaller effective step (`v` is large, denominator is large).
- A parameter with consistently small gradients gets a larger effective step (`v` is small, denominator is small).

This per-parameter adaptation is what makes Adam robust: you can use a single global learning rate and trust that Adam will scale it appropriately for each parameter based on its gradient history.

`eps_adam = 1e-8` is a small constant added before the square root to prevent division by zero when `v_hat` is extremely small.

### Hyperparameter choices in microgpt.py

| Hyperparameter | microgpt value | Standard Adam | Effect |
|---|---|---|---|
| `beta1` | 0.85 | 0.9 | Less gradient smoothing; adapts faster to new gradient signal |
| `beta2` | 0.99 | 0.999 | Slightly faster variance adaptation |
| `eps_adam` | 1e-8 | 1e-8 | Same |
| `learning_rate` | 0.01 | 0.001–0.003 | Higher than typical; compensated by the decay below |

---

## 7.5 Learning rate decay

```python
lr_t = learning_rate * (1 - step / num_steps)
```

This is **linear learning rate decay**. At step 0, `lr_t = 0.01`. At step 999 (the last step), `lr_t = 0.01 * (1 - 999/1000) = 0.00001`.

The intuition: early in training, parameters are far from a good solution and large steps help cover ground quickly. Later, parameters are near a local minimum and large steps would cause them to overshoot and oscillate. Decaying the learning rate lets the optimizer converge tightly at the end.

A graph of `lr_t` over 1 000 steps:

```
lr_t
0.010 |*
      | \
0.005 |   \
      |     \
0.000 |       *----
      +------------> step
      0         1000
```

Note that this differs from the popular **cosine annealing** schedule used in larger models (which follows a smooth cosine curve), but for 1 000 steps on a tiny model the linear schedule is sufficient.

---

## 7.6 Code walkthrough with source links

The complete training section of `microgpt.py` with section labels:

```python
# ── 1. Adam buffers  (lines 146–149) ────────────────────────────────────────
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)   # first moment (mean)  — one float per parameter
v = [0.0] * len(params)   # second moment (var)  — one float per parameter

# ── 2. Training loop  (lines 151–184) ───────────────────────────────────────
num_steps = 1000
for step in range(num_steps):

    # ── 2a. Document selection and tokenisation ──────────────────────────────
    doc = docs[step % len(docs)]          # cycle through shuffled dataset
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)  # positions to process

    # ── 2b. Forward pass: build the computation graph ────────────────────────
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)  # grows the graph
        probs = softmax(logits)                        # 28 probabilities
        loss_t = -probs[target_id].log()               # cross-entropy
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)           # mean cross-entropy over sequence

    # ── 2c. Backward pass ────────────────────────────────────────────────────
    loss.backward()   # sets .grad on every Value node in the graph

    # ── 2d. Adam parameter update ────────────────────────────────────────────
    lr_t = learning_rate * (1 - step / num_steps)   # decayed learning rate
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad           # update mean
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2      # update var
        m_hat = m[i] / (1 - beta1 ** (step + 1))             # bias-correct mean
        v_hat = v[i] / (1 - beta2 ** (step + 1))             # bias-correct var
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)   # update weight
        p.grad = 0                                            # zero gradient

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')
```

---

## 7.7 Hands-on steps

### Step 1 — verify the baseline loss

Before training, confirm that a random model produces loss ≈ 3.3. Add a print immediately after the first `loss = ...` calculation inside the loop, guarded so it only fires once:

```python
loss = (1 / n) * sum(losses)

# add this:
if step == 0:
    print(f"\nInitial loss: {loss.data:.4f}  (expected ≈ {math.log(vocab_size):.4f})")
```

Run the script. You should see something like:

```
Initial loss: 3.3142  (expected ≈ 3.3322)
```

The slight difference from the theoretical value is because the model is not perfectly uniform — the small random weights in `lm_head` create mild preferences — but the value should be within 0.1 of 3.33.

### Step 2 — track and print loss every 100 steps

The default `end='\r'` print overwrites itself. To keep a history, add a conditional print after the existing one:

```python
print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# add this:
if (step + 1) % 100 == 0:
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")
```

A healthy run should show a curve roughly like:

```
step  100 / 1000 | loss 2.6183
step  200 / 1000 | loss 2.3705
step  300 / 1000 | loss 2.2014
...
step 1000 / 1000 | loss 1.8xxx
```

Loss decreasing from ~3.3 toward ~1.8–2.2 in 1 000 steps confirms the optimizer is working. If loss stays flat or increases, check that `p.grad = 0` is present and that `loss.backward()` is called before the Adam update.

### Step 3 — observe gradient accumulation bugs

Make a temporary edit: comment out `p.grad = 0`:

```python
        # p.grad = 0    <-- temporarily disabled
```

Run for 20 steps and observe the loss. You will likely see it increase rapidly or produce `nan`. Restore `p.grad = 0` and confirm the run is healthy again.

This exercise makes the gradient zeroing requirement concrete rather than theoretical.

### Step 4 — inspect Adam buffers mid-training

After step 50 or so, print a sample of the first-moment values to see that they are nonzero and vary across parameters:

```python
if step == 49:
    print("\nSample m values:", [f"{mi:.6f}" for mi in m[:8]])
    print("Sample v values:", [f"{vi:.8f}" for vi in v[:8]])
```

You should see small but distinct floats, confirming that Adam has built up gradient history per parameter.

---

## What's happening — summary callouts

> **Cross-entropy loss** is not an arbitrary choice. It is the maximum-likelihood objective for a categorical distribution: minimising it is equivalent to making the model's predicted distribution as close as possible to the true (one-hot) distribution of the next token.

> **The computation graph is discarded implicitly.** After `loss.backward()`, the Python garbage collector is free to reclaim the tens of thousands of `Value` objects created during the forward pass — they are no longer referenced. A fresh graph is built on the next step.

> **Adam does not care about the raw gradient magnitude.** It divides by the root-mean-square of past gradients, so what reaches `p.data` is a *normalised* signal. This is why the effective learning rate per parameter can differ by orders of magnitude even though `lr_t` is the same for all.

---

## Check your understanding

1. A model is trained on a vocabulary of 100 tokens and its initial loss is 5.1. Is this consistent with a random initialisation? (Hint: what is log(100)?)

2. Suppose you double the document length from 8 characters to 16 characters but keep everything else the same. How does the `loss` value change, and why? How does the magnitude of the gradient update change compared to a length-8 document?

3. Explain in one sentence why `m_hat = m[i] / (1 - beta1 ** (step + 1))` gives a larger correction at step 0 than at step 500.

<details>
<summary>Answers</summary>

1. Yes. For a uniform distribution over 100 tokens, the expected cross-entropy loss is −log(1/100) = log(100) ≈ 4.605. An initial loss of 5.1 is slightly above this, which is plausible if the random weights create a slightly non-uniform distribution that happens to down-weight the correct tokens more than average. A value very close to 4.605 is what you would expect on average.

2. Both a length-8 and a length-16 document produce the same *average* (mean) cross-entropy loss `(1/n) * sum(losses)`, because the division by `n` normalises for length. The gradient magnitude per step is therefore roughly similar for both lengths. (The graph is larger for the longer document, so the absolute sum of gradient contributions is higher, but after dividing by `n` the mean loss and its gradient are comparable.)

3. At step 0, `beta1^1 = 0.85`, so `1 - 0.85 = 0.15` — a small denominator, meaning `m_hat` is scaled up significantly to correct for the fact that `m[i]` has only seen one gradient sample. At step 500, `beta1^501 ≈ 6.6e-34 ≈ 0`, so the denominator is essentially 1 and `m_hat ≈ m[i]` — no correction needed because the moving average has had hundreds of steps to warm up.

</details>

---

## What's next

With training complete, the model's parameters contain everything it learned about plausible character sequences in human names. Chapter 8 covers **inference and sampling**: how to feed the trained model a BOS token, sample from its output distribution using temperature scaling, and generate names one character at a time until another BOS token signals the end of a sequence.
