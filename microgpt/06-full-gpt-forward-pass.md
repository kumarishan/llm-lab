# Chapter 6: The Full GPT Forward Pass

## What you'll learn

- How the MLP block works and why it uses a 4x expansion
- What the residual stream is and why every block adds to it rather than replacing it
- Why `rmsnorm` appears before every sub-block (pre-norm architecture)
- The complete data flow from a single integer token ID to 28 raw logit scores
- How `n_layer` controls depth and what happens when you increase it

---

## Prerequisites

- Chapters 1–5 of this tutorial series
- Chapter 5 in particular: multi-head attention, the KV cache, and how `keys`/`values` accumulate across tokens
- Familiarity with `Value`, `rmsnorm`, `linear`, and `softmax` from earlier chapters

---

## Full data-flow diagram

Before diving into code, here is the complete picture of one `gpt()` call — processing a single token at one position:

```
token_id ──► wte[token_id] ───┐
                               ├──► element-wise add ──► rmsnorm ──► x (residual stream)
pos_id   ──► wpe[pos_id]   ───┘                                           │
                                                                           │
                             ┌─────────────────────────────────────────────┘
                             │
                             ▼           [Transformer Block 0]
                   ┌─────────────────────────────────────────────────────────────┐
                   │                                                             │
                   │  x ──► rmsnorm ──► Q,K,V projections ──► Multi-head       │
                   │  │                  Attention ──► attn_wo ──► + x ──► x'  │
                   │  └──────────────────────────────────────────────────────┘  │
                   │                                                             │
                   │  x' ──► rmsnorm ──► fc1 (16→64) ──► ReLU ──► fc2 (64→16) │
                   │  │                                              + x' ──► x │
                   │  └─────────────────────────────────────────────────────┘   │
                   └─────────────────────────────────────────────────────────────┘
                             │
                             ▼
                       lm_head (16 → 28)
                             │
                             ▼
                     logits  [28 raw scores]
```

With `n_layer=1` the loop runs exactly once. If you set `n_layer=2`, the output of Block 0 feeds into Block 1 before hitting `lm_head`.

---

## The complete `gpt()` function

Source: [`microgpt.py` lines 108–144](../microgpt.py#L108-L144)

```python
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits
```

We will walk through every line in order.

---

## Step 1: Embedding lookup and initial normalization

```python
tok_emb = state_dict['wte'][token_id]   # list of 16 Value objects
pos_emb = state_dict['wpe'][pos_id]     # list of 16 Value objects
x = [t + p for t, p in zip(tok_emb, pos_emb)]
x = rmsnorm(x)
```

`wte` (word token embeddings) maps the integer `token_id` to a 16-dimensional learned vector. `wpe` (word position embeddings) maps `pos_id` to another 16-dimensional vector. Adding them together fuses "what token" with "where in the sequence".

The immediate `rmsnorm` may look redundant — we just looked up learnable weights, why normalize them? The comment in the source explains it:

> `# note: not redundant due to backward pass via the residual connection`

During the backward pass, gradients flow back through the residual connections and accumulate. Without this normalization, the scale of `x` entering the first block would be inconsistent with the scale of every subsequent residual addition. Normalizing here once gives the whole network a stable starting scale.

After this step `x` is a list of 16 `Value` objects. This is the **residual stream** — it will travel through every block, collecting updates along the way.

---

## Step 2: The transformer block loop

```python
for li in range(n_layer):
```

With `n_layer=1` this runs exactly once. The loop index `li` is used to index into `state_dict` using f-strings like `f'layer{li}.attn_wq'`, so each layer has its own independent set of weights.

Inside the loop there are two sub-blocks, always in order: attention first, then MLP.

### Sub-block 1: Multi-head attention (covered in Chapter 5)

```python
x_residual = x          # snapshot x before modification
x = rmsnorm(x)          # pre-norm
# ... Q, K, V projections, attention, output projection ...
x = [a + b for a, b in zip(x, x_residual)]  # residual add
```

The attention block is the "communication" step — it lets the current token look at and aggregate information from all past tokens via the KV cache. Chapter 5 covers this in depth. Note the pattern: save `x` as `x_residual`, compute something, add `x_residual` back. This is the residual connection.

---

## Step 3: The MLP block

Source: [`microgpt.py` lines 136–141](../microgpt.py#L136-L141)

```python
x_residual = x
x = rmsnorm(x)
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
x = [xi.relu() for xi in x]
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
x = [a + b for a, b in zip(x, x_residual)]
```

The MLP block is the "computation" or "memory" step. It processes each token's representation independently — no cross-token communication happens here. Here is what each line does:

### Pre-norm

```python
x_residual = x
x = rmsnorm(x)
```

Same pattern as the attention block. Save the current stream as the residual, then normalize before computing. Normalizing *before* each sub-block (rather than after) is called **pre-norm** and is what GPT-2 uses. It improves gradient flow during training.

### fc1: expand 16 → 64

```python
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
```

`mlp_fc1` has shape `(4*n_embd, n_embd)` = `(64, 16)`. The `linear` function computes a dot product of `x` (length 16) with each of the 64 rows, producing a length-64 output. The dimension expands from 16 to 64 — a **4x bottleneck expansion**. This is the standard transformer ratio going back to the original "Attention Is All You Need" paper.

Why 4x? The expanded hidden layer gives the network additional capacity. Research and practice have found that this intermediate width is where transformers store learned factual associations. A model with only n_embd-wide MLPs would be significantly underpowered relative to its attention mechanism.

### ReLU activation

```python
x = [xi.relu() for xi in x]
```

This applies ReLU element-wise across all 64 values. The comment at the top of `microgpt.py` notes "GeLU -> ReLU" — GPT-2 uses GeLU (Gaussian Error Linear Unit), but this implementation uses the simpler ReLU to keep the `Value` autograd engine minimal. Both serve the same purpose: introduce non-linearity so the MLP is not just a linear transformation.

Without an activation function, `fc1` followed by `fc2` would collapse into a single linear map, eliminating the benefit of the intermediate expansion.

> **What's happening:** ReLU outputs zero for any negative input and passes positive inputs unchanged. After fc1 projects into 64 dimensions, ReLU selectively "zeroes out" dimensions the network has learned to suppress, keeping only the activated features for fc2 to work with.

### fc2: contract 64 → 16

```python
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
```

`mlp_fc2` has shape `(n_embd, 4*n_embd)` = `(16, 64)`. This projects back down to 16 dimensions, matching the residual stream width. The expand-activate-contract structure is sometimes called a "bottleneck" block.

### Residual add

```python
x = [a + b for a, b in zip(x, x_residual)]
```

The MLP's output (16 values) is added back to `x_residual` (the stream as it entered the MLP block). The MLP does not replace the stream — it contributes a delta to it.

---

## The residual stream architecture

The variable `x` flows through the entire function and is never replaced wholesale — only modified by addition. This design pattern is called the **residual stream** view of transformers.

```
x (16 values)
  │
  ├── rmsnorm → attention → attn_wo → [+] ──► x updated
  │                                    │
  │                               x_residual
  │
  ├── rmsnorm → fc1 → ReLU → fc2 → [+] ──► x updated
  │                                   │
  │                              x_residual
  │
  └── lm_head → logits
```

Each block reads from the stream and writes a correction back via addition. Because gradients flow through addition unattenuated (the local gradient of `a + b` with respect to both `a` and `b` is 1), gradients from the loss can travel all the way back to the embedding lookup without vanishing. This is why deep networks (many layers) became trainable once residual connections were introduced.

In this model the residual stream is only 16 values wide. In GPT-2 (small) it is 768 wide, and in larger models it can be 12,288 wide. The stream width is `n_embd`.

---

## Step 4: The language model head

```python
logits = linear(x, state_dict['lm_head'])
return logits
```

Source: [`microgpt.py` lines 143–144](../microgpt.py#L143-L144)

After the transformer block(s), `x` is still a 16-dimensional vector. The `lm_head` weight matrix has shape `(vocab_size, n_embd)` = `(28, 16)`. Applying `linear` gives a list of 28 raw scores — one per token in the vocabulary.

These are **logits**: unnormalized log-probabilities. A higher logit for token `i` means the model predicts token `i` is more likely to follow the current sequence. The caller converts them to probabilities via `softmax`:

```python
# in the training loop (line 166):
probs = softmax(logits)
loss_t = -probs[target_id].log()

# in inference (line 195):
probs = softmax([l / temperature for l in logits])
token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

> **Note on weight tying:** GPT-2 ties `lm_head` weights to `wte` (the token embedding matrix is reused as the output projection). `microgpt.py` keeps them as independent matrices. Weight tying reduces parameter count at the cost of an architectural constraint.

---

## End-to-end data flow summary

| Stage | Operation | Input shape | Output shape |
|---|---|---|---|
| Token lookup | `wte[token_id]` | scalar | 16 |
| Position lookup | `wpe[pos_id]` | scalar | 16 |
| Add + rmsnorm | element-wise | 16 + 16 | 16 |
| Attention pre-norm | rmsnorm | 16 | 16 |
| Q, K, V projections | linear (×3) | 16 | 16 each |
| Multi-head attention | dot products + softmax | 16 | 16 |
| attn_wo | linear | 16 | 16 |
| Attention residual add | element-wise | 16 + 16 | 16 |
| MLP pre-norm | rmsnorm | 16 | 16 |
| fc1 | linear | 16 | 64 |
| ReLU | element-wise | 64 | 64 |
| fc2 | linear | 64 | 16 |
| MLP residual add | element-wise | 16 + 16 | 16 |
| lm_head | linear | 16 | 28 |

This table describes one pass through a single transformer block (`n_layer=1`). For `n_layer=2`, the "Attention pre-norm" through "MLP residual add" rows repeat a second time before `lm_head`.

---

## Hands-on steps

### Step 1: Instrument `gpt()` to observe the stream

Add print statements after each major stage to see how `x` changes numerically. Open `microgpt.py` and modify `gpt()` temporarily:

```python
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    print(f"[after initial rmsnorm] x[:4] = {[round(xi.data, 4) for xi in x[:4]]}")

    for li in range(n_layer):
        # attention block (unchanged) ...
        x_residual = x
        x = rmsnorm(x)
        # ... (keep full attention code) ...
        x = [a + b for a, b in zip(x, x_residual)]
        print(f"[after attn residual, layer {li}] x[:4] = {[round(xi.data, 4) for xi in x[:4]]}")

        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
        print(f"[after mlp  residual, layer {li}] x[:4] = {[round(xi.data, 4) for xi in x[:4]]}")

    logits = linear(x, state_dict['lm_head'])
    print(f"[logits] top-5 indices = {sorted(range(len(logits)), key=lambda i: logits[i].data, reverse=True)[:5]}")
    return logits
```

Run the script for a few steps (reduce `num_steps` to 2 for speed). Observe:
- The scale of `x` stays bounded through rmsnorm
- The attention and MLP residual adds shift the values but do not explode them
- The logits are raw and can be any real number

### Step 2: Count `Value` operations in a single `gpt()` call

Every arithmetic operation on `Value` objects creates a new node in the computation graph. Try counting the total number of `Value` objects created per `gpt()` call. Before calling `gpt()`, you can monkey-patch `Value.__init__` to count:

```python
_value_count = 0
_orig_value_init = Value.__init__

def _counting_init(self, data, children=(), local_grads=()):
    global _value_count
    _value_count += 1
    _orig_value_init(self, data, children, local_grads)

Value.__init__ = _counting_init

# Reset and call gpt once
_value_count = 0
keys_tmp = [[] for _ in range(n_layer)]
values_tmp = [[] for _ in range(n_layer)]
_ = gpt(0, 0, keys_tmp, values_tmp)
print(f"Value nodes created in one gpt() call: {_value_count}")

Value.__init__ = _orig_value_init  # restore
```

Consider: why does this number grow with longer sequences? (Hint: the KV cache grows with `pos_id`, so attention creates more `Value` nodes per token as context grows.)

### Step 3: Increase `n_layer` and observe parameter count

Near the top of `microgpt.py`, change:

```python
n_layer = 1
```

to:

```python
n_layer = 2
```

Run the script. The `num params` line printed at startup will increase. Work out the math yourself:

- Each layer adds: `attn_wq + attn_wk + attn_wv + attn_wo + mlp_fc1 + mlp_fc2`
- In dimensions: `(16×16) + (16×16) + (16×16) + (16×16) + (64×16) + (16×64)`
- That is: `256 + 256 + 256 + 256 + 1024 + 1024 = 3072` parameters per layer

Verify this against the printed parameter counts for `n_layer=1` vs `n_layer=2`.

> **What's happening:** The `for li in range(n_layer)` loop in `gpt()` already handles multiple layers — no other code change is needed. The `state_dict` initialization loop at lines 82–88 creates the correct number of weight matrices for whatever `n_layer` is set to.

---

## Check your understanding

1. The MLP block uses `fc1` (16 → 64) followed by `fc2` (64 → 16). If you removed the ReLU activation between them, what would the combined effect of `fc1` and `fc2` be? Would the 4x expansion still provide extra capacity?

2. In the residual add `x = [a + b for a, b in zip(x, x_residual)]`, `x` is the MLP's output and `x_residual` is the stream before the MLP ran. If the MLP's weights were all zero (producing all-zero output), what would the result be? What does this imply about the network's behavior early in training?

3. The file comment says "layernorm -> rmsnorm". LayerNorm subtracts the mean and divides by the standard deviation, then applies a learned scale and bias. RMSNorm only divides by the root mean square and applies no bias. What does RMSNorm simplify compared to LayerNorm, and what does `microgpt.py`'s `rmsnorm` skip that even standard RMSNorm includes?

4. With `n_layer=1`, the `for li in range(n_layer)` loop body runs once. With `n_layer=3`, how many times is `rmsnorm` called inside `gpt()` (include the initial call before the loop)?

5. `lm_head` maps 16 dimensions to `vocab_size=28`. Why must `lm_head`'s output dimension equal `vocab_size` exactly, rather than some other number?

---

## What's next

You now have a complete picture of the forward pass: a token ID enters, passes through embeddings, one or more transformer blocks (each with attention + MLP, both wrapped in residual connections and pre-norm), and emerges as 28 logit scores.

But the model starts with random weights and knows nothing. Chapter 7 covers **training**: how the cross-entropy loss is computed from those logits, how `loss.backward()` walks the computation graph to assign blame to every `Value` node, and how the Adam optimizer uses those gradients to nudge the weights toward producing correct predictions. By the end of Chapter 7 you will understand the full train loop — the outer `for step in range(num_steps)` block — and why the loss decreases over time.
