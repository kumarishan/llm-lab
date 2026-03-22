# Chapter 8: Inference and Sampling

## What you'll learn

- How autoregressive generation works: one token at a time, feeding the model's own output back in
- What temperature scaling does to a probability distribution and why it matters
- How `random.choices()` samples from the model's learned distribution using plain floats
- Why the KV cache resets between names but accumulates within each name
- Why `Value.backward()` is never called during inference, and what that costs
- How training (teacher forcing) and inference (autoregressive) differ structurally
- What "hallucinated names" means and why the model generalizes rather than memorizes

---

## Prerequisites

- Completed Chapters 1–7
- Comfortable with all of `microgpt.py`: the `Value` class, the `gpt()` function, the training loop, and the Adam optimizer
- The ideas of softmax, cross-entropy, and the KV cache from Chapters 5–7

---

## The inference loop — source and context

Source: [`microgpt.py` lines 186–200](../microgpt.py)

```python
temperature = 0.5  # in (0, 1], control the "creativity"
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

These 15 lines are the payoff for everything that came before. The model has spent 1000 training steps adjusting 21,532 parameters to assign high probability to likely next characters. Now you ask it: given nothing but a start signal, what name would you invent?

---

## Autoregressive generation

"Autoregressive" means the model uses its own previous output as the next input. Each step produces one token; that token feeds directly back in as the input for the following step. There is no external script deciding what comes next — the model drives itself forward.

### The loop structure

```
outer loop: repeat for each name to generate (20 names)
  |
  reset KV cache and sample list
  seed: token_id = BOS
  |
  inner loop: up to block_size=16 steps
    |
    call gpt(token_id, pos_id, keys, values)  →  logits (28 numbers)
    apply temperature scaling to logits
    softmax  →  probs (28 numbers, sum to 1)
    sample one token_id from probs
    |
    if token_id == BOS:  stop (model says "name is complete")
    else:                append character, continue
  |
  print the completed name
```

### Why BOS serves as both start and stop

`BOS` is token id `27` (the last index, one past all the character tokens). At the start of generation, you give the model `BOS` as the very first input at position 0. The model learned during training that after `BOS` comes the first letter of a name — it learned this because every training document was wrapped as `[BOS, c1, c2, ..., cn, BOS]`.

The second `BOS` in that training sequence was the target that the model had to predict at the final position. After 1000 training steps the model has learned: when the name is finished, the most likely next token is `BOS`. The inference loop checks for exactly this:

```python
if token_id == BOS:
    break
```

So `BOS` is reused: it is the start signal going in and the end signal coming out. No separate `EOS` (end-of-sequence) token is needed.

### The safety cap

```python
for pos_id in range(block_size):
```

`block_size = 16`. The longest name in the training dataset is 15 characters, so a correctly trained model should learn to stop before position 16. The `range(block_size)` loop is a hard safety cap: if for some reason the model never predicts `BOS`, generation stops anyway. Without it, a degenerate model that always predicts a letter could loop forever.

### Contrast with training

During training (Chapter 7) the loop looks like this:

```python
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
for pos_id in range(n):
    token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
    logits = gpt(token_id, pos_id, keys, values)
    probs = softmax(logits)
    loss_t = -probs[target_id].log()
```

The key difference: during training, `token_id` at each step comes from the **original document** (`tokens[pos_id]`). The model is shown the correct previous character regardless of what it would have predicted. This technique is called **teacher forcing** — you force the model to attend to the ground truth at every step.

During inference there is no ground truth. The model's own prediction from step `t` becomes the input at step `t+1`. If the model makes a mistake early in the name, that mistake propagates forward. This is why inference and training are structurally different even though the same `gpt()` function runs in both.

```
Training (teacher forcing):
  Input sequence:  [BOS,  'e',  'm',  'i',  'l',  'y']
  Token source:     given  given given given given given
  All correct, regardless of model's predictions

Inference (autoregressive):
  Step 0 input: BOS  →  predict 'e'
  Step 1 input: 'e'  →  predict 'm'    (using model's own output)
  Step 2 input: 'm'  →  predict 'i'    (using model's own output)
  ...and so on
```

> **What's happening:** Teacher forcing speeds up training by preventing early mistakes from corrupting all subsequent positions in the same document. Autoregressive generation is the only option at inference time, because you have no ground truth to force.

---

## Temperature scaling

### The mechanism

Before applying softmax, the inference loop divides every logit by the temperature:

```python
probs = softmax([l / temperature for l in logits])
```

Temperature is a single positive scalar, `0.5` here. Dividing logits by a number less than 1 is equivalent to multiplying them — it makes the numbers larger in magnitude. The softmax then turns those scaled logits into probabilities.

To understand the effect, consider a simplified case with three tokens whose raw logits are `[2.0, 1.0, 0.0]`.

```
Raw logits:               [2.0,  1.0,  0.0]
After softmax (T=1.0):    [0.67, 0.24, 0.09]   ← original distribution

After dividing by T=0.5:  [4.0,  2.0,  0.0]
After softmax (T=0.5):    [0.88, 0.12, 0.00]   ← peaks sharpen

After dividing by T=2.0:  [1.0,  0.5,  0.0]
After softmax (T=2.0):    [0.51, 0.31, 0.18]   ← distribution flattens

After dividing by T=0.1:  [20.0, 10.0, 0.0]
After softmax (T=0.1):    [1.00, 0.00, 0.00]   ← approaches argmax
```

The pattern is consistent:

| Temperature | Effect on distribution | Character of output |
|-------------|------------------------|---------------------|
| Near 0 (e.g. 0.1) | Collapses to argmax — the top token gets nearly all the probability | Repetitive, always picks the most common continuation |
| Less than 1 (e.g. 0.5) | Sharpens peaks — high-probability tokens get more weight | Conservative, name-like, somewhat predictable |
| Exactly 1.0 | No change — original learned distribution | Exactly what the model learned |
| Greater than 1 (e.g. 2.0) | Flattens distribution — tokens become more equally likely | More varied, occasionally surprising or strange |
| Very large | Approaches uniform — any token is equally likely | Essentially random noise |

The script uses `temperature = 0.5`, which makes the model more conservative than its raw learned distribution. The generated names look more conventionally English as a result.

### Why not always use temperature 1.0?

At temperature 1.0 the model samples from exactly what it learned, which is reasonable. But language models trained on small datasets can overfit in subtle ways — the distribution they learn has the right rough shape but noisy estimates in the tails. A slight downward temperature tightens the distribution and reduces the chance of sampling a low-probability (often incorrect or strange-looking) token.

Temperature is a dial, not a correct answer. The right value depends on your use case: name generation benefits from conservatism; creative writing benefits from more variance.

> **What's happening:** Temperature does not change the ranking of tokens — the highest-logit token is still the most likely at any temperature. It only changes how concentrated the probability mass is around that top token. Low temperature makes the model more decisive; high temperature makes it more adventurous.

---

## Sampling with `random.choices()`

```python
token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

`random.choices(population, weights)` draws one element from `population` with probability proportional to the corresponding weight. Here:

- `population` is `range(28)` — the integers `0, 1, 2, ..., 27`, one per token
- `weights` is the list of softmax probabilities for each token

The function returns a list (even when drawing one element), so `[0]` extracts the single result.

### Why `p.data` and not `p`

`probs` is a list of `Value` objects — every number in the model's computation is a `Value`. `random.choices` expects plain Python floats as weights. Passing `Value` objects would cause a `TypeError` because Python does not know how to compare or sum them in the way `random.choices` requires internally.

`p.data` extracts the underlying float. This is a boundary point: you are leaving the computation graph behind and returning to plain Python land. From this point on, the sampled `token_id` is an ordinary Python `int` — no gradient tracking, no `Value` wrapping.

### Weighted sampling is not argmax

Calling `argmax` would always pick the most probable token. The names would be deterministic and identical for every run (with the same weights). Sampling from the distribution introduces variety: sometimes the model picks the second or third most likely character, which leads to different — and often still plausible — names.

This is the same distinction as the difference between a language model that always outputs the most likely response (deterministic but repetitive) and one that samples from its distribution (varied and sometimes surprising).

---

## The KV cache in inference

```python
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
```

This line appears at the top of the outer loop — once per name being generated. It resets the KV cache to empty lists before generation begins. Within the inner loop, every call to `gpt()` appends new key and value vectors to these lists:

```python
# inside gpt(), lines 121-122:
keys[li].append(k)
values[li].append(v)
```

So as the inner loop advances position by position, the cache grows:

```
Step 0 (pos_id=0, token=BOS):      keys[0] has 1 entry
Step 1 (pos_id=1, token='e'):      keys[0] has 2 entries
Step 2 (pos_id=2, token='m'):      keys[0] has 3 entries
...
Step n (pos_id=n):                  keys[0] has n+1 entries
```

At each step, the attention mechanism in `gpt()` computes attention scores from the current query against all accumulated keys. This is why the model can attend to every character it has generated so far, not just the immediately previous one.

This is precisely the KV cache described in Chapter 5 — the inference code uses the exact same mechanism as training. No special inference mode is needed. The only structural difference is that the cache is reset between names (outer loop) but accumulated within a single name (inner loop).

> **What's happening:** The KV cache is how the transformer "remembers" the characters it has already generated. Without it, each step would only see the single current token with no context of what came before. With it, the model at step 5 has access to the keys and values computed at steps 0, 1, 2, 3, and 4.

---

## No gradients during inference

During training, after the forward pass, the code calls:

```python
loss.backward()
```

This traverses the entire computation graph, fills in `.grad` for every `Value` node, and then Adam uses those gradients to update the parameters.

During inference, `backward()` is never called. The model runs the forward pass, samples a token, and discards the computation graph. Gradients serve no purpose here: you are not learning anything, only making predictions.

However, the `Value` objects are still created. Every addition, multiplication, and softmax in `gpt()` still builds `Value` nodes with `_children` and `_local_grads`. The computation graph is constructed and then abandoned. This is wasteful — Python allocates and then garbage-collects thousands of `Value` objects for each generated character.

In a production system, inference would use plain Python floats (or NumPy arrays, or PyTorch tensors with `torch.no_grad()`). The extra overhead in `microgpt.py` is the cost of sharing one code path between training and inference. For a 200-line teaching file, the tradeoff is correct: simplicity over efficiency.

---

## Putting it all together — one name, step by step

Let's trace the generation of a single name to make every moving part concrete. Assume the model has been trained for 1000 steps and the weights reflect what the model has learned.

**Setup:** `block_size = 16`, `vocab_size = 28`, `n_layer = 1`, `temperature = 0.5`

```
keys   = [[]]   # one empty list per layer — cache is empty
values = [[]]
token_id = 27   # BOS
sample   = []
```

---

**Step 0 — pos_id = 0:**

```
Input:    token_id = 27 (BOS)
          pos_id   = 0

gpt(27, 0, keys, values)
  → looks up token embedding wte[27]  (the BOS embedding)
  → looks up position embedding wpe[0]
  → runs through attention and MLP
  → keys[0] now has 1 entry (key vector for position 0)
  → returns logits: 28 raw scores

logits / 0.5  →  each logit doubled
softmax       →  probs over 28 tokens
              →  after training, tokens like 'a','e','m','s' have high mass here
                 (names commonly start with these letters)

random.choices samples: token_id = 4  (say, 'e')
token_id != BOS → continue
sample = ['e']
```

---

**Step 1 — pos_id = 1:**

```
Input:    token_id = 4  ('e')
          pos_id   = 1

gpt(4, 1, keys, values)
  → looks up wte[4]  (embedding for 'e')
  → looks up wpe[1]  (embedding for position 1)
  → attention: query from position 1 attends to keys at positions 0 and 1
  → keys[0] now has 2 entries
  → returns logits over 28 tokens

Given the context [BOS, 'e'] the model assigns high probability to
letters that commonly follow 'e' at the start of a name: 'm','l','v','r'

random.choices samples: token_id = 12 (say, 'm')
sample = ['e', 'm']
```

---

**Steps 2, 3, 4, ... continue similarly.**

Each step adds one character to `sample`, appends one new key/value pair to the cache, and the attention mechanism incorporates all of the context seen so far.

---

**Final step — say pos_id = 5:**

```
Input:    token_id = 8  ('a')  — from the previous step
sample at this point = ['e', 'm', 'i', 'l', 'i']

gpt(8, 5, keys, values)
  → attention attends to all 6 positions (0 through 5)
  → logits: BOS (token 27) has relatively high score
             because names of this pattern commonly end here

random.choices samples: token_id = 27 (BOS)
token_id == BOS → break

sample = ['e', 'm', 'i', 'l', 'i']
print: "sample  1: emili"
```

The name is not in the training dataset — it is a novel combination of letter patterns the model has internalized. It looks plausible because the model learned, over 1000 training steps, which character transitions are common in English names.

---

## Hands-on steps

Work through these in order. Each step takes only a few minutes.

### Step 1: Run the script as-is

```bash
python microgpt.py
```

Note the 20 generated names. They are deterministic with `random.seed(42)`.

---

### Step 2: Observe temperature = 1.0

Edit line 187 of `microgpt.py`:

```python
temperature = 1.0
```

Re-run. The model now samples from its raw learned distribution. Names should become slightly more varied — you may see some unusual letter combinations that temperature 0.5 would have suppressed.

---

### Step 3: Observe temperature = 0.1

```python
temperature = 0.1
```

Re-run. With very low temperature the distribution collapses toward argmax. You should see more repetition: several samples may start with the same letters, and some names may be identical or near-identical. The model is now almost always picking its single most-probable next token.

---

### Step 4: Observe temperature = 2.0

```python
temperature = 2.0
```

Re-run. The distribution flattens. You will likely see outputs that are harder to pronounce and less name-like — the model is sampling from its lower-probability predictions more often.

---

### Step 5: Measure memorization vs. generalization

Restore `temperature = 0.5` and increase the sample count from 20 to 100:

```python
for sample_idx in range(100):
```

After running, compare the 100 generated names against `input.txt`:

```python
# Run this separately after collecting output
names = open('input.txt').read().splitlines()
names_set = set(n.strip().lower() for n in names)

generated = [
    "emilia", "kael", "brionna",  # paste your actual output here
    # ...
]

memorized = [g for g in generated if g.lower() in names_set]
print(f"{len(memorized)} / {len(generated)} names appeared in the training data")
```

A well-trained model with 1000 steps on 32,033 names should generalize significantly: most generated names will not appear verbatim in `input.txt`. The model has not memorized names — it has internalized letter-transition patterns.

---

### Step 6: Watch the KV cache grow

Add a print statement inside the inference loop to watch the cache grow:

```python
for pos_id in range(block_size):
    logits = gpt(token_id, pos_id, keys, values)
    print(f"  pos {pos_id}: cache length = {len(keys[0])}")
    probs = softmax([l / temperature for l in logits])
    token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
    if token_id == BOS:
        break
    sample.append(uchars[token_id])
```

You will see the cache length increment by 1 at each step, confirming that `gpt()` appends to `keys[0]` on every call. Remove the print statement when you are done.

---

## Check your understanding

**1.** The inference inner loop runs `for pos_id in range(block_size)` but the loop often terminates early via `break`. What are the two conditions that cause the loop to end, and what does each one mean semantically about the name being generated?

**2.** If you set `temperature = 0.001` (extremely low), the distribution would be so sharp that effectively only the highest-logit token would be sampled. What would happen to the 20 generated names in that case — would they all be different, or mostly the same? Why?

**3.** During inference, `p.data` is used in `weights=[p.data for p in probs]` to extract plain floats from `Value` objects. Why can't you pass the `Value` objects themselves to `random.choices`? What would happen if you tried?

---

## Congratulations

You have now read, understood, and experimented with every line of `microgpt.py`.

Here is what you have built up across these eight chapters:

| Chapter | What you learned |
|---------|-----------------|
| 1 | How to run the file, what the output means, the five-stage structure |
| 2 | Character-level tokenization, the `BOS` token, input/target pairs |
| 3 | The `Value` class, computation graphs, backpropagation, the chain rule |
| 4 | Weight matrix initialization, `state_dict`, `linear`, `rmsnorm`, `softmax` |
| 5 | Multi-head self-attention, queries/keys/values, the KV cache |
| 6 | The full GPT forward pass: embeddings, residual connections, the MLP block |
| 7 | Cross-entropy loss, teacher forcing, the Adam optimizer, learning rate decay |
| 8 | Autoregressive generation, temperature scaling, sampling, inference vs. training |

The complete system — from raw character data to generated names — runs in 200 lines of pure Python. No NumPy, no PyTorch, no GPU. Every floating-point number that passes through the model is a `Value` object you understand completely.

### What makes this remarkable

- **21,532 parameters** — smaller than many images
- **1000 training steps** — a few seconds on a GPU, a few minutes on a CPU
- **Pure Python** — every operation is a plain Python function call
- **No external libraries** — `os`, `math`, `random`, and `urllib.request` from the standard library
- **Generates plausible English names** — the model has learned character-level statistics well enough to produce names that sound real

This is not a toy that works by magic. It is the same transformer architecture that powers GPT-2, GPT-3, and their descendants, reduced to its essential skeleton. The ideas — embeddings, attention, residual connections, softmax, cross-entropy, backpropagation, Adam — are identical. The only difference is scale.

### Where to go next

**Stay in this codebase:**
- Increase `n_layer` to 2 or 4 and watch the loss improve (and training slow down)
- Increase `n_embd` to 32 or 64 and observe the change in parameter count
- Train for 5000 steps instead of 1000 and compare name quality
- Change the dataset: replace `input.txt` with any line-per-document text file

**Go deeper on the foundations:**

- [micrograd](https://github.com/karpathy/micrograd) — Andrej Karpathy's minimal autograd engine, the direct ancestor of the `Value` class in `microgpt.py`. The accompanying lecture "The spelled-out intro to neural networks and backpropagation" walks through every line.

- [makemore](https://github.com/karpathy/makemore) — the name-generation model that `microgpt.py` is derived from. The lecture series builds from a bigram model all the way to a transformer, each step in detail.

- [nanoGPT](https://github.com/karpathy/nanoGPT) — a clean, minimal GPT implementation in PyTorch that scales to real language modeling tasks. Once you are comfortable with `microgpt.py`, `nanoGPT` is the natural next step. The architecture is the same; the difference is that NumPy arrays replace scalar `Value` objects, which makes training thousands of times faster.

**Go deeper on the math:**

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) — the original transformer paper. You can now read the architecture description and recognize every component.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) — a line-by-line walkthrough of the paper in PyTorch.

**Go to production:**

- [PyTorch](https://pytorch.org/) — the standard framework for deep learning research and production. The `Value` class you studied is conceptually identical to a PyTorch scalar tensor; `Value.backward()` does the same thing as `tensor.backward()`. PyTorch replaces scalar arithmetic with batched matrix operations, adds CUDA support, and provides an optimized autograd engine — but the ideas are unchanged.

---

The line at the top of `microgpt.py` reads:

> *The most atomic way to train and run inference for a GPT in pure, dependency-free Python. This file is the complete algorithm. Everything else is just efficiency.*

You now know exactly what that means.
