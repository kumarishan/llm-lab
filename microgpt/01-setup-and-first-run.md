# Chapter 1: Setup and First Run

## What you'll learn

- How to run `microgpt.py` with zero dependencies
- What the three startup lines tell you about the dataset and model
- How to read the training output and know things are working
- What the final inference section produces and why it looks the way it does
- A high-level mental map of the entire file from data loading to name generation

### Tutorial road map

| Chapter | Title |
|---------|-------|
| **1** | **Setup and First Run** (this chapter) |
| 2 | Dataset and Tokenization |
| 3 | Autograd from Scratch |
| 4 | Model Architecture Foundations |
| 5 | Multi-Head Self-Attention |
| 6 | The Full GPT Forward Pass |
| 7 | Training — Loss, Backprop, and Adam |
| 8 | Inference and Sampling |

---

## Prerequisites

- Python 3.8 or later installed and available on your `PATH`
- An internet connection for the first run (the script downloads the dataset automatically; subsequent runs use the cached file)
- No third-party packages required — the file uses only `os`, `math`, `random`, and `urllib.request` from the standard library

---

## Overview: what this file actually is

`microgpt.py` is a single, self-contained implementation of a character-level GPT that trains on a list of human names and then generates new, plausible-sounding names. Every concept that makes a modern language model work — tokenization, a transformer with multi-head attention, autograd-based backpropagation, and the Adam optimizer — is present in roughly 200 lines of pure Python with no external dependencies.

The file can be read as five sequential stages:

```
1. Dataset loading        lines  15–21    download + parse names.txt
2. Tokenizer              lines  23–27    map characters ↔ integers
3. Autograd engine        lines  29–72    scalar Value class with .backward()
4. Model + parameters     lines  74–144   GPT architecture definition
5. Train / infer loop     lines  146–200  Adam, 1000 steps, then 20 samples
```

You do not need to understand any of this deeply yet — each stage gets its own chapter. Right now the goal is simply to run it, see what it prints, and build a rough mental model of the whole.

---

## Step 1: Clone or locate the repository

If you are reading this inside the repo, you already have the file. Confirm it is where you expect it:

```bash
ls llm-lab/microgpt.py
```

Expected output:

```
llm-lab/microgpt.py
```

---

## Step 2: Run the script

From the root of the repository, run:

```bash
python microgpt.py
```

> If your system defaults to Python 2, use `python3 microgpt.py`.

The script will:

1. Download `names.txt` from GitHub on the first run and save it as `input.txt` in the current directory. This is roughly 150 KB and takes a second or two.
2. Print three startup lines immediately.
3. Train for 1000 steps, overwriting a single progress line in your terminal.
4. Print 20 generated names.

On a modern laptop this takes **roughly 5–15 minutes** because the autograd engine operates on individual Python floats — there is no NumPy or tensor library doing batch operations. That is intentional: every multiplication you see in the source is one floating-point number, which makes the mechanics maximally transparent.

---

## Step 3: Read the startup output

The first three lines printed are:

```
num docs: 32033
vocab size: 28
num params: 21532
```

Each line corresponds to a specific section of the source file. [view source: microgpt.py](../microgpt.py)

### `num docs: 32033`

```python
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")
```

`input.txt` is a plain text file with one name per line — 32,033 of them. The file strips whitespace and drops blank lines. It then shuffles the list with a fixed seed (`random.seed(42)` appears at the top of the file) so training order is randomized but reproducible.

### `vocab size: 28`

```python
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")
```

All the names together contain 27 unique characters (the 26 lowercase letters plus a handful of others such as apostrophes — the exact set depends on the dataset). One additional special token, `BOS` (Beginning of Sequence), is added to mark the start and end of each name. Total: 28 tokens.

This is a **character-level** tokenizer, meaning the model predicts one character at a time rather than one word or sub-word at a time. Chapter 2 goes into the details.

### `num params: 21532`

```python
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")
```

All the weight matrices in the model are flattened into a single list of `Value` objects. 21,532 is a tiny number by modern standards (GPT-2 small has 117 million parameters), but it is large enough to learn the statistical patterns of English names. Each element in this list is one learnable scalar.

---

## Step 4: Watch the training loop

After the three startup lines, a single line is written and continually overwritten:

```
step  137 / 1000 | loss 2.8401
```

The fields are:

| Field | Meaning |
|-------|---------|
| `step` | Current training step (1-indexed). One step = one document (name). |
| `/ 1000` | Total number of steps. Controlled by `num_steps = 1000` near the bottom of the file. |
| `loss` | Cross-entropy loss for this step. Lower is better; the model should trend downward over time. |

> **What is cross-entropy loss?** At each position in a name, the model predicts a probability distribution over the 28 possible next characters. Cross-entropy measures how surprised the model was by the actual next character — a perfect prediction gives loss 0, a completely wrong uniform guess gives roughly `log(28) ≈ 3.33`. You should see the loss start near 3.3 and gradually decrease toward 2.0–2.5 by step 1000.

The line uses `end='\r'` to overwrite itself:

```python
print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')
```

This means your terminal only ever shows one training line at a time. The loss from the final step is what remains visible when training finishes.

---

## Step 5: Read the generated names

When training finishes the script prints:

```
--- inference (new, hallucinated names) ---
sample  1: Ava
sample  2: Emilia
sample  3: Kael
...
sample 20: Brionna
```

The exact names vary between runs if you change the random seed, but with `random.seed(42)` they are deterministic. The model generates each name one character at a time: it starts with the `BOS` token and repeatedly asks "given everything so far, what character comes next?" until it predicts `BOS` again (meaning end-of-name) or reaches the maximum length of 16 characters.

```python
temperature = 0.5
```

The `temperature` parameter at the top of the inference section controls how "creative" the sampling is. At `0.5` the model is fairly conservative and produces names that look English. Chapter 8 explores what happens as you move this value toward `0.0` (repetitive, very common names) or `1.0` (more varied, sometimes stranger outputs).

---

## A tour of the file's structure

Now that you have seen the output, here is the same five-stage view with the purpose of each stage made explicit. This is the mental model you will build out across the remaining chapters.

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: DATASET                                               │
│  Download input.txt, parse into list of strings (docs)          │
├─────────────────────────────────────────────────────────────────┤
│  STAGE 2: TOKENIZER                                             │
│  Build a character ↔ integer mapping; define BOS token          │
├─────────────────────────────────────────────────────────────────┤
│  STAGE 3: AUTOGRAD ENGINE                                       │
│  Value class — each scalar tracks its own gradient              │
│  .backward() walks the computation graph and fills .grad        │
├─────────────────────────────────────────────────────────────────┤
│  STAGE 4: MODEL ARCHITECTURE                                    │
│  state_dict holds weight matrices as lists-of-lists of Value    │
│  gpt() function: embeddings → attention → MLP → logits          │
├─────────────────────────────────────────────────────────────────┤
│  STAGE 5: TRAINING + INFERENCE                                  │
│  1000 steps: forward → loss → backward → Adam update           │
│  Then 20 samples via autoregressive generation                  │
└─────────────────────────────────────────────────────────────────┘
```

One thing worth noting now: there is **no class for the model**. The architecture is a plain function `gpt()` that takes token and position ids and returns logits. The parameters live in a plain dictionary `state_dict`. This makes the data flow easy to follow — there is no magic inside a framework, only Python lists and arithmetic.

---

## Troubleshooting

**The script hangs at startup.**
It is downloading `input.txt`. Wait 10–20 seconds. If it still hangs, check your internet connection or manually download the file:

```bash
curl -o input.txt https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt
```

**I see `ModuleNotFoundError`.**
Make sure you are running with Python 3, not Python 2: `python3 microgpt.py`.

**Training seems frozen.**
It is not frozen — the `\r` at the end of the print statement overwrites the line in place. If your terminal does not support carriage returns the line may appear to not update, but training is still running.

**The loss goes up, not down.**
With `random.seed(42)` the training is deterministic and should decrease. If you have edited the file, revert to the original. If you are seeing this on an unmodified file, open an issue in the repository.

---

## Check your understanding

1. The script prints `vocab size: 28`. The names dataset only uses the 26 letters of the English alphabet, yet the vocab size is 28. What accounts for the extra two tokens, and where in the file are they defined?

2. The training loop uses `end='\r'` when printing the loss. What would you change to see the loss printed on a new line for every step, so you could redirect training output to a log file?

3. The model has 21,532 parameters. The embedding dimension is `n_embd = 16` and the vocabulary size is 28. The token embedding matrix `wte` and position embedding matrix `wpe` are defined as `matrix(vocab_size, n_embd)` and `matrix(block_size, n_embd)` respectively. How many parameters do these two matrices contribute in total, and what fraction of 21,532 is that?

---

## What's next

In Chapter 2 you will go deeper into stages 1 and 2 of the file: how the dataset is loaded, what character-level tokenization means in practice, and exactly how the `BOS` token is used to frame each training example. You will also see how the tokenizer encodes a full name into a sequence of integers and how that sequence is sliced into input–target pairs for training.
