# Chapter 2: Dataset and Tokenization

Before the model can learn anything, raw text must be converted into numbers. This chapter
walks through exactly how `microgpt.py` loads a list of baby names, shuffles them for
training, and encodes every character as an integer. You will build the same tokenizer by
hand in a Python REPL so that nothing feels like magic.

---

## What you'll learn

- How character-level tokenization works and how it differs from subword methods like BPE
- How a sorted unique-character list becomes a direct mapping between letters and integer IDs
- What the BOS token is, why its ID is placed *after* all character IDs, and why it appears
  on **both ends** of every training sequence
- How the sliding-window loop turns a tokenized sequence into (input, target) pairs
- Why `random.seed(42)` and `random.shuffle` matter for reproducible training

---

## Prerequisites

- Chapter 1 completed: you can run `microgpt.py` and read its top-level structure
- Python 3.8+ REPL available (`python3` or `ipython`)
- `input.txt` already downloaded to `/Users/kumarishan/Projects/llm-lab/` (running the
  script once creates it automatically)

---

## 2.1 Background: what is tokenization?

Every neural network works on numbers, not text. Tokenization is the process that bridges
the gap: it converts a string into a sequence of integers called *tokens*, and provides a
reverse mapping so you can decode tokens back to text.

Modern large language models (GPT-4, Llama, etc.) use **subword tokenization** (typically
Byte-Pair Encoding, or BPE). BPE starts with individual bytes, then iteratively merges the
most frequent adjacent pairs into single tokens. This gives a vocabulary of roughly
30,000–100,000 tokens, balancing efficiency with coverage.

`microgpt.py` takes the simplest possible approach instead: **character-level tokenization**.
Every unique character in the dataset gets its own integer ID. There is no merging step, no
special encoding library, no external dependency. The entire vocabulary is derived directly
from the data in three lines of code.

Character-level models were popularized by Andrej Karpathy's 2015 blog post
*"The Unreasonable Effectiveness of Recurrent Neural Networks"*. The approach trades
vocabulary efficiency for extreme simplicity: the model must learn to compose characters
into words and words into meaning entirely on its own.

---

## 2.2 Downloading and loading the dataset

[view source: microgpt.py lines 14-21](../microgpt.py)

```python
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")
```

`names.txt` is the same dataset used in Karpathy's `makemore` project: 32,033 first names
drawn from US census data, one name per line.  The `if not os.path.exists` guard means the
file is only downloaded once; subsequent runs load it from disk.

The list comprehension on the `docs = ...` line does two things at once:

- `line.strip()` removes leading/trailing whitespace and the trailing newline
- `if line.strip()` skips any blank lines (the `strip()` is evaluated twice, but Python
  short-circuits on the filter so blank lines are never added)

`random.shuffle(docs)` randomizes the order. This ensures that even though the training
loop visits documents in a simple round-robin (`docs[step % len(docs)]`), the model does
not see names in alphabetical order, which would bias early training.

> **Why shuffle at the start instead of sampling randomly during training?**
> Sampling with replacement can cause the model to see popular names many times before it
> ever sees rare ones. A single upfront shuffle followed by a deterministic cycle gives
> uniform coverage over the full dataset each epoch while keeping the training loop simple.

`random.seed(42)` is called on line 12, *before* the shuffle, so the shuffled order is
identical every time you run the script. This is essential for reproducibility: two
researchers running the same file will produce the same training trajectory.

---

## 2.3 Building the tokenizer

[view source: microgpt.py lines 23-27](../microgpt.py)

```python
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")
```

Three lines, but a lot is packed in. Let's unpack each one.

### `''.join(docs)` — flatten everything into one string

`docs` is a list of strings like `["emma", "olivia", "ava", ...]`. Joining them with an
empty separator produces one long string of every character in the dataset. This is the
universe of characters the model will ever see.

### `set(...)` — deduplicate

A Python `set` keeps only unique elements. Applying it to the joined string gives us the
set of distinct characters.

### `sorted(...)` — impose a stable order

Sets in Python have no guaranteed iteration order. `sorted` turns the set into a list in
ascending Unicode code-point order. This matters: the mapping from character to integer ID
must be *deterministic*. If the order changed between runs, the model weights learned in
one run would be meaningless in another.

The result is a list like:
```
['-', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
```

The list index *is* the token ID: `'-'` → 0, `'a'` → 1, `'b'` → 2, and so on up to
`'z'` → 26. That gives 27 character tokens.

### `BOS = len(uchars)` — the special token sits at the end

`BOS` (Beginning of Sequence) is a **special token** that marks the boundary of a
sequence. Its ID is set to `len(uchars)`, which is 27 — one past the last character ID.

> **Why not use 0 for BOS?**
> If BOS were 0, it would collide with `'-'` (or whatever the first character is), making
> it impossible to distinguish the boundary token from a real character in the data. By
> placing BOS after all character IDs, every ID has a unique, unambiguous meaning.

### `vocab_size = len(uchars) + 1`

The full vocabulary is 27 characters + 1 BOS = **28 tokens**. The model's token embedding
table (`wte`) has 28 rows, one learned vector per token ID.

---

## 2.4 Hands-on: tokenize a name step by step

Open a Python REPL in the repo directory and follow along.

```python
# Step 1 — load the same data the script uses
import random
random.seed(42)

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)          # same shuffle as the script
print(f"Total names: {len(docs)}")
print(f"First few after shuffle: {docs[:5]}")
```

```python
# Step 2 — build the vocabulary
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1

print(f"uchars ({len(uchars)}): {uchars}")
print(f"BOS token ID: {BOS}")
print(f"vocab_size: {vocab_size}")
```

You should see output similar to:
```
uchars (27): ['-', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
              'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
              'x', 'y', 'z']
BOS token ID: 27
vocab_size: 28
```

```python
# Step 3 — encode a name manually
name = "emma"
tokens = [BOS] + [uchars.index(ch) for ch in name] + [BOS]
print(f"Encoded '{name}': {tokens}")
```

Expected output:
```
Encoded 'emma': [27, 5, 13, 13, 1, 27]
```

The sequence is: `[BOS, 'e', 'm', 'm', 'a', BOS]` → `[27, 5, 13, 13, 1, 27]`.

```python
# Step 4 — decode back to a string
decoded = ''.join(uchars[t] for t in tokens if t != BOS)
print(f"Decoded: '{decoded}'")
# -> Decoded: 'emma'
```

You have just completed an encode-decode round-trip. The tokenizer is lossless: you can
always recover the original string from the token IDs (ignoring BOS tokens).

```python
# Step 5 — inspect the (input, target) pairs the training loop will see
block_size = 16
n = min(block_size, len(tokens) - 1)

print(f"\nTraining pairs for '{name}':")
print(f"{'pos':>3}  {'input token':>12}  {'input char':>10}  {'target token':>13}  {'target char':>11}")
for pos_id in range(n):
    token_id  = tokens[pos_id]
    target_id = tokens[pos_id + 1]
    in_char  = 'BOS' if token_id  == BOS else repr(uchars[token_id])
    out_char = 'BOS' if target_id == BOS else repr(uchars[target_id])
    print(f"{pos_id:>3}  {token_id:>12}  {in_char:>10}  {target_id:>13}  {out_char:>11}")
```

You should see:
```
pos  input token  input char  target token  target char
  0           27         BOS             5          'e'
  1            5         'e'            13          'm'
  2           13         'm'            13          'm'
  3           13         'm'             1          'a'
  4            1         'a'            27          BOS
```

> **What's happening:** Each row is one prediction task. The model is given the token at
> `pos_id` as input and must predict the token at `pos_id + 1` as output. Reading the
> table: given BOS, predict 'e'; given 'e', predict 'm'; given 'm', predict 'm'; given
> 'm', predict 'a'; given 'a', predict BOS (i.e., "the name ends here").

---

## 2.5 Why BOS appears on both sides

[view source: microgpt.py lines 155-158](../microgpt.py)

```python
doc = docs[step % len(docs)]
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
n = min(block_size, len(tokens) - 1)
```

The BOS token does double duty.

**Leading BOS (position 0) — "start of generation" signal**

During inference (lines 191–199), the model is seeded with `token_id = BOS` at `pos_id =
0`. It then autoregressively predicts the next token, and the next, until it predicts BOS
again. The model can only learn to start generating from a cold BOS if it has been trained
on sequences that *begin* with BOS.

**Trailing BOS (last position) — "end of name" signal**

In language modelling, the model must learn when to stop. Without a stop signal, it would
produce infinitely long sequences. By appending BOS at the end of every training sequence,
the model learns to predict BOS after the final character, which is the stopping condition
checked in the inference loop (`if token_id == BOS: break`).

Using the *same* token for both start and end is an elegant simplification. In larger
models these roles are typically split into separate `<BOS>` and `<EOS>` tokens, but in a
single-domain character-level model the symmetry works perfectly.

---

## 2.6 The sliding window and `block_size`

```python
n = min(block_size, len(tokens) - 1)
```

`block_size = 16` is the maximum context length the attention mechanism can see (defined on
line 77). The longest name in the dataset is 15 characters, which tokenizes to 17 tokens
(`[BOS] + 15 chars + [BOS]`). That would require `n = 16` prediction steps, which fits
exactly within `block_size`. No name is ever truncated — `block_size` is chosen with the
data in mind.

The `- 1` is because the last token can only be an *output*, never an *input*: there is
no token after it to serve as its target. A sequence of length `L` yields `L - 1` training
pairs.

---

## 2.7 Summary of the token ID space

| Range | What it represents |
|---|---|
| `0` | `'-'` (the hyphen character, appears in names like "mary-jane") |
| `1 – 26` | `'a'` through `'z'` |
| `27` | BOS (Beginning of Sequence / End of Sequence boundary token) |

Total: **28 token IDs**, matching `vocab_size = 28`.

---

## Check your understanding

1. The dataset contains a name like `"mary-jane"`. What token ID sequence does the
   training loop produce for it? (Hint: check the index of `'-'` in `uchars`.)

2. Suppose you replaced `sorted(set(...))` with just `list(set(...))`. What could go
   wrong when you save and reload model weights between two separate Python processes?

3. `n = min(block_size, len(tokens) - 1)`. For a one-character name like `"a"`, what is
   `n`? Trace the training pairs manually.

<details>
<summary>Answers</summary>

**1.** `"mary-jane"` has 9 characters. Encoded:
`[BOS] + [uchars.index(c) for c in "mary-jane"] + [BOS]`
= `[27, 13, 1, 18, 25, 0, 10, 1, 14, 5, 27]`
(`'-'` is at index 0, so the hyphen maps to token ID 0.)

**2.** `set` iteration order is not guaranteed to be stable across Python processes
(though it often is in practice on CPython 3.7+). If the order changed, the same
character would map to a different integer ID, so model weights trained in one session
would be applied to the wrong token embeddings in another. `sorted` eliminates this risk.

**3.** `"a"` tokenizes to `[BOS, 1, BOS]`, length 3. `n = min(16, 3 - 1) = 2`.
Training pairs: `(BOS → 'a')` at pos 0, `('a' → BOS)` at pos 1. The model sees
"given start-of-sequence, predict 'a'; given 'a', predict end-of-sequence."

</details>

---

## What's next

You now understand exactly how text becomes numbers. In **Chapter 3: Autograd from
Scratch**, we turn to the `Value` class (lines 30–72). This tiny object is the engine
behind all learning: it records every arithmetic operation in a computation graph so that
`loss.backward()` can compute gradients for every parameter automatically using the chain
rule. No PyTorch, no NumPy — just plain Python arithmetic and a topological sort.
