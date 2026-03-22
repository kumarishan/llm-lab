# Chapter 3: Tokenization — Training BPE from Scratch

## What you'll learn

- Why models need integers instead of raw text, and the trade-offs behind different tokenization strategies
- How the Byte-Pair Encoding (BPE) algorithm works — implemented from scratch in ~30 lines of Python
- What the GPT-4 split pattern does and why pre-tokenization matters
- How nanochat's `RustBPETokenizer` combines `rustbpe` for training with `tiktoken` for fast inference
- How to train your own tokenizer, understand special tokens, and interpret the chat tokenization format with loss masking

## Prerequisites

- Chapters 1 and 2 completed (environment set up, concepts of tokens and embeddings understood)
- Comfortable reading Python; no prior NLP library experience required
- nanochat repo checked out and dependencies installed (`uv sync`)

---

## 3.1 Why tokenization?

A language model is ultimately a function that takes a sequence of integers and predicts the next integer. It cannot consume raw Unicode strings directly. The job of a tokenizer is to map text to integers and back, losslessly.

Three strategies exist along a spectrum:

| Strategy | Vocabulary size | Sequence length | Notes |
|---|---|---|---|
| Character-level | ~100–200 | Very long | Simple but sequences are huge; hard to model long-range dependencies |
| Word-level | 100 K+ | Short | Vocabulary explosion; every new word needs an entry; can't handle typos |
| Subword (BPE, WordPiece, Unigram) | 32 K–100 K | Medium | Sweet spot; common words get a single token, rare words split into pieces |

nanochat uses subword BPE with a vocabulary of 32 768 tokens (2^15). That number is a deliberate choice: large enough to give good compression of common English text, small enough that the embedding table does not dominate model parameters at smaller scales.

**The fundamental trade-off:** A larger vocabulary means shorter token sequences (fewer steps for the model to process), but it also means a bigger embedding matrix and a bigger output projection layer. For a 32 K vocabulary and a 512-dimensional model, the embedding table alone is 32 768 × 512 = 16.7 M parameters — already a significant fraction of a small model.

---

## 3.2 Byte-Pair Encoding from scratch

BPE was originally a text-compression algorithm. In the NLP context it was popularized by Sennrich et al. (2016) and later adopted by GPT-2 and all GPT-family models.

### The core idea

1. Start with a vocabulary of individual bytes (256 entries, covering every possible byte value).
2. Count how often every consecutive pair of tokens appears in the training corpus.
3. Merge the most frequent pair into a new single token. Add that token to the vocabulary.
4. Repeat from step 2 until you reach the target vocabulary size.

Every merge reduces sequence length and increases vocabulary size by exactly one. After `V - 256` merges you have a vocabulary of size `V`.

### A concrete walk-through

Let's tokenize the string `"aaabdaaabac"` to see BPE in action.

**Initial state** (each character is its own token):

```
tokens: ['a','a','a','b','d','a','a','a','b','a','c']
vocab:  {'a':0, 'b':1, 'd':2, 'c':3}
```

**Step 1:** Count pairs: `(a,a)` appears 4 times, `(a,b)` appears 2 times, others once. Best pair: `(a,a)`. Merge it to a new token `aa`:

```
tokens: ['aa','a','b','d','aa','a','b','a','c']
vocab:  {..., 'aa':4}
```

**Step 2:** Count again: `(aa,a)` appears 2 times. Merge to `aaa`:

```
tokens: ['aaa','b','d','aaa','b','a','c']
vocab:  {..., 'aaa':5}
```

**Step 3:** `(aaa,b)` appears 2 times. Merge to `aaab`:

```
tokens: ['aaab','d','aaab','a','c']
vocab:  {..., 'aaab':6}
```

The string that started as 11 characters is now 5 tokens. Common substrings compress to single tokens.

### Python implementation

Here is a complete toy BPE tokenizer. It works on characters (not bytes), which keeps the code readable. nanochat's real implementation operates on bytes, but the algorithm is identical.

✍️ **Create `scratch/bpe_toy.py` and type this out:**

```python
# scratch/bpe_toy.py
# Toy BPE tokenizer (character-level, ~30 lines of core logic)

def get_stats(ids):
    """Count occurrences of every consecutive pair in ids."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in ids with `new_id`."""
    result = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result

def train_bpe(text, vocab_size):
    """Train BPE on text, building a vocab of `vocab_size` tokens."""
    # Start: one token per unique character
    chars = sorted(set(text))
    vocab = {ch: i for i, ch in enumerate(chars)}  # str -> int
    ids = [vocab[ch] for ch in text]               # encode initial text

    merges = {}  # (int, int) -> int  (pair -> new token id)
    num_merges = vocab_size - len(vocab)

    for step in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break
        best_pair = max(stats, key=stats.get)
        new_id = len(vocab) + step
        ids = merge(ids, best_pair, new_id)
        merges[best_pair] = new_id
        print(f"step {step+1}: merged {best_pair} -> {new_id}  (count={stats[best_pair]})")

    return merges, vocab

def encode(text, vocab, merges):
    ids = [vocab[ch] for ch in text]
    # Apply merges in the order they were learned
    for pair, new_id in merges.items():
        ids = merge(ids, pair, new_id)
    return ids

def decode(ids, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    # Reconstruct strings for merged tokens
    def token_str(tid):
        if tid in inv_vocab:
            return inv_vocab[tid]
        # Find the pair that was merged to produce this token
        for (a, b), mid in merges.items():
            if mid == tid:
                return token_str(a) + token_str(b)
        raise ValueError(f"Unknown token id: {tid}")
    return "".join(token_str(tid) for tid in ids)

# --- Try it out ---
text = "aaabdaaabac"
merges, vocab = train_bpe(text, vocab_size=len(set(text)) + 4)
ids = encode(text, vocab, merges)
print(f"\nOriginal:  {list(text)}  ({len(text)} chars)")
print(f"Encoded:   {ids}  ({len(ids)} tokens)")
print(f"Decoded:   {decode(ids, vocab)}")
```

✍️ **Run it:**

```bash
python scratch/bpe_toy.py
```

You should see output like:

```
step 1: merged (0, 0) -> 4  (count=4)
step 2: merged (4, 0) -> 5  (count=2)
step 3: merged (5, 1) -> 6  (count=2)
step 4: merged (6, 3) -> 7  (count=1)

Original:  ['a', 'a', 'a', 'b', 'd', 'a', 'a', 'a', 'b', 'a', 'c']  (11 chars)
Encoded:   [6, 3, 6, 0, 2]  (5 tokens)
Decoded:   aaabdaaabac
```

> **What's happening:** Each merge step greedily picks the most frequent pair in the *current* token sequence — not the original characters. This means early merges influence which pairs are available in later steps. The vocabulary is deterministic given the training corpus and target size.

**Compression intuition:** A well-trained BPE tokenizer on English text typically compresses roughly 3–5 bytes per token. GPT-4's tokenizer achieves about 4 bytes/token on English prose. nanochat targets approximately the same range with its 32 K vocabulary.

---

## 3.3 The GPT-4 split pattern

Raw BPE would happily merge across word boundaries — for example, the space before a word with the word itself, or the end of one word with the beginning of the next. This produces weird tokens that generalize poorly.

To prevent this, GPT-4 applies a **pre-tokenization split** using a regular expression before BPE sees any text. The text is split into non-overlapping chunks; BPE then runs independently within each chunk, so merges never cross chunk boundaries.

nanochat uses this pattern (from `nanochat/tokenizer.py`):

```python
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

This looks intimidating. Let's break it into its alternation branches:

| Branch | What it matches | Example |
|---|---|---|
| `'(?i:[sdmt]\|ll\|ve\|re)` | English contractions | `'s`, `'re`, `'ll`, `'ve`, `'d`, `'m`, `'t` |
| `[^\r\n\p{L}\p{N}]?+\p{L}+` | Optional leading punctuation + letters | `"Hello"`, ` world`, `(function` |
| `\p{N}{1,2}` | Numbers, max 2 digits | `42`, `9` (not `123`) |
| ` ?[^\s\p{L}\p{N}]++[\r\n]*` | Punctuation / symbols | `...`, `!!`, `@#` |
| `\s*[\r\n]` | Newlines | line endings |
| `\s+(?!\S)` | Trailing whitespace | space at end of line |
| `\s+` | Other whitespace | tabs, spaces |

> **Note on numbers:** The original GPT-4 pattern uses `\p{N}{1,3}` (up to 3-digit numbers per token). nanochat deliberately uses `\p{N}{1,2}` (max 2 digits). The comment in `tokenizer.py` explains: at 32 K vocab, spending tokens on 3-digit sequences is less efficient than allowing BPE to form them naturally from 2-digit pieces. This is an empirical finding specific to smaller vocabulary sizes.

**Why this matters:** Without the split pattern, a tokenizer trained on "New York" might create a token `" New"` and a token `" York"`, but also `" New York"` as a single merged token. The presence of that merged token depends on corpus frequency and can produce inconsistent representations. The split pattern ensures punctuation, contractions, and numbers always start fresh BPE contexts.

---

## 3.4 Byte fallback: encoding anything

The toy BPE above broke on any character not in the training corpus. A production tokenizer must handle every possible input: emoji, Chinese, Arabic, code, mixed languages.

The solution: start the vocabulary with all 256 possible byte values (0–255) rather than the characters in the training set. Every Unicode string can be encoded to UTF-8 bytes first, so it can always be represented — worst case, one token per byte.

In practice:
- Common English words compress to 1–2 tokens (`" the"` is one token in most GPT-family tokenizers)
- Rare scripts or emoji may expand to 3–4 tokens per character (e.g., an emoji like 🌍 is 4 UTF-8 bytes, potentially 4 tokens if it has never been seen during training)
- But encoding is always possible and always reversible — no unknown-token problem

The `HuggingFaceTokenizer` in nanochat makes this explicit with `byte_fallback=True` in the BPE model configuration.

---

## 3.5 nanochat's RustBPETokenizer

nanochat ships two tokenizer classes in `nanochat/tokenizer.py`:

- `HuggingFaceTokenizer`: wraps the `tokenizers` library; useful for compatibility
- `RustBPETokenizer`: trains with `rustbpe`, runs inference with `tiktoken` — **this is the one used everywhere**

### Why the split design?

Training BPE on 2 billion characters is slow in pure Python. The `rustbpe` library implements the training loop in Rust, which handles the character counting and merging efficiently. However, `rustbpe`'s inference is not optimized for throughput.

For inference (encoding text during dataset preparation and model evaluation), nanochat uses OpenAI's `tiktoken` library. `tiktoken` wraps a C/Rust extension that is extremely fast at applying a learned BPE vocabulary — it is the same library used to tokenize inputs to GPT-4.

The two-library design means: **train once with rustbpe, run forever with tiktoken.**

### The constructor pipeline

Look at `train_from_iterator` in `RustBPETokenizer`:

```python
# nanochat/tokenizer.py (abridged)

@classmethod
def train_from_iterator(cls, text_iterator, vocab_size):
    # 1) Train using rustbpe
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)

    # 2) Extract the learned merge table
    pattern = tokenizer.get_pattern()
    mergeable_ranks_list = tokenizer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}

    # 3) Hand it to tiktoken for inference
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,   # dict[bytes, int]
        special_tokens=special_tokens,     # dict[str, int]
    )
    return cls(enc, "<|bos|>")
```

> **What's happening:** rustbpe produces `mergeable_ranks`: a dictionary mapping token bytes to their rank (lower rank = earlier / more frequent merge). tiktoken uses this table at inference time to greedily apply merges in rank order, which reconstructs the same tokenization that the BPE training algorithm would produce.

### Encoding and decoding

✍️ **Open a Python REPL inside the nanochat repo and try this:**

```python
from nanochat.tokenizer import get_tokenizer

tok = get_tokenizer()  # loads from ~/.cache/nanochat/tokenizer/

# Basic encode/decode roundtrip
text = "Hello, world! I'm learning to build LLMs."
ids = tok.encode(text)
print(ids)          # list of integers
print(tok.decode(ids))  # should equal text exactly

# Batch encoding (uses multiple threads internally)
texts = ["The cat sat on the mat.", "Transformers are attention mechanisms."]
batch_ids = tok.encode(texts)  # list of lists

# Vocabulary size
print(tok.get_vocab_size())  # 32768

# Map a token ID back to its string
print(tok.id_to_token(1000))
```

> **What's happening:** `encode` calls `tiktoken`'s `encode_ordinary`, which applies the split pattern first, then greedily merges byte pairs using the learned rank table. `decode` is the inverse: look up each token ID's byte sequence and concatenate. Batched encoding (`list` input) uses `encode_ordinary_batch` which parallelizes across threads.

---

## 3.6 Special tokens

Regular tokens are learned from data and represent byte sequences. Special tokens are different: they are added manually *after* BPE training, assigned IDs beyond the learned vocabulary, and are never produced by the BPE merging process. They must be inserted explicitly by application code.

nanochat defines nine special tokens in `nanochat/tokenizer.py`:

```python
SPECIAL_TOKENS = [
    "<|bos|>",            # Beginning-of-sequence; every document starts with this
    "<|user_start|>",     # Opens a user turn in a conversation
    "<|user_end|>",       # Closes a user turn
    "<|assistant_start|>",# Opens an assistant turn
    "<|assistant_end|>",  # Closes an assistant turn
    "<|python_start|>",   # Opens a Python tool-call block (assistant invoking the REPL)
    "<|python_end|>",     # Closes a Python tool-call block
    "<|output_start|>",   # Opens REPL output returned to the assistant
    "<|output_end|>",     # Closes REPL output
]
```

Their IDs are assigned sequentially starting at `vocab_size_no_special` (i.e., immediately after the BPE tokens). With a 32 768 target vocabulary and 9 special tokens, the BPE portion is 32 759 tokens (IDs 0–32758) and special tokens occupy IDs 32759–32767.

**Why do we need them?**

1. **Document delimiting:** `<|bos|>` signals the start of a new sequence to the model. During pretraining, documents are concatenated into long token streams; the model learns that seeing `<|bos|>` resets context.

2. **Chat structure:** The `<|user_start|>` / `<|assistant_start|>` pairs tell the model whose turn it is. Without them, the model would see user and assistant text as a uniform stream and could not learn role-specific behavior.

3. **Tool use:** `<|python_start|>` and `<|output_start|>` mark code-execution boundaries, enabling the model to learn when to invoke Python and how to interpret outputs.

---

## 3.7 Training a tokenizer

### The training script

`scripts/tok_train.py` orchestrates the full pipeline. Read it section by section:

**Arguments:**

```python
# scripts/tok_train.py
parser.add_argument('--max-chars',  type=int, default=2_000_000_000)  # 2 billion chars
parser.add_argument('--doc-cap',    type=int, default=10_000)          # 10 KB per document
parser.add_argument('--vocab-size', type=int, default=32768)
```

`--doc-cap` prevents a single very long document from dominating the pair-frequency counts. Capping at 10 000 characters means a 1 MB document contributes the same weight as a 10 KB document.

**The text iterator:**

```python
def text_iterator():
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc[:args.doc_cap]   # trim long documents
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return
```

This is a Python generator. `rustbpe.train_from_iterator` pulls from it lazily — the entire 2 B characters never need to be in memory at once.

**Training:**

```python
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
```

**Saving:**

```python
tokenizer_dir = os.path.join(base_dir, "tokenizer")  # ~/.cache/nanochat/tokenizer/
tokenizer.save(tokenizer_dir)
```

`save` pickles the `tiktoken.Encoding` object to `tokenizer.pkl`. On the next load, `RustBPETokenizer.from_directory` unpickles it and hands it to tiktoken — no re-training needed.

**Token byte counts:**

After saving, the script computes a tensor `token_bytes` that records how many UTF-8 bytes each token ID decodes to. This is used later when evaluating the model with **bits-per-byte** loss, a metric that normalizes cross-entropy for the tokenizer's compression ratio. Special tokens get a byte count of 0 (they don't represent real text content).

### Running the training

✍️ **Train a tokenizer** (this assumes you have the dataset downloaded per Chapter 1):

```bash
# Quick tokenizer for experimentation (100M chars, small vocab)
python scripts/tok_train.py --max-chars 100_000_000 --vocab-size 8192

# Production tokenizer (matches the defaults used in nanochat)
python scripts/tok_train.py --max-chars 2_000_000_000 --vocab-size 32768
```

The training output will look like:

```
max_chars: 2,000,000,000
doc_cap: 10,000
vocab_size: 32,768
Training time: 142.37s
Saved tokenizer encoding to /Users/you/.cache/nanochat/tokenizer/tokenizer.pkl
Saved token_bytes to /Users/you/.cache/nanochat/tokenizer/token_bytes.pt
```

> **What's happening:** rustbpe reads the text stream, counts byte-pair frequencies in parallel, and performs 32 759 − 256 = 32 503 merges. At each step it records the winning pair and assigns it a new ID. The resulting merge table is serialized into a `tiktoken.Encoding`, pickled, and written to disk.

---

## 3.8 Chat tokenization: render_conversation()

Pretraining tokenizes plain text. Fine-tuning tokenizes *conversations* — structured exchanges between a user and an assistant. The tokenizer needs to:

1. Wrap each turn in the right special tokens
2. Indicate which tokens the model should learn to predict (the assistant's words) and which are just context (the user's words)

This is the job of `render_conversation`.

### Loss masking

During supervised fine-tuning the loss is computed only on tokens where `mask == 1`. User tokens, system tokens, and structural tokens all have `mask == 0` — the model sees them but is not penalized for not predicting them. Only assistant response tokens have `mask == 1`.

If you trained on all tokens equally, the model would waste capacity learning to predict user messages (which arrive from outside the model at inference time) instead of focusing on generating good assistant responses.

### The token layout

For a two-turn conversation:

```
User: "What is 2+2?"
Assistant: "It's 4."
```

The token stream becomes:

```
<|bos|> <|user_start|> What is 2+2? <|user_end|>
        <|assistant_start|> It's 4. <|assistant_end|>
```

With masks:

```
mask:  0       0           0...0       0
                    0               1...1         1
```

Everything in angle brackets and the user's words: mask 0. The assistant's words and `<|assistant_end|>`: mask 1.

### Walking through the code

```python
# nanochat/tokenizer.py (render_conversation, abridged)

def render_conversation(self, conversation, max_tokens=2048):
    ids, mask = [], []

    def add_tokens(token_ids, mask_val):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))

    # System message: prepend to the first user message
    if conversation["messages"][0]["role"] == "system":
        conversation = copy.deepcopy(conversation)
        messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
        messages = messages[1:]
    # ...

    add_tokens(bos, 0)  # <|bos|>, not supervised

    for i, message in enumerate(messages):
        if message["role"] == "user":
            add_tokens(user_start, 0)
            add_tokens(self.encode(content), 0)  # user text, not supervised
            add_tokens(user_end, 0)
        elif message["role"] == "assistant":
            add_tokens(assistant_start, 0)       # opening bracket, not supervised
            add_tokens(self.encode(content), 1)  # assistant text, supervised
            add_tokens(assistant_end, 1)         # closing bracket, supervised

    return ids[:max_tokens], mask[:max_tokens]
```

> **Note on `<|assistant_end|>`:** The closing bracket is included in the supervised tokens (mask 1). This teaches the model to generate it — without learning to produce `<|assistant_end|>`, the model would never know when to stop generating.

### Tool calls

For conversations where the assistant invokes Python, the content is a list of parts rather than a plain string:

```python
# Example conversation with a tool call
{
  "messages": [
    {"role": "user", "content": "What is sqrt(144)?"},
    {"role": "assistant", "content": [
      {"type": "text",          "text": "Let me compute that."},
      {"type": "python",        "text": "import math; print(math.sqrt(144))"},
      {"type": "python_output", "text": "12.0"},
      {"type": "text",          "text": "The answer is 12."},
    ]}
  ]
}
```

The resulting token stream:

```
<|bos|>
<|user_start|> What is sqrt(144)? <|user_end|>
<|assistant_start|>
  Let me compute that.
  <|python_start|> import math; print(math.sqrt(144)) <|python_end|>
  <|output_start|> 12.0 <|output_end|>
  The answer is 12.
<|assistant_end|>
```

Masks: the Python code and the final text are supervised (mask 1); the output block is not supervised (mask 0) because at inference time Python produces the output — the model is not responsible for predicting it.

✍️ **Try visualizing a conversation tokenization:**

```python
from nanochat.tokenizer import get_tokenizer

tok = get_tokenizer()

conversation = {
    "messages": [
        {"role": "user",      "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
}

ids, mask = tok.render_conversation(conversation)
print(f"Token count: {len(ids)}")
print(f"Supervised tokens: {sum(mask)}")
print()
print(tok.visualize_tokenization(ids, mask))
```

The terminal output will color supervised tokens green and non-supervised tokens red, separated by `|`. This is your ground truth for debugging data pipelines.

---

## 3.9 Evaluating a tokenizer

A tokenizer is a compression function. The primary metric is **bytes per token** (also called the compression ratio): how many UTF-8 bytes does one token represent on average? Higher is better — it means the model processes more information per step.

```
bytes per token = total_bytes(text) / total_tokens(encoded_text)
```

The reciprocal — tokens per byte, or "fertility" — is sometimes used instead. GPT-2 achieves roughly 3.5 bytes/token on English; GPT-4 achieves about 4.0; a well-trained 32 K tokenizer on a mixed English+code corpus should land around 3.5–4.0.

### Running tok_eval.py

`scripts/tok_eval.py` evaluates your tokenizer against GPT-2 and GPT-4 on six text samples: news, Korean, Python code, LaTeX math, scientific prose, and a sample from the training and validation splits.

✍️ **Run the evaluation:**

```bash
python scripts/tok_eval.py
```

Sample output (exact numbers depend on your training run):

```
Vocab sizes:
GPT-2: 50257
GPT-4: 100277
Ours:  32768

Comparison with GPT-2:
============================================================================================
Text Type  Bytes    GPT-2           Ours            Relative     Better
                    Tokens  Ratio   Tokens  Ratio   Diff %
--------------------------------------------------------------------------------------------
news       2341     604     3.88    551     4.25    + 8.8%       Ours
korean     512      706     0.73    498     1.03    +29.5%       Ours
code       1289     386     3.34    341     3.78    +11.7%       Ours
math       2108     648     3.25    587     3.59    + 9.4%       Ours
science    891      237     3.76    215     4.14    + 9.3%       Ours
...

Comparison with GPT-4:
...
```

**Interpreting the results:**

- **Korean:** GPT-2 scores ~0.73 bytes/token on Korean because it was trained almost exclusively on English. Your tokenizer, trained on more diverse data, should score significantly higher.
- **Code:** Both GPT-2 and a domain-matched tokenizer compress code well, since code has repetitive structure.
- **Relative diff:** Positive means your tokenizer uses fewer tokens for the same text — better compression.

> **What's happening:** The script encodes each text with three tokenizers, counts bytes and tokens, and computes ratios. It then prints a color-coded comparison table (green = better compression, red = worse). The `assert decoded == text` check verifies lossless roundtrip for each tokenizer.

### Bits per byte

During model training, nanochat reports loss as **bits per byte** rather than the raw cross-entropy value. This normalization makes loss comparable across different tokenizer vocabulary sizes:

```
bits_per_byte = cross_entropy_loss / log(2) / mean_token_bytes
```

where `mean_token_bytes` is read from `token_bytes.pt` (built by `tok_train.py`). A good language model on English text should reach below 1.0 bits per byte; random prediction on 8-bit data is 8.0 bits per byte.

---

## Check your understanding

1. **BPE mechanics:** In the toy BPE walk-through, the pair `(a, a)` was merged before `(a, b)` even though `(a, b)` also appeared multiple times. Why? What would happen if you ran one more merge step after the three shown?

2. **Split pattern:** Suppose the split pattern did not exist and BPE ran on raw text. A tokenizer trained on a corpus where `"New York"` always appears together would likely learn a token for `" New"`, `" York"`, and possibly `" New York"`. What problem would this cause when encoding the sentence `"New Delhi is in India"`?

3. **Loss masking:** A training example has 80 tokens total: 30 for the user turn and 50 for the assistant turn. If you computed the average cross-entropy loss over all 80 tokens instead of only the 50 masked ones, how would this distort training? Would the model over- or under-weight learning to generate assistant responses?

---

## What's next

Chapter 4 covers the model architecture: how the token IDs produced by this chapter's tokenizer flow through an embedding table, a stack of Transformer blocks, and a final projection head to produce next-token probability distributions. We will read nanochat's `model.py` and connect it back to the attention math from Chapter 2.
