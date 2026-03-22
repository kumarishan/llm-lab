# Chapter 1: Setup and First Run

## What you'll learn

- How to install `uv` and set up the nanochat project from scratch
- What each major dependency does and why the project needs it
- How nanochat automatically picks the right compute device (CPU, Apple Silicon, or NVIDIA GPU)
- How to run a tiny model end-to-end on CPU: tokenizer training, pretraining, and SFT
- The full pipeline you'll build across this tutorial — from raw text to a chatting LLM

---

## Prerequisites

- Python experience: you're comfortable writing functions, classes, and working with the command line
- A working terminal (macOS, Linux, or WSL on Windows)
- Internet access (the first run downloads ~2 GB of training data)
- **No prior machine learning or LLM knowledge required**

You do NOT need a GPU to complete this chapter. Everything here runs on a laptop CPU. Later chapters cover GPU training.

---

## Background: what is nanochat?

Before touching a single file, it helps to understand what you're looking at.

An LLM — a Large Language Model — is a neural network trained to predict the next token in a sequence of text. "Token" roughly means "word fragment": the string `"Hello, world!"` might become something like `["Hello", ",", " world", "!"]`. By training on billions of tokens of internet text and repeatedly asking "given what came before, what comes next?", the model learns grammar, facts, reasoning patterns, and the texture of human writing.

nanochat is a complete, self-contained training harness for doing this from scratch. It is intentionally minimal: you can read every important file in a day. The entire pipeline — tokenizer training, base model pretraining, supervised fine-tuning, reinforcement learning, evaluation, and a web chat UI — lives in one repository with no giant framework abstractions. This makes it an ideal learning environment.

The project was created by Andrej Karpathy. At full scale (8 × H100 GPUs, ~2 hours, ~$48), it produces a model with GPT-2-class capability. In this chapter, you will run a drastically smaller version that fits on a laptop.

---

## The full pipeline at a glance

Below is the end-to-end pipeline you will understand by the end of this tutorial. Each stage is covered in its own chapter.

```
Raw internet text (FineWeb dataset)
        │
        ▼
┌───────────────────┐
│  1. Tokenizer     │  scripts/tok_train.py
│  Training         │  Learns a 32K-token vocabulary from raw text
│                   │  using Byte-Pair Encoding (BPE)
└────────┬──────────┘
         │  vocabulary file + merge rules
         ▼
┌───────────────────┐
│  2. Pretraining   │  scripts/base_train.py
│  (Base Model)     │  Trains a GPT transformer to predict
│                   │  the next token on billions of tokens
└────────┬──────────┘
         │  base model checkpoint (.pt file)
         ▼
┌───────────────────┐
│  3. Supervised    │  scripts/chat_sft.py
│  Fine-Tuning      │  Teaches the model the User/Assistant
│  (SFT)            │  conversation format using labeled dialogues
└────────┬──────────┘
         │  chat-tuned checkpoint
         ▼
┌───────────────────┐
│  4. Reinforcement │  scripts/chat_rl.py
│  Learning (RL)    │  Rewards the model for correct answers
│                   │  on structured tasks (math, coding, etc.)
└────────┬──────────┘
         │  RL-tuned checkpoint
         ▼
┌───────────────────┐
│  5. Evaluation    │  scripts/base_eval.py, scripts/chat_eval.py
│                   │  Measures quality: bits-per-byte, DCLM CORE
│                   │  score, task-specific benchmarks
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  6. Inference     │  scripts/chat_cli.py
│  (CLI or Web UI)  │  scripts/chat_web.py
│                   │  Talk to your model in a terminal or browser
└───────────────────┘
```

In this chapter you will run the entire pipeline in miniature — tiny model, tiny dataset, fast enough for a laptop — so you understand what each stage produces before you study any of them in depth.

---

## Step 1: Install `uv`

nanochat uses [`uv`](https://github.com/astral-sh/uv), a fast Python package manager written in Rust. It replaces `pip`, `venv`, and `pyenv` in one tool: it creates virtual environments, installs packages at high speed, and pins exact versions in a lockfile.

If you already have `uv` installed, skip ahead. Otherwise:

✍️ Run this in your terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your terminal (or run `source ~/.zshrc` / `source ~/.bashrc`) so the `uv` command is on your PATH. Verify:

```bash
uv --version
```

Expected output (exact version will vary):

```
uv 0.7.x (...)
```

> **What's happening:** The installer downloads a pre-built binary and places it in `~/.local/bin`. `uv` is not a Python package itself — it's a standalone executable, which is why you install it this way rather than with `pip install`.

---

## Step 2: Clone the repository

✍️ Run:

```bash
git clone https://github.com/karpathy/nanochat.git
cd nanochat
```

---

## Step 3: Create a virtual environment and install dependencies

nanochat pins its Python version to `3.10` in a file called `.python-version`. `uv` reads this file automatically when you run `uv venv`, so you don't need to specify the version manually.

✍️ Run:

```bash
uv venv
uv sync --extra cpu
source .venv/bin/activate
```

**What each command does:**

- `uv venv` — Creates a virtual environment in `.venv/` using Python 3.10 (read from `.python-version`). A virtual environment is an isolated copy of Python for this project so packages don't collide with your system Python.
- `uv sync --extra cpu` — Reads `pyproject.toml` and `uv.lock`, then installs all pinned dependencies. The `--extra cpu` flag selects the CPU-only build of PyTorch (smaller download). If you have an NVIDIA GPU you would use `--extra gpu` instead.
- `source .venv/bin/activate` — Puts the virtual environment's Python and scripts on your PATH for this terminal session.

After `uv sync`, you'll see output like:

```
Resolved 147 packages in 0.42s
Prepared 63 packages in 1m 14s
Installed 63 packages in 3.4s
```

> **What's happening:** `uv.lock` contains the exact version and hash of every package (including transitive dependencies) so the install is perfectly reproducible. Unlike `pip install -r requirements.txt`, a lockfile guarantees byte-for-byte identical installs across machines.

---

## Tour of the repository

Before running anything, take two minutes to understand the directory layout:

```
nanochat/
├── nanochat/          # Core library — importable Python package
│   ├── gpt.py         # The GPT transformer neural network (nn.Module)
│   ├── tokenizer.py   # BPE tokenizer: text ↔ token IDs
│   ├── dataloader.py  # Streams tokenized training data to the GPU
│   ├── common.py      # Device detection, dtype selection, utilities
│   ├── engine.py      # Fast inference with KV cache
│   ├── optim.py       # AdamW + Muon optimizer
│   └── ...
│
├── scripts/           # Runnable entry points (python -m scripts.X)
│   ├── tok_train.py   # Train a tokenizer from raw text
│   ├── tok_eval.py    # Measure how well the tokenizer compresses text
│   ├── base_train.py  # Pretrain the base language model
│   ├── base_eval.py   # Evaluate the base model (bits-per-byte, CORE score)
│   ├── chat_sft.py    # Supervised fine-tuning for chat
│   ├── chat_rl.py     # Reinforcement learning fine-tuning
│   ├── chat_cli.py    # Chat with the model in the terminal
│   └── chat_web.py    # Serve a ChatGPT-style web UI
│
├── tasks/             # Evaluation tasks (math, coding, multiple choice)
│   ├── gsm8k.py       # 8K grade-school math problems
│   ├── arc.py         # Science multiple-choice questions
│   ├── mmlu.py        # Broad knowledge multiple-choice
│   └── ...
│
├── runs/              # Shell scripts that wire stages together
│   ├── runcpu.sh      # Full pipeline, tiny model, for CPU/MacBook
│   └── speedrun.sh    # Full pipeline, GPT-2 size, for 8×H100
│
├── dev/               # Development utilities (logo, data tools)
├── tests/             # Unit tests (pytest)
├── pyproject.toml     # Project metadata and dependency declarations
└── uv.lock            # Exact pinned versions of all packages
```

The separation between `nanochat/` (library) and `scripts/` (entry points) is intentional. The library contains reusable components; the scripts wire them together for each training stage. You run scripts as modules: `python -m scripts.base_train`, not `python scripts/base_train.py`.

---

## What each dependency does

`pyproject.toml` lists the project's direct dependencies. Here is what each one is for:

| Package | Purpose |
|---|---|
| `torch` | PyTorch — the deep learning framework. The neural network, training loop, and tensor math all live here. |
| `tiktoken` | OpenAI's fast tokenizer library — used at inference time for efficient text-to-token-ID conversion. |
| `rustbpe` | A Rust-based BPE tokenizer trainer — used when training your own tokenizer vocabulary from raw text. Faster than pure Python. |
| `tokenizers` | HuggingFace's tokenizer library — an alternative tokenizer backend used in some code paths. |
| `datasets` | HuggingFace Datasets — downloads and streams training data (FineWeb, SmolTalk, etc.) from HuggingFace Hub. |
| `transformers` | HuggingFace Transformers — used to load pre-trained models and tokenizers (e.g., for evaluation baselines). |
| `fastapi` | The web framework powering the ChatGPT-style UI (`chat_web.py`). |
| `uvicorn` | The ASGI server that runs the FastAPI application. |
| `wandb` | Weights & Biases — logs training metrics (loss, throughput, etc.) to a dashboard. Setting `--run=dummy` disables it. |
| `scipy`, `matplotlib` | Scientific computing and plotting utilities — used for evaluation and scaling-law analysis. |
| `zstandard` | Compression library — training data shards are stored in `.zst` compressed format. |
| `regex` | Extended regular expressions — used in the tokenizer's text pre-splitting pattern. |
| `psutil` | System resource queries — used to report CPU/RAM usage during training. |
| `python-dotenv` | Loads environment variables from a `.env` file — useful for storing API keys (e.g., your WandB key). |

---

## How device autodetection works

One of the first things nanochat does when any script starts is decide where to run computations: on an NVIDIA GPU (CUDA), on Apple Silicon (MPS), or on a regular CPU. This logic is in `nanochat/common.py`.

Open the file and look at lines 16–31:

```python
# nanochat/common.py

_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

def _detect_compute_dtype():
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"

COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()
```

This runs once at module import time and sets a global `COMPUTE_DTYPE`. Here is what each branch means:

**CUDA with SM 80+** (A100, H100): Uses `bfloat16`. These GPUs have hardware support for bfloat16 matrix multiplications ("tensor cores"), which are 2–4× faster than float32 and use half the memory.

**CUDA with SM < 80** (V100, T4): Falls back to `float32`. These older GPUs support float16 tensor cores but not bfloat16. float16 training is possible but requires a `GradScaler` to prevent numerical underflow — you can opt in with `NANOCHAT_DTYPE=float16`.

**CPU or Apple MPS**: Uses `float32`. The CPU has no reduced-precision hardware, and Apple's Metal Performance Shaders (MPS) backend does not reliably support bfloat16.

The `COMPUTE_DTYPE` constant propagates through the entire codebase. The custom `Linear` layer in `gpt.py` uses it to cast weights on every forward pass:

```python
# nanochat/gpt.py
class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))
```

Master weights are always stored in float32 (for optimizer precision), but computations happen in `COMPUTE_DTYPE`. This is the same mixed-precision technique used in production training, but implemented explicitly rather than through `torch.amp.autocast`.

You can override the detected dtype at any time:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"
```

---

## Step 4: Set the cache directory

nanochat stores downloaded data and checkpoints in a cache directory. The default is `~/.cache/nanochat`. You can override it:

✍️ Run:

```bash
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
```

If you skip this, nanochat will create `~/.cache/nanochat` automatically on first use. Setting it explicitly makes the following commands easier to follow.

---

## Step 5: Download training data and train the tokenizer

This is the first stage of the pipeline: **tokenization**.

### What is a tokenizer and why do we train one?

A tokenizer converts raw text into a sequence of integer IDs that the neural network can process. The mapping is defined by a *vocabulary*: a list of strings (called tokens) along with their IDs.

nanochat trains its own vocabulary from scratch using **Byte-Pair Encoding (BPE)**. The algorithm works by:
1. Starting with a vocabulary of individual bytes (256 tokens).
2. Finding the most frequent pair of adjacent tokens in the training text.
3. Merging that pair into a new single token and adding it to the vocabulary.
4. Repeating until the vocabulary reaches the target size (32,768 tokens for nanochat).

The result: common words like `" the"` or `" and"` become single tokens, while rare sequences are represented as multiple tokens. A well-trained tokenizer compresses text efficiently, which means the model sees more content per forward pass.

nanochat trains its tokenizer on a 2-billion-character sample of internet text (from the FineWeb dataset on HuggingFace). First, download 8 data shards:

✍️ Run (this downloads ~2 GB; takes a few minutes depending on your connection):

```bash
python -m nanochat.dataset -n 8
```

Expected output:

```
2026-03-22 10:00:01 - nanochat.dataset - INFO - Downloading shard 0...
2026-03-22 10:00:14 - nanochat.dataset - INFO - Shard 0: 256 MB
...
2026-03-22 10:03:45 - nanochat.dataset - INFO - Downloaded 8 shards (2.1 GB)
```

The shards are saved as `.zst` compressed files in `~/.cache/nanochat/`. Now train the tokenizer:

✍️ Run (takes ~34 seconds on a modern laptop):

```bash
python -m scripts.tok_train --max-chars=2000000000
```

Expected output:

```
Training tokenizer on 2,000,000,000 characters...
Final vocab size: 32768
Saved tokenizer to ~/.cache/nanochat/tokenizer.model
```

Verify the tokenizer compresses text well:

✍️ Run:

```bash
python -m scripts.tok_eval
```

Expected output (numbers will vary slightly):

```
Compression rate: 4.21 bytes/token
GPT-4 compression: 4.37 bytes/token
```

This tells you that on average, each token in your vocabulary encodes 4.21 bytes of text. Closer to GPT-4's 4.37 is better — it means the model processes more text per step.

> **What's happening:** `tok_train.py` calls `rustbpe` (a Rust-compiled BPE trainer) for speed. Training BPE on 2 billion characters in pure Python would take hours; Rust brings it down to seconds.

---

## Step 6: Pretrain a tiny base model

This is the second pipeline stage: **pretraining**.

### What pretraining does

The neural network (`nanochat/gpt.py`) is a GPT-style Transformer. It starts with random weights and learns by repeatedly:

1. Receiving a sequence of tokens (e.g., 512 tokens from an internet document).
2. Predicting the next token at every position.
3. Computing how wrong it was (the *cross-entropy loss*).
4. Adjusting all its weights slightly to be less wrong next time (backpropagation + optimizer step).

After millions of these steps, the model internalizes statistical patterns of language. This is the most compute-intensive stage — GPT-2 took 168 GPU-hours in 2019 and costs ~$48 today on modern hardware. The run below is intentionally tiny: a 6-layer model trained for 5000 steps, designed to finish in about 30 minutes on a MacBook.

✍️ Run:

```bash
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    --run=dummy
```

**What each flag means:**

| Flag | Value | Why |
|---|---|---|
| `--depth=6` | 6 transformer layers | Full GPT-2 is ~24–26 layers. 6 is tiny but trains quickly. |
| `--head-dim=64` | 64-dimensional attention heads | Smaller heads → smaller model. |
| `--window-pattern=L` | All layers use full context | `L` = Long (full sequence). Simplified for CPU. |
| `--max-seq-len=512` | 512 token context window | Full size is 2048. Shorter = less memory and faster. |
| `--device-batch-size=32` | 32 sequences per step | Reduce to 8 or 4 if you run out of RAM. |
| `--total-batch-size=16384` | 16384 tokens per optimizer step | The effective batch is accumulated across multiple device batches. |
| `--eval-every=100` | Evaluate every 100 steps | Measures validation loss to track learning progress. |
| `--eval-tokens=524288` | ~512K tokens per evaluation | Less than the full validation set for speed. |
| `--core-metric-every=-1` | Disable CORE score | This benchmark requires a GPU; skipped on CPU. |
| `--sample-every=100` | Print a text sample every 100 steps | Shows you what the model generates. |
| `--num-iterations=5000` | 5000 optimizer steps | Enough to see learning; nowhere near convergence. |
| `--run=dummy` | Disable WandB logging | `dummy` is the special name that skips remote logging. |

Expected output (first few lines):

```
Step    0 | loss 10.397 | val_bpb - | lr 6.00e-04 | norm 4.21 | dt 8.23s
Step  100 | loss  7.832 | val_bpb 2.431 | lr 5.98e-04 | norm 1.73 | dt 4.11s

Sample at step 100:
the quick brown fox jumped over the lazy d is is is is is is is is is is

Step  200 | loss  6.241 | val_bpb 1.987 | ...
...
Step 5000 | loss  3.817 | val_bpb 1.612 | ...
```

The `loss` decreases as the model learns. `val_bpb` is *validation bits-per-byte*: a measure of compression quality that is independent of vocabulary size (unlike raw cross-entropy loss). Lower is better; a random model starts near 8 bpb and a GPT-2-grade model reaches ~0.74 bpb.

The text samples will start as incoherent repetition and gradually become more word-like by step 5000, though with a 6-layer model trained for only 5000 steps, don't expect coherent sentences.

> **What's happening:** The training loop in `base_train.py` uses gradient accumulation to achieve the `--total-batch-size` even when `--device-batch-size` is small. If `total_batch_size / device_batch_size = 512` sequences, the code performs 512 forward passes, accumulates their gradients, then runs one optimizer step. This matches the math of a single large batch while keeping memory usage low.

---

## Step 7: Evaluate the base model

✍️ Run:

```bash
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16
```

This runs a quick evaluation on a subset of the validation set (`--split-tokens=16384`) and a sample of benchmark tasks (`--max-per-task=16`). Expect weak results — this tiny model is not meant to score well.

---

## Step 8: Supervised fine-tuning (SFT)

The base model learned to continue text. Now SFT teaches it to follow the User/Assistant conversation format. First, download a small set of labeled conversations:

✍️ Run:

```bash
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
```

Then fine-tune:

✍️ Run (takes ~10 minutes on a MacBook M3):

```bash
python -m scripts.chat_sft \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --run=dummy
```

SFT starts from the pretrained checkpoint and continues training, but now only on conversations structured as:

```
<|user_start|>What is the capital of France?<|user_end|>
<|assistant_start|>The capital of France is Paris.<|assistant_end|>
```

These special tokens (`<|user_start|>`, `<|assistant_start|>`, etc.) are defined in `nanochat/tokenizer.py` and were added to the tokenizer vocabulary before pretraining. After SFT, the model has learned to respond to prompts rather than just continuing text.

---

## Step 9: Talk to your model

Once SFT completes, you can have a conversation:

✍️ Run:

```bash
python -m scripts.chat_cli -p "What is the capital of France?"
```

Expected output (after a tiny underpowered model):

```
Assistant: The capital of France is Paris.
```

Or serve the web UI:

✍️ Run:

```bash
python -m scripts.chat_web
```

Then open your browser to `http://localhost:8000`. You'll see a ChatGPT-style interface connected to your locally trained model.

> **Managing expectations:** With a 6-layer model trained for 5000 steps, responses will be short and often incorrect. The comment in `runcpu.sh` puts it well: "The model should be able to say that it is Paris. It might even know that the color of the sky is blue." That is a reasonable bar for this exercise. The point is to watch the entire pipeline run, not to produce a capable model.

---

## Running the full pipeline in one shot

All of steps 5–8 above are captured in `runs/runcpu.sh`. Once you have `uv` and the repo, you can run the entire thing with:

✍️ Run:

```bash
bash runs/runcpu.sh
```

Reading that script is a good exercise — it is short (66 lines including comments) and shows exactly how the stages chain together.

---

## Checkpoint: what just happened?

Let's map the commands you ran back to the pipeline diagram:

```
python -m nanochat.dataset -n 8     →  Downloaded raw training data
python -m scripts.tok_train         →  Stage 1: Tokenizer (vocabulary)
python -m scripts.tok_eval          →  Measured tokenizer quality
python -m scripts.base_train        →  Stage 2: Pretraining (base model)
python -m scripts.base_eval         →  Measured base model quality
python -m scripts.chat_sft          →  Stage 3: SFT (chat format)
python -m scripts.chat_cli          →  Stage 6: Inference
```

Stages 4 (RL) and 5 (evaluation) are part of the full pipeline but skipped in the CPU run because they require more compute or a GPU to be meaningful.

---

## Check your understanding

1. **Device detection:** Open `nanochat/common.py` and find the `_detect_compute_dtype` function. What dtype would be selected on a machine with a V100 GPU (CUDA compute capability 7.0)? Why doesn't it use bfloat16?

2. **Batch math:** The CPU training run uses `--device-batch-size=32` and `--total-batch-size=16384` with a sequence length of 512. How many forward passes does the training loop perform before each optimizer step? (Hint: `total_batch_size` is measured in tokens, not sequences.)

3. **Tokenizer purpose:** Why does nanochat train its own tokenizer instead of reusing the GPT-4 tokenizer directly? What property of the trained vocabulary does `tok_eval` measure, and what unit is it reported in?

---

## What's next

In Chapter 2, you'll open `nanochat/gpt.py` and understand exactly how the GPT Transformer is built — from token embeddings through attention layers to the final prediction — so that when you train it, you know what every parameter is doing.
