# Tutorial: nanochat

A hands-on guide to building, training, and running your own large language model from scratch — covering tokenization, pretraining, fine-tuning, reinforcement learning, evaluation, inference, and a chat web UI.

**Estimated time:** 12–15 hours
**Prerequisites:** Python proficiency. No prior ML or LLM experience required.

---

## What you'll build

By the end of this tutorial you will have:

- Trained a BPE tokenizer from scratch on 2B characters of text
- Pretrained a GPT-class transformer on billions of tokens (CPU or GPU)
- Understood every architectural innovation in nanochat's GPT: RoPE, GQA, sliding window attention, smearing, value embeddings, Muon optimizer, and more
- Fine-tuned the base model into a conversational assistant using SFT
- Applied reinforcement learning (GRPO) to improve math reasoning with tool use
- Evaluated your model with Bits-Per-Byte and the CORE benchmark
- Run efficient inference with a KV cache and Flash Attention
- Deployed a streaming web chat UI served across multiple GPUs

---

## Chapters

| # | Chapter | What you'll learn |
|---|---------|-------------------|
| 1 | [Setup and First Run](./01-setup-and-first-run.md) | Install dependencies, run a tiny CPU model, tour the full pipeline |
| 2 | [LLM Fundamentals — Tokens, Embeddings, and Attention](./02-llm-fundamentals.md) | Autoregressive generation, token embeddings, self-attention math, transformer blocks, cross-entropy loss |
| 3 | [Tokenization — Training BPE from Scratch](./03-tokenization.md) | BPE algorithm, GPT-4 split pattern, RustBPE + tiktoken, special tokens, chat tokenization, loss masking |
| 4 | [The GPT Architecture — Attention, RoPE, GQA, and Modern Innovations](./04-gpt-architecture.md) | Rotary embeddings (RoPE), QK-norm, grouped-query attention, sliding windows, FA3, smearing, backout, per-layer scalars, logit softcap |
| 5 | [Data Pipeline — Loading, Packing, and Distributing Training Data](./05-data-pipeline.md) | Parquet + ClimbMix-400B, BOS-aligned best-fit cropping, DDP sharding, resumable state |
| 6 | [Pretraining — The Training Loop, Optimizers, and Mixed Precision](./06-pretraining.md) | Training loop, cross-entropy + BPB, AdamW math, Muon + Polar Express, mixed precision, gradient accumulation, LR schedule |
| 7 | [Distributed Training — Multi-GPU with DDP](./07-distributed-training.md) | Data parallelism, process groups, torchrun, all_reduce, checkpointing, MFU |
| 8 | [Evaluation — Bits-Per-Byte, CORE, and In-Context Learning](./08-evaluation.md) | Bits-per-byte, in-context learning, CORE benchmark (SQuAD, HellaSwag, WinoGrande, LAMBADA), MMLU, running base_eval.py |
| 9 | [Supervised Fine-Tuning — Teaching the Model to Chat](./09-supervised-finetuning.md) | Chat templates, loss masking on assistant tokens, MMLU, GSM8K, TaskMixture, ChatCORE, catastrophic forgetting |
| 10 | [Reinforcement Learning — GRPO, Rewards, and Tool Use](./10-reinforcement-learning.md) | Policy gradient theorem, REINFORCE, GRPO/DAPO normalization, GSM8K rewards, sandboxed Python execution |
| 11 | [Inference — KV Cache, Flash Attention, and Sampling](./11-inference.md) | KV cache math and code, prefill vs decode phases, FA3 vs SDPA, temperature + top-k sampling, streaming |
| 12 | [Chat Interfaces — CLI and Web Server](./12-chat-interfaces.md) | chat_cli.py multi-turn state, FastAPI, SSE streaming, OpenAI-compatible API, async worker pool, multi-GPU serving |
| 13 | [Scaling Laws, Compute-Optimal Training, and Going Further](./13-scaling-laws-and-going-further.md) | Chinchilla scaling laws, compute-optimal training, the `--depth` single dial, MFU, FP8, end-to-end runs, what to try next |

---

## How to use this tutorial

Follow chapters in order. Each chapter has:

- **Concept explanations** with math where needed
- **✍️ hands-on steps** — code blocks you type and run
- **"What's happening" callouts** — explaining *why*, not just *what*
- **"Check your understanding" questions** at the end

Don't just read — do the hands-on steps. The goal is to have real intuition for every line of nanochat's code by the end.

---

## Quick Start

```bash
# Install uv (package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install (CPU)
git clone https://github.com/your-org/nanochat
cd nanochat
uv sync --extra cpu

# Run a tiny model on CPU (see Chapter 1 for full explanation)
bash runs/runcpu.sh
```

Then start at [Chapter 1](./01-setup-and-first-run.md).

---

## Hardware guide

| Goal | Hardware | Time |
|------|----------|------|
| Follow along, test concepts | CPU (any laptop) | Minutes per run |
| Train a real tiny model | Single GPU (8GB+) | Hours |
| Train GPT-2 class model | 8× H100 | 2–3 hours, ~$48 |

All chapters include CPU-compatible commands. GPU commands are clearly marked.
