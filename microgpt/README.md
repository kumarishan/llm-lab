# Tutorial: microgpt.py

> "The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
> This file is the complete algorithm. Everything else is just efficiency."
> — @karpathy

A hands-on, chapter-by-chapter guide to understanding every line of `microgpt.py` — a fully working GPT built from scratch in ~200 lines of pure Python.

**Estimated time:** 4–6 hours
**Prerequisites:** Intermediate Python (list comprehensions, classes, closures). No ML background required.
**External dependencies:** None — the file uses only `os`, `math`, `random`, and `urllib`.

---

## Chapters

| # | Chapter | What you'll learn |
|---|---------|-------------------|
| 1 | [Setup and First Run](./01-setup-and-first-run.md) | Run the file, read the output, get a mental map of all 8 chapters |
| 2 | [Dataset and Tokenization](./02-dataset-and-tokenization.md) | Character-level vocab, integer token IDs, the BOS token trick, sliding-window targets |
| 3 | [Autograd from Scratch](./03-autograd-from-scratch.md) | The `Value` class, computation graphs, local gradients, reverse-mode autodiff |
| 4 | [Model Architecture Foundations](./04-model-architecture-foundations.md) | Parameter initialization, `linear()`, numerically stable `softmax()`, `rmsnorm()` |
| 5 | [Multi-Head Self-Attention](./05-multi-head-self-attention.md) | Q/K/V projections, scaled dot-product attention, KV cache, causal masking |
| 6 | [The Full GPT Forward Pass](./06-full-gpt-forward-pass.md) | MLP block, residual stream, complete token→logits data flow |
| 7 | [Training — Loss, Backprop, and Adam](./07-training-loss-backprop-adam.md) | Cross-entropy loss, `loss.backward()` through the full graph, Adam optimizer |
| 8 | [Inference and Sampling](./08-inference-and-sampling.md) | Autoregressive generation, temperature scaling, sampling, what's next |

---

## How to use this tutorial

Follow the chapters in order — each one builds on the last. Don't just read; **do the hands-on steps**. Every chapter has exercises you can run in a Python REPL or by modifying the file directly.

By the end you will have read, traced, and experimented with every line of a working GPT — from the tokenizer through the optimizer to the name generator.

---

## Quick Start

```bash
# No pip install needed — pure Python stdlib only
python microgpt.py
```

Expected output:
```
num docs: 32033
vocab size: 28
num params: <N>
step 1000 / 1000 | loss 2.XXXX
--- inference (new, hallucinated names) ---
sample  1: Ava
sample  2: Emilia
...
```

Start with [Chapter 1 →](./01-setup-and-first-run.md)

---

## What makes this file remarkable

- **~200 lines** of pure Python, zero external libraries
- **Complete GPT** — tokenizer, autograd engine, transformer architecture, Adam optimizer, inference
- **Single file** — the entire algorithm lives in `microgpt.py`
- **Trains in minutes** on CPU, generating plausible names after 1000 steps

> "Everything else is just efficiency." — and now you'll understand what that efficiency is optimizing.
