```
  _     _     __  __   _          _         
 | |   | |   |  \/  | | |    __ _| |__  ___ 
 | |   | |   | |\/| | | |   / _` | '_ \/ __|
 | |___| |___| |  | | | |__| (_| | |_) \__ \
 |_____|_____|_|  |_| |_____\__,_|_.__/|___/
                                                
```

Made with ❤️ by **Kumar Ishan** — with a lot of help from ✨ 🤖 So please be kind 😍.

This repo holds **two tutorial tracks** that follow code from [**Andrej Karpathy**](https://gist.github.com/karpathy): a minimal **[microgpt](https://karpathy.github.io/2026/02/12/microgpt/)** walkthrough (single-file GPT, no ML frameworks) and a **[nanochat](https://github.com/karpathy/nanochat)** walkthrough (full tokenizer → pretrain → SFT → RL → eval → inference → web UI). Each chapter is reading-first: you trace real code, run hands-on steps, and build intuition for *why* each piece exists.

The material here was generated with the [**repo-tutorial** skill](https://github.com/kumarishan/kistack) from [**kumarishan/kistack**](https://github.com/kumarishan/kistack) — structured, chapter-by-chapter guides derived from the repositories themselves.

---

## [microgpt](https://karpathy.github.io/2026/02/12/microgpt/) — `microgpt.py`

*“The most atomic way to train and run inference for a GPT in pure, dependency-free Python.”*


| Chapter                                                                                      | What your reading covers                                                                     |
| -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| [Chapter 1: Setup and First Run](microgpt/01-setup-and-first-run.md)                         | Running the file with zero deps, reading train/inference output, a map of the whole pipeline |
| [Chapter 2: Dataset and Tokenization](microgpt/02-dataset-and-tokenization.md)               | Character vocab, token IDs, BOS trick, sliding-window targets                                |
| [Chapter 3: Autograd from Scratch](microgpt/03-autograd-from-scratch.md)                     | The `Value` type, computation graphs, local grads, reverse-mode autodiff                     |
| [Chapter 4: Model Architecture Foundations](microgpt/04-model-architecture-foundations.md)   | Init, `linear()`, stable `softmax()`, `rmsnorm()`                                            |
| [Chapter 5: Multi-Head Self-Attention](microgpt/05-multi-head-self-attention.md)             | Q/K/V, scaled dot-product attention, KV cache, causal masking                                |
| [Chapter 6: The Full GPT Forward Pass](microgpt/06-full-gpt-forward-pass.md)                 | MLP block, residual stream, token → logits                                                   |
| [Chapter 7: Training — Loss, Backprop, and Adam](microgpt/07-training-loss-backprop-adam.md) | Cross-entropy, `backward()` through the graph, Adam                                          |
| [Chapter 8: Inference and Sampling](microgpt/08-inference-and-sampling.md)                   | Autoregressive generation, temperature, sampling, what to try next                           |


**Start here:** [microgpt/README.md](microgpt/README.md)

---

## [nanochat](https://github.com/karpathy/nanochat) — full LLM stack

A path from **BPE and architecture** through **distributed pretraining**, **SFT**, **RL (GRPO)**, **evaluation**, **fast inference**, and a **streaming chat** server.


| Chapter | What your reading covers |
| --- | --- |
| [Chapter 1: Setup and First Run](nanochat/01-setup-and-first-run.md) | `uv`, project layout, device selection, tiny CPU end-to-end run, full pipeline overview |
| [Chapter 2: LLM Fundamentals — Tokens, Embeddings, and Attention](nanochat/02-llm-fundamentals.md) | Autoregressive LM, embeddings, self-attention, transformer blocks, cross-entropy |
| [Chapter 3: Tokenization — Training BPE from Scratch](nanochat/03-tokenization.md) | BPE, split patterns, RustBPE / tiktoken, specials, chat tokens, loss masking |
| [Chapter 4: The GPT Architecture — Attention, RoPE, GQA, and Modern Innovations](nanochat/04-gpt-architecture.md) | RoPE, QK-norm, GQA, sliding windows, FA3, smearing, scalars, logit softcap |
| [Chapter 5: Data Pipeline — Loading, Packing, and Distributing Training Data](nanochat/05-data-pipeline.md) | Parquet / mix data, BOS-aligned cropping, DDP sharding, resumable state |
| [Chapter 6: Pretraining — The Training Loop, Optimizers, and Mixed Precision](nanochat/06-pretraining.md) | Training loop, CE + BPB, AdamW, Muon / Polar Express, AMP, accumulation, LR schedule |
| [Chapter 7: Distributed Training — Multi-GPU with DDP](nanochat/07-distributed-training.md) | Data parallel, process groups, `torchrun`, `all_reduce`, checkpoints, MFU |
| [Chapter 8: Evaluation — Bits-Per-Byte, CORE, and In-Context Learning](nanochat/08-evaluation.md) | Bits-per-byte, in-context learning, CORE / MMLU-style evals, running eval scripts |
| [Chapter 9: Supervised Fine-Tuning — Teaching the Model to Chat](nanochat/09-supervised-finetuning.md) | Chat templates, assistant-only loss, benchmarks, mixtures, forgetting |
| [Chapter 10: Reinforcement Learning — GRPO, Rewards, and Tool Use](nanochat/10-reinforcement-learning.md) | Policy gradients, REINFORCE, GRPO/DAPO, rewards, tool use (e.g. Python sandbox) |
| [Chapter 11: Inference — KV Cache, Flash Attention, and Sampling](nanochat/11-inference.md) | KV cache, prefill vs decode, Flash vs SDPA, temperature / top-k, streaming |
| [Chapter 12: Chat Interfaces — CLI and Web Server](nanochat/12-chat-interfaces.md) | CLI state, FastAPI, SSE, OpenAI-style API, async workers, multi-GPU serve |
| [Chapter 13: Scaling Laws, Compute-Optimal Training, and Going Further](nanochat/13-scaling-laws-and-going-further.md) | Chinchilla-style scaling, compute-optimal training, depth dial, MFU, FP8, where to go next |


**Start here:** [nanochat/README.md](nanochat/README.md)

---

## How to use this repo

1. Pick **[microgpt](https://karpathy.github.io/2026/02/12/microgpt/)** if you want every idea visible in ~200 lines of Python, and then head over to **[nanochat](https://github.com/karpathy/nanochat)** when you want the full production-shaped stack.
2. Open the track **README** for time estimates and prerequisites, then follow chapters **in order**.
3. Do the **hands-on steps** in each chapter — the goal is not skimming but tracing and running the real code paths.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please **open an issue first** to discuss improvements, vetting, or ideas for **other repos** that could use a similar tutorial.

## License

[MIT](LICENSE)
