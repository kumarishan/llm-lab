# Chapter 13: Scaling Laws, Compute-Optimal Training, and Going Further

You have reached the final chapter. You trained a language model from raw bytes, aligned it through
SFT and reinforcement learning, deployed it behind a web UI, and watched it generate text. Now we
step back and ask: what forces govern how good that model can become, and what single knob in
nanochat turns all of those forces at once?

---

## What you will learn

- Why model performance follows **power laws** as you scale compute, and what the Chinchilla paper
  changed about how the field thinks about the tradeoff between parameters and training tokens.
- How to derive the number of parameters and tokens to train on from a fixed compute budget.
- How nanochat's `--depth` parameter automatically places every model on the compute-optimal
  frontier by deriving all other hyperparameters from that single integer.
- How **Model FLOPS Utilization (MFU)** is calculated and what numbers to expect in practice.
- Concrete next experiments to run and resources to go deeper.

## Prerequisites

Chapters 1 through 12. You should have already run `runs/speedrun.sh` or at least walked through
it, and have a trained checkpoint in `~/.cache/nanochat/base_checkpoints/`.

---

## 13.1 Looking Back: What You Built

Before thinking about scale, it is worth naming what the pipeline you built actually does.

```
raw text bytes
    │
    ▼ scripts/tok_train.py
BPE tokenizer  (vocab = 32,768 tokens)
    │
    ▼ nanochat/dataloader.py
token IDs packed into fixed-length windows  (T = 2,048)
    │
    ▼ nanochat/gpt.py  (GPTConfig, GPT)
embeddings → rotary attention → MLP → residual stream  (× n_layer)
    │
    ▼  logit projection  (n_embd → vocab)
    │
    ▼ scripts/base_train.py
cross-entropy loss → Muon + AdamW → weight update  (× num_iterations)
    │
    ▼ scripts/chat_sft.py
SFT: teach conversation format, tools, multiple-choice
    │
    ▼ scripts/chat_rl.py
RL: REINFORCE on verifiable reward signals (math, code, spelling)
    │
    ▼ scripts/chat_web.py
HTTP server + nanochat/ui.html  →  ChatGPT-style browser UI
```

Every stage has been covered in the previous twelve chapters. The entire pipeline — from a blank
machine to a running chat assistant — lives in under 4,000 lines of Python and a handful of shell
scripts.

**Historical note.** When OpenAI trained the original GPT-2 (1.5 B parameters) in 2019, the
compute cost was approximately $43,000 and the wall-clock time was measured in weeks. The nanochat
equivalent runs in roughly two to three hours on an eight-GPU H100 node for about $48 on-demand,
or closer to $15 on a spot instance. The model quality target — a DCLM CORE score above 0.256525,
matching GPT-2 — is now achievable by anyone with a cloud account and an afternoon. That gap
closed because of better data curation, better architectures (GQA, rotary embeddings, Flash
Attention 3), better optimizers (Muon), and much faster hardware. The model itself is not
fundamentally smarter; the *infrastructure* improved by roughly three orders of magnitude.

---

## 13.2 What Are Scaling Laws?

### The empirical observation

If you train a transformer language model to convergence at many different sizes, a clean
relationship emerges: validation loss falls smoothly as you spend more compute. The curve is not
linear; it is a **power law**.

Kaplan et al. (2020, "Scaling Laws for Neural Language Models") fit the following relationship for
models trained on enough data:

```
L ≈ (N_c / N)^α_N + (D_c / D)^α_D
```

where:
- `L` is the validation cross-entropy loss (nats per token)
- `N` is the number of non-embedding parameters
- `D` is the number of training tokens
- `N_c ≈ 8.8 × 10^13`, `D_c ≈ 5.4 × 10^13` are fitted constants
- `α_N ≈ 0.076`, `α_D ≈ 0.095` are the power-law exponents

Each term decays independently. Double the parameters and the parameter term drops by `2^0.076 ≈
1.054` — a 5% improvement. Double the tokens and the data term drops by `2^0.095 ≈ 1.067` — a
7% improvement. There is also a third term `(C_c / C)^α_C` for the irreducible effect of finite
compute, but for our purposes the two-term form is sufficient.

The key insight from Kaplan et al. was that **if compute is fixed**, you should prefer making the
model larger over training it longer, because loss falls faster with parameters than with tokens
(the α exponents differ). This led to the practice of training very large models on relatively few
tokens — GPT-3 (175 B parameters) trained on only 300 B tokens.

### The Chinchilla correction

In 2022, Hoffmann et al. at DeepMind published "Training Compute-Optimal Large Language Models"
(colloquially, the **Chinchilla paper**). They reran the scaling study more carefully, with models
that were allowed to train to their optimal point rather than being cut short. Their finding
contradicted Kaplan: the exponents for parameters and tokens are nearly **equal**.

The Chinchilla result:

```
Optimal N ∝ C^0.5
Optimal D ∝ C^0.5
```

In other words, as you double your compute budget `C`, you should simultaneously double the
number of parameters *and* double the number of training tokens. The ratio `D / N` should stay
roughly constant — Chinchilla estimated the optimal ratio at about 20 tokens per parameter.

GPT-3's 300 B tokens for 175 B parameters gives a ratio of about 1.7, far below the Chinchilla
optimum of 20. GPT-3 was, in Chinchilla's framing, "over-parameterized and under-trained." A
smaller model trained on far more data would have achieved the same loss for the same compute
spend.

### The three variables

Any discussion of scaling reduces to three quantities:

| Symbol | What it measures |
|--------|-----------------|
| `C` | Compute budget (floating-point operations, FLOPs) |
| `N` | Model parameters |
| `D` | Training tokens |

These are linked by the approximation:

```
C ≈ 6 · N · D
```

The factor of 6 comes from: 2 FLOPs per multiply-accumulate, multiplied by 3 because backward
pass is roughly twice the cost of the forward pass (2 forward + 4 backward = 6 total). This is
exact for pure linear layers; the attention softmax and embeddings add a few percent, which is why
nanochat's `estimate_flops()` method adds a separate attention term on top of the `6N` term.

---

## 13.3 Compute Budget and the Param-Data Ratio

### Given a budget, how many parameters and tokens?

Start from the Chinchilla equal-scaling rule and the `C = 6ND` identity.

If you want to spend `C` FLOPs and keep `D = r · N` (where `r` is the target tokens-per-parameter
ratio):

```
C = 6 · N · (r · N) = 6 · r · N²
=> N = sqrt(C / (6r))
=> D = r · N = r · sqrt(C / (6r)) = sqrt(r · C / 6)
```

Example: you have a budget of `C = 4 × 10^19` FLOPs (the approximate speedrun budget) and you
want `r = 10.5` tokens per parameter (nanochat's default).

```
N = sqrt(4e19 / (6 × 10.5)) = sqrt(4e19 / 63) ≈ sqrt(6.35e17) ≈ 7.97 × 10^8 ≈ 800 M params
D = 10.5 × 8e8 ≈ 8.4 × 10^9 ≈ 8.4 B tokens
```

An ~800 M parameter model trained on ~8 B tokens for ~4 × 10^19 FLOPs. That is roughly
consistent with nanochat's `depth=24` speedrun.

### Why 10.5 instead of Chinchilla's 20?

nanochat uses `--target-param-data-ratio=10.5` by default, rather than the Chinchilla value of
20. The reasoning is **inference cost**. At serving time you pay for every forward pass, so a
smaller model is cheaper per query. If you train a smaller model on more tokens than Chinchilla
says is optimal for a given compute budget, you end up with a model that performs comparably to
the Chinchilla-optimal model but requires less memory and less compute per token at inference. This
is the philosophy behind Meta's LLaMA series: deliberately train smaller models for longer
(higher `r`) to produce inference-efficient checkpoints.

The value 10.5 was calibrated empirically by the nanochat project through the scaling laws
experiments in `runs/scaling_laws.sh` — it is the ratio where the val_bpb vs. compute curve for
the nanochat architecture is minimized (see `dev/LOG.md Jan 27, 2026` referenced in the code).

---

## 13.4 nanochat's Single `--depth` Dial

### The hyperparameter explosion problem

Training a transformer means choosing: number of layers, embedding dimension, number of attention
heads, number of KV heads, sequence length, batch size, learning rate, weight decay, warmup
schedule, warmdown schedule... and all of these should change together as you scale the model up
or down. Getting this wrong means you leave performance on the table: too wide a model for its
depth fails to generalize well; too many heads relative to the KV heads wastes memory.

### One number to rule them all

nanochat solves this by making `--depth` the single free variable. Every other architectural
hyperparameter is derived from `depth` using fixed ratios discovered empirically. From
`scripts/base_train.py`:

```python
def build_model_meta(depth):
    base_dim = depth * args.aspect_ratio          # args.aspect_ratio = 64 (default)
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim        # args.head_dim = 128 (default)
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
```

Breaking this down:

| Hyperparameter | Formula | Why |
|----------------|---------|-----|
| `n_embd` (model dim) | `depth × 64`, rounded to nearest `head_dim` multiple | Fixed **aspect ratio** keeps model "square" — depth and width grow together. A too-wide/shallow or too-narrow/deep model is suboptimal. |
| `n_head` (attention heads) | `model_dim // 128` | Fixed **head dimension** of 128 — empirically optimal for Flash Attention and matches LLaMA/Mistral conventions. |
| `n_kv_head` (KV heads) | equals `n_head` here; `window_pattern` controls which layers use sliding window | **Grouped Query Attention (GQA)** reduces KV cache memory at inference. The speedrun uses `n_kv_head = n_head // 4` implicitly through training config. |
| `n_layer` | equals `depth` | Direct — the `--depth` argument *is* the number of transformer blocks. |

The training horizon is then derived automatically in the same script:

```python
# num_scaling_params = transformer_matrices + lm_head (cleanest fit to scaling laws)
target_tokens = int(args.target_param_data_ratio * num_scaling_params)
num_iterations = target_tokens // total_batch_size
```

And the batch size scales with the training horizon via the Power Lines paper (Bopt ∝ D^0.383):

```python
batch_size_ratio = target_tokens / D_REF    # relative to reference d12
predicted_batch_size = B_REF * batch_size_ratio ** 0.383
total_batch_size = 2 ** round(math.log2(predicted_batch_size))  # snap to power of 2
```

Learning rates and weight decay scale with batch size in the same block. The result: you set
`--depth=24`, and everything else — architecture, training horizon, batch size, learning rates,
weight decay — is computed to be near-optimal for that scale.

### The depth ladder

Here is an approximate mapping from depth to model size and compute, using the nanochat
conventions (`aspect_ratio=64`, `head_dim=128`, `target_param_data_ratio=10.5`):

| depth | `n_embd` | `n_layer` | Approx. params (transformer + lm_head) | Approx. training tokens | Approx. compute (FLOPs) |
|-------|---------|-----------|----------------------------------------|------------------------|------------------------|
| 6     | 384     | 6         | ~12 M                                  | ~125 M                 | ~9 × 10^15             |
| 8     | 512     | 8         | ~24 M                                  | ~250 M                 | ~3 × 10^16             |
| 10    | 640     | 10        | ~43 M                                  | ~450 M                 | ~8 × 10^16             |
| 12    | 768     | 12        | ~72 M                                  | ~760 M                 | ~2 × 10^17             |
| 14    | 896     | 14        | ~112 M                                 | ~1.2 B                 | ~4 × 10^17             |
| 16    | 1,024   | 16        | ~166 M                                 | ~1.7 B                 | ~9 × 10^17             |
| 18    | 1,152   | 18        | ~235 M                                 | ~2.5 B                 | ~2 × 10^18             |
| 20    | 1,280   | 20        | ~322 M                                 | ~3.4 B                 | ~4 × 10^18             |
| 22    | 1,408   | 22        | ~430 M                                 | ~4.5 B                 | ~8 × 10^18             |
| 24    | 1,536   | 24        | ~560 M                                 | ~5.9 B                 | ~1.5 × 10^19           |
| 26    | 1,664   | 26        | ~715 M                                 | ~7.5 B                 | ~3 × 10^19             |

*Exact counts depend on vocab size and model config; use `python -m scripts.base_train --depth=N
--num-iterations=0` to print precise figures for any depth.*

Because every model on this table was derived from the same set of scaling-law-calibrated
relationships, each one is near compute-optimal. A sweep across all these depths at the same
compute budget is called a **scaling laws experiment**: you train each depth until the same number
of FLOPs has been spent, then plot loss vs. depth. The minimum of that curve tells you which model
size extracts the most value from your hardware.

This is exactly what `runs/scaling_laws.sh` does.

---

## 13.5 Model FLOPS Utilization (MFU)

### What MFU measures

A GPU has a **peak theoretical throughput** — the maximum number of floating-point operations it
can perform per second if every cycle is spent on compute. In practice, real workloads are slower
because memory bandwidth, kernel launch overhead, and synchronization all steal time. MFU measures
how much of that theoretical peak you are actually using:

```
MFU = (actual FLOPs per second) / (peak FLOPs per second)
```

From `scripts/base_train.py`, nanochat computes it every training step:

```python
tok_per_sec = int(total_batch_size / dt)
flops_per_sec = num_flops_per_token * total_batch_size / dt
mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
```

`num_flops_per_token` comes from `model.estimate_flops()`, which counts `6 × matmul_params +
attention_flops` per token. `gpu_peak_flops` comes from the lookup table in `nanochat/common.py`.

### The peak FLOPS table

```python
# From nanochat/common.py (BF16 peak, representative entries)
(["h100"],      989e12),   # H100 SXM: 989 TFLOPS BF16
(["h100", "pcie"], 756e12), # H100 PCIe: 756 TFLOPS BF16
(["a100"],      312e12),   # A100: 312 TFLOPS BF16
(["l40s"],      362e12),   # L40S: 362 TFLOPS BF16
(["4090"],      165.2e12), # RTX 4090: 165 TFLOPS BF16
(["b200"],      2.25e15),  # B200: 2,250 TFLOPS BF16
```

Eight H100 SXMs give a combined peak of `8 × 989 × 10^12 ≈ 7.9 × 10^15` FLOPs/sec. The
nanochat speedrun processes roughly 3–4 × 10^15 FLOPs/sec of actual model compute, giving an MFU
of around 40–50%.

### Interpreting MFU

| MFU range | Interpretation |
|-----------|---------------|
| < 20%     | Something is wrong: batch too small, poor kernel choice, or heavy CPU overhead. |
| 20–40%    | Typical for smaller models or non-optimized setups. Memory bandwidth bound. |
| 40–60%    | Good. The nanochat speedrun lands here with FP8 + Flash Attention 3. |
| > 60%     | Excellent. Usually requires carefully tuned batching and large matmuls. |
| ~100%     | Theoretical maximum; never achieved in practice. |

The reason MFU rarely exceeds 60% is that **matrix multiplications are memory-bandwidth bound** at
typical LLM sizes. The GPU has to read weights from HBM (high-bandwidth memory) for every matmul,
and HBM bandwidth is a smaller bottleneck than raw FP operations. Flash Attention 3 helps by
fusing the Q/K/V matmuls and the softmax into a single kernel that reads each weight block only
once.

MFU is logged to wandb as `train/mfu` and printed to stdout every step. If you are getting below
30% MFU on an H100, investigate: try increasing `--device-batch-size`, check that Flash Attention
3 is enabled (look for the `✓ Using Flash Attention 3` line in the output), and verify that
`COMPUTE_DTYPE` is `bfloat16`.

---

## 13.6 FP8: Squeezing More Out of H100s

### The idea

BF16 uses 16 bits per number. FP8 uses 8 bits per number. Halving the bit width roughly doubles
the throughput of matrix multiplications on H100+ hardware, because the cuBLAS FP8 kernel
(`torch._scaled_mm`) can pack twice as many operands into each tensor core cycle.

The challenge with 8-bit floats is their small dynamic range. `float8_e4m3fn` (4-bit exponent,
3-bit mantissa) can only represent values in roughly `[-448, 448]`. Activations and gradients
routinely exceed this range. The solution is **dynamic scaling**: before each matmul, compute
`scale = FP8_MAX / max(|tensor|)`, multiply the tensor by this scale to bring it into FP8 range,
cast to FP8, run the matmul, and let `torch._scaled_mm` divide by the scale internally.

### What `nanochat/fp8.py` does

The entire FP8 implementation is about 150 lines. It provides:

- `_to_fp8(x, fp8_dtype)`: tensorwise dynamic quantization — one scalar scale per tensor.
- `_Float8Matmul`: a custom `torch.autograd.Function` that runs all three GEMMs of a linear layer
  (forward `output = input @ weight.T`, and two backward passes) in FP8 using
  `torch._scaled_mm`.
- `Float8Linear`: a drop-in `nn.Linear` replacement that calls `_Float8Matmul.apply`.
- `convert_to_float8_training`: walks the model tree and swaps eligible `nn.Linear` layers with
  `Float8Linear`. Layers whose dimensions are not divisible by 16 (a hardware requirement) or
  that are too small (< 128 features) are skipped.

The two FP8 formats serve different roles:

| Format | Bits | Max value | Used for |
|--------|------|-----------|---------|
| `float8_e4m3fn` | 4E + 3M | 448 | inputs and weights (higher precision needed) |
| `float8_e5m2`   | 5E + 2M | 57,344 | gradients (wider range needed) |

### When to use FP8

FP8 requires:
1. An H100 or newer GPU (or AMD MI300X/MI325 equivalent).
2. `COMPUTE_DTYPE = bfloat16` (FP8 quantizes from BF16, not from FP32).
3. The `--fp8` flag in `base_train.py`.

Enable it like this:

```bash
# ✍️
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --fp8 \
    --fp8-recipe=tensorwise
```

The `--fp8-recipe=tensorwise` is the default and the faster option. The `rowwise` recipe (one
scale per row rather than one per tensor) is more numerically accurate but requires the full
`torchao` library and is slower.

With FP8 enabled on an H100, MFU typically improves by roughly 10–15 percentage points. The
speedrun entry #2 in the README (`d26 slightly undertrained + fp8`) was the first entry that used
FP8, shaving the speedrun time from 3.04 hours to 2.91 hours.

---

## 13.7 The `runs/` Scripts: End-to-End Pipelines

The `runs/` directory contains four reference pipelines. Each is a self-contained bash script
that documents one use case.

### `runs/runcpu.sh` — CPU / Apple Silicon

Trains a `depth=6`, 30-minute demo model on CPU or MPS. Not useful for real research, but useful
for verifying that the codebase runs correctly on a laptop before committing to a GPU job.

```bash
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --total-batch-size=16384 \
    --num-iterations=5000
```

Notable differences from a GPU run: `--window-pattern=L` (full context, no sliding window — the
SDPA fallback used on CPU does not support sliding windows), `--head-dim=64` (smaller heads fit
in CPU cache), and an explicit `--num-iterations` override because the Chinchilla-optimal horizon
for a d6 model would take too long on a laptop.

### `runs/speedrun.sh` — Train to GPT-2 capability

The canonical end-to-end pipeline. It:

1. Downloads ~170 shards of the FineWebEdu/ClimbMix pretraining corpus (in the background while
   the tokenizer trains).
2. Trains a BPE tokenizer on 2 B characters.
3. Runs the pretraining loop on `depth=24`, `--target-param-data-ratio=8` (slightly undertrained
   for compute efficiency), `--fp8`, 8 GPUs.
4. Downloads the identity conversations synthetic data and runs SFT.
5. Generates a markdown report.

The key line:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size=16 \
    --fp8 \
    --run=$WANDB_RUN
```

`--target-param-data-ratio=8` is below the default 10.5 because the speedrun prioritizes hitting
the GPT-2 CORE target quickly — slightly fewer tokens means slightly fewer iterations and faster
wall-clock time, at the cost of a marginally higher loss.

### `runs/miniseries.sh` — Sweep across depths

Trains `depth ∈ {12, 14, 16, 18, 20, 22, 24, 26}` sequentially, each at its compute-optimal
horizon. Results are written to a CSV:

```
depth, model_dim, num_params, num_scaling_params, num_iterations,
tokens_trained, param_data_ratio, val_bpb, core_score, train_time_sec
```

This produces the **nanochat miniseries**: a family of compute-optimal models at different scales
that you can compare by plotting `val_bpb` vs. `log(params)`.

### `runs/scaling_laws.sh` — Characterize scaling

Trains a grid of `(flops_budget, depth)` pairs where `flops_budget ∈ {1e18, 2.15e18, 4.64e18, 1e19}`
and `depth ∈ {8, 10, 12, 14, 16, 18, 20}`. For each pair, `--target-flops` overrides the
training horizon so that exactly that many FLOPs are spent regardless of model size.

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=$d \
    --target-flops=$flops \
    --target-param-data-ratio=-1 \
    ...
```

The results allow you to fit the power-law constants for your specific architecture and dataset,
and to identify the compute-optimal depth for any budget. This is how the 10.5 default for
`--target-param-data-ratio` was derived.

---

## 13.8 Hands-On: Run the Miniseries (or a Scaled-Down Version)

If you are on a GPU node with time to spare, the miniseries is the most instructive experiment you
can run.

```bash
# ✍️  Full miniseries on 8×H100 (~8 hours total)
bash runs/miniseries.sh my_series
```

If you are resource-constrained, run just two depths at a fixed small budget to see scaling in
action:

```bash
# ✍️  Quick scaling comparison: d12 vs d20, same 2×10^17 FLOPs each (~10 min on 8×H100)
export OMP_NUM_THREADS=1
source .venv/bin/activate

for depth in 12 20; do
    torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
        --depth=$depth \
        --target-flops=2e17 \
        --target-param-data-ratio=-1 \
        --run="scaling_demo_d${depth}" \
        --model-tag="scaling_demo_d${depth}" \
        --core-metric-every=999999 \
        --sample-every=-1 \
        --save-every=-1
done
```

You should see that d12 achieves a lower `val_bpb` than d20 at this budget, because d20 is
"underfit" — the 2×10^17 FLOP budget is less than the compute-optimal horizon for a model that
large. The crossover point — where d20 starts beating d12 — occurs around 4–8 × 10^17 FLOPs.
That crossover is the empirical signature of the scaling law.

---

## 13.9 Comparing to GPT-2

The nanochat speedrun targets GPT-2-class capability, measured by the DCLM CORE score. A few
numbers to keep in mind:

| | GPT-2 (2019, OpenAI) | nanochat speedrun (2026) |
|--|---------------------|--------------------------|
| Parameters | 1.5 B | ~560 M (depth=24) |
| Training tokens | ~100 B | ~5–8 B |
| Training cost | ~$43,000 | ~$48 (on-demand) / ~$15 (spot) |
| Wall-clock time | ~168 hours | ~2 hours |
| Architecture | Standard transformer, learned PE | Rotary PE, GQA, Flash Attention 3, Muon |
| CORE score | 0.256525 | ≥ 0.256525 (target) |

The nanochat model is smaller and trained on far fewer tokens, yet matches GPT-2's downstream
benchmark performance. The difference is almost entirely explained by:

1. **Better data.** FineWebEdu and ClimbMix are much higher quality than the raw WebText corpus
   used for GPT-2. Quality of tokens matters more than quantity up to a point.
2. **Better architecture.** Rotary embeddings generalize better than learned positional embeddings.
   GQA reduces memory pressure. Flash Attention 3 processes longer contexts without quadratic cost.
3. **Better optimizer.** Muon converges in far fewer steps than Adam for transformer matrices.
4. **Better hardware.** H100 BF16 tensor cores are roughly 30× faster than V100 FP16 tensor cores,
   which were what OpenAI used in 2019.

---

## 13.10 What to Try Next

You now have a working LLM training codebase and a conceptual model of how scaling works. Here are
concrete experiments ordered from easy to ambitious.

### Easy (1–2 hours of GPU time)

**Compare datasets.** The speedrun uses ClimbMix, but FineWebEdu is also available. Change the
`dataset.py` source and rerun `base_train` at `depth=12` to compare `val_bpb` and CORE score.
Dataset quality has an outsized effect on final model capability.

**Run the CPU demo.** If you have not yet run `runs/runcpu.sh` on a laptop, do it. Watching the
loss curve on a tiny model is useful for building intuition about what "training" looks like
before spending money on GPUs.

**Probe window patterns.** The default `--window-pattern=SSSL` alternates three "short context"
(sliding window) layers with one "long context" (full attention) layer. Try `--window-pattern=L`
(all full attention) or `--window-pattern=SSL` at `depth=12` and compare MFU and final loss.

### Medium (half a day, requires 8×H100)

**Scale up to depth=26.** This is the upper end of the speedrun range and trains to the full
GPT-2 capability CORE target. Approximate cost: $60–80 on-demand.

**Run the miniseries.** `bash runs/miniseries.sh` produces eight checkpoints across the depth
ladder. Plot `val_bpb` vs. `log(params)` to visualize the scaling law for the nanochat
architecture.

**Add a task to `tasks/`.** Look at `tasks/spellingbee.py` or `tasks/gsm8k.py` for examples.
Adding a domain-specific evaluation task (e.g., medical QA, code completion for a specific
framework) and training RL on it is a concrete way to specialize the model.

### Ambitious (multi-day, significant compute)

**Full RLHF loop.** The current RL implementation uses REINFORCE with a verifiable reward. A
proper RLHF pipeline requires training a separate reward model (to score outputs that do not have
automatic verifiers, like open-ended text quality) and then running PPO (Proximal Policy
Optimization) against that reward model. PPO is significantly more stable than REINFORCE for
complex reward landscapes.

**Multi-node training.** The current codebase uses DDP within one node. Scaling to two or more
nodes requires `torchrun --nnodes=2 --node_rank=0 --rdzv_backend=c10d ...`. The nanochat
codebase is structured for single-node, but the PyTorch DDP primitives it uses are fully
multi-node compatible.

**Quantization for inference.** A trained BF16 model at `depth=24` uses roughly 1 GB of VRAM.
INT8 quantization (e.g., via `bitsandbytes`) halves that to ~500 MB, enabling deployment on
smaller GPUs or consumer hardware. INT4 reduces it further to ~250 MB at some quality cost.

**Speculative decoding.** A small "draft model" (`depth=8`) generates tokens quickly. A large
"verifier model" (`depth=24`) then checks them in parallel. Accepted tokens come for free; only
rejected tokens require a full verifier forward pass. In practice, this yields 2–4× inference
speedup for greedy or near-greedy sampling.

---

## 13.11 The Broader LLM Landscape

### Where nanochat fits

nanochat is a research and learning tool. Its strengths:
- Minimal, readable, hackable code — under 4,000 lines for the full pipeline.
- Self-contained: tokenizer, pretraining, SFT, RL, inference, and a web UI in one repo.
- Compute-optimal by default: the `--depth` dial does the right thing.
- A concrete benchmark: the GPT-2 CORE leaderboard.

Its limitations:
- Single-node only (at present).
- No mixture-of-experts, no long-context (beyond 2,048 tokens without modification).
- No dataset curation pipeline — it relies on pre-prepared shards.
- Inference engine is functional but not production-grade (no continuous batching, no paged KV
  cache).

### What the large labs do differently

The capability gap between nanochat and GPT-4 or Claude is not primarily a matter of code
sophistication. It is compute, data, and iteration cycles:

- **Data curation at scale.** State-of-the-art models train on trillions of tokens filtered
  through many rounds of deduplication, quality scoring, and domain balancing. The DCLM paper
  (DataComp for LLMs) showed that dataset quality improvements alone can close a large fraction of
  the gap between similarly-sized models.
- **RLHF at scale.** Aligning a large model requires thousands of human preference labels and a
  carefully trained reward model. The reward model itself is a large language model.
- **Infrastructure.** Thousand-GPU training runs require custom networking (InfiniBand),
  fault-tolerance (checkpoint-on-every-step), and careful gradient synchronization. This is
  engineering, not research.

### Resources to go deeper

**Primary papers:**
- Vaswani et al. (2017), "Attention Is All You Need" — the original transformer.
- Kaplan et al. (2020), "Scaling Laws for Neural Language Models" — first scaling laws.
- Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models" — Chinchilla.
- Touvron et al. (2023), "LLaMA: Open and Efficient Foundation Language Models" — inference-
  optimal training strategy in practice.
- Peng et al. (2023), "RWKV: Reinventing RNNs for the Transformer Era" — alternative to attention.
- Guo et al. (2025), "DeepSeek-R1" — RLHF with verifiable rewards at scale.

**Code:**
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — the spiritual predecessor of
  nanochat, pretraining only, excellent for understanding the basics.
- [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — nanochat's
  inspiration for the speedrun format and Muon optimizer.
- [huggingface/transformers](https://github.com/huggingface/transformers) — production-grade
  model hub; useful for comparing architectures.
- [pytorch/torchtitan](https://github.com/pytorch/torchtitan) — production distributed training.

**Datasets and evaluations:**
- [FineWeb / FineWebEdu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — the default
  pretraining data for nanochat.
- [NVIDIA ClimbMix](https://huggingface.co/datasets/nvidia/ClimbMix) — the current speedrun
  dataset.
- [DCLM (DataComp for LLMs)](https://github.com/mlfoundations/dclm) — source of the CORE metric
  used throughout nanochat.

---

## 13.12 Your Complete Mental Model

Here is the full pipeline, annotated with the key design decision at each stage:

```
raw text (web, books, code)
    │
    │  Quality filtering: only high-educational-value text
    │  [FineWebEdu / ClimbMix]
    ▼
BPE tokenizer  (32,768 subword tokens)
    │  Why BPE: balances vocabulary coverage with token count.
    │  Why 32,768: large enough for efficient compression, small
    │  enough to keep the embedding table manageable.
    ▼
fixed-length windows of 2,048 tokens
    │  Why 2,048: fits in GPU SRAM for Flash Attention; long enough
    │  for multi-sentence context.
    ▼
token embeddings  (n_embd = depth × 64)
    │  Plus rotary positional encoding — applied inside attention,
    │  not added to the residual stream.
    ▼
n_layer transformer blocks (n_layer = depth)
    │  Each block: LayerNorm → GQA → residual
    │              LayerNorm → MLP (2× gated) → residual
    │  GQA: n_head query heads, n_head // 4 KV heads.
    │  Sliding window layers (S) alternate with full-context (L).
    ▼
logit projection  (n_embd → vocab_size)
    │  Weight-tied with embedding table.
    ▼
cross-entropy loss
    │  Bits-per-byte (bpb) = loss × log2(e) / mean_token_bytes
    │  Vocab-size independent, hardware independent.
    ▼
Muon optimizer (matrix params) + AdamW (embeddings, biases)
    │  Muon: Newton-Schulz orthogonalization of gradients.
    │  ~3× faster convergence on transformer matrices than Adam.
    ▼
pretrained base model checkpoint
    │
    ▼  SFT (scripts/chat_sft.py)
conversation format: <|user|> ... <|assistant|> ... <|end|>
    │  Also teaches tool use (<|code_start|> ... <|code_end|>)
    │  and multiple-choice reasoning.
    ▼
SFT checkpoint
    │
    ▼  RL (scripts/chat_rl.py)
REINFORCE on verifiable tasks:
    │  math (GSM8K), spelling, code execution
    ▼
RL-aligned checkpoint
    │
    ▼  Engine (nanochat/engine.py)
KV cache + batch generation
    │
    ▼  scripts/chat_web.py
HTTP API + nanochat/ui.html → browser chat interface
```

The key design decisions and their rationale:

| Decision | Rationale |
|----------|-----------|
| Single `--depth` dial | Removes the need to understand all hyperparameter interactions. Ensures every run is compute-optimal. |
| `target_param_data_ratio = 10.5` | Smaller model, more tokens = better inference-time cost/quality tradeoff than Chinchilla's 20. |
| BF16 compute + FP32 master weights | BF16 speeds up matmuls on Ampere+; FP32 optimizer state prevents precision loss during weight updates. |
| FP8 with tensorwise scaling | ~2× matmul throughput on H100+ with minimal code (150 lines vs. torchao's 2,000). |
| GQA (n_kv_head = n_head // 4) | Reduces KV cache size 4× at inference with negligible quality loss. |
| Muon optimizer | Empirically faster convergence than Adam on transformer weight matrices; stabilizes training. |
| Flash Attention 3 | O(1) memory in sequence length for attention; 2–3× faster than naive SDPA on Hopper GPUs. |
| Weight tying (embedding ↔ lm_head) | Reduces parameters by ~vocab_size × n_embd without quality loss; enforces consistency. |

---

## Summary of the Full Pipeline

| Chapter | Stage | Key files | Key concept |
|---------|-------|-----------|-------------|
| 1 | Setup | `runs/speedrun.sh`, `pyproject.toml` | uv, venv, GPU drivers |
| 2 | Fundamentals | `nanochat/gpt.py` | Tokens, embeddings, attention, loss |
| 3 | Tokenization | `scripts/tok_train.py`, `nanochat/tokenizer.py` | BPE, compression ratio |
| 4 | Architecture | `nanochat/gpt.py` | GPTConfig, GQA, rotary PE, MLP gating |
| 5 | Data pipeline | `nanochat/dataloader.py`, `nanochat/dataset.py` | Streaming shards, bestfit packing |
| 6 | Pretraining | `scripts/base_train.py` | Loss, optimizer, LR schedule |
| 7 | Distributed | `scripts/base_train.py` | DDP, torchrun, gradient accumulation |
| 8 | Evaluation | `scripts/base_eval.py`, `nanochat/core_eval.py` | BPB, CORE metric, DCLM |
| 9 | SFT | `scripts/chat_sft.py` | Chat format, multi-task fine-tuning |
| 10 | RL | `scripts/chat_rl.py` | REINFORCE, verifiable rewards |
| 11 | Inference | `nanochat/engine.py` | KV cache, batch generation, temperature |
| 12 | Web UI | `scripts/chat_web.py`, `nanochat/ui.html` | HTTP API, streaming |
| 13 | Scaling laws | `nanochat/common.py`, `scripts/base_train.py` | Chinchilla, MFU, FP8, depth ladder |

---

## Going Further: Concrete Next Steps

1. **Read the Chinchilla paper.** [arXiv:2203.15556](https://arxiv.org/abs/2203.15556). It is 67
   pages but Section 3 and Appendix D are the core. The experimental methodology for finding the
   optimal token/param ratio is directly reproduced by `runs/scaling_laws.sh`.

2. **Read the `dev/LOG.md` in the nanochat repo.** This is the internal research log documenting
   every architectural decision and ablation. It is the best secondary source for understanding
   *why* the current defaults are what they are.

3. **Submit to the leaderboard.** If you make a change that improves the speedrun time or the
   CORE score, consider opening a PR. The leaderboard accepts contributions; see
   `dev/LEADERBOARD.md` for the submission format.

4. **Fork and specialize.** nanochat is explicitly designed to be forked. If you have a domain
   — medical text, legal documents, a specific programming language — the code is small enough
   that you can understand every line and make targeted changes.

---

## Check Your Understanding

**1.** A colleague wants to train a model with a compute budget of `C = 10^20` FLOPs and uses the
Chinchilla equal-scaling rule (`r = 20`). What is the optimal number of parameters `N` and
training tokens `D`?

*Work it out:* `N = sqrt(C / (6r)) = sqrt(1e20 / 120) ≈ sqrt(8.33e17) ≈ 9.1 × 10^8 ≈ 910 M`.
`D = 20 × N ≈ 18 B`. A ~910 M parameter model trained on ~18 B tokens.

**2.** nanochat reports `bf16_mfu: 38.5` during training on an 8×H100 node. The peak BF16 FLOPS
for one H100 SXM is 989 TFLOPS. The model's `estimate_flops()` returns `7.2 × 10^9` FLOPs per
token, and `total_batch_size = 1,048,576` tokens. What is the actual compute throughput, and does
the 38.5% MFU figure make sense?

*Work it out:* Actual FLOPs/sec = `7.2e9 × 1,048,576 / dt`. If MFU = 38.5%, then
`actual = 0.385 × 8 × 989e12 ≈ 3.05 × 10^15` FLOPs/sec.
`dt = 7.2e9 × 1,048,576 / 3.05e15 ≈ 2.47` seconds per step. That is a plausible step time.

**3.** You run `base_train` with `--depth=20 --target-param-data-ratio=10.5`. Later, you re-run
the same model with `--depth=20 --target-param-data-ratio=20` (Chinchilla optimal). Which run
trains for more iterations? Which run should produce a lower `val_bpb` at the end of training,
and why?

*Answer:* The `--target-param-data-ratio=20` run trains for more iterations (20 / 10.5 ≈ 1.9×
more tokens). Because it trains longer, it sees more data and should achieve a lower `val_bpb`.
However, it also spends nearly twice the compute, so comparing the two fairly requires fixing the
compute budget (use `--target-flops`), not the model size.

---

## Closing

You started this tutorial knowing Python and not much else. Over thirteen chapters you have
tokenized raw text, built a transformer from first principles, trained it on internet-scale data
across multiple GPUs, aligned it to follow instructions, deployed it behind a web interface, and
now understand the mathematical laws that govern how all of this scales.

The LLM field moves quickly, but the fundamentals do not: power laws, compute budgets, the tension
between model size and training horizon, and the engineering challenge of making hardware
utilization approach its theoretical maximum. These are the same problems whether you are training
a 12-layer research model for $5 or a trillion-parameter production system for $500 million. The
difference is scale; the underlying questions are identical.

nanochat gives you a working system small enough to hold in your head and big enough to produce
real, usable results. That combination — complete, understandable, and functional — is rarer than
it sounds. Use it well, improve it, and pass the improvements on.

---

*Chapter 13 of the nanochat tutorial. For questions and discussion, see the*
*[nanochat GitHub Discussions](https://github.com/karpathy/nanochat/discussions) or the*
*`#nanochat` channel on Discord.*
