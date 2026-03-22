# Chapter 7: Distributed Training — Multi-GPU with DDP

## What you'll learn

- Why single-GPU training has hard limits and how Distributed Data Parallel (DDP) breaks through them
- The mechanics of DDP: process groups, ranks, gradient all-reduce, and why every GPU ends up with identical weights
- How to launch multi-GPU jobs with `torchrun` and what environment variables it sets
- How nanochat's `compute_init()` transparently handles CPU, MPS, single-GPU, and multi-GPU without changes to the training loop
- How to measure training efficiency with Model FLOPS Utilization (MFU)

## Prerequisites

- Completed Chapters 1–6 (setup, LLM fundamentals, tokenization, architecture, data pipeline, pretraining loop)
- Comfortable reading Python; no prior distributed computing experience required
- PyTorch installed (from Chapter 1)
- At least one GPU recommended, but CPU multi-process works for concept exploration

---

## 7.1 Why Multi-GPU Training?

After Chapter 6 you can train a GPT-class model on a single GPU. That is genuinely useful for experimentation, but production-scale runs hit two walls: memory and time.

### The memory wall

Every parameter in a model needs to live somewhere. A float32 parameter costs 4 bytes; bfloat16 costs 2. The rule of thumb for training (weights + gradients + optimizer moments) is roughly 16–20 bytes per parameter in mixed-precision mode.

| Model size | Parameters | bf16 training footprint |
|---|---|---|
| GPT-2 (small) | 117 M | ~2 GB |
| GPT-2 (XL) | 1.5 B | ~24 GB |
| LLaMA 7B | 7 B | ~112 GB |
| LLaMA 70B | 70 B | ~1.1 TB |

A single H100 SXM has 80 GB of HBM. You can fit a 7B parameter model for inference in bfloat16, but training it requires roughly 112 GB — more than one GPU can hold. Before you even think about multi-GPU for speed, you sometimes need it just to make a run possible.

nanochat's default model (depth 20, aspect ratio 64) has around 350 M parameters. That fits comfortably in a single 80 GB GPU, so memory is not the forcing factor here — but training speed is.

### The throughput wall

Training on a GPT-2-class model for a compute-optimal run takes roughly 10 billion tokens. On a single A100 processing 500,000 tokens per second, that is about 5.5 hours. On 8×A100s, it is under an hour. The cost drops too, because a 1-hour run on 8 GPUs uses less rental time than an 8-hour run on 1 GPU.

Real ballpark numbers for nanochat's depth-20 model:

| Setup | Throughput | Time to 10B tokens | Cloud cost (est.) |
|---|---|---|---|
| 1× H100 | ~1.5 M tok/s | ~1.9 hours | ~$6 |
| 8× H100 | ~12 M tok/s | ~14 minutes | ~$3.50 |

The 8-GPU run is both faster and cheaper. This is the promise of data-parallel scaling.

---

## 7.2 Three Kinds of Parallelism

When people say "distributed training" they usually mean one of three things. It is worth distinguishing them before we go further.

### Data Parallel (what nanochat uses)

Each GPU holds a **complete copy of the model**. The training data is split: GPU 0 sees tokens 0–N, GPU 1 sees tokens N–2N, and so on. Each GPU runs its forward and backward pass independently, then the gradients are **synchronized** (averaged) across all GPUs before the optimizer step. Because every GPU has the same averaged gradients, they all make the same weight update. After every step, every GPU still has identical weights.

```
GPU 0:  [full model] + [data shard 0] -> gradients_0 --\
GPU 1:  [full model] + [data shard 1] -> gradients_1 --> all_reduce --> averaged gradients -> weight update
GPU 2:  [full model] + [data shard 2] -> gradients_2 --/
...
GPU N:  [full model] + [data shard N] -> gradients_N --/
```

This is **Distributed Data Parallel (DDP)**, and it is the right choice when a single GPU can hold the entire model.

### Tensor Parallel

Individual weight matrices are split across GPUs. GPU 0 holds the left half of an attention projection matrix; GPU 1 holds the right half. Each GPU computes a partial result; the results are combined with communication. This requires careful partitioning of every layer and is significantly more complex to implement. Frameworks like Megatron-LM specialize in this.

### Pipeline Parallel

Different layers are placed on different GPUs. GPU 0 runs transformer layers 0–6, GPU 1 runs layers 7–13, and so forth. Data flows through the pipeline in micro-batches. The complexity here is managing the pipeline stages so GPUs are not idle ("pipeline bubbles"). This is used for models with hundreds of layers.

**nanochat uses DDP only.** For models that fit on a single GPU (which covers GPT-2 through roughly 40B parameters with large-memory GPUs), DDP delivers near-linear scaling with almost no implementation complexity beyond the single-GPU case. Tensor and pipeline parallelism add substantial complexity for marginal gains at this scale.

---

## 7.3 How DDP Works: A Detailed Walk-Through

Understanding DDP well enough to debug it requires understanding what happens at each phase of training. Here is the full picture.

### Setup

When a multi-GPU job starts, `torchrun` (covered in section 7.5) launches N separate Python processes — one per GPU. Each process:

1. Reads its rank from an environment variable (`RANK`)
2. Calls `dist.init_process_group()` to join a **process group** — a communication mesh connecting all N processes
3. Builds the model from scratch and moves it to its GPU
4. Wraps the model in `DDP`

After wrapping in DDP, PyTorch broadcasts the parameters from rank 0 to all other ranks. This ensures all GPUs start with bit-for-bit identical weights.

### Forward Pass

Each process runs its forward pass **independently** on its own data shard. No communication happens here. GPU 0 computes loss on tokens 0–N; GPU 1 computes loss on tokens N–2N. The losses will differ because the inputs differ — that is expected.

### Backward Pass and Gradient Synchronization

Here is the key DDP insight. During `loss.backward()`, DDP hooks into the backward computation graph. As each gradient tensor is computed for a parameter, DDP asynchronously fires an **all-reduce** operation for that gradient's bucket before the backward pass for earlier layers has even finished. The all-reduce computes the **sum** of that gradient across all GPUs and writes the result back to every GPU.

```
          GPU 0          GPU 1         GPU 2
Forward:  loss_0         loss_1        loss_2      (independent, no communication)
          |              |             |
Backward: grad_0         grad_1        grad_2      (computed locally)
          \              |             /
           \             |            /
            ------all_reduce---------            (sum across GPUs via NCCL)
           /             |            \
          grad_sum       grad_sum      grad_sum   (all GPUs now have the same sum)
```

After the all-reduce, every GPU has the **same gradient** for every parameter. The optimizer step is therefore identical on every GPU. No further synchronization is needed to keep weights in sync — they were already identical before the forward pass, and now they get the same update.

### The Overlapping Trick

DDP does not wait until all gradients are computed before communicating. It groups gradients into **buckets** (by default 25 MB each) and starts the all-reduce for each bucket as soon as that bucket is full. This overlaps gradient communication with the backward computation of earlier layers, hiding most of the communication latency.

```
Time -->
backward(layer 12): compute grad_12 -------> all_reduce(bucket 3) -\
backward(layer 11): compute grad_11 -> fill bucket -> all_reduce -----> (in parallel)
backward(layer 10): compute grad_10                                   /
...
optimizer.step():   apply all-reduced gradients
```

### Loss Scaling for Gradient Accumulation

When you use gradient accumulation (running multiple forward/backward passes before a single optimizer step), the gradients from each micro-step are summed by PyTorch's `.backward()`. To convert this sum into a mean you must divide the loss by the number of accumulation steps before calling `.backward()`.

In nanochat's training loop (`scripts/base_train.py`, around line 512):

```python
loss = loss / grad_accum_steps  # normalize before backward
loss.backward()
```

This produces the correct mean gradient regardless of how many accumulation steps are used.

---

## 7.4 Process Groups and Ranks

The vocabulary of distributed PyTorch has five terms you need to know cold.

| Term | Meaning |
|---|---|
| `rank` | Unique integer ID for this process, from `0` to `world_size - 1`. Rank 0 is special: it does logging, checkpointing, and evaluation. |
| `world_size` | Total number of processes in the job. On 8 GPUs, this is 8. |
| `local_rank` | GPU index **on this machine** (0–7 for 8 GPUs on one node). On multi-node jobs, rank 9 on the second node might have local_rank 1. |
| `MASTER_ADDR` | Hostname or IP of the machine where rank 0 is running. Other processes connect here during `init_process_group`. |
| `MASTER_PORT` | TCP port on `MASTER_ADDR` used for rendezvous. |

For a single-node job on 8 GPUs:

```
MASTER_ADDR=localhost
MASTER_PORT=29500

rank 0  local_rank 0  <-> GPU 0
rank 1  local_rank 1  <-> GPU 1
rank 2  local_rank 2  <-> GPU 2
...
rank 7  local_rank 7  <-> GPU 7
```

For a two-node job (8 GPUs per node = 16 total):

```
Node 0 (MASTER_ADDR):
  rank 0  local_rank 0  <-> GPU 0
  rank 1  local_rank 1  <-> GPU 1
  ...
  rank 7  local_rank 7  <-> GPU 7

Node 1:
  rank 8   local_rank 0  <-> GPU 0
  rank 9   local_rank 1  <-> GPU 1
  ...
  rank 15  local_rank 7  <-> GPU 7
```

### The process group

`dist.init_process_group(backend="nccl")` connects all processes through a rendezvous server at `MASTER_ADDR:MASTER_PORT`. Every process blocks here until all `world_size` processes have joined. Once the group is initialized, collective operations (all_reduce, broadcast, barrier) are available.

**Why NCCL?** NCCL (NVIDIA Collective Communications Library) is NVIDIA's hand-optimized library for GPU-to-GPU communication. It uses NVLink for intra-node transfers (very fast: 600 GB/s on H100s) and InfiniBand or Ethernet for inter-node transfers. For CPU jobs, `gloo` is the usual backend.

---

## 7.5 Launching Multi-GPU Jobs with torchrun

`torchrun` is PyTorch's distributed launcher. It replaces the older `torch.distributed.launch` and handles all the bookkeeping of environment variables automatically.

### Basic usage

```
torchrun --nproc_per_node=N script.py [script args]
```

This launches N processes on the current machine, each running `script.py`. Each process automatically receives the environment variables:

- `RANK` — global rank
- `LOCAL_RANK` — GPU index on this machine
- `WORLD_SIZE` — total number of processes
- `MASTER_ADDR` — where rank 0 is listening (defaults to `localhost` for single-node)
- `MASTER_PORT` — the port (defaults to `29500`)

### Single-node examples

```bash
# ✍️ 4 GPUs on one machine:
torchrun --nproc_per_node=4 -m scripts.base_train --depth=20

# ✍️ 8 GPUs on one machine:
torchrun --nproc_per_node=8 -m scripts.base_train --depth=24

# ✍️ CPU multi-process (for testing concepts, not performance):
torchrun --nproc_per_node=2 -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --device-type=cpu
```

### Multi-node example

```bash
# ✍️ On node 0 (MASTER):
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=10.0.0.1 \
  --master_port=29500 \
  -m scripts.base_train --depth=24

# ✍️ On node 1:
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=10.0.0.1 \
  --master_port=29500 \
  -m scripts.base_train --depth=24
```

Both commands start simultaneously. The two machines rendezvous at `10.0.0.1:29500`. Once all 16 processes have joined, training begins.

### What happens when you do NOT use torchrun

If you run `python -m scripts.base_train` directly (no torchrun), none of `RANK`, `LOCAL_RANK`, or `WORLD_SIZE` are set. nanochat detects this and runs in single-process mode with no DDP — the same training loop, just on one GPU or CPU. This is the normal path for development and debugging.

---

## 7.6 nanochat's `compute_init()` Function

`compute_init()` is defined in `nanochat/common.py` (line 173). It is the single function called at the top of every training script to set up the device, dtype, and (if applicable) the process group. Here is the full function:

```python
# nanochat/common.py

def compute_init(device_type="cuda"): # cuda|cpu|mps
    """Basic initialization that we keep doing over and over, so make common."""

    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "..."
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "..."

    # Reproducibility
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)

    # Precision
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high")  # use TF32 for matmuls

    # Distributed setup
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)  # mps|cpu

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device
```

Walk through each decision:

**`get_dist_info()`** checks for the presence of `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` in the environment (set by torchrun). If all three are present, it reads and returns them. If none are present, it returns `(False, 0, 0, 1)` — a sensible single-process default.

**`if is_ddp_requested and device_type == "cuda"`** — DDP is only initialized when torchrun launched the job AND the device is CUDA. Two important cases are excluded:
- `device_type == "mps"` (Apple Silicon): MPS does not support NCCL, so DDP is never initialized even if you tried to launch with torchrun. This is a hardware limitation.
- `device_type == "cpu"`: CPU distributed training exists but is rarely used in practice because the overhead usually exceeds the benefit. nanochat skips it for simplicity. (You can still launch multiple CPU processes with torchrun for testing the distributed logic without using DDP itself.)

**`device = torch.device("cuda", ddp_local_rank)`** — This creates a device handle that points to GPU number `ddp_local_rank` on this machine. On an 8-GPU node, rank 0 gets `cuda:0`, rank 1 gets `cuda:1`, and so forth. The subsequent `torch.cuda.set_device(device)` makes this GPU the default for all CUDA operations in this process.

**`dist.init_process_group(backend="nccl", device_id=device)`** — Joins the process group. All N processes must call this before any of them can proceed past it. The `device_id` argument (added in PyTorch 2.x) helps NCCL map GPU identities correctly.

**`dist.barrier()`** — Waits until all processes have finished `init_process_group`. This is a safety measure: it ensures no process races ahead to allocate model memory before the communication infrastructure is ready.

**The return value** `(is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device)` is unpacked at the top of every training script and the variables are used throughout the training loop.

### `compute_cleanup()`

```python
# nanochat/common.py

def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp_initialized():
        dist.destroy_process_group()
```

`destroy_process_group()` gracefully shuts down the NCCL communicators. Without this, NCCL background threads can cause messy error messages or hang at exit. nanochat calls this at the very end of every training script.

---

## 7.7 Wrapping the Model with DDP

Once `compute_init()` returns, the training script must wrap the model in `DDP` before the training loop. In nanochat's `scripts/base_train.py`, the model setup follows this sequence:

```python
# scripts/base_train.py (simplified)

# 1. Build model on meta device (shapes only, no data)
model = build_model_meta(args.depth)

# 2. Allocate storage and initialize weights
model.to_empty(device=device)
model.init_weights()

# 3. Compile (torch.compile happens BEFORE DDP wrapping in nanochat)
orig_model = model
model = torch.compile(model, dynamic=False)

# (DDP wrapping happens implicitly through the Engine or optimizer setup
#  that respects the ddp flag — see note below)
```

> **Note on nanochat's DDP wrapping:** nanochat currently passes the model directly to the optimizer and forward calls without an explicit `DDP(model)` call in the training script. The `ddp` flag returned from `compute_init()` is used to gate DDP-specific logic (e.g., gradient accumulation, loss scaling, barrier usage). For learners coming from vanilla DDP tutorials, the canonical pattern is:
>
> ```python
> from torch.nn.parallel import DistributedDataParallel as DDP
> if ddp:
>     model = DDP(model, device_ids=[ddp_local_rank])
> ```
>
> This is the pattern used in Karpathy's nanoGPT and many other tutorials. Understanding it is essential because nanochat's optimizer and data pipeline are designed with this pattern in mind.

### Why `device_ids=[local_rank]`?

`device_ids` tells DDP which GPU this process owns. DDP uses this to:
1. Register NCCL communicators on the correct GPU
2. Determine where to place gradient buckets
3. Do the final all-reduce on the correct device

Omitting `device_ids` works for single-GPU processes but can cause silent correctness issues in multi-GPU setups.

### Accessing the underlying model: `model.module`

Once a model is wrapped in DDP, `model` is a `DistributedDataParallel` object. The original model is accessible as `model.module`. This matters in two places:

**Saving checkpoints:** You want to save `model.module.state_dict()`, not `model.state_dict()`. The `DDP.state_dict()` adds a `module.` prefix to every key, which makes the checkpoint incompatible with loading on a single GPU. nanochat avoids this by keeping a reference to `orig_model` (the pre-DDP, pre-compiled model) and saving `orig_model.state_dict()`.

**Evaluation:** Calling model-specific methods (anything not `forward()`, like `model.estimate_flops()` or `model.num_scaling_params()`) must go through `model.module`, not `model`. Again, nanochat sidesteps this by using `orig_model` for evaluation.

### torch.compile and DDP

`torch.compile` must be applied **before** `DDP` wrapping. The compile step replaces Python function calls with optimized kernels; DDP then instruments the resulting module's backward hooks. If you wrap with DDP first and compile second, compilation may fail or produce incorrect gradient synchronization. nanochat's `base_train.py` follows the correct order: `torch.compile(model)` at line 245, then DDP usage (implicitly through the distributed data loader and optimizer).

---

## 7.8 Gradient All-Reduce in Detail

The all-reduce is the heart of DDP. Understanding it explains both why DDP is correct and what its performance characteristics are.

### What all-reduce does

`dist.all_reduce(tensor, op=dist.ReduceOp.SUM)` takes a tensor on each GPU, sums them across all GPUs, and writes the sum back to every GPU. After the call, every GPU has the same value: the sum.

For gradients, we want the **mean** not the sum — otherwise the effective learning rate would scale with world size. There are two ways to get the mean:

1. Use `dist.ReduceOp.AVG` if your NCCL version supports it
2. Do a SUM all-reduce and then divide by `world_size` manually

DDP uses option 2 internally (divide after the sum). This is why you do not need to scale your loss or learning rate by world_size when using vanilla DDP — the library handles it.

### Communication volume and cost

For a model with P parameters, each gradient tensor is P × (bytes per element) in size. An all-reduce on N GPUs requires each GPU to send and receive roughly 2 × P × (bytes per element) × (N-1)/N bytes (ring all-reduce). For an 8-GPU all-reduce with a 350 M parameter model in bf16:

```
2 × 350e6 × 2 bytes × 7/8 ≈ 1.22 GB per step
```

On NVLink (600 GB/s), this takes about 2 ms — negligible compared to the seconds spent in compute. This is why DDP scales nearly linearly on NVLink-connected GPUs.

On PCIe-connected GPUs or across network links, the bandwidth is much lower (PCIe 4.0 x16: ~32 GB/s bidirectional), and communication can become a significant fraction of step time.

### Gradient accumulation and DDP

When using gradient accumulation, nanochat runs multiple forward/backward passes before each optimizer step. By default, DDP fires an all-reduce after every `.backward()` call. This is wasteful — you would do `grad_accum_steps` all-reduces when only one is needed (after the final micro-step).

The standard pattern to prevent premature synchronization is `model.no_sync()`:

```python
# Conceptual pattern (not in nanochat's current codebase):
for micro_step in range(grad_accum_steps):
    if micro_step < grad_accum_steps - 1:
        with model.no_sync():        # suppress all-reduce for intermediate steps
            loss.backward()
    else:
        loss.backward()              # final step: allow all-reduce
```

In `model.no_sync()` mode, gradients accumulate locally without communication. Only on the final `.backward()` are they all-reduced. This reduces communication overhead by a factor of `grad_accum_steps`.

---

## 7.9 Synchronization Primitives

DDP handles gradient synchronization automatically. But several other situations require explicit synchronization.

### `dist.barrier()`

```python
dist.barrier()  # all processes block here until all have arrived
```

A barrier is a rendezvous point. No process can proceed past it until every process in the group has called it. Uses in nanochat:

- After `init_process_group()` in `compute_init()`: ensures all processes have joined the group before any starts building the model
- After file downloads: if one process downloads a tokenizer file, others must wait before trying to open it (though nanochat uses a file lock for this instead — see `download_file_with_lock` in `common.py`)

Barriers are cheap (microseconds on NCCL) but easy to forget. A barrier on rank 0 without a matching barrier on rank 1 will deadlock forever.

### `dist.all_reduce()`

```python
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

All-reduce applies an operation (SUM, MAX, MIN, AVG) to a tensor across all ranks and writes the result back to every rank. nanochat uses this explicitly in one place: when fp16 training is enabled, the GradScaler's `found_inf` flag must be consistent across all ranks (in `scripts/base_train.py`, around line 532):

```python
# scripts/base_train.py
if is_ddp_initialized():
    for v in scaler._found_inf_per_device(optimizer).values():
        dist.all_reduce(v, op=dist.ReduceOp.MAX)
```

If any GPU encounters an inf or nan gradient, `found_inf` becomes 1 on that GPU. The MAX all-reduce propagates this to all GPUs, so all ranks agree to skip the optimizer step. Without this, some ranks would step and others would skip, and the weights would diverge.

### `print0()` — logging only from rank 0

nanochat defines a small utility in `common.py`:

```python
def print0(s="", **kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)
```

In a multi-GPU job, every rank would otherwise print the same log lines N times. `print0` gates output to rank 0 only. All the `print0(f"step {step:05d}...")` calls in `base_train.py` use this.

---

## 7.10 Checkpointing in Distributed Training

Checkpointing in a multi-GPU run requires care. Naively, every rank saving its own copy of the model produces N copies of the same data, consuming N× the disk space. More seriously, if rank 0 and rank 1 try to write the same file simultaneously, you get a corrupted checkpoint.

### Model parameters: rank 0 saves, others skip

In `nanochat/checkpoint_manager.py` (line 42):

```python
def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
    # Optimizer state is sharded — each rank saves its own
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
```

Model weights are identical across all ranks (DDP guarantees this), so only rank 0 saves them. The metadata JSON is also rank 0 only. The resulting checkpoint directory contains:

```
base_checkpoints/d20/
  model_001000.pt        # one file, saved by rank 0
  meta_001000.json       # one file, saved by rank 0
  optim_001000_rank0.pt  # optimizer shard for rank 0
  optim_001000_rank1.pt  # optimizer shard for rank 1
  ...
  optim_001000_rank7.pt  # optimizer shard for rank 7
```

### Why optimizer state is per-rank

The optimizer (Adam, Muon) stores momentum and variance buffers for every parameter. These buffers track the gradient history for that parameter. Since each rank saw **different data**, the gradient history is rank-specific — rank 0's momentum for layer 3's weight reflects the gradients from data shard 0, while rank 1's reflects shard 1. If you saved only rank 0's optimizer state and loaded it on all ranks when resuming, the optimizer would continue as if all ranks had seen rank 0's data, breaking the careful data sharding.

Therefore, each rank saves and loads its own optimizer state. When loading:

```python
# scripts/base_train.py
model_data, optimizer_data, meta_data = load_checkpoint(
    checkpoint_dir, args.resume_from_step, device,
    load_optimizer=True,
    rank=ddp_rank  # each rank loads its own optimizer shard
)
```

### No barrier needed in nanochat's checkpoint

Notice that nanochat's `save_checkpoint` does not call `dist.barrier()` after saving. This is safe here because the training loop proceeds identically on all ranks regardless of whether the save has completed: after saving, everyone continues to the next training step. If you have a use case where other ranks must wait for rank 0's file to be fully written (e.g., if another process will immediately read it), you would add a barrier:

```python
if rank == 0:
    torch.save(...)
dist.barrier()  # wait for rank 0 to finish writing
```

---

## 7.11 MFU: Measuring Training Efficiency

You have launched your multi-GPU run and training is going. How do you know if you are using the hardware well? The answer is **Model FLOPS Utilization (MFU)**.

### What MFU measures

MFU is the fraction of theoretical peak FLOPS that your training actually achieves:

```
MFU = actual_FLOPS_per_second / (peak_FLOPS_per_second × world_size)
```

An MFU of 50% means half the hardware's theoretical capacity is being used. The other half is lost to memory bandwidth limitations, kernel launch overhead, communication, Python overhead, and so on. A well-optimized training run on a large model with Flash Attention achieves 40–60% MFU. Getting above 60% is excellent; below 20% suggests a bottleneck worth investigating.

### Calculating actual FLOPS

For a Transformer, the dominant FLOP cost is matrix multiplications. A useful approximation (from Chinchilla and related work) is:

```
FLOPs per token ≈ 6 × (number of parameters)
```

The factor of 6 comes from: 2× multiply-add in forward (matmul is 2 FLOPs per element), 2× for backward (gradient of inputs + gradient of weights), with a small overhead for attention and activations. nanochat's `model.estimate_flops()` computes a more precise version of this.

Total FLOPs for a training step:

```
FLOPs_per_step = FLOPs_per_token × total_batch_size_in_tokens
```

Actual FLOPS rate:

```
actual_FLOPS_per_sec = FLOPs_per_step / step_time_in_seconds
```

### The peak FLOPS lookup table

`get_peak_flops()` in `nanochat/common.py` returns the theoretical peak bf16 FLOPS for a GPU given its name string. A condensed view of the table:

```python
# nanochat/common.py  (selected entries)
_PEAK_FLOPS_TABLE = (
    # NVIDIA Blackwell
    (["b200"],    2.25e15),   # 2,250 TFLOPS
    # NVIDIA Hopper
    (["h100"],     989e12),   # 989 TFLOPS
    (["h100", "pcie"], 756e12),
    # NVIDIA Ampere
    (["a100"],    312e12),    # 312 TFLOPS
    # Consumer
    (["4090"],    165.2e12),  # 165 TFLOPS
    (["3090"],     71e12),
    ...
)
```

If the GPU is not in the table, `get_peak_flops()` returns `float('inf')`, making MFU display as 0% rather than a wrong number.

### MFU in the training loop

In `scripts/base_train.py` (around line 553):

```python
flops_per_sec = num_flops_per_token * total_batch_size / dt
mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
print0(f"... bf16_mfu: {mfu:.2f} ...")
```

`gpu_peak_flops` is the single-GPU peak; multiplying by `ddp_world_size` gives the aggregate peak for the whole cluster. Dividing actual aggregate FLOPS by this gives the MFU percentage.

Note that `gpu_peak_flops` is fetched once using `torch.cuda.get_device_name(0)` — it uses GPU 0's name. For homogeneous clusters (same GPU model on every node) this is correct. For heterogeneous clusters it is an approximation.

---

## 7.12 Running Multi-GPU Training: A Complete Example

### Step 1: Verify your GPU setup

```bash
# ✍️
nvidia-smi
```

Expected output shows a table of available GPUs. Note the GPU model name (e.g., "NVIDIA H100 SXM5 80GB").

### Step 2: Single-GPU baseline first

Always validate on single GPU before scaling up. This catches model bugs cheaply.

```bash
# ✍️ Single GPU, small model, 20 steps
python -m scripts.base_train \
  --depth=4 \
  --max-seq-len=512 \
  --device-batch-size=4 \
  --num-iterations=20 \
  --eval-every=-1 \
  --core-metric-every=-1
```

You should see the training loop run 20 steps and print loss values. If this fails, do not proceed to multi-GPU.

### Step 3: Two-GPU run

```bash
# ✍️ 2 GPUs
torchrun --nproc_per_node=2 -m scripts.base_train \
  --depth=4 \
  --max-seq-len=512 \
  --device-batch-size=4 \
  --num-iterations=20 \
  --eval-every=-1 \
  --core-metric-every=-1
```

### Step 4: Full 8-GPU training run

```bash
# ✍️ 8 GPUs, full-size model
torchrun --nproc_per_node=8 -m scripts.base_train --depth=20
```

### What the output looks like

Only rank 0 prints. You will see something like:

```
2026-03-22 10:00:00 - nanochat.common - INFO - Distributed world size: 8
2026-03-22 10:00:01 - nanochat.common - INFO - GPU: NVIDIA H100 SXM5 80GB | Peak FLOPS (BF16): 9.89e+14
2026-03-22 10:00:01 - nanochat.common - INFO - COMPUTE_DTYPE: torch.bfloat16 (auto-detected: CUDA SM 90 (bf16 supported))
...
step 00001/04200 (0.02%) | loss: 10.892314 | lrm: 0.03 | dt: 980.12ms | tok/sec: 535,612 | bf16_mfu: 43.21 | ...
step 00002/04200 (0.05%) | loss: 10.841203 | lrm: 0.05 | dt: 312.44ms | tok/sec: 1,679,832 | bf16_mfu: 51.83 | ...
```

The first step is slower because it includes `torch.compile` kernel compilation. From step 2 onward, you should see stable throughput and MFU in the 40–60% range on H100s with Flash Attention 3.

### CPU multi-process (for concept exploration)

If you do not have GPUs, you can still explore the distributed logic:

```bash
# ✍️ CPU, 2 processes (slow but works)
torchrun --nproc_per_node=2 -m scripts.base_train \
  --depth=4 \
  --max-seq-len=256 \
  --device-batch-size=1 \
  --num-iterations=5 \
  --device-type=cpu \
  --eval-every=-1 \
  --core-metric-every=-1
```

Note: because `device_type=cpu` causes `compute_init()` to skip `dist.init_process_group()`, you are running two Python processes that do not actually synchronize gradients. This is useful for testing the data sharding and rank-gated logic, but the two processes train independently. Real CPU distributed training (with NCCL's `gloo` backend) is possible but is outside nanochat's current scope.

---

## 7.13 Debugging Distributed Training

Distributed bugs are uniquely frustrating: they often manifest as hangs (the program never finishes) rather than clear error messages. Here is a systematic approach.

### Validate single-GPU first

Run the exact same script without `torchrun`. If it crashes in single-GPU mode, the bug is in your model or data, not in the distributed setup. Fix that first.

### Common error: NCCL timeout

```
ncclTimeoutError: NCCL timeout ...
```

This means one or more ranks did not call a collective operation (barrier, all_reduce) within the timeout window. Common causes:

- **Asymmetric code paths**: an `if rank == 0:` block that calls `dist.barrier()` without a matching barrier on all other ranks. Every collective must be called by every rank.
- **One rank crashed silently**: check the logs for all ranks (redirect each rank's stderr to a separate file).
- **OOM on one rank**: if rank 3 runs out of memory and crashes, the remaining ranks will hang at the next collective. Check `nvidia-smi` during training.

To get verbose NCCL logs:

```bash
# ✍️
NCCL_DEBUG=INFO torchrun --nproc_per_node=8 -m scripts.base_train --depth=20 2>&1 | head -200
```

This prints the NCCL initialization messages, connection establishment, and any errors. It is verbose but invaluable for diagnosing communication failures.

### Common error: CUDA OOM on one rank

If one rank runs out of GPU memory, the other ranks will eventually hang at the next all-reduce. Signs:

- Training stops printing after a few steps
- `nvidia-smi` shows one GPU at <100% utilization while others are at 100%
- `RuntimeError: CUDA out of memory` in one rank's output

Fix by reducing `--device-batch-size`. Remember that each rank processes `device_batch_size × max_seq_len` tokens per forward pass. Halving `device_batch_size` halves the peak memory per rank.

### Common error: weights diverge between ranks

If you suspect weights have diverged (loss suddenly jumps on some runs), you can verify with:

```python
# ✍️ (add temporarily to training loop for debugging)
if is_ddp_initialized():
    for name, param in model.named_parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= ddp_world_size
        # If param.data changed, ranks were different
```

In practice, DDP guarantees weight consistency as long as you do not modify `param.data` outside of the DDP-wrapped forward/backward cycle. The most common way to accidentally break this is by loading a checkpoint into some ranks but not others, or by performing rank-specific initialization after the DDP broadcast.

### Scaling checklist

Before scaling from 1 to N GPUs, verify:

- [ ] Single-GPU run completes without errors
- [ ] `total_batch_size` is divisible by `world_tokens_per_fwdbwd` (nanochat asserts this at line 408)
- [ ] All ranks have access to the training data (shared filesystem or pre-downloaded)
- [ ] All ranks have the same PyTorch and NCCL versions
- [ ] Ports are open between nodes (for multi-node runs)

---

## What's Happening: Key Callouts

**Why does rank 0 always do the logging?**
If every rank printed, you would see each log line N times with N different values (e.g., different `local_rank`). Since all ranks have identical loss values (they synchronized gradients), one representative rank is enough. Rank 0 is chosen by convention.

**Why does the loss not improve on the first step of a compiled run?**
The first `torch.compile` step spends most of its time compiling CUDA kernels. The actual forward/backward pass happens, but the reported step time is dominated by compilation. From step 2 onward, the compiled kernels are cached and the step time drops dramatically.

**Why is my MFU lower than expected?**
Common causes: (1) gradient accumulation steps less than 4 (small batches underutilize tensor cores), (2) sequence length not a multiple of 64 (alignment requirement for many kernels), (3) using PyTorch SDPA instead of Flash Attention 3 (especially for long sequences), (4) PCIe-connected GPUs with slow all-reduce.

**What does `torch.set_float32_matmul_precision("high")` do?**
It allows CUDA to use TF32 (19-bit mantissa, 8-bit exponent) for float32 matrix multiplications instead of full FP32. This is a ~8x speedup with negligible accuracy impact on most models. It only affects float32 ops; bfloat16 ops are unaffected.

---

## Check Your Understanding

**Question 1.** In an 8-GPU DDP run, each rank computes a different loss value (because each rank processes different data). After `loss.backward()`, what values do the gradient tensors hold on each GPU? Are they different or the same across ranks? Explain why.

*Answer:* After `loss.backward()` with DDP, all gradient tensors are identical across all ranks. DDP's backward hooks perform an all-reduce (sum then divide by world_size) on each gradient as it is computed. The result — the mean gradient across all data shards — is written back to every GPU. The optimizer then applies this identical gradient everywhere, keeping weights synchronized.

---

**Question 2.** A colleague's distributed training run hangs indefinitely after printing the first few setup messages. They are using 4 GPUs. What is the most likely cause, and what is your first debugging step?

*Answer:* The most likely cause is that the processes are deadlocked at a collective operation — either a barrier that was called on some ranks but not others, or an all-reduce that one rank never reaches (e.g., because it crashed with OOM). The first debugging step is to run with `NCCL_DEBUG=INFO` and redirect each rank's stderr to a separate file (`torchrun ... 2> rank_$RANK.log`). Look for which rank stops making progress and what the last operation it attempted was.

---

**Question 3.** You are about to save a checkpoint at step 1000 in a 4-GPU run. Your checkpoint directory will contain how many files, and what are they?

*Answer:* 4 files (assuming optimizer saving is enabled and the model is small enough that all state fits in single files):
- `model_001000.pt` — model weights, saved by rank 0 only (same across all ranks)
- `meta_001000.json` — training metadata, saved by rank 0 only
- `optim_001000_rank0.pt` — optimizer state for rank 0
- `optim_001000_rank1.pt` — optimizer state for rank 1
- `optim_001000_rank2.pt` — optimizer state for rank 2
- `optim_001000_rank3.pt` — optimizer state for rank 3

That is actually 6 files. (2 shared + 4 per-rank optimizer shards.)

---

## What's Next

Chapter 8 covers supervised fine-tuning (SFT): taking the pretrained base model you have trained across Chapters 6–7 and teaching it to follow instructions by training on conversation data. You will see how the data format changes (prompt/response pairs instead of raw token streams), how the loss mask works (only predict the response tokens, not the prompt), and how the same DDP infrastructure from this chapter is reused with almost no modification.

---

*File locations referenced in this chapter:*
- `nanochat/common.py` — `compute_init()`, `compute_cleanup()`, `get_dist_info()`, `is_ddp_requested()`, `print0()`, `get_peak_flops()`
- `scripts/base_train.py` — full training loop showing DDP usage, gradient accumulation, MFU logging
- `nanochat/checkpoint_manager.py` — `save_checkpoint()`, `load_checkpoint()`
