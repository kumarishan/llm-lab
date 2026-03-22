# Chapter 5: Data Pipeline — Loading, Packing, and Distributing Training Data

## What you'll learn

- Why training data cannot simply be loaded into RAM, and how streaming from parquet files solves that
- What the ClimbMix-400B dataset is, why it is organized in shards, and how nanochat downloads it on demand
- How the BOS-aligned Best-Fit Cropping algorithm packs variable-length documents into fixed-length training rows with zero wasted positions
- How nanochat automatically shards data across multiple GPUs so no two ranks train on the same rows
- How the training state is checkpointed so a crashed run can pick up where it left off

---

## Prerequisites

- Chapter 1 completed: you have the repo installed and have run at least the tiny CPU pipeline
- Chapter 3 completed: you understand what tokens and BOS are, and what the tokenizer produces
- Chapter 4 completed: you understand the GPT forward pass takes a (B, T) tensor of token IDs
- Familiarity with Python generators (`yield`) and basic file I/O

---

## 1. The data challenge

Training an LLM is fundamentally a data-throughput problem.

A GPT-2-grade model needs to see roughly 10 billion tokens during pretraining to reach its capability ceiling. The ClimbMix-400B dataset used in nanochat contains 400 billion tokens — 40x that amount, so you can train multiple times or train larger models without re-using data.

400 billion tokens, stored as UTF-8 text, works out to roughly 1.6 terabytes. Four practical constraints follow from this number:

**You cannot fit the dataset in RAM.** A modern workstation has 64–256 GB of RAM. The dataset is 6–25x that. You must stream it: read a little, process it, move on.

**You cannot tokenize everything in advance.** Pre-tokenized data takes less space than raw text but still amounts to hundreds of gigabytes. nanochat tokenizes text on the fly during training, in parallel with the GPU doing forward/backward passes on the previous batch.

**Multiple GPUs need non-overlapping data.** When you train across 8 GPUs using Distributed Data Parallel (DDP), each GPU is an independent process computing its own forward and backward pass. If all 8 processes read the same rows, you are doing 8x the compute to learn from 1x the data. Each GPU must see a unique slice of the dataset.

**Training runs take hours or days, and things crash.** A power blip, a preempted cloud instance, or an out-of-memory error will kill your training process. If you cannot resume from where you crashed, you lose all the compute spent up to that point. The data pipeline must track its position precisely.

These four constraints — streaming, on-the-fly tokenization, DDP sharding, and resumability — shape every design decision in `dataset.py` and `dataloader.py`.

---

## 2. The ClimbMix-400B dataset

### What it is

ClimbMix-400B is a pretraining dataset assembled by NVIDIA and published in March 2026. It is a curated mixture of web text, books, code, and academic papers, totalling 400 billion tokens. The name "Climb" reflects its goal: to assemble a mixture that gives LLMs a quality advantage at the same token budget compared to raw web crawls.

The dataset lives on HuggingFace Hub at:

```
https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle
```

It is organized as 6,543 parquet files (shards 0 through 6542), each one an independently downloadable chunk. Shard 6542 is reserved as the validation set; all other shards are training data.

### Why 400 billion tokens?

The *Chinchilla scaling laws* (Hoffmann et al., 2022) established an empirical relationship between model size and the optimal number of training tokens. Roughly: for compute-optimal training, you want to train on about 20 tokens per model parameter. A 7-billion-parameter model would want ~140 billion tokens. A GPT-2-sized model (~124M parameters) would want ~2.5 billion tokens.

nanochat targets the GPT-2 capability range (124M–1B parameters). 400B tokens is intentionally far more than needed for a single run, giving you room to experiment with larger models or to train multiple times.

In the tiny CPU run from Chapter 1, you only download 8 shards (~2 GB) — enough for a proof-of-concept run. A full production run uses 170+ shards.

### Parquet format

Parquet is a columnar binary file format designed for efficient storage and retrieval of structured data. "Columnar" means that all values from the same column are stored together on disk, rather than row-by-row as in a CSV. For a dataset with a single `text` column, this means:

- The file is essentially one long compressed block of text strings.
- Reading only the `text` column (which is all we ever need) requires no skipping over other columns.
- Integer and string data compress extremely well in parquet's internal format (roughly 3–5x vs raw text).

Each parquet file is further divided internally into **row groups**: blocks of rows that can be read independently. This is important for nanochat because the DDP sharding happens at the row-group level: rank 0 reads row groups 0, 4, 8, ... and rank 1 reads row groups 1, 5, 9, ... (with 4 GPUs). No seek to byte offsets required; pyarrow handles the row-group indexing.

```
shard_00000.parquet
├── row_group 0  (e.g., 1024 documents)
├── row_group 1  (1024 documents)
├── row_group 2  (1024 documents)
└── ...

shard_00001.parquet
├── row_group 0
└── ...
```

---

## 3. Downloading data: `dataset.py`

### Listing available shards

`list_parquet_files()` in `nanochat/dataset.py` returns the sorted list of parquet files found in the local data directory:

```python
# nanochat/dataset.py

DATA_DIR = os.path.join(base_dir, "base_data_climbmix")

def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir
    # ...
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths
```

The `.tmp` exclusion is important: `download_single_file()` writes to a `.tmp` path first and only renames it to the final name after the download succeeds. This prevents a partially-downloaded corrupt shard from ever being seen as a valid file.

### Downloading a shard

```python
# nanochat/dataset.py

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
index_to_filename = lambda index: f"shard_{index:05d}.parquet"

def download_single_file(index):
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    url = f"{BASE_URL}/{filename}"
    # ...
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            return True
        except (requests.RequestException, IOError) as e:
            # exponential backoff: wait 2^attempt seconds before retry
            wait_time = 2 ** attempt
            time.sleep(wait_time)
    return False
```

Three details worth noting:

**Idempotent:** If the file already exists, the function returns immediately. You can run the downloader script multiple times safely.

**Atomic write:** The file is written to `shard_00001.parquet.tmp`, then `os.rename()` moves it to `shard_00001.parquet`. On POSIX systems, `rename` is atomic. If the process crashes mid-download, only the `.tmp` file is left — and `list_parquet_files()` ignores `.tmp` files.

**Exponential backoff:** If the HuggingFace server returns an error, the retry wait doubles each time (2s, 4s, 8s, 16s). This is standard practice for network-dependent scripts: hammering a server immediately after a failure usually does not help and can make things worse.

### Parallel downloads

The `__main__` block in `dataset.py` uses `multiprocessing.Pool` to download multiple shards in parallel:

```python
# nanochat/dataset.py  (__main__ block)

with Pool(processes=args.num_workers) as pool:
    results = pool.map(download_single_file, ids_to_download)
```

The default is 4 parallel workers. Each worker calls `download_single_file()` independently — there is no shared state to synchronize.

### Lock files for multi-rank coordination

When training starts with `torchrun` (the DDP launcher), multiple processes start simultaneously. All of them will call `list_parquet_files()`. If a shard is missing, they might all try to download it at the same time.

nanochat handles a different but related problem in `nanochat/common.py` using `FileLock`:

```python
# nanochat/common.py

def download_file_with_lock(url, filename, postprocess_fn=None):
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock.
        # All other ranks block until it is released.
        if os.path.exists(file_path):  # recheck after acquiring
            return file_path
        # ... download ...
```

The pattern — check, lock, recheck, act — is the standard double-checked locking idiom:

1. Fast path: if the file exists, return immediately without ever touching the lock.
2. Slow path: acquire the lock. Now only one process can proceed.
3. Recheck inside the lock: another rank might have downloaded the file while we were waiting for the lock. If so, skip the download.
4. Download and write the file.
5. Release the lock: waiting ranks wake up, find the file exists (recheck), and return.

---

## 4. The tokenization step

Once a row group is read from parquet, you have a list of Python strings — the raw document texts. Before the model can do anything with them, they must be converted to token IDs.

From Chapter 3, you know that the tokenizer maps text to a list of integers:

```
"The quick brown fox" -> [450, 4996, 8864, 272]  (example IDs)
```

nanochat also prepends the BOS (Beginning of Sequence) token to every document:

```
"The quick brown fox" -> [1, 450, 4996, 8864, 272]
```

where `1` is the BOS token ID (obtained via `tokenizer.get_bos_token_id()`).

The tokenizer call inside `dataloader.py` processes an entire batch of documents at once, using multiple threads:

```python
# nanochat/dataloader.py  (inside refill_buffer())

token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
```

`tokenizer_threads` defaults to 4. Tokenization is CPU-bound and parallelizes well across threads because each document is independent.

> **Why not pre-tokenize the entire dataset?** Two reasons. First, storage: 400B tokens at 2 bytes per token (int16) is 800 GB. Second, flexibility: if you change the tokenizer vocabulary, you must re-tokenize everything. On-the-fly tokenization costs a small fraction of GPU compute time and removes these constraints entirely.

---

## 5. Packing sequences: the problem

The GPT model takes a fixed-size tensor of shape `(B, T)` — a batch of `B` sequences each of exactly `T` tokens. A typical value is `T = 2048`. Every training step needs a new `(B, T)` tensor.

The raw training data is a stream of variable-length documents. A Wikipedia article might be 5,000 tokens; a web comment might be 12 tokens. The question is: how do you fill a 2048-token row from these variable-length documents?

### Naive approach 1: Truncation

Take each document, trim it to 2048 tokens, discard the rest.

Problems:
- Short documents waste most of the row (a 50-token document leaves 1998 positions empty, filled with padding tokens the model must learn to ignore).
- Long documents are silently truncated: the model never sees the endings of long articles.

### Naive approach 2: Concatenation

Concatenate all documents end-to-end into one long stream, then slice it into 2048-token windows.

This is better — utilization is 100%, no padding needed. But it has a subtle correctness problem: a window might start in the middle of a document. Token 200 of the window came from the end of one document, and token 201 came from the beginning of an entirely different document. The model's attention mechanism will let token 400 attend back to token 201 of a different document. That is not how language works, and it can confuse the model.

### The BOS alignment requirement

The BOS token is the signal the model uses to know "a new document started here." From Chapter 3, you know that in nanochat, every document begins with BOS. The model's attention heads learn to use the presence of BOS as a delimiter.

For this to work correctly in training, every row of the training batch must start with BOS. If a row starts in the middle of a document — say at token 500 of a 3,000-token article — the model sees no BOS, has no positional anchor, and learns something confusing.

This constraint rules out naive concatenation unless you guarantee row boundaries always align with document boundaries — which is exactly what the BOS-aligned Best-Fit Cropping algorithm does.

---

## 6. BOS-aligned Best-Fit Cropping

This is the central algorithm in `dataloader.py`. Understand this section and the rest of the dataloader will make immediate sense.

### Goal

Fill every row of the training batch such that:
- The row begins with BOS (start of a real document)
- The row is exactly T+1 tokens long (one extra token because inputs are `row[:-1]` and targets are `row[1:]`)
- No position in the row is left empty (100% utilization, no padding)

### The buffer

The algorithm maintains a `doc_buffer`: a list of tokenized documents (each a Python list of token IDs, starting with BOS). Before filling a row, the buffer is topped up to `buffer_size` documents (default: 1000). Having many documents in the buffer at once is what makes the "best fit" part of the name possible.

### Algorithm for filling one row

Repeat until the row is full:

1. Let `remaining` = number of positions left unfilled in this row.
2. Scan the buffer for all documents whose length is `<= remaining`.
3. Among those, pick the **longest** one (greedy: "best fit" maximizes how much we fill with each placement).
4. Place it into the row starting at the current position. Advance position by its length. Remove it from the buffer.
5. If no document in the buffer is short enough to fit entirely: **crop** the shortest document to exactly `remaining` tokens, place those tokens, and mark the row complete. Discard the cropped portion.

Step 5 is the "cropping" in the name. The discarded tokens are lost — they will not appear in a future row, because the next row will start fresh with a different document beginning with BOS. This is the source of the ~35% token waste.

### A worked example

Say T = 8 (toy example), so each row needs 9 tokens (T+1 = 9).

Buffer contains four documents (already tokenized, each starting with BOS token `B`):

```
Doc A: [B, a, a, a, a, a]     length 6
Doc B: [B, b, b]               length 3
Doc C: [B, c, c, c, c, c, c]  length 7
Doc D: [B, d, d, d, d]        length 5
```

**Filling row 0 (capacity = 9):**

- remaining = 9. Docs that fit: A (6), B (3), C (7), D (5). Longest: C (7). Place C. remaining = 2.
- remaining = 2. Docs that fit: B (3)? No, 3 > 2. D (5)? No. A (6)? No. Nothing fits.
- Crop: take the shortest remaining doc (B, length 3), use only first 2 tokens: [B, b]. Row is now full.
- Row 0: [B, c, c, c, c, c, c, B, b]   (C placed fully, then B cropped to 2 tokens)
- Discarded from B: [b] (1 token lost). B is consumed.

**State of buffer after row 0:** A, D remain. B is gone (cropped).

**Filling row 1 (capacity = 9):**

- remaining = 9. Docs that fit: A (6), D (5). Longest: A (6). Place A. remaining = 3.
- remaining = 3. Docs that fit: D (5)? No. Nothing fits entirely.
- Crop: D (only doc left), use first 3 tokens: [B, d, d]. Row full.
- Row 1: [B, a, a, a, a, a, B, d, d]
- Discarded from D: [d, d] (2 tokens lost). D is consumed.

Visual summary:

```
Row 0: [ B c c c c c c | B b ]
         ^--- Doc C full ---^ ^crop

Row 1: [ B a a a a a | B d d ]
         ^--- Doc A --^ ^crop

Tokens discarded: 3 out of 18 placed = 17% in this toy example
(real workloads with longer documents approach ~35%)
```

Every row starts with BOS. No row has empty (padding) positions. Every row is exactly 9 tokens. This is what the algorithm guarantees.

### The code

```python
# nanochat/dataloader.py
# tokenizing_distributed_data_loader_with_state_bos_bestfit()

row_capacity = T + 1
# ...
for row_idx in range(B):
    pos = 0
    while pos < row_capacity:
        # Ensure buffer has documents
        while len(doc_buffer) < buffer_size:
            refill_buffer()

        remaining = row_capacity - pos

        # Find largest doc that fits entirely
        best_idx = -1
        best_len = 0
        for i, doc in enumerate(doc_buffer):
            doc_len = len(doc)
            if doc_len <= remaining and doc_len > best_len:
                best_idx = i
                best_len = doc_len

        if best_idx >= 0:
            doc = doc_buffer.pop(best_idx)
            doc_len = len(doc)
            row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
            pos += doc_len
        else:
            # No doc fits - crop shortest in buffer to fill remaining and minimize waste
            shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
            doc = doc_buffer.pop(shortest_idx)
            row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
            pos += remaining
```

A few points to notice:

- `row_capacity = T + 1`, not `T`. The row stores T+1 tokens so we can form input as `row[:-1]` (T tokens) and target as `row[1:]` (T tokens) with a one-position shift.
- In the crop branch, the code picks the *shortest* document in the buffer (not the one being placed, which could be anything). This is a heuristic to minimize waste: cropping a short document means fewer tokens are discarded than cropping a long one.
- `doc_buffer.pop(best_idx)` removes the chosen document from the buffer. The document is consumed in full when placed without cropping, or destroyed (the remainder silently discarded) when cropped.

### Why ~35% waste?

In practice, real document lengths are heavily skewed. The ClimbMix dataset contains many short web documents (blog posts, forum replies, news articles) alongside a smaller number of very long documents. The crop event — which discards tokens — happens roughly once per row on average. The average crop discards about half the tokens from the document being cropped. When aggregated across all rows, this works out to approximately 35% of the total tokens produced by the tokenizer being discarded before the model ever sees them.

35% waste sounds alarming. Is it acceptable? Consider the alternatives:

| Strategy | Utilization | BOS-aligned? | Notes |
|---|---|---|---|
| Truncation | ~10–50% (depends on doc length) | Yes | Wastes most short-doc capacity; loses long-doc endings |
| Naive concatenation | 100% | No | Mid-document row starts confuse attention |
| BOS-aligned best-fit (nanochat) | 100% | Yes | ~35% tokens discarded, all placed tokens are correct |
| Multi-packing with padding | ~70–90% | Yes | Requires attention masks; more complex model code |

BOS-aligned best-fit achieves full utilization and correct alignment at the cost of discarding some tokens. Given that ClimbMix has 400B tokens to burn, this is a worthwhile trade.

---

## 7. Distributed Data Parallel sharding

When training with multiple GPUs, PyTorch uses `torchrun` to launch one process per GPU. Each process gets two environment variables that identify its position:

- `RANK`: global rank (0 to world\_size - 1)
- `WORLD_SIZE`: total number of processes

`nanochat/common.py` reads these via `get_dist_info()`:

```python
# nanochat/common.py

def get_dist_info():
    if is_ddp_requested():
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1
```

In single-GPU or CPU training, this returns `(False, 0, 0, 1)`, so all the sharding logic below reduces to "just read everything."

### Row-group-level sharding

Inside `_document_batches()`, the sharding is applied to row groups within each parquet file:

```python
# nanochat/dataloader.py  (_document_batches)

ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
# ...
rg_idx = ddp_rank  # start from this rank's first row group
while rg_idx < pf.num_row_groups:
    rg = pf.read_row_group(rg_idx)
    batch = rg.column('text').to_pylist()
    # ...
    rg_idx += ddp_world_size  # skip world_size row groups at a time
```

Illustrated with 4 GPUs and a parquet file with 8 row groups:

```
Row group index:  0    1    2    3    4    5    6    7
                  |    |    |    |    |    |    |    |
Rank 0 reads:     0              4              ...
Rank 1 reads:          1              5         ...
Rank 2 reads:               2              6    ...
Rank 3 reads:                    3              7   ...
```

Each rank steps through row groups at stride `world_size`, starting from its own `rank`. The pattern `start=rank, step=world_size` is the interleaved sharding pattern: ranks process the same parquet file simultaneously, reading different row groups.

No two ranks ever read the same row group. Across an entire epoch, each rank sees exactly `1 / world_size` of the data.

---

## 8. Resumable training state

Training a GPT-2-sized model takes several hours. A robust pipeline must be able to restart from the exact point where it stopped.

### What state must be saved?

The data pipeline position is fully described by three numbers:

```python
state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
```

- `pq_idx`: which parquet file we are currently reading (0-indexed into the sorted list of parquet files)
- `rg_idx`: which row group within that file we last yielded
- `epoch`: how many times we have cycled through the full dataset (starts at 1)

The dataloader yields this state dict alongside every batch:

```python
# nanochat/dataloader.py

yield inputs, targets, state_dict
```

The training script saves this state dict into the checkpoint file (together with model weights and optimizer state) after every `N` steps.

### Resuming

To resume training, pass the saved state dict back into the dataloader:

```python
# nanochat/dataloader.py  (_document_batches, resume logic)

resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
resume_epoch  = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
```

On the first pass through the file list, the code starts from `resume_pq_idx` and `resume_rg_idx`. It advances by one row group before reading, to avoid repeating the row group that was already in-flight when the checkpoint was taken:

```python
# nanochat/dataloader.py

base_idx = resume_rg_idx // ddp_world_size
base_idx += 1  # advance by 1 so we don't repeat data after resuming
rg_idx = base_idx * ddp_world_size + ddp_rank
```

The `+ 1` advance means the resume is *approximate*: you skip the row group that was being processed at crash time, losing at most a few thousand documents. For pretraining at the scale of hundreds of billions of tokens, losing a few thousand documents at restart is completely negligible.

> **Why only approximate?** The dataloader maintains a `doc_buffer` of up to 1000 tokenized documents in memory. The buffer state is not saved in the checkpoint — only the parquet position is saved. Even with an exact parquet position, the buffer state cannot be perfectly reconstructed. The `+ 1` advance is a pragmatic acknowledgment of this: skip one row group to avoid any risk of double-counting.

---

## 9. Memory layout and GPU transfer

The dataloader pre-allocates buffers once at startup to avoid repeated memory allocation during training:

```python
# nanochat/dataloader.py

row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
cpu_inputs = cpu_buffer[:B * T].view(B, T)
cpu_targets = cpu_buffer[B * T:].view(B, T)
inputs  = gpu_buffer[:B * T].view(B, T)
targets = gpu_buffer[B * T:].view(B, T)
```

The layout is designed for a single, efficient host-to-device (HtoD) copy per batch:

```
row_buffer (CPU)          shape (B, T+1)   where rows are built
                              |
                              v  slice [:-1] and [1:]
cpu_buffer (CPU, pinned)  shape (2*B*T)    [inputs | targets] interleaved
                              |
                              v  single non-blocking copy
gpu_buffer (GPU)          shape (2*B*T)    [inputs | targets]
```

`pin_memory=True` on the CPU buffer allocates it in page-locked (pinned) memory. The CUDA DMA engine can copy from pinned memory directly to the GPU without an intermediate kernel copy, enabling the `non_blocking=True` HtoD copy at the end:

```python
# nanochat/dataloader.py

cpu_inputs.copy_(row_buffer[:, :-1])    # inputs: tokens 0..T-1
cpu_targets.copy_(row_buffer[:, 1:])   # targets: tokens 1..T
gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
yield inputs, targets, state_dict
```

`non_blocking=True` means the copy is enqueued in the CUDA stream and returns immediately. The GPU can begin the forward pass on the previous batch while this copy runs concurrently in the background. Synchronization happens automatically when the new `inputs` tensor is first accessed.

---

## 10. Putting it all together

Let's trace a full pass from raw text to a training batch.

### Step-by-step

1. **`torchrun` launches** one process per GPU. Each process calls `tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, B=8, T=2048, split="train", ...)`.

2. **`_document_batches()`** begins iterating over parquet files. Rank 0 reads row group 0, rank 1 reads row group 1, etc. From each row group, it extracts the list of raw text strings.

3. **`refill_buffer()`** calls `tokenizer.encode(doc_batch, prepend=bos_token, num_threads=4)`, converting a batch of 128 strings into 128 token-ID lists, each beginning with BOS. These are appended to `doc_buffer`.

4. **Best-fit packing**: the outer loop fills `B = 8` rows, each of T+1 = 2049 positions, using the algorithm described in Section 6. Documents are pulled from the buffer greedily; the final fragment of each row is cropped.

5. **Tensor assembly**: `cpu_inputs` gets `row_buffer[:, :-1]` (first T tokens of each row); `cpu_targets` gets `row_buffer[:, 1:]` (last T tokens of each row). This is the standard language-modeling setup: given the first T tokens, predict the next T tokens, one position shifted.

6. **HtoD copy**: `gpu_buffer.copy_(cpu_buffer, non_blocking=True)` moves both inputs and targets to the GPU in a single transfer.

7. **Yield**: the generator yields `(inputs, targets, state_dict)`. The training loop calls `model(inputs)`, computes cross-entropy loss against `targets`, and backpropagates.

8. **Next batch**: control returns to the generator for the next iteration. The process repeats, pulling new documents from `doc_buffer` (and refilling it as needed from the parquet stream).

### Final shape

```
inputs:  shape (B, T) = (8, 2048)   dtype=torch.long   on GPU
targets: shape (B, T) = (8, 2048)   dtype=torch.long   on GPU
```

Each row of `inputs[i]` is a sequence of 2048 token IDs beginning with BOS. `targets[i]` is the same sequence shifted one position to the right. The model sees `inputs[i]` and must predict `targets[i]` at every position.

---

## Hands-on: explore the pipeline interactively

The following steps let you observe the data pipeline without starting a full training run.

### Step 1: Download a few shards

✍️ Run (downloads shards 0–4 plus the validation shard, ~1.2 GB total):

```bash
python -m nanochat.dataset -n 5
```

Expected output:

```
Downloading 6 shards using 4 workers...
Target directory: /home/you/.cache/nanochat/base_data_climbmix
Downloading shard_00000.parquet...
Downloading shard_00001.parquet...
...
Successfully downloaded shard_00000.parquet
...
Done! Downloaded: 6/6 shards to /home/you/.cache/nanochat/base_data_climbmix
```

### Step 2: Inspect the parquet structure

✍️ Launch a Python REPL and run:

```python
import pyarrow.parquet as pq
from nanochat.dataset import list_parquet_files

paths = list_parquet_files()
print(f"Found {len(paths)} parquet files")

# Inspect the first shard
pf = pq.ParquetFile(paths[0])
print(f"Number of row groups: {pf.num_row_groups}")

# Read the first row group
rg = pf.read_row_group(0)
texts = rg.column('text').to_pylist()
print(f"Documents in row group 0: {len(texts)}")
print(f"First document (first 200 chars):\n{texts[0][:200]}")
```

You should see something like:

```
Found 6 parquet files
Number of row groups: 16
Documents in row group 0: 1024
First document (first 200 chars):
The Global Fund to Fight AIDS, Tuberculosis and Malaria is an international financing institution...
```

### Step 3: Observe BOS-aligned packing

✍️ Run:

```python
from nanochat.tokenizer import Tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

tokenizer = Tokenizer()
bos_id = tokenizer.get_bos_token_id()
loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, B=2, T=64, split="train", device="cpu"
)

inputs, targets = next(loader)
print(f"inputs shape: {inputs.shape}")    # (2, 64)
print(f"targets shape: {targets.shape}")  # (2, 64)

for row_idx in range(2):
    row = inputs[row_idx].tolist()
    bos_positions = [i for i, t in enumerate(row) if t == bos_id]
    print(f"Row {row_idx}: BOS at positions {bos_positions}")
    print(f"  First 10 tokens: {row[:10]}")
```

You will see that every row's first token is the BOS token ID, and BOS appears at multiple positions within each row (marking document boundaries). All positions after the last BOS are the beginning of the next document.

### Step 4: Observe state tracking

✍️ Run:

```python
from nanochat.tokenizer import Tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

tokenizer = Tokenizer()
loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B=2, T=64, split="train", device="cpu"
)

for step in range(5):
    inputs, targets, state = next(loader)
    print(f"Step {step}: pq_idx={state['pq_idx']}, rg_idx={state['rg_idx']}, epoch={state['epoch']}")
```

You will see `rg_idx` advancing (in steps of `world_size`, which is 1 in single-process mode) as the loader moves through row groups.

---

## What's happening (summary callout)

> Every batch of training data goes through this chain:
>
> 1. Parquet shard file on disk
> 2. Row group read by `pyarrow` (interleaved by rank)
> 3. Text strings tokenized by BPE tokenizer (prepend BOS, parallel threads)
> 4. Token lists placed into `doc_buffer`
> 5. Best-fit greedy packing fills a `(B, T+1)` row buffer (crop when nothing fits)
> 6. Inputs = `row[:, :-1]`, Targets = `row[:, 1:]`
> 7. Pinned-memory HtoD copy delivers `(B, T)` tensors to the GPU
> 8. Training loop calls `loss = model(inputs, targets)` and backpropagates
>
> The state dict `{pq_idx, rg_idx, epoch}` is checkpointed at every save interval, enabling exact-ish resume.

---

## Check your understanding

1. **Parquet row groups:** A parquet file has 64 row groups and you are running with 8 GPUs. How many row groups will rank 3 read during one pass through this file? Which row group indices will it read?

2. **Cropping math:** Suppose `T = 2048` and the best-fit algorithm has placed documents totaling 2040 tokens into a row, leaving 9 positions unfilled (remember, row capacity is T+1 = 2049). The smallest document in the buffer is 200 tokens long. How many tokens from that document will be placed into the row? How many will be discarded?

3. **State resume:** You checkpoint training after step 1000. The saved state dict is `{"pq_idx": 3, "rg_idx": 24, "epoch": 1}`. Training crashes at step 1050 and you restart from the step-1000 checkpoint. Explain what `_document_batches()` does on startup with this state dict. Specifically: which row group does rank 0 begin reading from, and why does the code advance by 1 before reading?

---

## What's next

Chapter 6 covers the training loop in `scripts/base_train.py`: how model weights are updated step by step, what gradient accumulation does, how the learning rate schedule works, and how all the pieces covered in Chapters 3–5 (tokenizer, architecture, data pipeline) connect into a working training run.
