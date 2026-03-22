# Chapter 10: Reinforcement Learning — GRPO, Rewards, and Tool Use

## What you'll learn

- Why reinforcement learning (RL) is used after SFT, and what problems it solves that imitation learning cannot
- How the REINFORCE policy gradient algorithm works mathematically, and why it applies to discrete token sequences
- What GRPO is — and how nanochat's version simplifies it to on-policy REINFORCE with group-relative advantage normalization
- How GSM8K is framed as an RL task, including the `<<expr=result>>` tool-call format
- How nanochat sandboxes model-generated Python code, and why sandboxing matters
- How to read and run `scripts/chat_rl.py` end to end

## Prerequisites

- Chapters 1–9 completed
- Chapter 9 in particular: you need a chat model produced by SFT as the starting point
- Comfortable with PyTorch autograd and the concept of a loss function
- No prior RL background required; all necessary theory is introduced here

---

## 10.1 Why RL after SFT?

Chapter 9 showed you supervised fine-tuning (SFT): the model is trained to imitate human-written answers. The learning signal is purely "copy this text exactly." That approach works well, but it has a hard ceiling.

### The imitation problem

Consider a math word problem. A human expert writes a solution, and the model is trained to reproduce it token by token. This gives the model one path to the answer — the path the human took. But math problems often have many valid solution strategies. SFT will never show the model a strategy that no human annotator happened to write down.

More critically, SFT gives the model no feedback about whether its answers are actually *correct*. If the model generates a plausible-sounding but wrong solution, there is no signal during training to penalize that. The model learns the *style* of correct solutions, not the *verification* of them.

### The RL solution

Reinforcement learning replaces imitation with outcome-based feedback:

1. Let the model generate its own answer.
2. Check if the answer is correct (using a verifier, not a human in the loop).
3. Reward correct answers; penalize or ignore incorrect ones.
4. Update the model to make correct answers more likely.

For tasks with a verifiable ground truth — math, code, formal logic — this is extremely powerful. The model receives a reward signal for every problem it attempts, not just for the small set of problems that had a human demonstration.

### An example: learning to use a calculator

During SFT, a model learns to write `<<12/60=0.2>>` because it saw examples of that pattern in the training data. During RL on GSM8K, the model can discover that writing calculator calls leads to higher rewards even if some training examples didn't include them. The model is incentivized by outcomes, not by imitation.

> **Why RL works here but not everywhere**
>
> RL on language models requires a reward signal that can be computed automatically. Math answers, code test outcomes, and structured format checks all qualify. Open-ended questions ("write me a poem") do not have an automatic verifier — those require human feedback (RLHF) or a learned reward model, both of which are more complex. nanochat focuses on the simpler, fully-verifiable case.

---

## 10.2 The RL setup for language models

Before we look at any code, let's establish the vocabulary that RL researchers use, mapped to the LLM context.

### The MDP framing

| RL concept | Meaning in this context |
|---|---|
| **Agent** | The language model |
| **Policy π** | The model's probability distribution over next tokens given a context |
| **State s** | The token sequence seen so far (prompt + partial completion) |
| **Action a** | The next token chosen (one of 32 768 possible vocabulary tokens) |
| **Episode / trajectory τ** | A full completion, from the first generated token to EOS |
| **Reward R(τ)** | Did the final answer match the ground truth? (0 or 1) |

The episode is the full assistant response. The reward is assigned once, at the end, when the answer can be checked. There is no intermediate reward for individual tokens — only a final binary signal.

### The objective

We want to maximize the *expected reward* over all trajectories the policy might generate:

```
J(θ) = E_{τ ~ π_θ} [R(τ)]
```

where `θ` are the model parameters and `τ` is a sampled trajectory (sequence of tokens). The expectation is over the randomness of sampling: different runs of top-k sampling produce different completions.

### Why this is hard to optimize directly

The reward `R(τ)` is a property of the complete output string. It's not differentiable with respect to the model parameters — you can't backpropagate through "is this the right number?" To work around this, we need the policy gradient theorem.

---

## 10.3 REINFORCE: the policy gradient algorithm

The core result of policy gradient methods is a formula that lets us compute a gradient of the expected reward even when the reward is not differentiable.

### Derivation

```
∇_θ J(θ) = ∇_θ E_{τ ~ π_θ} [R(τ)]
```

Expand the expectation as an integral over all trajectories:

```
= ∇_θ ∫ π_θ(τ) R(τ) dτ
```

Apply the log-derivative trick (`∇_θ π_θ = π_θ ∇_θ log π_θ`):

```
= ∫ π_θ(τ) ∇_θ log π_θ(τ) R(τ) dτ
```

Recognize this as another expectation:

```
= E_{τ ~ π_θ} [R(τ) · ∇_θ log π_θ(τ)]
```

This is the **policy gradient theorem**. The gradient of the expected reward equals the expected product of the reward and the gradient of the log-probability of the trajectory.

### Log-probability of a trajectory

A trajectory is a sequence of tokens `(t_1, t_2, ..., t_T)`. By the chain rule of probability:

```
log π_θ(τ) = Σ_{i=1}^{T} log P(t_i | t_1, ..., t_{i-1})
```

This is exactly what the language model computes — the sum of per-token log-probabilities. That sum is differentiable through the softmax, so we can backpropagate through it.

### The REINFORCE update rule

Sample a trajectory τ, observe its reward R(τ), then update:

```
θ ← θ + α · R(τ) · ∇_θ log π_θ(τ)
```

Equivalently, define the loss to *minimize*:

```
L_PG = -R(τ) · Σ_{i=1}^{T} log P(t_i | context)
```

Minimizing this loss (gradient descent) is equivalent to gradient ascent on the expected reward.

### Intuition

- If the reward is high (R = 1, correct answer): the gradient increases the log-probability of each token in the sequence. The model becomes more likely to generate this completion.
- If the reward is low (R = 0, wrong answer): the gradient is zero for that completion. Nothing changes. (This is different from PPO, which actively suppresses bad completions.)
- Over many examples, completions that tend to be correct accumulate positive gradient signal. Completions that tend to be wrong receive no positive signal and are relatively suppressed.

This is the core mechanic: **reinforce what works, ignore what doesn't.**

> **Why this works with discrete tokens**
>
> The key insight is the log-derivative trick. We never backpropagate *through* the sampling operation (which is non-differentiable). Instead, we backpropagate through the log-probability that the model assigned to the token it happened to sample. The actual token choice is treated as a fixed sample; only the probability assigned to it is trained.

---

## 10.4 Baseline subtraction and advantage

### The variance problem

There's a practical problem with REINFORCE as described: if R(τ) is always non-negative (as it is here — reward is 0 or 1), then the gradient always pushes probabilities *up*. The model never gets a signal to do *less* of something. The gradient estimator has very high variance because it depends on whether any particular rollout happened to succeed.

To see why this matters: if the model is already getting 90% of problems right, the remaining 10% still receive gradient signal pushing up their (usually wrong) completions. The signal is noisy.

### The baseline

The standard fix is to subtract a baseline `b` from the reward:

```
A(τ) = R(τ) - b    (the "advantage")
```

The baseline doesn't change the *expected* gradient (it can be shown that `E[b · ∇ log π] = 0` as long as `b` doesn't depend on the current trajectory), but it reduces variance significantly.

A good baseline is the mean reward for the same problem across multiple rollouts. If the model gets 0.7 of a group of rollouts correct, then:
- A correct rollout has advantage `1.0 - 0.7 = +0.3` (better than expected)
- An incorrect rollout has advantage `0.0 - 0.7 = -0.7` (worse than expected)

Now both correct and incorrect rollouts contribute meaningful gradient signals.

### DAPO normalization (what nanochat uses)

nanochat uses a specific variant called DAPO-style normalization. For each group of rollouts on the same problem, compute:

```
μ = mean(r_1, r_2, ..., r_G)       # mean reward across the group
A_i = r_i - μ                       # advantage for each rollout
```

Note what is **not** done here: there is no division by the standard deviation (no z-score). A true z-score normalization would be `(r_i - μ) / σ`. The choice to omit division by σ is deliberate: when all rollouts have the same reward (σ = 0, e.g., all correct or all wrong), z-score would divide by zero or produce NaN. DAPO avoids this degenerate case by only centering, not scaling.

From `scripts/chat_rl.py`:

```python
# Calculate the advantages by simply subtracting the mean (instead of z-score (x-mu)/sigma)
mu = rewards.mean()
advantages = rewards - mu
```

This is applied at the sequence level (one advantage per rollout). The same advantage value is then broadcast to every token in that rollout.

### Token-level normalization

The advantage `A_i` is a scalar for the whole sequence. In the loss computation, it multiplies the per-token log-probabilities:

```
L = -Σ_i A_i · Σ_j log P(t_j^(i) | context)
```

The "token-level" aspect means the normalization divisor in the code is `num_valid_tokens` (the total number of trained-on tokens across all sequences in the batch), not the number of sequences. This prevents shorter sequences from dominating the gradient. Sequences that hit EOS early naturally contribute fewer token-level gradient updates.

---

## 10.5 GRPO: Group Relative Policy Optimization

GRPO is the algorithm nanochat implements, though as the file header of `chat_rl.py` honestly notes, the implementation is closer to REINFORCE with group-relative baselines than to the full GRPO paper.

### The group idea

For each training problem, generate `G` independent completions (nanochat defaults to `--num-samples 16`). Evaluate the reward for each. Compute advantages relative to the group mean:

```
μ_group = (1/G) Σ_{i=1}^{G} r_i
A_i = r_i - μ_group
```

This is exactly what we described in section 10.4, applied per-question.

### What GRPO drops (compared to PPO)

Full GRPO as described in the DeepSeekMath paper includes:

1. A KL penalty term to keep the policy from drifting too far from a reference model
2. An importance sampling ratio `π_θ(a) / π_θ_old(a)` because GRPO in the original paper reuses rollouts across multiple gradient steps
3. Clipping of that ratio (PPO-style) to prevent large updates

nanochat's implementation drops all three:

- **No KL penalty:** No reference model is loaded. The training signal alone is assumed to keep the model from collapsing.
- **No importance sampling ratio:** Because nanochat is *on-policy* — it generates fresh rollouts at every step using the current weights — the old and new policies are identical at the time of the gradient update, so the ratio is exactly 1.0 and can be omitted.
- **No clipping:** Same reason. Clipping is a guard against large updates when reusing stale rollouts. With fresh rollouts, it's unnecessary.

The result is mathematically equivalent to REINFORCE with a group-relative baseline, evaluated on-policy. The comment in `chat_rl.py` is explicit about this:

```python
# Note, there is no need to add PPO ratio+clip because we are on policy
```

### The GRPO objective (as implemented)

For a batch of `N` problems, each with `G` rollouts:

```
L_GRPO = -(1 / N_tokens) · Σ_i Σ_j A_i · log P_θ(t_j^(i) | context)
```

where:
- `i` indexes rollouts
- `j` indexes tokens within rollout `i`
- `A_i = r_i - mean(r_group)` for the group containing rollout `i`
- `N_tokens` is the total number of valid (non-padding, non-prompt) tokens

---

## 10.6 On-policy training

The phrase "on-policy" means the data used for the gradient update was generated by the same model that will be updated.

### The rollout-update cycle

```
Step t:
  1. Sample the current model weights θ_t
  2. Generate G completions for each of the B problems using π_{θ_t}
  3. Compute rewards and advantages
  4. Compute gradient and update: θ_{t+1} = θ_t - α · ∇L
  (The rollouts from step t are now stale — never reused)

Step t+1:
  1. Generate fresh rollouts using π_{θ_{t+1}}
  ...
```

Each step requires a fresh batch of rollouts. This is computationally expensive (generation is slow), but it keeps the training data distribution aligned with the current policy.

### Why not reuse rollouts?

Suppose you generated rollouts using an old policy `π_old` and now want to update a new policy `π_new`. The data distribution is wrong — `π_old` over-represents some completions and under-represents others relative to `π_new`. To correct for this, you'd need importance weights `π_new(τ) / π_old(τ)`. That ratio is what PPO introduces — but computing it adds complexity, and clipping it introduces bias. nanochat avoids the entire issue by always using the current policy.

The practical consequence: generating rollouts is the bottleneck. The `get_batch()` generator in `chat_rl.py` calls `engine.generate_batch()` before every gradient step. The model is in `eval()` mode during generation, then switched to `train()` mode for the backward pass.

---

## 10.7 GSM8K: math reasoning with tool use

### The dataset

GSM8K (Grade School Math 8K) is a benchmark of 8 500 grade-school math problems created by OpenAI. Each problem requires 2–8 reasoning steps. Problems are stated in plain English; answers are integers or simple decimals.

Example problem from the dataset:

```
Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting.
How much did she earn?

Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10
```

Two things to notice:

1. **The `<<expr=result>>` inline tool call format.** Wherever arithmetic is performed, the dataset includes the expression and its evaluated result. This is the "calculator use" format.
2. **The `#### number` marker.** The final numerical answer always follows `####`. This is what the evaluation code extracts.

### Parsing the tool call format (`tasks/gsm8k.py`)

`GSM8K.get_example()` parses each training example into nanochat's internal conversation format. The assistant message is not a plain string — it's a list of typed parts:

```python
parts = re.split(r'(<<[^>]+>>)', answer)
for part in parts:
    if part.startswith('<<') and part.endswith('>>'):
        inner = part[2:-2]  # Remove << >>
        if '=' in inner:
            expr, result = inner.rsplit('=', 1)
        else:
            expr, result = inner, ""
        assistant_message_parts.append({"type": "python", "text": expr})
        assistant_message_parts.append({"type": "python_output", "text": result})
    else:
        assistant_message_parts.append({"type": "text", "text": part})
```

A `python` part contains the expression (e.g., `12/60`). A `python_output` part contains the result (e.g., `0.2`). A `text` part contains surrounding prose. During SFT training (Chapter 9), the model learned to generate text and python parts; the python_output parts were provided by the tokenizer/engine, not trained on.

### Answer extraction (`tasks/gsm8k.py`)

```python
GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")  # normalize: "1,000" -> "1000"
        return match_str
    return None
```

The regex finds `#### <number>` anywhere in the string. Commas are stripped to normalize formats like `1,000` to `1000`. If no `####` marker is found, the function returns `None` — and a comparison against the ground truth will fail, giving reward 0.

### Reward function (`tasks/gsm8k.py`)

```python
def evaluate(self, conversation, assistant_response):
    last_text_part = conversation['messages'][-1]['content'][-1]['text']
    ref_num = extract_answer(last_text_part)
    pred_num = extract_answer(assistant_response)
    is_correct = int(pred_num == ref_num)
    return is_correct

def reward(self, conversation, assistant_response):
    is_correct = self.evaluate(conversation, assistant_response)
    return float(is_correct)
```

The reward is simply binary: `1.0` if the extracted predicted number exactly matches the extracted ground-truth number, `0.0` otherwise. No partial credit; no format bonus.

This binary reward is important for understanding the training dynamics. The model either gets a problem right or it doesn't. The group-relative advantage then tells the model whether a particular completion was *better or worse than its other attempts* on the same problem.

---

## 10.8 Sandboxed Python execution

### Why a sandbox?

During RL rollouts, the model generates Python expressions to evaluate. These are fed to a Python `exec()` call. Executing arbitrary model-generated code without restrictions is dangerous:

- The model might generate `import os; os.system("rm -rf /")` (destructive filesystem operation)
- It might open network connections, fork processes, or consume unbounded memory
- It might enter an infinite loop and hang the training process

nanochat's `nanochat/execution.py` implements a defense-in-depth sandbox to mitigate these risks.

### The subprocess boundary

The primary protection is process isolation. All execution happens in a separate Python process:

```python
# nanochat/execution.py — execute_code()
manager = multiprocessing.Manager()
result_dict = manager.dict()

p = multiprocessing.Process(
    target=_unsafe_execute,
    args=(code, timeout, maximum_memory_bytes, result_dict)
)
p.start()
p.join(timeout=timeout + 1)

if p.is_alive():
    p.kill()
    return ExecutionResult(..., timeout=True)
```

If the subprocess hangs or crashes, the parent process kills it and reports a timeout. The main training process is never at risk of hanging.

### Resource limits

Inside `_unsafe_execute`, before running any code, `reliability_guard()` is called:

```python
# On Linux (not macOS):
resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))
```

The default memory limit is 256 MB. If the generated code tries to allocate more (e.g., a list comprehension producing a billion elements), it hits a `MemoryError`.

A signal-based timer provides a hard execution timeout:

```python
# nanochat/execution.py — time_limit()
signal.setitimer(signal.ITIMER_REAL, seconds)
signal.signal(signal.SIGALRM, signal_handler)
```

After the timeout elapses, `SIGALRM` fires, which raises `TimeoutException`. This catches infinite loops even if they don't allocate new memory.

### Function-level disabling

`reliability_guard()` also replaces dangerous functions with `None` in their modules:

```python
os.kill = None
os.system = None
os.remove = None
os.removedirs = None
os.fork = None
shutil.rmtree = None
shutil.move = None
subprocess.Popen = None
```

If generated code calls `os.system(...)`, it raises `TypeError: 'NoneType' object is not callable`. This blocks the most obvious destructive patterns.

### ExecutionResult

The result of every execution attempt is returned as a dataclass:

```python
@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    error: Optional[str] = None
    timeout: bool = False
    memory_exceeded: bool = False
```

The training code only cares about `success` and `stdout` (the computed result). The other fields are useful for debugging.

### What the sandbox does *not* protect

The file header of `execution.py` is honest about the limitations:

- **Network access is not blocked.** The model could open a socket.
- **Python's dynamic features could bypass restrictions.** `ctypes` can call C functions that bypass Python-level restrictions.
- **No kernel-level isolation.** There is no `seccomp` filter, no container, no VM boundary.

The sandbox is appropriate for evaluation of model-generated code where the model is not adversarial. It is not appropriate for executing code from untrusted external users.

### How the Engine uses the calculator

Note that during generation (in `nanochat/engine.py`), the calculator tool uses a *different*, lighter mechanism than `execute_code`. The `Engine` uses `eval_with_timeout` via `use_calculator()`:

```python
# nanochat/engine.py
def use_calculator(expr):
    if all([x in "0123456789*+-/.() " for x in expr]):
        return eval_with_timeout(expr)
    ...
```

This is a synchronous, in-process evaluation. It's faster than spawning a subprocess for every arithmetic step during generation. The full `execute_code` sandbox from `execution.py` is available for more complex Python code evaluation and is used in other task contexts.

---

## 10.9 The RL training loop in chat_rl.py

Let's walk through the code structure of `scripts/chat_rl.py` in detail.

### Initialization

```python
# scripts/chat_rl.py
model, tokenizer, meta = load_model("sft", device, phase="eval",
                                    model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer)
```

The script loads an SFT checkpoint (the output of Chapter 9). The `Engine` wraps the model for efficient batched generation with KV-cache and tool use.

```python
train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
```

GSM8K is loaded for both training (7 473 examples) and testing (1 319 examples). The number of optimizer steps is computed from the dataset size, examples per step, and number of epochs.

### The rollout generator: `get_batch()`

The function `get_batch()` is a Python generator decorated with `@torch.no_grad()`. It yields one batch at a time, producing rollouts on demand. The training loop calls `next(batch_iterator)` each step.

**Step 1: Get a problem from the dataset**

```python
# scripts/chat_rl.py — get_batch()
conversation = train_task[example_idx]
tokens = tokenizer.render_for_completion(conversation)
prefix_length = len(tokens)
```

`render_for_completion` returns the tokenized prompt with the assistant turn primed: the `<|assistant_start|>` token is present, but the assistant's answer is not. The model will complete from here.

**Step 2: Generate G completions**

```python
for sampling_step in range(num_sampling_steps):
    seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
    generated_token_sequences_batch, masks_batch = engine.generate_batch(
        tokens,
        num_samples=args.device_batch_size,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=seed,
    )
```

`generate_batch` internally handles the tool-use state machine: when the model emits a `<|python_start|>` token, the engine captures subsequent tokens as an expression, evaluates it with `use_calculator`, and force-injects the result tokens wrapped in `<|output_start|>` / `<|output_end|>` tokens.

The `seed` is derived from `(step, example_idx, sampling_step)` so that re-running the same step with the same model produces identical rollouts — important for reproducibility.

**Step 3: Compute rewards**

```python
for sample_tokens in generated_token_sequences:
    generated_tokens = sample_tokens[prefix_length:]
    generated_text = tokenizer.decode(generated_tokens)
    reward = train_task.reward(conversation, generated_text)
    rewards.append(reward)
```

Each completion is decoded to text, and `GSM8K.reward()` checks whether the `#### answer` in the generated text matches the ground truth. Reward is `1.0` or `0.0`.

**Step 4: Prepare targets with masking**

```python
targets = ids[:, 1:].clone()
targets[mask_ids[:, 1:] == 0] = -1  # -1 = ignore_index
```

The mask is `0` for prompt tokens and for forced tokens (tool outputs). By setting those positions to `-1`, the model loss at those positions is zeroed out. The model is only trained on tokens it actually *chose* to generate, not on the boilerplate of the tool call response.

**Step 5: Compute advantages**

```python
mu = rewards.mean()
advantages = rewards - mu
```

Mean reward over the `num_samples` completions of this problem. Each completion's advantage is its reward minus the mean.

### The training step

```python
# scripts/chat_rl.py — training loop
for example_step in range(examples_per_rank):
    sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
    model.train()
    for pass_idx in range(num_passes):
        inputs = inputs_all[b0:b1]
        targets = targets_all[b0:b1]
        advantages = advantages_all[b0:b1]

        logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)  # (B, T)
        pg_obj = (logp * advantages.unsqueeze(-1)).sum()
        num_valid = (targets >= 0).sum().clamp(min=1)
        pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
        loss = -pg_obj
        loss.backward()
```

Walk through this:

1. `model(inputs, targets, loss_reduction='none')` runs a forward pass and returns the per-token NLL (negative log-likelihood). Negating it gives per-token log-probability.
2. `.view_as(inputs)` reshapes to `(B, T)` — one log-prob per token per sequence.
3. `advantages.unsqueeze(-1)` broadcasts the per-sequence advantage across all T token positions.
4. The sum computes `Σ_i Σ_j A_i · logp(t_j^(i))` — the policy gradient objective.
5. Dividing by `num_valid` normalizes by the number of trained-on tokens. Dividing by `num_passes * examples_per_rank` normalizes by the batch size.
6. Negating gives the loss (we minimize the negative objective).
7. `.backward()` computes gradients. No optimizer step yet — gradients accumulate over `examples_per_rank` examples.

After the inner loop, one optimizer step is taken:

```python
lrm = get_lr_multiplier(step)
for group in optimizer.param_groups:
    group["lr"] = group["initial_lr"] * lrm
optimizer.step()
model.zero_grad(set_to_none=True)
```

The learning rate follows a simple linear decay from `init_lr_frac * base_lr` to 0 over `num_steps`.

### Evaluation

Every `--eval-every` steps, the model is evaluated on a subset of the GSM8K test set:

```python
# scripts/chat_rl.py — run_gsm8k_eval()
for k in range(1, args.device_batch_size + 1):
    passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
```

**Pass@k** is the fraction of problems where at least one of the first `k` samples is correct. Pass@1 is the standard greedy accuracy metric. Pass@k (k > 1) measures whether the model *can* solve the problem with multiple attempts, which is a more generous upper bound.

### What metrics to watch

| Metric | What it tells you |
|---|---|
| `reward` (per step) | Average reward over rollouts this step; should trend upward |
| `pass@1` | Greedy accuracy on test set; the primary quality metric |
| `pass@k` (k > 1) | How often the model can solve it with multiple tries |
| `sequence_length` | Average length of generated completions; watch for collapse (very short) or runaway generation |
| `lrm` | Learning rate multiplier; should decrease from 1.0 to 0.0 linearly |

If `reward` is flat and `pass@1` is not improving, the model may be stuck. Check that rollouts are reaching the `####` marker — if the model never generates the answer marker, all rewards are zero and advantages are all zero, giving zero gradient.

---

## 10.10 Running chat_rl.py

### Prerequisites

You need an SFT checkpoint from Chapter 9. By default the script looks for a model tagged `sft`. If you saved it under a different tag, pass `--model-tag <your-tag>`.

### Single GPU

✍️
```bash
python -m scripts.chat_rl \
  --run my-rl-run \
  --num-epochs 1 \
  --num-samples 16 \
  --examples-per-step 16 \
  --max-new-tokens 256
```

This generates 16 samples per problem, processes 16 problems per optimizer step, and runs one pass through the training set.

### Key flags

| Flag | Default | What it controls |
|---|---|---|
| `--num-samples` | 16 | Completions per problem (group size G) |
| `--examples-per-step` | 16 | Problems per optimizer step |
| `--max-new-tokens` | 256 | Max tokens generated per completion |
| `--temperature` | 1.0 | Sampling temperature; use 1.0 for diversity during training |
| `--eval-every` | 60 | Steps between pass@k evaluations |
| `--model-tag` | None | Tag of the SFT checkpoint to load |

### Multi-GPU

✍️
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run default
```

In the distributed case, each GPU handles a different slice of the training data (`rank_indices = range(ddp_rank, len(train_task), ddp_world_size)`). Reward aggregation uses `dist.all_reduce` so that the logged metrics reflect the full training batch across all GPUs.

### Expected training dynamics

During the first few hundred steps you should expect:

- `reward` starts somewhere around the SFT model's baseline accuracy on training problems (often 0.3–0.5 for a mid-size model)
- `reward` should gradually increase as the model learns to write `#### answer` reliably and to use the calculator for arithmetic
- `pass@1` on the test set may initially fluctuate before trending upward
- `sequence_length` may increase as the model learns to generate more complete reasoning chains

If `reward` stays near zero for many steps, check that the model is generating the `####` marker at all. You can decode a few rollouts manually by adding a print statement inside `get_batch()`.

### Checkpoints

Checkpoints are saved to `chatrl_checkpoints/<model_tag>/` every `--save-every` steps, and always at the final step. Unlike SFT checkpoints, optimizer state is not saved (the comment in the code says "we don't bother to save the optimizer state"), so you cannot resume an interrupted RL run from mid-training.

---

## 10.11 What RL can and can't fix

### Where RL excels

**Verifiable tasks.** Anything where "correct" has an automatic definition is a natural fit:
- Math (GSM8K, MATH, competition problems)
- Code (does it pass test cases?)
- Formal proofs (does the proof checker accept it?)
- Structured output tasks (does the output parse to valid JSON?)

For these tasks, RL can find solution strategies that SFT never demonstrated. The model can also learn reliability: not just *sometimes* producing the right answer but consistently doing so.

**Tool use.** RL is particularly good at teaching reliable tool use. A model trained with SFT on calculator examples learns to imitate the format. A model trained with RL on math problems learns that using the calculator *actually helps get the reward*. The incentive is aligned with correct behavior.

### Where RL struggles

**Knowledge.** RL cannot teach the model new facts. If the model doesn't know that the speed of light is 3×10^8 m/s, no reward signal can insert that fact — RL only reshapes the probability distribution over existing knowledge. For knowledge acquisition, you need pretraining or retrieval.

**Open-ended quality.** If you can't define a reward function automatically, you can't do RL. Fluency, style, creativity, and helpfulness are hard to quantify. This is why RLHF (RL from Human Feedback) uses a learned reward model trained on human preferences — but that introduces its own failure modes.

**Reward hacking.** The model will find the shortest path to high reward, which sometimes isn't the intended behavior. For example, a model trained on GSM8K might learn to hallucinate plausible-looking `#### 42` markers without doing the reasoning, if that happens to get some rewards. Binary rewards are somewhat resistant to this — a hallucinated answer usually fails verification — but the risk increases if the reward function has exploitable shortcuts.

**Distributional shift.** As the policy changes, the distribution of generated text changes. Problems that were easy early in training may become hard if the model drifts away from the right format. This is the fundamental instability of RL on language models, and it's why PPO clips the update ratio and why the original GRPO paper adds a KL penalty. nanochat's simpler approach works because the updates are small (fresh on-policy rollouts, small learning rate).

### The alignment angle

RLHF (RL from Human Feedback) is how GPT-4, Claude, and most commercial models are made helpful and safe. The mechanism is the same as what you've seen here:

1. A reward model is trained on human preference data (pairs of model outputs, labeled by which is better).
2. That reward model scores completions during RL training.
3. REINFORCE or PPO updates the policy to maximize predicted human preference.

The key insight — that you can train language models with reward signals that don't require differentiating through the reward — is exactly the policy gradient theorem from section 10.3. nanochat's GSM8K RL is a clean, small-scale version of the same machinery.

---

## Check your understanding

**Question 1:** In nanochat's RL loop, the mask `mask_ids` is set to 0 for both prompt tokens and tool-output tokens. Why is it important that tool-output tokens are excluded from training, even though they appear in the middle of the assistant's response?

**Question 2:** Suppose every rollout for a given problem gets reward 1.0 (the model already solves this problem reliably). What is the advantage for each rollout? What does this mean for the gradient update on this problem?

**Question 3:** The `execute_code` function in `execution.py` uses `multiprocessing.Process` and sets `p.join(timeout=timeout + 1)`. Why is the join timeout set to `timeout + 1` (one second longer than the code execution timeout), rather than exactly `timeout`?

---

## What's next

This chapter completes the core training pipeline: pretraining, SFT, and RL fine-tuning. The model can now learn from outcomes, not just imitation.

Possible directions from here:

- **Extend the reward function.** `GSM8K.reward()` is deliberately simple. You could add partial credit for correct reasoning format, or penalize excessive length.
- **Try a harder benchmark.** GSM8K is designed for small models. The MATH dataset is significantly harder and rewards more careful multi-step reasoning.
- **Add a learned reward model.** Replace the binary math reward with a trained classifier — the first step toward RLHF.
- **Experiment with the group size.** Try `--num-samples 4` vs `--num-samples 32`. Larger groups give lower-variance advantage estimates but require more generation compute per step.
- **Look at what the model learned.** Compare rollouts from the SFT model and the RL model on the same problem. Does the RL model use the calculator more consistently? Does it reliably write the `####` marker?
