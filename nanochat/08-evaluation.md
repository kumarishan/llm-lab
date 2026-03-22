# Chapter 8: Evaluation — Bits-Per-Byte, CORE, and In-Context Learning

## What you'll learn

- Why raw training loss is insufficient, and what properties a good evaluation metric must have
- How Bits-Per-Byte (BPB) turns cross-entropy loss into a number that is invariant to tokenizer vocabulary size
- What in-context learning (ICL) is and why it is the right lens for measuring a pretrained model's capabilities
- How the CORE benchmark evaluates language understanding through four complementary tasks using likelihood scoring
- How to run `scripts/base_eval.py` to obtain BPB, CORE score, and sample-based outputs for any checkpoint

## Prerequisites

- Chapters 1–7: you have a trained base model checkpoint
- You understand cross-entropy loss and nats-per-token from Chapter 2
- You understand how tokenization turns text into integer ids from Chapter 3
- You understand the pretraining loop from Chapter 6

---

## 8.1 Why raw training loss is not enough

After a few thousand gradient steps, training loss reliably falls. That is a good sign, but it is not a sufficient signal. Three problems make raw loss a poor sole evaluation criterion.

**Problem 1: Memorisation vs. generalisation.** A model that memorises its training data achieves zero loss on training examples but will fail on anything new. You need loss on a held-out validation set. Even then, if your validation set shares the same domain, a trivially overfit model can look impressive.

**Problem 2: Vocab-size dependence.** Cross-entropy is measured in nats-per-token, and a token is not a fixed-size unit of meaning. A tokenizer with a 50 K vocabulary assigns one token to most common English words; a 32 K tokenizer may need two tokens for the same words. The same text receives different nats-per-token scores under different tokenizers, making comparisons across models invalid.

**Problem 3: Loss does not directly tell you what the model knows.** A model that memorises a narrow corpus can achieve excellent loss while failing every factual question. Downstream capability benchmarks reveal the gap.

A complete evaluation therefore requires three complementary measurements:

| Signal | What it measures | Where in nanochat |
|---|---|---|
| BPB (val split) | Compression quality; generalisation | `nanochat/loss_eval.py` |
| CORE benchmark score | Language understanding / reasoning | `nanochat/core_eval.py` |
| Sampled text | Qualitative coherence | `nanochat/engine.py` |

All three are run by `scripts/base_eval.py`.

---

## 8.2 Bits-Per-Byte: the vocab-invariant loss metric

### From nats-per-token to bits-per-byte

At the end of a forward pass the model produces, for each position `t`, a probability distribution over the vocabulary. The cross-entropy loss at that position is:

```
loss_t = -log(p(correct_token_t))
```

When using natural logarithm the unit is **nats**. The total loss across a batch is the average over all token positions.

Converting nats to bits is a constant factor:

```
bits_t = loss_t / ln(2)  =  loss_t / 0.6931...
```

This gives **bits-per-token**. Still vocab-dependent.

To make the metric vocab-independent, instead of dividing by the number of tokens, divide by the **number of bytes** those tokens represent in UTF-8. Each token in the vocabulary has a fixed byte length; for example, the token `" the"` is 4 bytes, while ` ` is 1 byte.

The formula is:

```
BPB = (sum of loss_t in nats, for all valid tokens) / (ln(2) * sum of byte_len(token_t))
```

Or equivalently:

```
BPB = (sum of bits_t) / (total bytes in target text)
```

### Why this is vocab-invariant

Suppose a 50 K tokenizer encodes the word "unbelievable" as one token (`unbelievable`, 12 bytes), while a 32 K tokenizer splits it into three tokens (`un`, `believ`, `able`, also 12 bytes total). The sum of losses and the sum of bytes both change proportionally, so BPB stays the same (up to the quality of the model, not the tokenizer choice). This is the key insight.

### Reference values

On high-quality English web text:

- Random model (uniform over vocab): ~3.5+ BPB
- GPT-2 124M: approximately 1.00 BPB on its validation set
- Well-trained nanochat base model: similar range, depending on data and scale

Lower BPB means the model assigns higher probability to correct tokens — it is a better compressor of the data.

### The implementation: `nanochat/loss_eval.py`

```python
@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none') # (B, T)
        loss2d = loss2d.view(-1)
        y = y.view(-1)
        if (y.int() < 0).any():
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    # ...distributed all_reduce...
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
```

Walk through each decision:

**`loss_reduction='none'`** — instead of averaging losses inside the model call, the function gets a per-token loss tensor of shape `(B, T)`. This is necessary because each token will be weighted differently (by its byte length).

**`token_bytes`** — a 1-D tensor of shape `(vocab_size,)` computed once before evaluation. `token_bytes[token_id]` is the number of UTF-8 bytes that token represents, or **zero** for special tokens like `<|bos|>`. Indexing with `token_bytes[y]` broadcasts: for every target token in the flattened batch, look up its byte length.

**`num_bytes2d > 0` masking** — when a target token has zero byte length (a special token), its loss is excluded from the sum. This ensures BPB counts only real text tokens.

**The masked-target fast path** — targets with value `< 0` (the `ignore_index`, usually `-1`) mark padding or prompt positions that should not contribute to loss. The code handles them explicitly to avoid out-of-bounds indexing into `token_bytes`.

**Distributed accumulation** — when running with multiple GPUs via `torchrun`, each rank processes a subset of batches. `dist.all_reduce` sums `total_nats` and `total_bytes` across all ranks before dividing, giving the correct global BPB rather than an average of per-rank estimates.

**Final line**: `bpb = total_nats / (math.log(2) * total_bytes)`. This divides total nats by `ln(2) * total_bytes`, which is identical to converting nats to bits then dividing by bytes.

---

## 8.3 In-Context Learning and few-shot evaluation

### What in-context learning is

A pretrained language model has no explicit "memory" beyond its weights. Yet if you write several worked examples into the prompt before asking a question, the model often answers correctly — without any gradient update. This ability is called **in-context learning (ICL)**.

The mechanism is not fully understood, but the intuition is: during pretraining the model saw vast amounts of text structured as examples followed by answers (textbooks, Q&A forums, tutorials). It learns a general pattern — "when the preceding context shows several examples of type X, the next piece of text is usually of type X too."

### Few-shot vs. zero-shot

- **Zero-shot**: just the question, no examples. Tests whether the model has direct knowledge.
- **K-shot (few-shot)**: K worked examples prepended before the question. Helps the model understand the format and reduces random error.

For evaluation, a fixed K is chosen per task (e.g. 5-shot). The few-shot examples are sampled randomly (but with a fixed seed) from the same dataset, excluding the test item. This keeps evaluations deterministic and reproducible.

### Why ICL is a better capability metric than loss

BPB measures how well the model predicts held-out text from the same distribution as training. That is a compression quality signal. ICL-based benchmarks ask whether the model can **generalise** its knowledge to structured tasks it was not explicitly trained on in that format. A model can have excellent BPB on web text and still fail at commonsense reasoning questions. Conversely, a model trained on narrow corpora may show low BPB on that corpus while being surprisingly capable on tasks that happen to be well-represented.

Both measurements are needed.

---

## 8.4 The CORE benchmark

CORE (from the DCLM paper, arxiv:2406.11794) is a lightweight benchmark that aggregates four tasks into one comparable number. The tasks were chosen to span language understanding, world knowledge, commonsense reasoning, and text prediction.

### The four tasks

| Task | Format | What it tests |
|---|---|---|
| **SQuAD** | Reading comprehension | Can the model identify the answer span in a passage? |
| **LAMBADA** | Predict the final word | Does the model track narrative context over a long paragraph? |
| **HellaSwag** | Pick the correct sentence ending (4-choice) | Commonsense physical and social reasoning |
| **WinoGrande** | Fill-in-the-blank (2-choice) | Pronoun resolution requiring common sense |

### Scoring and the random baseline

Raw accuracy varies by task: WinoGrande (binary) has a random baseline of 50%, HellaSwag (4-choice) has a random baseline of 25%. A model that merely picks randomly would score differently on each task, making direct averaging misleading.

CORE centers each task score relative to its random baseline:

```
centered_score = (accuracy - random_baseline) / (1 - random_baseline)
```

The `random_baseline` values (as proportions, not percentages) are read from `eval_bundle/eval_meta_data.csv`. After centering, a random guesser scores 0.0 and a perfect model scores 1.0 on every task. The CORE metric is the mean of centered scores across all tasks:

```
CORE = mean(centered_score_i  for all tasks i)
```

From `scripts/base_eval.py`:

```python
centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
```

Note the `0.01 *` factor: the CSV stores random baselines as integers (e.g. `50` for 50%), so the code converts to a proportion by multiplying by 0.01 before centering.

### The GPT-2 baseline

GPT-2 124M achieves a CORE score of approximately **0.256525**. This is a useful reference point: a model that matches GPT-2 data efficiency has cleared a meaningful bar for a ~125 M parameter base model.

---

## 8.5 Multiple-choice evaluation: HellaSwag and WinoGrande

### The likelihood scoring method

Multiple-choice benchmarks (HellaSwag, WinoGrande, ARC) share a common evaluation strategy. Given a question and K answer candidates, the model does not generate text; instead it **scores each candidate** using the language model's own probabilities.

For each candidate `c_i`:

1. Construct the full string: `[few-shot examples] + question + delimiter + c_i`
2. Tokenize and run the model in forward (not generation) mode to get per-token losses
3. Identify the **continuation tokens** — those that belong to `c_i` and are not shared across all candidates
4. Compute the **mean loss** over those continuation tokens
5. Pick the candidate with the **lowest mean loss** (= highest log-probability)

This is principled: the model assigns probabilities to completions, and we pick the one it considers most likely.

### Why we need `find_common_length`

All K candidates share the same question prefix. When tokenized, the first N tokens of every candidate string are identical (up to where the answer choices diverge). Scoring the prefix tokens would be redundant and would favour longer shared prefixes.

`find_common_length` in `nanochat/core_eval.py` identifies the length of the shared prefix (or suffix, for schema tasks):

```python
def find_common_length(token_sequences, direction='left'):
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len
```

It scans token-by-token from `direction='left'` (for MC tasks where the prefix is shared) and returns the index of the first position where any sequence differs. The loss is summed only from `answer_start_idx` onward — only the tokens unique to each candidate.

### Prompt rendering: `render_prompts_mc`

```python
def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts
```

The function returns **one prompt per candidate**. Each prompt has the same few-shot examples and the same question, but ends with a different choice. For a 4-choice question this returns 4 strings, each tokenized and forwarded through the model separately.

The few-shot examples are rendered with the correct answer appended (so the model sees complete worked examples before the question under evaluation).

---

## 8.6 Language modeling evaluation: LAMBADA

### What LAMBADA tests

LAMBADA asks the model to predict the final word of a passage. The final word is chosen to be one that a human can predict if they read the whole passage (using narrative context), but cannot predict from just the last sentence. It tests long-range semantic coherence.

### The evaluation protocol

LAMBADA uses the `language_modeling` task type, which is different from multiple choice. There is only one correct answer (the true continuation), and the model must predict every token of that continuation correctly. Any wrong token counts as failure.

`render_prompts_lm` returns **two** prompts:

```python
def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    # ...template...
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with    = template.render(include_continuation=True,  **context)
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]
```

- `prompt_without`: the context up to (but not including) the continuation
- `prompt_with`: context + continuation

After tokenizing both, `batch_sequences_lm` verifies that the without-continuation tokens are a strict prefix of the with-continuation tokens, then sets `start_idx = len(tokens_without)` and `end_idx = len(tokens_with)`. Only the continuation tokens (`start_idx` to `end_idx`) are used for evaluation.

### Correctness check

```python
if task_type == 'language_modeling':
    si = start_idxs[0]
    ei = end_idxs[0]
    predicted_tokens = predictions[0, si-1:ei-1]
    actual_tokens = input_ids[0, si:ei]
    is_correct = torch.all(predicted_tokens == actual_tokens).item()
```

`predictions[i]` is the `argmax` of the model's output at position `i`, which predicts position `i+1`. So `predictions[0, si-1:ei-1]` are the argmax predictions for positions `si` through `ei-1`. If every predicted token matches the actual token at those positions, the example is correct.

This is strict: **all tokens** of the continuation must be predicted correctly. For LAMBADA, the continuation is typically one or two tokens (one word), so this is reasonable.

---

## 8.7 Schema tasks

Some tasks (WinoGrande in the CORE bundle) use a `schema` format rather than `multiple_choice`. In schema tasks, the **continuation is fixed** and the **context varies** across options. For example:

> Context A: "John could not fit in the closet because [he/it] was too large."
> Continuation: "he" or "it" — but here the continuation is the same pronoun resolved differently in different contexts.

The scoring is the mirror image of MC: `find_common_length` is called with `direction='right'` to find the **shared suffix**, and the loss is summed over those suffix tokens. The option whose context leads to the lowest loss on the fixed continuation wins.

---

## 8.8 Running `scripts/base_eval.py`

### Step 1: download the eval bundle

The CORE benchmark data (SQuAD, LAMBADA, HellaSwag, WinoGrande) is bundled as `eval_bundle.zip`. The script auto-downloads it on first run:

```python
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
```

On first run with `--eval core`, the script detects the missing bundle and downloads it automatically. You do not need to do this manually.

### Step 2: evaluate your nanochat model

✍️ Run all three evaluation modes on your trained checkpoint:

```bash
python -m scripts.base_eval --model-tag=YOUR_TAG --eval=core,bpb,sample
```

For a quick approximate evaluation (useful during active development), subsample each CORE task to 100 examples and use a smaller BPB token budget:

✍️
```bash
python -m scripts.base_eval \
    --model-tag=YOUR_TAG \
    --eval=core,bpb,sample \
    --max-per-task=100 \
    --split-tokens=524288
```

If you have multiple GPUs:

✍️
```bash
torchrun --nproc_per_node=4 -m scripts.base_eval \
    --model-tag=YOUR_TAG \
    --device-batch-size=16
```

### Step 3: evaluate the GPT-2 baseline

✍️ Install `transformers` if you have not:

```bash
pip install transformers
```

✍️ Run GPT-2 as a reference:

```bash
python -m scripts.base_eval --hf-path openai-community/gpt2
```

This loads GPT-2 124M through `ModelWrapper`, which adapts the HuggingFace interface to the nanochat evaluation harness:

```python
class ModelWrapper:
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )
        return loss
```

GPT-2's context window is 1024 tokens. The script sets `max_seq_len=1024`, and `evaluate_example` in `core_eval.py` respects this by truncating sequences longer than the model's context:

```python
if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
    max_tokens = model.max_seq_len
    # ... truncate each sequence to max_tokens, adjusting start/end indices
```

### Expected output format

```
================================================================================
CORE Evaluation
================================================================================
Evaluating: squad (0-shot, type: multiple_choice)... accuracy: 0.3100 | centered: 0.2933 | time: 42.18s
Evaluating: lambada (0-shot, type: language_modeling)... accuracy: 0.2450 | centered: 0.2450 | time: 18.33s
Evaluating: hellaswag (10-shot, type: multiple_choice)... accuracy: 0.2940 | centered: 0.0587 | time: 213.45s
Evaluating: winogrande (5-shot, type: schema)... accuracy: 0.5210 | centered: 0.0420 | time: 31.22s

Results written to: /path/to/base_eval/openai-community-gpt2.csv
CORE metric: 0.2565
```

The CSV file is written to `base_eval/<model_slug>.csv` in the base directory. It contains per-task raw accuracy, centered score, and the final CORE metric.

**What is happening** during evaluation:

For each task, `evaluate_core` loads the task's JSONL data file, shuffles it with a fixed seed (1337) for reproducibility, optionally subsamples to `max_per_task`, then calls `evaluate_task` which strides examples across DDP ranks. Each example is evaluated independently by `evaluate_example`. The few-shot sample for example `idx` uses seed `1234 + idx`, ensuring each example always gets the same few-shot context regardless of how many examples are evaluated in total.

---

## 8.9 MMLU: Massive Multitask Language Understanding

MMLU is a 57-subject academic benchmark with 4-choice multiple-choice questions. It is primarily used to evaluate **instruction-tuned** (chat) models in Chapter 9, but it is worth understanding its structure here because the evaluation mechanism is the same likelihood scoring used in CORE.

The 57 subjects span: `abstract_algebra`, `anatomy`, `high_school_chemistry`, `professional_law`, `virology`, and 52 more. Each subject is a separate subset in the HuggingFace dataset.

### Structure of `tasks/mmlu.py`

```python
class MMLU(Task):

    letters = ('A', 'B', 'C', 'D')
    groups = ('abstract_algebra', 'anatomy', ... )  # 57 subjects

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        choices  = row["choices"]    # list of 4 strings
        answer   = row["answer"]     # int 0-3
        subject  = row["subject"]
        user_message = render_mc(question, self.letters, choices)
        assistant_message = self.letters[answer]
        messages = [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        return {"messages": messages, "subject": subject, "letters": self.letters}
```

The key rendering function is `render_mc` from `tasks/common.py`:

```python
def render_mc(question, letters, choices):
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query
```

Two design decisions documented in the code are worth understanding:

1. The letter comes **after** the choice (`- The mitochondria=A`), not before. Small models have better binding this way — they are better at associating the answer letter with the choice text when the letter appears at the end.

2. There is **no whitespace before the letter** in the prompt (`=A`, not `= A`). The tokenizer assigns different token IDs to `" A"` (space-A) vs `"A"`. Since the assistant response is expected to be the bare letter `"A"`, the prompt must use the same tokenization. This is a subtle but important detail that can affect accuracy on small models.

For MMLU evaluation on a chat model, the evaluation function simply checks:

```python
def evaluate(self, conversation, assistant_response):
    assistant_message = conversation['messages'][-1]['content']  # e.g. "A"
    return assistant_response == assistant_message
```

The expected `assistant_response` is a single letter. The chat evaluation harness (Chapter 9) restricts the model's generation to one of the four letters, then checks correctness.

---

## 8.10 Validation loss vs. benchmark scores

### The expected relationship

In general, lower BPB correlates with better CORE scores. A model that assigns higher probability to correct tokens has, by definition, learned better representations of text — which translates to better performance on downstream tasks. This correlation is well-documented in scaling law research.

For nanochat models trained on the same data distribution, you should expect a roughly monotonic relationship: as BPB falls during training, CORE score rises.

### When they diverge

Several factors can break the correlation:

**Distribution shift.** If the val split differs in domain from the CORE tasks, the model can overfit to the val distribution while failing at CORE. Example: a model trained exclusively on code has low BPB on code validation but may fail LAMBADA (narrative text) and WinoGrande (commonsense).

**Tokenizer effects.** BPB is normalised by bytes, but CORE is not. If the tokenizer handles certain text types (numbers, rare words, non-English) inefficiently, BPB can look poor even when the model has good semantic understanding of those domains.

**Training signal mismatch.** A model trained with heavy instruction-following data may have slightly higher BPB on raw web text but score much better on structured tasks like WinoGrande, because it has learned to follow reasoning patterns.

### What this means in practice

- Use **BPB** as your primary fast feedback loop during training (it is computable on every val step, cheap, and responsive)
- Run **CORE** at key checkpoints (it takes minutes to hours, but reveals capabilities BPB cannot)
- Look at **samples** whenever something seems off: generated text shows you things that neither number captures

---

## 8.11 Sample-based evaluation

The `--eval sample` mode generates text from the model using a fixed set of prompts and unconditioned seeds. This is not a quantitative metric — it is a qualitative sanity check.

From `scripts/base_eval.py`:

```python
prompts = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]
```

For each prompt, the model generates 16 tokens at temperature 0 (greedy). The script also generates 8 unconditioned samples of 128 tokens at temperature 1.0 (from just `<|bos|>`).

✍️ Run sampling only:

```bash
python -m scripts.base_eval --model-tag=YOUR_TAG --eval=sample
```

**What to look for in conditioned samples:**

- Does "The capital of France is" continue with "Paris"?
- Does "If 5*x + 3 = 13, then x is" get the algebra right?
- Are completions grammatically coherent?

A model that gets none of these right probably has not been trained long enough or has a bug somewhere.

**What to look for in unconditioned samples:**

- Does the text resemble natural English sentences or random tokens?
- Are proper nouns capitalised? Are sentences grammatically complete?
- Do paragraphs have a coherent topic?

Early in training (high BPB), unconditioned samples look like character salad. By GPT-2-level quality, they should read like plausible (though sometimes incoherent) news or Wikipedia text.

---

## 8.12 Putting it all together

Here is a complete evaluation session for a trained nanochat checkpoint, annotated with what each number tells you.

✍️
```bash
# Full evaluation run: all three modes
python -m scripts.base_eval --model-tag=YOUR_TAG --eval=core,bpb,sample
```

Read the output in this order:

1. **Samples first.** If the model is outputting nonsense, stop and diagnose before running expensive benchmarks.
2. **BPB next.** Compare train BPB vs. val BPB. A large gap means overfitting. Compare val BPB to the previous checkpoint to confirm the model is still improving.
3. **CORE last.** Compare against the GPT-2 reference of 0.2565. Task-level breakdowns (written to the CSV) tell you which capabilities have been acquired and which are still weak.

A healthy training run at GPT-2-equivalent scale (~125 M parameters, ~10 B tokens) should produce:

| Metric | Expected range |
|---|---|
| Val BPB | ~1.0 |
| Train BPB | Slightly below val BPB |
| CORE score | ~0.25 (approaching GPT-2 baseline) |
| Conditioned samples | Mostly correct factual completions |

---

## Check your understanding

1. **Why does BPB require `token_bytes` rather than just counting tokens?** Consider a tokenizer that encodes the word "cat" as one token and another that encodes it as three tokens (`c`, `a`, `t`). Show with numbers that nats-per-token would differ between the two tokenizers but BPB would remain the same (assuming identical model quality).

2. **In `evaluate_example`, for a 4-choice HellaSwag item, the code runs four separate forward passes and picks the choice with the lowest mean loss over continuation tokens. Why mean loss rather than sum loss?** Hint: consider candidates of different lengths.

3. **LAMBADA evaluates a `language_modeling` task using `torch.all(predicted_tokens == actual_tokens)` — a strict all-or-nothing correctness criterion. For HellaSwag, correctness is determined by `pred_idx == item['gold']` — the chosen option. Why is LAMBADA strict-token accuracy while HellaSwag uses mean loss comparison?** What would go wrong if you used mean loss for LAMBADA?

---

## What's next

Chapter 9 covers supervised fine-tuning (SFT) and instruction following. You will adapt the pretrained base model to follow chat-style instructions, then evaluate the resulting chat model on MMLU and other tasks from the `tasks/` directory. The evaluation framework you built intuitions for in this chapter carries over directly — MMLU uses the same likelihood-scoring approach, and BPB on instruction-following data will replace the pretraining BPB you measured here.
