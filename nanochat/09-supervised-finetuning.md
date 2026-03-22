# Chapter 9: Supervised Fine-Tuning — Teaching the Model to Chat

## What you'll learn

- Why a pretrained base model cannot hold a conversation, and what SFT fixes
- How chat templates and special tokens structure a conversation into a token sequence
- How loss masking ensures the model trains only on assistant tokens — not user messages
- How multi-task learning with MMLU, GSM8K, SmolTalk, and spelling tasks is assembled as a single shuffled mixture
- How to read and run `scripts/chat_sft.py` and interpret the training output

## Prerequisites

- Chapter 3 (tokenization, special tokens, `render_conversation`)
- Chapter 6 (pretraining loop — gradient accumulation, optimizer, learning-rate schedule)
- Chapter 8 (evaluation fundamentals)
- A pretrained base model checkpoint on disk (produced by `scripts/base_train.py`)

---

## 9.1 From base model to chat model: the gap

After pretraining, the model has learned a remarkable amount from the internet. It knows that Paris is the capital of France, that Einstein developed relativity, and roughly how Python loops work. But ask it a question and something strange happens:

```
User: What is the capital of France?
Model: What is the capital of Germany? What is the capital of Italy? What is the…
```

The model did not answer. It continued the pattern of a trivia quiz because that is the only thing it knows how to do: predict the next token in a sequence. It has no idea it is supposed to be a helpful assistant. It has never seen a conversation.

The gap between a base model and a chat model is a gap in *format knowledge*. The base model knows facts; it does not know the role it is supposed to play. **Supervised Fine-Tuning (SFT)** closes that gap by showing the model thousands of (question, ideal answer) pairs, formatted as conversations, and training it to produce the answers.

After SFT, when the model sees:

```
<|user_start|>What is the capital of France?<|user_end|><|assistant_start|>
```

it has learned that the right next token begins an answer like "The capital of France is Paris." It has internalized the *assistant persona*.

Two important things are not happening here:

1. The model is not learning new facts in any deep sense. Its world knowledge comes from pretraining. SFT teaches format and behavior, not content.
2. The model is not learning to be clever or creative on its own. That comes later in reinforcement learning from human feedback (RLHF). SFT just teaches the supervised baseline.

---

## 9.2 Chat templates and special tokens

From Chapter 3 you know that nanochat defines eight special tokens in `nanochat/tokenizer.py`:

```python
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]
```

A conversation is a Python dictionary with a `"messages"` key:

```python
conversation = {
    "messages": [
        {"role": "user",      "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
}
```

When this is tokenized for training, it becomes the following flat token sequence (shown with the special tokens written out):

```
<|bos|>
<|user_start|> What is the capital of France? <|user_end|>
<|assistant_start|> The capital of France is Paris. <|assistant_end|>
```

The special tokens serve as structural anchors. Without them, the model cannot distinguish "user talking" from "assistant talking". With them, every assistant turn begins with the same token `<|assistant_start|>`, which the model quickly learns to treat as the trigger for generating an answer.

Why does this structure matter so much for small models? Because they have fewer parameters and less capacity for implicit pattern recognition. A large model might infer conversation structure from subtle statistical cues; a small model needs explicit markers it can reliably attend to.

For multi-turn conversations the pattern simply extends:

```
<|bos|>
<|user_start|> Hello, who are you? <|user_end|>
<|assistant_start|> I am nanochat, an AI assistant. <|assistant_end|>
<|user_start|> What can you do? <|user_end|>
<|assistant_start|> I can answer questions, do math, and write code. <|assistant_end|>
```

One practical complication: some datasets include an opening system message. nanochat handles this by merging the system message into the first user message with a double newline (`tokenizer.py`, lines 283–289). The conversation format requires user and assistant to strictly alternate, so the system message is folded in rather than given its own turn.

---

## 9.3 Loss masking: only train on assistant tokens

### During pretraining

The pretraining objective is simple: given tokens `t_1, t_2, ..., t_{n-1}`, predict token `t_n`. Every single position contributes to the loss equally. This is what teaches the model general language statistics.

### During SFT

During SFT the objective narrows: **only compute loss on the assistant tokens**. User tokens, special tokens, and tool outputs are masked out with the label value `-1`, which PyTorch's cross-entropy ignores.

Why? Consider the alternative. If we trained on every token, the model would be penalized for not predicting the user's exact words. But the user can say anything — there is no "correct" user message to predict. Worse, training on user tokens can cause the model to learn to parrot user phrasing rather than generate useful answers.

The technical mechanism is a parallel `mask` array produced alongside the token IDs. Wherever `mask = 1`, the loss is computed. Wherever `mask = 0`, the target is set to `-1` (the `ignore_index` for `nn.CrossEntropyLoss`).

### Walking through `render_conversation()`

The function lives in `nanochat/tokenizer.py` at line 266. Here is the full logic, annotated:

```python
def render_conversation(self, conversation, max_tokens=2048):
    ids, mask = [], []

    def add_tokens(token_ids, mask_val):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))
```

`add_tokens` is a helper that appends token IDs to `ids` and the same `mask_val` (0 or 1) to `mask` for each token added. The two lists are always the same length.

```python
    # fetch all the special tokens we need
    bos = self.get_bos_token_id()
    user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
    assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
    python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
    output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

    add_tokens(bos, 0)          # BOS: mask=0, not trained on
```

The conversation starts with `<|bos|>` at mask value 0. This token is structural — it signals the beginning of a sequence but is not something the model should be predicting.

```python
    for i, message in enumerate(messages):

        if message["role"] == "user":
            add_tokens(user_start, 0)   # structural: mask=0
            add_tokens(value_ids, 0)    # user text: mask=0
            add_tokens(user_end, 0)     # structural: mask=0

        elif message["role"] == "assistant":
            add_tokens(assistant_start, 0)  # structural: mask=0
            # ...
            add_tokens(value_ids, 1)        # assistant text: mask=1  <-- TRAINED ON
            # ...
            add_tokens(assistant_end, 1)    # end token: mask=1       <-- TRAINED ON
```

Notice the asymmetry: the `assistant_start` token itself gets `mask=0` (the model does not need to predict it, because at inference time we append it to the prompt manually to begin generation), but `assistant_end` gets `mask=1` (the model must learn to emit it to signal completion).

For tool-using assistant messages (a list of parts):

```python
            if part["type"] == "text":
                add_tokens(value_ids, 1)        # assistant prose: trained on

            elif part["type"] == "python":
                add_tokens(python_start, 1)     # model generates this
                add_tokens(value_ids, 1)        # model generates the expression
                add_tokens(python_end, 1)       # model generates this

            elif part["type"] == "python_output":
                add_tokens(output_start, 0)     # comes from Python: NOT trained on
                add_tokens(value_ids, 0)        # comes from Python: NOT trained on
                add_tokens(output_end, 0)       # comes from Python: NOT trained on
```

The tool outputs (`python_output`) are masked out entirely. At inference time the Python interpreter produces these tokens — the model cannot and should not predict them. Training on them would be wrong. But the expressions the model sends *to* the interpreter are trained on, because those are generated by the assistant.

### How the mask becomes `targets = -1`

Back in `scripts/chat_sft.py`, the data generator `sft_data_generator_bos_bestfit` converts this mask array into the targets tensor:

```python
mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
mask_targets = mask_tensor[:, 1:].to(device=device)
targets[mask_targets == 0] = -1
```

The `[:, 1:]` shift aligns the mask with the targets (targets are the inputs shifted one step to the right). Every position where `mask == 0` gets its target overwritten to `-1`. Cross-entropy with `ignore_index=-1` skips those positions entirely — they contribute zero gradient and zero loss.

### A visual summary

Here is a simplified tokenized conversation with the mask values shown:

```
Token:        <bos>  <usr_s>  What  is  Paris?  <usr_e>  <ast_s>  Paris.  <ast_e>
Mask:            0       0      0    0      0        0        0       1        1
Trains on?      no      no     no   no     no       no       no     yes      yes
```

The model is trained on exactly two things: the word "Paris." and the `<|assistant_end|>` token. Nothing else.

---

## 9.4 The Task infrastructure

All training data in nanochat flows through a common abstraction defined in `tasks/common.py`. Understanding it makes the data pipeline readable.

### The `Task` base class

```python
class Task:
    @property
    def eval_type(self):
        # 'categorical' or 'generative'
        raise NotImplementedError

    def get_example(self, index) -> dict:
        # Returns a conversation dict: {"messages": [...], ...}
        raise NotImplementedError

    def evaluate(self, conversation, completion) -> bool | int:
        # Returns whether the completion is correct
        raise NotImplementedError
```

Every dataset is a `Task`. Each task:
- Reports whether it is `categorical` (multiple choice, we can check the logit) or `generative` (open-ended, we must sample and check the text)
- Returns conversations in the standard format via `get_example(index)`
- Knows how to grade a completion via `evaluate(conversation, completion)`

Slicing is built in: `Task(start=100, stop=200)` gives you examples 100–199 without loading the rest.

### `TaskMixture`: shuffled union of datasets

```python
class TaskMixture(Task):
    def __init__(self, tasks, **kwargs):
        # Build list of all (task_idx, local_idx) pairs
        self.index_map = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))
        # Deterministically shuffle to mix tasks throughout training
        rng = random.Random(42)
        rng.shuffle(self.index_map)
```

`TaskMixture` pools every example from every task into a flat list and shuffles it once (with a fixed seed for reproducibility). When the data loader accesses index `i`, it looks up `self.index_map[i]` to find which task and which local index to pull from.

The practical effect: instead of training on all SmolTalk examples, then all MMLU examples, then all GSM8K examples (which would cause sequential forgetting), every mini-batch sees a random mix of all three. Multi-task training is more stable this way.

The comment in the source is deliberate: "if you wish to oversample any task, just pass it in multiple times in the list." This is how `--mmlu-epochs 3` works — the MMLU `Task` object is instantiated three times and all three copies appear in the mixture.

### `TaskSequence`: ordered curriculum

`TaskSequence` keeps tasks in order — all of task 1, then all of task 2, and so on. This is useful when a curriculum is needed (e.g., easy concepts before hard ones), but nanochat's SFT script uses `TaskMixture` everywhere.

---

## 9.5 MMLU: teaching the model general knowledge

**Massive Multitask Language Understanding** (MMLU) is a benchmark of 57 academic subjects spanning abstract algebra, anatomy, college chemistry, professional law, world religions, and more. Each question has exactly four answer choices labeled A, B, C, D.

### How a question becomes a conversation

In `tasks/mmlu.py`, `get_example()` calls `render_mc()` from `tasks/common.py`:

```python
def render_mc(question, letters, choices):
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query
```

Note the deliberate formatting: the answer letter appears *after* the choice text (`- choice text=A`), not before. The comment in the source explains the reasoning: small models exhibit better binding when the letter follows the text. When choice text comes first, the model has already processed the content before seeing which letter it maps to, making the association clearer.

There is a second subtle detail: there is no whitespace before the letter (`=A`, not `= A`). The nanochat tokenizer — like GPT-4's — produces different token IDs for `"A"` and `" A"`. Since the assistant response is just the bare letter `"A"` (no preceding space), it must match the exact token used in the prompt for the association to be clean.

A complete MMLU conversation looks like this:

```python
{
  "messages": [
    {
      "role": "user",
      "content": "Multiple Choice question: What is the powerhouse of the cell?\n"
                 "- Nucleus=A\n"
                 "- Mitochondria=B\n"
                 "- Ribosome=C\n"
                 "- Golgi apparatus=D\n\n"
                 "Respond only with the letter of the correct answer."
    },
    {
      "role": "assistant",
      "content": "B"
    }
  ],
  "subject": "college_biology",
  "letters": ("A", "B", "C", "D"),
}
```

The assistant response is a single letter. After `render_conversation()`, only that letter and `<|assistant_end|>` have `mask=1`. The model is being trained on an enormous breadth of world knowledge, one letter at a time.

### Why include MMLU in SFT?

A conversational dataset like SmolTalk improves response quality and style, but it is heavily English, heavily informal, and heavily biased toward common questions. MMLU's 57 subjects provide systematic coverage of academic knowledge. Including it (at 100 K examples per epoch) ensures the fine-tuning does not wash out factual capabilities learned during pretraining.

---

## 9.6 GSM8K: teaching math with tool use

**Grade School Math 8K** contains roughly 8 000 grade-school math word problems with step-by-step solutions. The tricky part: the solutions contain embedded arithmetic inside `<< >>` markers:

```
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10
```

The `<<expression=result>>` format is GSM8K's way of making the arithmetic explicit. nanochat repurposes this format as actual tool calls.

### How GSM8K becomes tool-using conversations

In `tasks/gsm8k.py`, `get_example()` parses the answer string with a regex:

```python
parts = re.split(r'(<<[^>]+>>)', answer)
for part in parts:
    if part.startswith('<<') and part.endswith('>>'):
        inner = part[2:-2]               # e.g. "12/60=0.2"
        expr, result = inner.rsplit('=', 1)
        assistant_message_parts.append({"type": "python", "text": expr})
        assistant_message_parts.append({"type": "python_output", "text": result})
    else:
        assistant_message_parts.append({"type": "text", "text": part})
```

The text between the `<<` and `=` becomes a `"python"` part (the expression sent to the interpreter). The text after the `=` and before `>>` becomes a `"python_output"` part (what the interpreter returns). Regular prose becomes `"text"` parts.

When this is passed to `render_conversation()`, the token sequence looks like:

```
<|assistant_start|>
  Weng earns 12/60 =          (mask=1, text)
  <|python_start|>             (mask=1)
  12/60                        (mask=1, expression)
  <|python_end|>               (mask=1)
  <|output_start|>             (mask=0)
  0.2                          (mask=0, output — not trained on)
  <|output_end|>               (mask=0)
  $0.2 per minute.             (mask=1, text)
  ...
<|assistant_end|>              (mask=1)
```

The model learns to generate the expression `12/60`. It does not learn to generate the output `0.2` — that is the calculator's job at inference time. The output tokens are masked entirely.

### Why is this powerful?

A model that uses a calculator is more reliable than one that tries to do arithmetic in its head. Language models are notoriously bad at multi-step arithmetic because each step has to be "stored" in the weights as a statistical pattern rather than computed exactly. Teaching the model to emit `<<12/60>>` and wait for the result offloads exact arithmetic to Python, where it is always correct.

GSM8K provides the training signal for this behavior. After SFT on GSM8K, when the model encounters an arithmetic step it has learned to emit `<|python_start|>` followed by the expression.

### Evaluation

The `evaluate()` method in `gsm8k.py` uses a regex to extract the number after `####`:

```python
GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
```

Both the reference answer (from the conversation's last text part) and the predicted answer (from the model's completion) are extracted and compared as strings. A prediction is correct if and only if the final numeric answer matches.

---

## 9.7 SmolTalk and conversational data

`tasks/smoltalk.py` wraps the `smol-smoltalk` dataset from HuggingFace (460 K training, 24 K test). This is a curated collection of multi-turn conversations covering:

- General question answering and factual lookup
- Writing assistance (summarization, paraphrasing, editing)
- Code explanation and debugging
- Step-by-step reasoning tasks
- Everyday instructions ("how do I do X")

Unlike MMLU or GSM8K, SmolTalk conversations have full natural-language assistant responses — sometimes several paragraphs, sometimes multiple turns. This teaches the model conversational fluency: how to be helpful, how to structure an explanation, when to use bullet points.

It also handles the system message case gracefully. When a SmolTalk conversation has an opening system message (setting a persona or context), `render_conversation()` folds it into the first user message. The model does not see a separate system-message turn; it sees the system content prepended to the user's text.

---

## 9.8 Spelling tasks: SimpleSpelling and SpellingBee

Two synthetic tasks in `tasks/spellingbee.py` address a known weakness of tokenization-based models: they struggle to reason about individual characters, because a token like `"strawberry"` is a single atomic unit with no visible internal structure.

**SimpleSpelling** (200 K examples) teaches the basic mapping from word to letters:
```
User:      Spell the word: apple
Assistant: apple:a,p,p,l,e
```

**SpellingBee** (80 K examples) extends this to letter-counting with tool use:
```
User:      How many 'r' are in strawberry?
Assistant: [manual walk-through counting each character]
           Let me double check using Python:
           <|python_start|>'strawberry'.count('r')<|python_end|>
           <|output_start|>3<|output_end|>
           Python gives us 3. My final answer is: #### 3
```

The SpellingBee task uses the same `#### answer` convention as GSM8K, so the evaluation and reward functions are identical. It also uses the same tool-call format, reinforcing the general pattern of "write expression, receive output, continue reasoning."

These tasks are particularly important for smaller models. A GPT-4-scale model can often spell words correctly without explicit training, but a small model at nanochat's scale genuinely benefits from this concentrated practice.

---

## 9.9 Multi-task learning in practice

Here is the complete training mixture assembled in `scripts/chat_sft.py`:

```python
train_tasks = [
    SmolTalk(split="train"),                         # 460 K conversations
    CustomJSON(filepath=identity_conversations_filepath),  # 1 K identity conversations
    CustomJSON(filepath=identity_conversations_filepath),  # 1 K (2nd epoch)
    *[MMLU(subset="auxiliary_train", split="train") for _ in range(args.mmlu_epochs)],   # 100 K × 3 = 300 K
    *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],            # 8 K × 4 = 32 K
    SimpleSpelling(size=200000, split="train"),       # 200 K
    SpellingBee(size=80000, split="train"),           # 80 K
]
train_dataset = TaskMixture(train_tasks)
```

With default flags (`--mmlu-epochs 3`, `--gsm8k-epochs 4`), this yields roughly 1.07 M training examples. `TaskMixture` shuffles all of them together into a single sequence.

### Why use different epoch counts per task?

The datasets are very different sizes and the tasks have different difficulties:

- SmolTalk is large (460 K) and general — one epoch is fine
- MMLU is medium (100 K) but broad — a few epochs give better coverage across all 57 subjects
- GSM8K is small (8 K) and difficult — more epochs let the model practice the tool-call format repeatedly without needing a larger dataset
- SimpleSpelling is synthetic and cheap to generate — 200 K provides intensive repetition of character-level reasoning

The multiplication trick (`*[GSM8K(...) for _ in range(4)]`) passes four independent instantiations of the GSM8K task to `TaskMixture`. Each instantiation is shuffled internally, and all four appear in the global shuffle, effectively giving GSM8K four times its natural representation.

### The role of shuffling in preventing forgetting

Without shuffling, a training run would look like: 460 K SmolTalk conversations, then 300 K MMLU, then 32 K GSM8K. The model would overfit to each task sequentially, forgetting earlier ones. With `TaskMixture`'s shuffle, every step in the training loop sees a random draw from the entire combined pool. The gradient updates for different task types interleave, and all tasks improve together.

---

## 9.10 SFT training loop differences from pretraining

The training loop in `chat_sft.py` is structurally identical to the pretraining loop: gradient accumulation, Muon optimizer, learning-rate schedule with warmup and warmdown, periodic evaluation. The key differences are:

### Smaller effective learning rate

```python
parser.add_argument("--init-lr-frac", type=float, default=0.8, ...)
```

The SFT run inherits the learning rates from the pretrained checkpoint, then starts at 80% of those values (`--init-lr-frac 0.8`). Fine-tuning requires smaller gradient steps than pretraining. The model's weights encode valuable knowledge; large updates can overwrite it.

### Fewer total steps

SFT stops after one pass through the training mixture (unless `--num-iterations` is set). With ~1 M examples packed into sequences of 2 048 tokens at a batch size of 524 288 tokens, the entire SFT run is a few thousand gradient steps — an order of magnitude less than pretraining.

### Optimizer warm-start

```python
if args.load_optimizer:
    optimizer_data = load_optimizer_state("base", ...)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    optimizer.load_state_dict(optimizer_data)
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr
```

The momentum buffers from pretraining are loaded (they encode the recent gradient direction and magnitude). But because pretraining ends with a warmdown that drives learning rates near zero, the LR values in the saved state are near-zero. The code saves the fresh SFT LRs first, loads the optimizer state, then restores those fresh LRs. This gives you warm momentum without dead learning rates.

### Loss masking changes the effective signal

During pretraining, every token in a sequence contributes to the loss. During SFT, only assistant tokens do. For a typical conversation where the user turn is roughly as long as the assistant turn, this halves the number of positions that generate gradient. In terms of effective batch size (tokens that actually update the model), SFT batches are smaller than they look. The model converges on fewer gradient steps partly because the training signal is more focused.

### Best-fit packing

The data loader (`sft_data_generator_bos_bestfit`) uses a best-fit bin-packing algorithm: for each row in the batch, it picks the largest conversation from the buffer that fits in the remaining space, then keeps packing until no conversation fits, at which point it pads the rest of the row with `<|bos|>` tokens at `mask=0`. This minimizes wasted capacity without truncating any conversation mid-thought.

---

## 9.11 Hands-on: run SFT on CPU with a tiny model

Before running, make sure you have a base model checkpoint. The SFT script loads it with `load_model("base", ...)`.

### ✍️ Step 1: Inspect the training mixture size

```python
from tasks.common import TaskMixture
from tasks.smoltalk import SmolTalk
from tasks.mmlu import MMLU
from tasks.gsm8k import GSM8K

mmlu_epochs = 3
gsm8k_epochs = 4

train_tasks = [
    SmolTalk(split="train"),
    *[MMLU(subset="auxiliary_train", split="train") for _ in range(mmlu_epochs)],
    *[GSM8K(subset="main", split="train") for _ in range(gsm8k_epochs)],
]
train_dataset = TaskMixture(train_tasks)
print(f"Total training examples: {len(train_dataset):,}")

# Inspect a few examples
for i in [0, 1, 2]:
    ex = train_dataset[i]
    role = ex["messages"][0]["role"]
    content_preview = str(ex["messages"][0]["content"])[:80]
    print(f"  [{i}] first message ({role}): {content_preview!r}")
```

**What you'll see:** a count around 1 M and a random mix of conversation types, because `TaskMixture` shuffles everything together.

### ✍️ Step 2: Inspect loss masking on a real example

```python
from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()

# A GSM8K-style example with tool calls
conversation = {
    "messages": [
        {"role": "user", "content": "If apples cost $3 each, how much do 7 apples cost?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "7 apples cost 7 x 3 = $"},
            {"type": "python", "text": "7*3"},
            {"type": "python_output", "text": "21"},
            {"type": "text", "text": "21.\n\n#### 21"},
        ]}
    ]
}

ids, mask = tokenizer.render_conversation(conversation)
print(f"Total tokens: {len(ids)}")
print(f"Trained-on tokens: {sum(mask)} ({100*sum(mask)/len(mask):.1f}%)")
print()
print(tokenizer.visualize_tokenization(ids, mask))
```

**What you'll see:** the assistant's prose and expression tokens highlighted in green (trained on), the user text, tool output, and structural tokens in red (masked out).

### ✍️ Step 3: Run SFT for 50 steps on CPU

This runs a short SFT trial. It does not train the model to convergence — it just validates the pipeline end-to-end:

```bash
python -m scripts.chat_sft \
    --device-type cpu \
    --num-iterations 50 \
    --device-batch-size 2 \
    --total-batch-size 4096 \
    --eval-every 25 \
    --chatcore-every -1 \
    --run dummy
```

**Flag breakdown:**

| Flag | Value | Purpose |
|---|---|---|
| `--device-type cpu` | `cpu` | Run on CPU (slower but no GPU required) |
| `--num-iterations 50` | `50` | Stop after 50 gradient steps instead of a full epoch |
| `--device-batch-size 2` | `2` | 2 sequences per micro-batch |
| `--total-batch-size 4096` | `4096` | Total tokens per gradient step |
| `--eval-every 25` | `25` | Run validation every 25 steps |
| `--chatcore-every -1` | `-1` | Disable ChatCORE evaluation (it requires sampling) |
| `--run dummy` | `dummy` | Disable wandb logging |

**Expected output:**

```
Inherited max_seq_len=2048 from pretrained checkpoint
Inherited device_batch_size=32 from pretrained checkpoint
...
Total training examples: 1,073,000
step 00001 (0.00%) | loss: 2.834521 | lrm: 0.80 | dt: 1823.14ms | ...
step 00002 (0.00%) | loss: 2.712309 | lrm: 0.80 | ...
...
Step 00025 | Validation bpb: 1.4321
...
step 00050 (0.00%) | loss: 1.983441 | ...
```

The loss should drop noticeably in the first 10–20 steps, faster than pretraining, because the task is more constrained (only assistant tokens, highly structured format).

> **What's happening:** The model starts from pretrained weights and immediately receives gradient signal from assistant tokens only. The structured format (user/assistant turns with special tokens) is quickly reinforced because these patterns are consistent across all tasks in the mixture.

### ✍️ Step 4: Spot-check the checkpoint

After training completes, a checkpoint is saved under `chatsft_checkpoints/`. Load and probe it:

```python
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

model, tokenizer, meta = load_model("sft", "cpu", phase="eval")
engine = Engine(model, tokenizer)

conversation = {
    "messages": [
        {"role": "user",      "content": "What is 6 times 7?"},
        {"role": "assistant", "content": ""},   # placeholder
    ]
}
prompt_ids = tokenizer.render_for_completion(conversation)
results, _ = engine.generate_batch(prompt_ids, num_samples=1, max_tokens=64, temperature=0.0, top_k=1)
prefix_len = len(prompt_ids)
print(tokenizer.decode(results[0][prefix_len:]))
```

Even after 50 steps on CPU, you should see the model attempting a structured answer rather than random text continuation.

---

## 9.12 ChatCORE evaluation

Once SFT is complete (a full run, not the 50-step trial), the model is evaluated with the **ChatCORE** metric, defined in `scripts/chat_eval.py`.

### The six tasks

| Task | Type | Baseline accuracy | Notes |
|---|---|---|---|
| ARC-Easy | categorical | 25% | Multiple choice science, elementary level |
| ARC-Challenge | categorical | 25% | Multiple choice science, harder |
| MMLU | categorical | 25% | 57-subject academic multiple choice |
| GSM8K | generative | 0% | Grade school math, open-ended |
| HumanEval | generative | 0% | Python programming problems |
| SpellingBee | generative | 0% | Letter counting with tool use |

### Categorical evaluation: logit-based

For categorical tasks, `run_categorical_eval()` in `chat_eval.py` never samples from the model. Instead, it:

1. Tokenizes the conversation up through `<|assistant_start|>` (using `render_for_completion()`)
2. Runs a forward pass and reads the logits at the last position
3. Narrows those logits down to only the token IDs corresponding to the valid answer letters (A, B, C, D)
4. Takes the argmax — whichever letter has the highest logit is the predicted answer

This is more reliable than sampling and dramatically faster because you can batch many problems together. The narrowing to valid letters also removes the possibility of the model outputting something invalid.

### Generative evaluation: sampling-based

For generative tasks, `run_generative_eval()` samples one or more completions from the model and calls `task.evaluate(conversation, completion)` on each. A problem is counted as "passed" if any sample passes.

The use of `render_for_completion()` matters here. That function:
1. Removes the last message (the ground-truth assistant response)
2. Appends `<|assistant_start|>` to the end of the token sequence

This primes the model to generate an assistant response without seeing the answer.

### The ChatCORE formula

Raw accuracy on MMLU is not very meaningful in isolation — even a random guesser gets 25%. ChatCORE centers each task's accuracy at its random baseline and averages:

```
centered_accuracy(task) = (accuracy - baseline) / (1 - baseline)

ChatCORE = mean(centered_accuracy(task) for all tasks)
```

A model that performs at random on every task scores 0.0. A perfect model scores 1.0. This makes the metric comparable across tasks with very different baselines.

During training in `chat_sft.py`, ChatCORE is computed every `--chatcore-every` steps:

```python
all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
categorical_tasks = {'ARC-Easy', 'ARC-Challenge', 'MMLU'}
chatcore = centered_mean(all_tasks)
chatcore_cat = centered_mean(categorical_tasks)
```

A healthy SFT run should see ChatCORE rise steeply in the first quarter of training as the model learns conversational format, then level off as the weights saturate.

---

## 9.13 Catastrophic forgetting

SFT presents a genuine danger. The model's pretrained knowledge lives in its weights as a dense, overlapping set of patterns. A new gradient signal — even a carefully designed one — can shift those weights and partially overwrite earlier knowledge. This is **catastrophic forgetting**.

### What it looks like in practice

Imagine you fine-tune on only SmolTalk. The model gets excellent at casual conversation. But SmolTalk has almost no math, so the math circuits in the model weights drift. A few hundred steps later, the model that could answer arithmetic questions during pretraining is now unreliable.

### nanochat's mitigations

**1. Data mixture diversity.** Including MMLU (world knowledge), GSM8K (math), and SmolTalk (conversation) forces the model to maintain performance across all three simultaneously. Any update that improves conversation but hurts math will be corrected by the math gradient in the next step.

**2. Reduced learning rate.** Starting at 80% of the pretrained LR (`--init-lr-frac 0.8`) and warming down to 0% over the last 50% of steps (`--warmdown-ratio 0.5`) means the total weight change from SFT is small relative to the total weight magnitude from pretraining.

**3. Warm optimizer state.** Loading the pretrained optimizer momentum means the gradient steps start "knowing" which directions caused improvement during pretraining. Fresh optimizer state would take many steps to re-learn the gradient landscape.

**4. One epoch.** Running one pass through the SFT mixture (rather than many epochs) limits total exposure. The model sees each example once, reducing the risk of overfitting to SFT patterns.

**5. Identity conversations.** The `identity_conversations.jsonl` file (loaded twice for two epochs) contains examples that teach the model its name, capabilities, and provider. These are repeated precisely because they are specific and easy to forget.

### The fundamental tension

SFT must be strong enough that the model actually learns to follow the chat format reliably, but gentle enough that it does not wipe out the pretraining. Getting this balance right is one of the central empirical challenges of building chat models. The hyperparameters `--mmlu-epochs`, `--gsm8k-epochs`, `--init-lr-frac`, and `--warmdown-ratio` all affect this trade-off and should be tuned for your specific model and dataset.

---

## Check your understanding

**1.** A colleague suggests that during SFT you should also compute loss on user tokens, arguing that "the model should understand what users say, and computing loss on those tokens helps". Is this reasoning correct? What would actually happen if you set all mask values to 1 across both user and assistant tokens?

**2.** In the best-fit packing data loader, padding tokens use `<|bos|>` and their targets are set to `-1`. Why use `<|bos|>` specifically for padding rather than a dedicated padding token? Would it matter if you used a random token instead?

**3.** You run SFT and notice that MMLU accuracy drops compared to the pretrained base model. Name two possible causes and one specific change to `chat_sft.py` arguments that would address each cause.

---

## What's next

Chapter 10 covers **Reinforcement Learning from Human Feedback (RLHF)** and **Group Relative Policy Optimization (GRPO)**. SFT teaches the model *what* an answer looks like. RL teaches the model *which* answers are actually better — it optimizes directly for quality signals like correctness and usefulness rather than token-by-token imitation of reference answers. You will see how the `render_for_completion()` function (introduced in this chapter) becomes the starting point for the RL generation loop.
