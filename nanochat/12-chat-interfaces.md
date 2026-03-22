# Chapter 12: Chat Interfaces — CLI and Web Server

## What you'll learn

- How nanochat manages multi-turn conversation state as a flat list of tokens
- How streaming generation works end-to-end: from the Engine loop to characters appearing in the terminal
- How FastAPI, async/await, and Server-Sent Events combine to push tokens to a browser in real time
- How a worker pool distributes concurrent requests across multiple GPUs without any request ever blocking another
- What abuse prevention looks like in practice and why every public-facing model server needs it

## Prerequisites

- Chapters 1–10 completed (you have a trained SFT or RL checkpoint)
- Chapter 11 completed (you understand the Engine, KV cache, and sampling)
- `uv sync` done; the nanochat repo is working
- For the web server: `fastapi` and `uvicorn` are already included in `pyproject.toml`

---

## 12.1 Two ways to talk to your model

After completing training you have a checkpoint file. The Engine from Chapter 11 can generate text from it. All that remains is a user interface — something that accepts a message, passes it through the conversation format, runs the Engine, and shows you the result.

nanochat ships two such interfaces:

| Interface | File | Use case |
|---|---|---|
| CLI | `scripts/chat_cli.py` | Development, debugging, scripting, no dependencies beyond PyTorch |
| Web server | `scripts/chat_web.py` | Sharing with others, concurrent users, browser UI, production serving |

Both interfaces use the same underlying `Engine` class and the same special-token conversation format. The CLI wraps the Engine in a terminal loop. The web server wraps it in an HTTP API with a browser UI, a worker pool, and streaming over Server-Sent Events.

Start with the CLI. Once you understand how tokens flow through a single conversation, the web server will feel like a natural extension.

---

## 12.2 The conversation format — a quick recap

Chapter 11 introduced the Engine. Before looking at either interface, it is worth being precise about the token format both interfaces produce.

A multi-turn conversation is serialized as a single flat list of integers:

```
[BOS] <|user_start|> ...user tokens... <|user_end|>
      <|assistant_start|> ...assistant tokens... <|assistant_end|>
      <|user_start|> ...user tokens... <|user_end|>
      <|assistant_start|> ...assistant tokens... <|assistant_end|>
      ...
      <|assistant_start|>          ← model generates from here
```

The model is given everything up to and including the final `<|assistant_start|>` token. It then generates tokens one at a time until it emits `<|assistant_end|>` or hits the token budget. Both interfaces build exactly this list, then feed it to `engine.generate()`.

---

## 12.3 CLI chat: `scripts/chat_cli.py`

### Overview

The CLI script is a single Python file — a synchronous `while True` loop with no web framework. It loads the model once at startup, maintains conversation state in memory, and prints tokens as they arrive from the Engine.

Run it:

✍️
```bash
python -m scripts.chat_cli --model-tag=my-sft-run
```

You should see:

```
NanoChat Interactive Mode
--------------------------------------------------
Type 'quit' or 'exit' to end the conversation
Type 'clear' to start a new conversation
--------------------------------------------------

User:
```

### Startup and argument parsing

```python
# scripts/chat_cli.py

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], ...)
args = parser.parse_args()
```

`--source` selects which checkpoint directory to look in (`sft` or `rl`). `--model-tag` is the run name you gave to your training script. `--step` lets you load a specific checkpoint step; omitting it loads the latest. `--prompt` puts the CLI into single-shot mode: send one message, get one response, exit — useful for scripting.

### Loading the model

```python
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
```

`autodetect_device_type()` returns `"cuda"` if CUDA is available, `"mps"` on Apple Silicon, and `"cpu"` otherwise. `compute_init` sets up the DDP context (here it is a no-op for single-GPU inference). `load_model` finds the checkpoint file, restores the model weights, and returns the tokenizer and metadata.

### Encoding special tokens

```python
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")
```

Special tokens were added to the vocabulary during SFT training (Chapter 7). `encode_special` returns the single integer ID for a special token. These integers are stored as local variables and reused every turn — no string lookups happen in the inner loop.

### Conversation state

```python
engine = Engine(model, tokenizer)
conversation_tokens = [bos]
```

The entire conversation history is a single Python list of integers — `conversation_tokens`. It starts with just `[BOS]` and grows with every turn. This is the key design choice: there is no separate "history" object, no list of message strings to re-encode. The token representation is the history.

### The main loop

```python
while True:
    try:
        user_input = input("\nUser: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        print("Conversation cleared.")
        continue

    if not user_input:
        continue
```

The loop reads one line at a time. Two special commands are handled before any token encoding: `clear` resets `conversation_tokens` back to `[bos]` — discarding the entire history so the model starts fresh — and `quit`/`exit` breaks the loop. Catching `EOFError` handles the case where the process receives EOF (for example, when piping input from a file).

### Encoding the user turn

```python
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)

    conversation_tokens.append(assistant_start)
```

Four operations: append `<|user_start|>`, extend with the BPE encoding of the user's text, append `<|user_end|>`, then append `<|assistant_start|>`. After these four lines the list ends with `<|assistant_start|>` — the cue for the model to generate.

### Streaming generation

```python
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": 256,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    response_tokens = []
    print("\nAssistant: ", end="", flush=True)
    for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
        token = token_column[0]  # pop the batch dimension (num_samples=1)
        response_tokens.append(token)
        token_text = tokenizer.decode([token])
        print(token_text, end="", flush=True)
    print()
```

`engine.generate()` is a Python generator (covered in Chapter 11). Each iteration yields one token across all samples — here `num_samples=1`, so `token_column` is a one-element tensor. Calling `tokenizer.decode([token])` turns that single integer back into a string fragment, which is printed immediately with `flush=True`. The `flush` is essential: without it, Python's stdout buffering holds the characters and the user sees nothing until an entire line is complete.

**What you're seeing in the terminal:** Each character (or small string) appears as soon as the GPU produces the corresponding token. You are watching the autoregressive process in real time — the model generates one token, it gets decoded and printed, then the model generates the next.

### Finishing the turn

```python
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)
```

After the generator exhausts its token budget, the response may not have ended with `<|assistant_end|>` (it hit `max_tokens` instead of a natural stopping point). The script checks and appends the end token if missing. This matters for the next turn: when the model sees the previous assistant response, it needs a clean `<|assistant_end|>` boundary to understand that the assistant's turn is over.

The full `response_tokens` list is then appended to `conversation_tokens`. The history grows. On the next user turn, the model will see all previous exchanges.

### Prompt mode (`--prompt`)

```python
    if args.prompt:
        user_input = args.prompt
    # ... (runs through the same generate loop)
    if args.prompt:
        break
```

When `--prompt` is set, the script skips `input()`, uses the provided string directly, generates one response, and exits. This lets you use the CLI non-interactively:

✍️
```bash
python -m scripts.chat_cli --model-tag=my-sft-run --prompt="What is 12 times 7?"
```

### Sampling parameters

`--temperature` (default 0.6) controls how peaked the probability distribution is before sampling. Lower values make the model more deterministic and focused; higher values introduce more randomness. `--top-k` (default 50) restricts sampling to the 50 highest-probability tokens at each step, preventing the model from accidentally sampling low-probability garbage.

Experiment with these values during development:

| Use case | temperature | top-k |
|---|---|---|
| Precise factual answers | 0.1–0.4 | 20–50 |
| General conversation | 0.6–0.8 | 50 |
| Creative writing | 0.9–1.2 | 100–200 |

---

> **What's happening: why `flush=True` matters**
>
> Python's standard output is line-buffered when connected to a terminal and fully buffered when redirected to a pipe or file. `print(token_text, end="", flush=True)` forces a flush after every token regardless of buffering mode. Without it, you would see nothing until the response is fully generated — which defeats the purpose of streaming.

---

## 12.4 FastAPI basics

The web server uses FastAPI, an async Python web framework. If you have used Flask or Django before, the mental model is similar: you register functions as request handlers for specific URL routes. FastAPI's key additions are:

1. **Pydantic request/response models** — request bodies are automatically parsed and validated against Python dataclasses.
2. **async/await** — handlers are `async def` functions. While a handler awaits I/O (e.g., waiting for a GPU computation to finish, or waiting for a worker), the event loop can handle other requests. This is what allows a single Python process to serve many concurrent users.
3. **Uvicorn** — the ASGI server that drives the event loop and handles the raw TCP connections.

### Defining an app and routes

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    ...

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    ...
```

`@app.get("/")` registers `root` as the handler for `GET /`. `@app.post("/chat/completions")` registers `chat_completions` for `POST /chat/completions`. The `async def` keyword means these functions are coroutines — they can `await` operations without blocking the process.

### Pydantic models for request validation

```python
# scripts/chat_web.py

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None
```

When a `POST /chat/completions` request arrives with a JSON body, FastAPI automatically deserializes the JSON into a `ChatRequest` object and calls `validate_chat_request` before your handler runs. If the JSON is malformed or missing required fields, FastAPI returns a 422 error automatically — you never write JSON parsing code.

### The lifespan handler

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on all GPUs on startup."""
    print("Loading nanochat models across GPUs...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    print(f"Server ready at http://localhost:{args.port}")
    yield

app = FastAPI(lifespan=lifespan)
```

The `lifespan` context manager runs setup code before the server starts accepting connections (everything before `yield`) and teardown code when the server stops (everything after `yield`, if any). This is where the worker pool is initialized — models are loaded onto GPUs once at startup, not on every request.

---

## 12.5 Server-Sent Events for streaming

HTTP is normally request-response: the client sends a request, the server sends a complete response, the connection closes. Streaming breaks that model: the server sends a response body incrementally, in chunks, over a connection that stays open until all chunks have been sent.

**Server-Sent Events (SSE)** is an HTTP streaming standard built on this idea. The server sets the `Content-Type` to `text/event-stream` and sends text lines with a specific format:

```
data: {"token": "Hello"}\n\n
data: {"token": " world"}\n\n
data: {"done": true}\n\n
```

Each message is one or more `field: value` lines followed by a blank line. Browsers have a built-in `EventSource` API for consuming SSE streams. In nanochat's web UI, the JavaScript instead uses the lower-level `fetch` + `ReadableStream` API for more control — but the format on the wire is the same.

### The `generate_stream` coroutine

```python
# scripts/chat_web.py

async def generate_stream(
    worker: Worker,
    tokens,
    temperature=None,
    max_new_tokens=None,
    top_k=None
) -> AsyncGenerator[str, None]:
    """Generate assistant response with streaming."""
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    accumulated_tokens = []
    last_clean_text = ""

    for token_column, token_masks in worker.engine.generate(
        tokens,
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=random.randint(0, 2**31 - 1)
    ):
        token = token_column[0]

        if token == assistant_end or token == bos:
            break

        accumulated_tokens.append(token)
        current_text = worker.tokenizer.decode(accumulated_tokens)
        if not current_text.endswith('â'):
            new_text = current_text[len(last_clean_text):]
            if new_text:
                yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"
```

Several things are worth examining carefully:

**`async def` with `yield`** makes this an async generator. Callers use `async for chunk in generate_stream(...)` to consume it. Each `yield` sends one SSE message.

**`engine.generate` is a synchronous generator** that runs GPU computation. In an async context, running synchronous blocking code without care would freeze the event loop. The nanochat Engine is designed to release the GIL during the PyTorch operations, so the event loop is not blocked in practice for CPU/async work, though the GPU kernel itself occupies the GPU. For true async decoupling on heavily loaded servers, you would wrap the engine call in `asyncio.to_thread`.

**UTF-8 multi-byte handling.** The tokenizer operates on bytes, and some tokens (emoji, accented characters) span multiple bytes. A single token ID may decode to an incomplete UTF-8 sequence that Python represents as `'â'` (the Unicode replacement character). The `accumulated_tokens` list grows with every token, and `tokenizer.decode(accumulated_tokens)` is called on the full list each time. The string is only yielded when it does not end with a replacement character — ensuring the browser always receives valid Unicode. `last_clean_text` tracks how much has already been sent so only the new suffix is emitted.

**The final `done` message** signals to the frontend that generation is complete.

### Returning a `StreamingResponse`

```python
return StreamingResponse(
    stream_and_release(),
    media_type="text/event-stream"
)
```

`StreamingResponse` takes any async generator and sends each yielded string as a chunk of the HTTP response body. Setting `media_type="text/event-stream"` tells the browser this is an SSE stream.

---

## 12.6 The OpenAI-compatible API

The `POST /chat/completions` endpoint deliberately mirrors the OpenAI Chat Completions API format. This is intentional: any code written against the OpenAI API can be pointed at nanochat instead by changing the base URL and providing a dummy API key.

### Request format

```json
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And its population?"}
  ],
  "temperature": 0.7,
  "top_k": 50,
  "max_tokens": 256
}
```

All fields except `messages` are optional. If omitted, server defaults (set at launch via `--temperature`, `--top-k`, `--max-tokens`) are used.

### Building the conversation tokens

```python
# scripts/chat_web.py — inside chat_completions()

conversation_tokens = [bos]
for message in request.messages:
    if message.role == "user":
        conversation_tokens.append(user_start)
        conversation_tokens.extend(worker.tokenizer.encode(message.content))
        conversation_tokens.append(user_end)
    elif message.role == "assistant":
        conversation_tokens.append(assistant_start)
        conversation_tokens.extend(worker.tokenizer.encode(message.content))
        conversation_tokens.append(assistant_end)

conversation_tokens.append(assistant_start)
```

This is identical in structure to what the CLI does each turn — except the CLI accumulates tokens across turns in memory, while the web server receives the full history as JSON on every request and rebuilds the token list from scratch. This stateless design is a fundamental property of HTTP: the server holds no per-session state between requests. The client (the browser) owns the conversation history.

### Response format

Each streamed chunk is:

```
data: {"token": " Paris", "gpu": 0}\n\n
```

When generation is complete:

```
data: {"done": true}\n\n
```

The `gpu` field in each chunk tells the client which GPU processed the request. This is informational — useful for debugging load distribution across multiple GPUs.

---

## 12.7 The worker pool

### The problem: one GPU, many users

An NVIDIA GPU can only run one forward pass at a time. (Batch inference can process multiple sequences in parallel, but nanochat's current Engine generates one request per forward pass.) If two users send messages simultaneously, one of them must wait.

The worker pool solves this by:
1. Maintaining a queue of available workers.
2. Each request `acquire_worker()` — removing a worker from the queue, blocking if all workers are busy.
3. The request runs generation using the acquired worker.
4. When done, the worker is returned to the queue via `release_worker()`.

Concurrent requests that arrive when all workers are busy are queued by `asyncio.Queue` and served in order as workers become free.

### The `Worker` dataclass

```python
# scripts/chat_web.py

@dataclass
class Worker:
    """A worker with a model loaded on a specific GPU."""
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object
```

A Worker bundles together everything needed to process one request: a specific GPU device, the Engine (model + KV cache) loaded on that device, and the tokenizer. Each Worker is a fully independent model replica.

### The `WorkerPool` class

```python
class WorkerPool:
    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source, model_tag=None, step=None):
        for gpu_id in range(self.num_gpus):
            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
            else:
                device = torch.device(device_type)
            model, tokenizer, _ = load_model(source, device, phase="eval", ...)
            engine = Engine(model, tokenizer)
            worker = Worker(gpu_id=gpu_id, device=device, engine=engine, tokenizer=tokenizer)
            self.workers.append(worker)
            await self.available_workers.put(worker)

    async def acquire_worker(self) -> Worker:
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        await self.available_workers.put(worker)
```

`asyncio.Queue` is the coordination primitive. `queue.get()` suspends the coroutine if the queue is empty (all workers busy) and resumes it when a worker is returned. `queue.put()` adds a worker back. This gives correct FIFO queuing with zero polling and zero busy-waiting — the event loop does all the work.

### Worker acquisition and release

```python
# scripts/chat_web.py — inside chat_completions()

worker_pool = app.state.worker_pool
worker = await worker_pool.acquire_worker()

try:
    # ... build tokens, return StreamingResponse ...
    async def stream_and_release():
        try:
            async for chunk in generate_stream(worker, ...):
                yield chunk
        finally:
            await worker_pool.release_worker(worker)

    return StreamingResponse(stream_and_release(), media_type="text/event-stream")
except Exception as e:
    await worker_pool.release_worker(worker)
    raise e
```

The `try/finally` in `stream_and_release` ensures the worker is always returned to the pool, even if generation raises an exception or the client disconnects mid-stream. The `except` block outside the generator handles the case where an exception occurs before the generator starts.

> **What's happening: async queuing in practice**
>
> Imagine three users send messages at the same time to a single-GPU server. The first request calls `await worker_pool.acquire_worker()` and immediately gets the single worker (queue had one item). The second and third requests call `acquire_worker()` and find an empty queue — their coroutines suspend. When the first request finishes and calls `release_worker()`, `asyncio.Queue.put()` wakes up one of the waiting coroutines, which immediately receives the worker and begins its request. The third request continues waiting. No threads, no locks, no busy polling.

---

## 12.8 Multi-GPU load balancing

### Launching with multiple GPUs

✍️
```bash
python -m scripts.chat_web --model-tag=my-sft-run --num-gpus=4
```

This loads four independent model replicas, one per GPU:

```
Initializing worker pool with 4 GPUs...
Loading model on GPU 0...
Loading model on GPU 1...
Loading model on GPU 2...
Loading model on GPU 3...
All 4 workers initialized!
Server ready at http://localhost:8000
```

### How load balancing works

The worker pool's `asyncio.Queue` holds all four workers. When four requests arrive simultaneously, each gets a different worker on a different GPU and all four run in parallel. If a fifth request arrives while all four GPUs are busy, it waits in the queue.

This is data parallelism: four identical model copies, each handling independent requests. There is no communication between GPUs, no model sharding. Each GPU holds the full model in its VRAM.

The assignment is effectively FIFO. Whichever GPU finishes its current request first returns its worker to the queue first, and the next waiting request gets that GPU. Under equal load this naturally distributes work roughly evenly — no explicit round-robin counter is needed.

### VRAM requirements

With `--num-gpus=4`, you need four GPUs each with enough VRAM to hold the full model plus KV cache. For nanochat's default model configuration this is well within 24 GB GPUs. Multi-GPU is appropriate when you have concurrent users and underutilized hardware — it increases throughput (requests per second) but does not reduce latency (time per request).

---

## 12.9 Abuse prevention

Any model server exposed to the public internet needs protection against requests designed to exhaust resources or extract harmful outputs. nanochat's abuse prevention is in `validate_chat_request`:

```python
# scripts/chat_web.py

MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096
```

### Message and length limits

```python
def validate_chat_request(request: ChatRequest):
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Too many messages. ...")

    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")
        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(status_code=400, detail=f"Message {i} is too long. ...")
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(status_code=400, detail=f"Total conversation is too long. ...")
```

**Why these limits matter:** A request with a 500,000-character message would produce a massive token sequence that fills the context window and occupies a GPU worker for a very long time, blocking all other users. These limits ensure no single request can monopolize the server.

**The limits are generous by design.** 8,000 characters per message is about 2,000 words — much more than a normal chat message. 32,000 total characters is a very long conversation. These limits stop pathological inputs, not normal usage.

### Parameter clamping

```python
    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(status_code=400, detail=f"Temperature must be between ...")

    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(status_code=400, detail=f"top_k must be between ...")

    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(status_code=400, detail=f"max_tokens must be between ...")
```

**Temperature clamping to 0.0–2.0:** Extremely high temperatures (e.g., 1000.0) would make the softmax nearly uniform — the model would sample random tokens from the entire vocabulary, producing incoherent output that still occupies GPU time. Extremely negative temperatures are not meaningful. The 0.0–2.0 range covers every reasonable use case.

**Top-k clamping to 0–200:** `top_k=0` disables top-k filtering entirely and uses the full vocabulary. `top_k=200` caps the breadth of sampling. Values above 200 would approach the same behavior as top_k=0 for a 32K vocabulary.

**Max tokens clamped to 1–4096:** A request asking for 100,000 tokens would occupy a GPU worker for minutes. 4,096 is more than enough for any conversational response.

### Role validation

```python
    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant"]:
            raise HTTPException(status_code=400, detail=f"Message {i} has invalid role. ...")
```

The token format only supports `user` and `assistant` roles. Accepting arbitrary role strings could cause undefined model behavior or encoding errors.

---

## 12.10 The browser UI

The browser interface lives in `nanochat/ui.html`. It is a single HTML file containing all styles and JavaScript inline — no build step, no JavaScript framework.

### How the server serves it

```python
# scripts/chat_web.py

@app.get("/")
async def root():
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"
    )
    return HTMLResponse(content=html_content)
```

The `ui.html` file contains a JavaScript line `const API_URL = ...` that points to the server. When served by the same server that handles the API, the URL should be an empty string (same-origin). The server does a string replacement before sending the file so the browser calls `/chat/completions` on the same host and port, rather than hardcoding `localhost:8000`.

### How the frontend calls the API

The core of the frontend is `generateAssistantResponse()` in `ui.html`:

```javascript
// nanochat/ui.html

const response = await fetch(`${API_URL}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        messages: messages,
        temperature: currentTemperature,
        top_k: currentTopK,
        max_tokens: 512
    }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let fullResponse = '';

while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            if (data.token) {
                fullResponse += data.token;
                assistantContent.textContent = fullResponse;
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
    }
}
```

`response.body.getReader()` gives access to the HTTP response body as a stream of `Uint8Array` chunks. Each chunk is decoded as text and split into lines. Lines that start with `data: ` are parsed as JSON. When a `token` field is present, it is appended to `fullResponse` and the DOM is updated immediately — this is what causes characters to appear incrementally in the browser.

The `messages` array is stored in JavaScript memory. Every `sendMessage()` call pushes to this array, and the full array is sent in the next `fetch` call. The server holds no session state.

### Slash commands in the UI

The UI supports slash commands typed into the input field:

| Command | Effect |
|---|---|
| `/temperature 0.9` | Set temperature for subsequent requests |
| `/topk 100` | Set top-k for subsequent requests |
| `/clear` | Reset conversation |
| `/help` | List available commands |

These are handled entirely in JavaScript before any fetch call is made — they never reach the server.

### Editing and regeneration

Clicking a user message copies it back to the input field and removes all subsequent messages from the conversation. Clicking an assistant message removes it and regenerates from the same history. Both features work by slicing the `messages` array and re-rendering the DOM.

---

## 12.11 Health and stats endpoints

### `GET /health`

```python
# scripts/chat_web.py

@app.get("/health")
async def health():
    worker_pool = getattr(app.state, 'worker_pool', None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0
    }
```

The `/health` endpoint returns a quick status snapshot. It is designed to be polled by a load balancer or monitoring system. `available_workers` tells you how many GPUs are currently free. The browser UI calls `/health` on page load to confirm the server is running.

✍️
```bash
curl http://localhost:8000/health
```

Expected response when one worker is free:

```json
{"status": "ok", "ready": true, "num_gpus": 1, "available_workers": 1}
```

### `GET /stats`

```python
@app.get("/stats")
async def stats():
    worker_pool = app.state.worker_pool
    return {
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [
            {"gpu_id": w.gpu_id, "device": str(w.device)}
            for w in worker_pool.workers
        ]
    }
```

`/stats` gives more detail: total workers, currently available, currently busy, and a list of each worker's GPU ID and device string. `busy_workers` is computed as `total - available` — workers that are in a request but not yet returned to the queue.

✍️
```bash
curl http://localhost:8000/stats
```

For a 2-GPU server with one request in progress:

```json
{
  "total_workers": 2,
  "available_workers": 1,
  "busy_workers": 1,
  "workers": [
    {"gpu_id": 0, "device": "cuda:0"},
    {"gpu_id": 1, "device": "cuda:1"}
  ]
}
```

---

## 12.12 Running the web server

### Single GPU

✍️
```bash
python -m scripts.chat_web --model-tag=my-sft-run
```

Output:

```
Initializing worker pool with 1 GPUs...
Loading model on GPU 0...
All 1 workers initialized!
Server ready at http://localhost:8000
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Open `http://localhost:8000` in your browser. You should see the nanochat chat interface.

### Multiple GPUs

✍️
```bash
python -m scripts.chat_web --model-tag=my-sft-run --num-gpus=4
```

### Custom port and host

✍️
```bash
python -m scripts.chat_web --model-tag=my-sft-run --port=9000 --host=127.0.0.1
```

`--host=0.0.0.0` (the default) binds to all network interfaces, making the server reachable from other machines. `--host=127.0.0.1` restricts it to localhost only.

### Deploying on a remote server (port forwarding)

If your model runs on a cloud GPU machine and you want to use it from your laptop:

✍️
```bash
# On the remote server:
python -m scripts.chat_web --model-tag=my-sft-run --host=0.0.0.0 --port=8000

# On your laptop (SSH tunnel):
ssh -L 8000:localhost:8000 user@remote-server-ip
```

Then open `http://localhost:8000` on your laptop. The SSH tunnel forwards all traffic from your laptop's port 8000 to the remote server's port 8000.

### Conversation logging

The server logs every conversation to the console:

```
2026-03-22 14:31:05 - ====================
2026-03-22 14:31:05 - [USER]: What is the capital of France?
2026-03-22 14:31:05 - --------------------
2026-03-22 14:31:09 - [ASSISTANT] (GPU 0): The capital of France is Paris.
2026-03-22 14:31:09 - ====================
```

This happens inside `stream_and_release()` in `chat_completions`. The full response is accumulated from the stream and logged after generation completes.

---

## 12.13 CPU mode

When no GPU is available (or when `--num-gpus=0` is passed), the server runs on CPU:

✍️
```bash
python -m scripts.chat_web --model-tag=my-sft-run --num-gpus=1 --device-type=cpu
```

The WorkerPool initializes a single worker using `torch.device("cpu")`. Generation still works — it is just much slower. A small model (e.g., 15M parameters) may produce tokens at 1–5 tokens per second on a modern laptop CPU, compared to 50–200+ tokens per second on a GPU.

CPU mode is useful for:
- **Development and testing** — verifying the server starts, routes work, and SSE streaming functions correctly without needing a GPU.
- **Running on a machine without NVIDIA GPUs** — macOS with Apple Silicon (`--device-type=mps`), CPU-only cloud instances, or developer laptops.

For MPS (Apple Silicon):

✍️
```bash
python -m scripts.chat_web --model-tag=my-sft-run --device-type=mps
```

The worker pool initializes a single worker on the MPS device. Multi-GPU (`--num-gpus > 1`) is only supported for CUDA; on CPU and MPS, the server always uses a single worker regardless of the `--num-gpus` flag.

---

## 12.14 Hands-on: run both interfaces

Try the following sequence to see both interfaces working:

### Step 1: Start the CLI

✍️
```bash
python -m scripts.chat_cli --model-tag=my-sft-run
```

Type a few messages. Notice tokens appearing as they are generated. Try `clear` to reset the conversation and observe that the model forgets what was discussed before.

### Step 2: Use prompt mode

✍️
```bash
python -m scripts.chat_cli --model-tag=my-sft-run --prompt="Explain what a transformer is in one sentence."
```

The process exits after printing one response. This is convenient for scripting or automated evaluation.

### Step 3: Start the web server

✍️
```bash
python -m scripts.chat_web --model-tag=my-sft-run
```

Open `http://localhost:8000`. Send a message and watch tokens stream in character by character.

### Step 4: Check the server status

✍️
```bash
curl http://localhost:8000/health
curl http://localhost:8000/stats
```

### Step 5: Send a request directly from the terminal

✍️
```bash
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "temperature": 0.7}' \
  --no-buffer
```

You should see SSE chunks printed to the terminal as generation proceeds.

### Step 6: Test abuse prevention

✍️
```bash
# This should return a 400 error
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hi"}], "temperature": 99.0}'
```

Expected:

```json
{"detail": "Temperature must be between 0.0 and 2.0"}
```

---

> **What's happening: the full token journey in the web server**
>
> 1. Browser calls `fetch('/chat/completions')` with JSON containing `messages`.
> 2. FastAPI parses the JSON into a `ChatRequest` Pydantic object.
> 3. `validate_chat_request` checks all limits and raises a 400 if anything is out of bounds.
> 4. `acquire_worker()` removes a Worker from the asyncio Queue (or suspends if all workers are busy).
> 5. The conversation history is encoded into a flat token list, ending with `<|assistant_start|>`.
> 6. `generate_stream()` runs `engine.generate()`, which runs the model forward pass on the GPU.
> 7. Each token is decoded, checked for UTF-8 completeness, and yielded as an SSE chunk: `data: {"token": "..."}\n\n`.
> 8. `StreamingResponse` writes each chunk to the HTTP response body as it arrives.
> 9. The browser's `ReadableStream` reader receives the chunks, parses the SSE format, and appends each token to the DOM.
> 10. When generation ends, `done: true` is sent, `release_worker()` returns the Worker to the Queue, and the HTTP response closes.

---

## Check your understanding

**1.** After a user sends three messages and receives three responses in the CLI, `conversation_tokens` contains the tokens for all six turns. If the user types `clear`, what happens to `conversation_tokens`, and how does this affect the model's behavior on the next turn?

**2.** Two users send requests simultaneously to a single-GPU web server. Describe what happens to each request at the `await worker_pool.acquire_worker()` line. What is the mechanism that prevents them from both trying to use the GPU at the same time?

**3.** The `generate_stream` function accumulates all response tokens in `accumulated_tokens` and calls `tokenizer.decode(accumulated_tokens)` on every token. Why not just call `tokenizer.decode([token])` on the current token alone? Under what condition would decoding a single token return the replacement character `'â'`?

---

## What's next

You have now built and run both chat interfaces. You understand how conversation state flows as a flat token list, how a Python generator becomes a real-time HTTP stream, and how a queue of GPU workers serves concurrent users.

The remaining chapters cover:

- **Chapter 13 — Evaluation:** Measuring model quality with benchmarks (DCLM CORE, math tasks), and understanding bits-per-byte as a training signal.
- **Chapter 14 — Scaling:** What changes when you scale from a laptop experiment to an 8×H100 run — batch size, gradient accumulation, mixed precision, and the tradeoffs involved.
