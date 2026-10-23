# llm-brake

**A Stop button that actually stops your local LLM.** Real deadlines, real cancellation, and proof that the GPU went idle, for Ollama, llama.cpp server, LM Studio, vLLM and any OpenAI-compatible server. Pure Python standard library (`psutil` optional).

## Install

```
py -m pip install git+https://github.com/jtclone22-coder/llm-brake
```

Python 3.9 or newer, standard library only. `psutil` is optional (it lets the Ollama check find the runner processes).

## The problem

You build a chat app on a local model and add a Stop button. The user presses it, your UI says "Stopped", and the GPU keeps generating for another two minutes. These are the reasons:

- A blocking socket read can't be interrupted from another thread, so your "cancel" waits for the next token. During a long prompt prefill, that wait has no end.
- Setting a socket timeout to get out breaks `http.client` for every read after it.
- In FastAPI/Starlette, when the browser disconnects, the server just stops reading from your generator. It is only closed when the garbage collector gets to it, and the model keeps running meanwhile.
- Even after you close the connection, nobody checks whether the server actually cancelled. The next request lands on a GPU that is still busy.

## What llm-brake does

- **`Budget`**: one object for a request's deadline and its stop signal, with the reason recorded (`cancelled`, `deadline`, `client_disconnected`). Child budgets never outlive their parent.
- **Interruptible streaming**: every read waits with `select()` in 0.25 s slices, so Stop or a deadline closes the connection within a quarter second, even before the first token.
- **Verified stop**: after closing the connection it asks the server whether anything is still generating (`GET /slots` on llama.cpp; for Ollama it finds the llama.cpp runner processes). A stop is reported as verified only when every slot is idle.
- **`Gate`**: one generation at a time per GPU, with a bounded queue. After an *unverified* stop it HOLDs, so new requests don't pile onto a busy GPU. The hold clears itself as soon as the verifier sees the server idle.
- **`pump()`**: drives a blocking generator from asyncio frameworks and cancels the budget the moment the client disconnects.

## Use it

```python
from llmbrake import Budget, Gate, GenerationStopped, OllamaClient

client = OllamaClient("http://127.0.0.1:11434", gate=Gate())
budget = Budget(90)                      # whole request: 90 s max
# from a UI button or another thread:  budget.cancel()

try:
    for text in client.chat({"model": "qwen3:14b", "messages": [{"role": "user", "content": "Hi"}]}, budget):
        print(text, end="", flush=True)
except GenerationStopped as stop:
    print(f"\n[{stop.reason}; server idle: {stop.verified}]")
```

llama.cpp server, LM Studio, vLLM or any `/v1/chat/completions` server:

```python
from llmbrake import OpenAICompatClient
client = OpenAICompatClient("http://127.0.0.1:8080")   # uses /slots automatically when the server has it
```

FastAPI / Starlette:

```python
from llmbrake import Budget, pump

@app.post("/chat")
def chat(req: ChatRequest):
    budget = Budget(120)
    events = (f"data: {json.dumps({'text': t})}\n\n" for t in client.chat(req.payload(), budget))
    return StreamingResponse(pump(events, budget), media_type="text/event-stream")
```

## Notes

- **Ollama verification needs `pip install psutil`.** Without it the client still stops, but reports the stop as *not verified* and never pretends otherwise.
- **The Ollama runner must be a llama.cpp server with `/slots`.** That is the case for Ollama versions that spawn `llama-server`.
- **Verification only checks what it can see.** It reads the server's own slot state, so if a different program is using the same server, its work counts as "busy" too.

## Tests

```
python -m unittest discover -s tests
```

The tests run against a fake local server that streams tokens and cancels on disconnect. It also has a "stubborn" mode that keeps generating after a disconnect, to prove the gate holds and then recovers.

## License

MIT
