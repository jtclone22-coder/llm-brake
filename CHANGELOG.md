# Changelog

## 0.1.0 (2026-10-22)

- `Budget`: a deadline plus a stop signal, with the reason recorded. Child budgets are capped by their parent.
- Interruptible HTTP streaming: select() in short slices, so a stop takes effect within about 0.25 s, even during prefill.
- `OllamaClient` (/api/chat) and `OpenAICompatClient` (/v1/chat/completions) with `GenerationStopped(reason, verified, partial)`.
- Stop verification: `SlotsVerifier` (llama.cpp /slots), `OllamaRunnerVerifier` (finds Ollama's llama-server runners; needs psutil) and `NullVerifier`.
- `Gate`: one generation at a time with a bounded queue. It holds after an unverified stop and recovers by itself once the server is verified idle.
- `pump()`: an asyncio bridge that cancels the budget when the client disconnects.
- Error responses are read reliably. With `Connection: close`, the socket stays open until the response body is read, even when the body arrives after the headers. Before this fix, such a body could fail with "file descriptor cannot be a negative integer"; a test now sends a split response to prove it.
