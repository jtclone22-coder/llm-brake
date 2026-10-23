"""Terminal chat with a working Stop: press Ctrl+C during an answer to stop it (verified when possible).

    python examples/chat_with_stop.py --model qwen3:14b
    python examples/chat_with_stop.py --openai http://127.0.0.1:8080 --model local
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmbrake import Budget, Gate, GateHeld, GenerationStopped, OllamaClient, OpenAICompatClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--openai", help="base URL of an OpenAI-compatible server instead of Ollama")
    parser.add_argument("--limit", type=float, default=120, help="seconds per answer")
    args = parser.parse_args()
    gate = Gate()
    client = OpenAICompatClient(args.openai, gate=gate) if args.openai else OllamaClient(args.ollama, gate=gate)
    history = []
    current: dict = {}
    signal.signal(signal.SIGINT, lambda *_: current["budget"].cancel() if current.get("budget") else sys.exit(0))
    print(f"Chatting with {args.model}. Ctrl+C stops an answer; Ctrl+C at the prompt quits.")
    while True:
        current.pop("budget", None)
        try:
            question = input("\nyou> ").strip()
        except EOFError:
            return 0
        if not question:
            continue
        history.append({"role": "user", "content": question})
        current["budget"] = Budget(args.limit)
        answer = []
        try:
            for text in client.chat({"model": args.model, "messages": history}, current["budget"]):
                answer.append(text)
                print(text, end="", flush=True)
        except GenerationStopped as stop:
            print(f"\n[stopped: {stop.reason}; server idle {'confirmed' if stop.verified else 'NOT confirmed'}]")
        except GateHeld as held:
            print(f"[waiting for the server to go idle: {held}]")
            history.pop()
            continue
        history.append({"role": "assistant", "content": "".join(answer)})


if __name__ == "__main__":
    sys.exit(main())
