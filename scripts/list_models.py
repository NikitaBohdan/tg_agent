"""List text models available to YOUR API keys, so LLM_MODEL is never guessed.

Run:  python -m scripts.list_models            # all providers with a key in .env
      python -m scripts.list_models --ping     # also send a tiny JSON request to LLM_MODEL
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from agent.config import get_settings


def list_openai(key: str) -> None:
    import openai

    client = openai.OpenAI(api_key=key)
    print("\n== OpenAI (chat models) ==")
    skip = ("audio", "realtime", "tts", "transcribe", "image", "search", "embedding", "moderation", "dall-e", "whisper")
    for m in sorted(client.models.list(), key=lambda m: m.id):
        if m.id.startswith(("gpt-", "o")) and not any(x in m.id for x in skip):
            print(f"  {m.id}")


def list_gemini(key: str) -> None:
    from google import genai

    client = genai.Client(api_key=key)
    print("\n== Gemini (supports generateContent) ==")
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            print(f"  {m.name.removeprefix('models/')}")


def list_groq(key: str) -> None:
    import groq

    client = groq.Groq(api_key=key)
    print("\n== Groq ==")
    for m in sorted(client.models.list().data, key=lambda m: m.id):
        if m.active and not any(x in m.id for x in ("whisper", "tts", "guard")):
            print(f"  {m.id}")


async def ping() -> None:
    from agent.llm import build_chain

    s = get_settings()
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    try:
        chain = build_chain(s)
    except ValueError as e:
        print(f"PING: config error: {e}")
        return
    for p in chain:
        print(f"PING {p.name} ... (timeout {s.llm_timeout_s:.0f}s)", flush=True)
        t0 = time.monotonic()
        try:
            out = await p.complete_json("You are a test.", 'Return {"ok": true}.', schema)
            print(f"  OK in {time.monotonic() - t0:.1f}s, answer: {out[:80]}")
        except Exception as e:  # report and continue with the next provider
            print(f"  FAILED after {time.monotonic() - t0:.1f}s: {type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", action="store_true")
    args = ap.parse_args()
    s = get_settings()

    any_key = False
    for name, fn in (("openai", list_openai), ("gemini", list_gemini), ("groq", list_groq)):
        try:
            key = s.api_key_for(name)
        except ValueError:
            print(f"(skip {name}: no API key)")
            continue
        any_key = True
        try:
            fn(key)
        except Exception as e:
            print(f"  failed to list {name} models: {type(e).__name__}: {e}")
    if not any_key:
        print("No LLM API keys in .env")
        return 1
    if args.ping:
        asyncio.run(ping())
    return 0


if __name__ == "__main__":
    sys.exit(main())
