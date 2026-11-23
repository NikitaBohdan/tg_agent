"""End-to-end voice check without Telegram: speak a phrase with edge-tts, convert it to
OGG/Opus and transcribe it back with the configured STT.

Run:  python -m scripts.voice_check
      python -m scripts.voice_check "Could we do half past five instead?"

The first run with STT_PROVIDER=local downloads the Whisper model (small.en ≈ 0.5 GB).
The voice note is kept at data/voice_check.ogg so you can listen to it.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import time
from pathlib import Path

from agent.config import get_settings
from agent.voice import build_stt, synthesize_voice_note, temp_dir


async def main(text: str) -> int:
    s = get_settings()
    s.ensure_dirs()
    with temp_dir() as d:
        t0 = time.monotonic()
        ogg, duration = await synthesize_voice_note(text, s.tts_voice, Path(d))
        keep = s.data_dir / "voice_check.ogg"
        shutil.copy(ogg, keep)
        print(f"TTS  ({s.tts_voice}): {duration}s voice note in {time.monotonic() - t0:.1f}s, saved to {keep}")

    stt = build_stt(s)
    t0 = time.monotonic()
    heard = await stt.transcribe(keep)
    print(f"STT  ({s.stt_provider}): {heard!r} in {time.monotonic() - t0:.1f}s")
    print(f"sent: {text!r}")
    return 0


if __name__ == "__main__":
    phrase = " ".join(sys.argv[1:]) or "Good afternoon, I'd like to book a table for two today at 5pm, please."
    sys.exit(asyncio.run(main(phrase)))
