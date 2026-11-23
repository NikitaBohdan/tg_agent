"""Voice in / voice out.

STT turns the manager's voice into text, selectable with STT_PROVIDER:
  local  — faster-whisper on CPU, free, offline after a one-time model download
  openai — OpenAI transcription API (paid, cheap), uses OPENAI_API_KEY
  groq   — Groq Whisper (free tier), uses GROQ_API_KEY

TTS turns the agent's text into a voice note: edge-tts with a British neural voice (free),
then ffmpeg converts MP3 to OGG/Opus, the format Telegram shows as a voice note.

Every function raises VoiceError on failure; the runner falls back to text.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from .config import Settings

log = logging.getLogger(__name__)


class VoiceError(Exception):
    pass


# --- STT -----------------------------------------------------------------------


class LocalWhisper:
    """faster-whisper; the model is loaded lazily on first use (download once)."""

    def __init__(self, model: str):
        self.model_name = model
        self._model = None
        self._lock = asyncio.Lock()

    async def transcribe(self, audio: Path) -> str:
        async with self._lock:  # one transcription at a time on CPU
            return await asyncio.to_thread(self._run, audio)

    async def warmup(self) -> None:
        """Load the model in the background so the first voice note isn't slow."""
        async with self._lock:
            await asyncio.to_thread(self._load)

    def _load(self) -> None:
        if self._model is None:
            from faster_whisper import WhisperModel

            log.info("loading local Whisper model %r (downloaded on first use only)…", self.model_name)
            self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")

    def _run(self, audio: Path) -> str:
        self._load()
        # Decode with ffmpeg ourselves instead of faster-whisper's PyAV path:
        # recent PyAV releases broke its decode_audio() (unexpected 'metadata_errors').
        pcm = decode_pcm16k(audio)
        segments, _info = self._model.transcribe(pcm, language="en", vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()


def decode_pcm16k(audio: Path):
    """Decode any audio file to a mono 16 kHz float32 numpy array (what Whisper expects)."""
    import subprocess

    import numpy as np

    if shutil.which("ffmpeg") is None:
        raise VoiceError("ffmpeg not found on PATH")
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(audio),
         "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, timeout=60,
    )
    if proc.returncode != 0:
        raise VoiceError(f"ffmpeg decode failed: {proc.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


class OpenAIWhisper:
    def __init__(self, api_key: str, model: str):
        import openai

        self._client = openai.AsyncOpenAI(api_key=api_key, timeout=60, max_retries=2)
        self.model_name = model

    async def transcribe(self, audio: Path) -> str:
        with audio.open("rb") as f:
            r = await self._client.audio.transcriptions.create(model=self.model_name, file=f, language="en")
        return (r.text or "").strip()


class GroqWhisper:
    def __init__(self, api_key: str, model: str):
        import groq

        self._client = groq.AsyncGroq(api_key=api_key, timeout=60, max_retries=2)
        self.model_name = model

    async def transcribe(self, audio: Path) -> str:
        with audio.open("rb") as f:
            r = await self._client.audio.transcriptions.create(
                model=self.model_name, file=(audio.name, f.read()), language="en"
            )
        return (r.text or "").strip()


def build_stt(s: Settings):
    if s.stt_provider == "local":
        return LocalWhisper(s.stt_model or "small.en")
    if s.stt_provider == "openai":
        return OpenAIWhisper(s.api_key_for("openai"), s.stt_model or "gpt-4o-mini-transcribe")
    if s.stt_provider == "groq":
        return GroqWhisper(s.api_key_for("groq"), s.stt_model or "whisper-large-v3-turbo")
    raise ValueError(f"Unknown STT_PROVIDER: {s.stt_provider}")


# --- TTS -----------------------------------------------------------------------


async def synthesize_voice_note(text: str, voice: str, workdir: Path) -> tuple[Path, int]:
    """Return (path to .ogg Opus file, duration in whole seconds)."""
    import edge_tts

    mp3, ogg = workdir / "reply.mp3", workdir / "reply.ogg"
    try:
        await edge_tts.Communicate(text, voice).save(str(mp3))
    except Exception as e:
        raise VoiceError(f"edge-tts failed: {e}") from e
    if not mp3.exists() or mp3.stat().st_size == 0:
        raise VoiceError("edge-tts produced no audio")

    await _run("ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
               "-c:a", "libopus", "-b:a", "32k", "-ac", "1", "-ar", "48000", str(ogg))
    out = await _run("ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(ogg))
    try:
        duration = max(1, round(float(out.strip())))
    except ValueError:
        duration = max(1, len(text) // 14)  # ~14 chars/sec speech
    return ogg, duration


async def _run(*cmd: str) -> str:
    if shutil.which(cmd[0]) is None:
        raise VoiceError(f"{cmd[0]} not found on PATH")
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=60)
    except TimeoutError as e:
        proc.kill()
        raise VoiceError(f"{cmd[0]} timed out") from e
    if proc.returncode != 0:
        raise VoiceError(f"{cmd[0]} failed: {err.decode(errors='replace')[:300]}")
    return out.decode(errors="replace")


def temp_dir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="tg-agent-voice-")
