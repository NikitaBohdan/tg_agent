"""TTS pipeline without network: edge-tts is replaced by an ffmpeg-generated tone,
the real ffmpeg/ffprobe conversion to Telegram's OGG/Opus voice format is exercised."""

import asyncio
import shutil
import subprocess
import sys
import types

import pytest

from agent import voice

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


class FakeCommunicate:
    def __init__(self, text, voice):
        self.text = text

    async def save(self, path):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=2", path], check=True)


def test_synthesize_produces_ogg_opus(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=FakeCommunicate))
    ogg, duration = asyncio.run(voice.synthesize_voice_note("Hello there", "en-GB-RyanNeural", tmp_path))
    codec = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                            "stream=codec_name", "-of", "csv=p=0", str(ogg)],
                           capture_output=True, text=True, check=True).stdout.strip()
    assert ogg.suffix == ".ogg" and codec == "opus" and duration == 2


def test_decode_pcm16k(tmp_path):
    ogg = tmp_path / "tone.ogg"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=2", "-c:a", "libopus", str(ogg)], check=True)
    pcm = voice.decode_pcm16k(ogg)
    assert pcm.dtype.name == "float32" and abs(len(pcm) - 32000) < 1600


def test_tts_failure_raises_voice_error(tmp_path, monkeypatch):
    class Broken:
        def __init__(self, *a):
            pass

        async def save(self, path):
            raise OSError("no network")

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=Broken))
    with pytest.raises(voice.VoiceError):
        asyncio.run(voice.synthesize_voice_note("Hi", "en-GB-RyanNeural", tmp_path))
