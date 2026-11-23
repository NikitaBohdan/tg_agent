"""Technical logging (console + rotating file).

This is the *debug* log. The per-message *decision log* required by the task
is a separate module (stage 4) so it stays clean and one-line-per-message.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
    root.addHandler(console)

    file = RotatingFileHandler(
        log_dir / "agent.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(file)

    # Telethon is chatty at INFO (reconnects etc.) — keep it at WARNING.
    for noisy in ("telethon", "httpx", "httpx2", "httpcore", "openai", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
