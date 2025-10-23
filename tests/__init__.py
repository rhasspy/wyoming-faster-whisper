"""Tests for wyoming-faster-whisper."""

from pathlib import Path

_DIR = Path(__file__).parent
_PROGRAM_DIR = _DIR.parent
_LOCAL_DIR = _PROGRAM_DIR / "local"
_SAMPLES_PER_CHUNK = 1024

# Need to give time for the model to download
_START_TIMEOUT = 60
_TRANSCRIBE_TIMEOUT = 60
