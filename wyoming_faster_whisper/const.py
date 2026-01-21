"""Constants."""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional, Union


class SttLibrary(str, Enum):
    """Speech-to-text library."""

    AUTO = "auto"
    FASTER_WHISPER = "faster-whisper"
    TRANSFORMERS = "transformers"
    SHERPA = "sherpa"
    ONNX_ASR = "onnx-asr"


AUTO_LANGUAGE = "auto"
AUTO_MODEL = "auto"

PARAKEET_LANGUAGES = {
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "it",
    "lv",
    "lt",
    "mt",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "es",
    "sv",
    "ru",
    "uk",
}


class Transcriber(ABC):
    """Base class for transcribers."""

    @abstractmethod
    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        pass
