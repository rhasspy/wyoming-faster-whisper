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

    @property
    def supports_streaming(self) -> bool:
        """Whether this transcriber can process audio chunks incrementally.

        When False (the default), callers must buffer the entire utterance and
        use transcribe(). When True, start_stream() returns a StreamingSession
        that transcribes audio as it arrives.
        """
        return False

    def start_stream(
        self,
        language: Optional[str] = None,
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> "StreamingSession":
        """Begin a new streaming transcription session.

        The returned session holds all per-utterance state, so a single
        (shared) transcriber can drive multiple concurrent sessions.

        Only valid when supports_streaming is True.
        """
        raise NotImplementedError


class StreamingSession(ABC):
    """A single in-progress streaming transcription.

    Created by Transcriber.start_stream(). Holds per-utterance state so it is
    safe to use one session per client connection even when the underlying
    transcriber is shared.
    """

    @abstractmethod
    def accept_chunk(self, audio_bytes: bytes) -> None:
        """Feed a chunk of audio (16Khz 16-bit mono PCM) to the stream."""

    @abstractmethod
    def finish(self) -> str:
        """Finish the stream and return the final transcript."""
