"""Constants."""

from enum import Enum


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
