"""Tests for pure model/library selection logic.

These are dependency-free: guess_stt_library takes backend-availability flags
as arguments, so the real STT backends need not be installed.
"""

import pytest

from wyoming_faster_whisper.const import SttLibrary
from wyoming_faster_whisper.models import guess_stt_library

_ALL_AVAILABLE = dict(
    has_transformers=True,
    has_sherpa=True,
    has_onnx_asr=True,
    has_funasr=True,
)


def _guess(preferred, language, model=None, **avail):
    flags = {**_ALL_AVAILABLE, **avail}
    return guess_stt_library(preferred, model, language, **flags)


# --- AUTO: per-language backend selection ---------------------------------


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        # FunASR (SenseVoice) languages, including locale-style codes.
        ("zh", SttLibrary.FUNASR),
        ("zh-CN", SttLibrary.FUNASR),
        ("zh-TW", SttLibrary.FUNASR),
        ("zh-HK", SttLibrary.FUNASR),  # Hong Kong -> Cantonese
        ("yue", SttLibrary.FUNASR),
        ("ja", SttLibrary.FUNASR),
        ("ko", SttLibrary.FUNASR),
        # Other specialized backends are unaffected.
        ("en", SttLibrary.SHERPA),
        ("ru", SttLibrary.ONNX_ASR),
        # Everything else defaults to faster-whisper.
        ("de", SttLibrary.FASTER_WHISPER),
        (None, SttLibrary.FASTER_WHISPER),
    ],
)
def test_auto_selects_per_language_backend(language, expected) -> None:
    assert _guess(SttLibrary.AUTO, language) == expected


def test_auto_funasr_languages_fall_back_when_funasr_missing() -> None:
    # zh would route to FunASR, but it isn't installed -> faster-whisper.
    assert (
        _guess(SttLibrary.AUTO, "zh-CN", has_funasr=False)
        == SttLibrary.FASTER_WHISPER
    )


def test_auto_with_explicit_model_skips_per_language_selection() -> None:
    # A forced model disables auto backend selection (stays faster-whisper).
    assert (
        _guess(SttLibrary.AUTO, "zh-CN", model="some/model")
        == SttLibrary.FASTER_WHISPER
    )


# --- Explicit library: dependency fallback --------------------------------


def test_explicit_funasr_kept_when_available() -> None:
    assert _guess(SttLibrary.FUNASR, "zh") == SttLibrary.FUNASR


def test_explicit_funasr_falls_back_when_missing() -> None:
    assert (
        _guess(SttLibrary.FUNASR, "zh", has_funasr=False)
        == SttLibrary.FASTER_WHISPER
    )


def test_explicit_faster_whisper_is_passthrough() -> None:
    assert (
        _guess(SttLibrary.FASTER_WHISPER, "en", has_sherpa=False)
        == SttLibrary.FASTER_WHISPER
    )
