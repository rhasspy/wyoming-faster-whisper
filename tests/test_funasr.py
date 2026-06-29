"""Multilingual tests for the FunASR (SenseVoice) backend.

SenseVoiceSmall officially supports five languages: Chinese (zh), Cantonese
(yue), English (en), Japanese (ja) and Korean (ko). These tests confirm the
backend transcribes each of them and document that other languages (e.g.
German/Spanish) are NOT reliably supported.

Audio fixtures were generated with Home Assistant Cloud TTS and resampled to
16 kHz mono. Expected text is the exact (normalized) output SenseVoiceSmall
produces for each clip, so the assertions are deterministic for a fixed model.
"""

import unicodedata

import pytest

from . import _DIR

# Skip the whole module unless FunASR is installed.
pytest.importorskip("funasr")

from wyoming_faster_whisper.funasr_handler import FunASRTranscriber  # noqa: E402

_MODEL = "FunAudioLLM/SenseVoiceSmall"


def _normalize(text: str) -> str:
    """Drop punctuation/whitespace and lowercase for script-agnostic compare.

    CJK output has no word spaces and SenseVoice adds ITN punctuation, so a
    plain string compare is too strict. Note Chinese/Cantonese output is in
    simplified characters regardless of the input script.
    """
    text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return text.replace(" ", "").strip().lower()


# (sensevoice language, fixture suffix, expected transcript)
_SUPPORTED = [
    ("en", "en", "Turn on the living room light."),
    ("zh", "zh", "打开客厅的灯。"),
    ("ja", "ja", "リビングの電気をつけて。"),
    ("ko", "ko", "거실 불 켜줘."),
    # SenseVoice emits Cantonese in simplified Han (keeps the colloquial 嘅).
    ("yue", "yue", "开客厅嘅灯。"),
    # Locale-style codes (as sent by Home Assistant / intent-sentences) are
    # normalized to the base SenseVoice language before decoding.
    ("zh-CN", "zh", "打开客厅的灯。"),
    ("zh-HK", "yue", "开客厅嘅灯。"),  # Hong Kong -> Cantonese
]

# Languages SenseVoiceSmall does not support; transcription is garbled. Kept as
# xfail so the limitation is documented rather than silently untested.
_UNSUPPORTED = [
    ("auto", "de", "Schalte das Wohnzimmerlicht ein."),
    ("auto", "es", "Enciende la luz del salón."),
]


@pytest.fixture(scope="module")
def transcriber(tmp_path_factory) -> FunASRTranscriber:
    cache_dir = tmp_path_factory.mktemp("funasr-cache")
    return FunASRTranscriber(_MODEL, cache_dir=cache_dir, device="cpu")


@pytest.mark.parametrize(("language", "suffix", "expected"), _SUPPORTED)
def test_funasr_supported_languages(
    transcriber: FunASRTranscriber, language: str, suffix: str, expected: str
) -> None:
    wav_path = _DIR / f"living_room_light_{suffix}.wav"
    text = transcriber.transcribe(wav_path, language=language)
    assert _normalize(text) == _normalize(expected)


@pytest.mark.parametrize(("language", "suffix", "expected"), _UNSUPPORTED)
@pytest.mark.xfail(reason="SenseVoiceSmall does not support this language", strict=True)
def test_funasr_unsupported_languages(
    transcriber: FunASRTranscriber, language: str, suffix: str, expected: str
) -> None:
    wav_path = _DIR / f"living_room_light_{suffix}.wav"
    text = transcriber.transcribe(wav_path, language=language)
    assert _normalize(text) == _normalize(expected)
