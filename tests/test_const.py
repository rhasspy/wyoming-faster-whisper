"""Tests for language-code helpers in const.

These are dependency-free (no STT backend required) so they run everywhere.
"""

import pytest

from wyoming_faster_whisper.const import sense_voice_language


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        # Bare SenseVoice codes pass through.
        ("zh", "zh"),
        ("yue", "yue"),
        ("ja", "ja"),
        ("ko", "ko"),
        ("en", "en"),
        # Locale-style codes (e.g. from Home Assistant / intent-sentences).
        ("zh-CN", "zh"),
        ("zh-TW", "zh"),
        ("zh-HK", "yue"),  # Hong Kong audio is typically Cantonese
        # Case-insensitive.
        ("ZH-cn", "zh"),
        ("JA", "ja"),
        # Unsupported / empty -> None (caller falls back to auto-detect).
        ("de", None),
        ("es", None),
        ("auto", None),
        ("", None),
        (None, None),
    ],
)
def test_sense_voice_language(code, expected) -> None:
    assert sense_voice_language(code) == expected
