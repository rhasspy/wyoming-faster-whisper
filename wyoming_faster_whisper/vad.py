"""Clip audio to its speech region with pysilero-vad before running STT.

Trimming leading/trailing non-speech reduces Whisper hallucinations on silence
and shortens the audio the model has to process. Audio is 16 kHz 16-bit mono PCM.
"""

import logging
import wave
from pathlib import Path
from typing import Optional, Union

from pysilero_vad import SileroVoiceActivityDetector

_LOGGER = logging.getLogger(__name__)

RATE = 16000
WIDTH = 2  # bytes per sample (16-bit)


def clip_to_speech(
    audio: bytes,
    threshold: float = 0.5,
    pad_ms: int = 400,
    detector: Optional[SileroVoiceActivityDetector] = None,
) -> bytes:
    """Trim leading/trailing non-speech, keeping pad_ms of context on each side.

    Returns the clipped 16 kHz 16-bit mono PCM. Returns b"" if no speech is
    detected (callers should decide whether to fall back to the full audio).
    A new detector is created per call unless one is supplied; the detector is
    stateful, so a shared instance must not be used across threads.
    """
    if detector is None:
        detector = SileroVoiceActivityDetector()
    else:
        detector.reset()

    chunk_bytes = detector.chunk_bytes()
    chunk_samples = detector.chunk_samples()

    # Not enough audio for even a single VAD window.
    if len(audio) < chunk_bytes:
        return b""

    probs = list(detector.process_chunks(audio))
    speech = [i for i, p in enumerate(probs) if p >= threshold]
    if not speech:
        return b""

    pad_chunks = max(0, (pad_ms * RATE // 1000) // chunk_samples)
    start = max(0, speech[0] - pad_chunks)
    end = min(len(probs), speech[-1] + 1 + pad_chunks)
    return audio[start * chunk_bytes : end * chunk_bytes]


def clip_wav_to_speech(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    threshold: float = 0.5,
    pad_ms: int = 400,
) -> bool:
    """Clip a 16 kHz 16-bit mono WAV to its speech region, writing to dst_path.

    Returns True if speech was found and dst_path now holds the clipped audio,
    or False if the WAV couldn't be read or no speech was detected (in which
    case dst_path is not written and callers should use the original audio).
    """
    try:
        with wave.open(str(src_path), "rb") as wav_file:
            audio = wav_file.readframes(wav_file.getnframes())
    except (wave.Error, OSError) as err:
        _LOGGER.warning("VAD clip: could not read %s: %s", src_path, err)
        return False

    clipped = clip_to_speech(audio, threshold=threshold, pad_ms=pad_ms)
    if not clipped:
        _LOGGER.debug("VAD clip: no speech detected, using full audio")
        return False

    clipped_file: wave.Wave_write = wave.open(str(dst_path), "wb")
    with clipped_file:
        clipped_file.setframerate(RATE)
        clipped_file.setsampwidth(WIDTH)
        clipped_file.setnchannels(1)
        clipped_file.writeframes(clipped)

    _LOGGER.debug(
        "VAD clip: %d ms -> %d ms",
        len(audio) // WIDTH * 1000 // RATE,
        len(clipped) // WIDTH * 1000 // RATE,
    )
    return True
