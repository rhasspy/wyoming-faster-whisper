"""Clip audio to its speech region with pysilero-vad before running STT.

Trimming leading/trailing non-speech reduces Whisper hallucinations on silence
and shortens the audio the model has to process. Audio is 16 kHz 16-bit mono PCM.
"""

import logging
from typing import Optional

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
