"""Event handler for clients of the server."""

import logging
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Union

import faster_whisper
import numpy as np

from .const import Transcriber
from .vad import clip_to_speech

_LOGGER = logging.getLogger(__name__)


class FasterWhisperTranscriber(Transcriber):
    """Event handler for clients."""

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        device: str = "cpu",
        compute_type: str = "default",
        cpu_threads: int = 4,
        vad_parameters: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None,
        vad_clip: bool = True,
        vad_clip_threshold: float = 0.5,
        vad_clip_pad_ms: int = 400,
    ) -> None:
        self.vad_filter = vad_parameters is not None
        self.vad_parameters = vad_parameters
        self.task = task

        # pysilero-vad pre-clipping (trims silence before Whisper runs).
        self.vad_clip = vad_clip
        self.vad_clip_threshold = vad_clip_threshold
        self.vad_clip_pad_ms = vad_clip_pad_ms

        self.model = faster_whisper.WhisperModel(
            model_id,
            download_root=str(cache_dir),
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:

        kwargs = {
            "beam_size": beam_size,
            "language": language,
            "initial_prompt": initial_prompt,
            "vad_filter": self.vad_filter,
            "vad_parameters": self.vad_parameters,
        }
        if self.task:
            kwargs["task"] = self.task

        # Pass either the WAV path or a pre-clipped float32 array to Whisper.
        audio_input: Union[str, np.ndarray] = str(wav_path)
        if self.vad_clip:
            clipped = self._clip_wav(wav_path)
            if clipped is not None:
                audio_input = clipped

        segments, _info = self.model.transcribe(audio_input, **kwargs)
        text = " ".join(segment.text for segment in segments)
        return text

    def _clip_wav(self, wav_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Clip a 16 kHz 16-bit mono WAV to its speech region with pysilero-vad.

        Returns a float32 array for faster-whisper, or None to fall back to the
        full audio (unreadable WAV or no speech detected).
        """
        try:
            with wave.open(str(wav_path), "rb") as wav_file:
                audio = wav_file.readframes(wav_file.getnframes())
        except (wave.Error, OSError) as err:
            _LOGGER.warning("VAD clip: could not read %s: %s", wav_path, err)
            return None

        clipped = clip_to_speech(
            audio,
            threshold=self.vad_clip_threshold,
            pad_ms=self.vad_clip_pad_ms,
        )
        if not clipped:
            _LOGGER.debug("VAD clip: no speech detected, using full audio")
            return None

        _LOGGER.debug(
            "VAD clip: %d ms -> %d ms",
            len(audio) // 2 * 1000 // 16000,
            len(clipped) // 2 * 1000 // 16000,
        )
        return np.frombuffer(clipped, dtype=np.int16).astype(np.float32) / 32768.0
