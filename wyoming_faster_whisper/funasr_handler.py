"""Code for transcription using the FunASR library."""

import os
import wave
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .const import Transcriber

_RATE = 16000

# Languages SenseVoice can be told to decode explicitly; otherwise auto-detect.
_SENSE_VOICE_LANGUAGES = {"auto", "zh", "en", "yue", "ja", "ko"}


class FunASRTranscriber(Transcriber):
    """Wrapper for a FunASR model (SenseVoice / Paraformer / Fun-ASR-Nano)."""

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        local_files_only: bool = False,
        device: str = "cpu",
    ) -> None:
        """Initialize model."""
        # FunASR (hub="hf") downloads via huggingface_hub; honor the cache dir.
        os.environ.setdefault("HF_HOME", str(Path(cache_dir).resolve()))

        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        self._postprocess = rich_transcription_postprocess
        self._is_sense_voice = "SenseVoice" in model_id
        self.model = AutoModel(
            model=model_id,
            hub="hf",
            device=device,
            disable_update=True,
        )

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """Returns transcription for WAV file.

        WAV file must be 16Khz 16-bit mono audio.
        """
        wav_file: wave.Wave_read = wave.open(str(wav_path), "rb")
        with wav_file:
            assert wav_file.getframerate() == _RATE, "Sample rate must be 16Khz"
            assert wav_file.getsampwidth() == 2, "Width must be 16-bit (2 bytes)"
            assert wav_file.getnchannels() == 1, "Audio must be mono"
            audio_bytes = wav_file.readframes(wav_file.getnframes())

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        gen_kwargs = {"input": audio, "cache": {}, "use_itn": True, "batch_size_s": 300}
        if self._is_sense_voice:
            lang = language if (language in _SENSE_VOICE_LANGUAGES) else "auto"
            gen_kwargs["language"] = lang

        result = self.model.generate(**gen_kwargs)
        text = result[0]["text"] if result else ""
        return self._postprocess(text).strip()
