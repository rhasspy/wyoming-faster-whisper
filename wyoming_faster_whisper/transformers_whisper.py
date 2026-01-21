"""Code for Whisper transcription using HuggingFace's transformers library."""

import wave
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .const import Transcriber

_RATE = 16000


class TransformersTranscriber(Transcriber):
    """Wrapper for HuggingFace transformers Whisper model."""

    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
    ) -> None:
        """Initialize Whisper model."""
        self.processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.model.eval()

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

        audio_tensor = (
            torch.frombuffer(audio_bytes, dtype=torch.int16).float() / 32768.0
        )

        inputs = self.processor(audio_tensor, sampling_rate=_RATE, return_tensors="pt")
        generate_args = {**inputs, "num_beams": beam_size}

        if language:
            generate_args["forced_decoder_ids"] = self.processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )

        with torch.no_grad():
            # Ignore warning about attention_mask because we're only doing a single utterance.
            generated_ids = self.model.generate(**generate_args)
            transcription = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        return transcription
