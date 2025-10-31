"""Code for transcription using the onnx-asr library."""

import asyncio
import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import Optional, Union
from unittest.mock import patch

import numpy as np
import onnx_asr
from huggingface_hub import snapshot_download
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

_RATE = 16000


class OnnxAsrModel:
    """Wrapper for onnx-asr model."""

    def __init__(
        self, model_id: str, cache_dir: Union[str, Path], local_files_only: bool
    ) -> None:
        """Initialize model."""

        # Force download to our cache dir
        def snapshot_download_with_cache(*args, **kwargs) -> str:
            kwargs["cache_dir"] = str(Path(cache_dir).resolve())
            kwargs["local_files_only"] = local_files_only

            return snapshot_download(*args, **kwargs)

        with patch("huggingface_hub.snapshot_download", snapshot_download_with_cache):
            self.onnx_model = onnx_asr.load_model(model_id)

    def transcribe(
        self, wav_path: Union[str, Path], language: Optional[str], *args, **kwargs
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

        audio_array = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        )

        recognize_kwargs = {}
        if language:
            recognize_kwargs["language"] = language

        text = self.onnx_model.recognize(  # type: ignore[call-overload]
            audio_array, sample_rate=_RATE, **recognize_kwargs
        )
        return text


class OnnxAsrEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        language: Optional[str],
        beam_size: int,
        model: OnnxAsrModel,
        model_lock: asyncio.Lock,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self._beam_size = beam_size
        self._language = language
        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None
        self._audio_converter = AudioChunkConverter(rate=_RATE, width=2, channels=1)

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = self._audio_converter.convert(AudioChunk.from_event(event))

            if self._wav_file is None:
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug(
                "Audio stopped. Transcribing with language=%s", self._language
            )
            assert self._wav_file is not None

            self._wav_file.close()
            self._wav_file = None

            async with self.model_lock:
                text = self.model.transcribe(
                    self._wav_path,
                    language=self._language,
                )

            _LOGGER.info(text)

            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            # Reset
            self._language = self._language

            return False

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self._language = transcribe.language
                _LOGGER.debug("Language set to %s", transcribe.language)
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True
