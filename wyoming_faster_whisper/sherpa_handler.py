"""Code for transcription using the sherpa-onnx library."""

import asyncio
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
import wave
from pathlib import Path
from typing import Optional, Union

import numpy as np
import sherpa_onnx as so
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

_RATE = 16000
_URL_FORMAT = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{model_id}.tar.bz2"


class SherpaModel:
    """Wrapper for sherpa-onnx model."""

    def __init__(self, model_id: str, cache_dir: Union[str, Path]) -> None:
        """Initialize model."""
        cache_dir = Path(cache_dir)
        model_dir = cache_dir / model_id
        _LOGGER.debug("Looking for sherpa model: %s", model_dir)

        if not model_dir.exists():
            url = _URL_FORMAT.format(model_id=model_id)
            _LOGGER.info("Downloading %s", url)
            cache_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Download/extract to cache dir.
                # We assume that the .tar.bz2 contains a directory named after
                # the model id.
                with urllib.request.urlopen(url) as response:
                    with tarfile.open(fileobj=response, mode="r|bz2") as tar:
                        for member in tar:
                            tar.extract(member, path=cache_dir)
            except Exception:
                # Delete directory so we'll download again next time
                shutil.rmtree(model_dir, ignore_errors=True)
                raise

        # Load model
        self.recognizer = so.OfflineRecognizer.from_transducer(
            encoder=f"{model_dir}/encoder.int8.onnx",
            decoder=f"{model_dir}/decoder.int8.onnx",
            joiner=f"{model_dir}/joiner.int8.onnx",
            tokens=f"{model_dir}/tokens.txt",
            provider="cpu",
            model_type="nemo_transducer",
        )

        # Prime model so that the first transcription will be fast
        stream = self.recognizer.create_stream()
        stream.accept_waveform(_RATE, np.zeros(shape=(128), dtype=np.float32))
        self.recognizer.decode_stream(stream)

    def transcribe(self, wav_path: Union[str, Path], *args, **kwargs) -> str:
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
        stream = self.recognizer.create_stream()
        stream.accept_waveform(_RATE, audio_array)
        self.recognizer.decode_stream(stream)
        return stream.result.text


class SherpaEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        language: Optional[str],
        beam_size: int,
        model: SherpaModel,
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
                    beam_size=self._beam_size,
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
