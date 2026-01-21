"""Code for transcription using the sherpa-onnx library."""

import logging
import shutil
import tarfile
import urllib.request
import wave
from pathlib import Path
from typing import Optional, Union

import numpy as np
import sherpa_onnx as so

from .const import Transcriber

_LOGGER = logging.getLogger(__name__)

_RATE = 16000
_URL_FORMAT = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{model_id}.tar.bz2"


class SherpaTranscriber(Transcriber):
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

        audio_array = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        )
        stream = self.recognizer.create_stream()
        stream.accept_waveform(_RATE, audio_array)
        self.recognizer.decode_stream(stream)
        return stream.result.text
