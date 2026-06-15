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

from .const import StreamingSession, Transcriber

_LOGGER = logging.getLogger(__name__)

_RATE = 16000
_URL_FORMAT = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{model_id}.tar.bz2"


def _ensure_model(model_id: str, cache_dir: Union[str, Path]) -> Path:
    """Download/extract a sherpa-onnx model if needed and return its directory."""
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

    return model_dir


def _find_model_file(model_dir: Path, prefix: str) -> str:
    """Find a model file by prefix, preferring the int8 quantized version.

    Streaming zipformer models use versioned file names (e.g.
    encoder-epoch-99-avg-1.int8.onnx), so we glob rather than hard-code.
    """
    int8_matches = sorted(model_dir.glob(f"{prefix}*.int8.onnx"))
    if int8_matches:
        return str(int8_matches[0])

    matches = sorted(p for p in model_dir.glob(f"{prefix}*.onnx"))
    if matches:
        return str(matches[0])

    raise FileNotFoundError(f"No '{prefix}*.onnx' model file found in {model_dir}")


def _bytes_to_samples(audio_bytes: bytes) -> np.ndarray:
    """Convert 16-bit mono PCM bytes to a float32 sample array."""
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0


class SherpaTranscriber(Transcriber):
    """Wrapper for sherpa-onnx model."""

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        cpu_threads: int = 4,
    ) -> None:
        """Initialize model."""
        model_dir = _ensure_model(model_id, cache_dir)

        # Load model
        self.recognizer = so.OfflineRecognizer.from_transducer(
            num_threads=cpu_threads,
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

        audio_array = _bytes_to_samples(audio_bytes)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(_RATE, audio_array)
        self.recognizer.decode_stream(stream)
        return stream.result.text


class SherpaStreamingTranscriber(Transcriber):
    """Wrapper for a streaming sherpa-onnx model (OnlineRecognizer).

    Use with natively-streaming models (e.g. streaming zipformer transducers).
    Offline models such as Parakeet TDT are NOT supported here.
    """

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        cpu_threads: int = 4,
    ) -> None:
        """Initialize model."""
        model_dir = _ensure_model(model_id, cache_dir)

        # Load model. Streaming zipformer file names are versioned, so locate
        # them by prefix instead of hard-coding.
        self.recognizer = so.OnlineRecognizer.from_transducer(
            encoder=_find_model_file(model_dir, "encoder"),
            decoder=_find_model_file(model_dir, "decoder"),
            joiner=_find_model_file(model_dir, "joiner"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=cpu_threads,
            provider="cpu",
        )

        # Prime model so that the first transcription will be fast
        stream = self.recognizer.create_stream()
        stream.accept_waveform(_RATE, np.zeros(shape=(128), dtype=np.float32))
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)

    @property
    def supports_streaming(self) -> bool:
        return True

    def start_stream(
        self,
        language: Optional[str] = None,
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> StreamingSession:
        return SherpaStreamingSession(self.recognizer)

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """Returns transcription for WAV file (batch fallback).

        WAV file must be 16Khz 16-bit mono audio.
        """
        wav_file: wave.Wave_read = wave.open(str(wav_path), "rb")
        with wav_file:
            assert wav_file.getframerate() == _RATE, "Sample rate must be 16Khz"
            assert wav_file.getsampwidth() == 2, "Width must be 16-bit (2 bytes)"
            assert wav_file.getnchannels() == 1, "Audio must be mono"
            audio_bytes = wav_file.readframes(wav_file.getnframes())

        session = self.start_stream(language, beam_size, initial_prompt)
        session.accept_chunk(audio_bytes)
        return session.finish()


class SherpaStreamingSession(StreamingSession):
    """A single in-progress streaming transcription for OnlineRecognizer."""

    def __init__(self, recognizer: "so.OnlineRecognizer") -> None:
        self.recognizer = recognizer
        self.stream = recognizer.create_stream()

    def accept_chunk(self, audio_bytes: bytes) -> None:
        if not audio_bytes:
            return

        self.stream.accept_waveform(_RATE, _bytes_to_samples(audio_bytes))
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

    def finish(self) -> str:
        # Signal end of audio and flush any remaining frames.
        self.stream.input_finished()
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        return self.recognizer.get_result(self.stream)
