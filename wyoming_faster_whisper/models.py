"""Logic for model selection, loading, and transcription."""

import asyncio
import logging
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .const import SttLibrary, Transcriber, sense_voice_language
from .faster_whisper_handler import FasterWhisperTranscriber

_LOGGER = logging.getLogger(__name__)

TRANSCRIBER_KEY = Tuple[SttLibrary, str, bool]  # (library, model id, streaming)


class ModelLoader:
    """Load transcribers for models."""

    def __init__(
        self,
        preferred_stt_library: SttLibrary,
        preferred_language: Optional[str],
        download_dir: Union[str, Path],
        local_files_only: bool,
        model: Optional[str],
        compute_type: str,
        device: str,
        beam_size: int,
        cpu_threads: int,
        initial_prompt: Optional[str],
        vad_parameters: Optional[Dict[str, Any]],
        whisper_task: Optional[str] = None,
        sherpa_streaming: bool = False,
        vad_clip: bool = False,
        vad_clip_threshold: float = 0.5,
        vad_clip_pad_ms: int = 400,
    ) -> None:
        self.preferred_stt_library = preferred_stt_library
        self.preferred_language = preferred_language
        self.sherpa_streaming = sherpa_streaming

        self.download_dir = Path(download_dir)
        self.local_files_only = local_files_only

        # faster-whisper only
        self.model = model
        self.compute_type = compute_type
        self.device = device
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters
        self.whisper_task = whisper_task
        self.vad_clip = vad_clip
        self.vad_clip_threshold = vad_clip_threshold
        self.vad_clip_pad_ms = vad_clip_pad_ms

        self._transcriber: Dict[TRANSCRIBER_KEY, Transcriber] = {}
        self._transcriber_lock: Dict[TRANSCRIBER_KEY, asyncio.Lock] = defaultdict(
            asyncio.Lock
        )

    async def load_transcriber(self, language: Optional[str] = None) -> Transcriber:
        """Load or get transcriber from cache for a language."""
        language = language or self.preferred_language

        # Check dependencies
        try:
            from .sherpa_handler import SherpaTranscriber

            has_sherpa = True
            _LOGGER.debug("Sherpa is available")
        except ImportError:
            has_sherpa = False
            _LOGGER.debug("Sherpa is NOT available")

        try:
            from .transformers_whisper import TransformersTranscriber

            has_transformers = True
            _LOGGER.debug("Transformers library is available")
        except ImportError:
            has_transformers = False
            _LOGGER.debug("Transformers library is NOT available")

        try:
            from .onnx_asr_handler import OnnxAsrTranscriber

            has_onnx_asr = True
            _LOGGER.debug("Onnx-ASR is available")
        except ImportError:
            has_onnx_asr = False
            _LOGGER.debug("Onnx-ASR is NOT available")
        try:
            from .funasr_handler import FunASRTranscriber

            has_funasr = True
            _LOGGER.debug("FunASR is available")
        except ImportError:
            has_funasr = False
            _LOGGER.debug("FunASR is NOT available")

        # Select speech-to-text library
        stt_library = guess_stt_library(
            self.preferred_stt_library,
            self.model,
            language,
            has_transformers=has_transformers,
            has_sherpa=has_sherpa,
            has_onnx_asr=has_onnx_asr,
            has_funasr=has_funasr,
        )

        # Streaming is only supported by the sherpa backend.
        streaming = self.sherpa_streaming and (stt_library == SttLibrary.SHERPA)

        # Select model
        model = self.model
        if model is None:  # auto
            machine = platform.machine().lower()
            is_arm = ("arm" in machine) or ("aarch" in machine)
            model = guess_model(stt_library, language, is_arm, streaming=streaming)

        _LOGGER.debug(
            "Selected stt-library '%s' with model '%s' (streaming=%s)",
            stt_library.value,
            model,
            streaming,
        )

        # Load transcriber
        assert stt_library != SttLibrary.AUTO
        assert model

        key = (stt_library, model, streaming)

        async with self._transcriber_lock[key]:
            transcriber = self._transcriber.get(key)
            if transcriber is not None:
                return transcriber

            if stt_library == SttLibrary.SHERPA:
                if streaming:
                    from .sherpa_handler import SherpaStreamingTranscriber  # noqa: F811

                    transcriber = SherpaStreamingTranscriber(
                        model,
                        self.download_dir,
                        cpu_threads=self.cpu_threads,
                        beam_size=self.beam_size,
                    )
                else:
                    from .sherpa_handler import SherpaTranscriber  # noqa: F811

                    transcriber = SherpaTranscriber(
                        model, self.download_dir, cpu_threads=self.cpu_threads
                    )
            elif stt_library == SttLibrary.ONNX_ASR:
                from .onnx_asr_handler import OnnxAsrTranscriber  # noqa: F811

                transcriber = OnnxAsrTranscriber(
                    model,
                    cache_dir=self.download_dir,
                    local_files_only=self.local_files_only,
                )
            elif stt_library == SttLibrary.TRANSFORMERS:
                from .transformers_whisper import TransformersTranscriber  # noqa: F811

                transcriber = TransformersTranscriber(
                    model,
                    cache_dir=self.download_dir,
                    local_files_only=self.local_files_only,
                )
            elif stt_library == SttLibrary.FUNASR:
                from .funasr_handler import FunASRTranscriber  # noqa: F811

                transcriber = FunASRTranscriber(
                    model,
                    cache_dir=self.download_dir,
                    local_files_only=self.local_files_only,
                    device=self.device,
                )
            else:
                transcriber = FasterWhisperTranscriber(
                    model,
                    cache_dir=self.download_dir,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                    vad_parameters=self.vad_parameters,
                    task=self.whisper_task,
                )

            self._transcriber[key] = transcriber

        return transcriber

    async def transcribe(
        self, wav_path: Union[str, Path], language: Optional[str]
    ) -> str:
        """Transcribe WAV file using appropriate transcriber.

        Assume WAV file is 16Khz 16-bit mono PCM.
        """
        transcriber = await self.load_transcriber(language)
        text = await asyncio.to_thread(
            transcriber.transcribe,
            wav_path,
            language=language,
            beam_size=self.beam_size,
            initial_prompt=self.initial_prompt,
        )
        _LOGGER.debug("Transcribed audio: %s", text)

        return text


def guess_stt_library(
    preferred_stt_library: SttLibrary,
    model: Optional[str],
    language: Optional[str],
    *,
    has_transformers: bool,
    has_sherpa: bool,
    has_onnx_asr: bool,
    has_funasr: bool,
) -> SttLibrary:
    """Resolve which speech-to-text library to use.

    When the preferred library is AUTO and no model is forced, pick the best
    available specialized backend for the language; otherwise faster-whisper.
    A non-AUTO library falls back to faster-whisper when its dependency is
    missing.
    """
    if preferred_stt_library == SttLibrary.AUTO:
        if model is None:  # auto-select a per-language backend
            if (language == "ru") and has_onnx_asr:
                # Prefer GigaAM via onnx-asr
                return SttLibrary.ONNX_ASR

            if (language == "en") and has_sherpa:
                # Prefer Parakeet via sherpa for English. The v3 Parakeet model
                # claims to auto detect other languages, but it doesn't work.
                return SttLibrary.SHERPA

            if (
                sense_voice_language(language) in ("zh", "yue", "ja", "ko")
            ) and has_funasr:
                # Prefer SenseVoice via FunASR for Chinese, Cantonese, Japanese,
                # and Korean (incl. locale codes like "zh-CN").
                return SttLibrary.FUNASR

        # Default to faster-whisper
        return SttLibrary.FASTER_WHISPER

    # Explicit library: fall back to faster-whisper if its dependency is absent.
    available = {
        SttLibrary.TRANSFORMERS: has_transformers,
        SttLibrary.SHERPA: has_sherpa,
        SttLibrary.ONNX_ASR: has_onnx_asr,
        SttLibrary.FUNASR: has_funasr,
    }
    if not available.get(preferred_stt_library, True):
        _LOGGER.debug("Falling back to faster-whisper (missing dependencies)")
        return SttLibrary.FASTER_WHISPER

    return preferred_stt_library


def guess_model(
    stt_library: SttLibrary,
    language: Optional[str],
    is_arm: bool,
    streaming: bool = False,
) -> str:
    """Automatically guess STT model id."""
    if stt_library == SttLibrary.SHERPA:
        if streaming:
            # Best available streaming (OnlineRecognizer) model. The Kroko
            # streaming zipformers produce mixed-case, punctuated output with
            # much better accuracy than the older LibriSpeech models. They are
            # per-language, so warn for languages we don't have a default for.
            if language in (None, "en"):
                return "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"

            if language in ("de", "es", "fr"):
                return f"sherpa-onnx-streaming-zipformer-{language}-kroko-2025-08-06"

            _LOGGER.warning(
                "No default streaming sherpa model for language '%s'; pass "
                "--model to choose one. Falling back to English.",
                language,
            )
            return "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"

        if language == "en":
            return "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"

        # Non-English
        return "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"

    if stt_library == SttLibrary.TRANSFORMERS:
        if language == "en":
            if is_arm:
                return "openai/whisper-tiny.en"

            return "openai/whisper-base.en"

        # Non-English
        if is_arm:
            return "openai/whisper-tiny"

        return "openai/whisper-base"

    if stt_library == SttLibrary.ONNX_ASR:
        return "gigaam-v2-rnnt"

    if stt_library == SttLibrary.FUNASR:
        return "FunAudioLLM/SenseVoiceSmall"

    # faster-whisper
    if is_arm:
        return "rhasspy/faster-whisper-tiny-int8"

    return "rhasspy/faster-whisper-base-int8"
