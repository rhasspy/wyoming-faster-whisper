"""Logic for model selection, loading, and transcription."""

import asyncio
import logging
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .const import SttLibrary, Transcriber
from .faster_whisper_handler import FasterWhisperTranscriber

_LOGGER = logging.getLogger(__name__)

TRANSCRIBER_KEY = Tuple[SttLibrary, str]  # model id


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
    ) -> None:
        self.preferred_stt_library = preferred_stt_library
        self.preferred_language = preferred_language

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

        self._transcriber: Dict[TRANSCRIBER_KEY, Transcriber] = {}
        self._transcriber_lock: Dict[TRANSCRIBER_KEY, asyncio.Lock] = defaultdict(
            asyncio.Lock
        )

    async def load_transcriber(self, language: Optional[str] = None) -> Transcriber:
        """Load or get transcriber from cache for a language."""
        language = language or self.preferred_language
        stt_library = self.preferred_stt_library

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

        # Select speech-to-text library
        if stt_library == SttLibrary.AUTO:
            # Default to faster-whisper
            stt_library = SttLibrary.FASTER_WHISPER

            if self.model is None:  # auto
                if (language == "ru") and has_onnx_asr:
                    # Prefer GigaAM via onnx-asr
                    stt_library = SttLibrary.ONNX_ASR
                elif (language == "en") and has_sherpa:
                    # Prefer Parakeet via sherpa for English.
                    # The v3 Parakeet model claims to auto detect other
                    # languages, but it doesn't work.
                    stt_library = SttLibrary.SHERPA
        elif (
            ((stt_library == SttLibrary.TRANSFORMERS) and (not has_transformers))
            or ((stt_library == SttLibrary.SHERPA) and (not has_sherpa))
            or ((stt_library == SttLibrary.ONNX_ASR) and (not has_onnx_asr))
        ):
            # Fall back to faster-whisper
            stt_library = SttLibrary.FASTER_WHISPER
            _LOGGER.debug("Falling back to faster-whisper (missing dependencies)")

        # Select model
        model = self.model
        if model is None:  # auto
            machine = platform.machine().lower()
            is_arm = ("arm" in machine) or ("aarch" in machine)
            model = guess_model(stt_library, language, is_arm)

        _LOGGER.debug(
            "Selected stt-library '%s' with model '%s'", stt_library.value, model
        )

        # Load transcriber
        assert stt_library != SttLibrary.AUTO
        assert model

        key = (stt_library, model)

        async with self._transcriber_lock[key]:
            transcriber = self._transcriber.get(key)
            if transcriber is not None:
                return transcriber

            if stt_library == SttLibrary.SHERPA:
                from .sherpa_handler import SherpaTranscriber  # noqa: F811

                transcriber = SherpaTranscriber(model, self.download_dir)
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
            else:
                transcriber = FasterWhisperTranscriber(
                    model,
                    cache_dir=self.download_dir,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                    vad_parameters=self.vad_parameters,
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


def guess_model(stt_library: SttLibrary, language: Optional[str], is_arm: bool) -> str:
    """Automatically guess STT model id."""
    if stt_library == SttLibrary.SHERPA:
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

    # faster-whisper
    if is_arm:
        return "rhasspy/faster-whisper-tiny-int8"

    return "rhasspy/faster-whisper-base-int8"
