#!/usr/bin/env python3
import argparse
import asyncio
import logging
import platform
import re
from functools import partial
from typing import Any, Optional

import faster_whisper
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer, AsyncTcpServer

from . import __version__
from .const import AUTO_LANGUAGE, AUTO_MODEL, PARAKEET_LANGUAGES, SttLibrary

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    #
    parser.add_argument(
        "--zeroconf",
        nargs="?",
        const="faster-whisper",
        help="Enable discovery over zeroconf with optional name (default: faster-whisper)",
    )
    #
    parser.add_argument(
        "--model", default=AUTO_MODEL, help=f"Name of model to use (or {AUTO_MODEL})"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into (default: first data dir)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for inference (default: cpu)",
    )
    parser.add_argument(
        "--language",
        default=AUTO_LANGUAGE,
        help=f"Default language to set for transcription (default: {AUTO_LANGUAGE})",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help="Compute type (float16, int8, etc.)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=0,
        help="Size of beam during decoding (0 for auto)",
    )
    parser.add_argument(
        "--cpu-threads",
        default=4,
        type=int,
        help="Number of CPU threads to use for inference (default: 4, faster-whisper ony)",
    )
    parser.add_argument(
        "--initial-prompt",
        help="Optional text to provide as a prompt for the first window (faster-whisper only)",
    )
    parser.add_argument(
        "--stt-library",
        choices=[lib.value for lib in SttLibrary],
        default=SttLibrary.AUTO,
        help="Set library to use for speech-to-text (may require extra dependencies)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Don't check HuggingFace hub for updates every time",
    )
    #
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    if not args.download_dir:
        # Download to first data dir by default
        args.download_dir = args.data_dir[0]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    # Automatic configuration
    stt_library = SttLibrary(args.stt_library)
    if stt_library == SttLibrary.AUTO:
        if args.model == AUTO_MODEL:
            if args.language in ("en", AUTO_LANGUAGE):
                # Prefer parakeet
                try:
                    from .sherpa_handler import SherpaModel

                    stt_library = SttLibrary.SHERPA
                except ImportError:
                    stt_library = SttLibrary.FASTER_WHISPER
            elif args.language == "ru":
                # Prefer GigaAM via onnx-asr
                try:
                    from .sherpa_handler import SherpaModel

                    stt_library = SttLibrary.ONNX_ASR
                except ImportError:
                    stt_library = SttLibrary.FASTER_WHISPER
        else:
            # Default to faster-whisper if model is provided
            stt_library = SttLibrary.FASTER_WHISPER

        _LOGGER.debug("Speech-to-text library automatically selected: %s", stt_library)

    machine = platform.machine().lower()
    is_arm = ("arm" in machine) or ("aarch" in machine)
    if args.model == AUTO_MODEL:
        args.model = guess_model(stt_library, args.language, is_arm)
        _LOGGER.debug("Model automatically selected: %s", args.model)

    if args.beam_size <= 0:
        args.beam_size = 1 if is_arm else 5
        _LOGGER.debug("Beam size automatically selected: %s", args.beam_size)

    # Resolve model name
    model_name = args.model
    match = re.match(r"^(tiny|base|small|medium)[.-]int8$", args.model)
    if match:
        # Original models re-uploaded to huggingface
        model_size = match.group(1)
        model_name = f"{model_size}-int8"
        args.model = f"rhasspy/faster-whisper-{model_name}"

    if args.language == AUTO_LANGUAGE:
        # Whisper does not understand auto
        args.language = None

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="faster-whisper",
                description="Faster Whisper transcription with CTranslate2",
                attribution=Attribution(
                    name="Guillaume Klein",
                    url="https://github.com/guillaumekln/faster-whisper/",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        attribution=Attribution(
                            name="Systran",
                            url="https://huggingface.co/Systran",
                        ),
                        installed=True,
                        languages=sorted(
                            list(
                                # pylint: disable=protected-access
                                set(faster_whisper.tokenizer._LANGUAGE_CODES).union(
                                    PARAKEET_LANGUAGES
                                )
                            )
                        ),
                        version=faster_whisper.__version__,
                    )
                ],
            )
        ],
    )

    # Load model
    _LOGGER.debug("Loading %s", args.model)
    whisper_model: Any = None

    if stt_library == SttLibrary.SHERPA:
        # Use Sherpa ONNX with nemo
        from .sherpa_handler import SherpaModel  # noqa: F811

        whisper_model = SherpaModel(args.model, args.download_dir)
    elif stt_library == SttLibrary.TRANSFORMERS:
        # Use HuggingFace transformers
        from .transformers_whisper import TransformersWhisperModel

        whisper_model = TransformersWhisperModel(
            args.model, args.download_dir, args.local_files_only
        )
    elif stt_library == SttLibrary.ONNX_ASR:
        # Use onnx-asr
        from .onnx_asr_handler import OnnxAsrModel

        whisper_model = OnnxAsrModel(
            args.model, args.download_dir, args.local_files_only
        )
    else:
        # Use faster-whisper
        whisper_model = faster_whisper.WhisperModel(
            args.model,
            download_root=args.download_dir,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=args.cpu_threads,
        )

    server = AsyncServer.from_uri(args.uri)

    if args.zeroconf:
        if not isinstance(server, AsyncTcpServer):
            raise ValueError("Zeroconf requires tcp:// uri")

        from wyoming.zeroconf import HomeAssistantZeroconf

        tcp_server: AsyncTcpServer = server
        hass_zeroconf = HomeAssistantZeroconf(
            name=args.zeroconf, port=tcp_server.port, host=tcp_server.host
        )
        await hass_zeroconf.register_server()
        _LOGGER.debug("Zeroconf discovery enabled")

    _LOGGER.info("Ready")
    model_lock = asyncio.Lock()

    if stt_library == SttLibrary.SHERPA:
        from .sherpa_handler import SherpaEventHandler

        await server.run(
            partial(
                SherpaEventHandler,
                wyoming_info,
                args.language,
                args.beam_size,
                whisper_model,
                model_lock,
            )
        )
    elif stt_library == SttLibrary.TRANSFORMERS:
        # Use HuggingFace transformers
        from .transformers_whisper import (
            TransformersWhisperEventHandler,
            TransformersWhisperModel,
        )

        assert isinstance(whisper_model, TransformersWhisperModel)

        await server.run(
            partial(
                TransformersWhisperEventHandler,
                wyoming_info,
                args.language,
                args.beam_size,
                whisper_model,
                model_lock,
            )
        )
    elif stt_library == SttLibrary.ONNX_ASR:
        # Use onnx-asr
        from .onnx_asr_handler import OnnxAsrEventHandler, OnnxAsrModel

        assert isinstance(whisper_model, OnnxAsrModel)

        await server.run(
            partial(
                OnnxAsrEventHandler,
                wyoming_info,
                args.language,
                args.beam_size,
                whisper_model,
                model_lock,
            )
        )
    else:
        # faster-whisper
        from .faster_whisper_handler import FasterWhisperEventHandler

        assert isinstance(whisper_model, faster_whisper.WhisperModel)
        await server.run(
            partial(
                FasterWhisperEventHandler,
                wyoming_info,
                args,
                whisper_model,
                model_lock,
                initial_prompt=args.initial_prompt,
            )
        )


# -----------------------------------------------------------------------------


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
        return "tiny-int8"

    return "base-int8"


# -----------------------------------------------------------------------------


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
