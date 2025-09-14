#!/usr/bin/env python3
import argparse
import asyncio
import logging
import platform
import re
from functools import partial
from typing import Any

import faster_whisper
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import FasterWhisperEventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        default=os.environ.get("WYOMING_MODEL_NAME"),
        help="Name of faster-whisper model to use (or auto)",
    )
    parser.add_argument(
        "--uri",
        required=True,
        help="unix:// or tcp://",
        default=os.environ.get("WYOMING_URI"),
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        default=os.environ.get("WYOMING_DATA_DIR"),
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into (default: first data dir)",
        default=os.environ.get("WYOMING_DOWNLOAD_DIR"),
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("WYOMING_DEVICE", "cpu"),
        help="Device to use for inference (default: cpu)",
    )
    parser.add_argument(
        "--language",
        default=os.environ.get("WYOMING_LANGUAGE", "en"),
        help="Default language to set for transcription",
    )
    parser.add_argument(
        "--compute-type",
        default=os.environ.get("WYOMING_COMPUTE_TYPE", "default"),
        help="Compute type (float16, int8, etc.)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=os.environ.get("WYOMING_BEAM_SIZE", 5),
        help="Size of beam during decoding (0 for auto)",
    )
    parser.add_argument(
        "--initial-prompt",
        default=os.environ.get("WYOMING_INITIAL_PROMPT"),
        help="Optional text to provide as a prompt for the first window",
    )
    parser.add_argument(
        "--use-transformers",
        action="store_true",
        default=os.environ.get("WYOMING_USE_TRANSFORMERS"),
        help="Use HuggingFace transformers library (requires transformers extras)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=os.environ.get("WYOMING_LOCAL_FILES_ONLY"),
        help="Don't check HuggingFace hub for updates every time",
    )
    #
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.environ.get("WYOMING_DEBUG"),
        help="Log DEBUG messages"
    )
    parser.add_argument(
        "--log-format", default=os.environ.get("WYOMING_LOG_FORMAT", logging.BASIC_FORMAT), help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        default=os.environ.get("WYOMING_VERSION", __version__),
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

    # Automatic configuration for ARM
    machine = platform.machine().lower()
    is_arm = ("arm" in machine) or ("aarch" in machine)
    if args.model == "auto":
        args.model = "tiny-int8" if is_arm else "base-int8"
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

    if args.language == "auto":
        # Whisper does not understand "auto"
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
                        languages=faster_whisper.tokenizer._LANGUAGE_CODES,  # pylint: disable=protected-access
                        version=faster_whisper.__version__,
                    )
                ],
            )
        ],
    )

    # Load model
    _LOGGER.debug("Loading %s", args.model)
    whisper_model: Any = None

    if args.use_transformers:
        # Use HuggingFace transformers
        from .transformers_whisper import TransformersWhisperModel

        whisper_model = TransformersWhisperModel(
            args.model, args.download_dir, args.local_files_only
        )
    else:
        # Use faster-whisper
        whisper_model = faster_whisper.WhisperModel(
            args.model,
            download_root=args.download_dir,
            device=args.device,
            compute_type=args.compute_type,
        )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")
    model_lock = asyncio.Lock()

    if args.use_transformers:
        # Use HuggingFace transformers
        from .transformers_whisper import (
            TransformersWhisperEventHandler,
            TransformersWhisperModel,
        )

        assert isinstance(whisper_model, TransformersWhisperModel)

        # TODO: initial prompt
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
    else:
        # Use faster-whisper
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


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
