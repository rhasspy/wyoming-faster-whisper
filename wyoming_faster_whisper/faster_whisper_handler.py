"""Event handler for clients of the server."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import faster_whisper

from .const import Transcriber


class FasterWhisperTranscriber(Transcriber):
    """Event handler for clients."""

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        device: str = "cpu",
        compute_type: str = "default",
        cpu_threads: int = 4,
        vad_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.vad_filter = vad_parameters is not None
        self.vad_parameters = vad_parameters

        self.model = faster_whisper.WhisperModel(
            model_id,
            download_root=str(cache_dir),
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        segments, _info = self.model.transcribe(
            str(wav_path),
            beam_size=beam_size,
            language=language,
            initial_prompt=initial_prompt,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters,
        )

        text = " ".join(segment.text for segment in segments)
        return text
