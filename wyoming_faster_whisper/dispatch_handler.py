"""Event handler for clients of the server."""

import asyncio
import logging
import os
import tempfile
import wave
from typing import List, Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .const import StreamingSession, Transcriber
from .models import ModelLoader

_LOGGER = logging.getLogger(__name__)


class DispatchEventHandler(AsyncEventHandler):
    """Dispatches to appropriate transcriber."""

    def __init__(
        self,
        wyoming_info: Info,
        loader: ModelLoader,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info_event = wyoming_info.event()

        self._loader = loader
        self._transcriber: Optional[Transcriber] = None
        self._transcriber_future: Optional[asyncio.Future] = None
        self._language: Optional[str] = None

        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None

        self._audio_converter = AudioChunkConverter(rate=16000, width=2, channels=1)

        # Streaming state.
        # _is_streaming is None until the transcriber is loaded and the path
        # (streaming vs batch) is decided. Audio that arrives before then is
        # buffered in _pending_audio.
        self._got_audio = False
        self._is_streaming: Optional[bool] = None
        self._session: Optional[StreamingSession] = None
        self._pending_audio: List[bytes] = []

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = self._audio_converter.convert(AudioChunk.from_event(event))
            self._got_audio = True

            if (self._transcriber is None) and (self._transcriber_future is None):
                # Load the transcriber in the background.
                # Hopefully it's ready by the time the audio stops.
                self._transcriber_future = asyncio.create_task(
                    self._loader.load_transcriber(self._language)
                )

            # Promote the background-loaded transcriber without blocking.
            self._resolve_transcriber()

            # Decide between streaming and batch once the transcriber is ready.
            if (self._is_streaming is None) and (self._transcriber is not None):
                await self._commit_path()

            if self._is_streaming:
                assert self._session is not None
                await asyncio.to_thread(self._session.accept_chunk, chunk.audio)
            elif self._is_streaming is False:
                self._ensure_wav_file()
                assert self._wav_file is not None
                self._wav_file.writeframes(chunk.audio)
            else:
                # Transcriber not ready yet: buffer until the path is known.
                self._pending_audio.append(chunk.audio)

            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped")

            # No audio was received before AudioStop — return empty transcript.
            # This happens when HA sends AudioStop without any AudioChunk
            # (e.g., VAD detected no speech, or the client disconnected early).
            if not self._got_audio:
                _LOGGER.warning("AudioStop received with no audio data")
                await self.write_event(Transcript(text="").event())
                self._reset()
                return False

            # Get the transcriber that was loading in the background.
            if self._transcriber is None:
                if self._transcriber_future is None:
                    _LOGGER.warning("No transcriber available")
                    await self.write_event(Transcript(text="").event())
                    self._reset()
                    return False
                self._transcriber = await self._transcriber_future

            # If audio arrived before the transcriber was ready, the path may
            # still be undecided — commit it now using the buffered audio.
            if self._is_streaming is None:
                await self._commit_path()

            if self._is_streaming:
                assert self._session is not None
                text = await asyncio.to_thread(self._session.finish)
            else:
                assert self._wav_file is not None
                self._wav_file.close()
                self._wav_file = None

                # Do transcription in a separate thread
                text = await asyncio.to_thread(
                    self._transcriber.transcribe,
                    self._wav_path,
                    self._language,
                    beam_size=self._loader.beam_size,
                    initial_prompt=self._loader.initial_prompt,
                )

            _LOGGER.info(text)

            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            self._reset()

            return False

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self._language = transcribe.language or self._loader.preferred_language
            _LOGGER.debug("Language set to %s", self._language)

            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True

    def _resolve_transcriber(self) -> None:
        """Promote the background-loaded transcriber if it's ready (no blocking)."""
        if self._transcriber is not None:
            return

        future = self._transcriber_future
        if (future is not None) and future.done() and (future.exception() is None):
            self._transcriber = future.result()

    async def _commit_path(self) -> None:
        """Decide streaming vs batch and flush any buffered audio accordingly."""
        assert self._transcriber is not None

        self._is_streaming = self._transcriber.supports_streaming
        if self._is_streaming:
            self._session = self._transcriber.start_stream(
                self._language,
                beam_size=self._loader.beam_size,
                initial_prompt=self._loader.initial_prompt,
            )
            if self._pending_audio:
                # Replay audio buffered before the transcriber was ready.
                replay = b"".join(self._pending_audio)
                self._pending_audio.clear()
                await asyncio.to_thread(self._session.accept_chunk, replay)
        else:
            # Batch path: flush buffered audio to the WAV file.
            self._ensure_wav_file()
            assert self._wav_file is not None
            for buffered in self._pending_audio:
                self._wav_file.writeframes(buffered)
            self._pending_audio.clear()

    def _ensure_wav_file(self) -> None:
        """Open the temp WAV file for batch transcription if not already open."""
        if self._wav_file is None:
            self._wav_file = wave.open(self._wav_path, "wb")
            # Audio is normalized to this format by the converter.
            self._wav_file.setframerate(16000)
            self._wav_file.setsampwidth(2)
            self._wav_file.setnchannels(1)

    def _reset(self) -> None:
        """Reset per-request state."""
        if self._wav_file is not None:
            self._wav_file.close()
            self._wav_file = None

        self._language = None
        self._transcriber = None
        self._transcriber_future = None
        self._got_audio = False
        self._is_streaming = None
        self._session = None
        self._pending_audio = []
