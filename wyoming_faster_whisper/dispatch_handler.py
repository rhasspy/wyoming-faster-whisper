"""Event handler for clients of the server."""

import asyncio
import logging
import os
import tempfile
import wave
from typing import Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .const import Transcriber
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

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            # Audio is saved to a WAV file for transcription later.
            # None of the underlying models support streaming.
            chunk = self._audio_converter.convert(AudioChunk.from_event(event))

            if self._wav_file is None:
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)

            if (self._transcriber is None) and (self._transcriber_future is None):
                # Load the transcriber in the background.
                # Hopefully it's ready by the time the audio stops.
                self._transcriber_future = asyncio.create_task(
                    self._loader.load_transcriber(self._language)
                )

            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped")

            # No audio was received before AudioStop — return empty transcript.
            # This happens when HA sends AudioStop without any AudioChunk
            # (e.g., VAD detected no speech, or the client disconnected early).
            if self._wav_file is None:
                _LOGGER.warning("AudioStop received with no audio data")
                await self.write_event(Transcript(text="").event())
                self._language = None
                self._transcriber = None
                self._transcriber_future = None
                return False

            if self._transcriber is None:
                # Get transcriber that was loading in the background
                if self._transcriber_future is None:
                    _LOGGER.warning("No transcriber available")
                    await self.write_event(Transcript(text="").event())
                    self._wav_file.close()
                    self._wav_file = None
                    self._language = None
                    return False
                self._transcriber = await self._transcriber_future

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

            # Reset
            self._language = None
            self._transcriber = None

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
