"""Standalone Gemini Live audio example with Google Search.

Set GEMINI_API_KEY, then run: python voice_google_search.py
Ask a question that needs current web information.
Type q at the prompt to quit.
"""

import asyncio
import os
from typing import Any

import pyaudio
from google import genai
from google.genai import types

MODEL = "models/gemini-3.8-live"
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=104857,
        sliding_window=types.SlidingWindow(target_tokens=52428),
    ),
    tools=[types.Tool(google_search=types.GoogleSearch())],
)


class AudioLoop:
    def __init__(self) -> None:
        self.session: Any = None
        self.audio_in_queue: asyncio.Queue[bytes] | None = None
        self.out_queue: asyncio.Queue[dict[str, Any]] | None = None
        self.audio: Any = None
        self.input_stream: Any = None
        self.output_stream: Any = None

    async def send_text(self) -> None:
        while True:
            text = await asyncio.to_thread(input, "message (q to quit) > ")
            if text.lower() == "q":
                return
            if self.session is not None:
                await self.session.send(input=text or ".", end_of_turn=True)

    async def send_realtime(self) -> None:
        while self.out_queue is not None:
            message = await self.out_queue.get()
            if self.session is not None:
                await self.session.send(input=message)

    async def listen_audio(self) -> None:
        mic_info = self.audio.get_default_input_device_info()
        self.input_stream = await asyncio.to_thread(
            self.audio.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        try:
            while self.out_queue is not None:
                data = await asyncio.to_thread(
                    self.input_stream.read,
                    CHUNK_SIZE,
                    exception_on_overflow=False,
                )
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
        finally:
            self.input_stream.close()

    async def receive_audio(self) -> None:
        while self.session is not None:
            async for response in self.session.receive():
                if response.data and self.audio_in_queue is not None:
                    await self.audio_in_queue.put(response.data)
                if response.text:
                    print(response.text, end="", flush=True)
            if self.audio_in_queue is not None:
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()

    async def play_audio(self) -> None:
        self.output_stream = await asyncio.to_thread(
            self.audio.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        try:
            while self.audio_in_queue is not None:
                audio_chunk = await self.audio_in_queue.get()
                await asyncio.to_thread(self.output_stream.write, audio_chunk)
        finally:
            self.output_stream.close()

    async def run(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set the GEMINI_API_KEY environment variable.")

        client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=api_key,
        )
        self.audio = pyaudio.PyAudio()
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as task_group,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                text_task = task_group.create_task(self.send_text())
                task_group.create_task(self.send_realtime())
                task_group.create_task(self.listen_audio())
                task_group.create_task(self.receive_audio())
                task_group.create_task(self.play_audio())
                await text_task
                raise asyncio.CancelledError("User requested exit")
        except asyncio.CancelledError:
            pass
        finally:
            self.session = None
            self.audio_in_queue = None
            self.out_queue = None
            self.audio.terminate()
            await asyncio.to_thread(client.close)


if __name__ == "__main__":
    asyncio.run(AudioLoop().run())
