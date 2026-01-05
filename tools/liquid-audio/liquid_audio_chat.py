#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pyaudio",
#     "soundfile",
#     "openai",
#     "prompt_toolkit",
# ]
# ///
"""Interactive CLI chat tool for LFM2.5-Audio server."""

import argparse
import base64
import io
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from queue import Queue

import numpy as np
import pyaudio
import soundfile as sf
from openai import OpenAI
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

# Suppress ALSA/JACK warnings during PyAudio init
import os
import contextlib

@contextlib.contextmanager
def suppress_stderr():
    """Temporarily redirect stderr to /dev/null."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)


class AudioPlayer:
    """Streams audio samples to speakers via PyAudio (non-blocking)."""

    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.all_samples = []
        self.pyaudio = None
        self.stream = None
        self.queue = Queue()
        self.thread = None
        self.running = False

    def _playback_thread(self):
        """Background thread that writes audio to the stream."""
        while self.running or not self.queue.empty():
            try:
                pcm_data = self.queue.get(timeout=0.1)
                if self.stream:
                    self.stream.write(pcm_data)
            except:
                pass

    def start(self):
        """Start the audio stream."""
        self.all_samples = []
        self.running = True
        with suppress_stderr():
            self.pyaudio = pyaudio.PyAudio()
            self.stream = self.pyaudio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                output=True,
            )
        self.thread = threading.Thread(target=self._playback_thread, daemon=True)
        self.thread.start()

    def add_samples(self, samples):
        """Add samples to playback queue (non-blocking)."""
        self.all_samples.extend(samples)
        pcm_data = np.array(samples, dtype=np.float32).tobytes()
        self.queue.put(pcm_data)

    def stop(self, output_file="output.wav"):
        """Stop the audio stream."""
        self.running = False
        if self.thread:
            self.thread.join()
            self.thread = None
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.pyaudio:
            self.pyaudio.terminate()
            self.pyaudio = None


class AudioRecorder:
    """Records audio from microphone using PyAudio."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recording = False
        self.samples = []
        self.available = self._check_available()

    def _check_available(self):
        """Check if audio input is available."""
        try:
            with suppress_stderr():
                p = pyaudio.PyAudio()
                has_input = p.get_default_input_device_info() is not None
                p.terminate()
            return has_input
        except Exception:
            return False

    def record(self, duration=None):
        """Record audio. Press Enter to stop if duration is None."""
        if not self.available:
            print("[No microphone available. Use /wav to load audio files.]")
            return None

        self.samples = []
        self.recording = True

        print("Recording... (Press Enter to stop)")

        # Start recording in background
        stop_event = threading.Event()

        def record_audio():
            try:
                with suppress_stderr():
                    p = pyaudio.PyAudio()
                    stream = p.open(
                        format=pyaudio.paFloat32,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=1024,
                    )
                while not stop_event.is_set():
                    data = stream.read(1024, exception_on_overflow=False)
                    samples = np.frombuffer(data, dtype=np.float32)
                    self.samples.extend(samples.tolist())
                stream.stop_stream()
                stream.close()
                p.terminate()
            except Exception as e:
                print(f"[Recording error: {e}]")

        record_thread = threading.Thread(target=record_audio)
        record_thread.start()

        # Wait for Enter key
        input()
        stop_event.set()
        record_thread.join()

        self.recording = False
        if self.samples:
            print(f"Recorded {len(self.samples) / self.sample_rate:.2f}s of audio")

        return self.samples if self.samples else None

    def to_wav_bytes(self):
        """Convert recorded samples to WAV bytes."""
        if not self.samples:
            return None
        buffer = io.BytesIO()
        sf.write(buffer, np.array(self.samples), self.sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()


SYSTEM_PROMPTS = {
    "asr": "Perform ASR.",
    "tts": "Perform TTS.",
    "interleaved": "Respond with interleaved text and audio.",
}


def create_text_message(text):
    """Create a text user message."""
    return {"role": "user", "content": text}


def create_audio_message(wav_data):
    """Create an audio user message."""
    encoded = base64.b64encode(wav_data).decode("utf-8")
    return {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": encoded, "format": "wav"},
            }
        ],
    }


def create_stream_single_shot(client, mode, text=None, wav_data=None, max_tokens=512):
    """Create a single-shot request for ASR/TTS (always resets context)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[mode]}]

    if mode == "asr" and wav_data:
        messages.append(create_audio_message(wav_data))
    elif mode == "tts" and text:
        messages.append(create_text_message(text))

    return client.chat.completions.create(
        model="",
        messages=messages,
        stream=True,
        max_tokens=max_tokens,
        extra_body={"reset_context": True},
    )


def create_stream_chat(client, messages, max_tokens=512, reset_context=False):
    """Create a chat request for interleaved mode (maintains context)."""
    return client.chat.completions.create(
        model="",
        messages=messages,
        stream=True,
        max_tokens=max_tokens,
        extra_body={"reset_context": reset_context},
    )


def process_stream(stream, audio_player=None):
    """Process streaming response, playing audio and printing text."""
    t0 = time.time()
    ttft = None
    text_chunks = []
    audio_chunks = []
    total_samples = 0
    completed = False

    for chunk in stream:
        if chunk.choices[0].finish_reason == "stop":
            completed = True
            break

        delta = chunk.choices[0].delta

        # Handle text
        if text := delta.content:
            if ttft is None:
                ttft = time.time() - t0
            text_chunks.append((time.time(), text))
            print(text, end="", flush=True)

        # Handle audio
        if hasattr(delta, "audio_chunk") and delta.audio_chunk:
            if ttft is None:
                ttft = time.time() - t0
            chunk_data = delta.audio_chunk["data"]
            pcm_bytes = base64.b64decode(chunk_data)
            samples = struct.unpack(f"<{len(pcm_bytes) // 4}f", pcm_bytes)
            audio_chunks.append((time.time(), samples))
            total_samples += len(samples)

            # Print note symbol for audio progress
            print("♪", end="", flush=True)

            if audio_player:
                audio_player.add_samples(samples)

    if text_chunks or audio_chunks:
        print()  # Newline after output

    if not completed:
        print("[Warning: Server disconnected before completion]")

    # Calculate and display stats (single line)
    total_time = time.time() - t0
    full_text = "".join(t for _, t in text_chunks)

    stats = []
    if ttft is not None:
        stats.append(f"ttft {ttft:.3f}s")

    if text_chunks and len(text_chunks) > 1:
        text_duration = text_chunks[-1][0] - text_chunks[0][0]
        if text_duration > 0:
            stats.append(f"text {len(text_chunks)} tok @ {len(text_chunks)/text_duration:.1f} tok/s")

    if audio_chunks:
        # Calculate from ttft to last chunk for accurate throughput
        first_audio_time = audio_chunks[0][0]
        last_audio_time = audio_chunks[-1][0]
        audio_duration = last_audio_time - first_audio_time
        audio_secs = total_samples / 24000
        stats.append(f"audio {audio_secs:.1f}s @ {total_samples/audio_duration:.0f} samples/s")

    stats.append(f"total {total_time:.3f}s")
    print(f"\n[{' | '.join(stats)}]")

    return full_text, total_samples


def print_help():
    """Print help information."""
    print(
        """
Commands:
  /mode <asr|tts|interleaved>  - Switch mode
  /reset                       - Reset context (interleaved mode only)
  /record                      - Record and transcribe/process audio
  /wav <path>                  - Load and transcribe/process audio file
  /help                        - Show this help
  /quit or /exit               - Exit the program

Modes:
  ASR (single-shot):
    - Use /record or /wav to transcribe audio
    - Each request is independent

  TTS (single-shot):
    - Type text to synthesize audio
    - Each request is independent

  Interleaved (chat):
    - Type text or use /record or /wav
    - Context is maintained across requests
    - Use /reset to start fresh
"""
    )


def main():
    parser = argparse.ArgumentParser(description="Interactive LFM2.5-Audio chat client")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8080/v1",
        help="Server base URL",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["asr", "tts", "interleaved"],
        default="interleaved",
        help="Initial mode",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--no-audio-playback",
        action="store_true",
        help="Disable audio playback (save to file instead)",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="dummy")
    recorder = AudioRecorder()
    mode = args.mode
    wav_data = None
    enable_playback = not args.no_audio_playback

    # Check audio capabilities
    audio_input_ok = recorder.available

    # Track if first message in interleaved mode (need to send system prompt)
    is_first_message = True

    # Command history for prompt_toolkit
    cmd_history = InMemoryHistory()

    print("=" * 50)
    print("LFM2.5-Audio Interactive Chat")
    print("=" * 50)
    print(f"Server: {args.base_url}")
    print("Audio output: pyaudio")
    print(f"Audio input:  {'microphone' if audio_input_ok else 'file only (/wav)'}")
    print("Type /help for commands")
    print("=" * 50)
    print(f"Mode: {mode}" + (" (single-shot)" if mode in ("asr", "tts") else " (chat)"))

    while True:
        try:
            # Show prompt with mode indicator
            mode_indicator = {"asr": "[ASR]", "tts": "[TTS]", "interleaved": "[INT]"}
            audio_indicator = " [audio]" if wav_data else ""

            prompt_str = f"{mode_indicator[mode]}{audio_indicator}> "
            user_input = prompt(
                prompt_str,
                history=cmd_history,
                auto_suggest=AutoSuggestFromHistory(),
            ).strip()

            if not user_input:
                if mode == "asr" and wav_data:
                    # In ASR mode with audio, pressing Enter transcribes
                    pass
                else:
                    continue

            # Handle commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else None

                if cmd in ("/quit", "/exit"):
                    print("Goodbye!")
                    break

                elif cmd == "/help":
                    print_help()
                    continue

                elif cmd == "/mode":
                    if arg in ("asr", "tts", "interleaved"):
                        if arg != mode:
                            mode = arg
                            is_first_message = True
                            print(f"Mode: {mode}" + (" (single-shot)" if mode in ("asr", "tts") else " (chat)"))
                        else:
                            print(f"Already in {mode} mode")
                    else:
                        print("Usage: /mode <asr|tts|interleaved>")
                    continue

                elif cmd == "/reset":
                    if mode != "interleaved":
                        print("Reset only available in interleaved mode")
                        continue
                    is_first_message = True
                    print("Context reset")
                    continue

                elif cmd == "/record":
                    if mode == "tts":
                        print("Recording not available in TTS mode")
                        continue
                    samples = recorder.record()
                    if samples:
                        wav_data = recorder.to_wav_bytes()
                        # Start inference immediately
                        user_input = ""
                    else:
                        continue

                elif cmd == "/wav":
                    if mode == "tts":
                        print("Audio input not available in TTS mode")
                        continue
                    if arg:
                        try:
                            with open(arg, "rb") as f:
                                wav_data = f.read()
                            # Start inference immediately
                            user_input = ""
                        except Exception as e:
                            print(f"Error loading file: {e}")
                            continue
                    else:
                        print("Usage: /wav <path>")
                        continue

                else:
                    print(f"Unknown command: {cmd}")
                    continue

            # Prepare request based on mode
            text_input = user_input if user_input and not user_input.startswith("/") else None

            if mode == "asr":
                if not wav_data:
                    print("ASR mode requires audio. Use /record or /wav first.")
                    continue
                text_input = None  # ASR ignores text input
            elif mode == "tts":
                if not text_input:
                    print("TTS mode requires text input.")
                    continue
                wav_data = None  # TTS ignores audio input

            # Create audio player if needed
            audio_player = None
            if enable_playback:
                audio_player = AudioPlayer()
                audio_player.start()

            try:
                print()  # Blank line before response

                if mode in ("asr", "tts"):
                    # Single-shot mode: system + one message, always reset
                    stream = create_stream_single_shot(
                        client, mode, text=text_input, wav_data=wav_data, max_tokens=args.max_tokens
                    )
                else:
                    # Interleaved chat mode: only send new messages
                    messages = []

                    # First message needs system prompt and reset
                    if is_first_message:
                        messages.append({"role": "system", "content": SYSTEM_PROMPTS["interleaved"]})

                    # Add user message(s)
                    if text_input:
                        messages.append(create_text_message(text_input))
                    if wav_data:
                        messages.append(create_audio_message(wav_data))

                    stream = create_stream_chat(
                        client, messages, max_tokens=args.max_tokens, reset_context=is_first_message
                    )
                    is_first_message = False

                response_text, _ = process_stream(stream, audio_player)

            except Exception as e:
                print(f"Error: {e}")

            finally:
                if audio_player:
                    audio_player.stop()

            # Clear audio after use
            if wav_data:
                wav_data = None

        except KeyboardInterrupt:
            print("\nUse /quit to exit")
        except EOFError:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
