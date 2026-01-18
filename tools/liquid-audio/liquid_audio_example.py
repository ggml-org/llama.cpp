import argparse
import base64
import struct
import time

import soundfile as sf
from openai import OpenAI


def interleaved(client, text=None, wav_data=None):
    messages = [
        {"role": "system", "content": "Respond with interleaved text and audio."},
    ]

    if text:
        messages.append({"role": "user", "content": text})

    if wav_data:
        encoded_wav_data = base64.b64encode(wav_data).decode("utf-8")

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded_wav_data, "format": "wav"},
                    }
                ],
            }
        )

    return client.chat.completions.create(
        model="",
        messages=messages,
        stream=True,
        max_tokens=512,
    )


def tts(client, text):
    return client.chat.completions.create(
        model="",
        messages=[
            {"role": "system", "content": "Perform TTS."},
            {"role": "user", "content": text},
        ],
        stream=True,
        max_tokens=512,
    )


def asr(client, wav_data):
    encoded_wav_data = base64.b64encode(wav_data).decode("utf-8")
    return client.chat.completions.create(
        model="",
        messages=[
            {"role": "system", "content": "Perform ASR."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded_wav_data, "format": "wav"},
                    }
                ],
            },
        ],
        stream=True,
        max_tokens=512,
    )


def collect_output(stream):
    t0 = time.time()
    received_text = []
    received_audio = []
    completed = False

    for chunk in stream:
        # Check for proper completion
        if chunk.choices[0].finish_reason == "stop":
            completed = True
            break

        delta = chunk.choices[0].delta

        # Handle text content
        if text := delta.content:
            received_text.append((time.time(), text))
            print(text, end="", flush=True)

        # Handle audio chunks
        if hasattr(delta, "audio_chunk") and delta.audio_chunk:
            chunk_data = delta.audio_chunk["data"]
            pcm_bytes = base64.b64decode(chunk_data)
            samples = struct.unpack(f"<{len(pcm_bytes) // 4}f", pcm_bytes)
            received_audio.append((time.time(), samples))

    if not completed:
        raise ConnectionError("Server disconnected before completion")

    text = "".join(t for _, t in received_text)
    audio = [s for _, samples in received_audio for s in samples]

    print("\n\n--- Performance Metrics ---")
    print(
        f"TTFT :                        {min(x[0][0] for x in [received_text, received_audio] if x) - t0:>5.3f}         s"
    )
    if text:
        print(
            f"Text : {len(received_text):>8}  tokens at {len(received_text) / (received_text[-1][0] - received_text[0][0]):>8.0f}  tokens/s"
        )
    if audio:
        print(
            f"Audio: {len(audio):>8} samples at {len(audio) / (received_audio[-1][0] - received_audio[0][0]):>8.0f} samples/s"
        )

    return text if text else None, audio if audio else None


def make_request(base_url, mode, wav_file, text, output):
    client = OpenAI(base_url=base_url, api_key="dummy")

    # Load WAV data if provided
    wav_data = None
    if wav_file:
        with open(wav_file, "rb") as f:
            wav_data = f.read()
        print(f"Loaded audio from {wav_file}")

    # Select mode and create stream
    if mode == "asr":
        print("Mode: ASR (Audio -> Text)")
        stream = asr(client, wav_data)
    elif mode == "tts":
        print("Mode: TTS (Text -> Audio)")
        print(f"Input text: {text}")
        stream = tts(client, text)
    elif mode == "interleaved":
        print("Mode: Interleaved (Audio + Text)")
        stream = interleaved(client, text=text, wav_data=wav_data)

    # Collect output
    text, audio_samples = collect_output(stream)

    # Display results
    if audio_samples:
        print(f"\nReceived {len(audio_samples)} audio samples")
        sample_rate = 24000
        sf.write(output, audio_samples, sample_rate)
        print(f"Saved audio to {output}")

    if text:
        print(f"\nTranscribed/Generated text: {text}")


def main():
    parser = argparse.ArgumentParser(
        description="Test LFM2-Audio server with OpenAI-compatible API"
    )
    parser.add_argument(
        "--wav", type=str, help="Path to input WAV file for ASR or interleaved mode"
    )
    parser.add_argument(
        "--text", type=str, help="Text prompt for TTS or interleaved mode"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["asr", "tts", "interleaved"],
        default="interleaved",
        help="Mode: asr (audio->text), tts (text->audio), or interleaved (both)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output WAV file path (default: output.wav)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8080/v1",
        help="Server base URL (default: http://127.0.0.1:8080/v1)",
    )

    args = parser.parse_args()

    # Validate inputs based on mode
    if args.mode == "asr" and not args.wav:
        parser.error("ASR mode requires --wav")
    if args.mode == "tts" and not args.text:
        parser.error("TTS mode requires --text")
    if args.mode == "interleaved" and not args.wav and not args.text:
        parser.error("Interleaved mode requires one of --wav or --text")

    make_request(args.base_url, args.mode, args.wav, args.text, args.output)


if __name__ == "__main__":
    main()
