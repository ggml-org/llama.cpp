import argparse
import base64
import time

import numpy as np
from openai import OpenAI

def tts(client, text):
    return client.chat.completions.create(
        model="",
        modalities=["audio"],
        messages=[
            {"role": "system", "content": "Perform TTS. Use the US male voice."},
            {"role": "user", "content": text},
        ],
        stream=True,
        max_tokens=512,
    )

def main():
    client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="dummy")
    t0 = time.time()
    try:
        stream = tts(client, "Hello, this is a test.")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return
    ttft = None
    first_audio_time = None
    total_samples = 0
    audio_sample_rate = None

    for chunk in stream:
        delta = chunk.choices[0].delta
        if hasattr(delta, "audio") and delta.audio and "data" in delta.audio:
            if first_audio_time is None:
                first_audio_time = time.time() - t0
                print(f"First audio chunk received at {first_audio_time:.3f}s")

            if audio_sample_rate is None and "sample_rate" in delta.audio:
                audio_sample_rate = delta.audio["sample_rate"]
            chunk_data = delta.audio["data"]
            pcm_bytes = base64.b64decode(chunk_data)
            samples = np.frombuffer(pcm_bytes, dtype=np.int16)
            total_samples += len(samples)

    print(f"Total time: {time.time() - t0:.3f}s")
    print(f"Total samples: {total_samples}")

if __name__ == "__main__":
    main()
