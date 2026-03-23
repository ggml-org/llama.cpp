"""
Tests for POST /v1/audio/transcriptions endpoint.

Requires a server started with an audio-capable model (e.g. Voxtral Realtime).
Since no tiny audio test model exists in ggml-org, these tests use the
tinygemma3 preset (which has mmproj) to verify error handling, and are
marked to skip if no audio model is available.

For full end-to-end testing with real audio transcription, run manually:
    llama-server -m voxtral.gguf --mmproj mmproj.gguf -ngl 99 --port 8080
    pytest test_audio_transcription.py -v
"""

import base64
import io
import struct
import wave
import pytest
import requests
from utils import *


server: ServerProcess


def generate_wav_bytes(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a silent WAV file in memory."""
    n_samples = int(sample_rate * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b'\x00\x00' * n_samples)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinygemma3()


# ─── Error handling tests (work with any multimodal model) ───


def test_audio_transcription_no_audio_support():
    """Verify endpoint returns error when model doesn't support audio."""
    global server
    # tinygemma3 is vision-only, not audio
    server.start()
    wav_data = generate_wav_bytes()
    url = f"http://{server.server_host}:{server.server_port}/v1/audio/transcriptions"
    res = requests.post(url, files={"file": ("test.wav", wav_data, "audio/wav")}, timeout=30)
    assert res.status_code in [400, 501, 503]  # should fail gracefully


def test_audio_transcription_missing_file():
    """Verify endpoint returns error when no file is provided."""
    global server
    server.start()
    url = f"http://{server.server_host}:{server.server_port}/v1/audio/transcriptions"
    res = requests.post(url, json={}, timeout=30)
    assert res.status_code == 400


def test_audio_transcription_empty_body():
    """Verify endpoint returns error on empty body."""
    global server
    server.start()
    url = f"http://{server.server_host}:{server.server_port}/v1/audio/transcriptions"
    res = requests.post(url, data="", headers={"Content-Type": "application/json"}, timeout=30)
    assert res.status_code in [400, 500]


def test_audio_transcription_json_format():
    """Verify endpoint accepts JSON with base64 audio."""
    global server
    server.start()
    wav_data = generate_wav_bytes()
    wav_b64 = base64.b64encode(wav_data).decode()
    url = f"http://{server.server_host}:{server.server_port}/v1/audio/transcriptions"
    res = requests.post(url, json={
        "file": wav_b64,
        "format": "wav",
        "language": "en",
    }, timeout=30)
    # should fail because tinygemma3 doesn't support audio, but NOT with 404
    assert res.status_code != 404


def test_audio_transcription_multipart_format():
    """Verify endpoint accepts multipart/form-data."""
    global server
    server.start()
    wav_data = generate_wav_bytes()
    url = f"http://{server.server_host}:{server.server_port}/v1/audio/transcriptions"
    res = requests.post(
        url,
        files={"file": ("test.wav", wav_data, "audio/wav")},
        data={"language": "en", "model": "voxtral"},
        timeout=30,
    )
    # should fail because tinygemma3 doesn't support audio, but NOT with 404
    assert res.status_code != 404


def test_audio_transcription_endpoint_exists():
    """Verify the endpoint is registered and returns something other than 404."""
    global server
    server.start()
    url = f"http://{server.server_host}:{server.server_port}/v1/audio/transcriptions"
    res = requests.post(url, json={"file": "dGVzdA=="}, timeout=30)
    assert res.status_code != 404


# ─── Integration test (requires Voxtral Realtime model) ───


@pytest.mark.skipif(
    not os.environ.get("LLAMA_TEST_AUDIO_MODEL"),
    reason="Set LLAMA_TEST_AUDIO_MODEL=1 to run with a real audio model"
)
def test_audio_transcription_real_model():
    """
    Full end-to-end test with a real audio model.
    Run manually with:
        LLAMA_TEST_AUDIO_MODEL=1 pytest test_audio_transcription.py::test_audio_transcription_real_model -v
    Expects the server to be already running with a Voxtral Realtime model.
    """
    global server
    server.start()

    # generate a short WAV with a 440Hz tone (not silence)
    import math
    sample_rate = 16000
    duration = 2.0
    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        samples.append(int(math.sin(2 * math.pi * 440 * i / sample_rate) * 16000))

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f'<{n_samples}h', *samples))
    wav_data = buf.getvalue()

    url = f"http://{server.server_host}:{server.server_port}/v1/audio/transcriptions"

    # test multipart
    res = requests.post(
        url,
        files={"file": ("test.wav", wav_data, "audio/wav")},
        data={"language": "en"},
        timeout=120,
    )
    assert res.status_code == 200
    body = res.json()
    assert "text" in body
    assert body["task"] == "transcribe"

    # test JSON base64
    wav_b64 = base64.b64encode(wav_data).decode()
    res = requests.post(url, json={
        "file": wav_b64,
        "format": "wav",
        "language": "en",
    }, timeout=120)
    assert res.status_code == 200
    body = res.json()
    assert "text" in body
    assert body["task"] == "transcribe"
