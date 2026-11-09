import pytest
import requests
import struct
from utils import *

server: ServerProcess

BODY = {
    "model": "tts",
    "input": "Hello there.",
    "voice": "female",
    "seed": 42,
    "n_predict": 8,
}


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinylfm3omni()


def post_audio(path: str, **extra) -> requests.Response:
    return requests.post(server.make_url(path), json={**BODY, **extra}, timeout=DEFAULT_REQUEST_TIMEOUT)


def assert_wav(res: requests.Response, data_size: int | None = None):
    assert res.status_code == 200
    assert res.headers["Content-Type"] == "audio/wav"
    assert res.content[:4] == b"RIFF" and res.content[8:12] == b"WAVE"
    assert struct.unpack("<H", res.content[34:36])[0] == 16 # bits per sample
    pos = res.content.index(b"data")
    size = struct.unpack("<I", res.content[pos + 4:pos + 8])[0]
    assert size == (len(res.content) - pos - 8 if data_size is None else data_size)
    assert len(res.content) > pos + 8


@pytest.mark.parametrize("path", ["/v1/audio/speech", "/audio/speech"])
def test_speech_default_format_is_wav(path: str):
    server.start()
    assert_wav(post_audio(path))


def test_speech_wav_same_as_tts():
    server.start()
    res = post_audio("/v1/audio/speech", response_format="wav")
    assert_wav(res)
    assert res.content == post_audio("/tts").content


def test_speech_pcm_is_signed_16_bit():
    server.start()
    res = post_audio("/v1/audio/speech", response_format="pcm")
    assert res.status_code == 200
    assert res.headers["Content-Type"] == "audio/pcm;rate=24000;encoding=signed-int;bits=16"

    # /tts keeps the float samples
    ref = post_audio("/tts", response_format="pcm")
    assert ref.headers["Content-Type"] == "audio/pcm;rate=24000;encoding=float;bits=32"

    samples = struct.unpack(f"<{len(res.content) // 2}h", res.content)
    floats = struct.unpack(f"<{len(ref.content) // 4}f", ref.content)
    assert len(samples) == len(floats) > 0
    assert any(v != 0 for v in samples)
    for s, f in zip(samples, floats):
        assert abs(s - max(-1.0, min(1.0, f)) * 32767) <= 1


def test_speech_stream_pcm():
    server.start()
    res = post_audio("/v1/audio/speech", response_format="pcm", stream_format="audio")
    assert res.status_code == 200
    assert res.headers.get("Transfer-Encoding") == "chunked"
    assert res.content == post_audio("/v1/audio/speech", response_format="pcm").content


def test_speech_stream_wav_has_unknown_size():
    server.start()
    res = post_audio("/v1/audio/speech", stream_format="audio")
    assert_wav(res, data_size=0xFFFFFFFF)


def test_speech_voice():
    server.start()
    female = post_audio("/v1/audio/speech", voice="female")
    male = post_audio("/v1/audio/speech", voice="male")
    assert_wav(female)
    assert_wav(male)
    assert female.content != male.content
    assert post_audio("/v1/audio/speech", voice={"id": "male"}).content == male.content


def test_speech_multipart():
    server.start()
    fields = {k: (None, str(v)) for k, v in BODY.items()}
    res = requests.post(server.make_url("/v1/audio/speech"), files=fields, timeout=DEFAULT_REQUEST_TIMEOUT)
    assert_wav(res)
    assert res.content == post_audio("/v1/audio/speech").content


@pytest.mark.parametrize("extra", [
    {"speed": 1.0},
    {"instructions": "Speak slowly."},
])
def test_speech_accepted_params(extra: dict):
    server.start()
    assert_wav(post_audio("/v1/audio/speech", **extra))


@pytest.mark.parametrize("extra,message", [
    ({"response_format": "mp3"},  "response_format"),
    ({"response_format": "opus"}, "response_format"),
    ({"stream_format": "sse"},    "stream_format"),
    ({"speed": 1.5},              "speed"),
    ({"voice": "alloy"},          "unknown voice"),
    ({"input": ""},               "input"),
])
def test_speech_invalid_request(extra: dict, message: str):
    server.start()
    res = post_audio("/v1/audio/speech", **extra)
    assert res.status_code == 400
    assert message in res.json()["error"]["message"]


def test_speech_no_tts_model():
    global server
    server = ServerPreset.tinyllama2()
    server.start()
    assert post_audio("/v1/audio/speech").status_code == 501 # ERROR_TYPE_NOT_SUPPORTED


def test_speech_router():
    global server
    model = server.model_hf_repo
    server = ServerPreset.router()
    server.start()
    assert_wav(post_audio("/v1/audio/speech", model=model))
