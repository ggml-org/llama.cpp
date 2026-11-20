import base64
import json
import pytest
import requests
from utils import *

server: ServerProcess

MP3_PATH = os.path.join(os.path.dirname(__file__), "../../../mtmd/test-2.mp3")

# full random sampling, so that the output depends on the audio
SAMPLING_RANDOM = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "seed": 42}


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinylfm3omni()
    server.jinja = True
    server.reasoning = "off"


def post_transcription(path: str = "/v1/audio/transcriptions", audio: bytes | None = None, filename: str = "audio.wav", **fields) -> requests.Response:
    data = {"max_tokens": "8", **fields}
    files = {"file": (filename, make_wav() if audio is None else audio)}
    return requests.post(server.make_url(path), data=data, files=files, timeout=DEFAULT_REQUEST_TIMEOUT)


def chat_with_audio(audio: bytes, fmt: str = "wav", extra: dict | None = None) -> str:
    res = server.make_request("POST", "/v1/chat/completions", data={
        "max_tokens": 8,
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(audio).decode(), "format": fmt}},
        ]}],
        **(extra or {}),
    })
    assert res.status_code == 200, res.body
    return res.body["choices"][0]["message"]["content"]


@pytest.mark.parametrize("path", ["/v1/audio/transcriptions", "/audio/transcriptions"])
def test_transcription(path: str):
    server.start()
    res = post_transcription(path)
    assert res.status_code == 200
    body = res.json()
    assert body["type"] == "transcript.text.done"
    assert len(body["text"]) > 0
    assert 0 < body["usage"]["output_tokens"] <= 8
    assert body["usage"]["input_tokens"] > 0


def test_transcription_mp3():
    server.start()
    with open(MP3_PATH, "rb") as f:
        res = post_transcription(audio=f.read(), filename="audio.mp3")
    assert res.status_code == 200
    assert len(res.json()["text"]) > 0


def test_transcription_stream():
    server.start()
    res = post_transcription(stream="true")
    assert res.status_code == 200
    lines = [line[len("data: "):] for line in res.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    events = [json.loads(line) for line in lines[:-1]]
    assert len(events) > 1
    assert all(ev["type"] == "transcript.text.delta" for ev in events[:-1])
    assert events[-1]["type"] == "transcript.text.done"
    text = "".join(ev["delta"] for ev in events[:-1])
    assert text == events[-1]["text"]
    assert text == post_transcription().json()["text"]


def test_transcription_language():
    server.start()
    # the LFM3 template puts the language in the system prompt
    n_default = post_transcription().json()["usage"]["input_tokens"]
    n_fr = post_transcription(language="fr").json()["usage"]["input_tokens"]
    assert n_fr > n_default


@pytest.mark.parametrize("fields,files,message", [
    ({"response_format": "text"}, None, "response_format"),
    ({}, {"file": ("audio.wav", b"not an audio file")}, None),
    ({}, {"other": ("audio.wav", b"")}, None),
])
def test_transcription_invalid_request(fields: dict, files: dict | None, message: str | None):
    server.start()
    res = requests.post(server.make_url("/v1/audio/transcriptions"), data={"max_tokens": "8", **fields},
                        files=files or {"file": ("audio.wav", make_wav())}, timeout=DEFAULT_REQUEST_TIMEOUT)
    assert res.status_code == 400
    if message:
        assert message in res.json()["error"]["message"]


@pytest.mark.parametrize("preset", [
    "tinyllama2",    # no audio input
    "tinyvoxtralrt", # live transcription only
])
def test_transcription_not_supported(preset: str):
    global server
    server = getattr(ServerPreset, preset)()
    server.start()
    assert post_transcription().status_code == 501 # ERROR_TYPE_NOT_SUPPORTED


def test_transcription_router():
    global server
    model = server.model_hf_repo
    server = ServerPreset.router()
    server.start()
    res = post_transcription(model=model)
    assert res.status_code == 200
    assert res.json()["type"] == "transcript.text.done"


def test_chat_input_audio():
    server.start()
    a = make_wav(freq=220)
    b = make_wav(freq=880)
    assert chat_with_audio(a, extra=SAMPLING_RANDOM) == chat_with_audio(a, extra=SAMPLING_RANDOM)
    assert chat_with_audio(a, extra=SAMPLING_RANDOM) != chat_with_audio(b, extra=SAMPLING_RANDOM)


def test_chat_input_audio_mp3():
    server.start()
    with open(MP3_PATH, "rb") as f:
        assert len(chat_with_audio(f.read(), fmt="mp3")) > 0


def test_chat_input_audio_no_audio_model():
    global server
    server = ServerPreset.tinyllama2()
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data={
        "max_tokens": 8,
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(make_wav()).decode(), "format": "wav"}},
        ]}],
    })
    assert res.status_code != 200
    assert "audio input is not supported" in res.body["error"]["message"]
