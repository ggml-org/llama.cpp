import base64
import json
import time
import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect, ClientConnection
from websockets.sync.connection import Connection
from utils import *

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinylfm3omni()
    server.jinja = True
    server.reasoning = "off"


def ws_connect(path: str = "/v1/realtime") -> ClientConnection:
    return connect(server.make_url(path).replace("http://", "ws://", 1), max_size=None, open_timeout=DEFAULT_REQUEST_TIMEOUT)


def send(ws: Connection, type: str, **fields):
    ws.send(json.dumps({"type": type, **fields}))


def send_audio(ws: Connection, pcm: bytes):
    send(ws, "input_audio_buffer.append", audio=base64.b64encode(pcm).decode())


def send_text(ws: Connection, text: str):
    send(ws, "conversation.item.create", item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})


def recv(ws: Connection) -> dict:
    return json.loads(ws.recv(timeout=DEFAULT_REQUEST_TIMEOUT))


def recv_until(ws: Connection, type: str) -> list[dict]:
    events = []
    while True:
        ev = recv(ws)
        events.append(ev)
        if ev["type"] == type:
            return events
        assert ev["type"] != "error", ev


def recv_all(ws: Connection) -> list[dict]:
    """ read events until the server closes the socket """
    events = []
    try:
        while True:
            events.append(recv(ws))
    except ConnectionClosed:
        return events


def get_transcripts(events: list[dict]) -> list[tuple[str, str]]:
    """ (item_id, transcript) of each completed item, checked against its deltas """
    deltas: dict[str, str] = {}
    out = []
    for ev in events:
        assert ev["type"] != "error", ev
        if ev["type"] == "conversation.item.input_audio_transcription.delta":
            deltas[ev["item_id"]] = deltas.get(ev["item_id"], "") + ev["delta"]
        elif ev["type"] == "conversation.item.input_audio_transcription.completed":
            assert ev["transcript"] == deltas.get(ev["item_id"], "")
            out.append((ev["item_id"], ev["transcript"]))
    return out


def session_update(ws: Connection, **session) -> dict:
    send(ws, "session.update", session=session)
    return recv_until(ws, "session.updated")[-1]["session"]


#
# live transcription
#

# [TAG_MTMD_NO_REALLOC] TODO: mtmd reallocs its graphs on audio stream steps
skip_no_realloc = pytest.mark.skipif(is_no_realloc(), reason="mtmd: graph realloc on audio stream step")

LIVE_ASR = [
    # a chat model with an "asr" chat template, live only in sessions opened with ?intent=transcription
    ("tinylfm3omni",  "/v1/realtime?intent=transcription"),
    # no chat, the text follows the audio by a few tokens
    pytest.param("tinyvoxtralrt", "/v1/realtime", marks=skip_no_realloc),
]


def start_live_asr(preset: str):
    global server
    if preset != "tinylfm3omni":
        server = getattr(ServerPreset, preset)()
    server.start()


@pytest.mark.parametrize("preset,path", LIVE_ASR)
def test_live_asr(preset: str, path: str):
    start_live_asr(preset)
    with ws_connect(path) as ws:
        session = recv(ws)
        assert session["type"] == "session.created"
        assert session["session"]["object"] == "realtime.transcription_session"
        assert session["session"]["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 16000}
        send_audio(ws, make_pcm16())
        send(ws, "input_audio_buffer.commit")
        send_audio(ws, make_pcm16(freq=880))
        send(ws, "input_audio_buffer.commit", final=True)
        events = recv_all(ws)
    committed = [ev for ev in events if ev["type"] == "input_audio_buffer.committed"]
    assert len(committed) == 1
    assert committed[0]["previous_item_id"] is None
    transcripts = get_transcripts(events)
    assert len(transcripts) == 2
    assert transcripts[0][0] == committed[0]["item_id"]
    assert transcripts[1][0] != transcripts[0][0]
    assert all(len(text) > 0 for _, text in transcripts)


@skip_no_realloc
@pytest.mark.parametrize("preset,path", LIVE_ASR)
def test_live_asr_chunked_audio(preset: str, path: str):
    start_live_asr(preset)
    pcm = make_pcm16()
    results = []
    for n_chunks in [1, 7]:
        size = len(pcm) // n_chunks // 2 * 2
        with ws_connect(path) as ws:
            recv(ws)
            for i in range(0, len(pcm), size):
                send_audio(ws, pcm[i:i + size])
            send(ws, "input_audio_buffer.commit", final=True)
            results.append([text for _, text in get_transcripts(recv_all(ws))])
    assert results[0] == results[1]


@pytest.mark.parametrize("preset,path", LIVE_ASR)
def test_live_asr_input_rate(preset: str, path: str):
    start_live_asr(preset)
    with ws_connect(path) as ws:
        recv(ws)
        send(ws, "session.update", session={"audio": {"input": {"format": {"type": "audio/pcm", "rate": 4000}}}})
        assert "between 8000 and 192000" in recv_until(ws, "error")[-1]["error"]["message"]
        send(ws, "session.update", session={"audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}}})
        assert recv_until(ws, "session.updated")[-1]["session"]["audio"]["input"]["format"]["rate"] == 24000
        send_audio(ws, make_pcm16(rate=24000))
        send(ws, "input_audio_buffer.commit", final=True)
        transcripts = get_transcripts(recv_all(ws))
    assert len(transcripts) == 1
    assert len(transcripts[0][1]) > 0


@pytest.mark.parametrize("preset,path", LIVE_ASR)
def test_live_asr_invalid_events(preset: str, path: str):
    start_live_asr(preset)
    with ws_connect(path) as ws:
        recv(ws)
        ws.send("not json")
        assert recv_until(ws, "error")[-1]["error"]["message"] == "invalid JSON"
        send(ws, "response.create")
        assert "unknown event type" in recv_until(ws, "error")[-1]["error"]["message"]
        send(ws, "input_audio_buffer.clear")
        recv_until(ws, "input_audio_buffer.cleared")
        # the session still works
        send_audio(ws, make_pcm16())
        send(ws, "input_audio_buffer.commit", final=True)
        assert len(get_transcripts(recv_all(ws))) == 1


@skip_no_realloc
def test_live_asr_depends_on_audio():
    # the random weights of this model give a different text for each input
    start_live_asr("tinyvoxtralrt")
    results = []
    for freq in [220, 880, 220]:
        with ws_connect() as ws:
            recv(ws)
            send_audio(ws, make_pcm16(freq=freq))
            send(ws, "input_audio_buffer.commit", final=True)
            results.append(get_transcripts(recv_all(ws))[0][1])
    assert results[0] != results[1]
    assert results[0] == results[2]


@pytest.mark.parametrize("preset", ["tinypockettts", "tinyllama2"])
def test_realtime_not_supported(preset: str):
    global server
    server = getattr(ServerPreset, preset)()
    server.start()
    with ws_connect() as ws:
        events = recv_all(ws)
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "does not support realtime" in events[0]["error"]["message"]


#
# realtime conversation
#

def test_conversation_text():
    server.start()
    with ws_connect() as ws:
        session = recv(ws)
        assert session["type"] == "session.created"
        assert session["session"]["object"] == "realtime.session"
        assert session["session"]["audio"]["output"]["format"]["rate"] > 0
        session_update(ws, output_modalities=["text"], max_output_tokens=8)
        send_text(ws, "hello")
        user_item = recv_until(ws, "conversation.item.done")[-1]["item"]
        assert user_item["role"] == "user"
        assert user_item["content"] == [{"type": "input_text", "text": "hello"}]
        send(ws, "response.create")
        events = recv_until(ws, "response.done")

    types = [ev["type"] for ev in events]
    n_deltas = types.count("response.output_text.delta")
    assert n_deltas == 8
    assert types == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        *["response.output_text.delta"] * n_deltas,
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.done",
    ]
    response = events[-1]["response"]
    assert response["status"] == "completed"
    assert response["id"] == events[0]["response"]["id"]
    assert response["usage"]["output_tokens"] == 8
    text = "".join(ev["delta"] for ev in events if ev["type"] == "response.output_text.delta")
    assert response["output"][0]["role"] == "assistant"
    assert response["output"][0]["content"] == [{"type": "output_text", "text": text}]


def test_conversation_audio():
    server.start()
    with ws_connect() as ws:
        recv(ws)
        session_update(ws, output_modalities=["audio"], max_output_tokens=8)
        send_audio(ws, make_pcm16())
        # response.create commits the pending audio
        send(ws, "response.create")
        events = recv_until(ws, "response.done")
    types = [ev["type"] for ev in events]
    assert types[:3] == ["input_audio_buffer.committed", "conversation.item.added", "conversation.item.done"]
    assert events[2]["item"]["content"] == [{"type": "input_audio"}]
    assert "response.output_text.delta" not in types
    audio = b"".join(base64.b64decode(ev["delta"]) for ev in events if ev["type"] == "response.output_audio.delta")
    assert len(audio) > 0 and len(audio) % 2 == 0
    response = events[-1]["response"]
    assert response["status"] == "completed"
    assert response["output"][0]["content"][0]["type"] == "output_audio"


def test_conversation_multi_turn():
    server.start()
    with ws_connect() as ws:
        recv(ws)
        session_update(ws, output_modalities=["text"], max_output_tokens=4)
        send_text(ws, "hello")
        recv_until(ws, "conversation.item.done")
        send(ws, "response.create")
        first = recv_until(ws, "response.done")[-1]["response"]
        send_audio(ws, make_pcm16())
        send(ws, "input_audio_buffer.commit")
        committed = recv_until(ws, "input_audio_buffer.committed")[-1]
        assert committed["previous_item_id"] == first["output"][0]["id"]
        send(ws, "response.create")
        second = recv_until(ws, "response.done")[-1]["response"]
        assert second["status"] == "completed"
        assert second["id"] != first["id"]


def test_conversation_cancel():
    server.start()
    with ws_connect() as ws:
        recv(ws)
        session_update(ws, output_modalities=["text"])
        send_text(ws, "hello")
        send(ws, "response.create")
        recv_until(ws, "response.output_text.delta")
        send(ws, "response.cancel")
        assert recv_until(ws, "response.done")[-1]["response"]["status"] == "cancelled"
        # the session still works
        send_text(ws, "hello again")
        send(ws, "response.create", response={"max_output_tokens": 4})
        response = recv_until(ws, "response.done")[-1]["response"]
        assert response["status"] == "completed"
        assert response["usage"]["output_tokens"] == 4


def test_conversation_session_update():
    server.start()
    tools = [{"type": "function", "name": "get_weather", "description": "Get the weather",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
    with ws_connect() as ws:
        recv(ws)
        session = session_update(ws, instructions="Be brief.", tools=tools, tool_choice="none", max_output_tokens=16,
                                 audio={"output": {"voice": {"id": "male"}}})
        assert session["instructions"] == "Be brief."
        assert session["tools"] == tools
        assert session["tool_choice"] == "none"
        assert session["max_output_tokens"] == 16
        assert session["audio"]["output"]["voice"] == "male"
        session = session_update(ws, max_output_tokens="inf", output_modalities=["text", "audio"])
        assert session["max_output_tokens"] == "inf"
        assert session["output_modalities"] == ["text", "audio"]


@pytest.mark.parametrize("events,message", [
    ([{"type": "response.create"}],                                              "commit audio or create a message first"),
    ([{"type": "input_audio_buffer.commit"}],                                    "the buffer is empty"),
    ([{"type": "conversation.item.create", "item": {"type": "message", "role": "assistant", "content": []}}], "only user messages"),
    ([{"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_image"}]}}], "unsupported content type"),
    ([{"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": "x", "output": "{}"}}], "must follow the response"),
    ([{"type": "session.update", "session": {"type": "foo"}}],                   "unknown session type"),
    ([{"type": "session.update", "session": {"output_modalities": []}}],         "non-empty array"),
    ([{"type": "session.update", "session": {"output_modalities": ["video"]}}],  "unknown modality"),
    ([{"type": "session.update", "session": {"max_output_tokens": 0}}],          "max_output_tokens"),
    ([{"type": "session.update", "session": {"tool_choice": "maybe"}}],          "tool_choice"),
    ([{"type": "session.update", "session": {"audio": {"output": {"voice": "alloy"}}}}], "unknown voice"),
    ([{"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}},
      {"type": "session.update", "session": {"instructions": "Be brief."}}],    "cannot change after"),
    ([{"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}},
      {"type": "session.update", "session": {"type": "transcription"}}],        "cannot change after"),
])
def test_conversation_invalid_events(events: list[dict], message: str):
    server.start()
    with ws_connect() as ws:
        recv(ws)
        for ev in events:
            ws.send(json.dumps(ev))
        err = recv_until(ws, "error")[-1]
        assert message in err["error"]["message"]


def test_transcription_session():
    server.start()
    with ws_connect() as ws:
        recv(ws)
        session = session_update(ws, type="transcription", max_output_tokens=8,
                                 audio={"input": {"transcription": {"language": "fr"}}})
        assert session["object"] == "realtime.transcription_session"
        assert session["audio"]["input"]["transcription"]["language"] == "fr"
        send_text(ws, "hello")
        assert "not supported in a transcription session" in recv_until(ws, "error")[-1]["error"]["message"]
        transcripts = []
        for freq in [440, 880]:
            send_audio(ws, make_pcm16(freq=freq))
            send(ws, "input_audio_buffer.commit")
            committed = recv_until(ws, "input_audio_buffer.committed")[-1]
            events = recv_until(ws, "conversation.item.input_audio_transcription.completed")
            transcripts += get_transcripts(events)
            assert transcripts[-1][0] == committed["item_id"]
    assert len(transcripts) == 2
    assert all(len(text) > 0 for _, text in transcripts)


#
# slot and server state
#

@pytest.mark.parametrize("path", ["/v1/realtime", "/v1/realtime?intent=transcription"])
def test_realtime_disconnect_frees_slot(path: str):
    server.n_slots = 1
    server.start()
    with ws_connect(path) as ws:
        recv(ws)
        send_audio(ws, make_pcm16())
        if path == "/v1/realtime":
            send(ws, "response.create")
            recv_until(ws, "response.output_item.added")
    # the socket is closed in the middle of the session
    res = server.make_request("POST", "/v1/chat/completions", data={
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert res.status_code == 200


def test_realtime_router():
    global server
    model = server.model_hf_repo
    server = ServerPreset.router()
    server.start()
    with ws_connect(f"/v1/realtime?model={model}") as ws:
        assert recv(ws)["type"] == "session.created"
        session_update(ws, output_modalities=["text"], max_output_tokens=4)
        send_text(ws, "hello")
        send(ws, "response.create")
        assert recv_until(ws, "response.done")[-1]["response"]["status"] == "completed"


def test_realtime_sleep():
    server.sleep_idle_seconds = 1
    server.start()
    start = time.time()
    while not server.make_request("GET", "/props").body["is_sleeping"]:
        assert time.time() - start < 10, "server did not go to sleep"
        time.sleep(0.1)
    # the session wakes the server up
    with ws_connect() as ws:
        assert recv(ws)["type"] == "session.created"
        session_update(ws, output_modalities=["text"], max_output_tokens=4)
        send_text(ws, "hello")
        send(ws, "response.create")
        assert recv_until(ws, "response.done")[-1]["response"]["status"] == "completed"
