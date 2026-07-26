import json
import threading
import time
from urllib.parse import quote
import pytest
from utils import *

server: ServerProcess

# a model name with slashes exercises the query string routing of the stream routes: the id
# cannot travel as a path param because the decoded slash would split it before capture
MODEL = "ggml-org/tinygemma3-GGUF:Q8_0"
STREAM_ID = f"conv-stream-test::{MODEL}"
QS = "conv_id=" + quote(STREAM_ID, safe="")


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.router()


def test_stream_resume_and_stop_with_slashed_model_name():
    global server
    server.start()

    content = ""
    for data in server.make_stream_request("POST", "/chat/completions", data={
        "model": MODEL,
        "stream": True,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hello"}],
    }, headers={"X-Conversation-Id": STREAM_ID}):
        if data["choices"]:
            content += data["choices"][0]["delta"].get("content") or ""
    assert len(content) > 0

    # the finished session replays from the beginning through the router
    res = server.make_request("GET", f"/v1/stream?{QS}&from=0")
    assert res.status_code == 200
    assert "data: " in str(res.body)

    # the explicit stop reaches the owning child and evicts the session
    res = server.make_request("DELETE", f"/v1/stream?{QS}")
    assert res.status_code == 204
    res = server.make_request("GET", f"/v1/stream?{QS}&from=0")
    assert res.status_code == 404


def test_stream_stop_during_model_load():
    global server
    server.start()

    thread_error: list[ServerError] = []
    thread_done = threading.Event()

    def fire_post():
        try:
            for _ in server.make_stream_request("POST", "/chat/completions", data={
                "model": MODEL,
                "stream": True,
                "max_tokens": 512,
                "messages": [{"role": "user", "content": "Count from 1 to 1000."}],
            }, headers={"X-Conversation-Id": STREAM_ID}):
                pass
        except ServerError as e:
            thread_error.append(e)
        finally:
            thread_done.set()

    t = threading.Thread(target=fire_post)
    t.start()

    # catch the autoload window, tiny models load fast so poll aggressively
    saw_loading = False
    deadline = time.time() + 5.0
    while time.time() < deadline and not thread_done.is_set():
        res = server.make_request("GET", "/models")
        status = next(m["status"]["value"] for m in res.body["data"] if m["id"] == MODEL)
        if status == "loading":
            saw_loading = True
            break
        time.sleep(0.002)
    if not saw_loading:
        t.join()
        pytest.skip("load window too short to be observed on this machine")

    # a stop during the load cancels the parked request instead of leaving an orphan
    res = server.make_request("DELETE", f"/v1/stream?{QS}")
    assert res.status_code == 204
    assert thread_done.wait(timeout=60)
    t.join()
    assert len(thread_error) == 1
    assert thread_error[0].code == 400
    assert "cancelled" in json.dumps(thread_error[0].body)
    res = server.make_request("GET", f"/v1/stream?{QS}&from=0")
    assert res.status_code == 404
