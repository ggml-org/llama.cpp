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
