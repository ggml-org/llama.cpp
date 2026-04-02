import json
import time
import pytest
import requests
import threading
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_monitor_disabled():
    global server
    server.server_slots = False
    server.start()
    res = server.make_request("GET", "/monitor")
    assert res.status_code == 501
    assert "error" in res.body


def test_monitor_sse_content_type():
    global server
    server.server_slots = True
    server.start()
    url = f"http://{server.server_host}:{server.server_port}/monitor"
    with requests.get(url, stream=True, timeout=10) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("Content-Type", "")
        assert response.headers.get("Cache-Control") == "no-cache"
        # read at least one event
        for line_bytes in response.iter_lines():
            line = line_bytes.decode("utf-8")
            if line.startswith("data: "):
                data = json.loads(line[6:])
                assert "timestamp_ms" in data
                assert "uptime_seconds" in data
                assert isinstance(data["uptime_seconds"], int)
                assert data["uptime_seconds"] >= 0
                assert "idle_slots" in data
                assert "processing_slots" in data
                assert "slots" in data
                assert "metrics" in data
                assert len(data["slots"]) == server.n_slots
                # verify metrics fields
                m = data["metrics"]
                assert "prompt_tokens_per_second" in m
                assert "predicted_tokens_per_second" in m
                assert "prompt_tokens_total" in m
                assert "predicted_tokens_total" in m
                assert "n_decode_total" in m
                break


def test_monitor_prompt_visible():
    """Verify prompt/generated text appears without LLAMA_SERVER_SLOTS_DEBUG."""
    global server
    server.server_slots = True
    server.start()

    # generate a completion first
    res = server.make_request("POST", "/completion", data={
        "n_predict": 16,
        "prompt": "Hello world",
        "temperature": 0.0,
    })
    assert res.status_code == 200

    # now check that monitor shows the prompt
    url = f"http://{server.server_host}:{server.server_port}/monitor"
    with requests.get(url, stream=True, timeout=10) as response:
        assert response.status_code == 200
        for line_bytes in response.iter_lines():
            line = line_bytes.decode("utf-8")
            if line.startswith("data: "):
                data = json.loads(line[6:])
                # find a slot that has prompt data
                for slot in data["slots"]:
                    if "prompt" in slot:
                        assert isinstance(slot["prompt"], str)
                        assert len(slot["prompt"]) > 0
                        assert "generated" in slot
                        return  # success
                break

    # if we got here, no slot had prompt data -- fail
    pytest.fail("No slot contained prompt text in monitor output")


def test_monitor_events_during_completion():
    global server
    server.server_slots = True
    server.n_predict = 32
    server.start()

    events = []
    stop_flag = threading.Event()

    def collect_events():
        url = f"http://{server.server_host}:{server.server_port}/monitor"
        try:
            with requests.get(url, stream=True, timeout=10) as response:
                for line_bytes in response.iter_lines():
                    if stop_flag.is_set():
                        break
                    line = line_bytes.decode("utf-8")
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        events.append(data)
                        if len(events) >= 10:
                            break
        except Exception:
            pass

    monitor_thread = threading.Thread(target=collect_events)
    monitor_thread.start()

    # wait for monitor to connect and receive first event
    time.sleep(1.0)

    # trigger a completion
    res = server.make_request("POST", "/completion", data={
        "n_predict": 16,
        "prompt": "Hello",
        "temperature": 0.0,
    })
    assert res.status_code == 200

    # give monitor time to observe the completion
    time.sleep(2.0)
    stop_flag.set()
    monitor_thread.join(timeout=5)

    # verify we got valid events
    assert len(events) >= 1
    for e in events:
        assert "timestamp_ms" in e
        assert "uptime_seconds" in e
        assert "slots" in e
        assert "metrics" in e


def test_monitor_max_clients():
    """Verify that the server rejects monitor connections beyond the limit."""
    global server
    server.server_slots = True
    server.start()

    connections = []
    try:
        for i in range(6):
            r = requests.get(
                f"http://{server.server_host}:{server.server_port}/monitor",
                stream=True, timeout=5)
            connections.append(r)
            time.sleep(0.1)  # small delay for sequential connection ordering

        ok_count = sum(1 for r in connections if r.status_code == 200)
        rejected_count = sum(1 for r in connections if r.status_code == 503)

        assert ok_count <= 4, f"Expected at most 4 accepted, got {ok_count}"
        assert rejected_count >= 2, f"Expected at least 2 rejected, got {rejected_count}"
    finally:
        for r in connections:
            r.close()
