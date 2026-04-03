import json
import os
import tempfile
import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.temperature = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_trace_records(trace_dir: str) -> list[dict]:
    """Read all JSONL records from the first (and only) trace file in trace_dir."""
    files = [f for f in os.listdir(trace_dir) if f.startswith("trace-") and f.endswith(".jsonl")]
    assert len(files) == 1, f"Expected exactly one trace file, found: {files}"
    records = []
    with open(os.path.join(trace_dir, files[0])) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def records_for(request_id: str, records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("request_id") == request_id]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_trace_non_streaming():
    """Non-streaming request produces a request + response record pair."""
    global server
    with tempfile.TemporaryDirectory() as trace_dir:
        server.http_trace_dir = trace_dir
        server.start()

        res = server.make_request("POST", "/completion", data={
            "prompt": "Say hello",
            "n_predict": 5,
            "stream": False,
        })
        assert res.status_code == 200

        server.stop()

        records = read_trace_records(trace_dir)
        req_records  = [r for r in records if r["type"] == "request"]
        resp_records = [r for r in records if r["type"] == "response"]

        assert len(req_records)  >= 1
        assert len(resp_records) >= 1

        # Find the /completion pair
        completion_reqs = [r for r in req_records if r["path"] == "/completion"]
        assert len(completion_reqs) == 1
        req = completion_reqs[0]

        # Verify request record fields
        assert "request_id"  in req
        assert "timestamp"   in req
        assert req["method"] == "POST"
        assert req["path"]   == "/completion"
        assert "body_size"   in req
        assert "body_hash"   in req
        assert req["body_size"] > 0
        assert "body_excerpt" in req

        req_id = req["request_id"]
        paired = [r for r in resp_records if r["request_id"] == req_id]
        assert len(paired) == 1, "Response record must share request_id with request record"
        resp = paired[0]

        # Verify response record fields
        assert resp["status"]      == 200
        assert resp["duration_ms"] >= 0
        assert "body_hash"         in resp
        assert "body_size"         in resp
        assert resp["body_size"]   > 0
        assert "body_excerpt"      in resp

        # Streaming records must NOT appear for this request
        for r in records_for(req_id, records):
            assert r["type"] not in ("stream_start", "chunk", "stream_end"), \
                "Non-streaming request should not produce stream records"


def test_trace_streaming():
    """Streaming request produces request + stream_start + chunk(s) + stream_end."""
    global server
    with tempfile.TemporaryDirectory() as trace_dir:
        server.http_trace_dir = trace_dir
        server.start()

        res = server.make_request("POST", "/completion", data={
            "prompt": "Count to five",
            "n_predict": 10,
            "stream": True,
        })
        assert res.status_code == 200

        server.stop()

        records = read_trace_records(trace_dir)

        req_records = [r for r in records if r["type"] == "request" and r["path"] == "/completion"]
        assert len(req_records) == 1
        req_id = req_records[0]["request_id"]

        by_type = {r["type"]: r for r in records_for(req_id, records)}
        all_types = [r["type"] for r in records_for(req_id, records)]

        assert "request"      in by_type, "Missing request record"
        assert "stream_start" in by_type, "Missing stream_start record"
        assert "stream_end"   in by_type, "Missing stream_end record"
        assert "chunk"        in all_types, "Missing at least one chunk record"

        # "response" must NOT appear for a streaming request
        assert "response" not in by_type, "Streaming request should not produce a response record"

        # Chunks must have monotonically increasing seq numbers
        chunks = sorted(
            [r for r in records_for(req_id, records) if r["type"] == "chunk"],
            key=lambda r: r["seq"],
        )
        for i, chunk in enumerate(chunks):
            assert chunk["seq"]        == i
            assert chunk["chunk_size"]  > 0

        # stream_end totals must be consistent
        end = by_type["stream_end"]
        assert end["total_chunks"] == len(chunks)
        assert end["total_bytes"]  == sum(c["chunk_size"] for c in chunks)
        assert end["duration_ms"]  >= 0

        # stream_start must appear before the first chunk in the file
        stream_start_idx = next(i for i, r in enumerate(records) if r.get("request_id") == req_id and r["type"] == "stream_start")
        first_chunk_idx  = next(i for i, r in enumerate(records) if r.get("request_id") == req_id and r["type"] == "chunk")
        assert stream_start_idx < first_chunk_idx


def test_trace_request_ids_are_unique():
    """Each request must receive a distinct request_id."""
    global server
    with tempfile.TemporaryDirectory() as trace_dir:
        server.http_trace_dir = trace_dir
        server.start()

        n = 5
        for _ in range(n):
            res = server.make_request("POST", "/completion", data={
                "prompt": "Hi",
                "n_predict": 3,
                "stream": False,
            })
            assert res.status_code == 200

        server.stop()

        records = read_trace_records(trace_dir)
        req_ids = [r["request_id"] for r in records if r["type"] == "request" and r["path"] == "/completion"]
        assert len(req_ids) == n
        assert len(set(req_ids)) == n, "Request IDs must be unique across requests"


def test_trace_error_response():
    """A malformed request still produces a request record followed by a response record."""
    global server
    with tempfile.TemporaryDirectory() as trace_dir:
        server.http_trace_dir = trace_dir
        server.start()

        # Send invalid JSON to trigger a 400 error
        res = server.make_request("POST", "/completion", data="not-json",
                                  headers={"Content-Type": "application/json"})
        assert res.status_code >= 400

        server.stop()

        records = read_trace_records(trace_dir)
        error_reqs = [r for r in records if r["type"] == "request" and r["path"] == "/completion"]
        assert len(error_reqs) >= 1, "Request record must be written even for failed requests"

        req_id = error_reqs[0]["request_id"]
        resp_records = [r for r in records_for(req_id, records) if r["type"] == "response"]
        assert len(resp_records) == 1
        assert resp_records[0]["status"] >= 400


def test_trace_disabled_by_default():
    """Without --http-trace-dir no trace files are produced."""
    global server
    # server.http_trace_dir is None — flag is not passed
    with tempfile.TemporaryDirectory() as trace_dir:
        # Start server normally (no trace flag)
        server.start()

        server.make_request("POST", "/completion", data={
            "prompt": "Hello",
            "n_predict": 3,
        })

        server.stop()

        # Nothing should be written to trace_dir since we never passed the flag
        files = [f for f in os.listdir(trace_dir) if f.startswith("trace-")]
        assert len(files) == 0, "Trace files must not be created when flag is absent"


def test_trace_headers_redaction():
    """Authorization header value must be redacted in the request record."""
    global server
    with tempfile.TemporaryDirectory() as trace_dir:
        server.http_trace_dir = trace_dir
        server.api_key = "supersecretkey"
        server.start()

        res = server.make_request("POST", "/completion",
                                  data={"prompt": "Hi", "n_predict": 3, "stream": False},
                                  headers={"Authorization": "Bearer supersecretkey"})
        assert res.status_code == 200

        server.stop()

        records = read_trace_records(trace_dir)
        req = next(r for r in records if r["type"] == "request" and r["path"] == "/completion")

        auth = req.get("headers", {}).get("authorization", "")
        assert "supersecretkey" not in auth, "Raw API key must not appear in trace logs"
        assert auth == "[REDACTED]"
