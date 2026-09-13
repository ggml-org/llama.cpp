import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_metrics = True


def fetch_metrics(server: ServerProcess) -> str:
    """get /metrics as raw prometheus text"""
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    assert "Process-Start-Time-Unix" in res.headers
    assert isinstance(res.body, str)
    return res.body


def parse_metrics(text: str) -> dict:
    """parse the prometheus text format into {name: (type, value)}"""
    out = {}
    types = {}
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split(" ", 3)
            types[name] = kind
        elif line.startswith("llamacpp:") and "{" not in line:
            name, value = line.split(" ", 1)
            assert name in types, f"{name} has no # TYPE line"
            out[name] = (types[name], float(value))
    return out


def test_metrics_disabled():
    global server
    server.server_metrics = False
    server.start()
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 501  # ERROR_TYPE_NOT_SUPPORTED


def test_metrics_prometheus_format():
    global server
    server.start()
    server.make_request("POST", "/completion", data={"prompt": "I believe", "n_predict": 8})

    text = fetch_metrics(server)
    metrics = parse_metrics(text)

    expected_counters = [
        "llamacpp:prompt_tokens_total",
        "llamacpp:prompt_tokens_cached_total",
        "llamacpp:prompt_seconds_total",
        "llamacpp:tokens_predicted_total",
        "llamacpp:tokens_predicted_seconds_total",
        "llamacpp:n_decode_total",
        "llamacpp:n_tokens_max",
        "llamacpp:spec_decode_num_draft_tokens_total",
        "llamacpp:spec_decode_num_accepted_tokens_total",
        "llamacpp:spec_decode_num_drafts_total",
    ]
    expected_gauges = [
        "llamacpp:prompt_tokens_seconds",
        "llamacpp:predicted_tokens_seconds",
        "llamacpp:requests_processing",
        "llamacpp:requests_deferred",
        "llamacpp:n_busy_slots_per_decode",
    ]

    for name in expected_counters:
        assert metrics[name][0] == "counter"
    for name in expected_gauges:
        assert metrics[name][0] == "gauge"

    # every metric must carry a help line
    for name in expected_counters + expected_gauges:
        assert f"# HELP {name} " in text

    assert metrics["llamacpp:n_decode_total"][1] > 0
    assert metrics["llamacpp:requests_processing"][1] == 0


def test_metrics_prompt_processed_and_cached():
    global server
    server.n_slots = 1  # keep the prompt cache on a single slot
    server.start()

    prompt = "the quick brown fox jumps over the lazy dog"

    n_processed = 0
    n_cached = 0
    for _ in range(2):
        res = server.make_request("POST", "/completion", data={"prompt": prompt, "n_predict": 4})
        assert res.status_code == 200
        n_processed += res.body["timings"]["prompt_n"]
        n_cached += res.body["timings"]["cache_n"]

    # the second request must reuse the prompt of the first one
    assert n_cached > 0

    metrics = parse_metrics(fetch_metrics(server))

    # cached tokens are counted apart, they cost no decode
    assert metrics["llamacpp:prompt_tokens_total"][1] == n_processed
    assert metrics["llamacpp:prompt_tokens_cached_total"][1] == n_cached


def test_metrics_predicted_total_matches_requests():
    global server
    server.start()

    n_predicted = 0
    for n_predict in [1, 4, 16]:
        res = server.make_request("POST", "/completion", data={"prompt": "I believe", "n_predict": n_predict})
        assert res.status_code == 200
        n_predicted += res.body["timings"]["predicted_n"]

    metrics = parse_metrics(fetch_metrics(server))
    assert metrics["llamacpp:tokens_predicted_total"][1] == n_predicted


def test_metrics_generation_rate_excludes_first_token():
    global server
    server.start()

    # the first token comes from the logits of the last prompt batch, so it costs no decode step
    res = server.make_request("POST", "/completion", data={"prompt": "I believe", "n_predict": 1})
    timings = res.body["timings"]
    assert timings["predicted_n"] == 1
    assert timings["predicted_per_second"] == 0.0
    assert timings["predicted_per_token_ms"] == 0.0

    res = server.make_request("POST", "/completion", data={"prompt": "I believe", "n_predict": 16})
    timings = res.body["timings"]
    assert timings["predicted_n"] == 16
    # the rate is over 15 decode steps, not 16 tokens
    expected = 1e3 / timings["predicted_ms"] * 15
    assert abs(timings["predicted_per_second"] - expected) < 1e-6


@pytest.mark.parametrize("n_predict", [1, 8])
def test_metrics_timings_are_finite(n_predict: int):
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": "I believe", "n_predict": n_predict})
    timings = res.body["timings"]

    # a null here means the server produced inf or nan
    for key, value in timings.items():
        assert value is not None, f"{key} is null"
        assert value >= 0, f"{key} is negative"

    assert timings["prompt_ms"] > 0
    assert timings["prompt_per_token_ms"] > 0


def test_metrics_timings_on_prompt_progress():
    global server
    server.start()

    # a long prompt so that it is split over several batches (n_batch = 32)
    prompt = "the quick brown fox jumps over the lazy dog " * 8
    chunks = list(server.make_stream_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": 4,
        "stream": True,
        "timings_per_token": True,
        "return_progress": True,
    }))

    progress = [c for c in chunks if "prompt_progress" in c]
    assert len(progress) > 1  # the prompt did not fit in a single batch

    # the very first update is sent before any prompt token is decoded
    first = progress[0]["timings"]
    assert first["prompt_n"] == 0
    assert first["prompt_ms"] == 0.0
    assert first["predicted_n"] == 0
    assert first["predicted_ms"] == 0.0

    # timings must never go backwards, nor report bogus values
    prompt_ms = 0.0
    for chunk in progress:
        timings = chunk["timings"]
        for key, value in timings.items():
            assert value is not None, f"{key} is null"
            assert value >= 0, f"{key} is negative"
        assert timings["prompt_ms"] >= prompt_ms
        prompt_ms = timings["prompt_ms"]

    assert prompt_ms > 0


def test_metrics_slots_idle_after_completion():
    global server
    server.server_slots = True
    server.start()
    server.make_request("POST", "/completion", data={"prompt": "I believe", "n_predict": 8})

    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    for slot in res.body:
        assert slot["is_processing"] is False
        if "next_token" in slot:
            # the budget of the finished task must not leak into the idle slot
            assert slot["next_token"][0]["n_remain"] == -1
            assert slot["next_token"][0]["n_decoded"] == 0


def test_metrics_embedding_prompt_is_counted():
    global server
    server = ServerPreset.bert_bge_small()
    server.server_metrics = True
    server.start()

    res = server.make_request("POST", "/v1/embeddings", data={"input": ["hello world", "goodbye world"]})
    assert res.status_code == 200

    # embedding tasks never sample a token, but their prompt still costs a decode
    metrics = parse_metrics(fetch_metrics(server))
    assert metrics["llamacpp:prompt_tokens_total"][1] > 0
    assert metrics["llamacpp:n_decode_total"][1] > 0
    assert metrics["llamacpp:tokens_predicted_total"][1] == 0
