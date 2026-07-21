import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 512
    server.n_slots = 1
    server.server_metrics = True  # appends --metrics (utils.py:192)
    # To run OFFLINE: uncomment and point at a local gguf:
    # server.model_hf_repo = None
    # server.model_hf_file = None
    # server.model_file = "/path/to/local.gguf"


def _metrics_text():
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    return res.body if isinstance(res.body, str) else res.body.decode()


def test_kv_cache_bytes_and_type_present():
    global server
    server.start()
    text = _metrics_text()
    assert "llamacpp:kv_cache_k_bytes" in text
    assert "llamacpp:kv_cache_v_bytes" in text
    assert 'llamacpp:kv_cache_type{' in text and 'cache="k"' in text


def test_histograms_have_bucket_sum_count():
    global server
    server.start()
    server.make_request("POST", "/completion", data={"prompt": "hello world", "n_predict": 8})
    text = _metrics_text()
    for name in ["prompt_tokens_size", "time_to_first_token_seconds", "generation_latency_seconds"]:
        assert f'llamacpp:{name}_bucket{{' in text, name
        assert f'llamacpp:{name}_sum' in text, name
        assert f'llamacpp:{name}_count' in text, name
    # +Inf bucket must equal _count for prompt_tokens_size
    inf_lines = [l for l in text.splitlines() if l.startswith("llamacpp:prompt_tokens_size_bucket") and 'le="+Inf"' in l]
    count_lines = [l for l in text.splitlines() if l.startswith("llamacpp:prompt_tokens_size_count")]
    assert inf_lines and count_lines
    assert inf_lines[0].split()[-1] == count_lines[0].split()[-1]


def test_vram_gauges_present_when_gpu():
    global server
    server.start()
    text = _metrics_text()
    # VRAM series only emitted when a GPU backend is present; assert well-formed if so.
    if "llamacpp:vram_total_bytes" in text:
        assert "llamacpp:vram_free_bytes" in text
        assert 'device="' in text
