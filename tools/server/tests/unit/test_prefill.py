import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
import requests

from utils import *


def test_disaggregated_prefill_matches_local():
    prompt = (
        "Once upon a time in a land far away, a traveler crossed the mountains "
        "to find a hidden library and asked the keeper for a story."
    )
    request = {
        "prompt": prompt,
        "n_predict": 16,
        "seed": 42,
        "temperature": 0.0,
        "cache_prompt": False,
        "return_tokens": True,
    }

    local = ServerPreset.tinyllama2()
    local.start()
    expected = local.make_request("POST", "/completion", data=request)
    local.stop()

    prefill = ServerPreset.tinyllama2()
    prefill.prefill_devices = ["none"]
    prefill.prefill_min_tokens = 1
    prefill.debug = True
    fd, prefill.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)

    try:
        prefill.start()
        actual = prefill.make_request("POST", "/completion", data=request)
        prefill.stop()

        assert actual.status_code == 200
        assert actual.body["content"] == expected.body["content"]
        assert actual.body["tokens"] == expected.body["tokens"]
        assert actual.body["timings"]["prompt_n"] == expected.body["timings"]["prompt_n"]
        assert actual.body["timings"]["cache_n"] == 0

        with open(prefill.log_path) as log:
            log_content = log.read()
            assert "__TEST_TAG_PREFILL_RESTORED__" in log_content
            assert log_content.count("__TEST_TAG_PREFILL_RANGE__") > 1
    finally:
        prefill.stop()
        os.remove(prefill.log_path)


def test_disaggregated_prefill_preserves_cached_prefix():
    server = ServerPreset.tinyllama2()
    server.prefill_devices = ["none"]
    server.prefill_min_tokens = 1
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)

    prompt = "The quick brown fox jumps over the lazy dog. " * 5
    request = {
        "n_predict": 1,
        "seed": 42,
        "temperature": 0.0,
        "cache_prompt": True,
    }

    try:
        server.start()
        first = server.make_request("POST", "/completion", data={**request, "prompt": prompt})
        second = server.make_request("POST", "/completion", data={**request, "prompt": prompt + "Then it rests."})
        server.stop()

        assert first.status_code == 200
        assert second.status_code == 200
        assert second.body["timings"]["cache_n"] > 1
        assert second.body["timings"]["prompt_n"] < first.body["timings"]["prompt_n"]
        with open(server.log_path) as log:
            assert log.read().count("__TEST_TAG_PREFILL_RESTORED__") == 1
    finally:
        server.stop()
        os.remove(server.log_path)


def test_disaggregated_prefill_multiple_workers():
    server = ServerPreset.tinyllama2()
    server.prefill_devices = ["none", "none"]
    server.prefill_min_tokens = 1
    server.n_slots = 2
    server.start()

    request = {
        "n_predict": 1,
        "seed": 42,
        "temperature": 0.0,
        "cache_prompt": False,
    }
    prompts = [
        "Alpha beta gamma delta epsilon. " * 8,
        "One two three four five. " * 8,
    ]
    results = parallel_function_calls([
        (server.make_request, ("POST", "/completion", {**request, "prompt": prompt}))
        for prompt in prompts
    ])

    assert all(result is not None and result.status_code == 200 for result in results)


def test_disaggregated_prefill_cancellation():
    server = ServerPreset.tinyllama2()
    server.prefill_devices = ["none"]
    server.prefill_min_tokens = 1
    server.n_ctx = 4096
    server.n_slots = 1
    server.server_slots = True
    server.start()

    with pytest.raises(requests.exceptions.ReadTimeout):
        server.make_request("POST", "/completion", data={
            "prompt": "The quick brown fox jumps over the lazy dog. " * 40,
            "n_predict": 1,
            "cache_prompt": False,
        }, timeout=0.001)

    for _ in range(100):
        slots = server.make_request("GET", "/slots")
        if not slots.body[0]["is_processing"]:
            break
        time.sleep(0.01)
    assert not slots.body[0]["is_processing"]


def test_disaggregated_prefill_shared_rpc_endpoint():
    server_bin = Path(os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")).resolve()
    rpc_bin = Path(os.environ.get("GGML_RPC_SERVER_BIN_PATH", server_bin.with_name("ggml-rpc-server")))
    if not rpc_bin.is_file():
        pytest.skip("ggml-rpc-server is not built")

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    rpc = subprocess.Popen(
        [str(rpc_bin), "--host", "127.0.0.1", "--port", str(port), "--device", "CPU"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    server = None
    try:
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if rpc.poll() is not None:
                    pytest.fail("ggml-rpc-server exited during startup")
                time.sleep(0.01)
        else:
            pytest.fail("ggml-rpc-server did not start")

        server = ServerPreset.tinyllama2()
        server.rpc_servers = f"127.0.0.1:{port}"
        server.devices = "none"
        server.prefill_devices = ["RPC0", "RPC0"]
        server.prefill_min_tokens = 1
        server.n_slots = 2
        server.start()

        request = {
            "n_predict": 1,
            "seed": 42,
            "temperature": 0.0,
            "cache_prompt": False,
        }
        prompts = [
            "Alpha beta gamma delta epsilon. " * 8,
            "One two three four five. " * 8,
        ]
        results = parallel_function_calls([
            (server.make_request, ("POST", "/completion", {**request, "prompt": prompt}))
            for prompt in prompts
        ])

        assert all(result is not None and result.status_code == 200 for result in results)
        assert server.process is not None and server.process.poll() is None
    finally:
        if server is not None:
            server.stop()
        rpc.terminate()
        rpc.wait(timeout=5)
