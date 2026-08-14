import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from utils import *


def start_rpc_server():
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

    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return rpc, port
        except OSError:
            if rpc.poll() is not None:
                pytest.fail("ggml-rpc-server exited during startup")
            time.sleep(0.01)

    rpc.terminate()
    rpc.wait(timeout=5)
    pytest.fail("ggml-rpc-server did not start")


def test_disaggregated_prefill_matches_local_and_preserves_cache():
    sentence = (
        "Once upon a time in a land far away, a traveler crossed the mountains "
        "to find a hidden library and asked the keeper for a story."
    )
    prompt = f"{sentence} {sentence}"
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

    rpc, port = start_rpc_server()
    server = ServerPreset.tinyllama2()
    server.prefill_nodes = [f"127.0.0.1:{port}"]
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)

    try:
        server.start()
        actual = server.make_request("POST", "/completion", data=request)
        multiple = server.make_request("POST", "/completion", data={
            **request,
            "n_predict": 2,
            "n_cmpl": 2,
        })
        cached = server.make_request("POST", "/completion", data={
            **request,
            "prompt": prompt + " Then it rested.",
            "n_predict": 1,
            "cache_prompt": True,
        })
        server.stop()

        assert actual.status_code == 200
        assert actual.body["content"] == expected.body["content"]
        assert actual.body["tokens"] == expected.body["tokens"]
        assert actual.body["timings"]["prompt_n"] == expected.body["timings"]["prompt_n"]
        assert actual.body["timings"]["prompt_n"] > 128
        assert multiple.status_code == 200
        assert len(multiple.body) == 2
        assert cached.status_code == 200
        assert cached.body["timings"]["cache_n"] > 1

        with open(server.log_path) as log:
            log_content = log.read()
            assert log_content.count("__TEST_TAG_PREFILL_RESTORED__") == 2
    finally:
        server.stop()
        rpc.terminate()
        rpc.wait(timeout=5)
        os.remove(server.log_path)
