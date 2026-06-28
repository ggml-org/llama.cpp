import os
import socket
import tempfile
import time

import pytest

from utils import *

server = ServerPreset.tinyllama2()
TINYLLAMA_MODEL_URL = "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"
STORIES15M_MOE_MODEL_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/stories15M_MOE-F16.gguf"
STORIES15M_DRAFT_MODEL_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0

    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


@pytest.fixture(scope="module", autouse=True)
def do_something():
    yield


def pick_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = make_target_only_server()
    yield

def configure_server(server):
    os.makedirs("./tmp", exist_ok=True)
    server.model_hf_repo = None
    server.model_hf_file = None
    server.offline = False
    server.server_port = pick_free_port()
    server.n_slots = 2
    server.n_predict = 32
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.kv_unified = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    return server

def make_target_only_server():
    server = ServerPreset.tinyllama2()
    server.model_file = download_file(TINYLLAMA_MODEL_URL)
    return configure_server(server)

def make_speculative_server():
    server = ServerPreset.stories15m_moe()
    server.model_file = download_file(STORIES15M_MOE_MODEL_URL)
    server.model_draft = download_file(STORIES15M_DRAFT_MODEL_URL)
    server.spec_draft_n_min = 4
    server.spec_draft_n_max = 8
    server.fa = "off"
    return configure_server(server)


LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom."
)


def test_trim_runs_only_after_all_slots_idle():
    global server
    server.start()
    log = LogReader(server.log_path)

    assert "__TEST_TAG_CUDA_GRAPH_TRIM__" not in log.drain()

    stream = server.make_stream_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "n_predict": 32,
        "stream": True,
    })

    first_chunk = next(stream)
    assert first_chunk["stop"] is False
    assert "__TEST_TAG_CUDA_GRAPH_TRIM__" not in log.drain()

    for _ in stream:
        pass

    assert wait_for_tags(log, "__TEST_TAG_CUDA_GRAPH_TRIM__")

def wait_for_tags(log, *tags):
    seen = set()
    for _ in range(50):
        content = log.drain()
        for tag in tags:
            if tag in content:
                seen.add(tag)
        if len(seen) == len(tags):
            return True
        time.sleep(0.1)
    return False

def test_trim_runs_for_draft_context_after_all_slots_idle(monkeypatch):
    global server
    monkeypatch.setenv("LLAMA_ARG_SPEC_TYPE", "draft-simple")
    server = make_speculative_server()
    server.start()
    log = LogReader(server.log_path)

    assert "__TEST_TAG_CUDA_GRAPH_TRIM__" not in log.drain()
    assert "__TEST_TAG_CUDA_GRAPH_TRIM_DRAFT__" not in log.drain()

    stream = server.make_stream_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "n_predict": 32,
        "stream": True,
        "temperature": 0.0,
        "top_k": 1,
    })

    first_chunk = next(stream)
    assert first_chunk["stop"] is False
    assert "__TEST_TAG_CUDA_GRAPH_TRIM__" not in log.drain()
    assert "__TEST_TAG_CUDA_GRAPH_TRIM_DRAFT__" not in log.drain()

    for _ in stream:
        pass

    assert wait_for_tags(
        log,
        "__TEST_TAG_CUDA_GRAPH_TRIM__",
        "__TEST_TAG_CUDA_GRAPH_TRIM_DRAFT__",
    )
