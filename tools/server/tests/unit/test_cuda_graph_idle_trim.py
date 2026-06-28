import os
import tempfile
import time

import pytest

from utils import *

server = ServerPreset.tinyllama2()
TINYLLAMA_MODEL_URL = "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"


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


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.model_file = download_file(TINYLLAMA_MODEL_URL)
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
    yield


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

    seen_trim_tag = False
    for _ in range(50):
        seen_trim_tag = seen_trim_tag or "__TEST_TAG_CUDA_GRAPH_TRIM__" in log.drain()
        if seen_trim_tag:
            break
        time.sleep(0.1)

    assert seen_trim_tag
