import glob
import os
import shutil
import tempfile

import pytest
from utils import *

server = ServerPreset.tinyllama2()

cache_dir: str = ""

TAG_DISK_LOAD = "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__"

# longer than a few hash blocks (64 tokens each), the token ids fit the small vocab of the test model
PROMPT_A = [10 + i for i in range(200)]
PROMPT_B = [300 + i for i in range(200)]


def read_log() -> str:
    with open(server.log_path) as f:
        return f.read()


def cache_files() -> list[str]:
    return sorted(glob.glob(os.path.join(cache_dir, "*.kvc")))


def process_prompt(prompt, **kwargs):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": 0,
        "cache_prompt": True,
        **kwargs,
    })
    assert res.status_code == 200
    return res.body["timings"]


def create_server_preset(preset):
    global server, cache_dir
    cache_dir = tempfile.mkdtemp(prefix="llama_cache_disk_")
    server = preset
    server.n_slots = 1
    server.n_ctx = 512
    server.temperature = 0.0
    server.debug = True
    server.cache_disk = cache_dir
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)


@pytest.fixture(autouse=True)
def create_server():
    create_server_preset(ServerPreset.tinyllama2())
    yield
    shutil.rmtree(cache_dir, ignore_errors=True)


def test_cache_disk_restart():
    server.start()
    process_prompt(PROMPT_A)
    assert len(cache_files()) == 0

    # the state of the idle slot is saved on shutdown
    server.stop()
    assert len(cache_files()) == 1

    server.start()
    timings = process_prompt(PROMPT_A + [7, 8, 9])
    assert TAG_DISK_LOAD in read_log()
    assert timings["cache_n"] == len(PROMPT_A)
    assert timings["prompt_n"] == 3

    # the longer prompt replaces its prefix on disk
    server.stop()
    assert len(cache_files()) == 1


def test_cache_disk_partial_match():
    server.start()
    process_prompt(PROMPT_A)
    server.stop()

    # the prompts differ in the middle of a hash block
    server.start()
    timings = process_prompt(PROMPT_A[:150] + PROMPT_B[:50])
    assert TAG_DISK_LOAD in read_log()
    assert timings["cache_n"] == 150
    assert timings["prompt_n"] == 50

    server.stop()
    assert len(cache_files()) == 2


def test_cache_disk_no_match():
    server.start()
    process_prompt(PROMPT_A)
    server.stop()

    server.start()
    timings = process_prompt(PROMPT_B)
    assert TAG_DISK_LOAD not in read_log()
    assert timings["cache_n"] == 0


def test_cache_disk_ram_first():
    server.start()
    process_prompt(PROMPT_A)

    # PROMPT_A goes to the RAM cache, nothing is written while the server runs
    process_prompt(PROMPT_B)
    assert len(cache_files()) == 0

    timings = process_prompt(PROMPT_A)
    assert TAG_DISK_LOAD not in read_log()
    assert timings["cache_n"] > 0


def test_cache_disk_ram_eviction():
    # the RAM cache has room for one of the two states
    server.cache_ram = 1
    server.n_ctx = 2048
    server.start()
    process_prompt(PROMPT_A * 5)
    process_prompt(PROMPT_B * 5)
    process_prompt(PROMPT_A)
    assert len(cache_files()) >= 1


def test_cache_disk_broken_file():
    server.start()
    process_prompt(PROMPT_A)
    server.stop()

    files = cache_files()
    assert len(files) == 1
    with open(files[0], "r+b") as f:
        f.truncate(os.path.getsize(files[0]) - 16)

    server.start()
    timings = process_prompt(PROMPT_A)
    assert timings["cache_n"] == 0
    assert files[0] not in cache_files()


def test_cache_disk_lora():
    create_server_preset(ServerPreset.stories15m_moe())
    server.lora_files = [download_file("https://huggingface.co/ggml-org-dev/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf")]
    server.start()

    # a state made with another lora config is not saved
    process_prompt(PROMPT_A, lora=[{"id": 0, "scale": 0.5}])
    server.stop()
    assert len(cache_files()) == 0

    server.start()
    process_prompt(PROMPT_A)
    server.stop()
    assert len(cache_files()) == 1
