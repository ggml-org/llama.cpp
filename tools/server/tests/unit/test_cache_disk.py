import base64
import glob
import os
import shutil
import tempfile
import time

import pytest
import requests
from utils import *

server = ServerPreset.tinyllama2()

cache_dir: str = ""


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
    def wait_for(self, tag, timeout=10) -> bool:
        # the server log is pumped to the file asynchronously - poll for the tag
        deadline = time.time() + timeout
        while time.time() < deadline:
            if tag in self.drain():
                return True
            time.sleep(0.25)
        return False


def kvc_files() -> list[str]:
    return sorted(glob.glob(os.path.join(cache_dir, "*.kvc")))


@pytest.fixture(autouse=True)
def create_server():
    global server, cache_dir
    cache_dir = tempfile.mkdtemp(prefix="llama_cache_disk_")
    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.temperature = 0.0
    server.debug = True
    server.cache_disk = cache_dir
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield
    shutil.rmtree(cache_dir, ignore_errors=True)


PROMPT_A = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers."
)

PROMPT_B = "The quick brown fox jumps over the lazy dog."


def make_prompt_request(prompt, n_predict=0):
    global server
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": n_predict,  # 0 = evaluate the prompt into the KV cache only
        "cache_prompt": True,
    })
    assert res.status_code == 200
    return res


def test_write_through_and_restart_hit():
    global server
    server.cache_disk_write_through = True
    server.start()
    log = LogReader(server.log_path)

    res = make_prompt_request(PROMPT_A)
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert prompt_n_full > 0

    # nothing is written while the prompt is still live in the slot
    assert len(kvc_files()) == 0

    # a different prompt takes over the only slot - the previous one is saved
    # to the RAM cache and, in write-through mode, to disk immediately
    make_prompt_request(PROMPT_B)
    assert log.wait_for("__TEST_TAG_CACHE_DISK_STORE__")
    assert len(kvc_files()) == 1

    # the state must survive a full server restart
    server.stop()
    server.start()
    log = LogReader(server.log_path)

    res = make_prompt_request(PROMPT_A)
    assert log.wait_for("__TEST_TAG_CACHE_DISK_HIT__")
    assert res.body["timings"]["prompt_n"] == 1  # only the last token is re-evaluated
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1


def test_spill_on_shutdown_flush():
    global server
    server.start()
    log = LogReader(server.log_path)

    make_prompt_request(PROMPT_A)
    make_prompt_request(PROMPT_B)  # forces PROMPT_A into the RAM cache

    # without write-through, nothing reaches the disk while running
    time.sleep(0.5)
    assert "__TEST_TAG_CACHE_DISK_STORE__" not in log.drain()
    assert len(kvc_files()) == 0

    # a graceful shutdown flushes the RAM cache entries to disk
    server.stop()
    assert len(kvc_files()) == 1

    server.start()
    log = LogReader(server.log_path)

    res = make_prompt_request(PROMPT_A)
    assert log.wait_for("__TEST_TAG_CACHE_DISK_HIT__")
    assert res.body["timings"]["prompt_n"] == 1


def test_ram_cache_hit_takes_priority():
    global server
    server.cache_disk_write_through = True
    server.start()
    log = LogReader(server.log_path)

    make_prompt_request(PROMPT_A)
    make_prompt_request(PROMPT_B)
    assert len(kvc_files()) == 1

    # PROMPT_A is in both the RAM cache and on disk - the RAM copy must win
    # (the disk entry is never longer than the RAM one here)
    res = make_prompt_request(PROMPT_A)
    time.sleep(0.5)
    assert "__TEST_TAG_CACHE_DISK_HIT__" not in log.drain()
    assert res.body["timings"]["cache_n"] > 0


def test_budget_eviction():
    global server
    server.n_ctx = 2048
    server.n_batch = 512
    server.cache_disk_write_through = True
    server.cache_disk_limit = 1  # MiB
    server.start()

    # three long, distinct token-array prompts; each state is close to 1 MiB
    n_len = 1500
    for i in range(3):
        make_prompt_request([100 + i] * n_len)

    # one final small prompt to force the last long prompt out of the slot
    make_prompt_request(PROMPT_B)

    files = kvc_files()
    assert len(files) >= 1
    assert len(files) < 3  # the oldest entries were evicted

    # the budget is respected (a single over-budget file is allowed to remain)
    if len(files) > 1:
        assert sum(os.path.getsize(f) for f in files) <= 1024 * 1024


def test_corrupt_file_is_removed():
    global server
    server.cache_disk_write_through = True
    server.start()

    make_prompt_request(PROMPT_A)
    make_prompt_request(PROMPT_B)
    files = kvc_files()
    assert len(files) == 1

    server.stop()

    # corrupt the serialized token section (starts right after the 48-byte header)
    with open(files[0], "r+b") as f:
        f.seek(48 + 4)
        f.write(b"\xff\xff\xff\xff")

    server.start()
    log = LogReader(server.log_path)

    # the request must still succeed, with the prompt fully re-processed
    res = make_prompt_request(PROMPT_A)
    time.sleep(0.5)
    assert "__TEST_TAG_CACHE_DISK_HIT__" not in log.drain()
    assert res.body["timings"]["prompt_n"] > 1

    # the corrupt file was deleted
    assert len(kvc_files()) == 0


IMG_URL_CAT = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"


def _get_img_base64(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    return base64.b64encode(response.content).decode("utf-8")


@pytest.fixture
def mmproj_server():
    global cache_dir
    os.environ['LLAMA_MEDIA_MARKER'] = '<__media__>'
    mm_server = ServerPreset.tinygemma3()
    mm_server.n_slots = 1
    mm_server.temperature = 0.0
    mm_server.debug = True
    # use the full SWA cache so the restored image prefix can be reused
    mm_server.swa_full = True
    mm_server.cache_disk = cache_dir
    mm_server.cache_disk_write_through = True
    fd, mm_server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    return mm_server


def test_image_prompt_across_restart(mmproj_server):
    server = mmproj_server
    server.start()

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }

    res = server.make_request("POST", "/completions", data={
        "n_predict": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/completions", data={
        "n_predict": 0,
        "cache_prompt": True,
        "prompt": "The quick brown fox",
    })
    assert res.status_code == 200
    assert len(kvc_files()) == 1

    server.stop()
    server.start()
    log = LogReader(server.log_path)

    # the image KV must be restored from disk in the new process
    res = server.make_request("POST", "/completions", data={
        "n_predict": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert log.wait_for("__TEST_TAG_CACHE_DISK_HIT__")
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
