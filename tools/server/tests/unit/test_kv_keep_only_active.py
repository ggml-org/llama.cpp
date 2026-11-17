import os
import random
import tempfile
import pytest
from utils import *

server = ServerPreset.tinyllama2()

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

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.kv_unified = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom."
)


# idle slot cleared on launch should restore from cache-ram
def test_clear_and_restore():
    global server
    server.start()
    log = LogReader(server.log_path)

    # verify feature is enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    # Slot 0 is the only slot with KV — should NOT be cleared
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()

    # Launching slot 1 clears idle slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.drain()

    # Re-send same prompt — should restore from cache-ram
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "updating prompt cache" in log.drain()
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n

    # Follow-up — slot 0 kept its KV, no clearing needed
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT + " The knight finally reached the castle gates.",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


def test_disabled_with_flag():
    global server
    server.no_cache_idle_slots = True
    server.start()
    log = LogReader(server.log_path)

    # Feature should not be enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" not in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # Request on different slot — should NOT trigger clearing
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


# with --no-cache-idle-slots, a restore that does not fit frees idle slots (each saved first) and stays within --cache-ram
def test_no_cache_idle_slots_restore_after_purge():
    global server
    server.no_cache_idle_slots = True
    server.n_slots = 3
    server.n_ctx = 1024
    server.n_predict = 1
    server.cache_ram = 1
    server.start()
    log = LogReader(server.log_path)

    rng = random.Random(7)

    def tokens(n):
        return [rng.randrange(10, 500) for _ in range(n)]

    def complete(prompt, id_slot=-1):
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt,
            "id_slot": id_slot,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        return res.body["timings"]["cache_n"]

    # A, B, C and D pass through slot 0, so the RAM cache holds A, B and C (about 0.95 MiB)
    a, b, c, d = tokens(500), tokens(500), tokens(500), tokens(500)
    for prompt in (a, b, c, d):
        complete(prompt, 0)

    # slot 1 takes 300 cells, so A does not fit next to D
    complete(tokens(300), 1)
    log.drain()

    # A goes to the empty slot 2: slot 0 is saved and cleared, and B is evicted because A still counts in RAM
    assert complete(a + tokens(10)) >= 500
    out = log.drain()
    assert "after purging 1 idle slots" in out
    assert "removing oldest entry" in out

    # D was saved before its cells were freed, B was evicted
    assert complete(d + tokens(10)) >= 500
    assert complete(b + tokens(10)) == 0
