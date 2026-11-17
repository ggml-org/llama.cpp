import os
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
LONG_PROMPT_EXT = LONG_PROMPT + " The knight finally reached the castle gates."


def completion(prompt, cache_prompt, **extra):
    return server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "cache_prompt": cache_prompt,
        **extra,
    })


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


# the state of a cache_prompt = false task is not saved to cache-ram
def test_nocache_prompt_not_saved():
    global server
    server.start()
    log = LogReader(server.log_path)
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": False,
    })
    assert res.status_code == 200
    log.drain()

    # Launching slot 1 would save idle slot 0, but its state came from a cache_prompt = false task
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


# the state of a cache_prompt = false task is not saved when another task takes its slot
def test_nocache_prompt_not_saved_on_slot_reuse():
    global server
    server.n_slots = 1
    server.start()

    assert completion(LONG_PROMPT, False).status_code == 200
    assert completion("The quick brown fox", True).status_code == 200

    res = completion(LONG_PROMPT, True)
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] <= 1  # only BOS


# a cache_prompt = false task overwrites its slot, so the slot is saved first
def test_nocache_task_saves_replaced_slot():
    global server
    server.n_slots = 1
    server.start()

    res = completion(LONG_PROMPT, True)
    assert res.status_code == 200
    n_prompt = res.body["timings"]["prompt_n"]

    assert completion(LONG_PROMPT_EXT, False).status_code == 200
    assert completion("The quick brown fox", True).status_code == 200

    res = completion(LONG_PROMPT, True)
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > n_prompt // 2


# a cache_prompt = false task does not take a matching prompt out of cache-ram
def test_nocache_task_keeps_cached_prompt():
    global server
    server.n_slots = 1
    server.start()

    res = completion(LONG_PROMPT, True)
    assert res.status_code == 200
    n_prompt = res.body["timings"]["prompt_n"]

    assert completion("The quick brown fox", True).status_code == 200
    log = LogReader(server.log_path)
    log.drain()
    assert completion(LONG_PROMPT_EXT, False).status_code == 200
    assert "looking for better prompt" not in log.drain()
    assert completion("Hello there, how are you today?", True).status_code == 200

    res = completion(LONG_PROMPT, True)
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > n_prompt // 2


# a cache_prompt = false task that fails before it runs does not lose the slot state
def test_failed_nocache_task_keeps_slot_state():
    global server
    server.n_slots = 1
    server.start()

    res = completion(LONG_PROMPT, True)
    assert res.status_code == 200
    n_prompt = res.body["timings"]["prompt_n"]

    res = completion(LONG_PROMPT + " and then" * 400, False)
    assert res.status_code != 200
    assert "exceeds the available context size" in res.body["error"]["message"]
    assert completion("The quick brown fox", True).status_code == 200

    res = completion(LONG_PROMPT, True)
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > n_prompt // 2


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
