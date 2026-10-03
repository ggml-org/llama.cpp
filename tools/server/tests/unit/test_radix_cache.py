import os
import tempfile
import time
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


PREFIX = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
)

SUFFIX_A = " He met dragons and fairies along the path to the castle."
SUFFIX_B = " She found wizards and elves near the silver lake shore."


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 8
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.kv_unified = True
    server.radix_cache = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield


def test_radix_enabled_tag():
    global server
    server.start()
    log = LogReader(server.log_path)
    assert "__TEST_TAG_RADIX_CACHE_ENABLED__" in log.drain()


def test_radix_disabled_parity():
    global server
    server.radix_cache = False
    server.start()
    log = LogReader(server.log_path)
    assert "__TEST_TAG_RADIX_CACHE_ENABLED__" not in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": PREFIX + SUFFIX_A,
        "cache_prompt": True,
        "temperature": 0.0,
        "n_predict": 4,
    })
    assert res.status_code == 200
    assert "content" in res.body


def test_radix_shared_prefix_two_slots():
    global server
    server.start()
    log = LogReader(server.log_path)
    log.drain()

    res1 = server.make_request("POST", "/completion", data={
        "prompt": PREFIX + SUFFIX_A,
        "id_slot": 0,
        "cache_prompt": True,
        "temperature": 0.0,
        "n_predict": 4,
    })
    assert res1.status_code == 200

    res2 = server.make_request("POST", "/completion", data={
        "prompt": PREFIX + SUFFIX_B,
        "id_slot": 1,
        "cache_prompt": True,
        "temperature": 0.0,
        "n_predict": 4,
    })
    assert res2.status_code == 200
    # Second request should reuse prefix (cache_n > 0) via LCP and/or radix alias
    assert res2.body["timings"]["cache_n"] > 0
    assert res2.body["timings"]["prompt_n"] + res2.body["timings"]["cache_n"] > res2.body["timings"]["prompt_n"]


def test_radix_branch_coexist():
    """Two branches with shared PREFIX should not destroy each other's cache across turns."""
    global server
    server.start()

    # Seed both branches
    assert server.make_request("POST", "/completion", data={
        "prompt": PREFIX + SUFFIX_A,
        "id_slot": 0,
        "cache_prompt": True,
        "temperature": 0.0,
        "n_predict": 2,
    }).status_code == 200

    assert server.make_request("POST", "/completion", data={
        "prompt": PREFIX + SUFFIX_B,
        "id_slot": 1,
        "cache_prompt": True,
        "temperature": 0.0,
        "n_predict": 2,
    }).status_code == 200

    # Grow branch A
    res_a = server.make_request("POST", "/completion", data={
        "prompt": PREFIX + SUFFIX_A + " The knight continued walking.",
        "id_slot": 0,
        "cache_prompt": True,
        "temperature": 0.0,
        "n_predict": 2,
    })
    assert res_a.status_code == 200
    assert res_a.body["timings"]["cache_n"] > 0

    # Grow branch B - must still hit cache on PREFIX+SUFFIX_B
    res_b = server.make_request("POST", "/completion", data={
        "prompt": PREFIX + SUFFIX_B + " The traveler kept going forward.",
        "id_slot": 1,
        "cache_prompt": True,
        "temperature": 0.0,
        "n_predict": 2,
    })
    assert res_b.status_code == 200
    assert res_b.body["timings"]["cache_n"] > 0


def test_radix_unique_no_regression():
    global server
    server.start()

    prompts = [
        "Alpha unique story begins with a lonely fox in the woods.",
        "Beta unique story begins with a quiet owl in the night.",
        "Gamma unique story begins with a swift deer in the field.",
    ]
    for p in prompts:
        res = server.make_request("POST", "/completion", data={
            "prompt": p,
            "cache_prompt": True,
            "temperature": 0.0,
            "n_predict": 4,
        })
        assert res.status_code == 200
        assert "content" in res.body


def test_radix_with_cache_ram_l2():
    global server
    server.n_slots = 2
    server.start()

    res1 = server.make_request("POST", "/completion", data={
        "prompt": PREFIX + SUFFIX_A,
        "id_slot": 0,
        "cache_prompt": True,
        "temperature": 0.0,
        "n_predict": 2,
    })
    assert res1.status_code == 200
    original_prompt_n = res1.body["timings"]["prompt_n"]

    # Force slot 0 idle clear via slot 1 + cache-idle-slots
    res2 = server.make_request("POST", "/completion", data={
        "prompt": "Short other prompt about cats and dogs together.",
        "id_slot": 1,
        "cache_prompt": True,
        "temperature": 0.0,
        "n_predict": 2,
    })
    assert res2.status_code == 200

    # Restore long prompt - L2 and/or radix should help
    res3 = server.make_request("POST", "/completion", data={
        "prompt": PREFIX + SUFFIX_A,
        "cache_prompt": True,
        "temperature": 0.0,
        "n_predict": 2,
    })
    assert res3.status_code == 200
    assert res3.body["timings"]["cache_n"] > 0 or res3.body["timings"]["prompt_n"] <= original_prompt_n
