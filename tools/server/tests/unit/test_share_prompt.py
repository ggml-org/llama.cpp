import os
import tempfile
import pytest
from utils import *

server = ServerPreset.tinyllama2()

TAG_WAIT = "__TEST_TAG_WAIT_DONOR_SLOT__"
TAG_COPY = "__TEST_TAG_SHARE_PROMPT__"

N_SAME = 200  # tokens shared by the prompts
N_OWN  = 40   # tokens of each prompt after the shared part


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
    server.n_slots = 3
    server.n_ctx = 2048
    server.n_predict = 8
    server.temperature = 0.0
    server.kv_unified = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


# the model has 512 tokens, the first n_same tokens do not depend on the seed
def make_prompt(seed: int, n_same: int = N_SAME, n_own: int = N_OWN):
    same = [10 + (i * 7) % 200 for i in range(n_same)]
    own  = [300 + (i * seed + seed) % 200 for i in range(n_own)]
    return same + own


# the prompts of one request are all given to the slots before the first batch, so the order is fixed
def completions(prompts, **kwargs):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompts,
        "cache_prompt": True,
        **kwargs,
    })
    assert res.status_code == 200
    return res.body


def completion(prompt, **kwargs):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "cache_prompt": True,
        **kwargs,
    })
    assert res.status_code == 200
    return res.body


# the answer without any cache
def reference(prompt):
    return completion(prompt, cache_prompt=False)["content"]


def test_share_prompt():
    global server
    server.start()
    log = LogReader(server.log_path)
    assert "prompt sharing between slots is enabled" in log.drain()

    # the first prompt is processed, the 2 others wait for it and copy the shared part
    prompts = [make_prompt(3), make_prompt(5), make_prompt(7)]
    res = completions(prompts)

    content = log.drain()
    assert content.count(TAG_WAIT) == 2
    assert content.count(TAG_COPY) == 2

    assert res[0]["timings"]["cache_n"] == 0
    for i in range(1, 3):
        assert res[i]["timings"]["cache_n"] == N_SAME
        assert res[i]["timings"]["prompt_n"] == N_OWN

    # the copied state gives the same answers
    for i in range(3):
        assert res[i]["content"] == reference(prompts[i])


def test_no_share():
    global server
    server.start()
    log = LogReader(server.log_path)

    # the shared part is too short
    prompts = [make_prompt(3, n_same=40), make_prompt(5, n_same=40)]
    res = completions(prompts)
    assert res[1]["timings"]["cache_n"] == 0

    # the prompts are not cached
    completions([make_prompt(7), make_prompt(9)], cache_prompt=False)

    content = log.drain()
    assert TAG_WAIT not in content
    assert TAG_COPY not in content


def test_cache_reuse_after_copy():
    global server
    server.no_cache_idle_slots = True  # the idle slots keep their cells
    server.server_slots = True
    server.start()
    log = LogReader(server.log_path)

    # slot 0 has the first prompt, slot 1 copies the shared part
    prompts = [make_prompt(3), make_prompt(5)]
    res = completions(prompts)
    assert TAG_COPY in log.drain()

    # the same prompt without its first tokens: the cells could be shifted, but they belong to slot 0 too
    prompt_c = prompts[1][50:]
    res_c = completion(prompt_c, id_slot=1, n_cache_reuse=16)
    assert "cache reuse is not supported for a prompt shared with another slot" in log.drain()
    assert res_c["content"] == reference(prompt_c)

    # slot 0 still has its prompt intact
    res_a = completion(prompts[0], id_slot=0)
    assert res_a["timings"]["cache_n"] == N_SAME + N_OWN - 1
    assert res_a["content"] == res[0]["content"]


@pytest.mark.parametrize("kv_unified,ctx_shift,reason", [
    (False, False, "it needs --kv-unified"),
    (True,  True,  "not compatible with --context-shift"),
])
def test_disabled(kv_unified: bool, ctx_shift: bool, reason: str):
    global server
    server.kv_unified = kv_unified
    server.enable_ctx_shift = ctx_shift
    server.start()
    log = LogReader(server.log_path)
    assert f"prompt sharing between slots is disabled: {reason}" in log.drain()

    prompts = [make_prompt(3), make_prompt(5)]
    res = completions(prompts)

    content = log.drain()
    assert TAG_WAIT not in content
    assert TAG_COPY not in content
    assert res[1]["timings"]["cache_n"] == 0
    assert res[1]["content"] == reference(prompts[1])
