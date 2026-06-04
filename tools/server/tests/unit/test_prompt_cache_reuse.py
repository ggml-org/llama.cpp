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
        with open(self.path, errors="ignore") as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    # single slot: a conversation that is not currently loaded lives ONLY
    # in the prompt cache, which is what exposes the consume-on-load bug.
    server.n_slots = 1
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    # force every slot reuse through the prompt cache (save + load) instead
    # of the in-place LCP-similarity reuse, so load() is actually exercised.
    server.slot_prompt_similarity = 0
    # isolate the cache save/load to get_available_slot (the idle-slot
    # clearing path would add another save and muddy the scenario).
    server.no_cache_idle_slots = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


# A and C share a long common prefix (so a request for C matches A's cached
# entry, f_keep >= 0.25), then diverge. B is unrelated to both.
COMMON_AC = (
    "Once upon a time in a quiet village by the sea there lived an old "
    "fisherman who every morning rowed his small wooden boat out past the "
    "harbour wall to cast his nets beneath the pale light of the rising sun."
)
CONV_A = COMMON_AC + (
    " On this particular day he caught a silver fish that spoke to him and "
    "promised three wishes in exchange for its freedom and a safe return home."
)
CONV_C = COMMON_AC + (
    " But the storm clouds gathered quickly that afternoon and the waves grew "
    "tall and angry as the wind tore the sails and scattered the frightened gulls."
)
CONV_B = (
    "In a bustling city far inland a young clockmaker tinkered late into the "
    "night with brass gears and tiny springs trying to build a machine that "
    "could measure not the hours but the quiet weight of a person's memories."
)
# A continuation of A (strict superset). Used for A's return so that
# n_past < task tokens and we avoid the identical-prompt path that, on
# SWA / hybrid / recurrent models, cannot partially remove the final
# token and would reset regardless of caching.
CONV_A_CONT = CONV_A + (
    " The fisherman closed his eyes and made his first wish very carefully."
)


def _total_prompt_tokens(res):
    t = res.body["timings"]
    return t["prompt_n"] + t["cache_n"]


# A prompt-cache entry must survive being matched by a DIFFERENT conversation.
# Regression test for load() consuming (erasing) the matched entry: with one
# slot and three conversations, conversation A lives only in the cache while
# conversation C — which shares A's prefix — is loaded. C's load must not
# destroy A's entry, otherwise A pays a full re-prefill when it returns.
def test_cache_entry_survives_cross_conversation_load():
    global server
    server.start()
    log = LogReader(server.log_path)

    # 1) Conversation A, cold. Capture its full token length.
    res_a1 = server.make_request("POST", "/completion", data={
        "prompt": CONV_A,
        "cache_prompt": True,
    })
    assert res_a1.status_code == 200
    assert res_a1.body["timings"]["cache_n"] == 0  # nothing cached yet
    n_tokens_a = _total_prompt_tokens(res_a1)

    # 2) Conversation B (unrelated). Selecting the slot saves A into the
    #    cache; B does not match A, so A is parked in the cache untouched.
    res_b = server.make_request("POST", "/completion", data={
        "prompt": CONV_B,
        "cache_prompt": True,
    })
    assert res_b.status_code == 200
    assert "updating prompt cache" in log.drain()

    # 3) Conversation C, which shares A's long prefix. Selecting the slot
    #    saves B; loading C matches A's cached entry (f_keep >= 0.25). The
    #    buggy behaviour erased A here; the fix keeps it.
    res_c = server.make_request("POST", "/completion", data={
        "prompt": CONV_C,
        "cache_prompt": True,
    })
    assert res_c.status_code == 200
    assert res_c.body["timings"]["cache_n"] > 0  # C reused A's shared prefix

    # 4) Conversation A returns (as a strict superset, so n_past < task
    #    tokens). It was only in the cache. With the fix its entry survived
    #    step 3, so all of A is reused and only the new continuation is
    #    processed. Without the fix A's entry was consumed in step 3 and
    #    only the prefix A shares with C can be reused.
    res_a2 = server.make_request("POST", "/completion", data={
        "prompt": CONV_A_CONT,
        "cache_prompt": True,
    })
    assert res_a2.status_code == 200
    cache_n_a2 = res_a2.body["timings"]["cache_n"]

    # The full original A prompt is reused from cache. Under the bug this
    # would only be the A/C shared prefix, which is well below n_tokens_a.
    assert cache_n_a2 >= n_tokens_a - 2
