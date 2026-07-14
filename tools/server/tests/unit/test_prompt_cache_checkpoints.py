import pytest
from utils import *


@pytest.fixture
def server():
    srv = ServerPreset.tinygemma3()
    # tinygemma3 uses SWA (n_swa > 0), which is required for the server to
    # ever create context checkpoints in the first place; a plain model like
    # tinyllama2 supports arbitrary partial KV removal and never checkpoints
    srv.n_ctx = 1024
    srv.n_batch = 16
    srv.n_ubatch = 16
    srv.n_slots = 1
    srv.n_predict = 1
    srv.temperature = 0.0
    srv.server_slots = True
    srv.cache_ram = 64
    srv.n_ctx_checkpoints = 8
    srv.checkpoint_min_step = 4
    # Interesting log entries are logged at level 4
    srv.log_verbosity = 4
    return srv


# long enough that prompt processing spans several batches and at least one
# context checkpoint gets created near the end
PROMPT_A = (
    "The brave knight traveled across the mountains and rivers and forests "
    "looking for treasure in the old castle ruins near the lake and the "
    "dragon guarding it fiercely every single day and night"
)

# shares no meaningful content with PROMPT_A, forcing a genuine prompt-cache miss
PROMPT_B = (
    "Quantum electron spin lattice diffraction pattern anomaly detector "
    "calibration sequence initiated for laboratory experiment number seven"
)


# Trigger full prompt re-processing
def test_prompt_cache_load_miss_and_hit(server, capfd):
    server.start()

    # trivial miss: empty cache, slot picks up PROMPT_A
    res = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": PROMPT_A,
            "cache_prompt": True,
        },
    )
    assert res.status_code == 200
    out, _ = capfd.readouterr()

    assert "created context checkpoint" in out

    # 2nd miss, this saves PROMPT_A's state into the cache and looks up
    # PROMPT_B against it
    res = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": PROMPT_B,
            "cache_prompt": True,
        },
    )
    assert res.status_code == 200
    out, _ = capfd.readouterr()
    assert "selected slot by LRU" in out
    assert "updating prompt cache" in out

    assert "failed to load prompt from cache" in out, (
        "expected server_prompt_cache::load() to report a miss for the "
        "unrelated prompt so the slot gets cleared; instead it appears to "
        "have reported a (bogus) hit:\n" + out
    )

    assert "erased invalidated context checkpoint" not in out, (
        "stale checkpoints from the previous conversation leaked into the "
        "new task and were erased one at a time instead of being cleared "
        "before processing started:\n" + out
    )
    assert "forcing full prompt re-processing" not in out, (
        "the new task was compared against stale leftover tokens from the "
        "previous conversation instead of starting from a freshly cleared "
        "slot:\n" + out
    )

    # PROMPT_A again, restore succeeds
    res = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": PROMPT_A,
            "cache_prompt": True,
        },
    )
    assert res.status_code == 200
    out, _ = capfd.readouterr()

    assert "selected slot by LRU" in out
    assert "found better prompt" in out, (
        "expected a genuine prompt-cache hit for the repeated PROMPT_A so "
        "this test actually exercises the erase-then-compare path in "
        "server_prompt_cache::load(); instead it looks like a miss:\n" + out
    )
    assert "failed to load prompt from cache" not in out
