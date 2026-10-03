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


@pytest.mark.parametrize("kv_unified", [False, True])
def test_pinned_prefixes(tmp_path, kv_unified):
    server.kv_unified = kv_unified
    server.n_ctx = 1024
    server.cache_ram = 1
    server.cache_prompt_dir = str(tmp_path)
    server.slot_save_path = str(tmp_path)
    short_prompt = LONG_PROMPT[:LONG_PROMPT.index(" He met")]
    # Load the longer prefix first to exercise containment in both directions.
    (tmp_path / "a.txt").write_text(LONG_PROMPT, encoding="utf-8")
    (tmp_path / "b.txt").write_text(short_prompt, encoding="utf-8")
    (tmp_path / "duplicate.txt").write_text(short_prompt, encoding="utf-8")
    (tmp_path / "ignored.md").write_text("ignored", encoding="utf-8")
    server.start()
    log = LogReader(server.log_path)
    startup = log.drain()
    assert startup.count("preloaded pinned prefix:") == 2
    assert "skipping duplicate prefix:" in startup

    prefixes = []
    for text in [short_prompt, LONG_PROMPT]:
        res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": True})
        assert res.status_code == 200
        prefixes.append(res.body["tokens"])
    assert prefixes[1][:len(prefixes[0])] == prefixes[0]

    suffix = server.make_request("POST", "/tokenize", data={"content": " A new adventure began."}).body["tokens"]

    for prefix in prefixes:
        prompt = prefix + suffix
        baseline = server.make_request("POST", "/completion", data={
            "prompt": prompt, "id_slot": 0, "cache_prompt": False, "return_tokens": True,
        })
        assert baseline.status_code == 200
        assert baseline.body["timings"]["cache_n"] == 0

        for id_slot in [0, 1, 0]:
            res = server.make_request("POST", f"/slots/{id_slot}?action=erase")
            assert res.status_code == 200
            res = server.make_request("POST", "/completion", data={
                "prompt": prompt, "id_slot": id_slot, "return_tokens": True,
            })
            assert res.status_code == 200
            assert res.body["timings"]["cache_n"] >= len(prefix)
            assert res.body["tokens"] == baseline.body["tokens"]

    # Ordinary conversations must make room without consuming the pinned entries.
    for i in range(12):
        res = server.make_request("POST", "/completion", data={
            "prompt": [100 + i] * 160, "n_predict": 0,
        })
        assert res.status_code == 200
    assert "removing oldest prompt cache entry" in log.drain()

    for prefix in prefixes:
        res = server.make_request("POST", "/slots/0?action=erase")
        assert res.status_code == 200
        res = server.make_request("POST", "/completion", data={
            "prompt": prefix, "id_slot": 0,
        })
        assert res.status_code == 200
        assert res.body["timings"]["cache_n"] == len(prefix) - 1


@pytest.mark.parametrize("tail", [0, 64, 768])
def test_pinned_prefix_longer_cached_prompt(tmp_path, tail):
    server.cache_prompt_dir = str(tmp_path)
    server.slot_save_path = str(tmp_path)
    server.n_ctx = 2048
    (tmp_path / "prefix.txt").write_text(LONG_PROMPT, encoding="utf-8")
    server.start()

    prefix = server.make_request("POST", "/tokenize", data={"content": LONG_PROMPT, "add_special": True}).body["tokens"]
    suffix = server.make_request("POST", "/tokenize", data={"content": " The knight returned home with the golden sword."}).body["tokens"]
    cached_prompt = prefix + suffix
    prompt = cached_prompt + suffix
    request = {"prompt": prompt, "id_slot": 0, "return_tokens": True}
    baseline = server.make_request("POST", "/completion", data={**request, "cache_prompt": False})
    assert baseline.status_code == 200
    assert server.make_request("POST", "/slots/0?action=erase").status_code == 200

    res = server.make_request("POST", "/completion", data={"prompt": cached_prompt + [100] * tail, "id_slot": 0, "n_predict": 0})
    assert res.status_code == 200
    res = server.make_request("POST", "/completion", data={"prompt": "The quick brown fox", "id_slot": 1, "n_predict": 0})
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    expected = len(cached_prompt) if len(cached_prompt) / (len(cached_prompt) + tail) >= 0.25 else len(prefix)
    assert res.body["timings"]["cache_n"] == expected
    assert res.body["tokens"] == baseline.body["tokens"]


def test_pinned_prefix_shorter_cached_prompt(tmp_path):
    server.n_ctx = 1024
    server.cache_prompt_dir = str(tmp_path)
    server.slot_save_path = str(tmp_path)
    (tmp_path / "prefix.txt").write_text(LONG_PROMPT, encoding="utf-8")
    server.start()

    prefix = server.make_request("POST", "/tokenize", data={"content": LONG_PROMPT, "add_special": True}).body["tokens"]
    cached_prompt = prefix[:len(prefix) * 2 // 5]
    request = {"prompt": cached_prompt, "id_slot": 0, "return_tokens": True}
    baseline = server.make_request("POST", "/completion", data={**request, "cache_prompt": False})
    assert baseline.status_code == 200
    assert server.make_request("POST", "/slots/0?action=erase").status_code == 200

    res = server.make_request("POST", "/completion", data={**request, "n_predict": 0})
    assert res.status_code == 200
    res = server.make_request("POST", "/completion", data={"prompt": "The quick brown fox", "id_slot": 1, "n_predict": 0})
    assert res.status_code == 200
    res = server.make_request("POST", "/completion", data={**request, "id_slot": -1})
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == len(cached_prompt) - 1
    assert res.body["tokens"] == baseline.body["tokens"]


@pytest.mark.skipif(not is_slow_test_allowed(), reason="skipping slow test")
@pytest.mark.parametrize("model", ["Qwen3-0.6B-GGUF:Q4_0", "gemma-3-270m-GGUF:Q8_0", "Qwen3.5-0.8B-GGUF:Q4_0"])
@pytest.mark.parametrize("kv_unified", [False, True])
def test_pinned_prefix_model_consistency(tmp_path, model, kv_unified):
    server.model_hf_repo = f"ggml-org/{model}"
    server.model_hf_file = None
    server.n_ctx = 4096
    server.n_batch = 128
    server.n_ubatch = 32
    server.n_predict = 8
    server.cache_ram = 512
    server.kv_unified = kv_unified
    server.no_cache_idle_slots = True
    server.cache_prompt_dir = str(tmp_path)
    server.slot_save_path = str(tmp_path)
    text = "The project is named Atlas. Its color is blue. The answer should use these facts.\n" * 40
    (tmp_path / "prefix.txt").write_text(text, encoding="utf-8")
    server.start(timeout_seconds=600)

    prefix = server.make_request("POST", "/tokenize", data={"content": text, "add_special": True}).body["tokens"]
    suffix = server.make_request("POST", "/tokenize", data={"content": " What is the project color?"}).body["tokens"]
    assert len(prefix) > 512  # Cross the Gemma 3 sliding window.
    for prompt in [prefix + suffix, prefix, prefix[:len(prefix) // 2] + suffix]:
        request = {"prompt": prompt, "return_tokens": True, "ignore_eos": True}
        for id_slot in [0, 1]:
            baseline = server.make_request("POST", "/completion", data={**request, "id_slot": id_slot, "cache_prompt": False})
            assert baseline.status_code == 200
            assert baseline.body["timings"]["cache_n"] == 0
            assert server.make_request("POST", f"/slots/{id_slot}?action=erase").status_code == 200
            res = server.make_request("POST", "/completion", data={**request, "id_slot": id_slot})
            assert res.status_code == 200
            assert res.body["tokens"] == baseline.body["tokens"]
            if len(prompt) >= len(prefix):
                assert res.body["timings"]["cache_n"] >= len(prefix) - server.n_ubatch
            else:
                assert res.body["timings"]["cache_n"] == 0


@pytest.mark.skipif(not is_slow_test_allowed(), reason="skipping slow test")
@pytest.mark.parametrize("kv_unified", [False, True])
@pytest.mark.parametrize("previous", ["restored", "uncached"])
def test_pinned_prefix_recurrent_fallback(tmp_path, kv_unified, previous):
    server.model_hf_repo = "ggml-org/Qwen3.5-0.8B-GGUF:Q4_0"
    server.model_hf_file = None
    server.n_ctx = 2048
    server.n_batch = 128
    server.n_ubatch = 32
    server.cache_ram = 512
    server.kv_unified = kv_unified
    server.no_cache_idle_slots = True
    server.cache_prompt_dir = str(tmp_path)
    server.slot_save_path = str(tmp_path)
    text = "The project is named Atlas. Its color is blue. The answer should use these facts.\n" * 3
    (tmp_path / "prefix.txt").write_text(text, encoding="utf-8")
    server.start(timeout_seconds=600)

    prefix = server.make_request("POST", "/tokenize", data={"content": text, "add_special": True}).body["tokens"]
    suffixes = [server.make_request("POST", "/tokenize", data={"content": f" Question {i}: What is the color?"}).body["tokens"] for i in [5, 6]]
    prompt = prefix + suffixes[0]
    request = {"prompt": prompt, "id_slot": 0, "return_tokens": True, "ignore_eos": True}
    baseline = server.make_request("POST", "/completion", data={**request, "cache_prompt": False})
    assert baseline.status_code == 200

    if previous == "restored":
        assert server.make_request("POST", "/slots/0?action=erase").status_code == 200
        res = server.make_request("POST", "/completion", data={**request, "cache_prompt": False, "n_predict": 0})
        assert res.status_code == 200
        res = server.make_request("POST", "/slots/0?action=save", data={"filename": "recurrent.bin"})
        assert res.status_code == 200
        assert server.make_request("POST", "/slots/0?action=erase").status_code == 200
        res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "recurrent.bin"})
    else:
        assert suffixes[0][0] == suffixes[1][0]
        res = server.make_request("POST", "/completion", data={**request, "prompt": prefix + suffixes[1], "cache_prompt": False})
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == len(prefix)
    assert res.body["timings"]["prompt_n"] == len(prompt) - len(prefix)
    assert res.body["tokens"] == baseline.body["tokens"]


@pytest.mark.parametrize("case", ["missing", "empty", "empty_file", "context", "disabled", "capacity"])
def test_pinned_prefix_startup_failure(tmp_path, case):
    server.cache_prompt_dir = str(tmp_path)
    if case == "missing":
        server.cache_prompt_dir = str(tmp_path / "missing")
    elif case == "empty_file":
        (tmp_path / "prefix.txt").write_text("", encoding="utf-8")
    elif case == "context":
        (tmp_path / "prefix.txt").write_text(LONG_PROMPT * 20, encoding="utf-8")
    elif case == "disabled":
        server.cache_ram = 0
        (tmp_path / "prefix.txt").write_text(LONG_PROMPT, encoding="utf-8")
    elif case == "capacity":
        server.cache_ram = 1
        for i in range(24):
            (tmp_path / f"{i:02}.txt").write_text(f"{i} {LONG_PROMPT}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Server process died"):
        server.start()
    errors = {
        "missing": "prefix directory does not exist",
        "empty": "contains no .txt files",
        "empty_file": "failed to read a nonempty prefix",
        "context": "prefix exceeds slot context",
        "disabled": "requires --cache-prompt and a nonzero --cache-ram",
        "capacity": "failed to pin prefix; increase --cache-ram",
    }
    log = LogReader(server.log_path).drain()
    assert "failed to preload prefixes:" in log
    assert errors[case] in log


def test_pinned_prefix_token_limit_and_reload(tmp_path):
    server.cache_prompt_dir = str(tmp_path)
    server.cache_ram = -1
    server.sleep_idle_seconds = 1
    for i in range(4):
        (tmp_path / f"{i}.txt").write_text(f"{i} {LONG_PROMPT}", encoding="utf-8")
    server.start()
    prompt = f"0 {LONG_PROMPT}"
    tokens = server.make_request("POST", "/tokenize", data={"content": prompt, "add_special": True}).body["tokens"]
    assert len(tokens) * 4 > server.n_ctx
    for cycle in range(2):
        res = server.make_request("POST", "/completion", data={"prompt": prompt, "n_predict": 0})
        assert res.status_code == 200
        assert res.body["timings"]["cache_n"] == len(tokens) - 1
        for slot, ordinary in [(0, [100] * 50), (1, [101] * 50), (-1, [100] * 50 + [102])]:
            res = server.make_request("POST", "/completion", data={"prompt": ordinary, "id_slot": slot, "n_predict": 0})
            assert res.status_code == 200
        assert res.body["timings"]["cache_n"] == 50
        if cycle == 0:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if server.make_request("GET", "/props").body["is_sleeping"]:
                    break
                time.sleep(0.1)
            else:
                pytest.fail("server did not sleep")
    assert LogReader(server.log_path).drain().count("preloaded pinned prefix:") == 8
