import math
import os
import re
import struct
import time

from utils import *


MIB = 1024 * 1024

LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom. "
) * 4


def make_server() -> ServerProcess:
    server = ServerPreset.tinygemma3()
    server.no_mmproj = True
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.kv_unified = True
    return server


def make_pure_attention_server() -> ServerProcess:
    server = ServerPreset.tinyllama2()
    server.n_predict = 1
    server.server_slots = True
    server.kv_unified = True
    return server


def completion(
        server: ServerProcess,
        prompt: str | list[int],
        id_slot: int | None = None,
        cache_prompt: bool = True) -> dict:
    data = {
        "prompt": prompt,
        "cache_prompt": cache_prompt,
    }
    if id_slot is not None:
        data["id_slot"] = id_slot
    res = server.make_request("POST", "/completion", data=data)
    assert res.status_code == 200
    return res.body["timings"]


def cache_files(cache_dir):
    return [path for path in cache_dir.iterdir() if path.is_file()]


def cache_data_files(cache_dir):
    return [path for path in cache_files(cache_dir) if path.name.endswith(".bin")]


def cache_meta_files(cache_dir):
    return {path for path in cache_files(cache_dir) if path.name.endswith(".bin.meta")}


def tokenize(server: ServerProcess, prompt: str, add_special: bool = True) -> list[int]:
    res = server.make_request("POST", "/tokenize", data={
        "content": prompt,
        "add_special": add_special,
    })
    assert res.status_code == 200
    return res.body["tokens"]


def common_prefix_length(lhs: list[int], rhs: list[int]) -> int:
    return next(
        (i for i, pair in enumerate(zip(lhs, rhs)) if pair[0] != pair[1]),
        min(len(lhs), len(rhs)))


def meta_checkpoint_count(path) -> int:
    # magic[8], version, flags, key_size, tokens_size, payload_size, main_size, drft_size
    header = path.read_bytes()[:64]
    assert header[:8] == b"LLPCACHE"
    return struct.unpack_from("<Q", header, 56)[0]


def test_disk_cache_restores_across_server_restart(tmp_path):
    server = make_server()
    server.cache_ram = 0
    server.cache_disk = str(tmp_path)
    server.cache_disk_max = 256
    server.start()

    timings_full = completion(server, LONG_PROMPT, 0)
    completion(server, "This prompt moves the first prompt into the disk cache.", 1)

    cached_files = {path.name for path in cache_files(tmp_path)}
    assert any(name.endswith(".bin") for name in cached_files)
    assert any(name.endswith(".bin.meta") for name in cached_files)

    # the test model uses SWA, so the cached state must carry context checkpoints
    metas = [path for path in cache_files(tmp_path) if path.name.endswith(".bin.meta")]
    assert all(meta_checkpoint_count(path) > 0 for path in metas)

    server.stop()
    assert cached_files.issubset({path.name for path in cache_files(tmp_path)})

    server.start()
    timings_restored = completion(server, LONG_PROMPT)
    assert timings_restored["cache_n"] + timings_restored["prompt_n"] == timings_full["prompt_n"]
    assert timings_restored["cache_n"] > timings_full["prompt_n"] * 0.8
    assert timings_restored["prompt_n"] < timings_full["prompt_n"] * 0.2
    assert cached_files.issubset({path.name for path in cache_files(tmp_path)})


def test_disk_cache_removes_incomplete_and_invalid_entries(tmp_path):
    incomplete = tmp_path / "llama-prompt-cache-incomplete.bin"
    invalid = tmp_path / "llama-prompt-cache-invalid.bin"
    invalid_metadata = tmp_path / "llama-prompt-cache-invalid.bin.meta"
    incomplete.write_bytes(b"incomplete")
    invalid.write_bytes(b"invalid")
    invalid_metadata.write_bytes(b"invalid metadata")

    server = make_server()
    server.cache_ram = 0
    server.cache_disk = str(tmp_path)
    server.cache_disk_max = 256
    server.start()

    assert not incomplete.exists()
    assert not invalid.exists()
    assert not invalid_metadata.exists()
    completion(server, "The server remains usable after cache cleanup.", 0)


def test_disk_cache_size_limit_keeps_recent_entries(tmp_path):
    server = make_server()
    server.cache_ram = 0
    server.cache_disk = str(tmp_path)
    server.cache_disk_max = -1
    server.start()

    completion(server, LONG_PROMPT, 0)
    completion(server, "Measure the disk state size.", 1)
    data_files = cache_data_files(tmp_path)
    assert len(data_files) == 1
    entry_size = data_files[0].stat().st_size

    server.stop()
    for path in cache_files(tmp_path):
        path.unlink()

    server.cache_disk_max = max(1, math.ceil(2.25 * entry_size / MIB))
    server.start()

    prompts = [f"Cache entry {index}. {LONG_PROMPT}" for index in range(4)]
    for index, prompt in enumerate(prompts):
        completion(server, prompt, index % 2)
    completion(server, "Move the last prompt into the disk cache.", 0)

    cache_limit = server.cache_disk_max * MIB
    assert sum(path.stat().st_size for path in cache_files(tmp_path)) <= cache_limit
    assert len(cache_data_files(tmp_path)) <= 2

    timings = completion(server, prompts[-1])
    assert timings["cache_n"] > 0
    assert timings["prompt_n"] < timings["cache_n"]


def test_disk_cache_strict_extension_matches_predicted_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_LOG_VERBOSITY", "4")

    server = make_server()
    server.n_predict = 1
    server.cache_ram = 0
    server.cache_disk = str(tmp_path)
    server.cache_disk_max = 256
    server.log_path = str(tmp_path / "server.log")
    server.start()

    cached_tokens = tokenize(server, LONG_PROMPT)
    suffix_tokens = tokenize(server, "The cached story continues with a new chapter.", add_special=False)
    extended_tokens = cached_tokens + suffix_tokens
    assert common_prefix_length(cached_tokens, extended_tokens) == len(cached_tokens)
    assert len(cached_tokens) < len(extended_tokens)

    completion(server, cached_tokens, 0)
    completion(server, "Move the strict-extension prompt into the disk cache.", 1)
    [metadata] = cache_meta_files(tmp_path)
    assert meta_checkpoint_count(metadata) > 0

    server.stop()
    server.start()
    timings = completion(server, extended_tokens)
    server.stop()

    log = (tmp_path / "server.log").read_text()
    predicted = re.findall(r"found better prompt .*effective_prefix_reuse = (\d+)", log)
    assert predicted
    assert timings["cache_n"] == int(predicted[-1])
    assert timings["cache_n"] > timings["prompt_n"]


def test_disk_cache_no_usable_checkpoint_skips_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_CTX_CHECKPOINTS", "0")

    server = make_server()
    server.n_predict = 1
    server.cache_ram = 0
    server.cache_disk = str(tmp_path)
    server.cache_disk_max = 256
    server.start()

    cached_prompt = LONG_PROMPT + " This cached branch ends at the red gate."
    divergent_prompt = LONG_PROMPT + " This requested branch ends at the blue harbor."
    cached_tokens = tokenize(server, cached_prompt)
    divergent_tokens = tokenize(server, divergent_prompt)
    raw_lcp = common_prefix_length(cached_tokens, divergent_tokens)
    assert raw_lcp < len(cached_tokens)
    assert raw_lcp / len(cached_tokens) > 0.25

    completion(server, cached_prompt, 0)
    completion(server, "Move the no-checkpoint prompt into the disk cache.", 1)
    [metadata] = cache_meta_files(tmp_path)
    assert meta_checkpoint_count(metadata) == 0
    server.stop()

    old_time = time.time() - 3600
    os.utime(metadata, (old_time, old_time))
    mtime_before = metadata.stat().st_mtime_ns

    server.start()
    timings = completion(server, divergent_prompt)
    assert timings["cache_n"] == 0
    assert metadata.stat().st_mtime_ns == mtime_before


def test_disk_cache_selector_prefers_smaller_equal_prefix(tmp_path):
    server = make_pure_attention_server()
    server.cache_ram = 0
    server.cache_disk = str(tmp_path)
    server.cache_disk_max = 256
    server.start()

    shared = LONG_PROMPT[:len(LONG_PROMPT) // 4]
    long_prompt = shared + (" The old branch follows the mountain road." * 12)
    short_prompt = shared + " The newer branch stops beside the river."
    request_prompt = shared + " The requested branch sails across the ocean."

    long_tokens = tokenize(server, long_prompt)
    short_tokens = tokenize(server, short_prompt)
    request_tokens = tokenize(server, request_prompt)
    raw_lcp = common_prefix_length(short_tokens, request_tokens)
    assert common_prefix_length(long_tokens, request_tokens) == raw_lcp
    assert raw_lcp < len(request_tokens)
    assert raw_lcp / len(long_tokens) > 0.25

    completion(server, long_prompt, 0)
    before = cache_meta_files(tmp_path)
    completion(server, "Move the older long prompt into the disk cache.", 1)
    [long_metadata] = cache_meta_files(tmp_path) - before

    completion(server, short_prompt, 0)
    before = cache_meta_files(tmp_path)
    completion(server, "Move the newer short prompt into the disk cache.", 1)
    [short_metadata] = cache_meta_files(tmp_path) - before
    server.stop()

    assert long_metadata.with_suffix("").stat().st_size > short_metadata.with_suffix("").stat().st_size

    old_time = time.time() - 3600
    os.utime(long_metadata, (old_time, old_time))
    os.utime(short_metadata, (old_time + 1, old_time + 1))
    long_mtime = long_metadata.stat().st_mtime_ns
    short_mtime = short_metadata.stat().st_mtime_ns

    server.start()
    timings = completion(server, request_prompt)
    assert timings["cache_n"] == raw_lcp
    assert long_metadata.stat().st_mtime_ns == long_mtime
    assert short_metadata.stat().st_mtime_ns > short_mtime


def test_disk_cache_reuses_long_entry_below_keep_threshold(tmp_path):
    server = make_pure_attention_server()
    server.cache_ram = 0
    server.cache_disk = str(tmp_path)
    server.cache_disk_max = 256
    server.start()

    shared = LONG_PROMPT[:len(LONG_PROMPT) // 8]
    shared_tokens = tokenize(server, shared)
    long_tail = tokenize(server, "Cached sessions keep going down the mountain road.", add_special=False)
    request_tail = tokenize(server, "Different requests ask something else entirely.", add_special=False)
    tail_lcp = common_prefix_length(long_tail, request_tail)
    long_tail = long_tail[tail_lcp:]
    request_tail = request_tail[tail_lcp:]
    assert long_tail and request_tail
    assert long_tail[0] != request_tail[0]
    target_tail_length = 3 * len(shared_tokens) + 1
    long_tail = (long_tail * math.ceil(target_tail_length / len(long_tail)))[:target_tail_length]
    long_tokens = shared_tokens + long_tail
    request_tokens = shared_tokens + request_tail
    raw_lcp = common_prefix_length(long_tokens, request_tokens)

    assert raw_lcp == len(shared_tokens)
    assert raw_lcp < len(request_tokens)
    assert raw_lcp / len(long_tokens) < 0.25
    assert raw_lcp / len(request_tokens) > 0.1
    assert server.n_ctx is not None and len(long_tokens) < server.n_ctx

    completion(server, long_tokens, 0)
    before = cache_meta_files(tmp_path)
    completion(server, "Move the long session into the disk cache.", 1)
    [long_metadata] = cache_meta_files(tmp_path) - before
    server.stop()

    old_time = time.time() - 3600
    os.utime(long_metadata, (old_time, old_time))
    long_mtime = long_metadata.stat().st_mtime_ns

    server.start()
    timings = completion(server, request_tokens)
    server.stop()

    assert timings["cache_n"] == raw_lcp
    assert long_metadata.stat().st_mtime_ns > long_mtime


def test_disk_cache_does_not_load_when_prompt_cache_disabled(tmp_path):
    server = make_server()
    server.n_predict = 1
    server.cache_ram = 0
    server.cache_disk = str(tmp_path)
    server.cache_disk_max = 256
    server.start()

    completion(server, LONG_PROMPT, 0)
    completion(server, "Move the disabled-cache prompt into the disk cache.", 1)
    [metadata] = cache_meta_files(tmp_path)
    server.stop()

    old_time = time.time() - 3600
    os.utime(metadata, (old_time, old_time))
    mtime_before = metadata.stat().st_mtime_ns

    server.start()
    timings = completion(server, LONG_PROMPT, cache_prompt=False)
    assert timings["cache_n"] == 0
    assert metadata.stat().st_mtime_ns == mtime_before
