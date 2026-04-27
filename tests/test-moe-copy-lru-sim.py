#!/usr/bin/env python3

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path


SAMPLE_LOG = """\
noise before
ggml_backend_sched_compute_splits: moe_copy split=1 input=0 tensor=blk.0.ffn_down_exps.weight node=ffn_down ids=topk src_backend=CPU dst_backend=CUDA0 n_expert=4 expert_size=100 used=2 used_bytes=200 ranges=1 copy_bytes=200 ids=[1,2]
ggml_backend_sched_compute_splits: moe_copy split=1 input=0 tensor=blk.0.ffn_down_exps.weight node=ffn_down ids=topk src_backend=CPU dst_backend=CUDA0 n_expert=4 expert_size=100 used=2 used_bytes=200 ranges=1 copy_bytes=200 ids=[2,3]
ggml_backend_sched_compute_splits: moe_copy split=1 input=0 tensor=blk.0.ffn_down_exps.weight node=ffn_down ids=topk src_backend=CPU dst_backend=CUDA0 n_expert=4 expert_size=100 used=2 used_bytes=200 ranges=1 copy_bytes=200 ids=[1,2]
ggml_backend_sched_compute_splits: moe_copy split=2 input=0 tensor=blk.1.ffn_down_exps.weight node=ffn_down ids=topk src_backend=CPU dst_backend=CUDA1 n_expert=4 expert_size=50 used=1 used_bytes=50 ranges=1 copy_bytes=50 ids=[0]
"""

SAMPLE_RUNTIME_LOG = """\
noise before
ggml_backend_sched_moe_cache_prepare: moe_cache tensor=blk.0.ffn_down_exps.weight backend=CUDA0 slots=2 used=2 hits=0 misses=2 copied=200 total_hits=0 total_misses=2 total_copied=200
ggml_backend_sched_moe_cache_prepare: moe_cache tensor=blk.0.ffn_down_exps.weight backend=CUDA0 slots=2 used=2 hits=1 misses=1 copied=100 total_hits=1 total_misses=3 total_copied=300
ggml_backend_sched_moe_cache_prepare: moe_cache tensor=blk.1.ffn_down_exps.weight backend=CUDA1 slots=1 used=1 hits=0 misses=1 copied=50 total_hits=0 total_misses=1 total_copied=50
ggml_backend_sched_compute_splits: moe_cache_bypass tensor=blk.2.ffn_down_exps.weight node=ffn_down ids=topk backend=CUDA0 slots=2 reason=too_many_experts n_expert=4 expert_size=100
ggml_backend_sched_compute_splits: moe_cache_bypass tensor=blk.2.ffn_down_exps.weight node=ffn_down ids=topk backend=CUDA0 slots=2 reason=ids_alloc_failed n_expert=4 expert_size=100
ggml_backend_sched_compute_splits: moe_cache_bypass tensor=blk.3.ffn_down_exps.weight node=ffn_down ids=topk backend=CUDA1 slots=1 reason=too_many_experts n_expert=4 expert_size=50
"""


def load_sim(repo_root: Path):
    script = repo_root / "scripts" / "moe-copy-lru-sim.py"
    spec = importlib.util.spec_from_file_location("moe_copy_lru_sim", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to create simulator module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser(sim) -> None:
    events = list(sim.read_events_from_lines(SAMPLE_LOG.splitlines()))
    assert len(events) == 4
    assert events[0].key == "CUDA0:blk.0.ffn_down_exps.weight"
    assert events[0].expert_size == 100
    assert events[0].used_bytes == 200
    assert events[0].copy_bytes == 200
    assert events[0].expert_ids == (1, 2)


def test_lru_batch_eviction(sim) -> None:
    events = list(sim.read_events_from_lines(SAMPLE_LOG.splitlines()))
    stats = sim.simulate_lru(events, [1, 2])

    k1 = stats[(1, "CUDA0:blk.0.ffn_down_exps.weight")]
    assert k1.events == 3
    assert k1.bypasses == 3
    assert k1.hits == 0
    assert k1.misses == 6
    assert k1.cache_bytes == 100
    assert k1.baseline_bytes == 600
    assert k1.cache_copy_bytes == 600

    k2 = stats[(2, "CUDA0:blk.0.ffn_down_exps.weight")]
    assert k2.events == 3
    assert k2.bypasses == 0
    assert k2.hits == 2
    assert k2.misses == 4
    assert k2.cache_bytes == 200
    assert k2.baseline_bytes == 600
    assert k2.cache_copy_bytes == 400

    aggregate = sim.aggregate_stats(stats)
    assert aggregate[2].cache_bytes == 300
    assert aggregate[2].baseline_bytes == 650
    assert aggregate[2].cache_copy_bytes == 450


def test_cli(repo_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "moe.log"
        log_path.write_text(SAMPLE_LOG, encoding="utf-8")
        script = repo_root / "scripts" / "moe-copy-lru-sim.py"
        result = subprocess.run(
            [sys.executable, str(script), "--slots", "2", str(log_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    assert "slots\tkey\tcache_bytes\tevents" in result.stdout
    assert "2\tALL\t300\t4\t0\t7\t2\t5\t0.285714\t650\t450\t200\t0.307692" in result.stdout


def test_runtime_cache_parser_and_summary(sim) -> None:
    events = list(sim.read_cache_events_from_lines(SAMPLE_RUNTIME_LOG.splitlines()))
    assert len(events) == 3
    assert events[0].key == "CUDA0:blk.0.ffn_down_exps.weight"
    assert events[0].slots == 2
    assert events[0].used == 2
    assert events[0].hits == 0
    assert events[0].misses == 2
    assert events[0].copied == 200

    stats = sim.summarize_runtime_cache(events)
    k0 = stats["CUDA0:blk.0.ffn_down_exps.weight"]
    assert k0.slots == 2
    assert k0.events == 2
    assert k0.accesses == 4
    assert k0.hits == 1
    assert k0.misses == 3
    assert k0.copied == 300
    assert k0.max_total_hits == 1
    assert k0.max_total_misses == 3
    assert k0.max_total_copied == 300

    aggregate = sim.aggregate_runtime_stats(stats)
    assert aggregate.events == 3
    assert aggregate.accesses == 5
    assert aggregate.hits == 1
    assert aggregate.misses == 4
    assert aggregate.copied == 350
    assert aggregate.max_total_copied == 350


def test_runtime_cache_cli(repo_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "moe-runtime.log"
        log_path.write_text(SAMPLE_RUNTIME_LOG, encoding="utf-8")
        script = repo_root / "scripts" / "moe-copy-lru-sim.py"
        result = subprocess.run(
            [sys.executable, str(script), "--runtime", "--details", str(log_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    assert "key\tslots\tevents\taccesses\thits\tmisses" in result.stdout
    assert "ALL\t-\t3\t5\t1\t4\t0.200000\t350\t1\t4\t350" in result.stdout
    assert "CUDA0:blk.0.ffn_down_exps.weight\t2\t2\t4\t1\t3\t0.250000\t300\t1\t3\t300" in result.stdout
    assert "bypass_key\treason\tevents" in result.stdout
    assert "ALL\ttoo_many_experts\t2" in result.stdout
    assert "CUDA0:blk.2.ffn_down_exps.weight\tids_alloc_failed\t1" in result.stdout


def test_runtime_cache_bypass_parser_and_summary(sim) -> None:
    events = list(sim.read_cache_bypass_events_from_lines(SAMPLE_RUNTIME_LOG.splitlines()))
    assert len(events) == 3
    assert events[0].key == "CUDA0:blk.2.ffn_down_exps.weight"
    assert events[0].slots == 2
    assert events[0].reason == "too_many_experts"
    assert events[0].n_expert == 4
    assert events[0].expert_size == 100

    stats = sim.summarize_runtime_bypasses(events)
    assert stats[("CUDA0:blk.2.ffn_down_exps.weight", "too_many_experts")] == 1
    assert stats[("CUDA0:blk.2.ffn_down_exps.weight", "ids_alloc_failed")] == 1
    assert stats[("CUDA1:blk.3.ffn_down_exps.weight", "too_many_experts")] == 1


def test_rejects_inconsistent_expert_size(sim) -> None:
    log = """\
ggml_backend_sched_compute_splits: moe_copy split=1 input=0 tensor=blk.0.ffn_down_exps.weight node=ffn_down ids=topk src_backend=CPU dst_backend=CUDA0 n_expert=4 expert_size=100 used=1 used_bytes=100 ranges=1 copy_bytes=100 ids=[1]
ggml_backend_sched_compute_splits: moe_copy split=1 input=0 tensor=blk.0.ffn_down_exps.weight node=ffn_down ids=topk src_backend=CPU dst_backend=CUDA0 n_expert=4 expert_size=200 used=1 used_bytes=200 ranges=1 copy_bytes=200 ids=[2]
"""
    events = list(sim.read_events_from_lines(log.splitlines()))
    try:
        sim.simulate_lru(events, [2])
    except ValueError as exc:
        assert "inconsistent expert_size" in str(exc)
    else:
        raise AssertionError("accepted inconsistent expert_size for a single cache key")


def test_rejects_inconsistent_copy_accounting(sim) -> None:
    bad_used_bytes = "ggml_backend_sched_compute_splits: moe_copy split=1 input=0 tensor=blk.0.ffn_down_exps.weight node=ffn_down ids=topk src_backend=CPU dst_backend=CUDA0 n_expert=4 expert_size=100 used=2 used_bytes=100 ranges=1 copy_bytes=200 ids=[1,2]"
    try:
        list(sim.read_events_from_lines([bad_used_bytes]))
    except ValueError as exc:
        assert "used_bytes" in str(exc)
    else:
        raise AssertionError("accepted inconsistent used_bytes")

    bad_copy_bytes = "ggml_backend_sched_compute_splits: moe_copy split=1 input=0 tensor=blk.0.ffn_down_exps.weight node=ffn_down ids=topk src_backend=CPU dst_backend=CUDA0 n_expert=4 expert_size=100 used=2 used_bytes=200 ranges=1 copy_bytes=150 ids=[1,2]"
    try:
        list(sim.read_events_from_lines([bad_copy_bytes]))
    except ValueError as exc:
        assert "copy_bytes" in str(exc)
    else:
        raise AssertionError("accepted copy_bytes smaller than used_bytes")


def test_rejects_inconsistent_runtime_cache_accounting(sim) -> None:
    bad_used = "ggml_backend_sched_moe_cache_prepare: moe_cache tensor=blk.0.ffn_down_exps.weight backend=CUDA0 slots=2 used=2 hits=2 misses=1 copied=100 total_hits=2 total_misses=1 total_copied=100"
    try:
        list(sim.read_cache_events_from_lines([bad_used]))
    except ValueError as exc:
        assert "used" in str(exc)
    else:
        raise AssertionError("accepted inconsistent runtime used/hit/miss counts")

    bad_total = "ggml_backend_sched_moe_cache_prepare: moe_cache tensor=blk.0.ffn_down_exps.weight backend=CUDA0 slots=2 used=1 hits=0 misses=1 copied=100 total_hits=0 total_misses=0 total_copied=100"
    try:
        list(sim.read_cache_events_from_lines([bad_total]))
    except ValueError as exc:
        assert "total" in str(exc)
    else:
        raise AssertionError("accepted runtime total counters below per-event counters")

    bad_slots = """\
ggml_backend_sched_moe_cache_prepare: moe_cache tensor=blk.0.ffn_down_exps.weight backend=CUDA0 slots=2 used=1 hits=0 misses=1 copied=100 total_hits=0 total_misses=1 total_copied=100
ggml_backend_sched_moe_cache_prepare: moe_cache tensor=blk.0.ffn_down_exps.weight backend=CUDA0 slots=3 used=1 hits=0 misses=1 copied=100 total_hits=0 total_misses=1 total_copied=100
"""
    events = list(sim.read_cache_events_from_lines(bad_slots.splitlines()))
    try:
        sim.summarize_runtime_cache(events)
    except ValueError as exc:
        assert "inconsistent slots" in str(exc)
    else:
        raise AssertionError("accepted inconsistent runtime slots for one key")

    bad_bypass = "ggml_backend_sched_compute_splits: moe_cache_bypass tensor=blk.0.ffn_down_exps.weight node=ffn_down ids=topk backend=CUDA0 slots=-1 reason=too_many_experts n_expert=4 expert_size=100"
    try:
        list(sim.read_cache_bypass_events_from_lines([bad_bypass]))
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("accepted negative runtime bypass slots")


def main() -> None:
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    sim = load_sim(repo_root)
    test_parser(sim)
    test_lru_batch_eviction(sim)
    test_cli(repo_root)
    test_runtime_cache_parser_and_summary(sim)
    test_runtime_cache_cli(repo_root)
    test_runtime_cache_bypass_parser_and_summary(sim)
    test_rejects_inconsistent_expert_size(sim)
    test_rejects_inconsistent_copy_accounting(sim)
    test_rejects_inconsistent_runtime_cache_accounting(sim)


if __name__ == "__main__":
    main()
