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
ggml_backend_sched_moe_cache_prepare: moe_cache tensor=blk.0.ffn_down_exps.weight backend=CUDA0 slots=2 expert_size=100 cache_bytes=300 used=2 hits=0 misses=2 copied=200 total_hits=0 total_misses=2 total_copied=200
ggml_backend_sched_moe_cache_prepare: moe_cache tensor=blk.0.ffn_down_exps.weight backend=CUDA0 slots=2 expert_size=100 cache_bytes=300 used=2 hits=1 misses=1 copied=100 total_hits=1 total_misses=3 total_copied=300
ggml_backend_sched_moe_cache_prepare: moe_cache tensor=blk.1.ffn_down_exps.weight backend=CUDA1 slots=1 expert_size=50 cache_bytes=50 used=1 hits=0 misses=1 copied=50 total_hits=0 total_misses=1 total_copied=50
ggml_backend_sched_compute_splits: moe_cache_bypass tensor=blk.2.ffn_down_exps.weight node=ffn_down ids=topk backend=CUDA0 slots=2 reason=too_many_experts n_expert=4 expert_size=100
ggml_backend_sched_compute_splits: moe_cache_bypass tensor=blk.2.ffn_down_exps.weight node=ffn_down ids=topk backend=CUDA0 slots=2 reason=ids_alloc_failed n_expert=4 expert_size=100
ggml_backend_sched_compute_splits: moe_cache_bypass tensor=blk.3.ffn_down_exps.weight node=ffn_down ids=topk backend=CUDA1 slots=1 reason=too_many_experts n_expert=4 expert_size=50
"""


SAMPLE_PROMPT_PRIME_LOG = """\
ggml_backend_sched_compute_splits: moe_copy split=1 input=0 tensor=blk.0.ffn_down_exps.weight node=ffn_down ids=topk src_backend=CPU dst_backend=CUDA0 n_expert=4 expert_size=100 used=3 used_bytes=300 ranges=1 copy_bytes=300 id_counts=[0:5,1:4,2:1] ids=[0,1,2]
ggml_backend_sched_compute_splits: moe_copy split=1 input=0 tensor=blk.0.ffn_down_exps.weight node=ffn_down ids=topk src_backend=CPU dst_backend=CUDA0 n_expert=4 expert_size=100 used=2 used_bytes=200 ranges=1 copy_bytes=200 ids=[0,1]
"""


SAMPLE_ORACLE_LOG = """\
ggml_backend_sched_compute_splits: moe_copy split=1 input=0 tensor=blk.0.ffn_down_exps.weight node=ffn_down ids=topk src_backend=CPU dst_backend=CUDA0 n_expert=4 expert_size=100 used=2 used_bytes=200 ranges=1 copy_bytes=200 ids=[0,1]
ggml_backend_sched_compute_splits: moe_copy split=1 input=0 tensor=blk.0.ffn_down_exps.weight node=ffn_down ids=topk src_backend=CPU dst_backend=CUDA0 n_expert=4 expert_size=100 used=2 used_bytes=200 ranges=1 copy_bytes=200 ids=[2,3]
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
    assert events[0].expert_counts == ((1, 1), (2, 1))


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


def test_prompt_prefetch_uses_bypass_hot_set(sim) -> None:
    events = list(sim.read_events_from_lines(SAMPLE_PROMPT_PRIME_LOG.splitlines()))
    assert events[0].expert_counts == ((0, 5), (1, 4), (2, 1))
    stats = sim.simulate_prefetch(events, [2], "prompt")
    stat = stats[("prompt", 2, "CUDA0:blk.0.ffn_down_exps.weight")]

    assert stat.events == 2
    assert stat.bypasses == 1
    assert stat.accesses == 5
    assert stat.speculative_hits == 2
    assert stat.demand_hits == 0
    assert stat.misses == 3
    assert stat.baseline_bytes == 500
    assert stat.demand_copy_bytes == 300
    assert stat.prefetch_copy_bytes == 200
    assert stat.prefetches == 2


def test_oracle_prefetch_bounds_next_event(sim) -> None:
    events = list(sim.read_events_from_lines(SAMPLE_ORACLE_LOG.splitlines()))
    stats = sim.simulate_prefetch(events, [2], "oracle")
    stat = stats[("oracle", 2, "CUDA0:blk.0.ffn_down_exps.weight")]

    assert stat.events == 2
    assert stat.accesses == 4
    assert stat.speculative_hits == 2
    assert stat.misses == 2
    assert stat.baseline_bytes == 400
    assert stat.demand_copy_bytes == 200
    assert stat.prefetch_copy_bytes == 200
    assert stat.prefetches == 2
    assert stat.wrong_prefetches == 0


def test_markov_prefetch_learns_repeated_sequence(sim) -> None:
    events = list(sim.read_events_from_lines(SAMPLE_ORACLE_LOG.splitlines())) * 2
    stats = sim.simulate_prefetch(events, [2], "markov")
    stat = stats[("markov", 2, "CUDA0:blk.0.ffn_down_exps.weight")]

    assert stat.events == 4
    assert stat.accesses == 8
    assert stat.speculative_hits == 2
    assert stat.misses == 6
    assert stat.prefetches == 4
    assert stat.demand_copy_bytes == 600
    assert stat.prefetch_copy_bytes == 400

    set_stats = sim.simulate_prefetch(events, [2], "setmarkov")
    set_stat = set_stats[("setmarkov", 2, "CUDA0:blk.0.ffn_down_exps.weight")]
    assert set_stat.speculative_hits == 2
    assert set_stat.prefetches == 4


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


def test_prefetch_cli(repo_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "moe-prompt.log"
        log_path.write_text(SAMPLE_PROMPT_PRIME_LOG, encoding="utf-8")
        script = repo_root / "scripts" / "moe-copy-lru-sim.py"
        result = subprocess.run(
            [sys.executable, str(script), "--slots", "2", "--policy", "prompt", str(log_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    assert "policy\tslots\tkey\tcache_bytes\tevents\tbypasses\taccesses" in result.stdout
    assert "prompt\t2\tALL\t200\t2\t1\t5\t2\t0\t2\t3\t0.400000\t500\t300\t200\t500\t200\t0.400000\t0\t0.000000\t2\t0\t0" in result.stdout


def test_runtime_cache_parser_and_summary(sim) -> None:
    events = list(sim.read_cache_events_from_lines(SAMPLE_RUNTIME_LOG.splitlines()))
    assert len(events) == 3
    assert events[0].key == "CUDA0:blk.0.ffn_down_exps.weight"
    assert events[0].slots == 2
    assert events[0].expert_size == 100
    assert events[0].cache_bytes == 300
    assert events[0].used == 2
    assert events[0].hits == 0
    assert events[0].misses == 2
    assert events[0].copied == 200

    stats = sim.summarize_runtime_cache(events)
    k0 = stats[(2, "CUDA0:blk.0.ffn_down_exps.weight")]
    assert k0.slots == 2
    assert k0.expert_size == 100
    assert k0.cache_bytes == 300
    assert k0.events == 2
    assert k0.accesses == 4
    assert k0.hits == 1
    assert k0.misses == 3
    assert k0.copied == 300
    assert k0.max_total_hits == 1
    assert k0.max_total_misses == 3
    assert k0.max_total_copied == 300

    aggregate = sim.aggregate_runtime_stats(stats)
    assert aggregate[1].cache_bytes == 50
    assert aggregate[1].events == 1
    assert aggregate[1].accesses == 1
    assert aggregate[1].hits == 0
    assert aggregate[1].misses == 1
    assert aggregate[1].copied == 50
    assert aggregate[1].max_total_copied == 50
    assert aggregate[2].cache_bytes == 300
    assert aggregate[2].events == 2
    assert aggregate[2].accesses == 4
    assert aggregate[2].hits == 1
    assert aggregate[2].misses == 3
    assert aggregate[2].copied == 300
    assert aggregate[2].max_total_copied == 300


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
    assert "key\tslots\tcache_bytes\tevents\taccesses\thits\tmisses" in result.stdout
    assert "ALL\t1\t50\t1\t1\t0\t1\t0.000000\t50\t0\t1\t50" in result.stdout
    assert "ALL\t2\t300\t2\t4\t1\t3\t0.250000\t300\t1\t3\t300" in result.stdout
    assert "CUDA0:blk.0.ffn_down_exps.weight\t2\t300\t2\t4\t1\t3\t0.250000\t300\t1\t3\t300" in result.stdout
    assert "bypass_key\tslots\treason\tevents" in result.stdout
    assert "ALL\t1\ttoo_many_experts\t1" in result.stdout
    assert "ALL\t2\ttoo_many_experts\t1" in result.stdout
    assert "CUDA0:blk.2.ffn_down_exps.weight\t2\tids_alloc_failed\t1" in result.stdout


def test_cli_tolerates_invalid_utf8(repo_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "moe-invalid-utf8.log"
        log_path.write_bytes(
            b"\xef\xbf\x00spinner\n"
            b"ggml_backend_sched_moe_cache_prepare: moe_cache tensor=t backend=CUDA0 slots=2 expert_size=100 "
            b"cache_bytes=300 used=1 hits=0 misses=1 copied=100 total_hits=0 total_misses=1 total_copied=100\n"
        )
        script = repo_root / "scripts" / "moe-copy-lru-sim.py"
        result = subprocess.run(
            [sys.executable, str(script), "--runtime", str(log_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    assert "ALL\t2\t300\t1\t1\t0\t1\t0.000000\t100\t0\t1\t100" in result.stdout


def test_runtime_cache_bypass_parser_and_summary(sim) -> None:
    events = list(sim.read_cache_bypass_events_from_lines(SAMPLE_RUNTIME_LOG.splitlines()))
    assert len(events) == 3
    assert events[0].key == "CUDA0:blk.2.ffn_down_exps.weight"
    assert events[0].slots == 2
    assert events[0].reason == "too_many_experts"
    assert events[0].n_expert == 4
    assert events[0].expert_size == 100

    stats = sim.summarize_runtime_bypasses(events)
    assert stats[("CUDA0:blk.2.ffn_down_exps.weight", 2, "too_many_experts")] == 1
    assert stats[("CUDA0:blk.2.ffn_down_exps.weight", 2, "ids_alloc_failed")] == 1
    assert stats[("CUDA1:blk.3.ffn_down_exps.weight", 1, "too_many_experts")] == 1


def test_runtime_bypass_only_cli(repo_root: Path) -> None:
    log = """\
ggml_backend_sched_compute_splits: moe_cache_bypass tensor=blk.2.ffn_down_exps.weight node=ffn_down ids=topk backend=CUDA0 slots=2 reason=too_many_experts n_expert=4 expert_size=100
ggml_backend_sched_compute_splits: moe_cache_bypass tensor=blk.2.ffn_down_exps.weight node=ffn_down ids=topk backend=CUDA0 slots=2 reason=ids_alloc_failed n_expert=4 expert_size=100
"""
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "moe-runtime-bypass.log"
        log_path.write_text(log, encoding="utf-8")
        script = repo_root / "scripts" / "moe-copy-lru-sim.py"
        result = subprocess.run(
            [sys.executable, str(script), "--runtime", "--details", str(log_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    assert "key\tslots\tcache_bytes\tevents\taccesses\thits\tmisses" not in result.stdout
    assert "bypass_key\tslots\treason\tevents" in result.stdout
    assert "ALL\t2\tids_alloc_failed\t1" in result.stdout
    assert "ALL\t2\ttoo_many_experts\t1" in result.stdout
    assert "CUDA0:blk.2.ffn_down_exps.weight\t2\ttoo_many_experts\t1" in result.stdout


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

    bad_id_counts = "ggml_backend_sched_compute_splits: moe_copy split=1 input=0 tensor=blk.0.ffn_down_exps.weight node=ffn_down ids=topk src_backend=CPU dst_backend=CUDA0 n_expert=4 expert_size=100 used=2 used_bytes=200 ranges=1 copy_bytes=200 id_counts=[1:2] ids=[1,2]"
    try:
        list(sim.read_events_from_lines([bad_id_counts]))
    except ValueError as exc:
        assert "id_counts" in str(exc)
    else:
        raise AssertionError("accepted id_counts that did not match ids")


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

    mixed_slots = """\
ggml_backend_sched_moe_cache_prepare: moe_cache tensor=blk.0.ffn_down_exps.weight backend=CUDA0 slots=2 cache_bytes=200 used=1 hits=0 misses=1 copied=100 total_hits=0 total_misses=1 total_copied=100
ggml_backend_sched_moe_cache_prepare: moe_cache tensor=blk.0.ffn_down_exps.weight backend=CUDA0 slots=3 cache_bytes=300 used=1 hits=0 misses=1 copied=100 total_hits=0 total_misses=1 total_copied=100
"""
    events = list(sim.read_cache_events_from_lines(mixed_slots.splitlines()))
    stats = sim.summarize_runtime_cache(events)
    assert stats[(2, "CUDA0:blk.0.ffn_down_exps.weight")].events == 1
    assert stats[(3, "CUDA0:blk.0.ffn_down_exps.weight")].events == 1
    assert stats[(2, "CUDA0:blk.0.ffn_down_exps.weight")].cache_bytes == 200
    assert stats[(3, "CUDA0:blk.0.ffn_down_exps.weight")].cache_bytes == 300

    inconsistent_cache_bytes = """\
ggml_backend_sched_moe_cache_prepare: moe_cache tensor=blk.0.ffn_down_exps.weight backend=CUDA0 slots=2 expert_size=100 cache_bytes=200 used=1 hits=0 misses=1 copied=100 total_hits=0 total_misses=1 total_copied=100
ggml_backend_sched_moe_cache_prepare: moe_cache tensor=blk.0.ffn_down_exps.weight backend=CUDA0 slots=2 expert_size=100 cache_bytes=300 used=1 hits=0 misses=1 copied=100 total_hits=0 total_misses=1 total_copied=100
"""
    try:
        sim.summarize_runtime_cache(list(sim.read_cache_events_from_lines(inconsistent_cache_bytes.splitlines())))
    except ValueError as exc:
        assert "cache_bytes" in str(exc)
    else:
        raise AssertionError("accepted inconsistent runtime cache footprint")

    inconsistent_expert_size = """\
ggml_backend_sched_moe_cache_prepare: moe_cache tensor=blk.0.ffn_down_exps.weight backend=CUDA0 slots=2 expert_size=100 cache_bytes=200 used=1 hits=0 misses=1 copied=100 total_hits=0 total_misses=1 total_copied=100
ggml_backend_sched_moe_cache_prepare: moe_cache tensor=blk.0.ffn_down_exps.weight backend=CUDA0 slots=2 expert_size=101 cache_bytes=200 used=1 hits=0 misses=1 copied=100 total_hits=0 total_misses=1 total_copied=100
"""
    try:
        sim.summarize_runtime_cache(list(sim.read_cache_events_from_lines(inconsistent_expert_size.splitlines())))
    except ValueError as exc:
        assert "expert_size" in str(exc)
    else:
        raise AssertionError("accepted inconsistent runtime cache expert size")

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
    test_prompt_prefetch_uses_bypass_hot_set(sim)
    test_oracle_prefetch_bounds_next_event(sim)
    test_markov_prefetch_learns_repeated_sequence(sim)
    test_cli(repo_root)
    test_prefetch_cli(repo_root)
    test_runtime_cache_parser_and_summary(sim)
    test_runtime_cache_cli(repo_root)
    test_cli_tolerates_invalid_utf8(repo_root)
    test_runtime_cache_bypass_parser_and_summary(sim)
    test_runtime_bypass_only_cli(repo_root)
    test_rejects_inconsistent_expert_size(sim)
    test_rejects_inconsistent_copy_accounting(sim)
    test_rejects_inconsistent_runtime_cache_accounting(sim)


if __name__ == "__main__":
    main()
