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


def main() -> None:
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    sim = load_sim(repo_root)
    test_parser(sim)
    test_lru_batch_eviction(sim)
    test_cli(repo_root)
    test_rejects_inconsistent_expert_size(sim)


if __name__ == "__main__":
    main()
