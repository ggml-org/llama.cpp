#!/usr/bin/env python3
"""Non-GPU fixture tests for DSV4 raw-TG summary and scheduler parsers."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parent
SUMMARIZE = ROOT / "summarize-tg.py"
PARSE = ROOT / "parse-sched-debug.py"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise AssertionError(f"command failed {result.returncode}: {' '.join(args)}\n{result.stdout}\n{result.stderr}")
    return result


def record(depth: int, accepted_ns: list[int] | None = None, **changes: object) -> dict[str, object]:
    if accepted_ns is None:
        accepted_ns = [3_200_000_000, 3_210_000_000, 3_190_000_000, 3_205_000_000, 3_195_000_000]
    samples_ns = [4_000_000_000, *accepted_ns]
    value: dict[str, object] = {
        "model_filename": "fixture.gguf",
        "n_prompt": 0,
        "n_gen": 32,
        "n_depth": depth,
        "n_batch": 512,
        "n_ubatch": 256,
        "split_mode": "tensor",
        "tensor_split": "1.00/1.00/1.00/1.00",
        "type_k": "f16",
        "type_v": "f16",
        "flash_attn": 1,
        "samples_ns": samples_ns,
        "samples_ts": [32e9 / ns for ns in samples_ns],
    }
    value.update(changes)
    return value


def write_jsonl(path: pathlib.Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in records))


def summarize(tmp: pathlib.Path, records: list[dict[str, object]], expected_depths: str = "0,2048") -> dict[str, object]:
    raw = tmp / "result.jsonl"
    out = tmp / "summary.json"
    tsv = tmp / "summary.tsv"
    write_jsonl(raw, records)
    run(
        sys.executable, str(SUMMARIZE), str(raw), "--json", str(out), "--tsv", str(tsv),
        "--expected-depths", expected_depths, "--expected-gen", "32", "--expected-reps", "6",
        "--discard-first", "1", "--stability-limit", "0.03", "--expected-batch", "512",
        "--expected-ubatch", "256", "--expected-tensor-split", "1/1/1/1",
    )
    assert tsv.read_text().startswith("run_complete\trun_stable\tdepth")
    return json.loads(out.read_text())


def test_summary(tmp: pathlib.Path) -> None:
    good = summarize(tmp, [record(0), record(2048)])
    assert good["complete"] is True and good["stable"] is True
    assert good["observational_baseline_accepted"] is True
    assert all(row["accepted_repetitions"] == 5 for row in good["records"])
    assert all(row["mad_over_median"] <= 0.03 for row in good["records"])

    missing = summarize(tmp, [record(0)])
    assert missing["complete"] is False and missing["missing_depths"] == [2048]

    unstable_ns = [1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000, 5_000_000_000]
    unstable = summarize(tmp, [record(0, unstable_ns), record(2048, unstable_ns)])
    assert unstable["complete"] is True and unstable["stable"] is False

    bad_contract = summarize(tmp, [record(0), record(2048, flash_attn=0)])
    assert bad_contract["complete"] is False
    assert any("flash_attn" in error for error in bad_contract["contract_errors"])


def scheduler_log(cpu_top_k: bool = False) -> str:
    top_backend = "CPU" if cpu_top_k else "Meta("
    return f"""llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: generation run 1/1
## SPLIT #0: Meta(ROCm0,ROCm1,ROCm2,ROCm3) # 1 inputs
node #  1 (   MUL_MAT): decode_out           ( 1KiB) [Meta(    2.sup] use=1,c=1:
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: depth run 1/1
llama-bench: benchmark 2/2: generation run 1/1
## SPLIT #0: Meta(ROCm0,ROCm1,ROCm2,ROCm3) # 2 inputs
node # 10 (     TOP_K): lid_top_k-1          ( 2KiB) [{top_backend:<5}    2.sup] use=1,c=1:
node # 11 ( LIGHTNING): lid_score_masked-1   ( 4KiB) [Meta(    2.sup] use=1,c=1:
"""


def parse_scheduler(tmp: pathlib.Path, cpu_top_k: bool = False) -> dict[str, object]:
    log = tmp / "bench.log"
    out = tmp / "scheduler.json"
    tsv = tmp / "scheduler.tsv"
    log.write_text(scheduler_log(cpu_top_k))
    run(
        sys.executable, str(PARSE), str(log), "--depths", "0,2048",
        "--json", str(out), "--tsv", str(tsv),
    )
    assert tsv.read_text().startswith("run_complete\tresidency_ok\tdepth")
    return json.loads(out.read_text())


def test_scheduler(tmp: pathlib.Path) -> None:
    good = parse_scheduler(tmp)
    assert good["complete"] is True and good["rocm_residency_ok"] is True
    depth_2048 = next(item for item in good["records"] if item["depth"] == 2048)
    assert depth_2048["top_k_total"] == 1 and depth_2048["lid_total"] == 1
    assert depth_2048["cpu_splits"] == 0
    assert depth_2048["total_split_input_copies"] == 2
    assert depth_2048["gpu_meta_split_input_copies"] == 2

    bad = parse_scheduler(tmp, cpu_top_k=True)
    assert bad["complete"] is True and bad["rocm_residency_ok"] is False
    depth_2048 = next(item for item in bad["records"] if item["depth"] == 2048)
    assert depth_2048["top_k_cpu"] == 1


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dsv4-tg-tools-") as name:
        tmp = pathlib.Path(name)
        test_summary(tmp)
        test_scheduler(tmp)
    print("dsv4 raw-TG tool fixtures: PASS")


if __name__ == "__main__":
    main()