#!/usr/bin/env python3
"""Non-GPU tests for DSV4 selected-region communication forensics."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent
TOOL = ROOT / "analyze-tg-communication.py"
spec = importlib.util.spec_from_file_location("dsv4_tg_communication", TOOL)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def expect_value_error(call, needle: str) -> None:
    try:
        call()
    except ValueError as exc:
        assert needle in str(exc), exc
    else:
        raise AssertionError(f"expected ValueError containing {needle!r}")


def replace_text(path: pathlib.Path, text: str) -> None:
    path.unlink()
    path.write_text(text)


def clone_fixture(source: pathlib.Path, destination: pathlib.Path) -> pathlib.Path:
    shutil.copytree(source, destination, copy_function=os.link)
    return destination


def write_csv(path: pathlib.Path, fields: list[str], rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_fixture(root: pathlib.Path) -> pathlib.Path:
    run = root / "good"
    rocprof = run / "rocprof"
    rocprof.mkdir(parents=True)
    (run / "contract.json").write_text(json.dumps({
        "profile": "kernel", "n_gen": 32, "discard_first": 1, "raw_repetitions": 6,
    }))
    intervals = [
        {"start": rep * 1_000_000, "end": rep * 1_000_000 + 900_000, "repetition": rep}
        for rep in range(2, 7)
    ]
    per_repetition = [
        {"repetition": rep, "wall_ns": 2_000_000_000 + rep}
        for rep in range(2, 7)
    ]
    (run / "profile-summary.json").write_text(json.dumps({
        "complete": True,
        "family_attribution_complete": True,
        "profiled_throughput_eligible": False,
        "csa_decision_eligible": False,
        "outside_selected_interval_events": {"kernel": 0, "memory_copy": 0, "rccl": 0, "hip": 0},
        "moe_dispatch_contract": {"block_count": 43, "gpu_agents": 4},
        "accepted_intervals_monotonic_ns": intervals,
        "profiled_repetitions": 5,
        "profiled_wall_stable": False,
        "depth": 16384,
        "per_repetition": per_repetition,
    }))
    write_csv(
        rocprof / "dsv4-tg_agent_info.csv",
        ["Logical_Node_Id", "Agent_Type", "Product_Name", "Gpu_Id"],
        (
            {"Logical_Node_Id": agent, "Agent_Type": "GPU", "Product_Name": "V620", "Gpu_Id": 100 + agent}
            for agent in range(1, 5)
        ),
    )
    kernel_fields = [
        "Agent_Id", "Kernel_Name", "Correlation_Id", "Start_Timestamp", "End_Timestamp", "Queue_Id", "Stream_Id",
    ]
    kernel_path = rocprof / "dsv4-tg_kernel_trace.csv"
    correlation = 1
    with kernel_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=kernel_fields)
        writer.writeheader()
        for rep in range(2, 7):
            base = rep * 1_000_000 + 10_000
            for agent in range(1, 5):
                for group in range(2752):
                    start = base + group * 100
                    writer.writerow({
                        "Agent_Id": f"Agent {agent}",
                        "Kernel_Name": "ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)",
                        "Correlation_Id": correlation,
                        "Start_Timestamp": start,
                        "End_Timestamp": start + 50,
                        "Queue_Id": 2 * agent - 1,
                        "Stream_Id": 13 + agent,
                    })
                    correlation += 1
    rccl_fields = ["Domain", "Function", "Process_Id", "Thread_Id", "Correlation_Id", "Start_Timestamp", "End_Timestamp"]
    rccl_path = rocprof / "dsv4-tg_rccl_api_trace.csv"
    correlation = 1_000_000
    with rccl_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rccl_fields)
        writer.writeheader()
        for rep in range(2, 7):
            base = rep * 1_000_000 + 400_000
            for group in range(2752):
                start = base + group * 100
                functions = ["ncclGroupStart"]
                functions.extend(["ncclAllReduce"] * 4)
                functions.extend(["ncclCommGetAsyncError"] * 4)
                functions.append("ncclGroupEnd")
                for index, function in enumerate(functions):
                    writer.writerow({
                        "Domain": "RCCL_API", "Function": function, "Process_Id": 1, "Thread_Id": 1,
                        "Correlation_Id": correlation, "Start_Timestamp": start + index,
                        "End_Timestamp": start + index + 1,
                    })
                    correlation += 1
    return run


def run_tool(run: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), str(run), *args],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def main() -> None:
    assert module.percentile([1, 2, 3, 4], 0.5) == 2
    assert module.percentile([1, 2, 3, 4], 0.95) == 4
    expect_value_error(lambda: module.percentile([], 0.5), "requires values")
    expect_value_error(lambda: module.percentile([1], 0), "0 < fraction")

    assert module.merge_intervals([(5, 10), (1, 3), (3, 6), (12, 13)]) == [(1, 10), (12, 13)]
    assert module.interval_duration([(1, 10), (12, 13)]) == 10
    assert module.intersection_duration([(1, 5), (7, 12)], [(3, 9)]) == 4
    expect_value_error(lambda: module.merge_intervals([(2, 1)]), "ends before")

    intervals = [(10, 20, 2), (30, 40, 3)]
    assert module.containing_repetition(intervals, 11, 19) == 2
    expect_value_error(lambda: module.containing_repetition(intervals, 19, 31), "belongs to 0")
    expect_value_error(lambda: module.containing_repetition([(10, 30, 2), (20, 40, 3)], 21, 22), "belongs to 2")

    expected = {
        "ncclAllReduce": 11008,
        "ncclCommGetAsyncError": 11008,
        "ncclGroupEnd": 2752,
        "ncclGroupStart": 2752,
    }
    module.require_exact_counts(Counter(expected), expected, "good")
    bad = Counter(expected)
    bad["ncclAllReduce"] -= 1
    expect_value_error(lambda: module.require_exact_counts(bad, expected, "bad"), "count mismatch")
    extra = Counter(expected)
    extra["ncclBroadcast"] = 1
    expect_value_error(lambda: module.require_exact_counts(extra, expected, "extra"), "count mismatch")

    with tempfile.TemporaryDirectory(prefix="dsv4-tg-communication-") as name:
        root = pathlib.Path(name)
        good = make_fixture(root / "accepted")
        output = good / "forensics.json"
        result = run_tool(good, "--json", str(output), "--top-long-kernels", "3")
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert "traced_collective_api=ncclAllReduce_only" in result.stdout
        value = json.loads(output.read_text())
        assert value["complete"] and value["critical_path_proven"] is False
        assert value["cross_run_cadence_invariant"] is False
        run = value["runs"][0]
        assert run["count_contract_complete"]
        assert run["groups_per_token"] == 86 and run["rank_allreduce_calls_per_token"] == 344
        assert run["expected_counts_per_repetition"] == {**expected, "nccl_device_kernels": 11008}
        assert run["message_metadata"] == {
            "available": False,
            "supported_rccl_schema_exact": True,
            "reason": (
                "the exact supported rocprof RCCL API schema contains no count/datatype/buffer/communicator/rank/stream/"
                "message-byte arguments; generic NCCL device-kernel launch geometry is not an attested payload size"
            ),
        }
        assert run["api_kernel_correlation"]["available"] is False
        assert len(run["longest_device_kernels"]) == 3
        assert len(run["repetitions"]) == 5
        assert all(item["kernel_count"] == 11008 for item in run["repetitions"])
        assert all(
            agent["overlap_fraction_of_nccl_union"] == 0.0
            for item in run["repetitions"] for agent in item["per_agent"].values()
        )

        missing_outside = clone_fixture(good, root / "missing-outside")
        summary_path = missing_outside / "profile-summary.json"
        summary = json.loads(summary_path.read_text())
        del summary["outside_selected_interval_events"]
        replace_text(summary_path, json.dumps(summary))
        result = run_tool(missing_outside)
        assert result.returncode == 2 and "missing or malformed outside" in result.stderr

        unknown_schema = clone_fixture(good, root / "unknown-schema")
        rccl_path = unknown_schema / "rocprof" / "dsv4-tg_rccl_api_trace.csv"
        lines = rccl_path.read_text().splitlines()
        lines[0] += ",Payload_Bytes"
        replace_text(rccl_path, "\n".join(lines) + "\n")
        result = run_tool(unknown_schema)
        assert result.returncode == 2 and "exact supported rocprof RCCL schema" in result.stderr

        duplicate_rep = clone_fixture(good, root / "duplicate-repetition")
        summary_path = duplicate_rep / "profile-summary.json"
        summary = json.loads(summary_path.read_text())
        summary["per_repetition"].append(dict(summary["per_repetition"][0]))
        replace_text(summary_path, json.dumps(summary))
        result = run_tool(duplicate_rep)
        assert result.returncode == 2 and "duplicate repetition" in result.stderr

        result = subprocess.run(
            [sys.executable, str(TOOL), str(good), str(good)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        assert result.returncode == 2 and "must be distinct" in result.stderr

        result = run_tool(good, "--json", str(root / "missing-parent" / "forensics.json"))
        assert result.returncode == 2
        assert "COMPLETE" not in result.stdout

    assert module.EXPECTED_BLOCK_COUNT == 43
    assert module.EXPECTED_ALLREDUCES_PER_BLOCK_TOKEN == 2
    assert module.EXPECTED_N_GEN * module.EXPECTED_BLOCK_COUNT * 2 == 2752
    assert module.EXPECTED_N_GEN * module.EXPECTED_BLOCK_COUNT * 2 * module.EXPECTED_GPU_AGENTS == 11008
    assert module.EXPECTED_RCCL_SCHEMA == {
        "Domain", "Function", "Process_Id", "Thread_Id", "Correlation_Id", "Start_Timestamp", "End_Timestamp",
    }
    print("dsv4 raw-TG communication forensic fixtures: PASS")


if __name__ == "__main__":
    main()