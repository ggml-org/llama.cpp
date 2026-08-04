#!/usr/bin/env python3
"""Summarize rocprofv3 events from accepted target-only raw-TG regions."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a selected-region DSV4 raw-TG kernel profile"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--tsv", type=Path)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--max-clock-drift-ns", type=int, default=1_000_000)
    return parser.parse_args()


def load_json(path: Path) -> object:
    with path.open() as handle:
        return json.load(handle)


def exactly_one(directory: Path, pattern: str, required: bool = True) -> Path | None:
    paths = sorted(directory.glob(pattern))
    if len(paths) == 1:
        return paths[0]
    if not paths and not required:
        return None
    raise ValueError(f"expected {'one' if required else 'at most one'} {pattern} under {directory}, found {len(paths)}")


def clock_offset(run_dir: Path, max_drift_ns: int) -> tuple[int, int, int]:
    values = dict(
        line.split("=", 1)
        for line in (run_dir / "clock-domain.txt").read_text().splitlines()
        if "=" in line
    )
    start = int(values["start_realtime_minus_monotonic_ns"])
    end = int(values["end_realtime_minus_monotonic_ns"])
    drift = end - start
    if abs(drift) > max_drift_ns:
        raise ValueError(f"realtime/monotonic drift {drift} ns exceeds {max_drift_ns}")
    span = max(int(values.get("start_calibration_span_ns", "0")), int(values.get("end_calibration_span_ns", "0")))
    return (start + end) // 2, drift, (abs(drift) + span + 1) // 2


def read_status(path: Path) -> dict[str, str]:
    required = {"process_exit_code", "truncated", "timeout_phase", "finished_at_ns"}
    found: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text().splitlines():
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key in required:
                found[key].append(value)
    for key in required:
        if len(found[key]) != 1:
            raise ValueError(f"status field {key} occurs {len(found[key])} times")
    values = {key: entries[0] for key, entries in found.items()}
    int(values["finished_at_ns"])
    if values["process_exit_code"] != "0" or values["truncated"] != "0" or values["timeout_phase"] != "none":
        raise ValueError(f"profile process status is not clean: {values}")
    return values


def accepted_intervals(run_dir: Path, depth: int, discard: int, reps: int) -> list[tuple[int, int, int]]:
    path = run_dir / "rocprof-selected-regions.tsv"
    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"event", "benchmark", "depth", "repetition", "timestamp_monotonic_ns"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing fields: {sorted(missing)}")
        rows = list(reader)
    expected_reps = list(range(discard + 1, reps + 1))
    if len(rows) != 2 * len(expected_reps):
        raise ValueError(f"selected-region boundary count is {len(rows)}, expected {2 * len(expected_reps)}")
    intervals: list[tuple[int, int, int]] = []
    for index, rep in enumerate(expected_reps):
        begin, end = rows[2 * index:2 * index + 2]
        if begin["event"] != "resume_return" or end["event"] != "pause_call":
            raise ValueError(f"selected-region events are not an ordered resume/pause pair for repetition {rep}")
        for row in (begin, end):
            if int(row["benchmark"]) != 1 or int(row["depth"]) != depth or int(row["repetition"]) != rep:
                raise ValueError(f"selected-region boundary identity mismatch for repetition {rep}: {row}")
        start, stop = int(begin["timestamp_monotonic_ns"]), int(end["timestamp_monotonic_ns"])
        if start <= 0 or stop <= start:
            raise ValueError(f"non-positive selected region for repetition {rep}")
        if intervals and start <= intervals[-1][1]:
            raise ValueError(f"selected regions overlap or are unordered at repetition {rep}")
        intervals.append((start, stop, rep))
    return intervals


def interval_contains(intervals: list[tuple[int, int, int]], start: int, end: int) -> bool:
    return any(start >= left and end <= right for left, right, _ in intervals)


def classify_kernel(name: str) -> str:
    lower = name.lower()
    if name.startswith("ncclDevKernel"):
        return "communication_nccl"
    if "lightning_indexer_kernel" in lower:
        return "lightning_indexer"
    if ("top_k" in lower or "topk" in lower) and "topk_moe" not in lower:
        return "lid_top_k"
    if "flash_attn" in lower:
        return "flash_attention"
    if "dsv4_hc_mixes" in lower:
        return "hc_mixes"
    if "mul_mat_q<" in name and re.search(r"\(ggml_type\)(16|18),", name):
        return "routed_expert_iq2_iq3_mmq"
    if "mul_mat_q<" in name:
        return "other_quantized_matmul"
    if "quantize" in lower and ("q8_1" in lower or "mmq" in lower):
        return "activation_quantization"
    if "build_mmq_active_tiles" in lower or "mm_ids" in lower or "topk_moe" in lower:
        return "moe_routing_support"
    if name.startswith("Cijk_") or "rocblas" in lower or "gemm" in lower:
        return "dense_gemm"
    if "mask" in lower:
        return "dense_mask"
    if "rope" in lower:
        return "rope"
    if "norm" in lower:
        return "normalization"
    if "copybuffer" in lower or "fillbuffer" in lower or "memcpy" in lower:
        return "device_copy_fill"
    return "other"


def shape_key(row: dict[str, str]) -> str:
    return (
        f"wg={row['Workgroup_Size_X']}x{row['Workgroup_Size_Y']}x{row['Workgroup_Size_Z']} "
        f"grid={row['Grid_Size_X']}x{row['Grid_Size_Y']}x{row['Grid_Size_Z']}"
    )


def main() -> int:
    args = parse_args()
    if args.top < 1 or args.max_clock_drift_ns < 0:
        raise ValueError("--top must be positive and --max-clock-drift-ns non-negative")
    run_dir = args.run_dir.resolve()
    contract = load_json(run_dir / "contract.json")
    summary = load_json(run_dir / "summary.json")
    if not isinstance(contract, dict) or not isinstance(summary, dict):
        raise ValueError("contract/summary must be JSON objects")
    if contract.get("profile") != "kernel" or contract.get("profile_scope") != "accepted-target-generation-selected-regions":
        raise ValueError("artifact is not a selected-region kernel profile")
    if contract.get("mode") != "performance" or contract.get("depth_state_api") != "context":
        raise ValueError("profile requires performance mode and full-context depth state")
    if not contract.get("target_only") or contract.get("draft_model_loaded") or contract.get("speculative_flags"):
        raise ValueError("target-only profile contract is invalid")
    if contract.get("model_hash_mode") != "full":
        raise ValueError("profile requires full model hashing")
    if int(contract.get("require_accepted_stack", 0)) != 1 or contract.get("accepted_stack") != {
        "mmq_j": 16, "hc_mixes": 1, "lid_subwave": 4,
    }:
        raise ValueError("profile does not use the exact accepted J16/HC1/LID4 stack")
    if (
        int(contract.get("batch", 0)) != 512 or int(contract.get("ubatch", 0)) != 256 or
        contract.get("tensor_split") != "1/1/1/1" or
        contract.get("cache_type_k") != "f16" or contract.get("cache_type_v") != "f16" or
        int(contract.get("threads", 0)) != 12 or contract.get("load_mode") != "mmap"
    ):
        raise ValueError("profile batch/tensor-split/KV contract is invalid")
    depths = contract.get("depths")
    if not isinstance(depths, list) or len(depths) != 1 or not isinstance(depths[0], int):
        raise ValueError("profile must contain exactly one integer depth")
    reps = int(contract["raw_repetitions"])
    discard = int(contract["discard_first"])
    accepted_reps = int(contract["accepted_repetitions"])
    n_gen = int(contract["n_gen"])
    if n_gen != 32 or reps < 6 or discard != 1 or accepted_reps != reps - 1 or accepted_reps < 5:
        raise ValueError("profile requires tg32, at least 6 raw repetitions, exactly first discarded, and at least 5 accepted")
    if int(contract.get("profile_skip_repetitions", -1)) != discard:
        raise ValueError("profiler skip count does not match discarded repetitions")
    if not summary.get("complete") or not summary.get("stable"):
        raise ValueError("raw-TG result is not complete/stable")
    read_status(run_dir / "status.txt")
    if (run_dir / "source-status.txt").read_text().strip() or (run_dir / "untracked-files.sha256").read_text().strip():
        raise ValueError("profile source identity is not clean")
    manifest = (run_dir / "manifest.txt").read_text()
    model_section = manifest.split("=== model_files ===", 1)
    if len(model_section) != 2 or "hash_mode=full" not in model_section[1].split("=== model_metadata ===", 1)[0]:
        raise ValueError("manifest does not attest full model hashing")
    model_hash_lines = re.findall(r"(?m)^[0-9a-f]{64}  .*\.gguf$", model_section[1].split("=== model_metadata ===", 1)[0])
    if len(model_hash_lines) != 3:
        raise ValueError(f"manifest has {len(model_hash_lines)} full GGUF hashes, expected 3")
    if "-- all resolved dependency hashes (local + ROCm/system DSOs) --" not in manifest:
        raise ValueError("manifest is missing all-resolved-DSO hashes")

    result_rows = [json.loads(line) for line in (run_dir / "result.jsonl").read_text().splitlines() if line.strip()]
    if len(result_rows) != 1:
        raise ValueError(f"profile requires exactly one result row, found {len(result_rows)}")
    samples = result_rows[0].get("samples_ns")
    if not isinstance(samples, list) or len(samples) != reps or not all(isinstance(v, int) and v > 0 for v in samples):
        raise ValueError("result samples_ns does not match repetition contract")
    accepted_samples = samples[discard:]
    accepted_wall_ns = sum(accepted_samples)
    evaluated_tokens = accepted_reps * n_gen

    offset, drift, uncertainty = clock_offset(run_dir, args.max_clock_drift_ns)
    intervals = accepted_intervals(run_dir, depths[0], discard, reps)
    rocprof_dir = run_dir / "rocprof"
    kernel_path = exactly_one(rocprof_dir, "*_kernel_trace.csv")
    agent_path = exactly_one(rocprof_dir, "*_agent_info.csv")
    copy_path = exactly_one(rocprof_dir, "*_memory_copy_trace.csv")
    rccl_path = exactly_one(rocprof_dir, "*_rccl_api_trace.csv")
    hip_path = exactly_one(rocprof_dir, "*_hip_api_trace.csv")
    assert kernel_path is not None and agent_path is not None and copy_path is not None and rccl_path is not None and hip_path is not None

    agents: dict[str, dict[str, str]] = {}
    with agent_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("Agent_Type") == "GPU":
                agents[f"Agent {int(row['Logical_Node_Id'])}"] = {
                    "product_name": row["Product_Name"],
                    "gpu_id": row["Gpu_Id"],
                    "domain": row["Domain"],
                    "location_id": row["Location_Id"],
                }
    if len(agents) != 4:
        raise ValueError(f"expected four GPU agents, found {len(agents)}")

    family_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    agent_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    kernel_totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    outside_events = 0
    kernel_rows = 0
    required_kernel_fields = {
        "Agent_Id", "Kernel_Name", "Start_Timestamp", "End_Timestamp",
        "Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z",
        "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z",
    }
    with kernel_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required_kernel_fields.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"kernel trace missing fields: {sorted(missing)}")
        for row in reader:
            kernel_rows += 1
            start, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
            if end < start:
                raise ValueError(f"kernel row {reader.line_num} ends before it starts")
            if not interval_contains(intervals, start, end):
                outside_events += 1
            agent = row["Agent_Id"]
            if agent not in agents:
                raise ValueError(f"kernel row references unknown GPU agent {agent}")
            duration = end - start
            family = classify_kernel(row["Kernel_Name"])
            family_totals[family][0] += 1
            family_totals[family][1] += duration
            agent_totals[agent][0] += 1
            agent_totals[agent][1] += duration
            kernel_totals[(row["Kernel_Name"], shape_key(row))][0] += 1
            kernel_totals[(row["Kernel_Name"], shape_key(row))][1] += duration
    if kernel_rows == 0:
        raise ValueError("kernel trace is empty")
    if outside_events:
        raise ValueError(f"{outside_events} kernel events fall outside accepted generation intervals")
    total_device_ns = sum(value[1] for value in family_totals.values())
    if total_device_ns <= 0:
        raise ValueError("summed device time is zero")

    def aggregate_api(path: Path, key: str) -> tuple[dict[str, dict[str, int]], int, str]:
        output: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        outside = 0
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {key, "Start_Timestamp", "End_Timestamp"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} missing fields: {sorted(missing)}")
            for row in reader:
                name = row[key]
                if not name:
                    raise ValueError(f"{path}:{reader.line_num} has a blank {key}")
                start, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
                if end < start:
                    raise ValueError(f"{path}:{reader.line_num} ends before it starts")
                if not interval_contains(intervals, start, end):
                    outside += 1
                output[name][0] += 1
                output[name][1] += end - start
        rendered = {name: {"calls": value[0], "duration_ns": value[1]} for name, value in output.items()}
        return rendered, outside, "present_with_events" if rendered else "present_empty"

    copies, copy_outside, copy_status = aggregate_api(copy_path, "Direction")
    rccl, rccl_outside, rccl_status = aggregate_api(rccl_path, "Function")
    hip, hip_outside, hip_status = aggregate_api(hip_path, "Function")
    if hip_status != "present_with_events":
        raise ValueError("HIP runtime trace is present but empty for accepted GPU decode")
    if copy_outside or rccl_outside or hip_outside:
        raise ValueError(
            f"non-kernel events outside selected intervals: copy={copy_outside} rccl={rccl_outside} hip={hip_outside}"
        )

    families = []
    for name, (calls, duration_ns) in sorted(family_totals.items(), key=lambda item: (-item[1][1], item[0])):
        families.append({
            "family": name,
            "calls": calls,
            "duration_ns": duration_ns,
            "share_of_summed_device_time": duration_ns / total_device_ns,
            "device_ms_per_token": duration_ns / evaluated_tokens / 1e6,
        })
    top_kernels = []
    for (name, shape), (calls, duration_ns) in sorted(kernel_totals.items(), key=lambda item: (-item[1][1], item[0]))[:args.top]:
        top_kernels.append({
            "kernel_name": name,
            "shape": shape,
            "calls": calls,
            "duration_ns": duration_ns,
            "share_of_summed_device_time": duration_ns / total_device_ns,
        })
    per_agent = {
        agent: {
            "calls": values[0],
            "duration_ns": values[1],
            "share_of_summed_device_time": values[1] / total_device_ns,
        }
        for agent, values in sorted(agent_totals.items())
    }
    output = {
        "complete": True,
        "scope": "accepted target-generation ROCTx selected regions only",
        "run_dir": str(run_dir),
        "depth": depths[0],
        "n_gen": n_gen,
        "raw_repetitions": reps,
        "discard_first": discard,
        "profiled_repetitions": accepted_reps,
        "evaluated_target_tokens": evaluated_tokens,
        "accepted_wall_ns": accepted_wall_ns,
        "accepted_median_ns": sorted(accepted_samples)[len(accepted_samples) // 2],
        "total_summed_device_ns": total_device_ns,
        "kernel_dispatches": kernel_rows,
        "clock_offset_ns": offset,
        "clock_drift_ns": drift,
        "clock_uncertainty_ns": uncertainty,
        "accepted_intervals_monotonic_ns": [
            {"start": start, "end": end, "repetition": rep} for start, end, rep in intervals
        ],
        "outside_selected_interval_events": {
            "kernel": outside_events, "memory_copy": copy_outside, "rccl": rccl_outside, "hip": hip_outside,
        },
        "trace_files": {
            "kernel": str(kernel_path), "agent": str(agent_path),
            "memory_copy": str(copy_path), "rccl": str(rccl_path), "hip": str(hip_path),
        },
        "trace_domain_status": {
            "kernel": "present_with_events", "memory_copy": copy_status,
            "rccl": rccl_status, "hip": hip_status,
        },
        "families": families,
        "top_kernel_shapes": top_kernels,
        "per_agent": per_agent,
        "memory_copies": copies,
        "rccl_api": rccl,
        "hip_api": hip,
        "classification_caveat": (
            "Families are exclusive kernel-name matches. routed_expert_iq2_iq3_mmq is inferred from the model's "
            "IQ2_XXS/IQ3_XXS MMQ types and must be checked against grid/call evidence; other may contain unclassified "
            "attention, mask, projection, elementwise, or selector support. Durations are summed across devices/queues."
        ),
    }

    print(f"run_dir={run_dir}")
    print(f"depth={depths[0]} profiled_repetitions={accepted_reps} evaluated_target_tokens={evaluated_tokens}")
    print(f"accepted_wall_ms={accepted_wall_ns / 1e6:.3f}")
    print(f"kernel_dispatches={kernel_rows} summed_device_ms={total_device_ns / 1e6:.3f}")
    print(f"clock_drift_ns={drift} clock_uncertainty_ns={uncertainty} outside_events=0")
    print("boundary_source=llama-bench CLOCK_MONOTONIC resume_return/pause_call")
    print("\nshare_pct\tdevice_ms\tms_per_token\tcalls\tfamily")
    for row in families:
        print(
            f"{100 * row['share_of_summed_device_time']:.3f}\t{row['duration_ns'] / 1e6:.3f}\t"
            f"{row['device_ms_per_token']:.6f}\t{row['calls']}\t{row['family']}"
        )
    print("\n[top kernel/shape groups]")
    print("share_pct\tdevice_ms\tcalls\tshape\tname")
    for row in top_kernels:
        short_name = row["kernel_name"] if len(row["kernel_name"]) <= 180 else row["kernel_name"][:177] + "..."
        print(
            f"{100 * row['share_of_summed_device_time']:.3f}\t{row['duration_ns'] / 1e6:.3f}\t"
            f"{row['calls']}\t{row['shape']}\t{short_name}"
        )
    print("\n[per-agent summed kernel time]")
    print("agent\tdevice_ms\tcalls\tshare_pct")
    for agent, item in per_agent.items():
        print(f"{agent}\t{item['duration_ns'] / 1e6:.3f}\t{item['calls']}\t{100 * item['share_of_summed_device_time']:.3f}")
    print("\n[trace domains]")
    for domain, status_value, values in (
        ("memory_copy", copy_status, copies), ("rccl", rccl_status, rccl), ("hip", hip_status, hip),
    ):
        calls = sum(item["calls"] for item in values.values())
        duration_ns = sum(item["duration_ns"] for item in values.values())
        print(f"{domain}\t{status_value}\tcalls={calls}\tduration_ms={duration_ns / 1e6:.3f}")
    print("[top HIP runtime calls]")
    for name, item in sorted(hip.items(), key=lambda entry: (-entry[1]["duration_ns"], entry[0]))[:10]:
        print(f"{item['duration_ns'] / 1e6:.3f} ms\t{item['calls']}\t{name}")
    print(f"\nclassification_caveat={output['classification_caveat']}")

    if args.json:
        args.json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if args.tsv:
        with args.tsv.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["depth", "profiled_repetitions", "target_tokens", "family", "calls", "duration_ns", "share", "device_ms_per_token"])
            for row in families:
                writer.writerow([
                    depths[0], accepted_reps, evaluated_tokens, row["family"], row["calls"], row["duration_ns"],
                    f"{row['share_of_summed_device_time']:.9f}", f"{row['device_ms_per_token']:.9f}",
                ])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)