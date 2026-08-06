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


def containing_repetition(intervals: list[tuple[int, int, int]], start: int, end: int) -> int | None:
    matches = [rep for left, right, rep in intervals if start >= left and end <= right]
    if len(matches) > 1:
        raise ValueError(f"trace event is contained by multiple selected regions: {start}-{end}")
    return matches[0] if matches else None


# Exact F16 concat signatures for the fully hashed 43-layer DSV4-Flash model.
# The grids distinguish CSA/HCA K and mask materialization at the two profiled
# decision depths. Counts are validated globally, per GPU, and per repetition.
DSV4_ATTENTION_CONCAT_SIGNATURES: dict[int, dict[tuple[int, str], tuple[str, str]]] = {
    16384: {
        (2, "2490368"): ("csa_k_concat", "csa_k_concat"),
        (2, "2359296"): ("csa_k_concat", "csa_k_concat"),
        (2, "393216"): ("hca_k_concat", "hca_k_concat"),
        (0, "4864"): ("csa_mask_concat", "csa_mask_concat"),
        (0, "4608"): ("csa_mask_concat", "csa_mask_concat"),
        (0, "768"): ("hca_mask_concat", "hca_mask_concat"),
    },
    65536: {
        (2, "8781824"): ("csa_k_concat", "csa_k_concat"),
        (2, "8650752"): ("csa_k_concat", "csa_k_concat"),
        (2, "524288"): ("hca_k_concat", "hca_k_concat"),
        (0, "17152"): ("csa_mask_concat", "csa_mask_concat"),
        (0, "16896"): ("csa_mask_concat", "csa_mask_concat"),
        (0, "1024"): ("hca_mask_concat", "hca_mask_concat"),
    },
}


def classify_kernel(row: dict[str, str], depth: int) -> tuple[str, str | None, str | None, str | None]:
    """Return an exclusive family plus optional exact MoE/attention roles.

    The MMVQ and attention-concat signatures are tied to the fully hashed
    V4-Flash IQ2_M inventory. Dispatch-count contracts later in the parser
    prevent partial or coincidental matches from being accepted.
    """
    name = row["Kernel_Name"]
    lower = name.lower()
    if name.startswith("ncclDevKernel"):
        return "communication_nccl", None, None, None
    if "lightning_indexer_kernel" in lower:
        return "lightning_indexer", None, None, None
    if ("top_k" in lower or "topk" in lower) and "topk_moe" not in lower:
        return "lid_top_k", None, None, None
    if "flash_attn" in lower:
        return "flash_attention", None, None, None
    if "dsv4_hc_" in lower:
        return "hc_mixes", None, None, None

    concat_names = {
        0: "void concat_cont<unsigned short, 0>(unsigned short const*, unsigned short const*, unsigned short*, long, long, long, long, long, long)",
        2: "void concat_cont<unsigned short, 2>(unsigned short const*, unsigned short const*, unsigned short*, long, long, long, long, long, long)",
    }
    concat_dim = next((dim for dim, exact_name in concat_names.items() if name == exact_name), None)
    if depth in DSV4_ATTENTION_CONCAT_SIGNATURES and name.startswith("void concat_cont<unsigned short,") and concat_dim is None:
        raise ValueError(f"unknown exact DSV4 F16 attention concat kernel name at depth {depth}: {name}")
    if concat_dim is not None and depth in DSV4_ATTENTION_CONCAT_SIGNATURES:
        workgroup = (row["Workgroup_Size_X"], row["Workgroup_Size_Y"], row["Workgroup_Size_Z"])
        grid = (row["Grid_Size_X"], row["Grid_Size_Y"], row["Grid_Size_Z"])
        key = (concat_dim, grid[0])
        signature = DSV4_ATTENTION_CONCAT_SIGNATURES[depth].get(key)
        if workgroup != ("256", "1", "1") or grid[1:] != ("1", "1") or signature is None:
            raise ValueError(
                f"unknown exact DSV4 F16 attention concat signature at depth {depth}: "
                f"dim={concat_dim} workgroup={workgroup} grid={grid}"
            )
        family, role = signature
        return family, None, None, role

    mmvq = re.search(r"mul_mat_vec_q<\(ggml_type\)(\d+),\s*1,\s*(true|false),", name)
    if mmvq and (row["Workgroup_Size_X"], row["Workgroup_Size_Y"], row["Workgroup_Size_Z"]) == ("32", "1", "1"):
        quant_type = int(mmvq.group(1))
        fused = mmvq.group(2) == "true"
        grid = (row["Grid_Size_X"], row["Grid_Size_Y"], row["Grid_Size_Z"])
        if grid == ("16384", "6", "1") and quant_type in {16, 22} and not fused:
            return "routed_expert_quant_matmul", "routed_gate_up", f"routed_gate_up_type{quant_type}", None
        if grid == ("131072", "6", "1") and quant_type in {18, 39} and not fused:
            return "routed_expert_quant_matmul", "routed_down", f"routed_down_type{quant_type}", None
        if grid == ("65536", "1", "1") and quant_type in {13, 14} and not fused:
            return "shared_expert_quant_matmul", "shared_gate_up", f"shared_gate_up_type{quant_type}", None
        if grid == ("131072", "1", "1") and quant_type in {8, 14} and fused:
            return "shared_expert_quant_matmul", "shared_down", f"shared_down_type{quant_type}", None
    if "mul_mat_q<" in name or "mul_mat_vec_q<" in name:
        return "non_moe_quantized_matmul", None, None, None
    if "quantize" in lower and ("q8_1" in lower or "mmq" in lower):
        return "activation_quantization", None, None, None
    if "build_mmq_active_tiles" in lower or "mm_ids" in lower or "topk_moe" in lower:
        return "moe_routing_support", None, None, None
    if name.startswith("Cijk_") or "rocblas" in lower or "gemm" in lower:
        return "dense_gemm", None, None, None
    if "mask" in lower:
        return "dense_mask", None, None, None
    if "rope" in lower:
        return "rope", None, None, None
    if "norm" in lower:
        return "normalization", None, None, None
    if "copybuffer" in lower or "fillbuffer" in lower or "memcpy" in lower:
        return "device_copy_fill", None, None, None
    return "other", None, None, None


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
    if not summary.get("complete"):
        raise ValueError("raw-TG result is incomplete")
    summary_records = summary.get("records")
    if not isinstance(summary_records, list) or len(summary_records) != 1 or summary_records[0].get("depth") != depths[0]:
        raise ValueError("raw-TG summary does not contain exactly the profiled depth")
    profile_wall_stable = bool(summary.get("stable")) and bool(summary_records[0].get("stable"))
    profile_wall_mad_over_median = float(summary_records[0]["mad_over_median"])
    if profile_wall_stable != (profile_wall_mad_over_median <= float(summary_records[0]["stability_limit"])):
        raise ValueError("raw-TG summary stability fields are inconsistent")
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
    repetition_family_totals: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    moe_role_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    moe_signature_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    moe_signature_agent_calls: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    moe_signature_repetition_calls: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    attention_concat_role_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    attention_concat_agent_calls: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    attention_concat_repetition_calls: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
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
            repetition = containing_repetition(intervals, start, end)
            if repetition is None:
                outside_events += 1
            agent = row["Agent_Id"]
            if agent not in agents:
                raise ValueError(f"kernel row references unknown GPU agent {agent}")
            duration = end - start
            family, moe_role, moe_signature, attention_concat_role = classify_kernel(row, depths[0])
            family_totals[family][0] += 1
            family_totals[family][1] += duration
            if moe_role is not None:
                assert moe_signature is not None
                moe_role_totals[moe_role][0] += 1
                moe_role_totals[moe_role][1] += duration
                moe_signature_totals[moe_signature][0] += 1
                moe_signature_totals[moe_signature][1] += duration
                moe_signature_agent_calls[moe_signature][agent] += 1
                if repetition is not None:
                    moe_signature_repetition_calls[moe_signature][repetition] += 1
            if attention_concat_role is not None:
                attention_concat_role_totals[attention_concat_role][0] += 1
                attention_concat_role_totals[attention_concat_role][1] += duration
                attention_concat_agent_calls[attention_concat_role][agent] += 1
                if repetition is not None:
                    attention_concat_repetition_calls[attention_concat_role][repetition] += 1
            if repetition is not None:
                repetition_family_totals[repetition][family][0] += 1
                repetition_family_totals[repetition][family][1] += duration
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

    block_counts = re.findall(r"(?m)^deepseek4\.block_count=(\d+)$", manifest)
    if len(block_counts) != 1:
        raise ValueError(f"manifest has {len(block_counts)} DeepSeek-V4 block-count records, expected exactly one")
    block_count = int(block_counts[0])
    if block_count != 43:
        raise ValueError(f"DeepSeek-V4 block count is {block_count}, expected exact profile model value 43")

    signature_layer_multiplicity = {
        "routed_gate_up_type16": (42, 2),
        "routed_gate_up_type22": (1, 2),
        "routed_down_type18": (41, 1),
        "routed_down_type39": (2, 1),
        "shared_gate_up_type13": (42, 2),
        "shared_gate_up_type14": (1, 2),
        "shared_down_type14": (42, 1),
        "shared_down_type8": (1, 1),
    }
    expected_signature_calls = {
        signature: evaluated_tokens * len(agents) * layers * multiplicity
        for signature, (layers, multiplicity) in signature_layer_multiplicity.items()
    }
    actual_signature_calls = {signature: moe_signature_totals[signature][0] for signature in signature_layer_multiplicity}
    if set(moe_signature_totals) != set(signature_layer_multiplicity) or actual_signature_calls != expected_signature_calls:
        raise ValueError(
            f"exact DSV4 MoE signature dispatch contract mismatch: actual={actual_signature_calls} "
            f"expected={expected_signature_calls} signatures={sorted(moe_signature_totals)}"
        )
    for signature, (layers, multiplicity) in signature_layer_multiplicity.items():
        expected_agent_calls = evaluated_tokens * layers * multiplicity
        actual_agents = dict(moe_signature_agent_calls[signature])
        if set(actual_agents) != set(agents) or any(value != expected_agent_calls for value in actual_agents.values()):
            raise ValueError(
                f"exact DSV4 MoE per-agent dispatch mismatch for {signature}: "
                f"actual={actual_agents} expected_each={expected_agent_calls}"
            )
        expected_repetition_calls = n_gen * len(agents) * layers * multiplicity
        actual_repetitions = dict(moe_signature_repetition_calls[signature])
        expected_repetitions = set(range(discard + 1, reps + 1))
        if set(actual_repetitions) != expected_repetitions or any(
                value != expected_repetition_calls for value in actual_repetitions.values()):
            raise ValueError(
                f"exact DSV4 MoE per-repetition dispatch mismatch for {signature}: "
                f"actual={actual_repetitions} expected_each={expected_repetition_calls}"
            )
    expected_moe_calls = {
        "routed_gate_up": evaluated_tokens * block_count * len(agents) * 2,
        "routed_down": evaluated_tokens * block_count * len(agents),
        "shared_gate_up": evaluated_tokens * block_count * len(agents) * 2,
        "shared_down": evaluated_tokens * block_count * len(agents),
    }
    actual_moe_calls = {role: moe_role_totals[role][0] for role in expected_moe_calls}
    if actual_moe_calls != expected_moe_calls:
        raise ValueError(f"exact DSV4 MoE role dispatch contract mismatch: actual={actual_moe_calls} expected={expected_moe_calls}")
    moe_dispatch_contract: dict[str, object] = {
        "complete": True, "block_count": block_count, "gpu_agents": len(agents),
        "target_tokens": evaluated_tokens, "actual_role_calls": actual_moe_calls,
        "expected_role_calls": expected_moe_calls, "actual_signature_calls": actual_signature_calls,
        "expected_signature_calls": expected_signature_calls, "per_agent_exact": True,
        "per_repetition_exact": True,
    }

    attention_concat_dispatch_contract: dict[str, object] = {
        "applicable": depths[0] in DSV4_ATTENTION_CONCAT_SIGNATURES,
        "complete": False,
    }
    if depths[0] in DSV4_ATTENTION_CONCAT_SIGNATURES:
        concat_role_layers = {
            "csa_k_concat": 21,
            "hca_k_concat": 20,
            "csa_mask_concat": 21,
            "hca_mask_concat": 20,
        }
        expected_concat_calls = {
            role: evaluated_tokens * len(agents) * layers
            for role, layers in concat_role_layers.items()
        }
        actual_concat_calls = {
            role: attention_concat_role_totals[role][0]
            for role in concat_role_layers
        }
        if set(attention_concat_role_totals) != set(concat_role_layers) or actual_concat_calls != expected_concat_calls:
            raise ValueError(
                f"exact DSV4 attention concat dispatch contract mismatch: "
                f"actual={actual_concat_calls} expected={expected_concat_calls} "
                f"roles={sorted(attention_concat_role_totals)}"
            )
        for role, layers in concat_role_layers.items():
            expected_agent_calls = evaluated_tokens * layers
            actual_agents = dict(attention_concat_agent_calls[role])
            if set(actual_agents) != set(agents) or any(value != expected_agent_calls for value in actual_agents.values()):
                raise ValueError(
                    f"exact DSV4 attention concat per-agent dispatch mismatch for {role}: "
                    f"actual={actual_agents} expected_each={expected_agent_calls}"
                )
            expected_repetition_calls = n_gen * len(agents) * layers
            actual_repetitions = dict(attention_concat_repetition_calls[role])
            expected_repetitions = set(range(discard + 1, reps + 1))
            if set(actual_repetitions) != expected_repetitions or any(
                    value != expected_repetition_calls for value in actual_repetitions.values()):
                raise ValueError(
                    f"exact DSV4 attention concat per-repetition dispatch mismatch for {role}: "
                    f"actual={actual_repetitions} expected_each={expected_repetition_calls}"
                )
        attention_concat_dispatch_contract = {
            "applicable": True, "complete": True, "depth": depths[0],
            "gpu_agents": len(agents), "target_tokens": evaluated_tokens,
            "actual_role_calls": actual_concat_calls,
            "expected_role_calls": expected_concat_calls,
            "per_agent_exact": True, "per_repetition_exact": True,
        }

    def aggregate_api(path: Path, key: str) -> tuple[dict[str, dict[str, int]], int, str, dict[int, dict[str, dict[str, int]]]]:
        output: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        repetition_output: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
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
                repetition = containing_repetition(intervals, start, end)
                if repetition is None:
                    outside += 1
                else:
                    repetition_output[repetition][name][0] += 1
                    repetition_output[repetition][name][1] += end - start
                output[name][0] += 1
                output[name][1] += end - start
        rendered = {name: {"calls": value[0], "duration_ns": value[1]} for name, value in output.items()}
        repetition_rendered = {
            rep: {name: {"calls": value[0], "duration_ns": value[1]} for name, value in values.items()}
            for rep, values in repetition_output.items()
        }
        return rendered, outside, "present_with_events" if rendered else "present_empty", repetition_rendered

    copies, copy_outside, copy_status, repetition_copies = aggregate_api(copy_path, "Direction")
    rccl, rccl_outside, rccl_status, repetition_rccl = aggregate_api(rccl_path, "Function")
    hip, hip_outside, hip_status, repetition_hip = aggregate_api(hip_path, "Function")
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
    per_repetition = []
    for rep in range(discard + 1, reps + 1):
        values = repetition_family_totals.get(rep)
        if not values:
            raise ValueError(f"accepted repetition {rep} contains no kernel events")
        repetition_total = sum(item[1] for item in values.values())
        ranked = sorted(values.items(), key=lambda item: (-item[1][1], item[0]))
        def domain_totals(source: dict[int, dict[str, dict[str, int]]]) -> dict[str, object]:
            entries = source.get(rep, {})
            return {
                "calls": sum(item["calls"] for item in entries.values()),
                "duration_ns": sum(item["duration_ns"] for item in entries.values()),
                "functions": entries,
            }

        per_repetition.append({
            "repetition": rep,
            "wall_ns": accepted_samples[rep - discard - 1],
            "total_summed_device_ns": repetition_total,
            "kernel_dispatches": sum(item[0] for item in values.values()),
            "top_family": ranked[0][0],
            "top_family_share": ranked[0][1][1] / repetition_total,
            "families": {
                name: {
                    "calls": item[0], "duration_ns": item[1],
                    "share_of_summed_device_time": item[1] / repetition_total,
                }
                for name, item in sorted(values.items())
            },
            "trace_domains": {
                "memory_copy": domain_totals(repetition_copies),
                "rccl_api": domain_totals(repetition_rccl),
                "hip_api": domain_totals(repetition_hip),
            },
        })
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
        "profiled_wall_stable": profile_wall_stable,
        "profiled_wall_mad_over_median": profile_wall_mad_over_median,
        "profiled_throughput_eligible": False,
        "csa_decision_eligible": False,
        "family_attribution_complete": True,
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
        "moe_roles": {
            role: {"calls": value[0], "duration_ns": value[1], "share_of_summed_device_time": value[1] / total_device_ns}
            for role, value in sorted(moe_role_totals.items())
        },
        "moe_signatures": {
            signature: {"calls": value[0], "duration_ns": value[1], "share_of_summed_device_time": value[1] / total_device_ns}
            for signature, value in sorted(moe_signature_totals.items())
        },
        "moe_dispatch_contract": moe_dispatch_contract,
        "attention_concat_roles": {
            role: {
                "calls": value[0], "duration_ns": value[1],
                "share_of_summed_device_time": value[1] / total_device_ns,
                "device_ms_per_token": value[1] / evaluated_tokens / 1e6,
            }
            for role, value in sorted(attention_concat_role_totals.items())
        },
        "attention_concat_dispatch_contract": attention_concat_dispatch_contract,
        "top_kernel_shapes": top_kernels,
        "per_agent": per_agent,
        "per_repetition": per_repetition,
        "memory_copies": copies,
        "rccl_api": rccl,
        "hip_api": hip,
        "classification_caveat": (
            "Families are exclusive. Routed/shared expert MMVQ signatures are tied to this fully hashed DSV4-Flash IQ2_M "
            "tensor inventory and must satisfy exact block_count*four-GPU*target-token dispatch counts. At 16K/64K, exact F16 "
            "concat dimension/grid signatures separately identify CSA/HCA K and final mask materialization and must satisfy "
            "21/20-layer counts globally, per GPU, and per repetition. non_moe_quantized_matmul combines attention/indexer/"
            "final-output projections; other may contain remaining unclassified attention, mask, elementwise, or scheduler work. "
            "Kernel durations are summed across devices/queues; API durations may overlap and are diagnostic."
        ),
    }

    print(f"run_dir={run_dir}")
    print(f"depth={depths[0]} profiled_repetitions={accepted_reps} evaluated_target_tokens={evaluated_tokens}")
    print(f"accepted_wall_ms={accepted_wall_ns / 1e6:.3f}")
    print(f"profiled_wall_stable={int(profile_wall_stable)} mad_over_median={profile_wall_mad_over_median:.6f} profiled_throughput_eligible=0 csa_decision_eligible=0")
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
    print("\n[per-repetition top family]")
    print("repetition\tdevice_ms\tdispatches\ttop_share_pct\ttop_family")
    for item in per_repetition:
        print(
            f"{item['repetition']}\t{item['total_summed_device_ns'] / 1e6:.3f}\t{item['kernel_dispatches']}\t"
            f"{100 * item['top_family_share']:.3f}\t{item['top_family']}"
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