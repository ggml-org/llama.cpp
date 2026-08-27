#!/usr/bin/env python3
"""Suggest RPC layer-placement benchmark candidates from trace output."""

import argparse
import json
import re
import shlex
from collections import defaultdict
from pathlib import Path


TRACE_OP_RE = re.compile(
    r"^\s+(?P<operation>\S+)\s+"
    r"(?P<endpoint>\S+:\d+)\s+"
    r"(?P<tensor>.+?)\s+"
    r"(?P<calls>\d+)\s+"
    r"(?P<bytes>\d+)\s+"
    r"(?P<wait_ms>[0-9.]+)\s*$"
)
CROSS_RE = re.compile(
    r"^\s+(?P<src>\S+:\d+)\s+->\s+"
    r"(?P<dst>\S+:\d+)\s+"
    r"(?P<calls>\d+)\s+"
    r"(?P<bytes>\d+)\s*$"
)
CROSS_TENSOR_RE = re.compile(
    r"^\s+(?P<src>\S+:\d+)\s+->\s+"
    r"(?P<dst>\S+:\d+)\s+"
    r"(?P<tensor>.+?)\s+"
    r"(?P<calls>\d+)\s+"
    r"(?P<bytes>\d+)\s*$"
)
TRACE_TENSOR_OP_HEADERS = {
    "ggml-rpc trace tensor operations (top by elapsed):",
    "ggml-rpc trace tensor operations (top by wait):",
}
FIT_FAILURE_KINDS = {
    "allocation_failed",
    "model_load_failed",
}


def parse_float_list(value):
    values = []
    for part in re.split(r"[,;/]+", value.strip()):
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid numeric split value {part!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("split list is empty")
    return values


def format_split(values):
    parts = []
    for value in values:
        if value == int(value):
            parts.append(str(int(value)))
        else:
            parts.append(f"{value:.6g}")
    return "/".join(parts)


def normalize_split_text(value):
    if value is None:
        return "auto"
    text = str(value).strip()
    if not text or text == "auto":
        return text or "auto"
    try:
        return format_split(parse_float_list(text))
    except argparse.ArgumentTypeError:
        return text


def placement_key(device, tensor_split):
    return str(device or ""), normalize_split_text(tensor_split)


def parse_list(value):
    return [part.strip() for part in re.split(r"[,;/]+", value) if part.strip()]


def parse_device_endpoints(value):
    mapping = {}
    if value is None:
        return mapping
    if isinstance(value, dict):
        return value
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        device, sep, endpoint = item.partition("=")
        if not sep or not device or not endpoint:
            raise argparse.ArgumentTypeError(
                f"invalid device endpoint mapping {item!r}; expected RPC0=host:port"
            )
        mapping[device.strip()] = endpoint.strip()
    return mapping


def parse_spill_device(value):
    target, sep, spill = value.partition("=")
    target = target.strip()
    spill = spill.strip()
    if not sep or not target or not spill:
        raise argparse.ArgumentTypeError(
            f"invalid spill device mapping {value!r}; expected TARGET=SPILL_DEVICE"
        )
    return target, spill


def parse_sweep_device_weight(value):
    device, sep, weights = value.partition("=")
    device = device.strip()
    weights = weights.strip()
    if not sep or not device or not weights:
        raise argparse.ArgumentTypeError(
            f"invalid device weight sweep {value!r}; expected DEVICE=WEIGHT_LIST"
        )
    return device, parse_float_list(weights)


def parse_trace(paths):
    tensor_ops = []
    cross_endpoint = []
    cross_tensors = []

    section = None
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped in TRACE_TENSOR_OP_HEADERS:
                    section = "tensor_ops"
                    continue
                if stripped == "ggml-rpc trace cross-endpoint copy fallbacks:":
                    section = "cross"
                    continue
                if stripped == "ggml-rpc trace cross-endpoint copy tensors (top by bytes):":
                    section = "cross_tensors"
                    continue
                if stripped.startswith("ggml-rpc trace "):
                    section = None
                    continue
                if not stripped or stripped.startswith("operation ") or stripped.startswith("endpoints "):
                    continue

                if section == "tensor_ops":
                    match = TRACE_OP_RE.match(line)
                    if match:
                        tensor_ops.append({
                            "operation": match.group("operation"),
                            "endpoint": match.group("endpoint"),
                            "tensor": match.group("tensor").strip(),
                            "calls": int(match.group("calls")),
                            "bytes": int(match.group("bytes")),
                            "wait_ms": float(match.group("wait_ms")),
                        })
                    continue

                if section == "cross":
                    match = CROSS_RE.match(line)
                    if match:
                        cross_endpoint.append({
                            "src": match.group("src"),
                            "dst": match.group("dst"),
                            "calls": int(match.group("calls")),
                            "bytes": int(match.group("bytes")),
                        })
                    continue

                if section == "cross_tensors":
                    match = CROSS_TENSOR_RE.match(line)
                    if match:
                        cross_tensors.append({
                            "src": match.group("src"),
                            "dst": match.group("dst"),
                            "tensor": match.group("tensor").strip(),
                            "calls": int(match.group("calls")),
                            "bytes": int(match.group("bytes")),
                        })

    return tensor_ops, cross_endpoint, cross_tensors


def aggregate_endpoint_waits(tensor_ops):
    endpoint = defaultdict(lambda: {
        "result_output_wait_ms": 0.0,
        "result_output_bytes": 0,
        "layer_boundary_wait_ms": 0.0,
        "layer_boundary_bytes": 0,
        "copy_tensor_wait_ms": 0.0,
        "copy_tensor_bytes": 0,
    })

    for op in tensor_ops:
        item = endpoint[op["endpoint"]]
        tensor = op["tensor"]
        if tensor == "result_output" and op["operation"] == "GET_TENSOR":
            item["result_output_wait_ms"] += op["wait_ms"]
            item["result_output_bytes"] += op["bytes"]
        if tensor.startswith("l_out-"):
            item["layer_boundary_wait_ms"] += op["wait_ms"]
            item["layer_boundary_bytes"] += op["bytes"]
            if op["operation"] == "COPY_TENSOR":
                item["copy_tensor_wait_ms"] += op["wait_ms"]
                item["copy_tensor_bytes"] += op["bytes"]

    result = {}
    for name, item in endpoint.items():
        result[name] = dict(item)
        for metric in ("result_output", "layer_boundary", "copy_tensor"):
            bytes_key = f"{metric}_bytes"
            wait_key = f"{metric}_wait_ms"
            rate_key = f"{metric}_ms_per_mib"
            bytes_value = item[bytes_key]
            result[name][rate_key] = (
                item[wait_key] / (bytes_value / (1024 * 1024))
                if bytes_value else None
            )
    return result


def aggregate_cross_tensors(cross_tensors, tensor_ops):
    wait_by_tensor = defaultdict(float)
    for op in tensor_ops:
        if op["tensor"].startswith("l_out-"):
            wait_by_tensor[(op["endpoint"], op["tensor"])] += op["wait_ms"]

    pairs = []
    for item in cross_tensors:
        pairs.append({
            **item,
            "src_wait_ms": wait_by_tensor.get((item["src"], item["tensor"]), 0.0),
        })
    return pairs


def infer_device_endpoints(devices, rpc_endpoints, explicit):
    if explicit:
        result = dict(explicit)
        for device in devices:
            result.setdefault(device, explicit.get(device))
        return result
    if len(devices) == len(rpc_endpoints):
        return dict(zip(devices, rpc_endpoints))
    return {device: None for device in devices}


def endpoint_suffix_cost(removed_endpoints, endpoint_metrics, cross_pairs):
    cost = 0.0
    for endpoint in removed_endpoints:
        metrics = endpoint_metrics.get(endpoint)
        if not metrics:
            continue
        cost += metrics["result_output_wait_ms"]
        cost += metrics["layer_boundary_wait_ms"]
    for pair in cross_pairs:
        if pair["src"] in removed_endpoints or pair["dst"] in removed_endpoints:
            cost += pair.get("src_wait_ms", 0.0)
    return cost


def endpoint_pair_cost(src_endpoint, dst_endpoint, cross_pairs):
    if src_endpoint is None or dst_endpoint is None:
        return 0.0
    cost = 0.0
    for pair in cross_pairs:
        if pair["src"] == src_endpoint and pair["dst"] == dst_endpoint:
            cost += pair.get("src_wait_ms", 0.0)
    return cost


def read_text_if_exists(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def resolve_case_path(summary_path, summary, value):
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path if path.is_file() else None

    candidates = []
    cwd = summary.get("cwd")
    if cwd:
        candidates.append(Path(cwd) / path)
    candidates.append(summary_path.parent / path.name)
    candidates.append(summary_path.parent / path)
    candidates.append(path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def infer_fit_failure_kind(case, stderr_text):
    kind = case.get("failure_kind")
    if kind in FIT_FAILURE_KINDS:
        return kind

    text = "\n".join([
        str(case.get("error") or ""),
        str(case.get("stderr_tail") or ""),
        stderr_text,
    ]).lower()
    if "failed to load model" in text:
        return "model_load_failed"
    if "out of memory" in text or "failed to allocate" in text or "memory allocation failed" in text:
        return "allocation_failed"
    return None


def load_fit_failures(paths):
    records = []
    by_placement = defaultdict(list)

    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        for case in summary.get("cases", []):
            if case.get("status") not in ("failed", "timeout"):
                continue

            stderr_path = resolve_case_path(path, summary, case.get("stderr_path"))
            stderr_text = read_text_if_exists(stderr_path) if stderr_path is not None else ""
            failure_kind = infer_fit_failure_kind(case, stderr_text)
            if failure_kind is None:
                continue

            device = case.get("device") or summary.get("device")
            tensor_split = case.get("tensor_split_display", case.get("tensor_split"))
            if device is None or tensor_split is None:
                continue

            record = {
                "summary": str(path),
                "name": case.get("name"),
                "status": case.get("status"),
                "failure_kind": failure_kind,
                "device": device,
                "tensor_split": normalize_split_text(tensor_split),
                "output_device": case.get("output_device"),
                "error": case.get("error"),
                "stderr_path": str(stderr_path) if stderr_path is not None else case.get("stderr_path"),
            }
            records.append(record)
            by_placement[placement_key(device, tensor_split)].append(record)

    return by_placement, records


def make_spill_candidates(args, devices, splits, device_endpoints, cross_pairs):
    candidates = []
    for target_device, spill_device in args.spill_device:
        if target_device not in devices:
            continue
        if spill_device in devices:
            continue

        target_index = devices.index(target_device)
        source_index = None
        for index in range(target_index + 1, len(splits)):
            if splits[index] > 0:
                source_index = index
                break
        if source_index is None:
            continue

        target_endpoint = device_endpoints.get(target_device)
        spill_endpoint = device_endpoints.get(spill_device)
        if target_endpoint is not None and spill_endpoint is not None and target_endpoint != spill_endpoint:
            continue

        source_device = devices[source_index]
        source_endpoint = device_endpoints.get(source_device)
        trace_cost = endpoint_pair_cost(target_endpoint, source_endpoint, cross_pairs)

        for spill_weight in args.spill_weight:
            if spill_weight <= 0 or splits[source_index] - spill_weight <= 0:
                continue

            candidate_devices = devices[:target_index + 1] + [spill_device] + devices[target_index + 1:]
            candidate_splits = splits[:target_index + 1] + [spill_weight] + splits[target_index + 1:]
            candidate_splits[source_index + 1] -= spill_weight

            candidates.append({
                "name": (
                    f"spill_after_{target_device}_to_{spill_device}_"
                    f"w{format_split([spill_weight]).replace('.', 'p')}_from_{source_device}"
                ),
                "description": (
                    "Move a small split weight from the next downstream device "
                    "onto a spill device that should live on the same RPC endpoint "
                    "as the target device."
                ),
                "device": "/".join(candidate_devices),
                "tensor_split": format_split(candidate_splits),
                "generation": "endpoint-spill",
                "spill_after_device": target_device,
                "spill_device": spill_device,
                "spill_source_device": source_device,
                "spill_weight": spill_weight,
                "spill_endpoint_estimate": spill_endpoint,
                "source_endpoint_estimate": source_endpoint,
                "trace_candidate_cost_ms": trace_cost,
                "trace_spill_boundary_cost_ms": trace_cost,
                "output_device_estimate": candidate_devices[-1],
                "output_endpoint_estimate": device_endpoints.get(candidate_devices[-1]),
            })

    return candidates


def make_weight_sweep_candidates(args, devices, splits, device_endpoints):
    candidates = []
    seen = set()
    sweep_order = 0
    for device, weights in args.sweep_device_weight:
        if device not in devices:
            continue

        index = devices.index(device)
        for weight in weights:
            if weight <= 0:
                continue

            candidate_splits = list(splits)
            if candidate_splits[index] == weight:
                continue
            candidate_splits[index] = weight
            if sum(candidate_splits) <= 0:
                continue
            key = (tuple(devices), tuple(candidate_splits))
            if key in seen:
                continue
            seen.add(key)

            output_device = None
            for output_index in range(len(candidate_splits) - 1, -1, -1):
                if candidate_splits[output_index] > 0:
                    output_device = devices[output_index]
                    break

            weight_name = format_split([weight]).replace(".", "p")
            candidates.append({
                "name": f"sweep_{device}_w{weight_name}",
                "description": (
                    "Change one existing device split weight while keeping the "
                    "device order and all other split weights unchanged."
                ),
                "device": "/".join(devices),
                "tensor_split": format_split(candidate_splits),
                "generation": "device-weight-sweep",
                "sweep_device": device,
                "sweep_device_index": index,
                "sweep_order": sweep_order,
                "sweep_weight": weight,
                "output_device_estimate": output_device,
                "output_endpoint_estimate": device_endpoints.get(output_device),
                "trace_candidate_cost_ms": 0.0,
                "trace_cost_source": "not-estimated",
            })
            sweep_order += 1

    return candidates


def make_candidates(args, devices, splits, device_endpoints, endpoint_metrics, cross_pairs):
    total = sum(splits)
    candidates = []
    baseline = {
        "name": "baseline",
        "description": "Original tensor split.",
        "device": "/".join(devices),
        "tensor_split": format_split(splits),
        "zero_suffix_after_device": None,
        "output_device_estimate": devices[-1] if devices else None,
        "output_endpoint_estimate": device_endpoints.get(devices[-1]) if devices else None,
        "omitted_split_weight": 0.0,
        "omitted_split_ratio": 0.0,
        "trace_removed_suffix_cost_ms": 0.0,
    }
    candidates.append(baseline)

    for cutoff in range(len(splits) - 2, -1, -1):
        candidate_splits = list(splits)
        for index in range(cutoff + 1, len(candidate_splits)):
            candidate_splits[index] = 0.0

        active_total = sum(candidate_splits)
        if active_total <= 0:
            continue

        omitted = total - active_total
        omitted_ratio = omitted / total if total else 0.0
        if omitted_ratio > args.max_omitted_ratio:
            continue

        output_device = devices[cutoff] if cutoff < len(devices) else None
        output_endpoint = device_endpoints.get(output_device)
        removed_devices = devices[cutoff + 1 :]
        removed_endpoints = {
            endpoint for device in removed_devices
            for endpoint in [device_endpoints.get(device)]
            if endpoint is not None
        }
        trace_cost = endpoint_suffix_cost(removed_endpoints, endpoint_metrics, cross_pairs)

        candidates.append({
            "name": f"zero_suffix_after_{output_device or cutoff}",
            "description": (
                "Set all split weights after this device to zero so the output "
                "slot lands on this device in layer split mode."
            ),
            "device": "/".join(devices),
            "tensor_split": format_split(candidate_splits),
            "zero_suffix_after_device": output_device,
            "output_device_estimate": output_device,
            "output_endpoint_estimate": output_endpoint,
            "removed_devices": removed_devices,
            "removed_endpoints": sorted(removed_endpoints),
            "omitted_split_weight": omitted,
            "omitted_split_ratio": omitted_ratio,
            "trace_removed_suffix_cost_ms": trace_cost,
            "trace_candidate_cost_ms": trace_cost,
        })

    candidates.extend(make_spill_candidates(args, devices, splits, device_endpoints, cross_pairs))
    candidates.extend(make_weight_sweep_candidates(args, devices, splits, device_endpoints))
    deduped = []
    seen_placements = set()
    for candidate in candidates:
        key = (candidate["device"], candidate["tensor_split"])
        if key in seen_placements:
            continue
        seen_placements.add(key)
        deduped.append(candidate)
    candidates = deduped
    non_baseline = sorted(
        candidates[1:],
        key=lambda item: (
            -item.get("trace_candidate_cost_ms", 0.0),
            item.get("omitted_split_ratio", 0.0),
            item.get("sweep_order", 1000000),
            item["name"],
        ),
    )
    return candidates[:1] + non_baseline[:args.max_candidates]


def apply_fit_constraints(candidates, fit_failures, include_known_failures):
    kept = []
    rejected = []

    for candidate in candidates:
        failures = fit_failures.get(placement_key(candidate["device"], candidate["tensor_split"]), [])
        if failures:
            candidate["fit_status"] = "known_failed"
            candidate["fit_failures"] = failures
            if candidate["name"] != "baseline" and not include_known_failures:
                rejected.append(candidate)
                continue
        else:
            candidate["fit_status"] = "unknown"
        kept.append(candidate)

    return kept, rejected


def build_bench_command(args, candidate):
    if args.llama_bench is None or args.model is None or args.rpc is None:
        return None
    if candidate["device"] != "/".join(parse_list(args.device)):
        return None
    cmd = [
        "python3",
        "tools/rpc/bench_rpc_remote.py",
        "--llama-bench",
        str(args.llama_bench),
        "--model",
        str(args.model),
        "--base-rpc",
        args.rpc,
        "--patch-rpc",
        args.rpc,
        "--prompt",
        str(args.prompt),
        "--gen",
        str(args.gen),
        "--repetitions",
        str(args.repetitions),
        "--ngl",
        str(args.ngl),
        "--device",
        candidate["device"],
        "--split-mode",
        args.split_mode,
        "--base-tensor-split",
        format_split(args.tensor_split),
        "--patch-tensor-split",
        candidate["tensor_split"],
    ]
    if args.flash_attn is not None:
        cmd.extend(["--flash-attn", args.flash_attn])
    for env in args.env:
        cmd.extend(["--env", env])
    if args.out_dir is not None:
        separator = "\\" if "\\" in args.out_dir and "/" not in args.out_dir else "/"
        cmd.extend(["--out-dir", args.out_dir.rstrip("/\\") + separator + candidate["name"]])
    return shlex.join(cmd)


def build_sweep_candidate_arg(candidate):
    return "--candidate-device " + shlex.quote(
        f"{candidate['name']}={candidate['device']}:{candidate['tensor_split']}"
    )


def build_summary(args):
    tensor_ops, cross_endpoint, cross_tensors = parse_trace(args.trace_stderr)
    devices = parse_list(args.device)
    rpc_endpoints = parse_list(args.rpc) if args.rpc else []
    explicit_mapping = parse_device_endpoints(args.device_endpoints)
    device_endpoints = infer_device_endpoints(devices, rpc_endpoints, explicit_mapping)
    endpoint_metrics = aggregate_endpoint_waits(tensor_ops)
    cross_pairs = aggregate_cross_tensors(cross_tensors, tensor_ops)
    fit_failures, fit_failure_records = load_fit_failures(args.fit_summary)
    raw_candidates = make_candidates(
        args,
        devices,
        args.tensor_split,
        device_endpoints,
        endpoint_metrics,
        cross_pairs,
    )
    candidates, rejected_candidates = apply_fit_constraints(
        raw_candidates,
        fit_failures,
        args.include_known_fit_failures,
    )
    for candidate in candidates:
        candidate["bench_rpc_sweep_arg"] = build_sweep_candidate_arg(candidate)
        candidate["bench_command"] = build_bench_command(args, candidate)
    for candidate in rejected_candidates:
        candidate["bench_rpc_sweep_arg"] = build_sweep_candidate_arg(candidate)
        candidate["bench_command"] = build_bench_command(args, candidate)

    return {
        "schema_version": 1,
        "trace_stderr": [str(path) for path in args.trace_stderr],
        "rpc": args.rpc,
        "device": "/".join(devices),
        "device_endpoints": device_endpoints,
        "tensor_split": format_split(args.tensor_split),
        "split_mode": args.split_mode,
        "flash_attn": args.flash_attn,
        "parsed": {
            "tensor_op_count": len(tensor_ops),
            "cross_endpoint_count": len(cross_endpoint),
            "cross_tensor_count": len(cross_tensors),
        },
        "endpoint_metrics": endpoint_metrics,
        "cross_endpoint": cross_endpoint,
        "cross_tensors": cross_pairs,
        "fit_summaries": [str(path) for path in args.fit_summary],
        "fit_failures": fit_failure_records,
        "candidates": candidates,
        "rejected_candidates": rejected_candidates,
        "notes": [
            "Candidates are benchmark suggestions, not proven optimizations.",
            "Layer split output placement is estimated as the last non-zero split device.",
            "Zeroing a suffix changes model placement and may reduce fit or prompt throughput.",
            "Known fit failures are skipped unless --include-known-fit-failures is set.",
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Suggest RPC tensor-split placement candidates from GGML_RPC_TRACE stderr."
    )
    parser.add_argument("--trace-stderr", action="append", required=True, type=Path, help="stderr file containing GGML_RPC_TRACE output; repeatable")
    parser.add_argument("--tensor-split", required=True, type=parse_float_list, help="current tensor split, for example 1/2/3/4")
    parser.add_argument("--device", required=True, help="current llama-bench device string, for example RPC0/RPC1")
    parser.add_argument("--rpc", default=None, help="RPC endpoint list used by llama-bench")
    parser.add_argument("--device-endpoints", default=None, help="explicit device to endpoint mapping, for example RPC0=a:50052,RPC1=a:50052")
    parser.add_argument("--fit-summary", action="append", default=[], type=Path, help="bench_rpc_sweep summary JSON containing failed candidates to avoid; repeatable")
    parser.add_argument("--include-known-fit-failures", action="store_true", help="emit known non-fitting candidates instead of moving them to rejected_candidates")
    parser.add_argument("--spill-device", action="append", default=[], type=parse_spill_device, metavar="TARGET=SPILL_DEVICE", help="generate same-endpoint spill candidates after TARGET using SPILL_DEVICE; repeatable")
    parser.add_argument("--spill-weight", default=[1.0], type=parse_float_list, help="spill weights to try for each --spill-device, for example 1 or 1/2")
    parser.add_argument("--sweep-device-weight", action="append", default=[], type=parse_sweep_device_weight, metavar="DEVICE=WEIGHT_LIST", help="generate candidates by changing one existing device split weight, for example RPC0=1/2/4/8")
    parser.add_argument("--max-omitted-ratio", default=0.35, type=float, help="skip candidates that zero more than this split-weight fraction")
    parser.add_argument("--max-candidates", default=6, type=int, help="maximum non-baseline candidates to emit")
    parser.add_argument("--llama-bench", default=None, help="optional llama-bench path for generated bench_rpc_remote commands")
    parser.add_argument("--model", default=None, help="optional model path for generated bench_rpc_remote commands")
    parser.add_argument("--prompt", default=16, type=int, help="generated command prompt tokens")
    parser.add_argument("--gen", default=16, type=int, help="generated command generation tokens")
    parser.add_argument("--repetitions", default=3, type=int, help="generated command repetitions")
    parser.add_argument("--ngl", default=99, type=int, help="generated command -ngl")
    parser.add_argument("--split-mode", default="layer", choices=("none", "layer", "row", "tensor"), help="generated command split mode")
    parser.add_argument("--flash-attn", default=None, choices=("on", "off", "auto"), help="generated command flash attention setting")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="environment override for generated commands; repeatable")
    parser.add_argument("--out-dir", default=None, help="optional output directory root for generated bench commands")
    parser.add_argument("--output", default=None, type=Path, help="write JSON summary to this path")
    args = parser.parse_args()

    if args.max_omitted_ratio < 0 or args.max_omitted_ratio > 1:
        parser.error("--max-omitted-ratio must be in the range 0..1")
    if args.max_candidates < 1:
        parser.error("--max-candidates must be >= 1")

    devices = parse_list(args.device)
    if len(devices) != len(args.tensor_split):
        parser.error(
            f"--device has {len(devices)} entries but --tensor-split has "
            f"{len(args.tensor_split)} entries"
        )
    unknown_sweep_devices = sorted({
        device
        for device, _weights in args.sweep_device_weight
        if device not in devices
    })
    if unknown_sweep_devices:
        parser.error(
            "--sweep-device-weight references unknown device(s): "
            + ", ".join(unknown_sweep_devices)
        )
    non_positive_sweep_weights = [
        f"{device}={format_split([weight])}"
        for device, weights in args.sweep_device_weight
        for weight in weights
        if weight <= 0
    ]
    if non_positive_sweep_weights:
        parser.error(
            "--sweep-device-weight must use positive weights; invalid: "
            + ", ".join(non_positive_sweep_weights)
        )

    return args


def main():
    args = parse_args()
    summary = build_summary(args)
    text = json.dumps(summary, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
