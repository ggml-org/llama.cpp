#!/usr/bin/env python3
import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Tuple

KERNEL_LAYOUT = {
    "mmvq_aos": "AOS",
    "mmvq_aos_baseline": "AOS",
    "mmvq_soa": "SOA",
    "mmvq_soa_baseline": "SOA",
    "mmvq_coalesced": "COALESCED",
    "mmvq_slm_cached": "AOS",
    "mmvq_prefetch": "AOS",
    "mmvq_wide_load": "AOS",
    "mmvq_esimd_block_load": "AOS",
    "mmvq_esimd_slm": "AOS",
    "mmvq_xmx_tile_8x8": "AOS",
    "mmvq_xmx_tile_16x16": "AOS",
    "mmvq_xmx_aos_direct": "AOS",
    "mmvq_xmx_soa_direct": "SOA",
    "mmvq_xmx_double_buffer": "AOS",
    "mmvq_esimd_dpas_1x16x32": "AOS",
    "mmvq_esimd_dpas_8x16x32": "AOS",
    "mmvq_esimd_dpas_chained": "AOS",
    "mmvq_xmx_tile_64x64": "AOS",
    "mmvq_xmx_register_accum": "AOS",
    "mmvq_xmx_multi_wg": "AOS",
    "mmvq_xmx_persistent": "AOS",
    "mmvq_esimd_large_tile": "AOS",
    "mmvq_esimd_persistent": "AOS",
    "mmvq_esimd_lsc_prefetch": "AOS",
    "mmvq_hybrid_adaptive": "AOS",
    "mmvq_xmx_fused": "AOS",
    "mmvq_coalesced_xmx_aligned": "COALESCED",
    "mmvq_esimd_hybrid": "AOS",
    "mmvq_esimd_cooperative": "AOS",
    "mmvq_q4_0_specialized": "AOS",
    "mmvq_q6_k_specialized": "AOS",
    "mmvq_mxfp4_native": "AOS",
    "dpas_memory_patterns": "DPAS",
}

SUPPORTED_LAYOUTS = {
    "Q4_0": {"AOS", "SOA", "COALESCED"},
    "Q8_0": {"AOS", "SOA", "COALESCED"},
    "Q6_K": {"AOS", "SOA", "COALESCED"},
    "MXFP4": {"AOS", "SOA", "COALESCED"},
    "Q4_K": {"AOS", "SOA"},
    "Q2_K": {"AOS"},
    "Q3_K": {"AOS"},
    "Q5_K": {"AOS"},
}

# (block_size, block_bytes)
QUANT_BLOCKS = {
    "Q4_0": (32, 18),
    "Q8_0": (32, 34),
    "MXFP4": (32, 17),
    "Q6_K": (256, 210),
    "Q4_K": (256, 144),
    "Q2_K": (256, 84),
    "Q3_K": (256, 110),
    "Q5_K": (256, 176),
}

Q8_1_BLOCK_SIZE = 32
Q8_1_BLOCK_BYTES = 36
COALESCED_TILE_BLOCKS = 32


def load_matrix(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_ref(matrix: Dict[str, Any], value: Any, key: str) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if value == "all":
            return list(matrix[key])
        if key == "dimensions":
            return list(matrix["dimension_sets"][value])
        if key == "batches":
            return list(matrix["batch_sets"][value])
        if key == "kernels":
            return list(matrix["kernels"][value])
    raise ValueError(f"Unable to resolve {key} reference: {value}")


def normalize_dims(entry: Dict[str, Any]) -> Tuple[int, int, int]:
    if "dim" in entry:
        dim = int(entry["dim"])
        return dim, dim, dim
    dim_m = int(entry.get("dim_m", 0))
    dim_k = int(entry.get("dim_k", 0))
    dim_n = int(entry.get("dim_n", dim_m))
    return dim_m, dim_n, dim_k


def expand_config_set(matrix: Dict[str, Any], config_set: Dict[str, Any]) -> List[Dict[str, Any]]:
    kernels = resolve_ref(matrix, config_set["kernels"], "kernels")
    quant_types = resolve_ref(matrix, config_set["quant_types"], "quant_types")
    dimensions = resolve_ref(matrix, config_set["dimensions"], "dimensions")
    batches = resolve_ref(matrix, config_set["batches"], "batches")
    memory_modes = list(config_set["memory_modes"])

    configs = []
    for kernel in kernels:
        for quant in quant_types:
            for dim in dimensions:
                for batch in batches:
                    for memory_mode in memory_modes:
                        configs.append({
                            "kernel": kernel,
                            "quant": quant,
                            "batch": int(batch),
                            "dim_m": int(dim),
                            "dim_n": int(dim),
                            "dim_k": int(dim),
                            "memory_mode": memory_mode,
                            "source": config_set.get("name", "unknown"),
                            "notes": config_set.get("notes", ""),
                        })
    return configs


def generate_configs(matrix: Dict[str, Any]) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for config_set in matrix.get("config_sets", []):
        configs.extend(expand_config_set(matrix, config_set))

    for entry in matrix.get("edge_cases", []):
        dim_m, dim_n, dim_k = normalize_dims(entry)
        configs.append({
            "kernel": entry["kernel"],
            "quant": entry["quant"],
            "batch": int(entry["batch"]),
            "dim_m": dim_m,
            "dim_n": dim_n,
            "dim_k": dim_k,
            "memory_mode": entry["memory_mode"],
            "expected_failure": bool(entry.get("expected_failure", False)),
            "skip": bool(entry.get("skip", False)),
            "source": entry.get("name", "edge_case"),
            "notes": entry.get("notes", ""),
        })

    for entry in matrix.get("stress_tests", []):
        dim_m, dim_n, dim_k = normalize_dims(entry)
        configs.append({
            "kernel": entry["kernel"],
            "quant": entry["quant"],
            "batch": int(entry["batch"]),
            "dim_m": dim_m,
            "dim_n": dim_n,
            "dim_k": dim_k,
            "memory_mode": entry["memory_mode"],
            "expected_failure": bool(entry.get("expected_failure", False)),
            "skip": bool(entry.get("skip", False)),
            "source": entry.get("name", "stress_test"),
            "notes": entry.get("notes", ""),
            "memory_pattern": entry.get("memory_pattern"),
            "repeat": entry.get("repeat"),
        })

    for entry in matrix.get("dpas_tests", []):
        dim_m, dim_n, dim_k = normalize_dims(entry)
        configs.append({
            "kernel": entry["kernel"],
            "quant": entry.get("quant", "Q8_0"),
            "batch": int(entry.get("batch", 1)),
            "dim_m": dim_m,
            "dim_n": dim_n,
            "dim_k": dim_k,
            "memory_mode": entry.get("memory_mode", "USM_DEVICE"),
            "expected_failure": bool(entry.get("expected_failure", False)),
            "skip": bool(entry.get("skip", False)),
            "source": entry.get("name", "dpas_test"),
            "notes": entry.get("notes", ""),
            "dpas_repeat": entry.get("dpas_repeat"),
            "dpas_device_opt": bool(entry.get("dpas_device_opt", False)),
            "memory_pattern": entry.get("memory_pattern"),
        })

    for cfg in configs:
        cfg["layout"] = KERNEL_LAYOUT.get(cfg["kernel"], "UNKNOWN")

    return configs


def config_id(cfg: Dict[str, Any]) -> Tuple[Any, ...]:
    base = (
        cfg.get("kernel"),
        cfg.get("quant"),
        cfg.get("batch"),
        cfg.get("dim_m"),
        cfg.get("dim_n"),
        cfg.get("dim_k"),
        cfg.get("memory_mode"),
    )
    kernel = cfg.get("kernel") or ""
    if str(kernel).startswith("dpas_"):
        return base + (
            cfg.get("dpas_repeat"),
            cfg.get("dpas_device_opt"),
            cfg.get("memory_pattern"),
        )
    return base


def find_duplicate_configs(configs: Iterable[Dict[str, Any]]) -> List[Tuple[Tuple[Any, ...], int]]:
    seen: Dict[Tuple[Any, ...], int] = {}
    duplicates = []
    for cfg in configs:
        cid = config_id(cfg)
        seen[cid] = seen.get(cid, 0) + 1
    for cid, count in seen.items():
        if count > 1:
            duplicates.append((cid, count))
    return duplicates


def block_size_for_quant(qtype: str) -> int:
    return QUANT_BLOCKS[qtype][0]


def row_size_bytes(qtype: str, k: int) -> int:
    block_size, block_bytes = QUANT_BLOCKS[qtype]
    blocks = math.ceil(k / block_size)
    return int(blocks * block_bytes)


def padded_k(k: int) -> int:
    return int(math.ceil(k / Q8_1_BLOCK_SIZE) * Q8_1_BLOCK_SIZE)


def estimate_total_bytes(cfg: Dict[str, Any]) -> int:
    k = int(cfg["dim_k"])
    m = int(cfg["dim_m"])
    batch = int(cfg["batch"])
    qtype = cfg["quant"]

    weights = row_size_bytes(qtype, k) * m
    k_pad = padded_k(k)
    act_blocks = k_pad // Q8_1_BLOCK_SIZE
    activations = batch * act_blocks * Q8_1_BLOCK_BYTES
    output = batch * m * 4
    return int(weights + activations + output)


def estimate_runtime_s(cfg: Dict[str, Any], estimation: Dict[str, Any]) -> float:
    batch = int(cfg["batch"])
    iterations = int(estimation.get("iterations", 100))
    tps_map = estimation.get("assumed_tps", {})

    if batch <= 4:
        tps = tps_map.get("memory_bound", 1.0)
    elif batch <= 64:
        tps = tps_map.get("transitional", 1.0)
    else:
        tps = tps_map.get("compute_bound", 1.0)

    if tps <= 0:
        return 0.0
    latency_s = batch / tps
    return iterations * latency_s


def validate_config(cfg: Dict[str, Any], matrix: Dict[str, Any]) -> List[str]:
    errors = []
    kernel = cfg.get("kernel")
    quant = cfg.get("quant")
    memory_mode = cfg.get("memory_mode")
    batch = int(cfg.get("batch", 0))
    dim_m = int(cfg.get("dim_m", 0))
    dim_k = int(cfg.get("dim_k", 0))
    layout = cfg.get("layout")

    if kernel not in matrix["kernels"]:
        errors.append(f"unknown kernel {kernel}")
    if quant not in matrix["quant_types"]:
        errors.append(f"unknown quant {quant}")
    if memory_mode not in matrix["memory_modes"]:
        errors.append(f"unknown memory mode {memory_mode}")

    if batch <= 0:
        errors.append("batch must be positive")
    if dim_m <= 0 or dim_k <= 0:
        errors.append("dim_m and dim_k must be positive")

    if quant in QUANT_BLOCKS:
        block_size = block_size_for_quant(quant)
        if dim_k % block_size != 0:
            errors.append(f"dim_k {dim_k} not divisible by block size {block_size}")

    if layout in {"AOS", "SOA", "COALESCED"}:
        allowed = SUPPORTED_LAYOUTS.get(quant, set())
        if layout not in allowed:
            errors.append(f"layout {layout} not supported for {quant}")

    if layout == "COALESCED" and quant in QUANT_BLOCKS:
        block_size = block_size_for_quant(quant)
        blocks_per_row = dim_k // block_size
        if blocks_per_row % COALESCED_TILE_BLOCKS != 0:
            errors.append("coalesced layout requires blocks_per_row multiple of 32")

    limits = matrix["limits"]
    max_bytes = int((limits["max_memory_gb"] - limits["headroom_gb"]) * (1024 ** 3))
    total_bytes = estimate_total_bytes(cfg)
    if total_bytes > max_bytes:
        errors.append(
            f"memory estimate {total_bytes / (1024 ** 3):.2f} GB exceeds budget {(max_bytes / (1024 ** 3)):.2f} GB"
        )

    return errors


def validate_matrix(matrix: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    configs = generate_configs(matrix)

    duplicates = find_duplicate_configs(configs)
    if duplicates:
        for cid, count in duplicates:
            errors.append(f"duplicate config {cid} count={count}")

    required_batches = set(matrix.get("required_batches", []))
    present_batches = {cfg["batch"] for cfg in configs if not cfg.get("skip", False)}
    missing_batches = sorted(required_batches - present_batches)
    if missing_batches:
        errors.append(f"missing required batch sizes: {missing_batches}")

    runnable = [cfg for cfg in configs if not cfg.get("skip", False) and not cfg.get("expected_failure", False)]
    limits = matrix["limits"]
    if len(runnable) < limits["min_configs"] or len(runnable) > limits["max_configs"]:
        errors.append(
            f"config count {len(runnable)} outside [{limits['min_configs']}, {limits['max_configs']}]"
        )

    for cfg in configs:
        cfg_errors = validate_config(cfg, matrix)
        if cfg.get("expected_failure", False):
            if not cfg_errors:
                errors.append(f"expected failure did not fail: {cfg.get('source')} {config_id(cfg)}")
            else:
                warnings.append(f"expected failure: {cfg.get('source')} -> {cfg_errors[0]}")
            continue
        if cfg_errors:
            errors.append(f"{cfg.get('source')} {config_id(cfg)}: {cfg_errors[0]}")

        if not cfg.get("skip", False):
            est = estimate_runtime_s(cfg, matrix.get("estimation", {}))
            if est <= 0.0:
                errors.append(f"invalid runtime estimate for {cfg.get('source')} {config_id(cfg)}")

    return {
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "total_configs": len(configs),
            "runnable_configs": len(runnable),
            "expected_failures": len([c for c in configs if c.get("expected_failure", False)]),
            "skipped": len([c for c in configs if c.get("skip", False)]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate sycl-kernel-bench test matrix")
    parser.add_argument(
        "--matrix",
        default=os.path.join(os.path.dirname(__file__), "test_matrix.json"),
        help="Path to test_matrix.json",
    )
    parser.add_argument("--summary", action="store_true", help="Print summary JSON")
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    result = validate_matrix(matrix)

    if args.summary:
        print(json.dumps(result["summary"], indent=2))

    if result["warnings"]:
        for warning in result["warnings"]:
            print(f"[warn] {warning}")

    if result["errors"]:
        for err in result["errors"]:
            print(f"[error] {err}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
