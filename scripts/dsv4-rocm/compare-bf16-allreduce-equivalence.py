#!/usr/bin/env python3
"""Compare the short FP32/BF16 DSV4 hidden-AllReduce correctness arms."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import pathlib
import re
import sys
from typing import Any

ABS_TOL = 0.05
REL_TOL = 0.01
MAX_RMSE = 0.02
EXPECTED_DEPTH = 2048
EXPECTED_N_GEN = 4
EXPECTED_CANDIDATE_CALLS = 86 * EXPECTED_N_GEN
SHA_LINE = re.compile(r"^([0-9a-f]{64})  (/.*)$")


def load_provenance(directory: pathlib.Path) -> dict[str, Any]:
    directory = directory.resolve()
    text = (directory / "manifest.txt").read_text(encoding="utf-8")
    hashes: dict[str, str] = {}
    for line in text.splitlines():
        match = SHA_LINE.fullmatch(line)
        if not match:
            continue
        path = pathlib.Path(match.group(2)).resolve(strict=False)
        try:
            key = f"<RUN>/{path.relative_to(directory).as_posix()}"
        except ValueError:
            key = str(path)
        if key in hashes:
            raise ValueError(f"duplicate normalized manifest hash: {key}")
        hashes[key] = match.group(1)
    if not hashes:
        raise ValueError("manifest contains no SHA-256 identities")
    required_run_keys = {
        "<RUN>/source.patch", "<RUN>/source-status.txt", "<RUN>/untracked-files.sha256",
        "<RUN>/model-identity.txt", "<RUN>/hardware-identity.txt",
    }
    missing = sorted(required_run_keys - hashes.keys())
    if missing:
        raise ValueError(f"manifest is missing required run-local hashes: {missing}")
    for key in required_run_keys:
        relative = key.removeprefix("<RUN>/")
        artifact = directory / relative
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual != hashes[key]:
            raise ValueError(f"manifest hash does not match run-local artifact: {key}")
    for key, expected in hashes.items():
        if not key.startswith("/"):
            continue
        artifact = pathlib.Path(key)
        if not artifact.is_file():
            raise ValueError(f"manifest absolute identity is not a file: {key}")
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
            raise ValueError(f"manifest hash does not match absolute artifact: {key}")
    if not re.search(r"^GGML_HIP_GRAPHS:BOOL=ON$", text, re.MULTILINE):
        raise ValueError("compiled GGML_HIP_GRAPHS is not BOOL=ON")
    commit_match = re.search(r"^commit=([0-9a-f]{40})$", text, re.MULTILINE)
    if not commit_match:
        raise ValueError("manifest source commit is missing")
    if (directory / "source-status.txt").read_text(encoding="utf-8") != "":
        raise ValueError("manifest source tree is not clean")
    model_identity = (directory / "model-identity.txt").read_text(encoding="utf-8")
    hardware_identity = (directory / "hardware-identity.txt").read_text(encoding="utf-8")
    if not model_identity.strip() or not hardware_identity.strip():
        raise ValueError("model or hardware identity is empty")
    contract: dict[str, str] = {}
    for line in (directory / "contract.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            contract[key] = value
    if contract.get("git_head") != commit_match.group(1):
        raise ValueError("contract/manifest source commit mismatch")
    binary_path = contract.get("binary")
    if not binary_path or hashes.get(str(pathlib.Path(binary_path).resolve(strict=False))) != contract.get("binary_sha256"):
        raise ValueError("contract/manifest binary hash mismatch")
    return {
        "commit": commit_match.group(1),
        "hashes": hashes,
        "model_identity": model_identity,
        "hardware_identity": hardware_identity,
    }


def reject_constant(text: str) -> None:
    raise ValueError(f"non-standard JSON constant: {text}")


def load_json(path: pathlib.Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def fnv1a(data: bytes) -> str:
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = math.ceil(fraction * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def strict_json_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(strict_json_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(strict_json_equal(actual[key], value) for key, value in expected.items())
    return actual == expected


def load_arm(directory: pathlib.Path, expected_candidate: str) -> dict[str, Any]:
    result_path = directory / "result.json"
    audit_path = directory / "audit.jsonl"
    result = load_json(result_path)
    errors: list[str] = []

    exact = {
        "schema_version": 1,
        "complete": True,
        "target_only": True,
        "state_restore_used": False,
        "sampling_used": False,
        "candidate_value": expected_candidate,
        "depth": EXPECTED_DEPTH,
        "n_gen": EXPECTED_N_GEN,
        "seed": 12345,
        "n_batch": 512,
        "n_ubatch": 256,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "flash_attn": "enabled",
        "logits_file": "logits.f32",
        "float_format": "IEEE-754 binary32 native little-endian",
    }
    for key, expected in exact.items():
        actual = result.get(key)
        if not strict_json_equal(actual, expected):
            errors.append(f"result.{key}={actual!r}, expected type-strict {expected!r}")

    depths = result.get("depths")
    if not isinstance(depths, list) or len(depths) != 1:
        errors.append("result.depths must contain exactly one record")
        depths = []
    records: list[dict[str, Any]] = []
    if depths:
        depth = depths[0]
        if not strict_json_equal(depth.get("depth"), EXPECTED_DEPTH):
            errors.append("depth record mismatch")
        inputs = depth.get("generation_input_tokens")
        if not isinstance(inputs, list) or len(inputs) != EXPECTED_N_GEN or any(type(v) is not int for v in inputs):
            errors.append("generation input token contract mismatch")
        raw_records = depth.get("records")
        if not isinstance(raw_records, list) or len(raw_records) != EXPECTED_N_GEN:
            errors.append("logit record count mismatch")
        else:
            records = raw_records

    logits_path = directory / "logits.f32"
    raw = logits_path.read_bytes() if logits_path.is_file() else b""
    if not raw:
        errors.append("missing or empty logits.f32")
    if not strict_json_equal(result.get("expected_logits_bytes"), len(raw)):
        errors.append("expected_logits_bytes does not type-strictly match file size")

    expected_offset = 0
    for index, record in enumerate(records):
        if not strict_json_equal(record.get("step"), index):
            errors.append(f"record {index}: step mismatch")
        n_vocab = record.get("n_vocab")
        byte_length = record.get("byte_length")
        if type(n_vocab) is not int or n_vocab <= 0 or type(byte_length) is not int or byte_length != n_vocab * 4:
            errors.append(f"record {index}: invalid vocabulary/length")
            continue
        if not strict_json_equal(record.get("byte_offset"), expected_offset):
            errors.append(f"record {index}: noncontiguous byte offset")
        if not strict_json_equal(
                record.get("input_token"), depths[0].get("generation_input_tokens", [None] * EXPECTED_N_GEN)[index]):
            errors.append(f"record {index}: input token mismatch")
        if type(record.get("argmax_token")) is not int:
            errors.append(f"record {index}: invalid argmax")
        chunk = raw[expected_offset:expected_offset + byte_length]
        if len(chunk) != byte_length:
            errors.append(f"record {index}: truncated logit bytes")
        elif fnv1a(chunk) != record.get("logits_fnv1a64"):
            errors.append(f"record {index}: FNV hash mismatch")
        expected_offset += byte_length
    if expected_offset != len(raw):
        errors.append("logit file has missing or trailing bytes")

    audits = []
    if audit_path.is_file():
        for line_number, line in enumerate(audit_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                errors.append(f"audit line {line_number}: blank")
                continue
            try:
                audits.append(json.loads(line, parse_constant=reject_constant))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"audit line {line_number}: {exc}")
    if len(audits) != 1:
        errors.append(f"audit must contain exactly one context, got {len(audits)}")
        audit = {}
    else:
        audit = audits[0]
        audit_exact = {
            "schema_version": 1,
            "context_id": 0,
            "candidate_enabled": expected_candidate == "1",
            "candidate_topology": True,
            "backend_count": 4,
            "logical_devices": [0, 1, 2, 3],
            "candidate_eligible_calls": EXPECTED_CANDIDATE_CALLS,
            "candidate_bf16_calls": EXPECTED_CANDIDATE_CALLS if expected_candidate == "1" else 0,
            "candidate_disabled_fp32_calls": EXPECTED_CANDIDATE_CALLS if expected_candidate == "0" else 0,
            "force_fp32_calls": 0,
            "force_candidate_conflict_calls": 0,
            "ne4096_calls": EXPECTED_CANDIDATE_CALLS,
            "ne4096_all_f32_calls": EXPECTED_CANDIDATE_CALLS,
            "ne4096_same_shape_calls": EXPECTED_CANDIDATE_CALLS,
            "first_ne4096_shape": [4096, 1, 1, 1],
            "complete": True,
        }
        for key, expected in audit_exact.items():
            actual = audit.get(key)
            if not strict_json_equal(actual, expected):
                errors.append(f"audit.{key}={actual!r}, expected type-strict {expected!r}")
        for key in (
            "allreduce_calls", "zero_element_calls", "legacy_fp32_calls", "legacy_bf16_calls",
            "force_fp32_calls",
        ):
            if type(audit.get(key)) is not int or audit.get(key, -1) < 0:
                errors.append(f"audit.{key} must be a non-negative integer")

    if errors:
        raise ValueError("; ".join(errors))
    return {
        "directory": str(directory),
        "result": result,
        "records": records,
        "raw": raw,
        "audit": audit,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def floats_for(arm: dict[str, Any], record: dict[str, Any]) -> array.array:
    offset = record["byte_offset"]
    length = record["byte_length"]
    values = array.array("f")
    values.frombytes(arm["raw"][offset:offset + length])
    if sys.byteorder != "little":
        values.byteswap()
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("control_dir", type=pathlib.Path)
    parser.add_argument("candidate_dir", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path, required=True)
    args = parser.parse_args()

    try:
        control_dir = args.control_dir.resolve()
        candidate_dir = args.candidate_dir.resolve()
        if control_dir.parent != candidate_dir.parent or control_dir.name != "control" or candidate_dir.name != "candidate":
            raise ValueError("control and candidate must be sibling arms in one run directory")
        provenance = load_provenance(control_dir.parent)
        control = load_arm(control_dir, "0")
        candidate = load_arm(candidate_dir, "1")
    except Exception as exc:  # noqa: BLE001
        output = {"schema_version": 1, "complete": False, "accepted": False, "classification": "INVALID", "errors": [str(exc)]}
        args.json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(f"INVALID: {exc}")
        return 2

    errors: list[str] = []
    failures: list[str] = []
    control_depth = control["result"]["depths"][0]
    candidate_depth = candidate["result"]["depths"][0]
    for key in ("prefix_fnv1a64", "generation_input_tokens"):
        if control_depth.get(key) != candidate_depth.get(key):
            errors.append(f"cross-arm {key} mismatch")

    control_audit = control["audit"]
    candidate_audit = candidate["audit"]
    for key in ("allreduce_calls", "zero_element_calls", "candidate_eligible_calls", "force_fp32_calls", "force_candidate_conflict_calls", "legacy_bf16_calls"):
        if control_audit.get(key) != candidate_audit.get(key):
            errors.append(f"cross-arm audit.{key} mismatch")
    if control_audit.get("legacy_fp32_calls") != candidate_audit.get("legacy_fp32_calls", 0) + EXPECTED_CANDIDATE_CALLS:
        errors.append("cross-arm legacy FP32 delta does not equal candidate dispatch count")

    comparisons = []
    for index, (control_record, candidate_record) in enumerate(zip(control["records"], candidate["records"])):
        for key in ("step", "input_token", "n_vocab", "byte_length"):
            if control_record.get(key) != candidate_record.get(key):
                errors.append(f"step {index}: {key} mismatch")
        a = floats_for(control, control_record)
        b = floats_for(candidate, candidate_record)
        if len(a) != len(b):
            errors.append(f"step {index}: float count mismatch")
            continue
        if control_record["argmax_token"] != candidate_record["argmax_token"]:
            failures.append(f"step {index}: argmax token mismatch")

        sum_abs = 0.0
        sum_sq = 0.0
        max_abs = 0.0
        max_rel = 0.0
        violations = 0
        nonfinite = 0
        abs_diffs: list[float] = []
        for lhs, rhs in zip(a, b):
            if not math.isfinite(lhs) or not math.isfinite(rhs):
                nonfinite += 1
                continue
            diff = abs(float(lhs) - float(rhs))
            scale = max(abs(float(lhs)), abs(float(rhs)))
            rel = diff / scale if scale else 0.0
            sum_abs += diff
            sum_sq += diff * diff
            max_abs = max(max_abs, diff)
            max_rel = max(max_rel, rel)
            abs_diffs.append(diff)
            if diff > ABS_TOL + REL_TOL * scale:
                violations += 1
        finite_count = len(a) - nonfinite
        rmse = math.sqrt(sum_sq / finite_count) if finite_count else math.inf
        mae = sum_abs / finite_count if finite_count else math.inf
        abs_diffs.sort()
        comparison = {
            "step": index,
            "input_token": control_record["input_token"],
            "control_argmax": control_record["argmax_token"],
            "candidate_argmax": candidate_record["argmax_token"],
            "n_vocab": len(a),
            "nonfinite_count": nonfinite,
            "combined_tolerance_violations": violations,
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
            "mean_abs_diff": mae,
            "rmse": rmse,
            "p99_abs_diff": percentile(abs_diffs, 0.99),
            "p999_abs_diff": percentile(abs_diffs, 0.999),
        }
        comparisons.append(comparison)
        if nonfinite:
            failures.append(f"step {index}: {nonfinite} nonfinite logits")
        if violations:
            failures.append(f"step {index}: {violations} combined-tolerance violations")
        if rmse > MAX_RMSE:
            failures.append(f"step {index}: RMSE {rmse:.9g} exceeds {MAX_RMSE}")

    classification = "INVALID" if errors else ("PASS" if not failures else "NO-GO")
    output = {
        "schema_version": 1,
        "complete": not errors,
        "accepted": not errors and not failures,
        "classification": classification,
        "provenance": {
            "source_commit": provenance["commit"],
            "identity_hash_count": len(provenance["hashes"]),
            "compiled_hip_graphs": True,
            "clean_source": True,
        },
        "contract": {
            "depth": EXPECTED_DEPTH,
            "n_gen": EXPECTED_N_GEN,
            "expected_candidate_calls": EXPECTED_CANDIDATE_CALLS,
            "absolute_tolerance": ABS_TOL,
            "relative_tolerance": REL_TOL,
            "maximum_rmse": MAX_RMSE,
            "argmax_must_match": True,
        },
        "control_logits_sha256": control["sha256"],
        "candidate_logits_sha256": candidate["sha256"],
        "control_audit": control_audit,
        "candidate_audit": candidate_audit,
        "comparisons": comparisons,
        "errors": errors,
        "failures": failures,
    }
    args.json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"DSV4 BF16 HIDDEN ALLREDUCE CORRECTNESS: {classification}")
    for comparison in comparisons:
        print(
            f"step={comparison['step']} argmax={comparison['control_argmax']}/{comparison['candidate_argmax']} "
            f"rmse={comparison['rmse']:.9g} max_abs={comparison['max_abs_diff']:.9g} "
            f"violations={comparison['combined_tolerance_violations']} nonfinite={comparison['nonfinite_count']}"
        )
    for message in errors + failures:
        print(f"- {message}")
    return 2 if errors else (0 if not failures else 1)


if __name__ == "__main__":
    raise SystemExit(main())