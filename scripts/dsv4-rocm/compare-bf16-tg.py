#!/usr/bin/env python3
"""Compare the short 0/2K/8K BF16 hidden-AllReduce TG triage arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
from typing import Any

EXPECTED_DEPTHS = [0, 2048, 8192]
EXPECTED_GEN = 8
EXPECTED_RAW_REPS = 6
EXPECTED_ACCEPTED = 5
MAX_REGRESSION_PERCENT = 2.0
MIN_PROMISING_8K_GAIN_PERCENT = 4.0
DISPATCH_MARKER = "using guarded RDNA2 BF16 hidden AllReduce"
ARMED_MARKER = "guarded RDNA2 BF16 hidden AllReduce armed across 4 devices"
SHA_LINE = re.compile(r"^([0-9a-f]{64})  (/.*)$")


def load_manifest(directory: pathlib.Path) -> dict[str, Any]:
    directory = directory.resolve()
    manifest_path = directory / "manifest.txt"
    text = manifest_path.read_text(encoding="utf-8", errors="strict")
    hashes: dict[str, str] = {}
    for line in text.splitlines():
        match = SHA_LINE.fullmatch(line)
        if not match:
            continue
        hashed_path = pathlib.Path(match.group(2)).resolve(strict=False)
        try:
            relative = hashed_path.relative_to(directory)
        except ValueError:
            key = str(hashed_path)
        else:
            key = f"<RUN>/{relative.as_posix()}"
        if key in hashes:
            raise ValueError(f"duplicate normalized manifest hash path: {key}")
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
    source_status = directory / "source-status.txt"
    if not source_status.is_file() or source_status.read_text(encoding="utf-8") != "":
        raise ValueError("manifest source tree is not clean")
    model_identity = (directory / "model-identity.txt").read_text(encoding="utf-8")
    hardware_identity = (directory / "hardware-identity.txt").read_text(encoding="utf-8")
    if not model_identity.strip() or not hardware_identity.strip():
        raise ValueError("model or hardware identity is empty")
    return {
        "hashes": hashes,
        "commit": commit_match.group(1),
        "model_identity": model_identity,
        "hardware_identity": hardware_identity,
    }


def reject_constant(text: str) -> None:
    raise ValueError(f"non-standard JSON constant: {text}")


def load_json(path: pathlib.Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def status_codes(path: pathlib.Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.endswith("exit_code"):
            values[key] = int(value)
    return values


def validate_arm(directory: pathlib.Path, candidate_value: str) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    required = [
        "summary.json", "contract.json", "status.txt", "stdout-classification.json",
        "bench.log", "bench.stdout.log", "bench.stdout-nonjson.log", "manifest.txt", "result.jsonl", "result-completed-at.ns",
    ]
    for name in required:
        if not (directory / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        return {}, errors

    summary = load_json(directory / "summary.json")
    contract = load_json(directory / "contract.json")
    classification = load_json(directory / "stdout-classification.json")
    codes = status_codes(directory / "status.txt")
    log = (directory / "bench.log").read_text(encoding="utf-8", errors="replace")
    try:
        manifest = load_manifest(directory)
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f"manifest: {exc}")
        manifest = {}

    exact_summary = {
        "complete": True,
        "truncated": False,
        "dropped_trailing_partial": False,
        "expected_depths": EXPECTED_DEPTHS,
        "seen_depths": EXPECTED_DEPTHS,
        "missing_depths": [],
        "extra_depths": [],
        "duplicate_depths": [],
        "depth_order_matches": True,
        "expected_gen": EXPECTED_GEN,
        "expected_raw_repetitions": EXPECTED_RAW_REPS,
        "discard_first": 1,
        "accepted_repetitions": EXPECTED_ACCEPTED,
        "stability_limit": 0.03,
        "contract_errors": [],
    }
    for key, expected in exact_summary.items():
        actual = summary.get(key)
        if actual != expected or (type(expected) is bool and type(actual) is not bool):
            errors.append(f"summary.{key}={actual!r}, expected {expected!r}")
    if type(summary.get("stable")) is not bool:
        errors.append("summary.stable must be a Boolean")

    records = summary.get("records")
    if not isinstance(records, list) or [r.get("depth") for r in records] != EXPECTED_DEPTHS:
        errors.append("summary records do not match exact depth order")
        records = []
    for record in records:
        for key, expected in (
            ("n_gen", EXPECTED_GEN), ("raw_repetitions", EXPECTED_RAW_REPS),
            ("discarded_first", 1), ("accepted_repetitions", EXPECTED_ACCEPTED),
            ("stability_limit", 0.03), ("contract_errors", []),
        ):
            if record.get(key) != expected:
                errors.append(f"depth {record.get('depth')}: {key} mismatch")
        stable = record.get("stable")
        median = record.get("median_ts")
        mad = record.get("mad_over_median")
        if type(stable) is not bool:
            errors.append(f"depth {record.get('depth')}: stable must be a Boolean")
        if type(median) not in (int, float) or type(median) is bool or not math.isfinite(median) or median <= 0:
            errors.append(f"depth {record.get('depth')}: median_ts must be finite and positive")
        if type(mad) not in (int, float) or type(mad) is bool or not math.isfinite(mad) or mad < 0:
            errors.append(f"depth {record.get('depth')}: MAD ratio must be finite and nonnegative")
        elif type(stable) is bool and stable != (mad <= 0.03):
            errors.append(f"depth {record.get('depth')}: stable flag disagrees with the 3% MAD limit")
    if records and type(summary.get("stable")) is bool and summary["stable"] != all(r.get("stable") is True for r in records):
        errors.append("summary.stable disagrees with record stability")

    exact_contract = {
        "target_only": True,
        "draft_model_loaded": False,
        "speculative_flags": [],
        "mode": "performance",
        "depths": EXPECTED_DEPTHS,
        "n_gen": EXPECTED_GEN,
        "raw_repetitions": EXPECTED_RAW_REPS,
        "discard_first": 1,
        "accepted_repetitions": EXPECTED_ACCEPTED,
        "depth_state_api": "context",
        "profile": "none",
        "model_hash_mode": "metadata",
        "require_accepted_stack": 1,
        "allow_busy_gpus": 0,
        "batch": 512,
        "ubatch": 256,
        "tensor_split": "1/1/1/1",
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "threads": 12,
        "load_mode": "mmap",
    }
    for key, expected in exact_contract.items():
        actual = contract.get(key)
        if actual != expected or (type(expected) is bool and type(actual) is not bool):
            errors.append(f"contract.{key}={actual!r}, expected {expected!r}")
    communication = contract.get("communication_candidate", {})
    if communication.get("backend") != "nccl":
        errors.append("contract communication backend must be nccl")
    if communication.get("hip_graphs") != "1" or communication.get("runtime_graph_disable") is not None:
        errors.append("HIP graph contract mismatch")
    if communication.get("bf16_hidden_allreduce") != candidate_value:
        errors.append("BF16 candidate contract mismatch")
    if communication.get("bf16_hidden_allreduce_audit") is not None:
        errors.append("performance screen must not enable correctness audit")
    for key in ("algorithm", "protocol", "min_channels", "max_channels", "debug", "debug_subsys"):
        if communication.get(key) is not None:
            errors.append(f"unexpected communication override {key}")

    for key in ("process_exit_code", "stderr_consumer_exit_code", "stdout_consumer_exit_code"):
        if codes.get(key) != 0:
            errors.append(f"status {key}={codes.get(key)!r}, expected 0")

    expected_classification = {
        "schema_version": 1,
        "json_lines": 3,
        "json_completion_timestamps": 3,
        "malformed_json_like_lines": 0,
        "consumer_success": True,
        "unterminated_final_data": False,
        "excessive_non_json_output": False,
        "max_non_json_lines": 4096,
        "raw_stream_preserved": True,
    }
    for key, expected in expected_classification.items():
        actual = classification.get(key)
        if actual != expected or (type(expected) is bool and type(actual) is not bool):
            errors.append(f"stdout classification {key}={actual!r}, expected {expected!r}")
    if (directory / "bench.stdout.log").stat().st_size != classification.get("total_bytes"):
        errors.append("raw stdout size does not match classification")
    if len((directory / "result-completed-at.ns").read_text(encoding="utf-8").splitlines()) != len(EXPECTED_DEPTHS):
        errors.append("result timestamp file count mismatch")

    dispatch_count = log.count(DISPATCH_MARKER)
    armed_count = log.count(ARMED_MARKER)
    if candidate_value == "1":
        if dispatch_count != len(EXPECTED_DEPTHS):
            errors.append(f"candidate dispatch marker count {dispatch_count}, expected {len(EXPECTED_DEPTHS)}")
        if armed_count != len(EXPECTED_DEPTHS):
            errors.append(f"candidate armed marker count {armed_count}, expected {len(EXPECTED_DEPTHS)}")
    elif dispatch_count != 0 or armed_count != 0:
        errors.append("control unexpectedly emitted candidate markers")

    if re.search(r"\b(?:nan|inf)\b", (directory / "result.jsonl").read_text(encoding="utf-8"), re.I):
        errors.append("result stream contains nonfinite text")

    return {"summary": summary, "contract": contract, "records": records, "manifest": manifest}, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("control_dir", type=pathlib.Path)
    parser.add_argument("candidate_dir", type=pathlib.Path)
    parser.add_argument("--correctness-dir", type=pathlib.Path, required=True)
    parser.add_argument("--json", type=pathlib.Path, required=True)
    args = parser.parse_args()

    try:
        control, control_errors = validate_arm(args.control_dir, "0")
    except Exception as exc:  # noqa: BLE001
        control, control_errors = {}, [f"unreadable arm: {exc}"]
    try:
        candidate, candidate_errors = validate_arm(args.candidate_dir, "1")
    except Exception as exc:  # noqa: BLE001
        candidate, candidate_errors = {}, [f"unreadable arm: {exc}"]
    errors = [f"control: {error}" for error in control_errors] + [f"candidate: {error}" for error in candidate_errors]
    comparisons = []
    failures: list[str] = []
    correctness_manifest: dict[str, Any] = {}

    if not errors:
        control_manifest = control["manifest"]
        candidate_manifest = candidate["manifest"]
        if control_manifest != candidate_manifest:
            errors.append("control/candidate binary, DSO, source, model, build, or hardware identity mismatch")
        try:
            correctness_result = load_json(args.correctness_dir / "comparison.json")
            if correctness_result.get("complete") is not True or correctness_result.get("accepted") is not True or \
                    correctness_result.get("classification") != "PASS":
                errors.append("correctness artifact is not an exact PASS")
            correctness_manifest = load_manifest(args.correctness_dir)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"correctness provenance: {exc}")
        if correctness_manifest:
            if correctness_manifest["commit"] != control_manifest["commit"]:
                errors.append("correctness/TG source commit mismatch")
            if correctness_manifest["model_identity"] != control_manifest["model_identity"]:
                errors.append("correctness/TG model identity mismatch")
            if correctness_manifest["hardware_identity"] != control_manifest["hardware_identity"]:
                errors.append("correctness/TG hardware identity mismatch")
            required_run_keys = {
                "<RUN>/source.patch", "<RUN>/source-status.txt", "<RUN>/untracked-files.sha256",
                "<RUN>/model-identity.txt", "<RUN>/hardware-identity.txt",
            }
            for key in required_run_keys:
                if correctness_manifest["hashes"][key] != control_manifest["hashes"][key]:
                    errors.append(f"correctness/TG provenance hash mismatch: {key}")

            correctness_caches = {
                key: value for key, value in correctness_manifest["hashes"].items()
                if pathlib.Path(key).name == "CMakeCache.txt"
            }
            control_caches = {
                key: value for key, value in control_manifest["hashes"].items()
                if pathlib.Path(key).name == "CMakeCache.txt"
            }
            if len(correctness_caches) != 1 or len(control_caches) != 1:
                errors.append("correctness and TG must each contain exactly one CMakeCache.txt hash")
            elif correctness_caches != control_caches:
                errors.append("correctness/TG CMakeCache.txt path or hash mismatch")

            correctness_dsos = {
                key: value for key, value in correctness_manifest["hashes"].items()
                if key.startswith("/") and re.search(r"\.so(?:\.\d+)*$", pathlib.Path(key).name)
                and pathlib.Path(key).is_file()
            }
            control_dsos = {
                key: value for key, value in control_manifest["hashes"].items()
                if key.startswith("/") and re.search(r"\.so(?:\.\d+)*$", pathlib.Path(key).name)
                and pathlib.Path(key).is_file()
            }
            common_dso_keys = sorted(correctness_dsos.keys() & control_dsos.keys())
            if len(common_dso_keys) < 4:
                errors.append("too few common correctness/TG DSO identities")
            for key in common_dso_keys:
                if correctness_dsos[key] != control_dsos[key]:
                    errors.append(f"correctness/TG DSO hash mismatch: {key}")

    if not errors:
        for control_record, candidate_record in zip(control["records"], candidate["records"]):
            depth = control_record["depth"]
            control_ts = float(control_record["median_ts"])
            candidate_ts = float(candidate_record["median_ts"])
            gain = (candidate_ts / control_ts - 1.0) * 100.0
            comparisons.append({
                "depth": depth,
                "control_median_ts": control_ts,
                "candidate_median_ts": candidate_ts,
                "gain_percent": gain,
                "control_mad_over_median": control_record["mad_over_median"],
                "candidate_mad_over_median": candidate_record["mad_over_median"],
                "control_stable": control_record["stable"],
                "candidate_stable": candidate_record["stable"],
            })
            if not control_record["stable"] or not candidate_record["stable"]:
                failures.append(f"depth {depth}: unstable arm")
            if gain < -MAX_REGRESSION_PERCENT:
                failures.append(f"depth {depth}: regression {gain:.3f}% exceeds {MAX_REGRESSION_PERCENT:.1f}%")
        gain_8k = comparisons[-1]["gain_percent"]
        if gain_8k < MIN_PROMISING_8K_GAIN_PERCENT:
            failures.append(
                f"8K gain {gain_8k:.3f}% is below the {MIN_PROMISING_8K_GAIN_PERCENT:.1f}% short-screen continuation threshold"
            )

    if errors:
        classification = "INVALID"
    elif failures:
        classification = "NO-GO"
    else:
        classification = "PROMISING_SHORT_SCREEN"

    output = {
        "schema_version": 1,
        "complete": not errors,
        "classification": classification,
        "optimization_accepted": False,
        "correctness_artifact": str(args.correctness_dir),
        "provenance_identity_matched": not errors,
        "contract": {
            "depths": EXPECTED_DEPTHS,
            "n_gen": EXPECTED_GEN,
            "raw_repetitions": EXPECTED_RAW_REPS,
            "discard_first": 1,
            "accepted_repetitions": EXPECTED_ACCEPTED,
            "maximum_regression_percent": MAX_REGRESSION_PERCENT,
            "minimum_promising_8k_gain_percent": MIN_PROMISING_8K_GAIN_PERCENT,
            "screen_can_accept_optimization": False,
        },
        "comparisons": comparisons,
        "errors": errors,
        "failures": failures,
    }
    args.json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"DSV4 BF16 HIDDEN ALLREDUCE SHORT TG SCREEN: {classification}")
    for row in comparisons:
        print(
            f"depth={row['depth']} control={row['control_median_ts']:.4f} candidate={row['candidate_median_ts']:.4f} "
            f"gain={row['gain_percent']:+.3f}% MAD={row['control_mad_over_median']:.3%}/{row['candidate_mad_over_median']:.3%}"
        )
    for message in errors + failures:
        print(f"- {message}")
    return 2 if errors else (0 if not failures else 1)


if __name__ == "__main__":
    raise SystemExit(main())