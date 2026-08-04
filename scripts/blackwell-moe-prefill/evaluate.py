from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def require_number(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{key} must be a positive finite number")
    return result


def fused_w13_result(data: dict[str, Any]) -> dict[str, Any] | None:
    if "baseline" not in data or "fused_w13" not in data:
        return None
    baseline = require_number(data["baseline"], "pp8192_ms")
    candidate = require_number(data["fused_w13"], "pp8192_ms")
    saved = baseline - candidate
    return {
        "baseline_ms": baseline,
        "candidate_ms": candidate,
        "saved_ms": saved,
        "speedup": baseline / candidate,
        "gate_ms": 15.0,
        "passed": saved >= 15.0,
    }


def grouped_mmq_result(data: dict[str, Any]) -> dict[str, Any] | None:
    if "grouped_mmq" not in data:
        return None
    measurements = data["grouped_mmq"]
    result: dict[str, Any] = {}
    passed = True
    for name, gate in (("m2048", 1.7), ("m8192", 2.2)):
        case = measurements.get(name)
        if not isinstance(case, dict):
            raise ValueError(f"grouped_mmq.{name} must be an object")
        generic_ms = require_number(case, "generic_ms")
        persistent_ms = require_number(case, "persistent_ms")
        speedup = generic_ms / persistent_ms
        case_passed = speedup >= gate
        passed = passed and case_passed
        result[name] = {
            "generic_ms": generic_ms,
            "persistent_ms": persistent_ms,
            "speedup": speedup,
            "gate": gate,
            "passed": case_passed,
        }

    full_model_ms = require_number(measurements, "full_model_gemm_ms")
    full_model_passed = full_model_ms < 150.0
    passed = passed and full_model_passed
    result["full_model_gemm"] = {
        "measured_ms": full_model_ms,
        "gate_ms": 150.0,
        "stretch_ms": 105.0,
        "passed": full_model_passed,
    }
    result["passed"] = passed
    return result


def moe_pipeline_result(data: dict[str, Any]) -> dict[str, Any] | None:
    if "moe_pipeline" not in data:
        return None
    measured = require_number(data["moe_pipeline"], "measured_ms")
    return {
        "measured_ms": measured,
        "target_upper_ms": 170.0,
        "stretch_ms": 140.0,
        "passed": measured <= 170.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("measurements", type=Path)
    args = parser.parse_args()

    data = json.loads(args.measurements.read_text(encoding="utf-8"))
    results = {
        "fused_w13": fused_w13_result(data),
        "grouped_mmq": grouped_mmq_result(data),
        "moe_pipeline": moe_pipeline_result(data),
    }
    present = [value for value in results.values() if value is not None]
    passed = all(bool(value["passed"]) for value in present)
    results["passed"] = passed
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
