from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_logits(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size == 0:
        raise ValueError(f"empty logits file: {path}")
    return values


def json_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--max-nmse", type=float, default=1e-8)
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()

    for name in ("rtol", "atol", "max_nmse"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")

    reference = load_logits(args.reference)
    candidate = load_logits(args.candidate)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"logit count differs: {reference.size} != {candidate.size}"
        )

    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    finite = bool(np.isfinite(reference64).all() and np.isfinite(candidate64).all())
    diff = candidate64 - reference64
    mse = float(np.mean(np.square(diff)))
    reference_power = float(np.mean(np.square(reference64)))
    nmse = mse / reference_power if reference_power != 0.0 else (0.0 if mse == 0.0 else float("inf"))
    close_mask = np.isclose(candidate, reference, rtol=args.rtol, atol=args.atol)
    close = bool(close_mask.all())
    different = np.flatnonzero(candidate != reference)
    mismatched = np.flatnonzero(~close_mask)
    passed = finite and close and nmse <= args.max_nmse

    first_diff_index = int(different[0]) if different.size else None
    first_mismatch_index = int(mismatched[0]) if mismatched.size else None

    print(json.dumps({
        "count": int(reference.size),
        "max_abs": json_number(float(np.max(np.abs(diff)))),
        "mean_abs": json_number(float(np.mean(np.abs(diff)))),
        "rmse": json_number(float(np.sqrt(mse))),
        "nmse": json_number(nmse),
        "finite": finite,
        "allclose": close,
        "first_diff_index": first_diff_index,
        "first_diff_reference": None if first_diff_index is None else json_number(float(reference[first_diff_index])),
        "first_diff_candidate": None if first_diff_index is None else json_number(float(candidate[first_diff_index])),
        "first_mismatch_index": first_mismatch_index,
        "metrics_only": args.metrics_only,
        "passed": None if args.metrics_only else passed,
    }, sort_keys=True))
    return 0 if (finite if args.metrics_only else passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
