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


def log_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    return shifted - np.log(np.sum(np.exp(shifted)))


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

    reference_norm = float(np.linalg.norm(reference64))
    candidate_norm = float(np.linalg.norm(candidate64))
    cosine = (
        float(np.dot(reference64, candidate64) / (reference_norm * candidate_norm))
        if reference_norm != 0.0 and candidate_norm != 0.0
        else float("nan")
    )
    abs_diff = np.abs(diff)
    percentiles = np.percentile(abs_diff, [50.0, 90.0, 99.0, 99.9])
    top1_reference = int(np.argmax(reference64))
    top1_candidate = int(np.argmax(candidate64))

    topk_overlap: dict[str, int] = {}
    for k in (10, 50):
        count = min(k, reference.size)
        reference_topk = np.argpartition(reference64, -count)[-count:]
        candidate_topk = np.argpartition(candidate64, -count)[-count:]
        topk_overlap[str(k)] = int(np.intersect1d(reference_topk, candidate_topk).size)

    reference_log_probability = log_softmax(reference64)
    candidate_log_probability = log_softmax(candidate64)
    reference_probability = np.exp(reference_log_probability)
    kl_divergence = float(np.sum(
        reference_probability * (reference_log_probability - candidate_log_probability)
    ))

    first_diff_index = int(different[0]) if different.size else None
    first_mismatch_index = int(mismatched[0]) if mismatched.size else None

    print(json.dumps({
        "count": int(reference.size),
        "max_abs": json_number(float(np.max(np.abs(diff)))),
        "mean_abs": json_number(float(np.mean(np.abs(diff)))),
        "p50_abs": json_number(float(percentiles[0])),
        "p90_abs": json_number(float(percentiles[1])),
        "p99_abs": json_number(float(percentiles[2])),
        "p999_abs": json_number(float(percentiles[3])),
        "rmse": json_number(float(np.sqrt(mse))),
        "nmse": json_number(nmse),
        "nrmse": json_number(float(math.sqrt(nmse))),
        "cosine_similarity": json_number(cosine),
        "kl_divergence": json_number(kl_divergence),
        "finite": finite,
        "allclose": close,
        "mismatch_count": int(mismatched.size),
        "mismatch_fraction": float(mismatched.size / reference.size),
        "top1_reference": top1_reference,
        "top1_candidate": top1_candidate,
        "top1_match": top1_reference == top1_candidate,
        "topk_overlap": topk_overlap,
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
