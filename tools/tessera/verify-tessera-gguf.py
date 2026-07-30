#!/usr/bin/env python3
"""Reject numerically corrupt Tessera GGUF artifacts before inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


FLOAT_KINDS = {"f"}
COMPONENT_SUFFIXES = (
    "_packed",
    "_page_scales",
    "_lane_scales",
    "_outlier_row_offsets",
    "_outlier_cols",
    "_outlier_vals",
    "_act_scale",
)


def verify(path: Path) -> dict:
    from gguf import GGUFReader

    reader = GGUFReader(path, "r")
    nonfinite: list[dict] = []
    component_names: set[str] = set()
    rope_constants: list[dict] = []
    components: dict[str, dict[str, np.ndarray]] = {}
    float_tensors = 0
    values_checked = 0

    for tensor in reader.tensors:
        name = tensor.name
        values = np.asarray(tensor.data)
        if any(name.endswith(suffix) for suffix in COMPONENT_SUFFIXES):
            component_names.add(name)
        for suffix in COMPONENT_SUFFIXES:
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                components.setdefault(stem, {})[suffix] = values.reshape(-1)
                break
        if values.dtype.kind not in FLOAT_KINDS:
            continue
        float_tensors += 1
        flat = values.reshape(-1)
        values_checked += flat.size
        finite = np.isfinite(flat)
        count = int(np.count_nonzero(~finite))
        if count:
            indices = np.flatnonzero(~finite)
            nonfinite.append(
                {
                    "tensor": name,
                    "dtype": str(values.dtype),
                    "count": count,
                    "first_indices": indices[:16].tolist(),
                }
            )
        if name.endswith("rope_freqs.weight"):
            rope_constants.append(
                {
                    "tensor": name,
                    "dtype": str(values.dtype),
                    "values": int(flat.size),
                    "max_abs": (
                        float(np.max(np.abs(flat[finite])))
                        if np.any(finite)
                        else None
                    ),
                }
            )

    quantized_rope = sorted(
        name
        for name in component_names
        if ".rope_freqs.weight_" in name
    )
    csr_issues: list[dict] = []
    for stem, parts in components.items():
        offsets = parts.get("_outlier_row_offsets")
        columns = parts.get("_outlier_cols")
        page_scales = parts.get("_page_scales")
        if offsets is None or columns is None or page_scales is None:
            continue
        offsets_i64 = offsets.astype(np.int64, copy=False)
        rows = offsets_i64.size - 1
        reasons: list[str] = []
        if rows <= 0:
            reasons.append("row-offset tensor has no rows")
            pages_per_row = 0
        elif page_scales.size % rows:
            reasons.append("page-scale count is not divisible by row count")
            pages_per_row = 0
        else:
            pages_per_row = page_scales.size // rows
        if offsets_i64[0] != 0:
            reasons.append("first row offset is not zero")
        differences = np.diff(offsets_i64)
        if np.any(differences < 0):
            reasons.append("row offsets are not monotonic")
        if offsets_i64[-1] != columns.size:
            reasons.append("final row offset does not equal column count")
        physical_width = pages_per_row * 640
        if physical_width and np.any(differences > physical_width):
            reasons.append("a row has more residuals than physical columns")
        columns_i64 = columns.astype(np.int64, copy=False)
        if physical_width and (
            np.any(columns_i64 < 0) or np.any(columns_i64 >= physical_width)
        ):
            reasons.append("residual column lies outside physical row width")
        if not reasons and columns_i64.size:
            for row in np.flatnonzero(differences):
                start = offsets_i64[row]
                end = offsets_i64[row + 1]
                row_columns = columns_i64[start:end]
                if np.unique(row_columns).size != row_columns.size:
                    reasons.append(f"row {int(row)} contains duplicate residual columns")
                    break
        if reasons:
            csr_issues.append(
                {
                    "tensor": stem,
                    "rows": rows,
                    "pages_per_row": int(pages_per_row),
                    "residuals": int(columns.size),
                    "max_residuals_per_row": (
                        int(np.max(differences)) if differences.size else 0
                    ),
                    "reasons": reasons,
                }
            )
    return {
        "path": str(path),
        "tensor_count": len(reader.tensors),
        "float_tensors": float_tensors,
        "float_values_checked": values_checked,
        "nonfinite": nonfinite,
        "quantized_rope_components": quantized_rope,
        "csr_issues": csr_issues,
        "rope_constants": rope_constants,
        "valid": not nonfinite and not quantized_rope and not csr_issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = verify(args.gguf)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(encoded + "\n")
    print(encoded)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
