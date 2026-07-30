#!/usr/bin/env python3
"""Safely import portable Tessera family priors into a local policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCHEMA = "llama.speculative.calibration-policy.v1"


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unsupported policy schema")
    return value


def portable_entries(policy: dict) -> dict:
    accepted = {}
    rejected = []
    for key, entry in policy.get("tensor_families", {}).items():
        matches = entry.get("match", [])
        unsafe = (
            bool(entry.get("exact", False))
            or key.startswith(("shadow_", "override:", "repair:"))
            or any("blk." in str(fragment) for fragment in matches)
        )
        if unsafe:
            rejected.append(key)
        else:
            accepted[key] = dict(entry)
    return accepted, rejected


def merge(base: dict, prior: dict, family: str) -> dict:
    result = dict(base)
    current = dict(base.get("tensor_families", {}))
    portable, rejected = portable_entries(prior)
    # Current local entries remain first, so equal-specificity matching keeps
    # direct current evidence ahead of a cross-epoch family prior.
    for key, entry in portable.items():
        current.setdefault(f"prior:{family}:{key}", entry)
    result["tensor_families"] = current
    result["tessera_calibration_prior"] = {
        "schema": "llama.tessera.calibration-prior.v1",
        "family": family,
        "portable_entries": sorted(portable),
        "rejected_nonportable_entries": sorted(rejected),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge portable Tessera calibration family priors")
    parser.add_argument("--base-policy", required=True)
    parser.add_argument("--prior-policy", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.family.replace(".", "").replace("-", "").isalnum():
        raise ValueError("family must be a simple architecture-family identifier")
    merged = merge(load(Path(args.base_policy)), load(Path(args.prior_policy)), args.family)
    Path(args.output).write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
