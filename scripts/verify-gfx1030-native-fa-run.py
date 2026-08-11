#!/usr/bin/env python3
"""Verify a completed gfx1030-native-fa harness artifact without using GPUs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys


EXPECTED_LABELS = {
    f"rep{rep:02d}-{graphs}-{impl}"
    for rep in range(1, 4)
    for graphs in ("on", "off")
    for impl in ("stock", "native")
}


def fail(message: str) -> "NoReturn":
    raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-profiles", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    try:
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest.get("failed"):
            fail("manifest reports failed commands")
        commands = manifest.get("commands", [])
        if not commands or any(x.get("returncode") != 0 for x in commands):
            fail("manifest contains a missing or nonzero command")

        test_files = sorted((run / "tests").glob("*.stdout.log"))
        if {p.stem.removesuffix(".stdout") for p in test_files} != EXPECTED_LABELS:
            fail(f"expected {len(EXPECTED_LABELS)} test logs, found {len(test_files)}")
        for path in test_files:
            text = path.read_text(errors="replace")
            if not re.search(r"\b5/5 backends passed\b", text):
                fail(f"correctness result missing from {path}")

        bench_files = sorted(run.glob("bench/*/*.stdout.log"))
        bench_labels = {p.parent.name for p in bench_files}
        if bench_labels != EXPECTED_LABELS:
            fail(f"expected {len(EXPECTED_LABELS)} benchmark labels, found {len(bench_labels)}")
        for path in bench_files:
            rows = json.loads(path.read_text())
            if len(rows) != 4:
                fail(f"expected PP512/4096/16384 and TG128 in {path}")
            if any(float(row["avg_ts"]) <= 0 for row in rows):
                fail(f"nonpositive benchmark result in {path}")

        profiles = sorted(run.glob("profiles/*/webhie/*_kernel_stats.csv"))
        if args.require_profiles and len(profiles) != 2:
            fail(f"expected two profiler stats files, found {len(profiles)}")
        for path in profiles:
            rows = list(csv.DictReader(path.open(newline="")))
            fa = [row for row in rows if "flash_attn_tile<" in row.get("Name", "")]
            if len(fa) != 1:
                fail(f"expected one FA tile row in {path}")
            name = fa[0]["Name"]
            expected_native = "native" in path.parent.parent.name
            if (", true>" in name) != expected_native:
                fail(f"kernel dispatch mismatch for {path}: {name}")
            if float(fa[0]["TotalDurationNs"]) <= 0:
                fail(f"nonpositive FA duration in {path}")

        print(f"verified run={run}")
        print(f"commands={len(commands)} tests={len(test_files)} benchmarks={len(bench_files)} profiles={len(profiles)}")
        print("all commands returned zero; all backend suites report 5/5; all benchmark JSON has 4 positive cases")
        return 0
    except (OSError, KeyError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())