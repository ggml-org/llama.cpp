#!/usr/bin/env python3
"""
latency_lut.py

Tessera Phase C (telemetry upgrade) latency LUT builder.

Reads L1 (and optionally L1.5) v3 dequant sidecar files in a directory
and produces a per-(shape, kernel_id) latency lookup table. The per-row
timing_ns values from the v3 strip (see common/tessera-debug/tessera-
debug.h) are aggregated into per-tensor totals, per-row means, and
per-row population standard deviations; the resulting per-tensor
records are then grouped by the chosen key.

The v3 strip layout (per row, 24 bytes):

    offset  size  field
    ------  ----  -----------------------------------------------
         0     8  timing_ns            uint64
         8     4  kernel_id            uint32
        12     4  dispatch_count       uint32
        16     8  reserved             uint64

For each input sidecar we compute:

    per-tensor total   = sum(per_row_timing_ns)
    per-row mean       = total / rows
    per-row std (pop)  = sqrt(sum((x - mean)^2) / rows)

For each (shape, kernel_id) group we aggregate:

    mean_ns       = mean of per-row means across tensors in the group
    std_ns        = sqrt(mean of (per-row std^2))   -- within-tensor variability
    count         = number of tensors in the group
    mean_total_ns = mean of per-tensor totals across tensors in the group

Output JSON schema ("llama.tessera.latency-lut.v1"):

    {
      "schema": "llama.tessera.latency-lut.v1",
      "group_by": "<shape|kernel|shape-kernel>",
      "entries": [
        {
          "shape":         "<rows>x<cols>",
          "kernel_id":     <uint32>,
          "mean_ns":       <float>,
          "std_ns":        <float>,
          "count":         <int>,
          "mean_total_ns": <float>
        },
        ...
      ],
      "summary": {
        "n_tensors":    <int>,
        "n_groups":     <int>,
        "n_kernel_ids": <int>
      }
    }

Table output is a human-readable rendering of the same data.

Usage:

    python3 tools/tessera/latency_lut.py \\
        --sidecar-dir /path/to/dequant --out lut.json --format json

    python3 tools/tessera/latency_lut.py \\
        --sidecar-dir /path/to/dequant --out lut.txt --format table \\
        --group-by shape-kernel --include-l15

Files written to the output are ASCII only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import l3_sidecar_v3_reader as reader  # noqa: E402

SUFFIX_L1 = ".dequant.f32"
SUFFIX_L15 = ".act.dequant.f32"
SCHEMA = "llama.tessera.latency-lut.v1"
GROUP_SHAPE_KERNEL = "shape-kernel"
GROUP_SHAPE = "shape"
GROUP_KERNEL = "kernel"


def _err(msg, *args):
    sys.stderr.write("latency_lut: " + (msg % args) + "\n")


def _group_key(rows, cols, kernel_id, group_by):
    """Return a hashable tuple identifying a (shape, kernel) group."""
    shape = "%dx%d" % (int(rows), int(cols))
    if group_by == GROUP_SHAPE:
        return (shape,)
    if group_by == GROUP_KERNEL:
        return (int(kernel_id),)
    if group_by == GROUP_SHAPE_KERNEL:
        return (shape, int(kernel_id))
    raise ValueError("unknown group_by: %s" % group_by)


def _tensor_record(sidecar):
    """Build a per-tensor record from a sidecar dict as returned by
    l3_sidecar_v3_reader.read_sidecar.

    The tensor-level kernel_id is the first row's value: in the current
    runtime hook every row of a single tensor is dispatched by the same
    kernel, so the first row's value is the canonical tensor-level
    kernel_id. A mixed-kernel file would surface the first encountered
    value (no test exercises this case).
    """
    rows = int(sidecar["rows"])
    cols = int(sidecar["cols"])
    timing = list(sidecar["row_timing_ns"])
    kernel_id_list = list(sidecar["row_kernel_id"])
    kernel_id = int(kernel_id_list[0]) if kernel_id_list else 0
    n = len(timing)
    if n == 0:
        total = 0
        mean = 0.0
        std = 0.0
    else:
        total = int(sum(timing))
        mean = float(total) / float(n)
        var = sum((float(x) - mean) ** 2 for x in timing) / float(n)
        std = math.sqrt(var)
    return {
        "path":           sidecar["path"],
        "rows":           rows,
        "cols":           cols,
        "kernel_id":      kernel_id,
        "n":              n,
        "total":          total,
        "mean":           mean,
        "std":            std,
        "actual_version": int(sidecar["actual_version"]),
    }


def collect_sidecars(sidecar_dir, include_l15=False):
    """Walk sidecar_dir and return a list of per-tensor records.

    Only v3 files are processed (the v3 strip is required for the
    per-row timing_ns field); v1/v2 files are skipped with a warning
    to stderr. Provenance JSON is not read (the LUT does not need it).
    """
    p = Path(sidecar_dir)
    if not p.is_dir():
        raise FileNotFoundError("not a directory: %s" % sidecar_dir)
    # Walk the directory manually so the L1.5 suffix ("act.dequant.f32")
    # is not also matched by the L1 suffix ("dequant.f32"): a naive
    # `*.dequant.f32` glob matches both. We treat the suffixes as
    # ordered: L1.5 (longest) is checked first.
    paths = []
    for entry in sorted(p.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name
        if name.endswith(SUFFIX_L15):
            if include_l15:
                paths.append(entry)
        elif name.endswith(SUFFIX_L1):
            paths.append(entry)
    records = []
    for path in paths:
        try:
            sc = reader.read_sidecar(str(path), mode="auto", provenance=False)
        except (reader.SidecarReadError, OSError) as e:
            _err("failed to read %s: %s", path, e)
            continue
        if sc["actual_version"] < 3:
            _err("skipping v%d file (need v3 for per_row_timing_ns): %s" % (
                sc["actual_version"], path))
            continue
        records.append(_tensor_record(sc))
    return records


def aggregate(records, group_by):
    """Group per-tensor records by `group_by` and compute per-group
    summary statistics. Returns (entries, summary)."""
    groups = {}
    for r in records:
        key = _group_key(r["rows"], r["cols"], r["kernel_id"], group_by)
        groups.setdefault(key, []).append(r)
    entries = []
    for key, recs in groups.items():
        n_t = len(recs)
        mean_of_means = sum(r["mean"] for r in recs) / float(n_t)
        mean_of_var = sum(r["std"] ** 2 for r in recs) / float(n_t)
        std_within = math.sqrt(mean_of_var)
        mean_total = sum(r["total"] for r in recs) / float(n_t)
        if group_by == GROUP_SHAPE:
            shape = key[0]
            kernel_id = recs[0]["kernel_id"]
        elif group_by == GROUP_KERNEL:
            shape = "%dx%d" % (recs[0]["rows"], recs[0]["cols"])
            kernel_id = key[0]
        else:
            shape = key[0]
            kernel_id = key[1]
        entries.append({
            "shape":         shape,
            "kernel_id":     int(kernel_id),
            "mean_ns":       mean_of_means,
            "std_ns":        std_within,
            "count":         n_t,
            "mean_total_ns": mean_total,
        })
    entries.sort(key=lambda e: (e["shape"], e["kernel_id"]))
    n_kernel_ids = len({e["kernel_id"] for e in entries})
    summary = {
        "n_tensors":    len(records),
        "n_groups":     len(entries),
        "n_kernel_ids": n_kernel_ids,
    }
    return entries, summary


def render_table(entries, summary, group_by):
    """Render the LUT as a human-readable table."""
    lines = []
    lines.append("Tessera latency LUT (schema=%s, group_by=%s)" % (SCHEMA, group_by))
    if not entries:
        lines.append("(no entries)")
    else:
        lines.append("%-12s %-10s %14s %14s %6s %16s" % (
            "shape", "kernel_id", "mean_ns", "std_ns", "count", "mean_total_ns"))
        for e in entries:
            lines.append("%-12s %-10d %14.3f %14.3f %6d %16.3f" % (
                e["shape"], e["kernel_id"],
                e["mean_ns"], e["std_ns"],
                e["count"], e["mean_total_ns"]))
    lines.append("")
    lines.append("summary: n_tensors=%d n_groups=%d n_kernel_ids=%d" % (
        summary["n_tensors"], summary["n_groups"], summary["n_kernel_ids"]))
    return "\n".join(lines) + "\n"


def render_json(entries, summary, group_by):
    obj = {
        "schema":   SCHEMA,
        "group_by": group_by,
        "entries":  entries,
        "summary":  summary,
    }
    return json.dumps(obj, indent=2, sort_keys=False) + "\n"


def _main(argv):
    p = argparse.ArgumentParser(
        description="Build a per-(shape, kernel_id) latency LUT from "
                    "Tessera v3 dequant sidecars.")
    p.add_argument("--sidecar-dir", required=True,
                   help="directory containing .dequant.f32 sidecars")
    p.add_argument("--out", required=True,
                   help="output path (JSON or table depending on --format)")
    p.add_argument("--format", choices=("json", "table"), default="json",
                   help="output format (default: json)")
    p.add_argument("--group-by",
                   choices=(GROUP_SHAPE, GROUP_KERNEL, GROUP_SHAPE_KERNEL),
                   default=GROUP_SHAPE_KERNEL,
                   help="grouping key (default: shape-kernel)")
    p.add_argument("--include-l15", action="store_true",
                   help="also process L1.5 .act.dequant.f32 sidecars")
    args = p.parse_args(argv)

    records = collect_sidecars(args.sidecar_dir, include_l15=args.include_l15)
    entries, summary = aggregate(records, args.group_by)
    if args.format == "json":
        out = render_json(entries, summary, args.group_by)
    else:
        out = render_table(entries, summary, args.group_by)
    with open(args.out, "w") as f:
        f.write(out)
    _err("wrote %s (n_tensors=%d, n_groups=%d, n_kernel_ids=%d)" % (
        args.out, summary["n_tensors"], summary["n_groups"],
        summary["n_kernel_ids"]))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
