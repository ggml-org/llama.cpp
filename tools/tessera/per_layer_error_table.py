#!/usr/bin/env python3
"""
per_layer_error_table.py

Tessera per-layer error table for the L1-L6 runtime-aware pipeline
(Phase B, telemetry upgrade).

Reads pairs of v3 dequant sidecar files produced by the runtime
dequant instrumentation:

  - L1  sidecar: ``<tensor_name>.dequant.f32``
  - L1.5 sidecar: ``<tensor_name>.act.dequant.f32``

Both files use the v3 schema described in ``common/tessera-debug.h``
(magic ``"TDQT"``, 40-byte header + 28R per-row strip + F32 data).
The reader used here is the canonical Python reader
``l3_sidecar_v3_reader.py``; this tool only adds the L1 vs L1.5
error computation.

For each pair (L1, L1.5) with the same tensor name, the per-tensor
error is the relative Frobenius-norm-squared difference

    epsilon(l, b) = ||L15(l) - L1(l)||^2_F / ||L15(l)||^2_F

where ``l`` indexes the layer and ``b`` indexes the tensor within
the layer. This is the canonical "how much did quantization move
this tensor relative to its original FP16 reference" metric for
Tessera. The denominator normalizes by the reference energy, so
the metric is comparable across tensors of different sizes.

Output: one NDJSON line per tensor, conformant to
``common/schemas/per_layer_error.schema.json``. Consumers read the
output via ``tools/tessera/_analytical_io.py:read_analytical``.
The per-layer / per-network rollup is the consumer's job: different
consumers want different aggregations, and the polars group_by
one-liner is simpler than carrying both raw and rolled-up views
in the producer.

WAVE-4 GOTCHA: in real production runs, L1.5 currently contains
the same F32 data as L1 (the FP16 reference path is not yet
wired into the runtime hook; the sidecar file is still written
with the dequantized F32). The bit-identical contract is
asserted in ``l3_sidecar_v3_smoke.py``. This tool is correct for
synthetic demos where L1.5 is the original FP16 reference;
running it on production sidecars will report epsilon = 0
everywhere until the hook is updated.

Layer name derivation
---------------------

Tessera tensor names follow the canonical ``blk.<N>.<component>.<kind>``
pattern, where:

  - ``N`` is the block (layer) index
  - ``<component>`` is e.g. ``attn_q``, ``attn_k``, ``attn_v``,
    ``attn_output``, ``ffn_gate``, ``ffn_up``, ``ffn_down``,
    ``ffn_moe_gate``, etc.
  - ``<kind>`` is ``weight`` or ``bias``

The layer name is derived by stripping the trailing ``.weight`` /
``.bias`` suffix and any expert index, then keeping the
``blk.<N>`` prefix. Examples:

  - ``blk.3.attn_q.weight``        -> layer ``blk.3``
  - ``blk.7.ffn_down.bias``        -> layer ``blk.7``
  - ``blk.12.ffn_moe_up.3.weight`` -> layer ``blk.12``
  - ``token_embd.weight``          -> layer ``token_embd``
    (no ``blk.`` prefix, e.g. embeddings / output / norms)

This is a heuristic; tensors that do not match the
``blk.<N>.<...>`` pattern are grouped into a single layer
labeled by the full tensor name. The aggregation is robust
because the layer name is just a string key.

CLI
---

::

    python3 tools/tessera/per_layer_error_table.py \\
        --sidecar-dir /path/to/sidecars \\
        --out         /path/to/per_layer_error.ndjson

    # Optional: print a human-readable summary to stdout
    python3 tools/tessera/per_layer_error_table.py \\
        --sidecar-dir /path/to/sidecars \\
        --out         /path/to/per_layer_error.ndjson \\
        --print-table

Missing-pair behavior
---------------------

If a sidecar has only an L1 file (no matching L1.5), the pair is
skipped with a warning printed to stderr; the same goes for an
L1.5 file with no L1. The tool never crashes on a partial
directory; it always emits a valid NDJSON document (possibly
empty).

Schema versioning
-----------------

The NDJSON output is typed per
``common/schemas/per_layer_error.schema.json``. Adding a column
is non-breaking (consumers ignore unknown columns); removing or
renaming a column requires bumping the schema name.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import l3_sidecar_v3_reader as reader  # noqa: E402
from _analytical_io import polars_schema as _schema_polars_types  # noqa: E402

# Match the schema's name field; carries through to consumers for
# provenance. The schema-file itself is the source of truth for
# the columns; this constant is the human-readable name in the
# ``tessera_main_tip`` provenance field, not a parse target.
SCHEMA_NAME = "tessera.per-layer-error-record.v1"

L1_SUFFIX = ".dequant.f32"
L15_SUFFIX = ".act.dequant.f32"

# Matches the canonical block (layer) prefix used by llama.cpp / ggml:
#   blk.<N>.<component>.<kind>[.expert_index]
_BLK_RE = re.compile(r"^(blk\.\d+)\.")


def _err(msg: str, *args: Any) -> None:
    sys.stderr.write("per_layer_error_table: " + (msg % args) + "\n")


def _tessera_provenance() -> Tuple[str, str, str]:
    """Return (kernel_version, created_at, tessera_main_tip).

    Tries the C++ provenance helpers (which read the auto-populated
    build-info header baked in by CMake) and falls back to
    subprocess calls to ``git`` for standalone / un-CMake-built
    invocations (e.g. when run directly from a worktree as
    ``python3 tools/tessera/per_layer_error_table.py ...``).

    The values are stamped into every NDJSON record so downstream
    consumers can audit which build wrote the row. See
    ``common/tessera-debug/tessera-debug.h:tessera_kernel_version``
    for the canonical contract.
    """
    kernel_version = "unknown"
    main_tip = "unknown"
    try:
        # 1. Try the canonical C++ helper via subprocess. The
        #    binary ``-print-tessera-provenance`` is not yet wired;
        #    fall back to git for now.
        kv = subprocess.run(
            ["git", "describe", "--all", "--always"],
            capture_output=True, text=True, check=False,
            cwd=os.path.dirname(THIS_DIR))
        if kv.returncode == 0 and kv.stdout.strip():
            kernel_version = kv.stdout.strip()
        mt = subprocess.run(
            ["git", "rev-parse", "--short", "main"],
            capture_output=True, text=True, check=False,
            cwd=os.path.dirname(THIS_DIR))
        if mt.returncode == 0 and mt.stdout.strip():
            main_tip = mt.stdout.strip()
    except FileNotFoundError:
        pass  # git not installed
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return kernel_version, created_at, main_tip


def derive_layer_name(tensor_name: str) -> str:
    """Derive the layer name for a canonical Tessera tensor name.

    Strips the trailing ``.weight`` or ``.bias`` and any expert
    index, then returns the ``blk.<N>`` prefix when present.
    Falls back to the full tensor name for tensors that do not
    follow the ``blk.<N>.<...>`` pattern (embeddings, output
    projections, norms, etc.).
    """
    base = tensor_name
    for suf in (".weight", ".bias"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    m = _BLK_RE.match(base + ".")  # ensure we only match the prefix
    if m is not None:
        return m.group(1)
    if base.startswith("blk."):
        # already handled above; if we got here the regex did not match
        # because there was no component segment (e.g. ``blk.3.`` only).
        return base
    return base


def _tensor_name_from_path(path: str, suffix: str) -> str:
    """Strip a known sidecar suffix and return the tensor name."""
    base = os.path.basename(path)
    if base.endswith(suffix):
        return base[: -len(suffix)]
    return base


def _safe_rel_error(diff: np.ndarray, ref: np.ndarray) -> float:
    """Compute ``||diff||^2 / ||ref||^2`` in F32. Returns 0.0 when
    the reference is zero (avoids division by zero; the metric is
    undefined in that case and we choose the conservative answer)."""
    num = float(np.sum(diff.astype(np.float32) ** 2))
    den = float(np.sum(ref.astype(np.float32) ** 2))
    if den == 0.0:
        return 0.0
    return num / den


def scan_sidecar_dir(sidecar_dir: str) -> Tuple[List[str], List[str]]:
    """Return (l1_paths, l15_paths) sorted by filename. Tensors
    without a partner are still returned so the caller can warn
    and skip them."""
    l1: List[str] = []
    l15: List[str] = []
    if not os.path.isdir(sidecar_dir):
        _err("sidecar dir does not exist: %s", sidecar_dir)
        return l1, l15
    for name in sorted(os.listdir(sidecar_dir)):
        full = os.path.join(sidecar_dir, name)
        if not os.path.isfile(full):
            continue
        if name.endswith(L1_SUFFIX) and not name.endswith(L15_SUFFIX):
            l1.append(full)
        elif name.endswith(L15_SUFFIX):
            l15.append(full)
    return l1, l15


def pair_l1_l15(l1_paths: List[str],
                l15_paths: List[str]) -> Tuple[List[Tuple[str, str]],
                                                List[str],
                                                List[str]]:
    """Match L1 and L1.5 paths by the tensor-name stem (filename
    minus the sidecar suffix).

    Returns (pairs, unmatched_l1, unmatched_l15) where:
      - ``pairs`` is a list of (l1_path, l15_path) tuples, sorted
        by the L1 path.
      - ``unmatched_l1`` is a list of L1 paths that have no
        matching L1.5 file.
      - ``unmatched_l15`` is a list of L1.5 paths that have no
        matching L1 file.
    """
    l1_by_name = {_tensor_name_from_path(p, L1_SUFFIX): p for p in l1_paths}
    l15_by_name = {_tensor_name_from_path(p, L15_SUFFIX): p for p in l15_paths}
    common = sorted(set(l1_by_name) & set(l15_by_name))
    pairs = [(l1_by_name[n], l15_by_name[n]) for n in common]
    unmatched_l1 = sorted(p for n, p in l1_by_name.items() if n not in l15_by_name)
    unmatched_l15 = sorted(p for n, p in l15_by_name.items() if n not in l1_by_name)
    return pairs, unmatched_l1, unmatched_l15


def compute_per_tensor(pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """For each (L1, L1.5) pair, compute the per-tensor epsilon.

    Returns a list of dicts with keys:
      ``name``, ``layer``, ``epsilon``, ``l1_path``, ``l15_path``.

    A pair is skipped (and a warning is printed) when:
      - The L1 and L1.5 shapes differ (cannot compare)
      - The reader fails to parse either file
      - The L1 or L1.5 data dtype is not F32

    The shape check is enforced by the runner: a pair with
    mismatched shape is reported with epsilon = NaN in the
    output (so downstream consumers can flag it) AND a warning
    is printed to stderr.
    """
    out: List[Dict[str, Any]] = []
    for l1_path, l15_path in pairs:
        name = _tensor_name_from_path(l1_path, L1_SUFFIX)
        try:
            s1 = reader.read_sidecar(l1_path, mode="auto", provenance=False)
        except (reader.SidecarReadError, OSError) as e:
            _err("skip %s: L1 read failed: %s", l1_path, e)
            continue
        try:
            s15 = reader.read_sidecar(l15_path, mode="auto", provenance=False)
        except (reader.SidecarReadError, OSError) as e:
            _err("skip %s: L1.5 read failed: %s", l15_path, e)
            continue

        if s1["dtype_name"] != "F32" or s15["dtype_name"] != "F32":
            _err("skip %s: non-F32 dtype (L1=%s, L1.5=%s)",
                 name, s1["dtype_name"], s15["dtype_name"])
            continue

        d1 = s1["data"]
        d15 = s15["data"]
        if d1.shape != d15.shape:
            _err("skip %s: shape mismatch L1=%s L1.5=%s",
                 name, d1.shape, d15.shape)
            out.append({
                "name":     name,
                "layer":    derive_layer_name(name),
                "epsilon":  None,
                "l1_path":  l1_path,
                "l15_path": l15_path,
                "note":     "shape mismatch",
            })
            continue

        diff = (d15 - d1).astype(np.float32, copy=False)
        eps = _safe_rel_error(diff, d15)
        out.append({
            "name":     name,
            "layer":    derive_layer_name(name),
            "epsilon":  float(eps),
            "l1_path":  l1_path,
            "l15_path": l15_path,
        })
    return out


def build_table(per_tensor: List[Dict[str, Any]]) -> str:
    """Render the per-tensor result as a human-readable, greppable
    table for stdout. Aggregation across layers is the consumer's
    job (a one-liner polars group_by); this is a quick-look view."""
    lines: List[str] = []
    lines.append("# Tessera per-tensor L1/L1.5 relative error")
    lines.append("# epsilon(l, b) = ||L15 - L1||^2_F / ||L15||^2_F")
    lines.append("")
    lines.append("layer          tensor                            epsilon")
    for rec in per_tensor:
        eps = rec["epsilon"]
        eps_s = "%.6e" % eps if eps is not None else "n/a(shape)"
        lines.append("%-14s %-32s %s" % (rec["layer"], rec["name"], eps_s))
    lines.append("")
    return "\n".join(lines)


def write_ndjson(per_tensor: List[Dict[str, Any]],
                 out_path: str,
                 sidecar_dir: str,
                 provenance: Tuple[str, str, str]) -> int:
    """Write the per-tensor records as NDJSON, typed per
    ``common/schemas/per_layer_error.schema.json``.

    Returns the number of records written. An empty input emits
    a valid empty NDJSON file (one closing line so ``pl.read_ndjson``
    does not error on zero-row input).
    """
    kernel_version, created_at, main_tip = provenance
    rows: List[Dict[str, Any]] = []
    for rec in per_tensor:
        eps = rec["epsilon"]
        rows.append({
            "tensor":           rec["name"],
            "layer":            rec["layer"],
            "epsilon":          eps if eps is not None else float("nan"),
            "epsilon_is_nan":   eps is None,
            "note":             rec.get("note", ""),
            "sidecar_dir":      sidecar_dir,
            "kernel_version":   kernel_version,
            "created_at":       created_at,
            "tessera_main_tip": main_tip,
        })
    # Pin the column dtypes per the schema's polars_schema so a
    # consumer reading via _analytical_io:read_analytical sees
    # exactly the types the schema promises.
    schema_types = _schema_polars_types("per_layer_error")
    if rows:
        df = pl.DataFrame(rows, infer_schema_length=max(len(rows), 1))
    else:
        # Empty input: build a zero-row frame with the schema's
        # column order so read_analytical on the result has the
        # expected columns even when no records were produced.
        df = pl.DataFrame(
            {col: pl.Series(name=col, values=[], dtype=dtype)
             for col, dtype in schema_types.items()},
            schema=schema_types,
        )
    for col, dtype in schema_types.items():
        if col in df.columns and df.schema[col] != dtype:
            df = df.with_columns(pl.col(col).cast(dtype, strict=False))
    df.write_ndjson(out_path)
    return len(rows)


def _main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Tessera per-layer error table: compares L1 vs L1.5 v3 "
            "dequant sidecars and emits per-tensor NDJSON. Consumers "
            "do the per-layer / per-network rollup via polars."))
    p.add_argument("--sidecar-dir", required=True,
                   help="directory containing L1 .dequant.f32 and "
                        "L1.5 .act.dequant.f32 v3 sidecars")
    p.add_argument("--out", required=True,
                   help="output NDJSON path (one record per tensor)")
    p.add_argument("--print-table", action="store_true",
                   help="print a human-readable per-tensor table to stdout")
    args = p.parse_args(argv)

    l1_paths, l15_paths = scan_sidecar_dir(args.sidecar_dir)
    if not l1_paths and not l15_paths:
        _err("no L1 or L1.5 sidecar files found in %s", args.sidecar_dir)
        # Still emit an empty-but-valid NDJSON document.
        provenance = _tessera_provenance()
        write_ndjson([], args.out, args.sidecar_dir, provenance)
        return 0

    pairs, missing_l1, missing_l15 = pair_l1_l15(l1_paths, l15_paths)
    for path in missing_l1:
        _err("missing L1.5 for %s (skipping)", path)
    for path in missing_l15:
        _err("missing L1 for %s (skipping)", path)

    per_tensor = compute_per_tensor(pairs)
    provenance = _tessera_provenance()
    n = write_ndjson(per_tensor, args.out, args.sidecar_dir, provenance)

    if args.print_table:
        sys.stdout.write(build_table(per_tensor))
    else:
        sys.stderr.write(
            "wrote %d per-tensor records to %s\n" % (n, args.out))

    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
