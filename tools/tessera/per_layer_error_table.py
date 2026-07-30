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
error computation and the per-layer aggregation.

For each pair (L1, L1.5) with the same tensor name, the per-tensor
error is the relative Frobenius-norm-squared difference

    epsilon(l, b) = ||L15(l) - L1(l)||^2_F / ||L15(l)||^2_F

where ``l`` indexes the layer and ``b`` indexes the tensor within
the layer. This is the canonical "how much did quantization move
this tensor relative to its original FP16 reference" metric for
Tessera. The denominator normalizes by the reference energy, so
the metric is comparable across tensors of different sizes.

The per-layer total is the sum of per-tensor epsilons within a
layer. The summary reports the mean and max across tensors, the
total number of tensors, and the number of layers.

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
        --out         /path/to/per_layer_error_table.json \\
        --format      json

    python3 tools/tessera/per_layer_error_table.py \\
        --sidecar-dir /path/to/sidecars \\
        --out         /path/to/per_layer_error_table.txt \\
        --format      table

Missing-pair behavior
---------------------

If a sidecar has only an L1 file (no matching L1.5), the pair is
skipped with a warning printed to stderr; the same goes for an
L1.5 file with no L1. The tool never crashes on a partial
directory; it always emits a valid output document.

Schema versioning
-----------------

The JSON output is versioned by a top-level ``schema`` field:

    "schema": "llama.tessera.per-layer-error-table.v1"

so downstream consumers (L3 metric, L5 IterQuant orchestrator,
the L6 plan writer) can dispatch on the schema name. Adding a
field is non-breaking; removing or renaming a field requires
bumping the schema name (``.v2``).
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import l3_sidecar_v3_reader as reader  # noqa: E402

SCHEMA = "llama.tessera.per-layer-error-table.v1"

L1_SUFFIX = ".dequant.f32"
L15_SUFFIX = ".act.dequant.f32"

# Matches the canonical block (layer) prefix used by llama.cpp / ggml:
#   blk.<N>.<component>.<kind>[.expert_index]
# Examples it should match:
#   blk.3.attn_q.weight
#   blk.7.ffn_down.bias
#   blk.12.ffn_moe_up.3.weight
#   blk.0.attn_norm.weight
_BLK_RE = re.compile(r"^(blk\.\d+)\.")


def _err(msg: str, *args: Any) -> None:
    sys.stderr.write("per_layer_error_table: " + (msg % args) + "\n")


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
    # Fall back: strip any trailing ``.<int>`` (expert index), but
    # only for tensors that already have a ``blk.`` shape; otherwise
    # keep the full name so embedding/output tensors stay
    # distinguishable from the block tensors.
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

        # F32 in, F32 out. Use the same dtype through the diff and
        # the norm; the reader already returns float32.
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


def aggregate_per_layer(per_tensor: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sum per-tensor epsilons per layer. Layers are sorted by
    block index when they are ``blk.<N>`` style, otherwise
    alphabetically for stability. Tensors with ``epsilon is None``
    (shape mismatch) are not summed into the layer total.
    """
    sums: Dict[str, Dict[str, Any]] = {}
    for rec in per_tensor:
        layer = rec["layer"]
        eps = rec["epsilon"]
        if eps is None:
            continue
        s = sums.setdefault(layer, {"layer": layer, "total_epsilon": 0.0,
                                    "n_tensors": 0})
        s["total_epsilon"] += float(eps)
        s["n_tensors"] += 1

    def _sort_key(item: Dict[str, Any]) -> Tuple[int, str]:
        m = re.match(r"^blk\.(\d+)$", item["layer"])
        if m is not None:
            return (0, "%010d" % int(m.group(1)))
        return (1, item["layer"])

    return [sums[k] for k in sorted(sums, key=lambda l: _sort_key(sums[l]))]


def build_summary(per_tensor: List[Dict[str, Any]],
                  per_layer: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the top-level summary block."""
    finite = [r["epsilon"] for r in per_tensor
              if r["epsilon"] is not None]
    if finite:
        mean_eps = float(sum(finite) / len(finite))
        max_eps = float(max(finite))
    else:
        mean_eps = 0.0
        max_eps = 0.0
    return {
        "n_tensors":    len(per_tensor),
        "n_layers":     len(per_layer),
        "mean_epsilon": mean_eps,
        "max_epsilon":  max_eps,
    }


def build_table(per_tensor: List[Dict[str, Any]],
                per_layer: List[Dict[str, Any]],
                summary: Dict[str, Any]) -> str:
    """Render the result as a human-readable, greppable table."""
    lines: List[str] = []
    lines.append("# Tessera per-layer error table (schema=%s)" % SCHEMA)
    lines.append("# epsilon(l, b) = ||L15 - L1||^2_F / ||L15||^2_F")
    lines.append("")
    lines.append("## Summary")
    lines.append("n_tensors     = %d" % summary["n_tensors"])
    lines.append("n_layers      = %d" % summary["n_layers"])
    lines.append("mean_epsilon  = %.6e" % summary["mean_epsilon"])
    lines.append("max_epsilon   = %.6e" % summary["max_epsilon"])
    lines.append("")
    lines.append("## Per-tensor")
    lines.append("layer          tensor                            epsilon")
    for rec in per_tensor:
        eps = rec["epsilon"]
        eps_s = "%.6e" % eps if eps is not None else "n/a(shape)"
        lines.append("%-14s %-32s %s" % (rec["layer"], rec["name"], eps_s))
    lines.append("")
    lines.append("## Per-layer")
    lines.append("layer          total_epsilon  n_tensors")
    for rec in per_layer:
        lines.append("%-14s %-13.6e %d" % (
            rec["layer"], rec["total_epsilon"], rec["n_tensors"]))
    lines.append("")
    return "\n".join(lines)


def _main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Tessera per-layer error table: compares L1 vs L1.5 v3 "
            "dequant sidecars and aggregates to per-layer totals."))
    p.add_argument("--sidecar-dir", required=True,
                   help="directory containing L1 .dequant.f32 and "
                        "L1.5 .act.dequant.f32 v3 sidecars")
    p.add_argument("--out", required=True,
                   help="output path (JSON or text, depending on --format)")
    p.add_argument("--format", choices=("json", "table"), default="json",
                   help="output format (default: json)")
    args = p.parse_args(argv)

    l1_paths, l15_paths = scan_sidecar_dir(args.sidecar_dir)
    if not l1_paths and not l15_paths:
        _err("no L1 or L1.5 sidecar files found in %s", args.sidecar_dir)
        # Still emit an empty-but-valid document.
        empty = {
            "schema":     SCHEMA,
            "per_tensor": [],
            "per_layer":  [],
            "summary":    build_summary([], []),
            "missing":    {"l1_only": [], "l15_only": []},
        }
        if args.format == "json":
            with open(args.out, "w") as f:
                json.dump(empty, f, indent=2)
                f.write("\n")
        else:
            with open(args.out, "w") as f:
                f.write(build_table([], [], empty["summary"]))
        return 0

    pairs, missing_l1, missing_l15 = pair_l1_l15(l1_paths, l15_paths)
    for path in missing_l1:
        _err("missing L1.5 for %s (skipping)", path)
    for path in missing_l15:
        _err("missing L1 for %s (skipping)", path)

    per_tensor = compute_per_tensor(pairs)
    per_layer = aggregate_per_layer(per_tensor)
    summary = build_summary(per_tensor, per_layer)

    if args.format == "json":
        doc = {
            "schema":     SCHEMA,
            "per_tensor": per_tensor,
            "per_layer":  per_layer,
            "summary":    summary,
            "missing":    {
                "l1_only":  missing_l1,
                "l15_only": missing_l15,
            },
        }
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
    else:
        with open(args.out, "w") as f:
            f.write(build_table(per_tensor, per_layer, summary))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
