"""Wire the Python calibration pipeline to ``tensor_stats``.

The C++ dispatch's GA-prep walk already upserts per-tensor
``kurtosis`` and ``eff_rank`` into ``tensor_stats`` (see
``tessera-dispatch.cpp``). This script handles the Python side:
reads the per-channel observer parquet written by
``evidence-store.py ingest-imatrix``, reduces to a per-tensor
summary, and upserts ``rms / mean_abs / tail_ratio`` (plus a
family / layer / dtype derivation) into ``tensor_stats``.

The reduction is: per-tensor mean of the per-channel rms and
mean_abs; per-tensor max of the per-channel tail_ratio. These
match the conventions the C++ side uses for the same fields
(see ``tessera-quantize-db.cpp::TS_QDB_SCHEMA_SQL`` for the
column list and the comment block above the ``tensors`` table
on the per-tensor reduction rules).

The source field is set to ``\"py_cal\"`` so the two sides'
writes are distinguishable. The upsert is via
``TesseraDB.insert_tensor_stats`` (typed helper, same
``ON CONFLICT (model_hash, name) DO UPDATE`` semantics as the
C++ side). Both sides can write the same row in any order;
the last writer wins on the columns it sets, the rest are
preserved by the upsert.

Usage::

    python3 tools/tessera/calibration_to_tensor_stats.py \\
        --imatrix imatrix.gguf \\
        --db tessera.duckdb \\
        --model-hash <hash> \\
        --evidence-store /path/to/evidence

Either ``--imatrix`` (the GGUF the calibration pipeline produced)
or ``--evidence-store`` (the directory holding
``observer/part-*.parquet``) must be provided. When both are
given, ``--evidence-store`` wins (the parquet is the post-parse
artifact; the GGUF is re-parsed only as a fallback when the
parquet is missing).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import polars as pl

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from tessera_db import TesseraDB


# Per-tensor reduction. rms / mean_abs are mean-of-channels
# (the calibration pipeline averages the per-channel magnitude
# per tensor). tail_ratio is max-of-channels (the worst case
# drives the L5 sensitivity scoring).
def _reduce_observer(observer_df: pl.DataFrame) -> pl.DataFrame:
    if observer_df.height == 0:
        return pl.DataFrame()
    # The observer parquet columns are per the legacy schema:
    # run_id, source, schema, tensor, expert, channel, count, sum,
    # sum2, sum4, maxabs, rms, mean_abs, kurtosis, tail_ratio.
    # Per-tensor summary: group by tensor, take mean of rms and
    # mean_abs, max of tail_ratio, mean of kurtosis (used as a
    # cross-check against the C++ side's value).
    summary = (
        observer_df.group_by("tensor")
        .agg(
            pl.col("rms").mean().alias("rms"),
            pl.col("mean_abs").mean().alias("mean_abs"),
            pl.col("tail_ratio").max().alias("tail_ratio"),
            pl.col("kurtosis").mean().alias("kurtosis_check"),
        )
    )
    return summary


def _read_observer_parquet(evidence_store: Path) -> pl.DataFrame:
    """Read every ``observer/part-*.parquet`` under the evidence
    store and concatenate. The polars lazy API handles the row
    group layout; the eager collect is fine for the calibration
    scope (typically 100-10000 tensors, 16-256 channels each)."""
    parts = sorted((evidence_store / "observer").glob("part-*.parquet"))
    if not parts:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(p) for p in parts], how="vertical_relaxed")


def _family_of(name: str) -> str:
    """Lightweight family inference: same convention as
    ts_regime_infer_family (attn_q, attn_k, attn_v, attn_output,
    ffn_gate, ffn_up, ffn_down, ffn_output, other). Used only
    for the ``family`` column on tensor_stats; the C++ side's
    family classification wins on a subsequent write because
    the upsert preserves whichever side writes last."""
    n = name
    for suf in (".weight", ".bias"):
        if n.endswith(suf):
            n = n[: -len(suf)]
            break
    for prefix, fam in (
        ("attn_q", "attn_q"), ("attn_k", "attn_k"),
        ("attn_v", "attn_v"), ("attn_output", "attn_output"),
        ("ffn_gate", "ffn_gate"), ("ffn_up", "ffn_up"),
        ("ffn_down", "ffn_down"), ("ffn_output", "ffn_output"),
    ):
        if n.endswith(prefix) or f".{prefix}." in n:
            return fam
    return "other"


def _layer_of(name: str) -> int:
    """Extract the block index from a tensor name (blk.N. / h.N. /
    blocks.N. / layers.N. / model.layers.N.). Returns 0 for
    non-block tensors (norm, embed, output)."""
    for prefix in ("blk.", "blocks.", "h.", "layers.", "model.layers."):
        idx = name.find(prefix)
        if idx < 0:
            continue
        start = idx + len(prefix)
        end = start
        while end < len(name) and name[end].isdigit():
            end += 1
        if end > start:
            try:
                return int(name[start:end])
            except ValueError:
                return 0
    return 0


def run(
    *,
    db_path: Path,
    model_hash: str,
    evidence_store: Optional[Path] = None,
    imatrix_path: Optional[Path] = None,
) -> int:
    """Reduce the per-channel observer data and upsert the
    per-tensor summary into ``tensor_stats``. Returns the number
    of rows written.
    """
    if evidence_store is not None:
        observer = _read_observer_parquet(evidence_store)
    elif imatrix_path is not None:
        # Fallback: re-parse the GGUF through evidence-store's
        # machinery. This is slower (parses the GGUF twice) but
        # works when the parquet directory is missing.
        from evidence_store import observer_frames  # type: ignore
        reader = import_gguf(imatrix_path)  # type: ignore
        frames = list(observer_frames(reader, "", imatrix_path, 1 << 30))
        reader.close()
        if not frames:
            return 0
        observer = pl.concat(frames, how="vertical_relaxed")
    else:
        raise ValueError(
            "either --evidence-store or --imatrix must be provided"
        )
    if observer.height == 0:
        return 0
    summary = _reduce_observer(observer)
    if summary.height == 0:
        return 0
    rows = []
    for r in summary.to_dicts():
        name = r["tensor"]
        rows.append({
            "name":         name,
            "family":       _family_of(name),
            "layer_depth":  _layer_of(name),
            "rms":          float(r["rms"]) if r["rms"] is not None else None,
            "mean_abs":     float(r["mean_abs"]) if r["mean_abs"] is not None else None,
            "tail_ratio":   float(r["tail_ratio"]) if r["tail_ratio"] is not None else None,
            "source":       "py_cal",
        })
    with TesseraDB.open(db_path) as db:
        return db.insert_tensor_stats(model_hash=model_hash, rows=rows)


def import_gguf(path: Path):
    """Late import of the gguf-py reader; the calibration harness
    container may not have it."""
    from gguf import GGUFReader  # type: ignore
    return GGUFReader(str(path), "r")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Calibration pipeline -> tensor_stats: read the per-channel "
            "observer data and upsert the per-tensor rms / mean_abs / "
            "tail_ratio into the unified tessera.duckdb store."
        ),
    )
    p.add_argument(
        "--db", required=True, type=Path,
        help="Path to the unified tessera.duckdb file",
    )
    p.add_argument(
        "--model-hash", required=True,
        help="Model hash (must match the C++ dispatch's model_hash)",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--evidence-store", type=Path,
        help="Directory holding observer/part-*.parquet (preferred)",
    )
    g.add_argument(
        "--imatrix", type=Path,
        help="Path to the imatrix.gguf (fallback when parquet is missing)",
    )
    p.add_argument(
        "--print-summary", action="store_true",
        help="Print a one-line summary after writing.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    n = run(
        db_path=args.db,
        model_hash=args.model_hash,
        evidence_store=args.evidence_store,
        imatrix_path=args.imatrix,
    )
    if args.print_summary:
        print(f"calibration_to_tensor_stats: wrote {n} rows to tensor_stats")
    return 0


if __name__ == "__main__":
    sys.exit(main())
