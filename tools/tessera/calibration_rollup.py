#!/usr/bin/env python3
"""Tessera calibration rollup: cross-pipeline join of the four
analytical outputs into a single tidy long-format table.

Phase C of the polars-integration scout
(``docs/tessera-polars-integration-scout.md``). The four Tessera
analytical outputs (L3 outlier, L4 probe, L5 plan, per-layer
error) are now written as typed NDJSON by their respective
producers; this script reads them and joins on ``tensor`` (and
``layer`` when present) into a single per-tensor row with every
relevant signal in one place. The output is the analytical force
multiplier the scout identified: a single table that answers
"for each tensor, what did the L3 outlier rate, the L4 MSE, the
L5 sensitivity score, and the L1/L1.5 sidecar error say?"

Output formats:

  * ``--out-parquet <path>`` (canonical): a single parquet file
    suitable for DuckDB and downstream polars work. The row
    group size mirrors ``tools/tessera/evidence-store.py:write_part``
    (zstd compression, statistics on, row_group_size=65536).
  * ``--out-ndjson <path>`` (optional): an NDJSON copy for
    human inspection or other tools. The schema is the union
    of the input schemas, with provenance columns (``_source_*``)
    prefixing the per-file source names so multiple L3 outlier
    runs (different sidecar labels) can co-exist in the same
    table.

Usage::

    python3 tools/tessera/calibration_rollup.py \\
        --l3-outlier     ckpt-v3-tile640.ndjson:q4_k=ckpt-v3-tile640.ndjson \\
        --l4-probe      l4.ndjson \\
        --l5-plan       l5.ndjson \\
        --per-layer-error per-layer-error.ndjson \\
        --out-parquet   evidence/calibration-rollup.parquet \\
        --out-ndjson    evidence/calibration-rollup.ndjson

The ``--l3-outlier`` form ``LABEL=PATH`` lets multiple L3 outlier
files (one per quant variant or per calibration corpus) feed
into the same rollup; without ``=`` the LABEL defaults to the
file's stem.

The rollup is the consumer-side joining the scout identified as
"the analytical force multiplier" - what used to take three
hand-written JSON parsers in three different scripts is now a
single ``pl.concat`` / ``pl.join`` chain.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import polars as pl

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from _analytical_io import read_analytical  # noqa: E402

# Map from the user-facing CLI arg name to the schema name in
# common/schemas/. The polars_schema lookup in _analytical_io is
# keyed by schema name; this map is the CLI -> schema bridge.
# The imatrix source is special: it is a parquet file (the
# per-channel observer written by evidence-store.py:ingest_imatrix),
# not a typed-NDJSON. The schema "imatrix" is in this map for
# the SCHEMA_BY_FLAG convention, but the reader is the
# parquet reducer below (no read_analytical call).
# The l5_outcome source is also special: it is a tessera.duckdb
# path (the unified-DB feedback-loop tables l5_outcome +
# l5_weights), not a typed-NDJSON; the reader is the SQL
# reducer below (no read_analytical call).
SCHEMA_BY_FLAG = {
    "l3_outlier":     "l3_outlier",
    "l4_probe":       "l4_probe",
    "l5_plan":        "l5_plan",
    "per_layer_error": "per_layer_error",
    "imatrix":       "imatrix",
    "l5_outcome":    "l5_outcome_db",
}

# Columns that participate in the join key. Every input has
# ``tensor``; ``layer`` is also present in all four (the scout
# confirmed the layer extraction is consistent across producers).
# ``iteration`` (L5 only) and ``sidecar_label`` (L3 only) are
# preserved as data columns, not join keys.
JOIN_KEYS = ["tensor"]


# imatrix (per-channel observer) -> per-tensor summary. The
# reduction is mean-of-channels for rms / mean_abs / kurtosis
# and max-of-channels for tail_ratio (the worst case drives
# the L5 sensitivity scoring). The column list below is the
# rollup-facing column set: tensor + the per-tensor summary +
# a derived layer (extracted from the tensor name). It does
# NOT include a schema in common/schemas/ — the C++ side does
# not produce an imatrix; the schema is Python-side only.
# ``tensor`` (not ``name``) is the join key so the imatrix
# source joins on the same key as L3 / L4 / L5 / PLE.
IMATRIX_TIDY_COLS: tuple[str, ...] = (
    "tensor", "layer", "rms", "mean_abs", "tail_ratio", "kurtosis",
)


# l5_outcome (tessera.duckdb unified-DB) -> per-tensor summary.
# The rollup joins the DB on (model_hash, name) for the
# ``l5_outcome`` table (per-iteration) and on (model_hash,
# family) for the ``l5_weights`` table (per-family). The
# reducer below picks the most-recent per (model_hash, name)
# from l5_outcome (highest iteration, then plan_id), joins
# l5_weights on (model_hash, family), and derives the
# ``recommended_action`` per tensor via the l5_action rules.
# The result uses ``tensor`` (not ``name``) as the join key so
# this source joins on the same key as L3 / L4 / L5 / PLE.
L5_OUTCOME_TIDY_COLS: tuple[str, ...] = (
    "tensor",
    "miscalibration_score",   # l5_weights.slope
    "hit_rate",               # l5_weights.hit_rate
    "recommended_weight_im",  # l5_weights.w_imatrix
    "recommended_weight_grad", # l5_weights.w_gradient
    "recommended_weight_layer", # l5_weights.w_layer
    "delta_mse",              # most recent l5_outcome.delta_mse
    "plan_accepted",          # most recent l5_outcome.plan_accepted
    "residual",               # most recent l5_outcome.residual
    "recommended_action",     # derived via l5_action rules
)


# Per-tensor layer extraction: same convention as
# l5_orchestrator.py::_layer and per_layer_error_table.py.
# Returns the integer block index for blk.N. / h.N. / blocks.N. /
# layers.N. / model.layers.N. tensors, 0 for non-block tensors.
def _layer_of(tensor_name: str) -> int:
    base = tensor_name
    for suf in (".weight", ".bias"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    for prefix in ("blk.", "blocks.", "h.", "layers.", "model.layers."):
        idx = base.find(prefix)
        if idx < 0:
            continue
        start = idx + len(prefix)
        end = start
        while end < len(base) and base[end].isdigit():
            end += 1
        if end > start:
            try:
                return int(base[start:end])
            except ValueError:
                return 0
    return 0


# ---------------------------------------------------------------------------
# CLI arg parsing helpers
# ---------------------------------------------------------------------------


def _parse_labeled_path(arg: str, default_label_from_path: bool = True) -> Tuple[str, Path]:
    """Parse a ``LABEL=PATH`` argument. PATH may not contain ``=``;
    LABEL is a human-readable name (e.g. 'q4_k', 'ckpt-v3') that
    becomes the per-source column prefix in the rollup output.

    Without ``=``, the label defaults to the file's stem (when
    ``default_label_from_path`` is True).
    """
    if "=" in arg:
        label, path = arg.split("=", 1)
        if not label or not path:
            raise argparse.ArgumentTypeError(
                f"expected LABEL=PATH, got {arg!r}")
        return label, Path(path)
    p = Path(arg)
    if default_label_from_path:
        return p.stem, p
    return "", p


def _arg_labeled_paths(label: str) -> List[Tuple[str, Path]]:
    """The argparse ``type=`` for repeatable labeled-path args."""
    def _parse(arg: str) -> Tuple[str, Path]:
        return _parse_labeled_path(arg, default_label_from_path=True)
    return _parse  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Per-source column prefixing
# ---------------------------------------------------------------------------


# Columns that should NOT be prefixed because they are join keys
# (and would collide across sources). Provenance columns
# (kernel_version, created_at, tessera_main_tip) are DROPPED
# from the rollup entirely; the rollup stamps its own. The
# per-source provenance is preserved in the source NDJSON
# files but does not need to be in the rollup table.
_KEEP_RAW = {
    "tensor",          # join key
    "layer",           # join key (when present in all inputs)
}

# Per-source provenance that is KEPT in the rollup (prefixed with
# the source label so multiple sources do not collide). These are
# useful for the consumer: e.g. ``l3.outlier_fraction`` tells the
# consumer the L3 outlier rate from the L3 source; knowing which
# L3 sidecar dir the row came from (``l3.sidecar_label``) lets the
# consumer cross-reference with the L3 source's CLI invocation.
_PER_SOURCE_KEEP = {
    "sidecar_label",   # L3 outlier
    "iteration",       # L5 plan
    "plan_id",         # L5 plan
}

# Provenance columns that the rollup drops from every source
# (the rollup stamps its own at write time). These are
# over-recorded by the producers and would only duplicate in
# the rollup output.
_DROP_FROM_ROLLUP = {
    "kernel_version",
    "created_at",
    "tessera_main_tip",
}


def _prefix_columns(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Rename every column except the join keys to ``<prefix>.<col>``,
    and drop the per-source provenance columns the rollup replaces
    with its own (kernel_version, created_at, tessera_main_tip).

    Schema columns that are not present in this source (e.g.
    ``sidecar_label`` in an L4 probe) are simply absent; we do
    NOT add nulls for missing columns here, because the rollup's
    outer join handles the per-column coverage naturally.
    """
    # Drop the per-source provenance first; the rollup stamps
    # its own at write time. Doing this BEFORE the rename keeps
    # the prefixing pure and predictable.
    drop = [c for c in df.columns if c in _DROP_FROM_ROLLUP]
    if drop:
        df = df.drop(drop)
    rename: dict[str, str] = {}
    for col in df.columns:
        if col in _KEEP_RAW:
            continue
        rename[col] = f"{prefix}.{col}"
    return df.rename(rename)


# ---------------------------------------------------------------------------
# Rollup driver
# ---------------------------------------------------------------------------


def _add_labeled_arg(p: argparse.ArgumentParser, flag: str, dest: str, schema: str) -> None:
    """Add a repeatable ``--flag LABEL=PATH`` arg to the parser.

    The schema name is held in the ``dest`` so the driver can
    look it up. ``schema`` is one of the four shipped schemas in
    common/schemas/ (or the special ``imatrix`` / ``l5_outcome_db``
    keys for the parquet + DuckDB read paths).
    """
    if schema in ("imatrix", "l5_outcome_db"):
        kind = "parquet" if schema == "imatrix" else "tessera.duckdb"
        help_text = (
            f"A {kind} file to roll up. May be repeated. "
            f"LABEL becomes the per-source column prefix; without "
            f"'=' the LABEL defaults to the file's stem. "
            f"Required companion: --model-hash for l5_outcome."
        )
    else:
        help_text = (
            f"A {schema} NDJSON file to roll up. May be repeated. "
            f"LABEL becomes the per-source column prefix; without "
            f"'=' the LABEL defaults to the file's stem. Schema: "
            f"common/schemas/{schema}.schema.json."
        )
    p.add_argument(
        f"--{flag}",
        action="append",
        type=_arg_labeled_paths(flag),
        dest=dest,
        metavar="LABEL=PATH",
        default=[],
        help=help_text,
    )


def _tessera_provenance() -> tuple[str, str, str]:
    """Mirror the convention from per_layer_error_table.py and
    l3_outlier_report.py: shell out to git for kernel_version
    and tessera_main_tip; stamp created_at in UTC.
    """
    kernel_version = "unknown"
    main_tip = "unknown"
    try:
        r = subprocess.run(
            ["git", "describe", "--all", "--always"],
            capture_output=True, text=True, check=False,
            cwd=str(THIS_DIR.parent.parent))
        if r.returncode == 0 and r.stdout.strip():
            kernel_version = r.stdout.strip()
        r = subprocess.run(
            ["git", "rev-parse", "--short", "main"],
            capture_output=True, text=True, check=False,
            cwd=str(THIS_DIR.parent.parent))
        if r.returncode == 0 and r.stdout.strip():
            main_tip = r.stdout.strip()
    except FileNotFoundError:
        pass
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return kernel_version, created_at, main_tip


def _read_source(label: str, path: Path, schema: str,
                 model_hash: Optional[str] = None) -> pl.DataFrame:
    """Read one source file, prefix non-join columns with ``label.``.

    Raises FileNotFoundError if the file is missing; ValueError
    if the schema name is unknown (re-raised from read_analytical).
    The imatrix source is special: it's a parquet file (the
    per-channel observer written by
    ``evidence-store.py:ingest_imatrix``), reduced to a per-tensor
    summary before prefixing. The l5_outcome source is also
    special: it's a tessera.duckdb path (the unified-DB
    feedback-loop tables l5_outcome + l5_weights), reduced to a
    per-tensor summary before prefixing. The prefix for the
    l5_outcome source is forced to ``l5`` (the user-facing
    contract is ``l5.*``; the file path doesn't carry useful
    provenance). Other sources are typed-NDJSON.

    The model_hash argument is required for ``l5_outcome_db`` and
    unused for the other schemas.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{label}: file not found: {path}")
    if schema == "imatrix":
        df = _read_imatrix_tidy(path)
    elif schema == "l5_outcome_db":
        if model_hash is None:
            raise ValueError(
                f"{label}: --l5-outcome requires --model-hash"
            )
        df = _read_l5_outcome_tidy(path, model_hash)
    else:
        df = read_analytical(path, schema)
    # Force the l5_outcome prefix to ``l5`` regardless of the
    # user-supplied label (the file path / stem is uninformative;
    # the user-facing contract is ``l5.*``).
    effective_label = "l5" if schema == "l5_outcome_db" else label
    return _prefix_columns(df, effective_label)


def _read_imatrix_tidy(path: Path) -> pl.DataFrame:
    """Read a per-channel observer parquet (the artifact of
    ``evidence-store.py:ingest_imatrix``) and reduce to a
    per-tensor summary. Per-tensor reduction:
      rms, mean_abs, kurtosis  -- mean across channels
      tail_ratio              -- max across channels
    The layer is derived from the tensor name (blk.N. / h.N. /
    blocks.N. / layers.N. / model.layers.N.) so the result
    joins cleanly with the L3 / L4 / L5 / PLE rollup on
    (tensor, layer).

    The output is a per-tensor tidy frame in the same shape as
    a typed-NDJSON source: name + layer + numeric columns.
    The column-prefix step in ``_rollup`` then writes the
    imatrix.* columns in the rollup output.
    """
    raw = pl.read_parquet(path)
    if raw.height == 0:
        return pl.DataFrame(
            {c: [] for c in IMATRIX_TIDY_COLS},
            schema={c: pl.Float64 if c not in ("name",) else pl.String
                    for c in IMATRIX_TIDY_COLS},
        )
    if "tensor" not in raw.columns:
        raise ValueError(
            f"imatrix parquet {path} has no 'tensor' column; "
            f"got columns: {raw.columns}"
        )
    # Per-tensor reduction. Some columns may be missing if the
    # imatrix observer was produced by a downstream tool that
    # trimmed them; coerce to nullable Float64.
    agg_cols: list[pl.Expr] = []
    for src, dst, fn in (
        ("rms",        "rms",        "mean"),
        ("mean_abs",   "mean_abs",   "mean"),
        ("tail_ratio", "tail_ratio", "max"),
        ("kurtosis",   "kurtosis",   "mean"),
    ):
        if src in raw.columns:
            agg_cols.append(getattr(pl.col(src), fn)().alias(dst))
    if not agg_cols:
        raise ValueError(
            f"imatrix parquet {path} has none of rms / mean_abs / "
            f"tail_ratio / kurtosis columns; got: {raw.columns}"
        )
    summary = raw.group_by("tensor").agg(agg_cols)
    # layer: derive from tensor name. Use the same string format
    # as the L3 / L4 / PLE / L5 producers ("blk.0", "blk.1", ...)
    # so the outer join in _rollup can coalesce the layer column
    # across sources. An integer block index would conflict
    # with the L3 / L4 string layer at the polars layer.
    def _row_layer(name: str) -> str:
        idx = _layer_of(name)
        return f"blk.{idx}"
    layers: list[str] = [_row_layer(t) for t in summary["tensor"].to_list()]
    summary = summary.with_columns(pl.Series("layer", layers, dtype=pl.Utf8))
    return summary.select(IMATRIX_TIDY_COLS)


def _read_l5_outcome_tidy(path: Path, model_hash: str) -> pl.DataFrame:
    """Read the per-tensor feedback-loop summary from the
    unified-DB ``l5_outcome`` + ``l5_weights`` tables.

    Per-tensor reduction:
      miscalibration_score, hit_rate, recommended_weight_*:
        per-(model_hash, family) row from l5_weights, joined
        onto the per-tensor l5_outcome rows by family.
      delta_mse, plan_accepted, residual:
        the most recent (model_hash, name) row from l5_outcome
        (highest iteration, then plan_id).
      recommended_action:
        derived via the l5_action rules from
        (miscalibration_score, hit_rate, delta_mse,
        plan_accepted).

    On a missing DB / missing tables / no rows, returns an
    empty frame with the L5_OUTCOME_TIDY_COLS schema so the
    outer join in _rollup produces no l5.* columns (the
    coverage is empty). This is the safe default for the
    cross-pipeline consumer: the rollup is robust to a
    not-yet-retuned model.
    """
    empty = pl.DataFrame(
        {c: [] for c in L5_OUTCOME_TIDY_COLS},
        schema={
            "tensor": pl.Utf8,
            "miscalibration_score": pl.Float64,
            "hit_rate": pl.Float64,
            "recommended_weight_im": pl.Float64,
            "recommended_weight_grad": pl.Float64,
            "recommended_weight_layer": pl.Float64,
            "delta_mse": pl.Float64,
            "plan_accepted": pl.Boolean,
            "residual": pl.Float64,
            "recommended_action": pl.Utf8,
        },
    )
    if not path.is_file():
        return empty
    import duckdb
    try:
        con = duckdb.connect(str(path), read_only=True)
    except Exception:
        return empty
    try:
        names = {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()}
        if "l5_outcome" not in names or "l5_weights" not in names:
            return empty
        # Most-recent per-(model_hash, name) from l5_outcome.
        # The ROW_NUMBER window picks the highest iteration,
        # then the lexicographically-largest plan_id as a
        # stable tiebreaker (the consumer is a long-running
        # retune; plan_id is opaque but stable per iteration).
        outcome_recent_sql = (
            "WITH ranked AS ("
            "  SELECT name, family, delta_mse, plan_accepted, "
            "         residual, "
            "         ROW_NUMBER() OVER ("
            "           PARTITION BY name "
            "           ORDER BY iteration DESC, plan_id DESC"
            "         ) AS rn "
            "  FROM l5_outcome WHERE model_hash = ?"
            ") "
            "SELECT name, family, delta_mse, plan_accepted, "
            "       residual FROM ranked WHERE rn = 1"
        )
        # l5_weights per (model_hash, family): the per-family
        # retune verdict. slope is the OLS slope of delta_mse on
        # sensitivity_score; the rollup surfaces it as
        # miscalibration_score.
        weights_sql = (
            "SELECT family, slope, hit_rate, w_imatrix, "
            "       w_gradient, w_layer "
            "FROM l5_weights WHERE model_hash = ?"
        )
        outcome_rows = con.execute(
            outcome_recent_sql, [model_hash]).fetchall()
        weight_rows = con.execute(
            weights_sql, [model_hash]).fetchall()
    except Exception:
        return empty
    finally:
        try:
            con.close()
        except Exception:
            pass
    if not outcome_rows:
        return empty
    # Index l5_weights by family for the per-row join below.
    weights_by_family: dict[str, tuple] = {
        r[0]: r for r in weight_rows
    }
    # Late import: l5_action is a small pure-python module;
    # importing it at module top would force the test fixtures
    # to sys.path it on every import of calibration_rollup.
    from l5_action import derive_recommended_action
    tensors: list[str] = []
    miscal: list[float | None] = []
    hit: list[float | None] = []
    w_im: list[float | None] = []
    w_grad: list[float | None] = []
    w_layer: list[float | None] = []
    d_mse: list[float | None] = []
    p_acc: list[bool | None] = []
    resid: list[float | None] = []
    actions: list[str] = []
    for (name, family, delta_mse, plan_accepted, residual) in outcome_rows:
        # The most recent outcome row's family determines which
        # l5_weights row applies. l5_outcome.family is written
        # by l5_orchestrator / l5_outcome.py and is the
        # canonical family tag for the tensor.
        weights = weights_by_family.get(family or "")
        if weights is not None:
            _, slope, hit_rate, wim, wgr, wly = weights
        else:
            slope, hit_rate = None, None
            wim, wgr, wly = None, None, None
        action = derive_recommended_action(
            slope, hit_rate, delta_mse, plan_accepted)
        tensors.append(name)
        miscal.append(slope)
        hit.append(hit_rate)
        w_im.append(wim)
        w_grad.append(wgr)
        w_layer.append(wly)
        d_mse.append(delta_mse)
        p_acc.append(plan_accepted)
        resid.append(residual)
        actions.append(action)
    return pl.DataFrame(
        {
            "tensor":                  tensors,
            "miscalibration_score":    miscal,
            "hit_rate":                hit,
            "recommended_weight_im":   w_im,
            "recommended_weight_grad": w_grad,
            "recommended_weight_layer":w_layer,
            "delta_mse":               d_mse,
            "plan_accepted":           p_acc,
            "residual":                resid,
            "recommended_action":      actions,
        },
        schema={
            "tensor": pl.Utf8,
            "miscalibration_score": pl.Float64,
            "hit_rate": pl.Float64,
            "recommended_weight_im": pl.Float64,
            "recommended_weight_grad": pl.Float64,
            "recommended_weight_layer": pl.Float64,
            "delta_mse": pl.Float64,
            "plan_accepted": pl.Boolean,
            "residual": pl.Float64,
            "recommended_action": pl.Utf8,
        },
    )


def _rollup(sources: List[Tuple[str, Path, str]],
            model_hash: Optional[str] = None) -> pl.DataFrame:
    """Outer-join the sources on ``tensor``.

    ``sources`` is a list of ``(label, path, schema_name)`` tuples
    in CLI-arg order. The first source seeds the join; subsequent
    sources are outer-joined onto the accumulator.

    ``model_hash`` is required when any source has the
    ``l5_outcome_db`` schema (the unified-DB feedback-loop
    tables are keyed by model_hash). It is unused for the
    NDJSON and imatrix sources.

    The ``layer`` column is added to the join keys when present
    in the first source AND in every subsequent source; otherwise
    it stays as a data column (possibly null for sources that do
    not record it).
    """
    if not sources:
        return pl.DataFrame()
    # Decide whether ``layer`` is a join key. It is when every
    # source has it; otherwise we outer-join on tensor only and
    # let layer be a possibly-null data column.
    use_layer = True
    frames: List[pl.DataFrame] = []
    for i, (label, path, schema) in enumerate(sources):
        df = _read_source(label, path, schema, model_hash=model_hash)
        if "layer" not in df.columns:
            use_layer = False
        frames.append(df)
    keys = ["tensor", "layer"] if use_layer else ["tensor"]
    # Full-outer-join sequentially. polars handles the per-column
    # coalescing via the join suffix convention; the
    # already-prefixed columns are unique so we never see a
    # ``_right`` suffix in the output.
    accumulator = frames[0]
    for nxt in frames[1:]:
        accumulator = accumulator.join(nxt, on=keys, how="full",
                                       coalesce=True)
    # Stamp rollup-level provenance.
    kernel_version, created_at, main_tip = _tessera_provenance()
    accumulator = accumulator.with_columns([
        pl.lit(kernel_version).alias("kernel_version"),
        pl.lit(created_at).alias("created_at"),
        pl.lit(main_tip).alias("tessera_main_tip"),
    ])
    return accumulator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Tessera calibration rollup: outer-join the four analytical "
            "outputs (L3 outlier, L4 probe, L5 plan, per-layer error) "
            "and the per-tensor imatrix summary on tensor (+ layer "
            "when present) into a single tidy parquet table. The output "
            "is the analytical force multiplier for cross-pipeline "
            "queries (DuckDB, polars, spreadsheet review)."
        ),
    )
    for flag, schema in SCHEMA_BY_FLAG.items():
        # The CLI flag uses dashes (--l3-outlier); the attribute
        # dest uses underscores (args.l3_outlier). Both forms are
        # documented; argparse's default dash-to-underscore
        # conversion only fires when no explicit dest is set, so
        # we set it explicitly here.
        _add_labeled_arg(p, flag.replace("_", "-"), flag, schema)
    p.add_argument(
        "--out-parquet",
        type=Path,
        default=None,
        help="Output parquet path (canonical artifact for DuckDB / polars).",
    )
    p.add_argument(
        "--out-ndjson",
        type=Path,
        default=None,
        help="Output NDJSON path (human-inspectable copy of the rollup).",
    )
    p.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a one-line-per-source coverage summary to stdout.",
    )
    p.add_argument(
        "--model-hash",
        default=None,
        help=(
            "Model hash for the --l5-outcome source (the unified-DB "
            "l5_outcome + l5_weights tables are keyed by model_hash). "
            "Required when --l5-outcome is provided; ignored otherwise."
        ),
    )
    return p


def _coverage_summary(sources: List[Tuple[str, Path, str]],
                      rollup: pl.DataFrame) -> str:
    """A short human-readable summary of the rollup: per-source
    row count, per-source column count in the rollup, total rows
    in the rollup, total columns. Useful as a sanity check."""
    lines: List[str] = ["calibration rollup summary"]
    lines.append("=" * 72)
    lines.append(f"  sources:  {len(sources)}")
    for label, path, schema in sources:
        # The l5_outcome_db source is force-prefixed to ``l5``;
        # the user-supplied label is informational only.
        effective = "l5" if schema == "l5_outcome_db" else label
        # The per-source column count is the number of columns
        # the source contributed to the rollup; we re-read the
        # source's schema to count its columns, but the rollup
        # has them prefixed with the source label.
        prefix = f"{effective}."
        contributed = sum(1 for c in rollup.columns if c.startswith(prefix))
        lines.append(
            f"    [{label}] schema={schema}  path={path}  "
            f"contributed_columns={contributed}")
    lines.append(f"  total_rows:   {rollup.height}")
    lines.append(f"  total_cols:   {rollup.width}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    sources: List[Tuple[str, Path, str]] = []
    has_l5_outcome_db = False
    for flag, schema in SCHEMA_BY_FLAG.items():
        for label, path in getattr(args, flag):
            sources.append((label, path, schema))
            if schema == "l5_outcome_db":
                has_l5_outcome_db = True
    if not sources:
        sys.stderr.write(
            "calibration_rollup: no source files provided; pass at "
            "least one of --l3-outlier / --l4-probe / --l5-plan / "
            "--per-layer-error / --imatrix-tidy / --l5-outcome.\n")
        return 2
    if has_l5_outcome_db and not args.model_hash:
        sys.stderr.write(
            "calibration_rollup: --l5-outcome requires --model-hash; "
            "the l5_outcome + l5_weights tables are keyed by model_hash.\n"
        )
        return 2
    rollup = _rollup(sources, model_hash=args.model_hash)
    if args.print_summary:
        print(_coverage_summary(sources, rollup))
    if args.out_parquet is None and args.out_ndjson is None:
        sys.stderr.write(
            "calibration_rollup: no --out-parquet or --out-ndjson "
            "specified; rollup is computed but not written. Re-run "
            "with --print-summary to see the coverage.\n")
        return 0
    if args.out_parquet is not None:
        args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
        # Mirror the evidence-store convention: zstd compression,
        # statistics on, row_group_size=65536.
        rollup.write_parquet(
            args.out_parquet,
            compression="zstd",
            statistics=True,
            row_group_size=65536,
        )
        sys.stderr.write(
            f"wrote {rollup.height} rows to {args.out_parquet}\n")
    if args.out_ndjson is not None:
        args.out_ndjson.parent.mkdir(parents=True, exist_ok=True)
        rollup.write_ndjson(args.out_ndjson)
        sys.stderr.write(
            f"wrote {rollup.height} rows to {args.out_ndjson}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
