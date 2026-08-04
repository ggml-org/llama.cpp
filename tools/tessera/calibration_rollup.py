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
SCHEMA_BY_FLAG = {
    "l3_outlier":     "l3_outlier",
    "l4_probe":       "l4_probe",
    "l5_plan":        "l5_plan",
    "per_layer_error": "per_layer_error",
}

# Columns that participate in the join key. Every input has
# ``tensor``; ``layer`` is also present in all four (the scout
# confirmed the layer extraction is consistent across producers).
# ``iteration`` (L5 only) and ``sidecar_label`` (L3 only) are
# preserved as data columns, not join keys.
JOIN_KEYS = ["tensor"]


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
    common/schemas/.
    """
    p.add_argument(
        f"--{flag}",
        action="append",
        type=_arg_labeled_paths(flag),
        dest=dest,
        metavar="LABEL=PATH",
        default=[],
        help=(
            f"A {schema} NDJSON file to roll up. May be repeated. "
            f"LABEL becomes the per-source column prefix; without "
            f"'=' the LABEL defaults to the file's stem. Schema: "
            f"common/schemas/{schema}.schema.json."
        ),
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


def _read_source(label: str, path: Path, schema: str) -> pl.DataFrame:
    """Read one source file, prefix non-join columns with ``label.``.

    Raises FileNotFoundError if the file is missing; ValueError
    if the schema name is unknown (re-raised from read_analytical).
    """
    if not path.is_file():
        raise FileNotFoundError(f"{label}: file not found: {path}")
    df = read_analytical(path, schema)
    return _prefix_columns(df, label)


def _rollup(sources: List[Tuple[str, Path, str]]) -> pl.DataFrame:
    """Outer-join the sources on ``tensor``.

    ``sources`` is a list of ``(label, path, schema_name)`` tuples
    in CLI-arg order. The first source seeds the join; subsequent
    sources are outer-joined onto the accumulator.

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
        df = _read_source(label, path, schema)
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
            "on tensor (+ layer when present) into a single tidy "
            "parquet table. The output is the analytical force "
            "multiplier for cross-pipeline queries (DuckDB, polars, "
            "spreadsheet review)."
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
        # The per-source column count is the number of columns
        # the source contributed to the rollup; we re-read the
        # source's schema to count its columns, but the rollup
        # has them prefixed with the source label.
        prefix = f"{label}."
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
    for flag, schema in SCHEMA_BY_FLAG.items():
        for label, path in getattr(args, flag):
            sources.append((label, path, schema))
    if not sources:
        sys.stderr.write(
            "calibration_rollup: no source files provided; pass at "
            "least one of --l3-outlier / --l4-probe / --l5-plan / "
            "--per-layer-error.\n")
        return 2
    rollup = _rollup(sources)
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
