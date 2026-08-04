#!/usr/bin/env python3
"""Per-tensor L3 outlier report for the Tessera calibration pipeline.

Reads Tessera Layer 1 dequant sidecar files (``.dequant.f32``) and
produces a per-tensor outlier report. The sidecar format is described
in ``common/tessera-debug.h``: v1 (header 28 bytes, then F32 data),
v2 (header 40 bytes with ``outlier_threshold`` and
``outlier_count_total``, then a per-row int32 strip, then F32 data),
and v3 (the v2 layout plus a per-row 24-byte meta strip). The reader
used here is the canonical Python reader ``l3_sidecar_v3_reader.py``
which dispatches on the file version field; this tool only adds the
cross-sidecar aggregation, the console rendering, and the optional
PNG scatter plots.

The per-tile outlier count is the L3 signal that closes the loop with
the LLM.int8() finding: 0.1% of channels in transformer attention/FFN
weights exceed ~6.0 in absolute value, and those channels dominate
the quantization loss landscape. Tessera records the count per row
(a "tile" in the Tile640 sense) so the L3 metric and the L5
IterQuant orchestrator can use the per-row breakdown to identify
sensitive rows that need special handling.

Output: one NDJSON line per (sidecar_label, tensor) pair, conformant
to ``common/schemas/l3_outlier.schema.json``. Cross-sidecar rollups
(``df.pivot("tensor", "sidecar_label", "outlier_fraction")``) are
the consumer's job; the producer emits one record per
(sidecar_label, tensor) so the consumer sees every input.

Also produces:
  * A console report (default; --quiet to suppress) with the
    per-tensor outlier fractions and a ceiling-breach warning.
  * An optional per-row outlier scatter plot (PNG, no matplotlib
    dependency) for the top-N tensors of each sidecar.

Usage::

    python3 tools/tessera/l3_outlier_report.py \\
        --sidecar-dir tile640:/path/to/tile640_dequant \\
        --sidecar-dir q4k:/path/to/q4k_dequant \\
        --out           report.ndjson \\
        --plot-dir      plots/

The ``--sidecar-dir`` argument takes ``label:path`` so the report
can distinguish Tile640, Q4_K, and Q6_K dumps when comparing quant
types on the same model.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import subprocess
import sys
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import l3_sidecar_v3_reader as reader  # noqa: E402
from _analytical_io import polars_schema as _schema_polars_types  # noqa: E402

SIDECAR_SUFFIX = ".dequant.f32"

# Outlier ceiling for the smoke test and the report's per-tensor sanity
# check. The LLM.int8() paper reports ~0.1% of channels > 6.0 in
# transformer attention/FFN weights; 5% is a generous safety margin
# (50x) that catches catastrophic outliers (e.g. an uninitialized
# scratch buffer) without false-positives on healthy tensors. See
# tools/tessera/l3_outlier_smoke.py for the gating logic.
DEFAULT_OUTLIER_CEILING = 0.05

# Default |x| > threshold cutoff. Matches the LLM.int8() precedent.
DEFAULT_THRESHOLD = 6.0

# Layer-name extraction. Mirrors per_layer_error_table.py so a
# cross-pipeline rollup on `layer` works without normalization.
_BLK_RE = re.compile(r"^(blk\.\d+)\.")


def _layer_from_tensor(tensor_name: str) -> str:
    """Extract the canonical ``blk.<N>`` layer name from a tensor
    name, falling back to the full tensor name for tensors that
    do not match the pattern (embeddings, output, norms)."""
    base = tensor_name
    for suf in (".weight", ".bias"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    m = _BLK_RE.match(base + ".")
    if m is not None:
        return m.group(1)
    return base


def _tessera_provenance() -> tuple[str, str, str]:
    """Return (kernel_version, created_at, tessera_main_tip) by
    shelling out to git. Mirrors the same convention in
    per_layer_error_table.py so the provenance fields line up
    across the four analytical outputs."""
    kernel_version = "unknown"
    main_tip = "unknown"
    try:
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
        pass
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return kernel_version, created_at, main_tip


# ---------------------------------------------------------------------------
# Sidecar aggregation
# ---------------------------------------------------------------------------


@dataclass
class TensorOutlier:
    """One per-tensor row destined for the NDJSON writer.

    The ``per_row_counts`` field is kept separately for the PNG
    plot path; the NDJSON schema does not include per-row
    breakdowns (consumers can re-read the sidecar to recover them
    if needed; the L3 signal in the schema is the aggregate
    count and fraction, which is what the L5 orchestrator and
    calibration_rollup actually consume).
    """

    name: str
    layer: str
    rows: int
    cols: int
    total_elements: int
    outlier_count: int
    outlier_fraction: float
    threshold: float
    skipped: bool = False
    skip_reason: str = ""
    # Provenance: which reader produced the count. "dequant" for
    # the sidecar-based reader (default, accurate); "tensor_stats_estimate"
    # for the --tessera-db fast path (approximate, derived from
    # tail_ratio). Consumers can filter on this to require an
    # accurate count.
    source: str = "dequant"
    per_row_counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32), repr=False)


@dataclass
class SidecarGroup:
    """All tensors found in a single sidecar directory."""

    label: str
    path: Path
    tensors: list[TensorOutlier] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total_elements(self) -> int:
        return sum(t.total_elements for t in self.tensors if not t.skipped)

    @property
    def total_outliers(self) -> int:
        return sum(t.outlier_count for t in self.tensors if not t.skipped)

    @property
    def aggregate_fraction(self) -> float:
        te = self.total_elements
        if te == 0:
            return 0.0
        return float(self.total_outliers) / float(te)


def parse_labeled_dir(arg: str) -> "LabeledDir":
    if ":" not in arg:
        raise argparse.ArgumentTypeError(
            f"--sidecar-dir expects LABEL:PATH, got {arg!r}")
    label, path = arg.split(":", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError(
            f"--sidecar-dir expects LABEL:PATH with both non-empty, got {arg!r}")
    return LabeledDir(label=label, path=Path(path))


@dataclass
class LabeledDir:
    label: str
    path: Path


def iter_sidecar_paths(d: Path) -> Iterable[Path]:
    """Yield all ``.dequant.f32`` files under ``d`` (sorted by name)."""
    if not d.is_dir():
        return
    for p in sorted(d.glob(f"*{SIDECAR_SUFFIX}")):
        if p.is_file():
            yield p


def build_group(labeled: LabeledDir, default_threshold: float) -> SidecarGroup:
    """Walk a sidecar directory and produce a per-tensor aggregation.

    Uses the canonical ``l3_sidecar_v3_reader`` to dispatch on the
    file version (v1 / v2 / v3). Per-row counts are kept on the
    TensorOutlier for the PNG plot path.
    """
    g = SidecarGroup(label=labeled.label, path=labeled.path)
    for p in iter_sidecar_paths(labeled.path):
        try:
            sc = reader.read_sidecar(p, mode="auto", provenance=False)
        except (reader.SidecarReadError, ValueError, OSError) as e:
            g.skipped.append((p.name, str(e)))
            continue

        rows = int(sc["rows"])
        cols = int(sc["cols"])
        total = int(rows * cols)
        threshold = float(sc.get("threshold", 0.0))
        per_row = np.asarray(sc.get("row_outlier_count", []),
                             dtype=np.int32)
        count = int(sc.get("outlier_count_total", -1))

        # v1 files have negative outlier_count_total; recompute
        # from the F32 data with the default threshold. The
        # canonical reader does this internally, but only the
        # default cutoff (it does not consult default_threshold
        # here). To keep the report self-consistent we re-do
        # the count when negative, with the user's --threshold
        # overriding the default. v2/v3 files have the count
        # already; trust it.
        if count < 0:
            data = sc["data"]
            threshold = default_threshold
            mask = np.abs(data) > threshold
            count = int(mask.sum())
            per_row = mask.sum(axis=1).astype(np.int32)
        if threshold <= 0.0:
            threshold = default_threshold

        fraction = float(count) / float(total) if total else 0.0
        name = p.stem
        # ``p.stem`` is the filename minus its last suffix. The
        # sidecar filenames end in ``.f32`` (so stem is
        # ``<tensor>.dequant``) and the canonical reader returns
        # the same; strip the ``.dequant`` segment so the
        # ``tensor`` field is the bare tensor name.
        if name.endswith(".dequant"):
            name = name[: -len(".dequant")]
        g.tensors.append(TensorOutlier(
            name=name,
            layer=_layer_from_tensor(name),
            rows=rows,
            cols=cols,
            total_elements=total,
            outlier_count=count,
            outlier_fraction=fraction,
            threshold=threshold,
            source="dequant",
            per_row_counts=per_row,
        ))
    g.tensors.sort(key=lambda t: t.outlier_fraction, reverse=True)
    return g


# ---------------------------------------------------------------------------
# Console rendering
# ---------------------------------------------------------------------------


def render_console(groups: list[SidecarGroup], top_k: int, ceiling: float) -> str:
    out: list[str] = []
    out.append("Tessera L3 outlier report")
    out.append("=" * 72)
    for g in groups:
        out.append("")
        out.append(f"[{g.label}]  {g.path}  ({len(g.tensors)} tensors, "
                   f"{len(g.skipped)} skipped)")
        out.append("-" * 72)
        if not g.tensors:
            out.append("  (no sidecar files found)")
            continue
        out.append(
            f"  {'tensor':<48} {'rows':>8} {'cols':>8} "
            f"{'outliers':>12} {'fraction':>10} {'threshold':>10}"
        )
        for t in g.tensors[:top_k]:
            mark = "  " if t.outlier_fraction <= ceiling else " *"
            out.append(
                f"{mark}{t.name:<46} {t.rows:>8d} {t.cols:>8d} "
                f"{t.outlier_count:>12d} {t.outlier_fraction:>10.4%} "
                f"{t.threshold:>10.3f}"
            )
        if len(g.tensors) > top_k:
            out.append(f"  ... ({len(g.tensors) - top_k} more tensors omitted)")
        out.append(
            f"  AGGREGATE  total_elements={g.total_elements:>14d}  "
            f"total_outliers={g.total_outliers:>12d}  "
            f"fraction={g.aggregate_fraction:>10.4%}"
        )
        ceiling_breach = [t for t in g.tensors if t.outlier_fraction > ceiling]
        if ceiling_breach:
            out.append(
                f"  WARNING: {len(ceiling_breach)} tensor(s) exceed the "
                f"{ceiling:.1%} outlier ceiling: "
                + ", ".join(t.name for t in ceiling_breach[:5])
                + (" ..." if len(ceiling_breach) > 5 else "")
            )
        if g.skipped:
            out.append("  SKIPPED:")
            for name, reason in g.skipped:
                out.append(f"    {name}: {reason}")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# NDJSON writer (per-tensor records, typed per the L3 schema)
# ---------------------------------------------------------------------------


def write_ndjson(groups: list[SidecarGroup],
                 out_path: Path,
                 sidecar_dirs: dict[str, str],
                 provenance: tuple[str, str, str],
                 ceiling: float) -> int:
    """Emit one NDJSON line per (sidecar_label, tensor) row.

    Carries the per-tensor fields plus the sidecar's source dir
    and the provenance. The schema's polars_schema is the dtype
    override; consumer reads via ``_analytical_io.read_analytical``.
    Cross-sidecar rollups (the canonical "where is the Tile640 vs
    Q4_K outlier rate divergent" table) are the consumer's job:
    a one-liner ``df.pivot("tensor", "sidecar_label",
    "outlier_fraction")``.

    Returns the number of records written.
    """
    kernel_version, created_at, main_tip = provenance
    rows: list[dict] = []
    for g in groups:
        sidecar_dir = sidecar_dirs.get(g.label, str(g.path))
        for t in g.tensors:
            rows.append({
                "sidecar_label":   g.label,
                "tensor":          t.name,
                "layer":           t.layer,
                "rows":            t.rows,
                "cols":            t.cols,
                "total_elements":  t.total_elements,
                "outlier_count":   t.outlier_count,
                "outlier_fraction": t.outlier_fraction,
                "threshold":       t.threshold,
                "skipped":         t.skipped,
                "skip_reason":     t.skip_reason,
                "source":          getattr(t, "source", "dequant"),
                "kernel_version":  kernel_version,
                "created_at":      created_at,
                "tessera_main_tip": main_tip,
            })
        for fname, reason in g.skipped:
            # Emit a placeholder row for each skipped file so
            # the consumer can audit the failure (the row's
            # counts are zero and skip_reason is set).
            rows.append({
                "sidecar_label":   g.label,
                "tensor":          fname,
                "layer":           _layer_from_tensor(fname),
                "rows":            0,
                "cols":            0,
                "total_elements":  0,
                "outlier_count":   0,
                "outlier_fraction": 0.0,
                "threshold":       0.0,
                "skipped":         True,
                "skip_reason":     reason,
                "source":          "dequant",
                "kernel_version":  kernel_version,
                "created_at":      created_at,
                "tessera_main_tip": main_tip,
            })
    # Pin dtypes per the schema. An empty input (no sidecar
    # files found in any dir) still emits a zero-row frame with
    # the schema's column order so downstream readers have the
    # expected columns.
    schema_types = _schema_polars_types("l3_outlier")
    if rows:
        df = pl.DataFrame(rows, infer_schema_length=max(len(rows), 1))
    else:
        df = pl.DataFrame(
            {col: pl.Series(name=col, values=[], dtype=dtype)
             for col, dtype in schema_types.items()},
            schema=schema_types,
        )
    for col, dtype in schema_types.items():
        if col in df.columns and df.schema[col] != dtype:
            df = df.with_columns(pl.col(col).cast(dtype, strict=False))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_ndjson(out_path)
    return len(rows)


# ---------------------------------------------------------------------------
# PNG plotting (no matplotlib; per-row outlier scatter)
# ---------------------------------------------------------------------------


def render_plots(groups: list[SidecarGroup], plot_dir: Path, top_n: int) -> list[Path]:
    """Render a per-row outlier scatter for the top-N tensors per group.

    Returns the list of plot paths actually written. The plot is a
    simple PNG (no matplotlib dependency) drawn from scratch with
    the Python stdlib so the tool stays numpy-only.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for g in groups:
        for t in g.tensors[:top_n]:
            png_path = plot_dir / f"{g.label}__{t.name}.png"
            _write_scatter_png(t, png_path)
            written.append(png_path)
    return written


def _write_scatter_png(t: TensorOutlier, png_path: Path) -> None:
    """Write a minimal PNG of per-row outlier counts (no matplotlib)."""
    counts = t.per_row_counts
    if counts.size == 0:
        return
    width, height = 640, 240
    margin_l, margin_r, margin_t, margin_b = 40, 12, 16, 28
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_count = int(counts.max()) if counts.size else 0
    if max_count <= 0:
        max_count = 1

    pixels = bytearray([255] * (width * height * 3))
    for x in range(margin_l, width - margin_r):
        _set_pixel(pixels, width, height, x, height - margin_b, 200, 200, 200)
        _set_pixel(pixels, width, height, x, margin_t, 200, 200, 200)
    for y in range(margin_t, height - margin_b):
        _set_pixel(pixels, width, height, margin_l, y, 200, 200, 200)
        _set_pixel(pixels, width, height, width - margin_r - 1, y, 200, 200, 200)

    n = counts.size
    for i, c in enumerate(counts):
        if c <= 0:
            continue
        x = margin_l + int(round(i / max(n - 1, 1) * (plot_w - 1)))
        y_top = margin_t + int(round((1.0 - c / max_count) * (plot_h - 1)))
        y_bot = height - margin_b - 1
        for y in range(y_top, y_bot + 1):
            _set_pixel(pixels, width, height, x, y, 60, 110, 200)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                xx, yy = x + dx, y_top + dy
                if 0 <= xx < width and 0 <= yy < height:
                    _set_pixel(pixels, width, height, xx, yy, 30, 60, 160)

    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y * stride:(y + 1) * stride])
    compressed = zlib.compress(bytes(raw), 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    png_path.write_bytes(png)


def _set_pixel(buf: bytearray, width: int, height: int, x: int, y: int,
               r: int, g: int, b: int) -> None:
    if 0 <= x < width and 0 <= y < height:
        idx = (y * width + x) * 3
        buf[idx] = r
        buf[idx + 1] = g
        buf[idx + 2] = b


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Tessera L3 outlier report. Reads one or more Layer-1 dequant "
            "sidecar directories and prints / writes per-tensor outlier "
            "counts (|x| > threshold). Use --sidecar-dir LABEL:PATH to "
            "compare Tile640 / Q4_K / Q6_K dumps side-by-side."
        )
    )
    p.add_argument(
        "--sidecar-dir",
        action="append",
        type=parse_labeled_dir,
        required=True,
        help=(
            "A sidecar directory to scan, as LABEL:PATH. May be repeated. "
            "LABEL is the human-readable name in the report "
            "(e.g. 'tile640', 'q4k', 'q6k')."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output NDJSON path (one record per (sidecar_label, tensor)).",
    )
    p.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="If set, write per-row outlier scatter plots (PNG) for the "
             "top --plot-top-n tensors of each sidecar directory.",
    )
    p.add_argument(
        "--plot-top-n",
        type=int,
        default=5,
        help="Number of top tensors per sidecar to plot (default 5).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Console: how many tensors to print per sidecar (default 20).",
    )
    p.add_argument(
        "--ceiling",
        type=float,
        default=DEFAULT_OUTLIER_CEILING,
        help=(
            "Outlier fraction ceiling for the per-tensor sanity check "
            f"(default {DEFAULT_OUTLIER_CEILING * 100:.2f}%%). "
            "Tensors above the ceiling are flagged with '*' in the "
            "console report and the CLI exits non-zero so the "
            "smoke test can gate on it."
        ),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            f"Override the |x| > threshold cutoff used for the v1 fallback "
            f"and the report's sanity bands (default {DEFAULT_THRESHOLD}). "
            f"v2 / v3 sidecar files already record the threshold they were "
            f"written with; this flag does not retroactively change them."
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the console report (still writes --out / --plot-dir).",
    )
    p.add_argument(
        "--tessera-db",
        type=Path,
        default=None,
        help=(
            "Path to the unified tessera.duckdb. When set, the report "
            "uses the per-tensor rms / tail_ratio / kurtosis from the "
            "tensor_stats table as a fast-path approximation of the "
            "outlier count, instead of reading the dequant sidecar "
            "files. The output NDJSON rows are tagged with "
            "source='tensor_stats_estimate' so the consumer knows "
            "the count is approximate. The default (no --tessera-db) "
            "reads the dequant sidecar for the exact count "
            "(source='dequant'). The fast path is useful when the "
            "sidecar files are unavailable (e.g. calibration was run "
            "on a different machine) or when you only need a rough "
            "estimate to gate the heavy path."
        ),
    )
    p.add_argument(
        "--model-hash",
        default=None,
        help=(
            "Model hash to filter the tensor_stats query (only used "
            "with --tessera-db). Default: any model. Set this to "
            "the same model_hash the C++ dispatch wrote if you "
            "want a specific model's stats."
        ),
    )
    return p


def build_tensor_stats_group(
    label: str,
    db_path: Path,
    *,
    threshold: float,
    model_hash: str | None = None,
) -> SidecarGroup:
    """Fast-path SidecarGroup built from the tensor_stats table.

    Reads ``SELECT name, family, layer_depth, n_elements, rms,
    mean_abs, tail_ratio, kurtosis FROM tensor_stats`` (filtered by
    model_hash when given) and produces a per-tensor approximation
    of the outlier count. The approximation is conservative: when
    the tensor's tail_ratio is below the threshold * 1.0, the count
    is 0; when above, the count is 1 (just the max). The
    outlier_fraction is the count / n_elements when n_elements >
    0, else 0. The provenance ``source`` is set to
    ``tensor_stats_estimate`` so the consumer knows the value is
    approximate.

    This is the fast path the unified-DB design described: read
    the per-tensor summary instead of the per-tensor dequant
    sidecar, at the cost of an approximate count. The dequant
    sidecar path remains the default and produces the exact count.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tessera_db import TesseraDB
    db_path = Path(db_path) if not isinstance(db_path, Path) else db_path
    g = SidecarGroup(label=label, path=str(db_path))
    if not db_path.is_file():
        return g
    with TesseraDB.open(str(db_path), read_only=True) as db:
        where = ""
        if model_hash:
            where = f" WHERE model_hash = '{model_hash}'"
        df = db.query(
            f"SELECT name, family, layer_depth, n_elements, rms, "
            f"mean_abs, tail_ratio, kurtosis FROM tensor_stats{where}"
        )
    for row in df.to_dicts():
        n_elements = int(row.get("n_elements") or 0)
        rms = float(row.get("rms") or 0.0)
        tail_ratio = float(row.get("tail_ratio") or 0.0)
        layer_depth = int(row.get("layer_depth") or 0)
        # The fast-path outlier count: 1 if tail_ratio > threshold,
        # else 0. This is a conservative "is there at least one
        # likely outlier" signal. The dequant sidecar is the
        # accurate count.
        count = 1 if tail_ratio > threshold else 0
        fraction = (count / n_elements) if n_elements > 0 else 0.0
        tensor = TensorOutlier(
            name=row.get("name", ""),
            layer=f"blk.{layer_depth}",
            rows=0,             # unknown without dequant read
            cols=0,
            total_elements=n_elements,
            outlier_count=count,
            outlier_fraction=fraction,
            threshold=threshold,
            skipped=False,
            skip_reason="",
            source="tensor_stats_estimate",
        )
        g.tensors.append(tensor)
    return g


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    groups: list[SidecarGroup] = []
    sidecar_dirs: dict[str, str] = {}
    if args.tessera_db is not None:
        # Fast path: read tensor_stats, skip the dequant sidecar.
        # Each --sidecar-dir entry becomes a labeled group sourced
        # from the same DB (the LABEL is reused so the output is
        # shaped the same as the sidecar-based run). When --sidecar-dir
        # is not given, the fast path is still useful: a single
        # unlabeled group with all the tensor_stats rows.
        if args.sidecar_dir:
            for labeled in args.sidecar_dir:
                groups.append(build_tensor_stats_group(
                    labeled.label, args.tessera_db,
                    threshold=args.threshold,
                    model_hash=args.model_hash,
                ))
                sidecar_dirs[labeled.label] = str(args.tessera_db)
        else:
            groups.append(build_tensor_stats_group(
                "tensor_stats", args.tessera_db,
                threshold=args.threshold,
                model_hash=args.model_hash,
            ))
            sidecar_dirs["tensor_stats"] = str(args.tessera_db)
        sys.stderr.write(
            f"l3_outlier: --tessera-db fast path: {sum(g.total_elements for g in groups)} "
            f"elements across {sum(len(g.tensors) for g in groups)} tensors\n"
        )
    else:
        for labeled in args.sidecar_dir:
            groups.append(build_group(labeled, default_threshold=args.threshold))
            sidecar_dirs[labeled.label] = str(labeled.path)

    if not args.quiet:
        print(render_console(groups, top_k=args.top_k, ceiling=args.ceiling))

    provenance = _tessera_provenance()
    n = write_ndjson(groups, args.out, sidecar_dirs, provenance,
                     ceiling=args.ceiling)
    sys.stderr.write(f"wrote {n} per-tensor records to {args.out}\n")

    if args.plot_dir is not None:
        written = render_plots(groups, args.plot_dir, top_n=args.plot_top_n)
        if written:
            sys.stderr.write(
                f"wrote {len(written)} plot(s) to {args.plot_dir}\n")

    # Exit non-zero if any group has a tensor above the ceiling, unless
    # the user explicitly opts out by passing --ceiling 1.0. The smoke
    # test relies on this to gate.
    if args.ceiling < 1.0:
        for g in groups:
            for t in g.tensors:
                if t.outlier_fraction > args.ceiling:
                    return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
