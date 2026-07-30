#!/usr/bin/env python3
"""Per-tensor L3 outlier report for the Tessera calibration pipeline.

Reads Tessera Layer 1 dequant sidecar files (``.dequant.f32``) and produces
a per-tensor outlier report. The sidecar format is described in
``common/tessera-debug.h``: v1 (header 28 bytes, then F32 data) and v2
(header 40 bytes with ``outlier_threshold`` and ``outlier_count_total``,
then a per-row int32 strip, then F32 data). This reader accepts both
versions.

The per-tile outlier count is the L3 signal that closes the loop with the
LLM.int8() finding: 0.1% of channels in transformer attention/FFN weights
exceed ~6.0 in absolute value, and those channels dominate the
quantization loss landscape. Tessera records the count per row (a "tile"
in the Tile640 sense) so the L3 metric and the L5 IterQuant orchestrator
can use the per-row breakdown to identify sensitive rows that need
special handling.

Outputs:

* A JSON report with per-tensor (and aggregate) counts for each sidecar
  directory passed in.
* An optional CSV with one row per (tensor, sidecar) for spreadsheet
  review.
* An optional scatter plot of per-row outlier count vs row index, useful
  for spotting whether outliers cluster in the first/last rows of a
  tensor (a known pathology in some attention-projection weights).

Usage::

    python3 tools/tessera/l3_outlier_report.py \\
        --sidecar-dir tile640:/path/to/tile640_dequant \\
        --sidecar-dir q4k:/path/to/q4k_dequant \\
        --sidecar-dir q6k:/path/to/q6k_dequant \\
        --output report.json \\
        --csv report.csv \\
        --plot-dir plots/

The ``--sidecar-dir`` argument takes ``label:path`` so the report can
distinguish Tile640, Q4_K, and Q6_K dumps when comparing quant types on
the same model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


SIDECAR_SUFFIX = ".dequant.f32"
SIDECAR_MAGIC = b"TDQT"
# v1: 28-byte header, F32 data immediately after.
# v2: 40-byte header, per-row int32 strip, then F32 data.
SIDECAR_HEADER_V1 = 28
SIDECAR_HEADER_V2 = 40
DTYPE_F32 = 0
DTYPE_F16 = 1
DTYPE_BF16 = 2

# Outlier ceiling for the smoke test and the report's per-tensor sanity
# check. The LLM.int8() paper reports ~0.1% of channels > 6.0 in
# transformer attention/FFN weights; 5% is a generous safety margin
# (50x) that catches catastrophic outliers (e.g. an uninitialized
# scratch buffer) without false-positives on healthy tensors. See
# tools/tessera/l3_outlier_smoke.py for the gating logic.
DEFAULT_OUTLIER_CEILING = 0.05

# Default |x| > threshold cutoff. Matches the LLM.int8() precedent.
DEFAULT_THRESHOLD = 6.0


@dataclass
class TensorOutlierReport:
    """Per-tensor outlier summary from a single sidecar directory."""

    name: str
    rows: int
    cols: int
    total_elements: int
    outlier_count: int
    outlier_fraction: float
    threshold: float
    # Per-row counts, kept for the optional scatter plot.
    per_row_counts: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "rows": int(self.rows),
            "cols": int(self.cols),
            "total_elements": int(self.total_elements),
            "outlier_count": int(self.outlier_count),
            "outlier_fraction": float(self.outlier_fraction),
            "threshold": float(self.threshold),
        }


# ---------------------------------------------------------------------------
# Sidecar reader (v1 + v2)
# ---------------------------------------------------------------------------


@dataclass
class Sidecar:
    """Parsed L1 dequant sidecar with v1/v2 backward compatibility."""

    path: Path
    version: int
    rows: int
    cols: int
    dtype: int
    threshold: float
    outlier_count_total: int
    per_row_counts: np.ndarray
    data_f32: np.ndarray

    @property
    def total_elements(self) -> int:
        return int(self.rows) * int(self.cols)

    @property
    def outlier_fraction(self) -> float:
        if self.total_elements == 0:
            return 0.0
        return float(self.outlier_count_total) / float(self.total_elements)

    def tensor_report(self) -> TensorOutlierReport:
        return TensorOutlierReport(
            name=self.path.stem.replace(".dequant", "").replace(".f32", ""),
            rows=self.rows,
            cols=self.cols,
            total_elements=self.total_elements,
            outlier_count=int(self.outlier_count_total),
            outlier_fraction=self.outlier_fraction,
            threshold=self.threshold,
            per_row_counts=self.per_row_counts,
        )


def read_sidecar(path: Path) -> Sidecar:
    """Parse a ``.dequant.f32`` sidecar. Accepts v1 and v2.

    v1: 28-byte header (magic[4], version[4], rows[8], cols[8], dtype[4])
        then F32 data starting at offset 28.
    v2: 40-byte header (v1 fields + threshold[4] + total[8]) then a
        per-row int32 strip of length rows*4, then F32 data.
    """
    with open(path, "rb") as f:
        header = f.read(SIDECAR_HEADER_V2)

    if len(header) < SIDECAR_HEADER_V1:
        raise ValueError(f"{path}: sidecar too short ({len(header)} bytes)")

    if header[:4] != SIDECAR_MAGIC:
        raise ValueError(
            f"{path}: bad magic {header[:4]!r} (expected {SIDECAR_MAGIC!r})"
        )

    version = int(np.frombuffer(header[4:8], dtype="<u4")[0])
    rows = int(np.frombuffer(header[8:16], dtype="<i8")[0])
    cols = int(np.frombuffer(header[16:24], dtype="<i8")[0])
    dtype = int(np.frombuffer(header[24:28], dtype="<u4")[0])

    if dtype != DTYPE_F32:
        raise ValueError(
            f"{path}: only DTYPE_F32 ({DTYPE_F32}) is supported, got dtype={dtype}"
        )

    if version == 1:
        threshold = float("nan")  # v1 does not record the threshold
        total = -1  # sentinel: will be recomputed from the F32 data
        per_row_counts = np.zeros(rows, dtype=np.int32)
    elif version == 2:
        threshold = float(np.frombuffer(header[28:32], dtype="<f4")[0])
        total = int(np.frombuffer(header[32:40], dtype="<i8")[0])
        with open(path, "rb") as f:
            f.seek(SIDECAR_HEADER_V2)
            strip_bytes = f.read(rows * 4)
        if len(strip_bytes) < rows * 4:
            raise ValueError(
                f"{path}: per-row strip truncated (expected {rows*4} bytes, "
                f"got {len(strip_bytes)})"
            )
        per_row_counts = np.frombuffer(strip_bytes, dtype="<i4").astype(np.int32)
    else:
        raise ValueError(f"{path}: unsupported sidecar version {version}")

    # F32 data block. v1: starts at offset 28. v2: starts at offset
    # 40 + rows*4. Recompute the total from the data if v1 or the v2
    # total looks uninitialized (negative).
    if version == 1:
        data_off = SIDECAR_HEADER_V1
    else:
        data_off = SIDECAR_HEADER_V2 + rows * 4
    expected_floats = rows * cols
    with open(path, "rb") as f:
        f.seek(data_off)
        data_bytes = f.read(expected_floats * 4)
    if len(data_bytes) < expected_floats * 4:
        raise ValueError(
            f"{path}: F32 data truncated (expected {expected_floats} floats, "
            f"got {len(data_bytes) // 4})"
        )
    data_f32 = np.frombuffer(data_bytes, dtype="<f4").reshape(rows, cols).copy()

    if total < 0:
        # v1: no header total; recompute from the F32 data using the
        # default threshold. v1 files predate the configurable cutoff,
        # so the default is the only signal we have.
        total = int(np.sum(np.abs(data_f32) > DEFAULT_THRESHOLD))
        per_row_counts = np.sum(np.abs(data_f32) > DEFAULT_THRESHOLD, axis=1).astype(np.int32)
        if np.isnan(threshold):
            threshold = DEFAULT_THRESHOLD

    return Sidecar(
        path=path,
        version=version,
        rows=rows,
        cols=cols,
        dtype=dtype,
        threshold=threshold,
        outlier_count_total=total,
        per_row_counts=per_row_counts,
        data_f32=data_f32,
    )


def iter_sidecar_paths(d: Path) -> Iterable[Path]:
    """Yield all ``.dequant.f32`` files under ``d`` (sorted by name)."""
    if not d.is_dir():
        return
    for p in sorted(d.glob(f"*{SIDECAR_SUFFIX}")):
        if p.is_file():
            yield p


# ---------------------------------------------------------------------------
# Aggregation across sidecar directories
# ---------------------------------------------------------------------------


@dataclass
class LabeledDir:
    label: str
    path: Path


def parse_labeled_dir(arg: str) -> LabeledDir:
    if ":" not in arg:
        raise argparse.ArgumentTypeError(
            f"--sidecar-dir expects LABEL:PATH, got {arg!r}"
        )
    label, path = arg.split(":", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError(
            f"--sidecar-dir expects LABEL:PATH with both non-empty, got {arg!r}"
        )
    return LabeledDir(label=label, path=Path(path))


@dataclass
class SidecarGroup:
    """All tensors found in a single sidecar directory."""

    label: str
    path: Path
    tensors: list[TensorOutlierReport] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total_elements(self) -> int:
        return sum(t.total_elements for t in self.tensors)

    @property
    def total_outliers(self) -> int:
        return sum(t.outlier_count for t in self.tensors)

    @property
    def aggregate_fraction(self) -> float:
        if self.total_elements == 0:
            return 0.0
        return float(self.total_outliers) / float(self.total_elements)


def build_group(labeled: LabeledDir) -> SidecarGroup:
    g = SidecarGroup(label=labeled.label, path=labeled.path)
    for p in iter_sidecar_paths(labeled.path):
        try:
            sc = read_sidecar(p)
        except (ValueError, OSError) as e:
            g.skipped.append((p.name, str(e)))
            continue
        g.tensors.append(sc.tensor_report())
    g.tensors.sort(key=lambda t: t.outlier_fraction, reverse=True)
    return g


# ---------------------------------------------------------------------------
# Report rendering
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


def render_json(groups: list[SidecarGroup], ceiling: float) -> dict:
    return {
        "schema": "tessera.l3-outlier-report.v1",
        "ceiling": ceiling,
        "groups": [
            {
                "label": g.label,
                "path": str(g.path),
                "tensor_count": len(g.tensors),
                "skipped": [{"name": n, "reason": r} for n, r in g.skipped],
                "aggregate": {
                    "total_elements": g.total_elements,
                    "total_outliers": g.total_outliers,
                    "outlier_fraction": g.aggregate_fraction,
                },
                "tensors": [t.to_dict() for t in g.tensors],
            }
            for g in groups
        ],
    }


def render_csv(groups: list[SidecarGroup], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sidecar_label", "tensor", "rows", "cols", "total_elements",
            "outlier_count", "outlier_fraction", "threshold",
        ])
        for g in groups:
            for t in g.tensors:
                w.writerow([
                    g.label, t.name, t.rows, t.cols, t.total_elements,
                    t.outlier_count, f"{t.outlier_fraction:.6f}",
                    f"{t.threshold:.4f}",
                ])


def render_plots(groups: list[SidecarGroup], plot_dir: Path, top_n: int) -> list[Path]:
    """Render a per-row outlier scatter for the top-N tensors per group.

    Returns the list of plot paths actually written. The plot is a
    simple PNG (no matplotlib dependency) drawn from scratch with the
    Python stdlib so the tool stays numpy-only. Each row is a stem:
    (row_idx, count) marked on a 1x1 PNG canvas.
    """
    try:
        import zlib  # noqa: F401  (stdlib, always available)
        import struct
    except ImportError:  # pragma: no cover - stdlib always present
        return []
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for g in groups:
        for t in g.tensors[:top_n]:
            png_path = plot_dir / f"{g.label}__{t.name}.png"
            _write_scatter_png(t, png_path)
            written.append(png_path)
    return written


def _write_scatter_png(t: TensorOutlierReport, png_path: Path) -> None:
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

    # White background row-major RGB.
    pixels = bytearray([255] * (width * height * 3))
    # Axes (light grey).
    _hline = lambda y: (_set_pixel(pixels, width, height, x, y, 200, 200, 200)
                       for x in range(margin_l, width - margin_r))
    _vline = lambda x: (_set_pixel(pixels, width, height, x, y, 200, 200, 200)
                       for y in range(margin_t, height - margin_b))
    for _ in _hline(height - margin_b):
        pass
    for _ in _hline(margin_t):
        pass
    for _ in _vline(margin_l):
        pass
    for _ in _vline(width - margin_r - 1):
        pass

    # Stems.
    n = counts.size
    for i, c in enumerate(counts):
        if c <= 0:
            continue
        x = margin_l + int(round(i / max(n - 1, 1) * (plot_w - 1)))
        y_top = margin_t + int(round((1.0 - c / max_count) * (plot_h - 1)))
        y_bot = height - margin_b - 1
        # Stem line
        for y in range(y_top, y_bot + 1):
            _set_pixel(pixels, width, height, x, y, 60, 110, 200)
        # Mark at the top.
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                xx, yy = x + dx, y_top + dy
                if 0 <= xx < width and 0 <= yy < height:
                    _set_pixel(pixels, width, height, xx, yy, 30, 60, 160)

    # PNG encode.
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)  # filter type 0 per row
        raw.extend(pixels[y * stride:(y + 1) * stride])
    import zlib
    import struct
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
        "--output",
        type=Path,
        default=None,
        help="Write the full report as JSON to this path.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write a flat per-tensor CSV (one row per tensor, all sidecars).",
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
            "console report."
        ),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            f"Override the |x| > threshold cutoff used for the v1 fallback "
            f"and the report's sanity bands (default {DEFAULT_THRESHOLD}). "
            f"v2 sidecar files already record the threshold they were "
            f"written with; this flag does not retroactively change them."
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the console report (still writes --output / --csv).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    groups: list[SidecarGroup] = []
    for labeled in args.sidecar_dir:
        groups.append(build_group(labeled))

    if not args.quiet:
        print(render_console(groups, top_k=args.top_k, ceiling=args.ceiling))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = render_json(groups, ceiling=args.ceiling)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)

    if args.csv is not None:
        render_csv(groups, args.csv)
        print(f"wrote {args.csv}", file=sys.stderr)

    if args.plot_dir is not None:
        written = render_plots(groups, args.plot_dir, top_n=args.plot_top_n)
        if written:
            print(f"wrote {len(written)} plot(s) to {args.plot_dir}", file=sys.stderr)

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
