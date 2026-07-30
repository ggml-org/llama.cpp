#!/usr/bin/env python3
"""Plot OpenVINO phase-tune CSVs (two devices, prefill + decode phases)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("error: matplotlib required (pip install matplotlib)", file=sys.stderr)
    sys.exit(1)


def load_csv(path: Path) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(int(row["token_index"]))
            ys.append(float(row["avg_ms"]))
    return xs, ys


def find_file(directory: Path, phase: str, device_slug: str) -> Path | None:
    direct = directory / f"{phase}_{device_slug}.csv"
    if direct.is_file():
        return direct
    matches = sorted(directory.glob(f"{phase}_*.csv"))
    for p in matches:
        if device_slug in p.stem:
            return p
    return None


def device_to_slug(device: str) -> str:
    return device.replace(".", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot phase-tune token latency CSVs")
    parser.add_argument("output_dir", type=Path, help="Directory with pp_*.csv and tg_*.csv")
    parser.add_argument("--device0", default="CPU", help="First tune device (matches CSV suffix)")
    parser.add_argument("--device1", default="GPU.0", help="Second tune device")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory for PNG outputs (default: output_dir)",
    )
    args = parser.parse_args()

    out_dir = args.out or args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    d0 = device_to_slug(args.device0)
    d1 = device_to_slug(args.device1)

    for phase, title in (("pp", "Prefill"), ("tg", "Token generation")):
        p0 = find_file(args.output_dir, phase, d0)
        p1 = find_file(args.output_dir, phase, d1)
        if not p0 or not p1:
            print(f"warning: missing {phase} CSVs in {args.output_dir}", file=sys.stderr)
            continue

        x0, y0 = load_csv(p0)
        x1, y1 = load_csv(p1)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(x0, y0, s=12, alpha=0.75, label=args.device0)
        ax.scatter(x1, y1, s=12, alpha=0.75, label=args.device1)
        ax.set_xlabel("Token index (within phase)")
        ax.set_ylabel("Average infer time (ms)")
        ax.set_title(f"{title}: per-token latency by device")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        png = out_dir / f"phase_tune_{phase}.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"wrote {png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
