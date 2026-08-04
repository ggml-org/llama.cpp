#!/usr/bin/env python3
"""Normalize alphaevolve run artifacts into tidy frames for analysis.

Reads every `.zcode/alphaevolve/*/gene-ledger.json` and the cross-run
`findings.jsonl`, flattens the heterogeneous champion_scores dicts into
tidy long format, and emits a per-wave summary table or CSV/JSON dumps.

Schema drift handled: waves name the same concept differently (peak_RSS_*
vs q4_0_peak_rss_B vs delta_RSS_pct). Regex buckets each metric key into
a family so cross-wave comparison is possible despite the naming.

Prereq: polars  (`pip install --user --break-system-packages polars`).

Usage:
  python3 scripts/alphaevolve-metrics.py                          # stdout table
  python3 scripts/alphaevolve-metrics.py --format csv --out DIR   # writes DIR/{metrics_long.csv,findings.csv}
  python3 scripts/alphaevolve-metrics.py --format json            # structured stdout
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import polars as pl
except ImportError:
    sys.stderr.write(
        "alphaevolve-metrics.py needs polars.\n"
        "  pip install --user --break-system-packages polars\n"
    )
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / ".zcode" / "alphaevolve"
FINDINGS = RUNS_DIR / "findings.jsonl"


# Metric key -> family. Same concept, different names across waves.
# Order matters: first match wins, so specific patterns precede generic.
METRIC_FAMILIES = [
    ("rss",          re.compile(r"rss|peak_rss|RSS", re.I)),
    ("tps",          re.compile(r"tps|t/s|tg\d|pp\d", re.I)),
    ("correctness",  re.compile(r"correctness|logit|token_match|ctest", re.I)),
    ("pct",          re.compile(r"pct|_pct_|percent", re.I)),
    ("bug",          re.compile(r"bug|broken|fallback", re.I)),
]


def metric_family(key: str) -> str:
    for fam, pat in METRIC_FAMILIES:
        if pat.search(key):
            return fam
    return "other"


def load_ledgers() -> pl.DataFrame:
    """Flatten every wave's gene-ledger.json champion_scores into long format."""
    rows = []
    for ledger_path in sorted(RUNS_DIR.glob("*/gene-ledger.json")):
        try:
            d = json.loads(ledger_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            sys.stderr.write(f"  skip {ledger_path}: {e}\n")
            continue
        run = d.get("run", ledger_path.parent.name)
        for gene in d.get("genes", []):
            if not isinstance(gene, dict):
                continue
            gid = gene.get("gene_id", "?")
            status = gene.get("status", "?")
            gen = gene.get("generation")
            updated = gene.get("last_update", "")
            for key, val in (gene.get("champion_scores") or {}).items():
                if not isinstance(val, (int, float)):
                    continue
                rows.append({
                    "run": run,
                    "gene_id": gid,
                    "status": status,
                    "metric_family": metric_family(key),
                    "metric_key": key,
                    "metric_value": float(val),
                    "generation": gen,
                    "last_update": updated,
                })
    if not rows:
        return pl.DataFrame(schema={
            "run": pl.Utf8, "gene_id": pl.Utf8, "status": pl.Utf8,
            "metric_family": pl.Utf8, "metric_key": pl.Utf8,
            "metric_value": pl.Float64, "generation": pl.Int64,
            "last_update": pl.Utf8,
        })
    return pl.DataFrame(rows)


def load_findings() -> pl.DataFrame:
    """Read findings.jsonl, skipping the header marker row."""
    rows = []
    if not FINDINGS.exists():
        return pl.DataFrame()
    for line in FINDINGS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("_") == "header":
            continue
        rows.append({
            "ts": d.get("ts", ""),
            "run": d.get("run", ""),
            "category": d.get("category", ""),
            "severity": d.get("severity", ""),
            "status": d.get("status", ""),
            "summary": d.get("summary", ""),
            "source": d.get("source", ""),
            "ref": d.get("ref", ""),
        })
    if not rows:
        return pl.DataFrame(schema={"ts": pl.Utf8, "run": pl.Utf8, "category": pl.Utf8,
                                    "severity": pl.Utf8, "status": pl.Utf8, "summary": pl.Utf8,
                                    "source": pl.Utf8, "ref": pl.Utf8})
    return pl.DataFrame(rows)


def print_table(metrics: pl.DataFrame, findings: pl.DataFrame) -> None:
    """Human-readable per-wave summary to stdout."""
    runs = sorted(metrics["run"].unique().to_list())
    print(f"alphaevolve metrics: {len(runs)} runs, {metrics.height} metric rows, "
          f"{findings.height} findings\n")

    # Findings-by-severity pivot per run
    if findings.height:
        fcounts = (findings.group_by(["run", "severity"])
                          .len().rename({"len": "n"})
                          .pivot("severity", values="n",
                                 aggregate_function="sum")
                          .fill_null(0))
    else:
        fcounts = pl.DataFrame()

    for run in runs:
        sub = metrics.filter(pl.col("run") == run)
        if sub.height == 0:
            continue
        genes = sub["gene_id"].unique().to_list()
        statuses = sub["status"].unique().to_list()
        fams = sorted(sub["metric_family"].unique().to_list())
        print(f"== {run} ==")
        print(f"  genes: {', '.join(genes)}  status: {', '.join(statuses)}")
        print(f"  metric families: {', '.join(fams)} ({sub.height} values)")

        # Show the headline metric (RSS or pct deltas are usually the story)
        headline = sub.filter(
            pl.col("metric_key").str.contains("delta|vs_baseline|paged_vs_flash", literal=False)
        )
        if headline.height:
            print("  headline deltas:")
            for row in headline.iter_rows(named=True):
                print(f"    {row['metric_key']:<55} {row['metric_value']:>14.4f}")

        if fcounts.height:
            row = fcounts.filter(pl.col("run") == run)
            if row.height:
                cols = [c for c in row.columns if c != "run"]
                parts = [f"{c}={int(row[c][0])}" for c in cols if row[c][0]]
                if parts:
                    print(f"  findings: {', '.join(parts)}")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", choices=["table", "csv", "json"], default="table",
                    help="output format (default: table to stdout)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir for --format csv (writes metrics_long.csv + findings.csv)")
    args = ap.parse_args()

    if not RUNS_DIR.exists():
        sys.stderr.write(f"no alphaevolve runs dir at {RUNS_DIR}\n")
        return 1

    metrics = load_ledgers()
    findings = load_findings()

    if args.format == "table":
        print_table(metrics, findings)
    elif args.format == "csv":
        if not args.out:
            sys.stderr.write("--out DIR is required for --format csv\n")
            return 1
        args.out.mkdir(parents=True, exist_ok=True)
        mpath = args.out / "metrics_long.csv"
        fpath = args.out / "findings.csv"
        metrics.write_csv(str(mpath))
        findings.write_csv(str(fpath))
        print(f"wrote {mpath} ({metrics.height} rows)")
        print(f"wrote {fpath} ({findings.height} rows)")
    elif args.format == "json":
        out = {
            "metrics": metrics.to_dicts(),
            "findings": findings.to_dicts(),
        }
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
