#!/usr/bin/env python3

import sys
import os
import re
import argparse
import statistics
import logging

from collections import defaultdict

# Mapping of cli-friendly names to (internal_data_key, Display Header, numeric_sort_key)
COL_MAP = {
    "op":         ("op",         "Op",         "op"),
    "dims":       ("dims",       "Dims",       "dims"),
    "dtypes":     ("dtypes",     "DTypes",     "dtypes"),
    "count":      ("count",      "Count",      "_sort_count"),
    "max-usec":   ("max_usec",   "Max usec",   "_sort_max_usec"),
    "avg-usec":   ("avg_usec",   "Avg usec",   "_sort_avg_usec"),
    "max-cycles": ("max_cycles", "Max Cycles", "_sort_max_cycles"),
    "avg-cycles": ("avg_cycles", "Avg Cycles", "_sort_avg_cycles"),
    "max-pmu":    ("max_pmu",    "Max PMU",    "_sort_max_pmu"),
    "avg-pmu":    ("avg_pmu",    "Avg PMU",    "_sort_avg_pmu"),
}

op_pattern = re.compile(
    r"profile-op\s+(?P<op_name>[A-Z_0-9]+):\s+.*?\s+:\s+(?P<dims>[\d:x\s\->!]+)\s+:\s+(?P<types>[a-z\d_\s\->x]+)\s+:\s+.*?\s+usec\s+(?P<usec>\d+)\s+cycles\s+(?P<cycles>\d+)(?:\s+pmu\s+\[(?P<pmu>[\d,\s]+)\])?"
)

logger = logging.getLogger("ggml-hexagon-profile")


def parse_log(file_path, pmu_index=None):
    try:
        if file_path != "-":
            f = open(file_path, 'r', encoding='utf-8', errors='ignore')
        else:
            f = os.fdopen(0, 'r', encoding='utf-8', errors='ignore')
    except FileNotFoundError:
        logger.error(f"file '{file_path}' not found.")
        sys.exit(1)

    all_ops = []
    for line in f:
        match = op_pattern.search(line)
        if not match: continue

        pmu_raw = match.group('pmu')
        pmu_val = None
        if pmu_raw and pmu_index is not None:
            try:
                pmu_list = [int(x.strip()) for x in pmu_raw.split(',')]
                if len(pmu_list) > pmu_index:
                    pmu_val = pmu_list[pmu_index]
            except (ValueError, IndexError):
                pmu_val = None

        all_ops.append({
            'name':    match.group('op_name'),
            'dims':    match.group('dims').strip(),
            'types':   match.group('types').strip(),
            'usec':    int(match.group('usec')),
            'cycles':  int(match.group('cycles')),
            'pmu_val': pmu_val
        })

    f.close()

    return all_ops


def generate_report(ops, top_n, width_overrides, sort_col, pmu_name=None):
    if not ops:
        logger.info("No valid records found.")
        return

    grouped = defaultdict(list)
    for op in ops:
        key = (op['name'], op['dims'], op['types'])
        grouped[key].append(op)

    group_stats = []
    for (name, dims, types), group_ops in grouped.items():
        usecs = [o['usec'] for o in group_ops]
        cycles = [o['cycles'] for o in group_ops]
        pmu_vals = [o['pmu_val'] for o in group_ops if o['pmu_val'] is not None]

        group_stats.append({
            'op':               name,
            'dims':             dims,
            'dtypes':           types,
            'count':            str(len(group_ops)),
            'max_usec':         str(max(usecs)),
            'avg_usec':         f"{statistics.mean(usecs):.2f}",
            'max_cycles':       str(max(cycles)),
            'avg_cycles':       f"{statistics.mean(cycles):.2f}",
            'max_pmu':          str(max(pmu_vals)) if pmu_vals else "0",
            'avg_pmu':          f"{statistics.mean(pmu_vals):.2f}" if pmu_vals else "0.00",
            # Numeric values for accurate sorting
            '_sort_count':      len(group_ops),
            '_sort_max_usec':   max(usecs),
            '_sort_avg_usec':   statistics.mean(usecs),
            '_sort_max_cycles': max(cycles),
            '_sort_avg_cycles': statistics.mean(cycles),
            '_sort_max_pmu':    max(pmu_vals) if pmu_vals else 0,
            '_sort_avg_pmu':    statistics.mean(pmu_vals) if pmu_vals else 0
        })

    # Sorting logic
    actual_sort_key = COL_MAP[sort_col][2]
    # We sort numeric fields descending, strings (op/dims) ascending
    is_numeric    = actual_sort_key.startswith("_") or actual_sort_key == "count"
    sorted_groups = sorted(group_stats, key=lambda x: x[actual_sort_key], reverse=is_numeric)[:top_n]

    # Define initial column order
    active_cols = ["op", "dims", "dtypes"]
    if pmu_name:
        active_cols += ["max-pmu", "avg-pmu"]
    active_cols += ["max-usec", "avg-usec", "max-cycles", "avg-cycles", "count"]

    final_headers, final_keys, final_widths = [], [], []

    for col_name in active_cols:
        data_key, header_text, _ = COL_MAP[col_name]
        if "pmu" in col_name and pmu_name:
            header_text = header_text.replace("PMU", pmu_name)

        natural_width = max([len(row[data_key]) for row in sorted_groups] + [len(header_text)])
        target_width  = width_overrides.get(col_name, natural_width)

        if target_width == 0:
            continue

        final_headers.append(header_text)
        final_keys.append(data_key)
        final_widths.append(target_width)

    # Print Report
    logger.info(f"\n# Profile Report (Top {top_n} Ops sorted by {sort_col})\n")
    header_line = "| " + " | ".join(f"{h:<{final_widths[i]}}" for i, h in enumerate(final_headers)) + " |"
    sep_line    = "| " + " | ".join("-" * final_widths[i] for i in range(len(final_headers))) + " |"
    logger.info(header_line)
    logger.info(sep_line)

    for group in sorted_groups:
        row_vals = []
        for i, key in enumerate(final_keys):
            val = group[key]
            if len(val) > final_widths[i]:
                val = val[:final_widths[i] - 3] + "..."
            row_vals.append(f"{val:<{final_widths[i]}}")
        logger.info("| " + " | ".join(row_vals) + " |")


def main():
    parser = argparse.ArgumentParser(description="Post-process Op profile info.")
    parser.add_argument("logfile")
    parser.add_argument("-n", "--top", type=int, default=100)
    parser.add_argument("--sort", type=str, default="max-usec", choices=list(COL_MAP.keys()))
    parser.add_argument("--pmu-index", type=int)
    parser.add_argument("--pmu-name", type=str)
    parser.add_argument("--width", action='append', default=['dims:40'], help="Override column width, e.g. --width dims:50")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Sort validation: can't sort by PMU if index isn't provided
    if "pmu" in args.sort and args.pmu_index is None:
        logger.error(f"Cannot sort by '{args.sort}' without --pmu-index.")
        sys.exit(1)

    overrides = {}
    if args.width:
        for w in args.width:
            try:
                name, val = w.split(':')
                overrides[name.lower()] = int(val)
            except ValueError:
                logger.warning(f"Invalid width format '{w}'")

    final_pmu_name = (args.pmu_name or f"#{args.pmu_index}") if args.pmu_index is not None else None
    ops = parse_log(args.logfile, pmu_index=args.pmu_index)
    generate_report(ops, args.top, overrides, args.sort, pmu_name=final_pmu_name)


if __name__ == "__main__":
    main()
