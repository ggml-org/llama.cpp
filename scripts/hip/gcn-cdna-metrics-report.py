#!/usr/bin/env python3

# Generate a markdown report of per-kernel occupancy and spill changes from
# the TSV deltas file written by gcn-cdna-metrics-check.py --dump.
#
# Usage:
#   python3 gcn-cdna-metrics-report.py <deltas.tsv> [-o FILE]
#
#   -o FILE : write the report to FILE (default: hip-kernel-metrics-report.md)

import argparse
import sys

METRIC_NAMES = {
    'occupancy': 'Occupancy [waves/SIMD]',
    'vgpr_spill': 'VGPRs Spill',
    'sgpr_spill': 'SGPRs Spill',
}

METRICS = ('occupancy', 'vgpr_spill', 'sgpr_spill')


def read_deltas(filepath):
    meta = {}
    rows = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('#'):
                    parts = line[1:].split('\t')
                    if len(parts) == 2:
                        meta[parts[0].strip()] = parts[1].strip()
                    continue
                cols = line.split('\t')
                if len(cols) == 6 and cols[0] in METRICS:
                    rows.append(cols)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found", file=sys.stderr)  # noqa: NP100
        sys.exit(1)
    return meta, rows


def write_report(path, meta, rows):
    tables = {}  # (metric, status) -> list of (kernel, location, baseline, pr)
    added = {}   # kernel -> [location, {metric: value}]
    removed = {} # kernel -> [location, {metric: value}]
    for metric, kernel, location, baseline, pr, status in rows:
        if status in ('regression', 'improved'):
            tables.setdefault((metric, status), []).append((kernel, location, baseline, pr))
        elif status == 'added':
            added.setdefault(kernel, [location, {}])[1][metric] = pr
        elif status == 'removed':
            removed.setdefault(kernel, [location, {}])[1][metric] = baseline

    with open(path, 'w') as f:
        f.write('# HIP kernel metrics comparison\n\n')
        f.write(f"- Baseline: `{meta.get('baseline_log', '?')}`\n")
        f.write(f"- PR: `{meta.get('pr_log', '?')}`\n\n")
        f.write(f"Kernels compared: {meta.get('kernels_compared', '?')} "
                f"(baseline {meta.get('baseline_kernels', '?')}, PR {meta.get('pr_kernels', '?')}). "
                f"Added kernels: {len(added)}, removed kernels: {len(removed)}.\n\n")

        for metric in METRICS:
            name = METRIC_NAMES[metric]
            for status, title in (('regression', 'regressions'), ('improved', 'improvements')):
                entries = tables.get((metric, status), [])
                f.write(f'## {name} {title} ({len(entries)})\n\n')
                if entries:
                    f.write('| Location | Kernel | Baseline | PR |\n|---|---|---|---|\n')
                    for kernel, location, baseline, pr in entries:
                        f.write(f'| {location} | `{kernel}` | {baseline} | {pr} |\n')
                else:
                    f.write('None.\n')
                f.write('\n')

        for title, kernels in (('Added kernels', added), ('Removed kernels', removed)):
            f.write(f'## {title} ({len(kernels)})\n\n')
            if kernels:
                f.write('| Location | Kernel | Occupancy | VGPRs Spill | SGPRs Spill |\n|---|---|---|---|---|\n')
                for kernel in sorted(kernels):
                    location, values = kernels[kernel]
                    occupancy = values.get('occupancy', '')
                    vspill = values.get('vgpr_spill', '')
                    sspill = values.get('sgpr_spill', '')
                    f.write(f'| {location} | `{kernel}` | {occupancy} | {vspill} | {sspill} |\n')
            else:
                f.write('None.\n')
            f.write('\n')


def main():
    parser = argparse.ArgumentParser(description='Generate a markdown report from a gcn-cdna-metrics-check.py TSV dump')
    parser.add_argument('deltas_tsv', help='TSV deltas file written by gcn-cdna-metrics-check.py --dump')
    parser.add_argument('-o', '--output', default='hip-kernel-metrics-report.md',
                        help='write the report to FILE (default: hip-kernel-metrics-report.md)')
    args = parser.parse_args()

    meta, rows = read_deltas(args.deltas_tsv)
    write_report(args.output, meta, rows)
    print(f"Report written to {args.output}")  # noqa: NP100


if __name__ == "__main__":
    main()
