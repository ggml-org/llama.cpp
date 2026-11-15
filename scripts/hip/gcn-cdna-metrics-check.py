#!/usr/bin/env python3

# Compare per-kernel occupancy and register spills between two build logs and
# fail on regressions.
#
# Works on the metrics.log files produced with GGML_HIP_EXPORT_METRICS=ON
# (-Rpass-analysis=kernel-resource-usage), with the same parsing principle as
# gcn-cdna-vgpr-check.py: the parallel build interleaves the remark lines, so
# each metric line is matched to the most recent function of the same location.
#
# Usage:
#   python3 gcn-cdna-metrics-check.py <baseline_metrics.log> <pr_metrics.log> \
#           [--tolerance N] [--ignore FILE] [--dump FILE]
#
#   --tolerance N : allow occupancy drops of up to N waves/SIMD (default 0)
#   --ignore FILE : file with one kernel symbol per line to exclude (exact
#                   match, lines starting with '#' are comments)
#   --dump FILE   : also write all deltas to FILE as TSV, for consumption by
#                   gcn-cdna-metrics-report.py
#
# Exit code: 1 if at least one kernel regresses (occupancy drop or new spills),
# 0 otherwise. Kernels present in only one log are reported for information
# but do not fail the check.

import argparse
import re
import sys
from collections import defaultdict

FUNC_RE = re.compile(r'(?:remark: )?([^:]+):(\d+):\d+: Function Name: (\S+)')
OCC_RE = re.compile(r'(?:remark: )?([^:]+):(\d+):\d+:\s+ Occupancy \[waves/SIMD\]: (\d+)')
VSPILL_RE = re.compile(r'(?:remark: )?([^:]+):(\d+):\d+:\s+ VGPRs Spill: (\d+)')
SSPILL_RE = re.compile(r'(?:remark: )?([^:]+):(\d+):\d+:\s+ SGPRs Spill: (\d+)')

METRICS = ('occupancy', 'vgpr_spill', 'sgpr_spill')


def parse_log_file(filepath):
    # A kernel may be instantiated in several TUs with different metrics;
    # keep every value and use the worst one when comparing.
    occupancies = defaultdict(list)  # kernel name -> occupancy values
    vspills = defaultdict(list)      # kernel name -> VGPR spill counts
    sspills = defaultdict(list)      # kernel name -> SGPR spill counts
    locations = defaultdict(str)     # kernel name -> last seen location
    func_stack = []

    def match_metric(metric_match, values):
        location = metric_match.group(1) + ':' + metric_match.group(2)
        for i in range(len(func_stack) - 1, -1, -1):
            if func_stack[i]['location'] == location:
                values[func_stack[i]['name']].append(int(metric_match.group(3)))
                break

    try:
        with open(filepath, 'r') as f:
            for line in f:
                func_match = FUNC_RE.search(line)
                if func_match:
                    location = func_match.group(1) + ':' + func_match.group(2)
                    func_name = func_match.group(3)
                    parts = location.split('/')
                    locations[func_name] = parts[-1] if parts else location
                    func_stack.append({'name': func_name, 'location': location})
                    continue

                occ_match = OCC_RE.search(line)
                if occ_match:
                    match_metric(occ_match, occupancies)
                    continue

                vspill_match = VSPILL_RE.search(line)
                if vspill_match:
                    match_metric(vspill_match, vspills)
                    continue

                sspill_match = SSPILL_RE.search(line)
                if sspill_match:
                    match_metric(sspill_match, sspills)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found", file=sys.stderr)  # noqa: NP100
        sys.exit(1)

    return occupancies, vspills, sspills, locations


def load_ignore_file(filepath):
    ignored = set()
    try:
        with open(filepath, 'r') as f:
            for line in f:
                name = line.strip()
                if name and not name.startswith('#'):
                    ignored.add(name)
    except FileNotFoundError:
        print(f"Error: Ignore file {filepath} not found", file=sys.stderr)  # noqa: NP100
        sys.exit(1)
    return ignored


def worst(values, metric):
    # Occupancy is worst at its minimum, spills at their maximum.
    return min(values) if metric == 'occupancy' else max(values)


def compare_metric(base_values, pr_values, names, worse, tolerance=0):
    # worse: comparison direction, 'min' for occupancy (lower is worse),
    # 'max' for spills (higher is worse). Only compare names that have the
    # metric on both sides.
    regressions = []
    gains = []
    for name in names:
        if not base_values[name] or not pr_values[name]:
            continue
        base_val = min(base_values[name]) if worse == 'min' else max(base_values[name])
        pr_val = min(pr_values[name]) if worse == 'min' else max(pr_values[name])
        if worse == 'min':
            if pr_val < base_val - tolerance:
                regressions.append((name, base_val, pr_val))
            elif pr_val > base_val:
                gains.append((name, base_val, pr_val))
        else:
            if pr_val > base_val:
                regressions.append((name, base_val, pr_val))
            elif pr_val < base_val:
                gains.append((name, base_val, pr_val))
    return regressions, gains


def dump_tsv(path, baseline_log, pr_log, base_count, pr_count, common_count, rows):
    with open(path, 'w') as f:
        f.write(f'#baseline_log\t{baseline_log}\n')
        f.write(f'#pr_log\t{pr_log}\n')
        f.write(f'#baseline_kernels\t{base_count}\n')
        f.write(f'#pr_kernels\t{pr_count}\n')
        f.write(f'#kernels_compared\t{common_count}\n')
        f.write('metric\tkernel\tlocation\tbaseline\tpr\tstatus\n')
        for row in rows:
            f.write('\t'.join(str(col) for col in row) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Check for occupancy and spill regressions between two build logs')
    parser.add_argument('baseline_log', help='metrics.log of the baseline (e.g. master)')
    parser.add_argument('pr_log', help='metrics.log of the PR branch')
    parser.add_argument('-t', '--tolerance', type=int, default=0,
                        help='allow occupancy drops of up to N waves/SIMD (default 0)')
    parser.add_argument('--ignore', help='file with kernel symbols to exclude, one per line')
    parser.add_argument('--dump', help='write all deltas to FILE as TSV, for gcn-cdna-metrics-report.py')
    args = parser.parse_args()

    ignored = load_ignore_file(args.ignore) if args.ignore else set()

    base_occ, base_vspill, base_sspill, base_loc = parse_log_file(args.baseline_log)
    pr_occ, pr_vspill, pr_sspill, pr_loc = parse_log_file(args.pr_log)
    locations = {**base_loc, **pr_loc}

    base_names = ({name for name in base_occ if base_occ[name]}
                  | set(base_vspill) | set(base_sspill))
    pr_names = ({name for name in pr_occ if pr_occ[name]}
                | set(pr_vspill) | set(pr_sspill))
    base_names -= ignored
    pr_names -= ignored
    common = base_names & pr_names

    occ_regressions, occ_gains = compare_metric(base_occ, pr_occ, common, 'min', args.tolerance)
    vspill_regressions, vspill_gains = compare_metric(base_vspill, pr_vspill, common, 'max')
    sspill_regressions, sspill_gains = compare_metric(base_sspill, pr_sspill, common, 'max')

    added = pr_names - base_names
    removed = base_names - pr_names

    print(f"Kernels compared: {len(common)} (baseline {len(base_names)}, PR {len(pr_names)})")  # noqa: NP100
    print(f"Added kernels: {len(added)}, removed kernels: {len(removed)}")  # noqa: NP100

    for name, base_val, pr_val in sorted(occ_regressions):
        print(f"\033[91m{locations[name]}: {name} - Occupancy: {base_val} -> {pr_val} (REGRESSION)\033[0m")  # noqa: NP100
    for name, base_val, pr_val in sorted(vspill_regressions):
        print(f"\033[91m{locations[name]}: {name} - VGPRs Spill: {base_val} -> {pr_val} (REGRESSION)\033[0m")  # noqa: NP100
    for name, base_val, pr_val in sorted(sspill_regressions):
        print(f"\033[91m{locations[name]}: {name} - SGPRs Spill: {base_val} -> {pr_val} (REGRESSION)\033[0m")  # noqa: NP100

    for name, base_val, pr_val in sorted(occ_gains):
        print(f"\033[92m{locations[name]}: {name} - Occupancy: {base_val} -> {pr_val} (improved)\033[0m")  # noqa: NP100
    for name, base_val, pr_val in sorted(vspill_gains):
        print(f"\033[92m{locations[name]}: {name} - VGPRs Spill: {base_val} -> {pr_val} (improved)\033[0m")  # noqa: NP100
    for name, base_val, pr_val in sorted(sspill_gains):
        print(f"\033[92m{locations[name]}: {name} - SGPRs Spill: {base_val} -> {pr_val} (improved)\033[0m")  # noqa: NP100

    for name in sorted(added):
        print(f"note: new kernel in PR: {name}")  # noqa: NP100
    for name in sorted(removed):
        print(f"note: kernel only in baseline: {name}")  # noqa: NP100

    if args.dump:
        rows = []
        pr_metrics = {'occupancy': pr_occ, 'vgpr_spill': pr_vspill, 'sgpr_spill': pr_sspill}
        base_metrics = {'occupancy': base_occ, 'vgpr_spill': base_vspill, 'sgpr_spill': base_sspill}
        for metric, regressions, gains in (
            ('occupancy', occ_regressions, occ_gains),
            ('vgpr_spill', vspill_regressions, vspill_gains),
            ('sgpr_spill', sspill_regressions, sspill_gains),
        ):
            for name, base_val, pr_val in sorted(regressions):
                rows.append((metric, name, locations[name], base_val, pr_val, 'regression'))
            for name, base_val, pr_val in sorted(gains):
                rows.append((metric, name, locations[name], base_val, pr_val, 'improved'))
        for name in sorted(added):
            for metric in METRICS:
                if pr_metrics[metric][name]:
                    rows.append((metric, name, locations[name], '', worst(pr_metrics[metric][name], metric), 'added'))
        for name in sorted(removed):
            for metric in METRICS:
                if base_metrics[metric][name]:
                    rows.append((metric, name, locations[name], worst(base_metrics[metric][name], metric), '', 'removed'))
        dump_tsv(args.dump, args.baseline_log, args.pr_log, len(base_names), len(pr_names), len(common), rows)
        print(f"Deltas written to {args.dump}")  # noqa: NP100

    regressions = occ_regressions + vspill_regressions + sspill_regressions
    if regressions:
        print(f"\033[91m{len(regressions)} regression(s) found\033[0m")  # noqa: NP100
        sys.exit(1)
    print("No regressions found")  # noqa: NP100


if __name__ == "__main__":
    main()
