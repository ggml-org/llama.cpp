#!/usr/bin/env python3

# Compare per-kernel occupancy between two build logs and fail on regressions.
#
# Works on the metrics.log files produced with GGML_HIP_EXPORT_METRICS=ON
# (-Rpass-analysis=kernel-resource-usage), with the same parsing principle as
# gcn-cdna-vgpr-check.py: the parallel build interleaves the remark lines, so
# each metric line is matched to the most recent function of the same location.
#
# Usage:
#   python3 gcn-cdna-occupancy-check.py <baseline_metrics.log> <pr_metrics.log> \
#           [--tolerance N] [--ignore FILE]
#
#   --tolerance N : allow occupancy drops of up to N waves/SIMD (default 0)
#   --ignore FILE : file with one kernel symbol per line to exclude (exact
#                   match, lines starting with '#' are comments)
#
# Exit code: 1 if at least one kernel regresses (PR occupancy below baseline),
# 0 otherwise. Kernels present in only one log are reported for information
# but do not fail the check.

import argparse
import re
import sys
from collections import defaultdict

FUNC_RE = re.compile(r'(?:remark: )?([^:]+):(\d+):\d+: Function Name: (\S+)')
OCC_RE = re.compile(r'(?:remark: )?([^:]+):(\d+):\d+:\s+ Occupancy \[waves/SIMD\]: (\d+)')


def parse_log_file(filepath):
    # A kernel may be instantiated in several TUs with different occupancy;
    # keep every value and use the worst (lowest) when comparing.
    functions = defaultdict(lambda: {'occupancies': [], 'location': ''})
    func_stack = []

    try:
        with open(filepath, 'r') as f:
            for line in f:
                func_match = FUNC_RE.search(line)
                if func_match:
                    location = func_match.group(1) + ':' + func_match.group(2)
                    func_name = func_match.group(3)
                    parts = location.split('/')
                    functions[func_name]['location'] = parts[-1] if parts else location
                    func_stack.append({'name': func_name, 'location': location})
                    continue

                occ_match = OCC_RE.search(line)
                if occ_match:
                    location = occ_match.group(1) + ':' + occ_match.group(2)
                    for i in range(len(func_stack) - 1, -1, -1):
                        if func_stack[i]['location'] == location:
                            functions[func_stack[i]['name']]['occupancies'].append(int(occ_match.group(3)))
                            break
    except FileNotFoundError:
        print(f"Error: File {filepath} not found", file=sys.stderr)  # noqa: NP100
        sys.exit(1)

    return functions


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


def main():
    parser = argparse.ArgumentParser(description='Check for occupancy regressions between two build logs')
    parser.add_argument('baseline_log', help='metrics.log of the baseline (e.g. master)')
    parser.add_argument('pr_log', help='metrics.log of the PR branch')
    parser.add_argument('-t', '--tolerance', type=int, default=0,
                        help='allow occupancy drops of up to N waves/SIMD (default 0)')
    parser.add_argument('--ignore', help='file with kernel symbols to exclude, one per line')
    args = parser.parse_args()

    ignored = load_ignore_file(args.ignore) if args.ignore else set()

    baseline = parse_log_file(args.baseline_log)
    pr = parse_log_file(args.pr_log)

    base_names = {name for name in baseline if baseline[name]['occupancies']}
    pr_names = {name for name in pr if pr[name]['occupancies']}
    common = base_names & pr_names

    regressions = []
    gains = []
    for name in common:
        if name in ignored:
            continue
        base_occ = min(baseline[name]['occupancies'])
        pr_occ = min(pr[name]['occupancies'])
        if pr_occ < base_occ - args.tolerance:
            regressions.append((name, base_occ, pr_occ))
        elif pr_occ > base_occ:
            gains.append((name, base_occ, pr_occ))

    added = pr_names - base_names - ignored
    removed = base_names - pr_names - ignored

    print(f"Kernels compared: {len(common)} (baseline {len(base_names)}, PR {len(pr_names)})")  # noqa: NP100
    print(f"Added kernels: {len(added)}, removed kernels: {len(removed)}")  # noqa: NP100

    for name, base_occ, pr_occ in sorted(regressions):
        location = pr[name]['location'] or baseline[name]['location']
        print(f"\033[91m{location}: {name} - Occupancy: {base_occ} -> {pr_occ} (REGRESSION)\033[0m")  # noqa: NP100

    for name, base_occ, pr_occ in sorted(gains):
        location = pr[name]['location'] or baseline[name]['location']
        print(f"\033[92m{location}: {name} - Occupancy: {base_occ} -> {pr_occ} (improved)\033[0m")  # noqa: NP100

    for name in sorted(added):
        print(f"note: new kernel in PR: {name} (occupancy {min(pr[name]['occupancies'])})")  # noqa: NP100
    for name in sorted(removed):
        print(f"note: kernel only in baseline: {name} (occupancy {min(baseline[name]['occupancies'])})")  # noqa: NP100

    if regressions:
        print(f"\033[91m{len(regressions)} occupancy regression(s) found\033[0m")  # noqa: NP100
        sys.exit(1)
    print("No occupancy regressions found")  # noqa: NP100


if __name__ == "__main__":
    main()
