#!/usr/bin/env python3

# Check HIP VGPR limits, compare kernel resource usage, and generate a report.
# Read Clang optimization records produced with -fsave-optimization-record=yaml
# and -Rpass-analysis=kernel-resource-usage.

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import cast

import yaml

try:
    from itanium_demangler import parse as demangle  # ty: ignore[import-not-found]
except ImportError:
    demangle = None

METRICS = ('occupancy', 'vgpr_spill', 'sgpr_spill')
REPORT_METRICS = METRICS + ('vgpr_total',)
RULES = {
    'vgpr_limit': 'blocking',
    'vgpr_spill_increase': 'blocking',
    'sgpr_spill_increase': 'blocking',
    'occupancy_vgpr_increase': 'blocking',
    'occupancy_drop': 'warning',
}
REMARK_NAMES = {'NumVGPR': 'vgprs', 'Occupancy': 'occupancy', 'VGPRSpill': 'vgpr_spill', 'SGPRSpill': 'sgpr_spill'}
METRIC_NAMES = {'occupancy': 'Occupancy [waves/SIMD]', 'vgpr_spill': 'VGPRs Spill', 'sgpr_spill': 'SGPRs Spill', 'vgpr_total': 'Total VGPRs (limit 256)'}

IGNORED_VGPR = {

    '_ZL13rwkv_wkv7_f32ILi128EEviiiiPKfS1_S1_S1_S1_S1_S1_Pf',
    '_ZL18flash_attn_ext_f16ILi512ELi512ELi16ELi2ELb0ELb0ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi512ELi512ELi16ELi2ELb1ELb0ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi512ELi512ELi16ELi4ELb0ELb0ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi512ELi512ELi16ELi4ELb1ELb0ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi512ELi512ELi32ELi2ELb0ELb0ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi512ELi512ELi32ELi2ELb1ELb0ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi512ELi512ELi4ELi8ELb0ELb0ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi512ELi512ELi4ELi8ELb1ELb0ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi512ELi512ELi8ELi4ELb0ELb0ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi512ELi512ELi8ELi4ELb1ELb0ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi512ELi512ELi8ELi8ELb0ELb0ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi512ELi512ELi8ELi8ELb1ELb0ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi576ELi512ELi16ELi4ELb0ELb1ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi576ELi512ELi1ELi16ELb0ELb1ELb1EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi576ELi512ELi1ELi32ELb0ELb1ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi576ELi512ELi2ELi16ELb0ELb1ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi576ELi512ELi2ELi32ELb0ELb1ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi576ELi512ELi4ELi16ELb0ELb1ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_f16ILi576ELi512ELi8ELi4ELb0ELb1ELb0EL9ggml_type1ELS0_1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_vecILi256ELi1EL9ggml_type2ELS0_2ELb0EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_vecILi256ELi1EL9ggml_type43ELS0_43ELb0EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL18flash_attn_ext_vecILi256ELi1EL9ggml_type43ELS0_43ELb1EEvPK15HIP_vector_typeIfLj2EEPKcS6_PK6__halfPKfPKiSD_PfPS2_ffffjfiS1_IjLj3EEiiiiiiiiiiiliiliiiill',
    '_ZL21gated_linear_attn_f32ILi128EEviiiifPKfS1_S1_S1_S1_Pf',
}


class RemarkLoader(yaml.SafeLoader):
    pass


def construct_remark(loader, tag, node):
    return loader.construct_mapping(node)


RemarkLoader.add_multi_constructor('!', construct_remark)


def format_name(name):
    if demangle is None:
        return name
    try:
        demangled = demangle(name)
    except Exception:
        return name
    return f'{name} [{demangled}]' if demangled is not None else name


def parse_remarks(filepath):
    path = Path(filepath)
    files = sorted(path.rglob('*.opt.yaml')) if path.is_dir() else [path] if path.is_file() else []
    if not files:
        raise ValueError(f'No optimization records found in {filepath}')

    functions = defaultdict(lambda: {'vgprs': [], 'occupancy': [], 'vgpr_spill': [], 'sgpr_spill': [], 'location': ''})
    for file in files:
        with file.open('r') as f:
            for remark in yaml.load_all(f, Loader=RemarkLoader):
                if not isinstance(remark, dict) or remark.get('Pass') != 'kernel-resource-usage':
                    continue
                key = remark.get('Name')
                if key not in REMARK_NAMES:
                    continue
                name = remark.get('Function')
                if not isinstance(name, str) or not name:
                    continue
                values = [arg[key] for arg in remark.get('Args', []) if isinstance(arg, dict) and key in arg]
                if len(values) != 1:
                    raise ValueError(f'Invalid {key} remark for {name} in {file}')
                try:
                    value = int(values[0])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'Invalid {key} value for {name} in {file}') from exc
                cast(list[int], functions[name][REMARK_NAMES[key]]).append(value)
                location = remark.get('DebugLoc') or {}
                if location.get('File') and location.get('Line'):
                    functions[name]['location'] = f"{Path(location['File']).name}:{location['Line']}"

    if not functions:
        raise ValueError(f'No kernel resource usage remarks found in {filepath}')
    return functions


def load_ignore_file(filepath):
    ignored = set()
    with open(filepath, 'r') as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith('#'):
                ignored.add(name)
    return ignored


def check_vgpr(functions, filepath, rows=None):
    found_issues = False
    for ignored in (True, False):
        for name, data in sorted(functions.items()):
            if (name in IGNORED_VGPR) != ignored:
                continue
            vgprs = data['vgprs'][-1] if data['vgprs'] else 0
            spill = data['vgpr_spill'][-1] if data['vgpr_spill'] else 0
            total_vgprs = vgprs + spill
            if total_vgprs <= 256:
                continue
            location = data['location'] or filepath
            suffix = ' [IGNORED]' if ignored else ' '
            message = f"{location}: {format_name(name)} - Total VGPRs: {total_vgprs} ({vgprs} + {spill}){suffix}"
            blocking = RULES['vgpr_limit'] == 'blocking'
            color = '91' if blocking else '93'
            print(message if ignored else f'\033[{color}m{message}\033[0m')  # noqa: NP100
            if not ignored:
                found_issues |= blocking
                if rows is not None:
                    rows.append(('vgpr_total', name, location, 256, total_vgprs, 'regression' if blocking else 'warning'))
    return int(found_issues)


def worst(values, metric):
    return min(values) if metric == 'occupancy' else max(values)


def compare_metric(base, pr, names, metric, tolerance=0):
    regressions = []
    gains = []
    for name in names:
        if not base[name][metric] or not pr[name][metric]:
            continue
        base_val = worst(base[name][metric], metric)
        pr_val = worst(pr[name][metric], metric)
        if metric == 'occupancy':
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


def check_comparison(base, pr, baseline_log, pr_log, tolerance=0, ignored=None, dump=None):
    ignored = ignored or set()
    base_names = {name for name, data in base.items() if any(data[metric] for metric in METRICS)} - ignored
    pr_names = {name for name, data in pr.items() if any(data[metric] for metric in METRICS)} - ignored
    common = base_names & pr_names
    added = pr_names - base_names
    removed = base_names - pr_names
    locations = {name: data['location'] for name, data in base.items()}
    locations.update({name: data['location'] for name, data in pr.items()})

    changes = {metric: compare_metric(base, pr, common, metric, tolerance) for metric in METRICS}
    issues = []  # metric, kernel, baseline, PR, severity, extra details
    for metric, rule in (('vgpr_spill', 'vgpr_spill_increase'), ('sgpr_spill', 'sgpr_spill_increase')):
        for name, base_val, pr_val in changes[metric][0]:
            issues.append((metric, name, base_val, pr_val, RULES[rule], ''))
    for name, base_val, pr_val in changes['occupancy'][0]:
        base_vgprs = worst(base[name]['vgprs'], 'vgprs') if base[name]['vgprs'] else None
        pr_vgprs = worst(pr[name]['vgprs'], 'vgprs') if pr[name]['vgprs'] else None
        increasing = base_vgprs is not None and pr_vgprs is not None and pr_vgprs > base_vgprs
        rule = 'occupancy_vgpr_increase' if increasing else 'occupancy_drop'
        detail = f', VGPRs: {base_vgprs} -> {pr_vgprs}' if increasing else ''
        issues.append(('occupancy', name, base_val, pr_val, RULES[rule], detail))

    print(f'Kernels compared: {len(common)} (baseline {len(base_names)}, PR {len(pr_names)})')  # noqa: NP100
    print(f'Added kernels: {len(added)}, removed kernels: {len(removed)}')  # noqa: NP100
    for severity, color, label in (('blocking', '91', 'REGRESSION'), ('warning', '93', 'WARNING')):
        for metric, name, base_val, pr_val, level, detail in sorted(issues):
            if level == severity:
                print(f"\033[{color}m{locations[name]}: {name} - {METRIC_NAMES[metric]}: {base_val} -> {pr_val}{detail} ({label})\033[0m")  # noqa: NP100
    for metric in METRICS:
        for name, base_val, pr_val in sorted(changes[metric][1]):
            print(f"\033[92m{locations[name]}: {name} - {METRIC_NAMES[metric]}: {base_val} -> {pr_val} (improved)\033[0m")  # noqa: NP100
    for name in sorted(added):
        print(f'note: new kernel in PR: {name}')  # noqa: NP100
    for name in sorted(removed):
        print(f'note: kernel only in baseline: {name}')  # noqa: NP100

    # Check the absolute VGPR limit as well, even for kernels excluded from comparisons.
    vgpr_rows = []
    vgpr_failed = check_vgpr(pr, pr_log, vgpr_rows)
    if dump:
        rows = []
        for metric, name, base_val, pr_val, severity, _ in issues:
            rows.append((metric, name, locations[name], base_val, pr_val, 'regression' if severity == 'blocking' else 'warning'))
        for metric in METRICS:
            for name, base_val, pr_val in sorted(changes[metric][1]):
                rows.append((metric, name, locations[name], base_val, pr_val, 'improved'))
        rows.extend(vgpr_rows)
        for name in sorted(added):
            for metric in METRICS:
                if pr[name][metric]:
                    rows.append((metric, name, locations[name], '', worst(pr[name][metric], metric), 'added'))
        for name in sorted(removed):
            for metric in METRICS:
                if base[name][metric]:
                    rows.append((metric, name, locations[name], worst(base[name][metric], metric), '', 'removed'))
        dump_tsv(dump, baseline_log, pr_log, len(base_names), len(pr_names), len(common), rows)
        print(f'Deltas written to {dump}')  # noqa: NP100

    blocking = sum(severity == 'blocking' for _, _, _, _, severity, _ in issues)
    warnings = sum(severity == 'warning' for _, _, _, _, severity, _ in issues)
    blocking += sum(row[5] == 'regression' for row in vgpr_rows)
    warnings += sum(row[5] == 'warning' for row in vgpr_rows)
    if blocking or warnings:
        print(f'{blocking} blocking regression(s), {warnings} warning(s)')  # noqa: NP100
    else:
        print('No regressions found')  # noqa: NP100
    return int(bool(blocking or vgpr_failed))


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
                if len(cols) == 6 and cols[0] in REPORT_METRICS:
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
        if status in ('regression', 'warning', 'improved'):
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

        for metric in REPORT_METRICS:
            name = METRIC_NAMES[metric]
            statuses = (('regression', 'regressions'), ('improved', 'improvements'))
            if metric == 'occupancy':
                statuses = (('regression', 'regressions'), ('warning', 'warnings'), ('improved', 'improvements'))
            elif metric == 'vgpr_total':
                statuses = (('regression', 'regressions'), ('warning', 'warnings'))
            if metric == 'vgpr_total':
                f.write('The baseline value for total VGPRs is the 256-register threshold.\n\n')
            for status, title in statuses:
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
    parser = argparse.ArgumentParser(description='Check HIP kernel resource usage and generate reports')
    commands = parser.add_subparsers(dest='command', required=True)
    vgpr = commands.add_parser('vgpr', help='check the 256 VGPR limit')
    vgpr.add_argument('records', help='directory of Clang .opt.yaml files')
    compare = commands.add_parser('compare', help='compare occupancy and register spills')
    compare.add_argument('baseline_log', help='baseline optimization records')
    compare.add_argument('pr_log', help='PR optimization records')
    compare.add_argument('-t', '--tolerance', type=int, default=0, help='allowed occupancy drop in waves/SIMD')
    compare.add_argument('--ignore', help='file with kernel symbols to exclude, one per line')
    compare.add_argument('--dump', help='write all deltas to a TSV file')
    report = commands.add_parser('report', help='generate the Markdown report from a TSV dump')
    report.add_argument('deltas_tsv', help='TSV deltas from compare --dump')
    report.add_argument('-o', '--output', default='hip-kernel-metrics-report.md', help='output Markdown file')
    args = parser.parse_args()
    try:
        if args.command == 'vgpr':
            return check_vgpr(parse_remarks(args.records), args.records)
        if args.command == 'compare':
            ignored = load_ignore_file(args.ignore) if args.ignore else set()
            return check_comparison(parse_remarks(args.baseline_log), parse_remarks(args.pr_log), args.baseline_log, args.pr_log, args.tolerance, ignored, args.dump)
        meta, rows = read_deltas(args.deltas_tsv)
        write_report(args.output, meta, rows)
        print(f'Report written to {args.output}')  # noqa: NP100
        return 0
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f'Error: {exc}', file=sys.stderr)  # noqa: NP100
        return 1


if __name__ == '__main__':
    sys.exit(main())
