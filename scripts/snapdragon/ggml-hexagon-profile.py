#!/usr/bin/env python3

import sys
import os
import re
import argparse
import statistics
import logging
from typing import Any, Dict, List, Optional

from collections import defaultdict

# Mapping of cli-friendly names to (internal_data_key, Display Header, numeric_sort_key)
COL_MAP = {
    "tot-usec":   ("tot_usec",   "Tot usec",   "_sort_tot_usec"),
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
    r"profile-op\s+(?P<op_name>[A-Z_0-9+]+):\s+.*?\s+:\s+(?P<dims>[\d:x\s\->!]+)\s+:\s+(?P<types>[a-z\d_\s\->x]+)\s+:\s+.*?\s+:\s+(?:op-)?usec\s+(?P<usec>\d+)\s+(?:op-)?cycles\s+(?P<cycles>\d+)(?:\s+start\s+(?P<start>\d+))?(?:\s+mhz\s+(?P<mhz>[\d.]+))?(?:\s+pmu\s+\[(?P<pmu>[\d,\s]+)\])?(?:\s+evt\s+\[(?P<evt>[\d,\s]+)\])?"
)

trace_pattern = re.compile(
    r"trace-op\s+(?P<op_name>[A-Z_0-9+]+):\s+thread\s+(?P<thread>\d+)\s+event\s+(?P<event>[A-Z_0-9\-]+)\s+info\s+(?P<info>\d+)\s+(?P<state>start|stop)\s+(?P<cycles>\d+)"
)

logger = logging.getLogger("ggml-hexagon-profile")


def normalize_event_name(evt_type):
    if evt_type == "HVX_COMP":
        return "V-COMP"
    if evt_type == "HMX_COMP":
        return "M-COMP"

    # Strip HVX_ or HMX_ prefixes
    name = evt_type
    if name.startswith("HVX_") or name.startswith("HMX_"):
        name = name[4:]
    return name.replace("_", "-")


class CycleUnwrapper:
    def __init__(self):
        self.last_raw = None
        self.high_part = 0

    def unwrap(self, raw):
        if self.last_raw is None:
            self.last_raw = raw
            return raw
        diff = raw - self.last_raw
        if diff < -0x80000000:
            self.high_part += 0x100000000
        elif diff > 0x80000000:
            self.high_part -= 0x100000000
        self.last_raw = raw
        return raw + self.high_part


def parse_log(file_path, pmu_index=None):
    try:
        if file_path != "-":
            f = open(file_path, 'r', encoding='utf-8', errors='ignore')
        else:
            f = os.fdopen(0, 'r', encoding='utf-8', errors='ignore')
    except FileNotFoundError:
        logger.error(f"file '{file_path}' not found.")
        sys.exit(1)

    all_ops: List[Dict[str, Any]] = []
    current_op: Optional[Dict[str, Any]] = None

    timestamp_pattern = re.compile(r"^(?P<min>\d+)\.(?P<sec>\d+)\.(?P<ms>\d+)\.(?P<us>\d+)\s+[A-Z]\s+")
    unwrapper = CycleUnwrapper()

    for line in f:
        ts_match = timestamp_pattern.match(line)
        abs_usec = 0
        if ts_match:
            abs_usec = (
                (int(ts_match.group('min')) * 60 + int(ts_match.group('sec'))) * 1000000
                + int(ts_match.group('ms')) * 1000
                + int(ts_match.group('us'))
            )

        if "|" in line and "profile-op" in line:
            parts = [p.strip() for p in line.split("|")]
            prefix = parts[0]
            prefix_match = re.search(r"profile-op\s+(?P<op_name>[A-Z_0-9+]+)", prefix)
            if not prefix_match:
                continue

            if len(parts) == 7:
                dims, types, timings = parts[2], parts[3], parts[6]
            elif len(parts) == 6:
                dims, types, timings = parts[2], parts[3], parts[5]
            else:
                continue

            timing_match = re.search(
                r"(?:op-)?usec\s+(?P<usec>\d+)\s+(?:op-)?cycles\s+(?P<cycles>\d+)(?:\s+start\s+(?P<start>\d+))?(?:\s+mhz\s+(?P<mhz>[\d.]+))?(?:\s+pmu\s+\[(?P<pmu>[\d,\s]+)\])?(?:\s+evt\s+\[(?P<evt>[\d,\s]+)\])?",
                timings
            )
            if not timing_match:
                continue

            op_match = timing_match
            op_name = prefix_match.group("op_name")
        else:
            op_match = op_pattern.search(line)
            if op_match:
                op_name = op_match.group('op_name')
                dims = op_match.group('dims').strip()
                types = op_match.group('types').strip()
            else:
                op_match = None

        if op_match:
            pmu_raw = op_match.group('pmu') if 'pmu' in op_match.groupdict() else None
            pmu_val = None
            if pmu_raw and pmu_index is not None:
                try:
                    pmu_list = [int(x.strip()) for x in pmu_raw.split(',')]
                    if len(pmu_list) > pmu_index:
                        pmu_val = pmu_list[pmu_index]
                except (ValueError, IndexError):
                    pmu_val = None

            evt_raw = op_match.group('evt') if 'evt' in op_match.groupdict() else None
            evt_val = None
            if evt_raw:
                try:
                    evt_val = [int(x.strip()) for x in evt_raw.split(',')]
                except ValueError:
                    evt_val = None

            cycles_start_raw = op_match.group('start')
            unwrapped_cycles_start = None
            if cycles_start_raw:
                unwrapped_cycles_start = unwrapper.unwrap(int(cycles_start_raw))

            idx = line.find("profile-op ")
            op_text = line[idx + 11:].strip() if idx != -1 else line.strip()

            current_op = {
                'name':         op_name,
                'dims':         dims,
                'types':        types,
                'op_text':      op_text,
                'usec':         int(op_match.group('usec')),
                'cycles':       int(op_match.group('cycles')),
                'cycles_start': int(cycles_start_raw) if cycles_start_raw else None,
                'unwrapped_cycles_start': unwrapped_cycles_start,
                'pmu_val':      pmu_val,
                'evt_val':      evt_val,
                'abs_usec':     abs_usec,
                'trace_events': []
            }
            all_ops.append(current_op)
            continue

        trace_match = trace_pattern.search(line)
        if trace_match and current_op:
            if trace_match.group('op_name') == current_op['name']:
                raw_cyc = int(trace_match.group('cycles'))
                current_op['trace_events'].append({
                    'thread': int(trace_match.group('thread')),
                    'event':  trace_match.group('event'),
                    'info':   int(trace_match.group('info')),
                    'cycles': raw_cyc,
                    'unwrapped_cycles': unwrapper.unwrap(raw_cyc),
                    'state':  trace_match.group('state')
                })

    f.close()
    return all_ops


def print_ascii_timeline(op_name, dims, types, usec, cycles, events, evt_val=None):
    evt_str = ""
    if evt_val:
        evt_str = " - evt [" + ",".join(str(x) for x in evt_val) + "]"
    logger.info("=" * 100)
    logger.info(f"{op_name} ({dims} : {types}) - {usec} usec {cycles} cycles{evt_str}")
    logger.info("=" * 100)

    events = sorted(events, key=lambda e: e['cycles'])
    if not events:
        logger.info("  No trace events recorded.")
        return

    min_cycles = events[0]['cycles']

    logger.info("Cycles      %-30s" % "EventDetails" + " ".join(f"T{i:<2}" for i in range(10)) + " HMX")
    logger.info("-" * 100)

    thread_stacks = [[] for _ in range(11)]

    for e in events:
        t = e['thread']
        if t < 0 or t > 10:
            continue

        if e['cycles'] >= min_cycles:
            rel_cycles = e['cycles'] - min_cycles
        else:
            rel_cycles = (e['cycles'] + 0x100000000) - min_cycles

        state = e['state']
        evt_type = e['event']

        # Determine char representing the event
        norm_evt = normalize_event_name(evt_type)
        char = '?'
        if norm_evt == 'V-COMP':
            char = 'V'
        elif norm_evt == 'M-COMP':
            char = 'H'
        elif norm_evt == 'A-QUANT':
            char = 'Q'
        elif norm_evt == 'A-PREP':
            char = 'A'
        elif norm_evt == 'Q-PREP':
            char = 'q'
        elif norm_evt == 'K-PREP':
            char = 'k'
        elif norm_evt == 'V-PREP':
            char = 'v'
        elif norm_evt == 'W-DEQUANT':
            char = 'D'
        elif norm_evt == 'O-PROC':
            char = 'O'
        elif norm_evt == 'W-PREP':
            char = 'P'
        elif norm_evt == 'DMA':
            char = 'M'

        if state == 'start':
            thread_stacks[t].append(char)
        elif state == 'stop':
            if thread_stacks[t]:
                if thread_stacks[t][-1] == char:
                    thread_stacks[t].pop()
                elif char in thread_stacks[t]:
                    thread_stacks[t].remove(char)
                else:
                    thread_stacks[t].pop()

        cols = []
        for i in range(11):
            if thread_stacks[i]:
                cols.append(f"[{thread_stacks[i][-1]}]")
            else:
                cols.append(" | ")

        evt_desc = f"T{t}: {evt_type} {state} ({e['info']})"
        logger.info(f"{rel_cycles:10d}  %-30s" % evt_desc + " ".join(cols[:10]) + "  " + cols[10])
    logger.info("-" * 100)


def print_ascii_summary(op_name, dims, types, usec, cycles, events, evt_val=None):
    evt_str = ""
    if evt_val:
        evt_str = " - evt [" + ",".join(str(x) for x in evt_val) + "]"
    logger.info("=" * 100)
    logger.info(f"{op_name} ({dims} : {types}) - {usec} usec {cycles} cycles{evt_str}")
    logger.info("=" * 100)

    events = sorted(events, key=lambda e: e['cycles'])
    if not events:
        logger.info("  No trace events recorded.")
        return

    active_starts = {}
    thread_totals = defaultdict(lambda: defaultdict(int))

    for e in events:
        t = e['thread']
        evt = e['event']
        info = e['info']
        cyc = e['cycles']
        state = e['state']

        key = (t, evt, info)
        if state == 'start':
            active_starts[key] = cyc
        elif state == 'stop':
            if key in active_starts:
                start_cyc = active_starts[key]
                del active_starts[key]

                if cyc >= start_cyc:
                    dur = cyc - start_cyc
                else:
                    dur = (cyc + 0x100000000) - start_cyc

                norm_evt = normalize_event_name(evt)
                thread_totals[t][norm_evt] += dur

    for t in sorted(thread_totals.keys()):
        thread_name = f"Thread {t} (HVX)" if t != 10 else "Thread 10 (HMX)"
        sorted_evts = sorted(thread_totals[t].items(), key=lambda item: item[0])

        evt_strs = []
        for evt, dur in sorted_evts:
            pct = (dur / cycles * 100) if cycles > 0 else 0
            evt_strs.append(f"{evt} {dur} ({pct:.1f}%)")

        logger.info(f"  {thread_name:<16}: " + " | ".join(evt_strs))


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

        avg_usec_val = statistics.mean(usecs)
        count_val = len(group_ops)
        tot_usec_val = avg_usec_val * count_val

        group_stats.append({
            'op':               name,
            'dims':             dims,
            'dtypes':           types,
            'count':            str(count_val),
            'max_usec':         str(max(usecs)),
            'avg_usec':         f"{avg_usec_val:.2f}",
            'tot_usec':         f"{tot_usec_val:.2f}",
            'max_cycles':       str(max(cycles)),
            'avg_cycles':       f"{statistics.mean(cycles):.2f}",
            'max_pmu':          str(max(pmu_vals)) if pmu_vals else "0",
            'avg_pmu':          f"{statistics.mean(pmu_vals):.2f}" if pmu_vals else "0.00",
            # Numeric values for accurate sorting
            '_sort_count':      count_val,
            '_sort_max_usec':   max(usecs),
            '_sort_avg_usec':   avg_usec_val,
            '_sort_tot_usec':   tot_usec_val,
            '_sort_max_cycles': max(cycles),
            '_sort_avg_cycles': statistics.mean(cycles),
            '_sort_max_pmu':    max(pmu_vals) if pmu_vals else 0,
            '_sort_avg_pmu':    statistics.mean(pmu_vals) if pmu_vals else 0
        })

    # Sorting logic
    actual_sort_key = COL_MAP[sort_col][2]
    is_numeric    = actual_sort_key.startswith("_") or actual_sort_key == "count"
    sorted_groups = sorted(group_stats, key=lambda x: x[actual_sort_key], reverse=is_numeric)[:top_n]

    # Define initial column order
    active_cols = ["op", "dims", "dtypes"]
    if pmu_name:
        active_cols += ["max-pmu", "avg-pmu"]
    active_cols += ["tot-usec", "avg-usec", "avg-cycles", "max-usec", "max-cycles", "count"]

    final_headers, final_keys, final_widths = [], [], []

    for col_name in active_cols:
        data_key, header_text, _ = COL_MAP[col_name]
        if "pmu" in col_name and pmu_name:
            header_text = header_text.replace("PMU", pmu_name)

        natural_width = max([len(str(row[data_key])) for row in sorted_groups] + [len(header_text)])
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
            val = str(group[key])
            if len(val) > final_widths[i]:
                val = val[:final_widths[i] - 3] + "..."
            row_vals.append(f"{val:<{final_widths[i]}}")
        logger.info("| " + " | ".join(row_vals) + " |")


def main():
    parser = argparse.ArgumentParser(description="Post-process Op profile info.")
    parser.add_argument("logfile")
    parser.add_argument("-n", "--top", type=int, default=100)
    parser.add_argument("--sort", type=str, default="tot-usec", choices=list(COL_MAP.keys()))
    parser.add_argument("--pmu-index", type=int)
    parser.add_argument("--pmu-name", type=str)
    parser.add_argument("--width", action='append', default=['dims:40'], help="Override column width, e.g. --width dims:50")
    parser.add_argument("--timeline", type=str, nargs='?', const='summary', choices=["summary", "diagram"],
                        help="Output ASCII art event summary or timing diagram (default: summary)")
    parser.add_argument("--filter", type=str, help="Regex filter matching against the original profile-op line")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--head", type=int, help="Limit to first N ops")
    group.add_argument("--tail", type=int, help="Limit to last N ops")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

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

    if args.filter:
        try:
            filter_re = re.compile(args.filter)
        except re.error as e:
            logger.error(f"Invalid regex filter: {e}")
            sys.exit(1)
        ops = [op for op in ops if filter_re.search(op['op_text'])]

    if args.head is not None:
        ops = ops[:args.head]
    elif args.tail is not None:
        ops = ops[-args.tail:]

    if args.timeline:
        logger.info(f"\n# ASCII Timing {args.timeline.capitalize()}\n")
        for op in ops:
            if args.timeline == "summary":
                print_ascii_summary(op['name'], op['dims'], op['types'], op['usec'], op['cycles'], op['trace_events'], op.get('evt_val'))
            elif args.timeline == "diagram":
                print_ascii_timeline(op['name'], op['dims'], op['types'], op['usec'], op['cycles'], op['trace_events'], op.get('evt_val'))
    else:
        generate_report(ops, args.top, overrides, args.sort, pmu_name=final_pmu_name)


if __name__ == "__main__":
    main()
