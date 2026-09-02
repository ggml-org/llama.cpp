#!/usr/bin/env python3

import sys
import os
import re
import argparse
import statistics
import logging
import bisect
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
    r"trace-evt\s+(?P<event>[A-Z_0-9\-]+):\s+thread\s+(?P<thread>\d+)\s+info\s+(?P<info>\d+)\s+(?P<state>start|stop)\s+(?P<cycles>\d+)"
)

device_pattern = re.compile(r"\b(HTP\d+(?::\d+)?)\s+(?:profile-op|trace-evt)\b")


def extract_device(line):
    m = device_pattern.search(line)
    if m:
        return m.group(1)
    return "HTP0"


def device_matches(record_device, target_device):
    targets = [t.strip() for t in target_device.split(',')]
    for target in targets:
        if record_device == target:
            return True
        if record_device.startswith(target + ":"):
            return True
    return False


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
    def __init__(self, initial_val=None):
        if initial_val is not None:
            self.last_raw = initial_val & 0xFFFFFFFF
            self.high_part = initial_val & 0xFFFFFFFF00000000
        else:
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


def parse_log(file_path, pmu_index=None, limit=None, device_filter=None, op_filter_re=None):
    try:
        if file_path != "-":
            f = open(file_path, 'r', encoding='utf-8', errors='ignore')
        else:
            f = os.fdopen(0, 'r', encoding='utf-8', errors='ignore')
    except FileNotFoundError:
        logger.error(f"file '{file_path}' not found.")
        sys.exit(1)

    all_ops: List[Dict[str, Any]] = []
    all_traces: List[Dict[str, Any]] = []
    current_op: Optional[Dict[str, Any]] = None
    ops_count_per_device = {}
    if device_filter is not None:
        for target in device_filter.split(','):
            ops_count_per_device[target.strip()] = 0
    limit_reached = False

    timestamp_pattern = re.compile(r"(?P<min>\d+)\.(?P<sec>\d+)\.(?P<ms>\d+)\.(?P<us>\d+)\s+[A-Z]\s+")
    unwrappers = {}
    last_batch_start = {}
    trace_unwrappers = {}

    for line in f:
        if "profile-op" not in line and "trace-evt" not in line:
            continue

        ts_match = timestamp_pattern.search(line)
        abs_usec = 0
        if ts_match:
            abs_usec = (
                (int(ts_match.group('min')) * 60 + int(ts_match.group('sec'))) * 1000000
                + int(ts_match.group('ms')) * 1000
                + int(ts_match.group('us'))
            )

        device = extract_device(line)

        idx = line.find("profile-op")
        if idx != -1 and "|" in line[idx:]:
            parts = [p.strip() for p in line[idx:].split("|")]
            prefix = parts[0]
            prefix_match = re.search(r"profile-op\s+(?P<op_name>[A-Z_0-9+]+)", prefix)
            if not prefix_match:
                continue

            names = parts[1]
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
                names = ""
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

            evt_val = None
            if types.startswith("evt-cnt "):
                try:
                    evt_val = [int(x.strip()) for x in types[8:].split(',')]
                except ValueError:
                    evt_val = None

            cycles_start_raw = op_match.group('start')
            unwrapped_cycles_start = None
            if op_name == "OPBATCH":
                if cycles_start_raw:
                    unwrapped_cycles_start = int(cycles_start_raw)
                    unwrappers[device] = CycleUnwrapper(unwrapped_cycles_start)
                    last_batch_start[device] = unwrapped_cycles_start
                    for k in list(trace_unwrappers.keys()):
                        if k[0] == device:
                            del trace_unwrappers[k]
            else:
                if cycles_start_raw:
                    device_unwrapper = unwrappers.get(device)
                    if device_unwrapper is not None:
                        unwrapped_cycles_start = device_unwrapper.unwrap(int(cycles_start_raw))

            op_text = re.sub(r"^profile-op\s+", "", line[idx:]).strip() if idx != -1 else line.strip()

            current_op = {
                'name':         op_name,
                'names':        names,
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
                'trace_events': [],
                'device':       device
            }
            all_ops.append(current_op)

            # Check if matching early exit criteria
            matched = False
            matched_target = None
            if device_filter is not None:
                targets = [t.strip() for t in device_filter.split(',')]
                for target in targets:
                    if device == target or device.startswith(target + ":"):
                        matched = True
                        matched_target = target
                        break
            else:
                matched = True
                matched_target = device

            if op_filter_re is not None and not op_filter_re.search(op_text):
                matched = False

            if matched:
                if matched_target not in ops_count_per_device:
                    ops_count_per_device[matched_target] = 0
                ops_count_per_device[matched_target] += 1

            if limit is not None and len(ops_count_per_device) > 0 and all(count >= limit for count in ops_count_per_device.values()):
                limit_reached = True

            if limit_reached and op_name == "OPBATCH":
                break
            continue

        trace_match = trace_pattern.search(line)
        if trace_match:
            thread = int(trace_match.group('thread'))
            raw_cyc = int(trace_match.group('cycles'))
            unwrapped_cyc = None
            th_key = (device, thread)
            if th_key not in trace_unwrappers:
                batch_start = last_batch_start.get(device)
                trace_unwrappers[th_key] = CycleUnwrapper(batch_start)
            unwrapped_cyc = trace_unwrappers[th_key].unwrap(raw_cyc)
            all_traces.append({
                'thread': thread,
                'event':  trace_match.group('event'),
                'info':   int(trace_match.group('info')),
                'cycles': raw_cyc,
                'unwrapped_cycles': unwrapped_cyc,
                'state':  trace_match.group('state'),
                'device': device
            })

    f.close()

    # Assign start/end cycles to all ops
    for op in all_ops:
        op['start_cycles'] = op['unwrapped_cycles_start']
        op['end_cycles'] = op['start_cycles'] + op['cycles'] if op['start_cycles'] is not None else None

    # Group ops by device
    valid_ops_by_dev = defaultdict(list)
    for op in all_ops:
        if op['start_cycles'] is not None and op['end_cycles'] is not None:
            valid_ops_by_dev[op['device']].append(op)

    # Group trace events by device
    traces_by_dev = defaultdict(list)
    for e in all_traces:
        if e['unwrapped_cycles'] is not None:
            traces_by_dev[e['device']].append(e)

    for device, dev_ops in valid_ops_by_dev.items():
        opbatch_ops = [op for op in dev_ops if op['name'] == "OPBATCH"]
        other_ops = [op for op in dev_ops if op['name'] != "OPBATCH"]

        opbatch_ops.sort(key=lambda op: op['start_cycles'])
        other_ops.sort(key=lambda op: op['start_cycles'])

        opbatch_starts = [op['start_cycles'] for op in opbatch_ops]
        other_starts = [op['start_cycles'] for op in other_ops]

        dev_traces = traces_by_dev.get(device, [])
        for e in dev_traces:
            cyc = e['unwrapped_cycles']

            # Map to OPBATCH
            idx = bisect.bisect_right(opbatch_starts, cyc) - 1
            if idx >= 0:
                op = opbatch_ops[idx]
                if op['start_cycles'] <= cyc <= op['end_cycles']:
                    op['trace_events'].append(e)

            # Map to other ops
            idx = bisect.bisect_right(other_starts, cyc) - 1
            if idx >= 0:
                op = other_ops[idx]
                if op['start_cycles'] <= cyc <= op['end_cycles']:
                    op['trace_events'].append(e)

    return all_ops


def print_bubbles_timeline(op):
    op_name = op['name']
    dims = op['dims']
    types = op['types']
    usec = op['usec']
    cycles = op['cycles']
    events = op['trace_events']
    logger.info("=" * 100)
    logger.info(f"{op_name} ({dims} : {types}) - {usec} usec {cycles} cycles")
    logger.info("=" * 100)

    if not events:
        logger.info("  No trace events recorded.")
        return

    # Identify start and end cycles for this operator
    op_start = op['start_cycles']
    op_end = op['end_cycles']
    if op_start is None or op_end is None:
        logger.info("  Cannot analyze bubbles: missing start/end cycle counts.")
        return

    batch_duration = op_end - op_start
    if batch_duration <= 0:
        logger.info("  Cannot analyze bubbles: batch duration is 0.")
        return

    # Group events by (thread, track_type)
    tracks = defaultdict(list)
    for e in events:
        t = e['thread']
        is_dma = (normalize_event_name(e['event']) == 'DMA')
        track_type = 'dma' if is_dma else 'compute'
        tracks[(t, track_type)].append(e)

    active_threads = sorted(list(set(t for (t, track_type) in tracks.keys())))
    if not active_threads:
        logger.info("  No active threads in trace.")
        return

    bubble_threshold = 10000  # 10k cycles

    thread_stats = {}
    for t in active_threads:
        thread_stats[t] = {
            'compute_idle_cycles': batch_duration,
            'compute_idle_pct': 100.0,
            'compute_bubbles': [],

            'dma_idle_cycles': batch_duration,
            'dma_idle_pct': 100.0,
            'dma_bubbles': []
        }

    total_compute_idle_pct = 0.0
    total_dma_idle_pct = 0.0

    for t in active_threads:
        for track_type in ['compute', 'dma']:
            key = (t, track_type)
            track_events = tracks.get(key, [])

            if not track_events:
                gaps = [(op_start, op_end)]
                idle_cycles = batch_duration
            else:
                track_events = sorted(track_events, key=lambda e: e.get('unwrapped_cycles') or e['cycles'])

                active_intervals = []
                active_count = 0
                curr_start = None

                for e in track_events:
                    cyc = e.get('unwrapped_cycles') or e['cycles']
                    cyc = max(op_start, min(op_end, cyc))
                    state = e['state']

                    if state == 'start':
                        if active_count == 0:
                            curr_start = cyc
                        active_count += 1
                    elif state == 'stop':
                        if active_count > 0:
                            active_count -= 1
                            if active_count == 0:
                                active_intervals.append((curr_start, cyc))
                        else:
                            active_intervals.append((op_start, cyc))

                if active_count > 0 and curr_start is not None:
                    active_intervals.append((curr_start, op_end))

                # Merge intervals
                active_intervals.sort(key=lambda x: x[0])
                merged_intervals = []
                for start, end in active_intervals:
                    if not merged_intervals:
                        merged_intervals.append([start, end])
                    else:
                        last_start, last_end = merged_intervals[-1]
                        if start <= last_end:
                            merged_intervals[-1][1] = max(last_end, end)
                        else:
                            merged_intervals.append([start, end])

                # Calculate gaps
                gaps = []
                curr_time = op_start
                for start, end in merged_intervals:
                    if start > curr_time:
                        gaps.append((curr_time, start))
                    curr_time = max(curr_time, end)
                if curr_time < op_end:
                    gaps.append((curr_time, op_end))

                idle_cycles = sum(end - start for start, end in gaps)

            idle_pct = (idle_cycles / batch_duration) * 100.0

            bubbles = []
            for start, end in gaps:
                dur = end - start
                if dur >= bubble_threshold:
                    bubbles.append((start, end, dur))

            if track_type == 'compute':
                thread_stats[t]['compute_idle_cycles'] = idle_cycles
                thread_stats[t]['compute_idle_pct'] = idle_pct
                thread_stats[t]['compute_bubbles'] = bubbles
                total_compute_idle_pct += idle_pct
            else:
                thread_stats[t]['dma_idle_cycles'] = idle_cycles
                thread_stats[t]['dma_idle_pct'] = idle_pct
                thread_stats[t]['dma_bubbles'] = bubbles
                total_dma_idle_pct += idle_pct

    avg_compute_idle = total_compute_idle_pct / len(active_threads)
    avg_dma_idle = total_dma_idle_pct / len(active_threads)

    logger.info("  Combined Idle Statistics:")
    logger.info(f"    Active Threads   : {', '.join(str(t) for t in active_threads)}")
    logger.info(f"    Avg Thread Compute IDLE : {avg_compute_idle:.1f}%")
    logger.info(f"    Avg Thread DMA IDLE     : {avg_dma_idle:.1f}%")
    logger.info("-" * 100)

    logger.info("  Per-Thread Idle Analysis:")
    for t in active_threads:
        stats = thread_stats[t]
        thread_name = f"Thread {t:<2} (HVX)" if t != 10 else "Thread 10 (HMX)"
        logger.info(f"    {thread_name} -> Compute Idle: {stats['compute_idle_pct']:.1f}% | DMA Idle: {stats['dma_idle_pct']:.1f}%")

    all_bubbles = []
    for t in active_threads:
        stats = thread_stats[t]
        for start, end, dur in stats['compute_bubbles']:
            pct = (dur / batch_duration) * 100.0
            all_bubbles.append((dur, f"Thread {t} Compute: bubble of {dur} cycles ({pct:.1f}%) at {start - op_start} to {end - op_start}"))
        for start, end, dur in stats['dma_bubbles']:
            pct = (dur / batch_duration) * 100.0
            all_bubbles.append((dur, f"Thread {t} DMA    : bubble of {dur} cycles ({pct:.1f}%) at {start - op_start} to {end - op_start}"))

    if all_bubbles:
        logger.info("-" * 100)
        logger.info(f"  Significant Bubbles (>= {bubble_threshold} cycles):")
        all_bubbles.sort(key=lambda x: x[0], reverse=True)
        for dur, desc in all_bubbles[:15]:
            logger.info(f"    {desc}")
    else:
        logger.info("-" * 100)
        logger.info(f"  No significant bubbles detected (all idle gaps < {bubble_threshold} cycles).")


def print_ascii_summary(op_name, dims, types, usec, cycles, events):
    logger.info("=" * 100)
    logger.info(f"{op_name} ({dims} : {types}) - {usec} usec {cycles} cycles")
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
    parser.add_argument("--timeline", type=str, nargs='?', const='summary', choices=["summary", "bubbles"],
                        help="Output ASCII art event summary or thread idle bubble analysis (default: summary)")
    parser.add_argument("--filter", type=str, help="Regex filter matching against the original profile-op line")
    parser.add_argument("--device", type=str, help="Device to filter by (e.g. HTP0, HTP0:0) or 'split' to generate separate reports per device")

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

    op_filter_re = None
    if args.filter:
        try:
            op_filter_re = re.compile(args.filter)
        except re.error as e:
            logger.error(f"Invalid regex filter: {e}")
            sys.exit(1)

    limit = args.head if args.head is not None else None
    device_filter = args.device if (args.device and args.device != "split") else None
    ops = parse_log(args.logfile, pmu_index=args.pmu_index, limit=limit, device_filter=device_filter, op_filter_re=op_filter_re)

    if args.device and args.device != "split":
        ops = [op for op in ops if device_matches(op['device'], args.device)]

    if args.device == "split":
        unique_devices = sorted(list(set(op['device'] for op in ops)))
        for dev in unique_devices:
            dev_ops = [op for op in ops if device_matches(op['device'], dev)]

            if args.filter:
                try:
                    filter_re = re.compile(args.filter)
                except re.error as e:
                    logger.error(f"Invalid regex filter: {e}")
                    sys.exit(1)
                dev_ops = [op for op in dev_ops if filter_re.search(op['op_text'])]

            if args.head is not None:
                dev_ops = dev_ops[:args.head]
            elif args.tail is not None:
                dev_ops = dev_ops[-args.tail:]

            logger.info("\n=========================================")
            logger.info(f" Device: {dev}")
            logger.info("=========================================")

            if args.timeline:
                for op in dev_ops:
                    if args.timeline == "summary":
                        print_ascii_summary(op['name'], op['dims'], op['types'], op['usec'], op['cycles'], op['trace_events'])
                    elif args.timeline == "bubbles":
                        print_bubbles_timeline(op)
            else:
                generate_report(dev_ops, args.top, overrides, args.sort, pmu_name=final_pmu_name)
    else:
        if args.filter:
            try:
                filter_re = re.compile(args.filter)
            except re.error as e:
                logger.error(f"Invalid regex filter: {e}")
                sys.exit(1)
            ops = [op for op in ops if filter_re.search(op['op_text'])]

        if args.head is not None or args.tail is not None:
            ops_by_dev = defaultdict(list)
            for op in ops:
                ops_by_dev[op['device']].append(op)

            filtered_ops = []
            for dev in sorted(ops_by_dev.keys()):
                dev_ops = ops_by_dev[dev]
                if args.head is not None:
                    dev_ops = dev_ops[:args.head]
                elif args.tail is not None:
                    dev_ops = dev_ops[-args.tail:]
                filtered_ops.extend(dev_ops)
            ops = filtered_ops

        if args.timeline:
            for op in ops:
                if args.timeline == "summary":
                    print_ascii_summary(op['name'], op['dims'], op['types'], op['usec'], op['cycles'], op['trace_events'])
                elif args.timeline == "bubbles":
                    print_bubbles_timeline(op)
        else:
            generate_report(ops, args.top, overrides, args.sort, pmu_name=final_pmu_name)


if __name__ == "__main__":
    main()
