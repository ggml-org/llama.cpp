#!/usr/bin/env python3

import sys
import os
import re
import argparse
import statistics
import logging
from typing import Any, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger("ggml-hexagon-trace")

op_pattern = re.compile(
    r"profile-op\s+(?P<op_name>[A-Z_0-9+]+):\s+.*?\s+:\s+(?P<dims>[\d:x\s\->!]+)\s+:\s+(?P<types>[a-z\d_\s\->x]+)\s+:\s+(?P<strides>[\d:x\s\->!]+)\s+:\s+(?:op-)?usec\s+(?P<usec>\d+)\s+(?:op-)?cycles\s+(?P<cycles>\d+)(?:\s+start\s+(?P<start>\d+))?(?:\s+mhz\s+(?P<mhz>[\d.]+))?(?:\s+pmu\s+\[(?P<pmu>[\d,\s]+)\])?(?:\s+evt\s+\[(?P<evt>[\d,\s]+)\])?"
)

trace_pattern = re.compile(
    r"trace-op\s+(?P<op_name>[A-Z_0-9+]+):\s+thread\s+(?P<thread>\d+)\s+event\s+(?P<event>[A-Z_0-9\-]+)\s+info\s+(?P<info>\d+)\s+(?P<state>start|stop)\s+(?P<cycles>\d+)"
)


def normalize_event_name(evt_type):
    if evt_type == "HVX_COMP":
        return "V-COMP"
    if evt_type == "HMX_COMP":
        return "M-COMP"
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


def parse_log(file_path):
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
    unwrapper = CycleUnwrapper()
    line_idx = 0

    for line in f:
        line_idx += 1
        op_match = op_pattern.search(line)
        if op_match:
            cycles_start_raw = op_match.group('start')
            unwrapped_cycles_start = None
            if cycles_start_raw:
                unwrapped_cycles_start = unwrapper.unwrap(int(cycles_start_raw))

            idx = line.find("profile-op ")
            op_text = line[idx + 11:].strip() if idx != -1 else line.strip()

            current_op = {
                'name':         op_match.group('op_name'),
                'dims':         op_match.group('dims').strip() if op_match.group('dims') else '',
                'types':        op_match.group('types').strip() if op_match.group('types') else '',
                'strides':      op_match.group('strides').strip() if op_match.group('strides') else '',
                'op_text':      op_text,
                'usec':         int(op_match.group('usec')),
                'cycles':       int(op_match.group('cycles')),
                'cycles_start': int(cycles_start_raw) if cycles_start_raw else None,
                'unwrapped_cycles_start': unwrapped_cycles_start,
                'trace_events': [],
                'line_num':     line_idx
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

# --- Simple protobuf encoder ---


def write_varint(val):
    if val < 0:
        val = (1 << 64) + val
    res = bytearray()
    while True:
        towrite = val & 0x7f
        val >>= 7
        if val > 0:
            res.append(towrite | 0x80)
        else:
            res.append(towrite)
            break
    return bytes(res)


def pb_field(num, wire, data):
    return write_varint((num << 3) | wire) + data


def pb_varint(num, val):
    return pb_field(num, 0, write_varint(val))


def pb_length_delimited(num, data):
    return pb_field(num, 2, write_varint(len(data)) + data)


def pb_string(num, text):
    return pb_length_delimited(num, text.encode('utf-8'))


# Message Encoders
def make_process_descriptor(pid, name):
    return pb_varint(1, pid) + pb_string(6, name)


def make_thread_descriptor(pid, tid, name, sort_index=None):
    payload = pb_varint(1, pid) + pb_varint(2, tid) + pb_string(5, name)
    if sort_index is not None:
        payload += pb_varint(3, sort_index)
    return payload


def make_track_descriptor(uuid, name=None, parent_uuid=None, thread=None, process=None, sibling_merge_behavior=None, child_ordering=None, sibling_order_rank=None):
    payload = pb_varint(1, uuid)
    if name is not None:
        payload += pb_string(2, name)
    if parent_uuid is not None:
        payload += pb_varint(5, parent_uuid)
    if process is not None:
        payload += pb_length_delimited(3, process)
    if thread is not None:
        payload += pb_length_delimited(4, thread)
    if sibling_merge_behavior is not None:
        payload += pb_varint(15, sibling_merge_behavior)
    if child_ordering is not None:
        payload += pb_varint(11, child_ordering)
    if sibling_order_rank is not None:
        payload += pb_varint(12, sibling_order_rank)
    return payload


def make_debug_annotation(name, string_val=None, int_val=None):
    payload = pb_string(10, name)
    if string_val is not None:
        payload += pb_string(6, string_val)
    elif int_val is not None:
        payload += pb_varint(4, int_val)
    return payload


def make_track_event(event_type, track_uuid, name=None, category=None, debug_annotations=None):
    payload = pb_varint(9, event_type)
    payload += pb_varint(11, track_uuid)
    if name is not None:
        payload += pb_string(23, name)
    if category is not None:
        payload += pb_string(22, category)
    if debug_annotations is not None:
        for da in debug_annotations:
            payload += pb_length_delimited(4, da)
    return payload


def make_trace_packet(timestamp, track_event=None, track_descriptor=None, seq_id=1):
    payload = pb_varint(8, timestamp)
    payload += pb_varint(10, seq_id)
    if track_event is not None:
        payload += pb_length_delimited(11, track_event)
    if track_descriptor is not None:
        payload += pb_length_delimited(60, track_descriptor)
    return payload


def write_trace_packet_to_file(f, packet_bytes):
    # Write as field 1 of top-level Trace message
    f.write(pb_length_delimited(1, packet_bytes))

# --- End Protobuf Encoder ---


def generate_perfetto_trace(filtered_ops, output_path):
    if not filtered_ops:
        logger.warning("No operators found after filtering.")
        return

    # Compute average frequency
    frequencies = []
    for op in filtered_ops:
        if op['usec'] > 0 and op['cycles'] > 0:
            frequencies.append(op['cycles'] / op['usec'])
    avg_freq_mhz = statistics.mean(frequencies) if frequencies else 1000.0
    if avg_freq_mhz <= 0:
        avg_freq_mhz = 1000.0

    # Assign start and end cycles to each operator
    for op in filtered_ops:
        op['start_cycles'] = op['unwrapped_cycles_start']
        op['end_cycles'] = op['start_cycles'] + op['cycles']

    global_min_cyc = min(op['start_cycles'] for op in filtered_ops if op['start_cycles'] is not None)

    # Process events
    completed_events = []
    for op in filtered_ops:
        events = op['trace_events']
        if not events:
            continue
        events = sorted(events, key=lambda e: e['unwrapped_cycles'])

        active_starts = {}
        for e in events:
            t = e['thread']
            evt = e['event']
            info = e['info']
            state = e['state']
            cyc = e['unwrapped_cycles']

            key = (t, evt, info)
            if state == 'start':
                active_starts[key] = cyc
            elif state == 'stop':
                if key in active_starts:
                    start_cyc = active_starts[key]
                    del active_starts[key]
                    completed_events.append({
                        'thread': t,
                        'event': evt,
                        'info': info,
                        'start_cyc': start_cyc,
                        'end_cyc': cyc,
                        'op_name': op['name']
                    })

    completed_events.sort(key=lambda e: e['start_cyc'])

    # Convert event times to microseconds and apply clamp rounded to 1ns resolution (3 decimals)
    for e in completed_events:
        start_us = (e['start_cyc'] - global_min_cyc) / avg_freq_mhz
        dur_us = (e['end_cyc'] - e['start_cyc']) / avg_freq_mhz
        e['ts_ns'] = int(round(start_us * 1000))
        e['dur_ns'] = int(round(max(dur_us, 0.1) * 1000))

    # Allocate slots (sub-tracks) to prevent overlaps on same virtual track
    active_slots = defaultdict(list)
    for e in completed_events:
        t = e['thread']
        evt = e['event']
        ts = e['ts_ns']
        dur = e['dur_ns']

        norm_evt = normalize_event_name(evt)
        if norm_evt == "DMA":
            track_key = (t, "DMA")
        elif t == 10:
            track_key = (t, "HMX")
        else:
            track_key = (t, "HVX")

        slots = active_slots[track_key]
        allocated_slot = -1
        for idx, slot_end_ns in enumerate(slots):
            if ts >= slot_end_ns:
                slots[idx] = ts + dur
                allocated_slot = idx
                break
        if allocated_slot == -1:
            slots.append(ts + dur)
            allocated_slot = len(slots) - 1
        e['slot'] = allocated_slot

    # Generate Track IDs and track definitions
    used_tracks = {}
    for e in completed_events:
        t = e['thread']
        evt = e['event']
        slot = e['slot']

        norm_evt = normalize_event_name(evt)
        if norm_evt == "DMA":
            track_evt = "DMA"
            evt_id = 1
        elif t == 10:
            track_evt = "HMX"
            evt_id = 3
        else:
            track_evt = "HVX"
            evt_id = 2

        t_sort = 1 if t == 10 else t + 2
        # Unique UUID for each sub-track
        if t == 10:
            uuid = 20  # HMX thread track UUID
        else:
            uuid = int(t_sort * 1000000 + evt_id * 1000 + slot)
        e['uuid'] = uuid
        used_tracks[uuid] = (t, track_evt, slot)

    with open(output_path, "wb") as f:
        # Define Process with EXPLICIT child sorting
        proc_desc = make_process_descriptor(1, "HTP NPU")
        proc_packet = make_trace_packet(0, track_descriptor=make_track_descriptor(1, process=proc_desc, child_ordering=3))
        write_trace_packet_to_file(f, proc_packet)

        # Define Operators Track (UUID = 2) as a thread track at rank 1, tid 8
        op_thread_desc = make_thread_descriptor(1, 8, "Ops", sort_index=1)
        op_packet = make_trace_packet(0, track_descriptor=make_track_descriptor(2, parent_uuid=1, thread=op_thread_desc))
        write_trace_packet_to_file(f, op_packet)

        # Define HMX Thread Track (UUID = 20) at rank 2, tid 9
        hmx_thread_desc = make_thread_descriptor(1, 9, "HMX", sort_index=2)
        hmx_packet = make_trace_packet(0, track_descriptor=make_track_descriptor(20, parent_uuid=1, thread=hmx_thread_desc))
        write_trace_packet_to_file(f, hmx_packet)

        # Define Thread Tracks (T0, T1, ..., T9)
        unique_threads = sorted(list(set(t for (t, _, _) in used_tracks.values() if t != 10)))
        for t in unique_threads:
            thread_uuid = 10 + t
            thread_name = f"T{t}"
            # Sort order starts from index 3 (T0 -> 3, T1 -> 4, etc.)
            sort_index = 3 + t
            tid = 10 + t
            thread_desc = make_thread_descriptor(1, tid, thread_name, sort_index=sort_index)
            thread_packet = make_trace_packet(0, track_descriptor=make_track_descriptor(
                thread_uuid,
                parent_uuid=1,
                thread=thread_desc,
                sibling_order_rank=sort_index,
                child_ordering=3  # Explicit child sorting for sub-tracks
            ))
            write_trace_packet_to_file(f, thread_packet)

        # Define Track descriptors for sub-tracks parented to thread tracks
        for uuid in sorted(used_tracks.keys()):
            if uuid == 20:
                continue
            t, evt, slot = used_tracks[uuid]
            name = f"T{t} {evt}"
            rank = 0 if evt == "HVX" else 1
            parent_thread_uuid = 10 + t
            # Sibling merge behavior: 1 (SIBLING_MERGE_BEHAVIOR_BY_TRACK_NAME)
            track_desc = make_track_descriptor(
                uuid=uuid,
                name=name,
                parent_uuid=parent_thread_uuid,
                sibling_merge_behavior=1,
                sibling_order_rank=rank
            )
            track_packet = make_trace_packet(0, track_descriptor=track_desc)
            write_trace_packet_to_file(f, track_packet)

        # Emit Operators
        last_op_end_ns = 0
        for op in filtered_ops:
            op_start_ns = int(round(((op['start_cycles'] - global_min_cyc) / avg_freq_mhz) * 1000))
            op_dur_ns = int(round((op['cycles'] / avg_freq_mhz) * 1000))
            if op_start_ns < last_op_end_ns:
                op_start_ns = last_op_end_ns
            clamped_dur = max(op_dur_ns, 100) # Clamp to 100ns (0.1us)

            # Debug annotations for Ops
            debug_annots = []
            if 'line_num' in op:
                debug_annots.append(make_debug_annotation("line", int_val=op['line_num']))
            if 'strides' in op and op['strides']:
                debug_annots.append(make_debug_annotation("strides", string_val=op['strides']))

            # Slice Begin
            evt_begin = make_track_event(1, 2, name=f"{op['name']} ({op['dims']})", category="operator", debug_annotations=debug_annots)
            packet_begin = make_trace_packet(op_start_ns, track_event=evt_begin)
            write_trace_packet_to_file(f, packet_begin)

            # Slice End
            evt_end = make_track_event(2, 2)
            packet_end = make_trace_packet(op_start_ns + clamped_dur, track_event=evt_end)
            write_trace_packet_to_file(f, packet_end)

            last_op_end_ns = op_start_ns + clamped_dur

        # Emit Thread Trace Events
        for e in completed_events:
            norm_name = normalize_event_name(e['event'])
            name = f"DMA {e['info']}" if norm_name == "DMA" else norm_name

            # Slice Begin
            evt_begin = make_track_event(1, e['uuid'], name=name, category="trace")
            packet_begin = make_trace_packet(e['ts_ns'], track_event=evt_begin)
            write_trace_packet_to_file(f, packet_begin)

            # Slice End
            evt_end = make_track_event(2, e['uuid'])
            packet_end = make_trace_packet(e['ts_ns'] + e['dur_ns'], track_event=evt_end)
            write_trace_packet_to_file(f, packet_end)

    logger.info(f"Successfully generated Perfetto trace at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Hexagon Op profile logs to native Perfetto Protobuf traces.")
    parser.add_argument("logfile", help="Path to hex-log profile file")
    parser.add_argument("-o", "--output", default="optrace.perfetto-trace", help="Output trace file path (default: optrace.perfetto-trace)")
    parser.add_argument("--filter", type=str, help="Regex filter matching against the original profile-op line")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--head", type=int, help="Limit to first N ops")
    group.add_argument("--tail", type=int, help="Limit to last N ops")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    ops = parse_log(args.logfile)

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

    generate_perfetto_trace(ops, args.output)


if __name__ == "__main__":
    main()
