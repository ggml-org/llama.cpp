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

logger = logging.getLogger("ggml-hexagon-trace")

op_pattern = re.compile(
    r"profile-op\s+(?P<op_name>[A-Z_0-9+]+):\s+.*?\s+:\s+(?P<dims>[\d:x\s\->!]+)\s+:\s+(?P<types>[a-z\d_\s\->x]+)\s+:\s+(?P<strides>[\d:x\s\->!]+?)\s+:\s+(?:(?P<params>.*?)\s+:\s+)?(?:op-)?usec\s+(?P<usec>\d+)\s+(?:op-)?cycles\s+(?P<cycles>\d+)(?:\s+start\s+(?P<start>\d+))?(?:\s+mhz\s+(?P<mhz>[\d.]+))?(?:\s+pmu\s+\[(?P<pmu>[\d,\s]+)\])?(?:\s+evt\s+\[(?P<evt>[\d,\s]+)\])?"
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


def get_split_output_path(base_path, device_name):
    safe_device = device_name.replace(':', '_')
    root, ext = os.path.splitext(base_path)
    return f"{root}-{safe_device}{ext}"


def normalize_event_name(evt_type, info=0):
    if evt_type == "HVX_COMP":
        return "V-COMP"
    if evt_type == "HMX_COMP":
        return "M-COMP"
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


class DeviceTimeMapper:
    def __init__(self, dev, ops):
        self.dev = dev
        self.batches = []
        for op in ops:
            if op.get('device') == dev and op.get('name') == 'OPBATCH' and op.get('unwrapped_cycles_start') is not None:
                cycles = op.get('cycles', 0)
                usec = op.get('usec', 0)
                start_cyc = op['unwrapped_cycles_start']
                freq = (cycles / usec) if usec > 0 and cycles > 0 else 1000.0
                if freq <= 0:
                    freq = 1000.0
                self.batches.append({
                    'start_cycles': start_cyc,
                    'cycles': cycles,
                    'end_cycles': start_cyc + cycles,
                    'usec': usec,
                    'dur_ns': usec * 1000,
                    'freq_mhz': freq,
                })

        self.batches.sort(key=lambda b: b['start_cycles'])

        for i, b in enumerate(self.batches):
            if i == 0:
                b['start_time_ns'] = 0
            else:
                prev = self.batches[i - 1]
                idle_cyc = max(0, b['start_cycles'] - prev['end_cycles'])
                idle_ns = int(round((idle_cyc / prev['freq_mhz']) * 1000))
                b['start_time_ns'] = prev['start_time_ns'] + prev['dur_ns'] + idle_ns

        self.batch_starts = [b['start_cycles'] for b in self.batches]

        valid_starts = [op['unwrapped_cycles_start'] for op in ops if op.get('device') == dev and op.get('unwrapped_cycles_start') is not None]
        self.min_cyc = min(valid_starts) if valid_starts else 0
        if self.batches:
            self.default_freq = self.batches[0]['freq_mhz']
        else:
            freqs = [op['cycles'] / op['usec'] for op in ops if op.get('device') == dev and op.get('usec', 0) > 0 and op.get('cycles', 0) > 0]
            self.default_freq = statistics.mean(freqs) if freqs else 1000.0

    def get_batch(self, cyc):
        if not self.batches:
            return None
        idx = bisect.bisect_right(self.batch_starts, cyc) - 1
        if idx >= 0:
            return self.batches[idx]
        return self.batches[0]

    def get_freq(self, cyc=None):
        if cyc is not None:
            b = self.get_batch(cyc)
            if b is not None:
                return b['freq_mhz']
        return self.default_freq

    def cycle_to_ns(self, cyc):
        if cyc is None:
            return 0
        b = self.get_batch(cyc)
        if b is not None:
            return b['start_time_ns'] + int(round(((cyc - b['start_cycles']) / b['freq_mhz']) * 1000))
        return int(round(((cyc - self.min_cyc) / self.default_freq) * 1000))

    def dur_cycles_to_ns(self, cyc_start, cyc_dur):
        if cyc_dur is None:
            return 0
        freq = self.get_freq(cyc_start)
        return int(round((cyc_dur / freq) * 1000))


def parse_log(file_path, limit=None, device_filter=None, op_filter_re=None):
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
    unwrappers = {}
    last_batch_start = {}
    trace_unwrappers = {}
    line_idx = 0

    for line in f:
        line_idx += 1
        if "profile-op" not in line and "trace-evt" not in line:
            continue
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
                dims, types, strides, params, timings = parts[2], parts[3], parts[4], parts[5], parts[6]
            elif len(parts) == 6:
                dims, types, strides, params, timings = parts[2], parts[3], parts[4], "", parts[5]
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
                dims = op_match.group('dims').strip() if op_match.group('dims') else ''
                types = op_match.group('types').strip() if op_match.group('types') else ''
                strides = op_match.group('strides').strip() if op_match.group('strides') else ''
                params = op_match.group('params').strip() if ('params' in op_match.groupdict() and op_match.group('params')) else ''
            else:
                op_match = None

        if op_match:
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

            evt_str = None
            if types.startswith("evt-cnt "):
                evt_str = types[8:].strip()

            current_op = {
                'name':         op_name,
                'names':        names,
                'dims':         dims,
                'types':        types,
                'strides':      strides,
                'params':       params,
                'evt':          evt_str,
                'op_text':      op_text,
                'usec':         int(op_match.group('usec')),
                'cycles':       int(op_match.group('cycles')),
                'cycles_start': int(cycles_start_raw) if cycles_start_raw else None,
                'unwrapped_cycles_start': unwrapped_cycles_start,
                'trace_events': [],
                'line_num':     line_idx,
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
                'line_num': line_idx,
                'device': device
            })

    f.close()
    return all_ops, all_traces

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


def generate_perfetto_trace(filtered_ops, trace_events, output_path):
    if not filtered_ops:
        logger.warning("No operators found after filtering.")
        return

    # Assign start and end cycles to each operator
    for op in filtered_ops:
        op['start_cycles'] = op['unwrapped_cycles_start']
        op['end_cycles'] = op['start_cycles'] + op['cycles'] if op['start_cycles'] is not None else None

    # Get list of unique devices present in the operations
    unique_devices = sorted(list(set(op['device'] for op in filtered_ops)))
    device_to_idx = {dev: idx for idx, dev in enumerate(unique_devices)}
    time_mappers = {dev: DeviceTimeMapper(dev, filtered_ops) for dev in unique_devices}

    # Process events
    completed_events = []
    if trace_events:
        trace_events = sorted(trace_events, key=lambda e: e['unwrapped_cycles'])

        one_usec_cycles = {}
        for dev in unique_devices:
            one_usec_cycles[dev] = max(time_mappers[dev].get_freq(), 1.0)

        active_starts = {}
        for e in trace_events:
            t = e['thread']
            evt = e['event']
            info = e['info']
            state = e['state']
            cyc = e['unwrapped_cycles']
            dev = e['device']

            key = (dev, t, evt, info)
            if state == 'start':
                # Handle missing stop (start followed by another start)
                if key in active_starts:
                    prev_e = active_starts[key]
                    completed_events.append({
                        'thread': t,
                        'event': evt,
                        'info': info,
                        'start_cyc': prev_e['unwrapped_cycles'],
                        'end_cyc': prev_e['unwrapped_cycles'] + one_usec_cycles.get(dev, 1000.0),
                        'line_num': prev_e.get('line_num'),
                        'missing_stop': True,
                        'device': dev
                    })
                active_starts[key] = e
            elif state == 'stop':
                if key in active_starts:
                    prev_e = active_starts[key]
                    del active_starts[key]
                    completed_events.append({
                        'thread': t,
                        'event': evt,
                        'info': info,
                        'start_cyc': prev_e['unwrapped_cycles'],
                        'end_cyc': cyc,
                        'line_num': prev_e.get('line_num'),
                        'device': dev
                    })
                else:
                    # Handle missing start (stop without start)
                    completed_events.append({
                        'thread': t,
                        'event': evt,
                        'info': info,
                        'start_cyc': cyc - one_usec_cycles.get(dev, 1000.0),
                        'end_cyc': cyc,
                        'line_num': e.get('line_num'),
                        'missing_start': True,
                        'device': dev
                    })

        # Clear remaining unmatched starts
        for key, prev_e in active_starts.items():
            dev, t, evt, info = key
            completed_events.append({
                'thread': t,
                'event': evt,
                'info': info,
                'start_cyc': prev_e['unwrapped_cycles'],
                'end_cyc': prev_e['unwrapped_cycles'] + one_usec_cycles.get(dev, 1000.0),
                'line_num': prev_e.get('line_num'),
                'missing_stop': True,
                'device': dev
            })

    completed_events.sort(key=lambda e: e['start_cyc'])

    # Convert event times to nanoseconds using per-device / per-batch time mapper
    for e in completed_events:
        dev = e['device']
        tm = time_mappers[dev]
        e['ts_ns'] = tm.cycle_to_ns(e['start_cyc'])
        dur_ns = tm.dur_cycles_to_ns(e['start_cyc'], e['end_cyc'] - e['start_cyc'])
        e['dur_ns'] = max(dur_ns, 100)

    # Allocate slots (sub-tracks) to prevent overlaps on same virtual track
    active_slots = defaultdict(list)
    for e in completed_events:
        t = e['thread']
        evt = e['event']
        ts = e['ts_ns']
        dur = e['dur_ns']
        dev = e['device']

        norm_evt = normalize_event_name(evt, e['info'])
        if norm_evt == "DMA":
            track_key = (dev, t, "DMA")
        elif t == 10:
            track_key = (dev, t, "HMX")
        else:
            track_key = (dev, t, "HVX")

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
        dev = e['device']

        norm_evt = normalize_event_name(evt, e['info'])
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
        dev_idx = device_to_idx[dev]

        # Unique UUID for each sub-track
        if t == 10:
            uuid = dev_idx * 10000000 + 20  # HMX thread track UUID
        else:
            uuid = int(dev_idx * 10000000 + t_sort * 1000000 + evt_id * 1000 + slot)
        e['uuid'] = uuid
        used_tracks[uuid] = (dev, t, track_evt, slot)

    with open(output_path, "wb") as f:
        for dev in unique_devices:
            dev_idx = device_to_idx[dev]
            pid = dev_idx + 1
            proc_uuid = dev_idx * 10000000 + 1

            # Define Process with EXPLICIT child sorting
            proc_name = dev
            proc_desc = make_process_descriptor(pid, proc_name)
            proc_packet = make_trace_packet(0, track_descriptor=make_track_descriptor(proc_uuid, process=proc_desc, child_ordering=3))
            write_trace_packet_to_file(f, proc_packet)

            # Define Operators Track as a thread track
            op_track_uuid = dev_idx * 10000000 + 2
            op_tid = pid * 100 + 8
            op_thread_desc = make_thread_descriptor(pid, op_tid, "Ops", sort_index=1)
            op_packet = make_trace_packet(0, track_descriptor=make_track_descriptor(op_track_uuid, parent_uuid=proc_uuid, thread=op_thread_desc))
            write_trace_packet_to_file(f, op_packet)

            # Define HMX Thread Track at rank 2
            hmx_track_uuid = dev_idx * 10000000 + 20
            hmx_tid = pid * 100 + 9
            hmx_thread_desc = make_thread_descriptor(pid, hmx_tid, "HMX", sort_index=2)
            hmx_packet = make_trace_packet(0, track_descriptor=make_track_descriptor(hmx_track_uuid, parent_uuid=proc_uuid, thread=hmx_thread_desc))
            write_trace_packet_to_file(f, hmx_packet)

            # Define Thread Tracks (T0, T1, ..., T9) for this device
            dev_used_tracks = {uuid: val for uuid, val in used_tracks.items() if val[0] == dev}
            unique_threads = sorted(list(set(t for (_, t, _, _) in dev_used_tracks.values() if t != 10)))
            for t in unique_threads:
                thread_uuid = dev_idx * 10000000 + 10 + t
                thread_name = f"T{t}"
                sort_index = 3 + t
                tid = pid * 100 + 10 + t
                thread_desc = make_thread_descriptor(pid, tid, thread_name, sort_index=sort_index)
                thread_packet = make_trace_packet(0, track_descriptor=make_track_descriptor(
                    thread_uuid,
                    parent_uuid=proc_uuid,
                    thread=thread_desc,
                    sibling_order_rank=sort_index,
                    child_ordering=3  # Explicit child sorting for sub-tracks
                ))
                write_trace_packet_to_file(f, thread_packet)

        # Define Track descriptors for sub-tracks parented to thread tracks
        for uuid in sorted(used_tracks.keys()):
            dev, t, evt, slot = used_tracks[uuid]
            dev_idx = device_to_idx[dev]
            if t == 10:
                continue
            name = f"T{t} {evt}"
            rank = 0 if evt == "HVX" else 1
            parent_thread_uuid = dev_idx * 10000000 + 10 + t
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
        last_op_end_ns = defaultdict(int)
        for op in filtered_ops:
            dev = op['device']
            dev_idx = device_to_idx[dev]
            tm = time_mappers[dev]
            op_start_ns = tm.cycle_to_ns(op['start_cycles'])
            op_dur_ns = tm.dur_cycles_to_ns(op['start_cycles'], op['cycles'])
            if op['name'] != "OPBATCH":
                if op_start_ns < last_op_end_ns[dev]:
                    op_start_ns = last_op_end_ns[dev]
                clamped_dur = max(op_dur_ns, 100) # Clamp to 100ns (0.1us)
                last_op_end_ns[dev] = op_start_ns + clamped_dur
            else:
                clamped_dur = max(op_dur_ns, 100)

            # Debug annotations for Ops
            debug_annots = []
            if 'line_num' in op:
                debug_annots.append(make_debug_annotation("line", int_val=op['line_num']))
            if 'names' in op and op['names'] and op['names'] != '----':
                debug_annots.append(make_debug_annotation("names", string_val=op['names']))
            if 'strides' in op and op['strides'] and op['strides'] != '----':
                debug_annots.append(make_debug_annotation("strides", string_val=op['strides']))
            if 'params' in op and op['params'] and op['params'] != '----':
                debug_annots.append(make_debug_annotation("params", string_val=op['params']))
            if 'evt' in op and op['evt']:
                debug_annots.append(make_debug_annotation("evt", string_val=op['evt']))

            op_track_uuid = dev_idx * 10000000 + 2

            # Slice Begin
            evt_begin = make_track_event(1, op_track_uuid, name=f"{op['name']} ({op['dims']})", category="operator", debug_annotations=debug_annots)
            packet_begin = make_trace_packet(op_start_ns, track_event=evt_begin)
            write_trace_packet_to_file(f, packet_begin)

            # Slice End
            evt_end = make_track_event(2, op_track_uuid)
            packet_end = make_trace_packet(op_start_ns + clamped_dur, track_event=evt_end)
            write_trace_packet_to_file(f, packet_end)

        # Emit Thread Trace Events
        for e in completed_events:
            norm_name = normalize_event_name(e['event'], e['info'])
            if norm_name == "DMA":
                name = f"DMA {e['info']}"
            elif norm_name == "FENCE":
                name = f"FENCE {e['info']}" if e.get('info') is not None and e['info'] != 0 else "FENCE"
            else:
                name = norm_name

            if e.get('missing_start') or e.get('missing_stop'):
                name += "!"

            debug_annots = []
            if 'line_num' in e and e['line_num'] is not None:
                debug_annots.append(make_debug_annotation("line", int_val=e['line_num']))
            if norm_name == "FENCE" and e.get('info') is not None:
                debug_annots.append(make_debug_annotation("seq", int_val=e['info']))
            elif norm_name == "DMA" and e.get('info') is not None:
                debug_annots.append(make_debug_annotation("channel", int_val=e['info']))
            elif e.get('info') is not None and e['info'] != 0:
                debug_annots.append(make_debug_annotation("info", int_val=e['info']))

            if e.get('missing_start'):
                debug_annots.append(make_debug_annotation("missing_start", string_val="true"))
            if e.get('missing_stop'):
                debug_annots.append(make_debug_annotation("missing_stop", string_val="true"))

            # Slice Begin
            evt_begin = make_track_event(1, e['uuid'], name=name, category="trace", debug_annotations=debug_annots if debug_annots else None)
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
    parser.add_argument("--device", type=str, help="Device to filter by (e.g. HTP0, HTP0:0) or 'split' to generate separate files per device")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--head", type=int, help="Limit to first N ops")
    group.add_argument("--tail", type=int, help="Limit to last N ops")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    op_filter_re = None
    if args.filter:
        try:
            op_filter_re = re.compile(args.filter)
        except re.error as e:
            logger.error(f"Invalid regex filter: {e}")
            sys.exit(1)

    limit = args.head if args.head is not None else None
    device_filter = args.device if (args.device and args.device != "split") else None
    ops, traces = parse_log(args.logfile, limit=limit, device_filter=device_filter, op_filter_re=op_filter_re)

    if args.device and args.device != "split":
        ops = [op for op in ops if device_matches(op['device'], args.device)]
        traces = [t for t in traces if device_matches(t['device'], args.device)]

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

    if args.filter or args.head is not None or args.tail is not None:
        # Group valid ranges by device
        valid_ranges_by_dev = defaultdict(list)
        for op in ops:
            start_cyc = op['unwrapped_cycles_start']
            end_cyc = start_cyc + op['cycles'] if start_cyc is not None else None
            if start_cyc is not None and end_cyc is not None:
                valid_ranges_by_dev[op['device']].append((start_cyc, end_cyc))

        for dev in valid_ranges_by_dev:
            valid_ranges_by_dev[dev].sort(key=lambda r: r[0])

        range_starts_by_dev = {dev: [r[0] for r in ranges] for dev, ranges in valid_ranges_by_dev.items()}

        filtered_traces = []
        for e in traces:
            cyc = e['unwrapped_cycles']
            if cyc is None:
                continue
            dev = e['device']
            range_starts = range_starts_by_dev.get(dev)
            if not range_starts:
                continue
            idx = bisect.bisect_right(range_starts, cyc) - 1
            if idx >= 0:
                start, end = valid_ranges_by_dev[dev][idx]
                if start <= cyc <= end:
                    filtered_traces.append(e)
        traces = filtered_traces

    if args.device == "split":
        unique_devices = sorted(list(set(op['device'] for op in ops)))
        for dev in unique_devices:
            dev_ops = [op for op in ops if device_matches(op['device'], dev)]
            dev_traces = [t for t in traces if device_matches(t['device'], dev)]
            out_path = get_split_output_path(args.output, dev)
            generate_perfetto_trace(dev_ops, dev_traces, out_path)
    else:
        generate_perfetto_trace(ops, traces, args.output)


if __name__ == "__main__":
    main()
