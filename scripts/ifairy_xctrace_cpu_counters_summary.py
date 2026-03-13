#!/usr/bin/env python3
import argparse
import sys
import xml.etree.ElementTree as ET


def parse_metric_aggregation_for_thread(xml_stream):
    start_times: dict[int, int] = {}
    durations: dict[int, int] = {}
    u64s: dict[int, int] = {}
    fdecs: dict[int, float] = {}
    strings: dict[int, str] = {}
    bools: dict[int, bool] = {}

    rows = []

    def get_int(elem: ET.Element, cache: dict[int, int]) -> int:
        ref = elem.get("ref")
        if ref is not None:
            return cache.get(int(ref), 0)
        return int((elem.text or "0").strip() or "0")

    def get_float(elem: ET.Element, cache: dict[int, float]) -> float:
        ref = elem.get("ref")
        if ref is not None:
            return cache.get(int(ref), 0.0)
        return float((elem.text or "0").strip() or "0")

    def get_str(elem: ET.Element, cache: dict[int, str]) -> str:
        ref = elem.get("ref")
        if ref is not None:
            return cache.get(int(ref), "")
        return (elem.text or "").strip()

    def get_bool(elem: ET.Element, cache: dict[int, bool]) -> bool:
        ref = elem.get("ref")
        if ref is not None:
            return cache.get(int(ref), False)
        return ((elem.text or "0").strip() or "0") == "1"

    for event, elem in ET.iterparse(xml_stream, events=("end",)):
        tag = elem.tag

        if tag == "start-time":
            elem_id = elem.get("id")
            if elem_id is not None and (elem.text or "").strip():
                start_times[int(elem_id)] = int(elem.text)

        elif tag == "duration":
            elem_id = elem.get("id")
            if elem_id is not None and (elem.text or "").strip():
                durations[int(elem_id)] = int(elem.text)

        elif tag == "uint64":
            elem_id = elem.get("id")
            if elem_id is not None and (elem.text or "").strip():
                u64s[int(elem_id)] = int(elem.text)

        elif tag == "fixed-decimal":
            elem_id = elem.get("id")
            if elem_id is not None and (elem.text or "").strip():
                fdecs[int(elem_id)] = float(elem.text)

        elif tag == "string":
            elem_id = elem.get("id")
            if elem_id is not None and (elem.text or "").strip():
                strings[int(elem_id)] = (elem.text or "").strip()

        elif tag == "boolean":
            elem_id = elem.get("id")
            if elem_id is not None and (elem.text or "").strip():
                bools[int(elem_id)] = ((elem.text or "0").strip() or "0") == "1"

        elif tag == "row":
            ts_elem = elem.find("./start-time")
            dur_elem = elem.find("./duration")
            u64_elem = elem.find("./uint64")
            fdec_elem = elem.find("./fixed-decimal")
            name_elem = elem.find("./string")
            precise_elem = elem.find("./boolean")

            if ts_elem is None or dur_elem is None or name_elem is None or precise_elem is None:
                elem.clear()
                continue

            metric = get_str(name_elem, strings)
            if not metric:
                elem.clear()
                continue

            ts = get_int(ts_elem, start_times)
            dur = get_int(dur_elem, durations)
            is_precise = get_bool(precise_elem, bools)
            metric_u64 = get_int(u64_elem, u64s) if u64_elem is not None else 0
            metric_f = get_float(fdec_elem, fdecs) if fdec_elem is not None else 0.0

            rows.append((ts, dur, metric, is_precise, metric_u64, metric_f))
            elem.clear()

    return rows


def summarize(rows):
    # Keyed by (timestamp, duration). Each interval has one "cycle" row (u64),
    # and ratio rows (fixed-decimal).
    cycles_by_interval = {}
    ratio_by_interval = {}
    duration_by_interval = {}

    for ts, dur, metric, is_precise, metric_u64, metric_f in rows:
        if not is_precise:
            continue

        key = (ts, dur)
        duration_by_interval[key] = dur

        if metric == "cycle":
            cycles_by_interval[key] = metric_u64
        else:
            ratio_by_interval.setdefault(key, {})[metric] = metric_f

    total_cycles = sum(cycles_by_interval.values())
    total_dur_ns = sum(duration_by_interval[k] for k in cycles_by_interval.keys())

    weighted = {"useful": 0.0, "processing": 0.0, "delivery": 0.0, "discarded": 0.0}
    denom_cycles = 0.0
    for key, cyc in cycles_by_interval.items():
        denom_cycles += float(cyc)
        ratios = ratio_by_interval.get(key, {})
        for name in weighted.keys():
            weighted[name] += float(cyc) * float(ratios.get(name, 0.0))

    avg = {k: (weighted[k] / denom_cycles if denom_cycles else 0.0) for k in weighted.keys()}

    return total_dur_ns, total_cycles, avg


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Summarize xctrace MetricAggregationForThread (CPU Counters) export: total cycles and weighted ratios."
    )
    ap.add_argument("--json", action="store_true", help="print as JSON-like (one line) for easy grep")
    args = ap.parse_args()

    rows = parse_metric_aggregation_for_thread(sys.stdin.buffer)
    total_dur_ns, total_cycles, avg = summarize(rows)

    if args.json:
        print(
            "{"
            f"\"duration_ns\":{total_dur_ns},"
            f"\"cycles\":{total_cycles},"
            f"\"useful\":{avg['useful']:.6f},"
            f"\"processing\":{avg['processing']:.6f},"
            f"\"delivery\":{avg['delivery']:.6f},"
            f"\"discarded\":{avg['discarded']:.6f}"
            "}"
        )
    else:
        print(f"duration_ns: {total_dur_ns}")
        print(f"cycles: {total_cycles}")
        print(f"useful: {avg['useful']:.6f}")
        print(f"processing: {avg['processing']:.6f}")
        print(f"delivery: {avg['delivery']:.6f}")
        print(f"discarded: {avg['discarded']:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
