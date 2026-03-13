#!/usr/bin/env python3
import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter


def parse_time_profile_leaf(xml_stream) -> tuple[Counter, int]:
    weights: dict[int, int] = {}
    frames: dict[int, str] = {}
    backtrace_leaf: dict[int, str] = {}

    leaf_weight: Counter[str] = Counter()
    leaf_total = 0

    def frame_name(frame_elem: ET.Element) -> str:
        name = frame_elem.get("name")
        ref = frame_elem.get("ref")
        if (not name) and ref is not None:
            name = frames.get(int(ref))
        return name or "<unknown>"

    def leaf_from_backtrace(bt: ET.Element) -> str:
        # The first frame is the leaf in this export.
        f = bt.find("./frame")
        if f is None:
            return "<unknown>"
        return frame_name(f)

    for event, elem in ET.iterparse(xml_stream, events=("end",)):
        tag = elem.tag

        if tag == "weight":
            elem_id = elem.get("id")
            if elem_id is not None and (elem.text or "").strip():
                weights[int(elem_id)] = int(elem.text)

        elif tag == "frame":
            elem_id = elem.get("id")
            name = elem.get("name")
            if elem_id is not None and name:
                frames[int(elem_id)] = name

        elif tag == "backtrace":
            elem_id = elem.get("id")
            if elem_id is not None:
                backtrace_leaf[int(elem_id)] = leaf_from_backtrace(elem)

        elif tag == "row":
            w = elem.find("./weight")
            bt = elem.find("./backtrace")
            if w is None or bt is None:
                elem.clear()
                continue

            w_ref = w.get("ref")
            if w_ref is not None:
                w_val = weights.get(int(w_ref), 0)
            else:
                w_val = int((w.text or "0").strip() or "0")

            bt_ref = bt.get("ref")
            if bt_ref is not None:
                leaf_sym = backtrace_leaf.get(int(bt_ref), "<unknown>")
            else:
                leaf_sym = leaf_from_backtrace(bt)
                bt_id = bt.get("id")
                if bt_id is not None:
                    backtrace_leaf[int(bt_id)] = leaf_sym

            leaf_weight[leaf_sym] += w_val
            leaf_total += w_val

            elem.clear()

    return leaf_weight, leaf_total


def pct(v: int, total: int) -> float:
    return 0.0 if total == 0 else (100.0 * v / total)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize xctrace time-profile leaf CPU time share from XML export.")
    ap.add_argument("--top", type=int, default=20, help="top N leaf symbols to print")
    ap.add_argument("--prefix", default="", help="only count symbols that start with this prefix (optional)")
    args = ap.parse_args()

    leaf_weight, total = parse_time_profile_leaf(sys.stdin.buffer)

    if args.prefix:
        leaf_weight = Counter({k: v for k, v in leaf_weight.items() if k.startswith(args.prefix)})

    print(f"total_ms: {total/1_000_000.0:.3f}")
    for sym, v in leaf_weight.most_common(args.top):
        print(f"{pct(v, total):6.2f}%  {v/1_000_000.0:9.3f} ms  {sym}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

