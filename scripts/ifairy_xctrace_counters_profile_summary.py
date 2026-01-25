#!/usr/bin/env python3

import argparse
import sys
import xml.etree.ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize xctrace 'counters-profile' XML exports for a single process.\n\n"
            "Typical usage:\n"
            "  xcrun xctrace export --input <trace> "
            "--xpath '/trace-toc/run[@number=\"1\"]/data/table[@schema=\"counters-profile\"]' "
            "> counters_profile.xml\n"
            "  python3 scripts/ifairy_xctrace_counters_profile_summary.py "
            "--process-name llama-bench --events ARM_STALL CORE_ACTIVE_CYCLE ... "
            "< counters_profile.xml\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--process-name",
        default="llama-bench",
        help="Process name to include (default: llama-bench).",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Optional PID filter (overrides --process-name).",
    )
    parser.add_argument(
        "--events",
        nargs="+",
        default=None,
        help=(
            "Names for the pmc-events array in order. If omitted, prints counters as e0..eN.\n"
            "Example: --events ARM_STALL CORE_ACTIVE_CYCLE ARM_L1D_CACHE_LMISS_RD ARM_L1D_CACHE_RD L1D_TLB_MISS"
        ),
    )
    parser.add_argument(
        "--running-only",
        action="store_true",
        default=True,
        help="Only include samples where thread-state is Running (default: true).",
    )
    parser.add_argument(
        "--include-nonrunning",
        dest="running_only",
        action="store_false",
        help="Also include non-Running samples.",
    )
    return parser.parse_args()


def _id_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def main() -> int:
    args = parse_args()

    process_name_by_id: dict[int, str] = {}
    pid_by_process_id: dict[int, int] = {}
    thread_state_fmt_by_id: dict[int, str] = {}
    weight_us_by_id: dict[int, int] = {}

    totals: list[int] | None = None
    samples = 0
    total_weight_us = 0

    # The XML is large; iterparse + periodic root.clear() keeps memory bounded.
    context = ET.iterparse(sys.stdin.buffer, events=("start", "end"))
    try:
        first_event, root = next(context)
        if first_event != "start":
            # Should not happen, but keep a sensible default.
            root = None
    except StopIteration:
        print("empty input", file=sys.stderr)
        return 2

    for event, elem in context:
        if event != "end":
            continue

        tag = elem.tag

        if tag == "process":
            pid_elem = elem.find("pid")
            pid = None
            if pid_elem is not None and pid_elem.text is not None:
                try:
                    pid = int(pid_elem.text)
                except ValueError:
                    pid = None

            elem_id = _id_int(elem.get("id"))
            if elem_id is not None:
                fmt = elem.get("fmt")
                if fmt is not None:
                    # "llama-bench (22103)" -> "llama-bench"
                    name = fmt.split(" (", 1)[0]
                    process_name_by_id[elem_id] = name
                if pid is not None:
                    pid_by_process_id[elem_id] = pid

        elif tag == "thread-state":
            elem_id = _id_int(elem.get("id"))
            if elem_id is not None:
                fmt = elem.get("fmt") or (elem.text or "")
                thread_state_fmt_by_id[elem_id] = fmt

        elif tag == "weight":
            elem_id = _id_int(elem.get("id"))
            if elem_id is not None and elem.text is not None:
                try:
                    weight_us_by_id[elem_id] = int(elem.text)
                except ValueError:
                    pass

        elif tag == "row":
            process_elem = elem.find("process")
            if process_elem is None:
                if root is not None:
                    root.clear()
                continue

            process_id = _id_int(process_elem.get("ref")) or _id_int(process_elem.get("id"))
            if process_id is None:
                if root is not None:
                    root.clear()
                continue

            pid = pid_by_process_id.get(process_id)
            name = process_name_by_id.get(process_id)

            if args.pid is not None:
                if pid != args.pid:
                    if root is not None:
                        root.clear()
                    continue
            else:
                if name != args.process_name:
                    if root is not None:
                        root.clear()
                    continue

            state_elem = elem.find("thread-state")
            if state_elem is None:
                if root is not None:
                    root.clear()
                continue

            state = state_elem.get("fmt")
            if state is None:
                ref_id = _id_int(state_elem.get("ref"))
                if ref_id is not None:
                    state = thread_state_fmt_by_id.get(ref_id)
            if state is None:
                state = state_elem.text

            if args.running_only and state != "Running":
                if root is not None:
                    root.clear()
                continue

            pmc_elem = elem.find("pmc-events")
            if pmc_elem is None or pmc_elem.text is None:
                if root is not None:
                    root.clear()
                continue

            parts = pmc_elem.text.strip().split()
            if not parts:
                if root is not None:
                    root.clear()
                continue

            values: list[int] = []
            ok = True
            for part in parts:
                try:
                    values.append(int(part))
                except ValueError:
                    ok = False
                    break
            if not ok:
                if root is not None:
                    root.clear()
                continue

            if totals is None:
                totals = [0] * len(values)
            if len(values) != len(totals):
                if root is not None:
                    root.clear()
                continue

            for i, v in enumerate(values):
                totals[i] += v

            weight_elem = elem.find("weight")
            if weight_elem is not None:
                w = None
                if weight_elem.text is not None:
                    try:
                        w = int(weight_elem.text)
                    except ValueError:
                        w = None
                if w is None:
                    ref_id = _id_int(weight_elem.get("ref"))
                    if ref_id is not None:
                        w = weight_us_by_id.get(ref_id)
                if w is not None:
                    total_weight_us += w

            samples += 1
            if root is not None:
                root.clear()

    if totals is None:
        print("no samples matched filters", file=sys.stderr)
        return 2

    names = args.events
    if names is None:
        names = [f"e{i}" for i in range(len(totals))]
    if len(names) != len(totals):
        print(
            f"events count mismatch: got {len(names)} names but {len(totals)} counters",
            file=sys.stderr,
        )
        return 2

    total_weight_ms = total_weight_us / 1e6
    print(f"samples: {samples}")
    print(f"total_weight_ms: {total_weight_ms:.3f}")
    for name, value in zip(names, totals):
        print(f"{name}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
