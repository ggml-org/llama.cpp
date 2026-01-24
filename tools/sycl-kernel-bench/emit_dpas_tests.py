#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict


def load_matrix(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def memory_flag(memory_mode: str) -> str:
    if memory_mode == "USM_SHARED":
        return "usm_shared"
    if memory_mode == "BUFFER":
        return "buffer"
    return "usm_device"


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit CLI commands for DPAS device-opt tests.")
    parser.add_argument(
        "--matrix",
        default=os.path.join(os.path.dirname(__file__), "test_matrix.json"),
        help="Path to test_matrix.json",
    )
    parser.add_argument("--binary", default=os.path.join("build", "bin", "sycl-kernel-bench"))
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--output", default="jsonl", choices=["csv", "json", "jsonl"])
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    tests = matrix.get("dpas_tests", [])
    if not tests:
        return 0

    for entry in tests:
        cmd = [
            args.binary,
            f"--kernel={entry['kernel']}",
            f"--dim_m={entry['dim_m']}",
            f"--dim_n={entry['dim_n']}",
            f"--dim_k={entry['dim_k']}",
            f"--memory={memory_flag(entry.get('memory_mode', 'USM_DEVICE'))}",
            f"--output={args.output}",
        ]
        repeat = entry.get("dpas_repeat")
        if repeat is not None:
            cmd.append(f"--dpas-repeat={repeat}")
        if entry.get("dpas_device_opt"):
            cmd.append("--dpas-device-opt")
        if args.device is not None:
            cmd.append(f"--device={args.device}")
        print(" ".join(cmd))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
