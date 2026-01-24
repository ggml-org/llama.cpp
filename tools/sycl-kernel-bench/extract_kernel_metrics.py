#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys


def _parse_value(raw: str):
    val = raw.strip()
    if not val:
        return ""
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [int(x.strip()) for x in inner.split(",")]
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


def parse_ze_info(path: str):
    kernels = {}
    current = None
    in_exec = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("- name:"):
                name = stripped.split(":", 1)[1].strip()
                current = {"name": name, "execution_env": {}}
                kernels[name] = current
                in_exec = False
                continue
            if stripped.startswith("execution_env:"):
                in_exec = True
                continue
            if stripped.startswith("payload_arguments:") or stripped.startswith("per_thread_payload_arguments:"):
                in_exec = False
                continue
            if current is None or not in_exec:
                continue
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                current["execution_env"][key.strip()] = _parse_value(value)
    return kernels


def find_ze_info(dump_dir: str):
    for root, _, files in os.walk(dump_dir):
        for name in files:
            if name == ".ze_info" or name.endswith(".ze_info"):
                return os.path.join(root, name)
    return None


def find_asm_for_kernel(dump_dir: str, kernel_name: str):
    candidates = []
    pattern = re.compile(re.escape(kernel_name))
    for root, _, files in os.walk(dump_dir):
        for name in files:
            if not name.endswith(".asm"):
                continue
            if pattern.search(name):
                candidates.append(os.path.join(root, name))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        candidates.sort()
        return candidates[0]
    return None


def count_instr(path: str):
    if not path or not os.path.exists(path):
        return {"dpas": 0, "send": 0}
    dpas = 0
    send = 0
    dpas_re = re.compile(r"\bdpas\b")
    send_re = re.compile(r"\bsend")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if dpas_re.search(line):
                dpas += 1
            if send_re.search(line):
                send += 1
    return {"dpas": dpas, "send": send}


def summarize_kernel(dump_dir: str, kernel: dict):
    name = kernel["name"]
    asm_path = find_asm_for_kernel(dump_dir, name)
    counts = count_instr(asm_path)
    dpas = counts["dpas"]
    send = counts["send"]
    send_per_dpas = (send / dpas) if dpas else 0.0
    exec_env = kernel.get("execution_env", {})
    return {
        "kernel": name,
        "asm": asm_path or "",
        "dpas": dpas,
        "send": send,
        "send_per_dpas": send_per_dpas,
        "execution_env": exec_env,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract DPAS/send counts and ze_info execution_env.")
    parser.add_argument("--dump-dir", required=True, help="SYCL_DUMP_DIR containing .ze_info and .asm files")
    parser.add_argument("--kernel", default="", help="Kernel name to inspect (exact match)")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args()

    if not os.path.isdir(args.dump_dir):
        print(f"Missing dump dir: {args.dump_dir}", file=sys.stderr)
        return 1

    ze_info = find_ze_info(args.dump_dir)
    if not ze_info:
        print("Missing .ze_info in dump dir.", file=sys.stderr)
        return 2

    kernels = parse_ze_info(ze_info)
    if args.kernel:
        kernel = kernels.get(args.kernel)
        if not kernel:
            print(f"Kernel not found in .ze_info: {args.kernel}", file=sys.stderr)
            return 3
        summary = summarize_kernel(args.dump_dir, kernel)
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"kernel: {summary['kernel']}")
            print(f"asm: {summary['asm']}")
            print(f"dpas: {summary['dpas']}")
            print(f"send: {summary['send']}")
            print(f"send_per_dpas: {summary['send_per_dpas']:.3f}")
            print(f"execution_env: {summary['execution_env']}")
        return 0

    summaries = [summarize_kernel(args.dump_dir, k) for k in kernels.values()]
    summaries.sort(key=lambda s: s["kernel"])
    if args.json:
        print(json.dumps(summaries, indent=2))
        return 0
    for summary in summaries:
        print(f"{summary['kernel']}: dpas={summary['dpas']} send={summary['send']} "
              f"send_per_dpas={summary['send_per_dpas']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
