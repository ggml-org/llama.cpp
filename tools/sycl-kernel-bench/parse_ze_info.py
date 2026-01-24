#!/usr/bin/env python3
import argparse
import json
import os
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


def main():
    parser = argparse.ArgumentParser(description="Parse zebin .ze_info metadata for kernel execution_env.")
    parser.add_argument("--file", required=True, help="Path to .ze_info file")
    parser.add_argument("--kernel", default="", help="Kernel name to extract (exact match)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Missing file: {args.file}", file=sys.stderr)
        return 1

    kernels = parse_ze_info(args.file)
    if args.kernel:
        entry = kernels.get(args.kernel)
        if not entry:
            print(f"Kernel not found: {args.kernel}", file=sys.stderr)
            return 2
        print(json.dumps(entry, indent=2))
        return 0

    print(json.dumps(sorted(kernels.keys()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
