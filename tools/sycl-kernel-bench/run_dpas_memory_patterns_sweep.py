#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from datetime import datetime


def parse_list(value: str):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_command(args, dims, repeat, pattern, ntiles, grf_mode, type_acc):
    dim_m, dim_n, dim_k = dims
    cmd = [
        args.binary,
        "--kernel=dpas_memory_patterns",
        f"--dim_m={dim_m}",
        f"--dim_n={dim_n}",
        f"--dim_k={dim_k}",
        f"--iterations={args.iterations}",
        f"--warmup={args.warmup}",
        f"--output={args.output_format}",
        f"--dpas-type-a={args.type_a}",
        f"--dpas-type-b={args.type_b}",
        f"--dpas-acc={type_acc}",
        f"--dpas-memory={pattern}",
        f"--dpas-grf={grf_mode}",
        f"--dpas-repeat={repeat}",
        f"--dpas-ntiles={ntiles}",
        f"--memory={args.memory}",
    ]
    if args.device is not None:
        cmd.append(f"--device={args.device}")
    return cmd


def parse_dims_list(values):
    dims = []
    for entry in values:
        parts = [int(v) for v in entry.split("x")]
        if len(parts) == 1:
            dims.append((parts[0], parts[0], parts[0]))
        elif len(parts) == 3:
            dims.append(tuple(parts))
        else:
            raise ValueError(f"Invalid dims entry: {entry}")
    return dims


def main():
    parser = argparse.ArgumentParser(description="Sweep dpas_memory_patterns across dims/repeats.")
    parser.add_argument("--binary", default=os.path.join("build", "bin", "sycl-kernel-bench"))
    parser.add_argument("--dims", default="4096,8192,16384",
                        help="Comma-separated dims (N or MxNxK).")
    parser.add_argument("--repeats", default="2,8")
    parser.add_argument("--patterns", default="lsc_streaming,lsc_prefetch,lsc_prefetch2")
    parser.add_argument("--ntiles", default="1,2,4,8")
    parser.add_argument("--grf", default="128,256")
    parser.add_argument("--acc", default="int32,float")
    parser.add_argument("--type-a", default="int8")
    parser.add_argument("--type-b", default="int8")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--memory", default="usm_device")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--output", default="")
    parser.add_argument("--log", default="")
    parser.add_argument("--output-format", default="jsonl", choices=["csv", "json", "jsonl"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dims = parse_dims_list(parse_list(args.dims))
    repeats = [int(v) for v in parse_list(args.repeats)]
    patterns = parse_list(args.patterns)
    ntiles = [int(v) for v in parse_list(args.ntiles)]
    grf_modes = parse_list(args.grf)
    acc_types = parse_list(args.acc)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.output:
        args.output = os.path.join("benchmark_results", f"dpas_memory_patterns_sweep_{stamp}.jsonl")
    if not args.log:
        args.log = os.path.join("benchmark_results", f"dpas_memory_patterns_sweep_{stamp}.log")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    with open(args.log, "w", encoding="utf-8") as logf, open(args.output, "w", encoding="utf-8") as outf:
        for dims_entry in dims:
            for repeat in repeats:
                for pattern in patterns:
                    for grf_mode in grf_modes:
                        for acc in acc_types:
                            for tile in ntiles:
                                cmd = build_command(args, dims_entry, repeat, pattern, tile, grf_mode, acc)
                                logf.write("RUN " + " ".join(cmd) + "\n")
                                logf.flush()
                                if args.dry_run:
                                    continue
                                proc = subprocess.run(cmd, capture_output=True, text=True)
                                logf.write(proc.stdout)
                                logf.write(proc.stderr)
                                logf.flush()
                                json_lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
                                if not json_lines:
                                    raise RuntimeError(f"No JSON output for dims={dims_entry} repeat={repeat} pattern={pattern}")
                                payload = json.loads(json_lines[-1])
                                outf.write(json.dumps(payload) + "\n")
                                outf.flush()

    print(f"Wrote {args.output} and {args.log}")


if __name__ == "__main__":
    main()
