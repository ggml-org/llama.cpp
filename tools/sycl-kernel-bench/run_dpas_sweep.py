#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from datetime import datetime


def k_per_dpas(type_a: str, type_b: str) -> int:
    bits_a = 8 if type_a == "int8" else 16
    bits_b = 8 if type_b == "int8" else 16
    max_bits = max(bits_a, bits_b)
    max_elems = 32 // max_bits
    ops_per_channel = 8 if max_elems > 8 else (1 if max_elems < 1 else max_elems)
    return 8 * ops_per_channel


def load_configs(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_matrix(cfg: dict):
    shape = cfg["shape"]
    for repeat in cfg["repeats"]:
        for combo in cfg["type_combos"]:
            for pattern in cfg["memory_patterns"]:
                for grf in cfg["grf_modes"]:
                    yield {
                        "name": f"{combo['type_a']}_{combo['type_acc']}_r{repeat}_{pattern}_grf{grf}",
                        "kernel": "dpas_sweep",
                        "repeat": repeat,
                        "type_a": combo["type_a"],
                        "type_b": combo["type_b"],
                        "type_acc": combo["type_acc"],
                        "memory_pattern": pattern,
                        "grf_mode": grf,
                        "m_tiles": shape["m_tiles"],
                        "n_tiles": shape["n_tiles"],
                        "k_tiles": shape["k_tiles"],
                    }


def iter_priority(cfg: dict):
    for entry in cfg.get("priority", []):
        yield entry


def build_command(args, entry, iterations, warmup, memory_mode):
    k_per = k_per_dpas(entry["type_a"], entry["type_b"])
    dim_m = entry["m_tiles"] * entry["repeat"]
    dim_n = entry["n_tiles"] * 16
    dim_k = entry["k_tiles"] * k_per
    cmd = [
        args.binary,
        f"--kernel={entry['kernel']}",
        "--quant=Q4_0",
        "--batch=1",
        f"--dim_m={dim_m}",
        f"--dim_n={dim_n}",
        f"--dim_k={dim_k}",
        f"--iterations={iterations}",
        f"--warmup={warmup}",
        "--output=jsonl",
        f"--dpas-config={entry['name']}",
        f"--dpas-type-a={entry['type_a']}",
        f"--dpas-type-b={entry['type_b']}",
        f"--dpas-acc={entry['type_acc']}",
        f"--dpas-memory={entry['memory_pattern']}",
        f"--dpas-grf={entry['grf_mode']}",
        f"--dpas-repeat={entry['repeat']}",
        f"--memory={memory_mode}",
    ]
    if entry.get("misaligned"):
        cmd.append("--dpas-misaligned")
    if args.device is not None:
        cmd.append(f"--device={args.device}")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run DPAS exploration sweep.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "dpas_configs.json"))
    parser.add_argument("--binary", default=os.path.join("build", "bin", "sycl-kernel-bench"))
    parser.add_argument("--output", default="")
    parser.add_argument("--log", default="")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--mode", choices=["matrix", "priority", "all"], default="all")
    parser.add_argument("--memory", default="usm_device")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_configs(args.config)
    iterations = cfg.get("iterations", 100)
    warmup = cfg.get("warmup", 10)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.output:
        args.output = os.path.join(os.path.dirname(args.config), f"dpas_sweep_{stamp}.jsonl")
    if not args.log:
        args.log = os.path.join(os.path.dirname(args.config), f"dpas_sweep_{stamp}.log")

    entries = []
    if args.mode in ("matrix", "all"):
        entries.extend(list(iter_matrix(cfg)))
    if args.mode in ("priority", "all"):
        entries.extend(list(iter_priority(cfg)))

    with open(args.log, "w", encoding="utf-8") as logf, open(args.output, "w", encoding="utf-8") as outf:
        for entry in entries:
            cmd = build_command(args, entry, iterations, warmup, args.memory)
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
                raise RuntimeError(f"No JSON output for {entry['name']}")
            payload = json.loads(json_lines[-1])
            outf.write(json.dumps(payload) + "\n")
            outf.flush()

    print(f"Wrote {args.output} and {args.log}")


if __name__ == "__main__":
    main()
