# SPDX-License-Identifier: MIT
"""Same-binary baseline/candidate sweep. No benchmark values fabricated in dry-run."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from architecture_common import VERSION, verify_build
QS = (1, 2, 4, 8, 16, 32, 64, 128, 184, 256, 512, 1024, 2048)
RECIPES = ((1, 1, 0), (1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (4, 4, 1))

def cases_for(suite):
    if suite == "smoke":
        return [dict(name="smoke-q13-t2-4", q=13, k=512, hidden=256, experts=8, topk=2,
                     gt=2, dt=4, overlap=1, split_gate=2, split_down=1, model_shape=False),
                dict(name="smoke-q1-t4-1", q=1, k=512, hidden=256, experts=3, topk=1,
                     gt=4, dt=1, overlap=1, split_gate=1, split_down=1, model_shape=False)]
    qs = (1, 13, 184, 512) if suite == "quick" else QS
    return [dict(name=f"q{q:04d}-t{gt}-{dt}-fork{overlap}", q=q, k=2048, hidden=512, experts=256,
                 topk=8, gt=gt, dt=dt, overlap=overlap, split_gate=1, split_down=1, model_shape=True)
            for q in qs for gt, dt, overlap in RECIPES]

def command_for(case, a):
    args = [str(a.exe), "--device", a.device, "--out", str(a.out / case["name"]),
            "--tokens", str(case["q"]), "--topk", str(case["topk"]), "--k", str(case["k"]),
            "--hidden", str(case["hidden"]), "--experts", str(case["experts"]),
            "--candidate", "grouped", "--gate-tile", str(case["gt"]), "--down-tile", str(case["dt"]),
            "--overlap", str(case["overlap"]), "--in-order", "0", "--gate-split", str(case["split_gate"]),
            "--down-split", str(case["split_down"]), "--local", "4", "--repeats", str(a.repeats),
            "--warmup", "3", "--skip-probe", "1", "--dump-contract", str(int(a.dump_contract and case["q"] == 184))]
    if case["model_shape"] and a.gate:
        for key in ("gate", "up", "down"):
            args += ["--" + key, str(getattr(a, key))]
    return args

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", type=Path, default=Path("build/maple-architecture-compare.exe"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--suite", choices=("smoke", "quick", "full"), default="smoke")
    ap.add_argument("--device", default="A750")
    ap.add_argument("--repeats", type=int, default=28)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--dump-contract", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    for key in ("gate", "up", "down"):
        ap.add_argument("--" + key, type=Path)
    a = ap.parse_args()
    if not (2 <= a.repeats <= 10000) or a.timeout <= 0:
        ap.error("invalid repeats/timeout")
    if any((a.gate, a.up, a.down)) and not all((a.gate, a.up, a.down)):
        ap.error("gate/up/down capsules must be supplied together")
    if a.out.exists() and any(a.out.iterdir()):
        ap.error("output directory is not empty; never overwrite evidence")
    a.out.mkdir(parents=True, exist_ok=True)
    a.exe = a.exe.resolve()
    cases = cases_for(a.suite)
    plan = {"version": VERSION, "dry_run": a.dry_run, "case_count": len(cases),
            "scope": "one MoE layer, baseline/candidate AB/BA; NOT server t/s",
            "cases": [{**c, "argv": command_for(c, a)} for c in cases]}
    if not a.dry_run:
        plan["build"] = verify_build(a.exe)
    (a.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if a.dry_run:
        print(f"Dry-run only: {len(cases)} cases, no GPU invocation, no performance results")
        return 0
    def run(argv, log):
        with log.open("w", encoding="utf-8") as stream:
            try:
                rc = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, timeout=a.timeout, check=False).returncode
            except subprocess.TimeoutExpired:
                stream.write("\nTIMEOUT; kernel acceptance not established\n")
                return 124
        return rc
    rc = run([str(a.exe), "--device", a.device, "--probe-only", "1"], a.out / "probe.log")
    if rc:
        print("Probe failed; preserve probe.log. No fallback/performance pass claimed", file=sys.stderr)
        return rc
    status = []
    for item in plan["cases"]:
        print(item["name"], flush=True)
        rc = run(item["argv"], a.out / (item["name"] + ".log"))
        ok = rc == 0
        if ok:
            path = a.out / item["name"] / "summary.csv"
            if not path.is_file():
                ok = False
            else:
                with path.open(newline="", encoding="utf-8-sig") as stream:
                    rows = list(csv.DictReader(stream))
                ok = len(rows) == 2 and {r["variant"] for r in rows} == {"baseline", "candidate"} and all(
                    r["kernel_pass"] == "1" and r["pair_pass"] == "1" for r in rows)
        status.append({"case": item["name"], "exit": rc, "pass": ok})
        (a.out / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        if not ok:
            print("FAILED: inspect quant_audit, numerical, pair_correctness, raw dumps; do not widen tolerances", file=sys.stderr)
            return rc or 1
    print("All case checks passed; performance and model quality remain separate judgments")
    return 0
if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
