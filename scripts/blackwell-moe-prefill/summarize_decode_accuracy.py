from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.run_dir.glob("*-vs-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append((path.stem, data))

    print("# MoE decode accuracy")
    print()
    print("| Comparison | NMSE | NRMSE | Mean abs | P99 abs | Max abs | Cosine | KL | Top-1 | Top-10 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for name, data in rows:
        topk = data.get("topk_overlap", {})
        print(
            f"| {name} | {data['nmse']:.6g} | {data['nrmse']:.6g} | "
            f"{data['mean_abs']:.6g} | {data['p99_abs']:.6g} | {data['max_abs']:.6g} | "
            f"{data['cosine_similarity']:.8f} | {data['kl_divergence']:.6g} | "
            f"{'yes' if data['top1_match'] else 'no'} | {topk.get('10', 0)}/10 |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
