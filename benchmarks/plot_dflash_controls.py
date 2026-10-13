#!/usr/bin/env python3
"""Plot accepted uninstrumented DFlash controls; do not infer PCIe rates from final counters."""

import argparse
import csv
import json
import math
import re
from pathlib import Path


def load_series(directory):
    points=[]
    seen=set()
    for line in (directory/"results.jsonl").read_text().splitlines():
        row=json.loads(line)
        if row.get("mode") != "timing":
            continue
        if row.get("status") != "ok" or row.get("numerical_control_only") or row.get("uvm"):
            raise ValueError("performance plot requires accepted non-UVM timing points")
        identity=(row["context"],row["depth"])
        if identity in seen:
            raise ValueError("duplicate context/depth timing point")
        seen.add(identity)
        timing=row["result"]["timings"]
        for key in ("prompt_per_second","predicted_per_second"):
            if not isinstance(timing.get(key),(int,float)) or not math.isfinite(timing[key]) or timing[key] <= 0:
                raise ValueError("invalid throughput")
        generated=timing.get("draft_n",0)
        accepted=timing.get("draft_n_accepted",0)
        if type(generated) is not int or type(accepted) is not int or generated <= 0 or not 0 <= accepted <= generated:
            raise ValueError("invalid draft counts")
        log=Path(row["log"]).read_text(errors="replace")
        grants=re.findall(r"memory_phase: phase=decode[^\n]*\bkv_pool=(\d+)",log)
        points.append({"context":row["context"],"depth":row["depth"],"arena_mib":row.get("arena_mib"),
            "decode_kv_pool_mib":int(grants[-1])/1048576 if grants else None,
            "prefill_tps":timing["prompt_per_second"],"decode_tps":timing["predicted_per_second"],
            "acceptance_pct":100*accepted/generated})
    if not points:
        raise ValueError("no accepted timing points")
    return sorted(points,key=lambda p:(p["depth"],p["context"]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series",nargs=2,action="append",required=True,metavar=("LABEL","DIRECTORY"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    rows=[]
    for label,directory in args.series:
        points=load_series(Path(directory))
        for depth in sorted({p["depth"] for p in points}):
            group=[p for p in points if p["depth"]==depth]
            name=f"{label}, D={depth}"
            x=[p["context"]/1024 for p in group]
            axes[0,0].plot(x,[p["decode_tps"] for p in group],marker=".",label=name)
            axes[0,1].plot(x,[p["prefill_tps"] for p in group],linestyle="--",marker=".",label=name)
            axes[1,0].plot(x,[p["decode_kv_pool_mib"] if p["decode_kv_pool_mib"] is not None else math.nan for p in group],marker=".",label=name)
            axes[1,1].plot(x,[p["acceptance_pct"] for p in group],marker=".",label=name)
            rows.extend(dict(series=label,**p) for p in group)
    for ax,title,unit in zip(axes.flat,("Accepted decode throughput","Prefill throughput","Effective decode KV grant","Draft acceptance"),
            ("tokens/s","tokens/s","MiB","percent")):
        ax.set_title(title);ax.set_ylabel(unit);ax.grid(alpha=.25);ax.legend(fontsize=8)
    for ax in axes[1]:ax.set_xlabel("Context capacity (Ki tokens)")
    figure.suptitle("Adaptive DFlash2: 256 generated tokens per point, non-UVM")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(args.output,dpi=180)
    with args.output.with_suffix(".csv").open("w",newline="") as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=="__main__":
    main()
