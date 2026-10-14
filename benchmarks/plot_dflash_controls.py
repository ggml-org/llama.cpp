#!/usr/bin/env python3
"""Plot accepted target/MTP/DFlash controls; do not infer PCIe rates from final counters."""

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
        target_control="mtp_length" in row
        if row.get("mode","timing" if target_control else None) != "timing":
            continue
        if row.get("status") != "ok" or row.get("numerical_control_only") or row.get("uvm"):
            raise ValueError("performance plot requires accepted non-UVM timing points")
        context=row["context_capacity"] if target_control else row["context"]
        depth=row["mtp_length"] if target_control else row["depth"]
        spec_type="mtp" if target_control and depth else "none" if target_control else "dflash"
        if type(context) is not int or context <= 0 or type(depth) is not int or depth < (0 if target_control else 1):
            raise ValueError("invalid context or draft depth")
        identity=(context,spec_type,depth)
        if identity in seen:
            raise ValueError("duplicate context/depth timing point")
        seen.add(identity)
        timing={"prompt_per_second":row["prefill_tps"],"predicted_per_second":row["decode_tps"],
                "draft_n":row["mtp_draft_generated"],"draft_n_accepted":row["mtp_draft_accepted"]} if target_control else row["result"]["timings"]
        for key in ("prompt_per_second","predicted_per_second"):
            if type(timing.get(key)) not in (int,float) or not math.isfinite(timing[key]) or timing[key] <= 0:
                raise ValueError("invalid throughput")
        generated=timing.get("draft_n",0)
        accepted=timing.get("draft_n_accepted",0)
        acceptance=None
        if spec_type == "none":
            if generated not in (None,0) or accepted not in (None,0):
                raise ValueError("target-only control contains draft counts")
        else:
            if type(generated) is not int or type(accepted) is not int or generated <= 0 or not 0 <= accepted <= generated:
                raise ValueError("invalid draft counts")
            acceptance=100*accepted/generated
        log=Path(row["log"]).read_text(errors="replace")
        grants=re.findall(r"memory_phase: phase=decode[^\n]*\bkv_pool=(\d+)",log)
        points.append({"context":context,"depth":depth,"spec_type":spec_type,"arena_mib":row.get("arena_mib"),
            "decode_kv_pool_mib":int(grants[-1])/1048576 if grants else None,
            "prefill_tps":timing["prompt_per_second"],"decode_tps":timing["predicted_per_second"],
            "acceptance_pct":acceptance})
    if not points:
        raise ValueError("no accepted timing points")
    return sorted(points,key=lambda p:(p["spec_type"],p["depth"],p["context"]))


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
        for spec_type,depth in sorted({(p["spec_type"],p["depth"]) for p in points}):
            group=[p for p in points if (p["spec_type"],p["depth"]) == (spec_type,depth)]
            mode="target only" if spec_type == "none" else f"MTP D={depth}" if spec_type == "mtp" else f"DFlash2 D={depth}"
            name=f"{label}, {mode}"
            x=[p["context"]/1024 for p in group]
            color=axes[0,0].plot(x,[p["decode_tps"] for p in group],marker=".",label=name)[0].get_color()
            axes[0,1].plot(x,[p["prefill_tps"] for p in group],color=color,linestyle="--",marker=".",label=name)
            axes[1,0].plot(x,[p["decode_kv_pool_mib"] if p["decode_kv_pool_mib"] is not None else math.nan for p in group],color=color,marker=".",label=name)
            if spec_type != "none":
                axes[1,1].plot(x,[p["acceptance_pct"] for p in group],color=color,marker=".",label=name)
            rows.extend(dict(series=label,**p) for p in group)
    for ax,title,unit in zip(axes.flat,("Accepted decode throughput","Prefill throughput","Effective decode KV grant","Draft acceptance"),
            ("tokens/s","tokens/s","MiB","percent")):
        ax.set_title(title);ax.set_ylabel(unit);ax.grid(alpha=.25);ax.legend(fontsize=8)
    for ax in axes[1]:ax.set_xlabel("Context capacity (Ki tokens)")
    figure.suptitle("Adaptive KV streaming: accepted uninstrumented non-UVM controls")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(args.output,dpi=180)
    with args.output.with_suffix(".csv").open("w",newline="") as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=="__main__":
    main()
