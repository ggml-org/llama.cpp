#!/usr/bin/env python3
"""Plot accepted target/MTP/DFlash controls; do not infer PCIe rates from final counters."""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


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
        spec_type=("mtp" if depth else "none") if target_control else row.get("spec_mode","dflash")
        if (spec_type not in ("none","mtp","dflash","dflash-k4v") or type(context) is not int or context <= 0
                or type(depth) is not int or depth < (0 if spec_type == "none" else 1)
                or (spec_type == "none" and depth != 0)):
            raise ValueError("invalid context or draft depth")
        config=row.get("ngram") if spec_type == "dflash-k4v" else None
        if spec_type == "dflash-k4v" and (not isinstance(config,dict) or not 1 <= depth <= 7 or
                any(type(config.get(name)) is not int or not 1 <= config[name] <= maximum
                    for name,maximum in (("ngram_n",65535),("ngram_m",7),("ngram_min_hits",65535)))):
            raise ValueError("mixed plot needs explicit bounded lookup/proposal/hit settings")
        settings={name:config[name] if config else None for name in ("ngram_n","ngram_m","ngram_min_hits")}
        identity=(context,spec_type,depth,*settings.values())
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
            "acceptance_pct":acceptance,**settings})
    if not points:
        raise ValueError("no accepted timing points")
    return sorted(points,key=lambda p:(p["spec_type"],p["depth"],p["context"]))


def group_key(point):
    return tuple(point.get(name) for name in ("spec_type","depth","ngram_n","ngram_m","ngram_min_hits"))


def mode_name(point):
    mode,depth=point["spec_type"],point["depth"]
    if mode == "none": return "target only"
    if mode == "mtp": return f"MTP = {depth}"
    if mode == "dflash": return f"DFlash2 = {depth}"
    return f"DFlash2 = {depth} + K4V {point['ngram_m']} (N={point['ngram_n']}, hits={point['ngram_min_hits']})"


def load_csv_points(path, stock=False):
    points=[]
    with path.open() as source:
        for row in csv.DictReader(source):
            point: dict[str, Any]={"context":int(row["context_capacity" if stock else "context"]),
                "spec_type":"none" if stock else row["spec_type"],"depth":0 if stock else int(row["depth"]),
                "prefill_tps":float(row["prefill_tps"]),"decode_tps":float(row["decode_tps"]),
                "ngram_n":None,"ngram_m":None,"ngram_min_hits":None}
            if point["spec_type"] not in ("none","mtp","dflash") or point["context"] <= 0 or any(
                    not math.isfinite(point[key]) or point[key] <= 0 for key in ("prefill_tps","decode_tps")):
                raise ValueError("invalid historical throughput data")
            points.append(point)
    if not points: raise ValueError("empty historical series")
    return points


def throughput_figure(series, historical=(), stock=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(14,6),sharex=True)
    def lines(points,label,**style):
        ordered=sorted(points,key=lambda p:p["context"])
        x=[p["context"]/1024 for p in ordered]
        color=axes[0].plot(x,[p["decode_tps"] for p in ordered],label=label,**style)[0].get_color()
        axes[1].plot(x,[p["prefill_tps"] for p in ordered],label=label,**(style|{"color":color}))
    for key in sorted({group_key(p) for p in historical}):
        points=[p for p in historical if group_key(p)==key]
        lines(points,"Earlier "+mode_name(points[0]),alpha=.4,linewidth=1.4,linestyle=":")
    for label,points in series:
        for key in sorted({group_key(p) for p in points}):
            group=[p for p in points if group_key(p)==key]
            lines(group,label+", "+mode_name(group[0]),linewidth=2.1,marker=".")
    if stock:
        lines(stock,"Stock + UVM (historical)",color="#444444",linewidth=1.6,marker="x",markersize=4)
    for axis,title in zip(axes,("Decode throughput","Prefill throughput")):
        axis.set_title(title);axis.set_ylabel("tokens/s");axis.set_xlabel("Context capacity (Ki tokens)")
        axis.set_ylim(bottom=0);axis.grid(alpha=.25)
    contexts=[p["context"]/1024 for _,points in series for p in points]+[p["context"]/1024 for p in (*historical,*stock)]
    if contexts:
        first,last=min(contexts),max(contexts)
        for axis in axes: axis.set_xlim(first,last if last>first else first+1)
    figure.suptitle("Adaptive KV streaming: distinct uninstrumented runtime cohorts")
    figure.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3,fontsize=8,frameon=False)
    figure.tight_layout(rect=(0,.19,1,.94))
    return figure,axes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series",nargs=2,action="append",required=True,metavar=("LABEL","DIRECTORY"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--csv-output",type=Path,help="separate current-cohort CSV; historical input files remain unchanged")
    parser.add_argument("--throughput-only",action="store_true",help="two-panel prefill/decode comparison with zero y origins")
    parser.add_argument("--historical-csv",type=Path,help="earlier adaptive cohort, shown as separate faded lines")
    parser.add_argument("--stock-uvm-csv",type=Path,help="distinct historical stock-UVM cohort, without extrapolation")
    parser.add_argument("--title",help="explicit model, hardware and workload caption")
    args=parser.parse_args()
    if (args.historical_csv or args.stock_uvm_csv) and not args.throughput_only:
        parser.error("historical overlays require --throughput-only")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series=[(label,load_series(Path(directory))) for label,directory in args.series]
    rows=[]
    if args.throughput_only:
        figure,axes=throughput_figure(series,load_csv_points(args.historical_csv) if args.historical_csv else (),
            load_csv_points(args.stock_uvm_csv,True) if args.stock_uvm_csv else ())
        rows=[dict(series=label,**point) for label,points in series for point in points]
    else:
        figure,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for label,points in (() if args.throughput_only else series):
        for key in sorted({group_key(p) for p in points}):
            group=[p for p in points if group_key(p)==key]
            spec_type=group[0]["spec_type"]
            mode=mode_name(group[0])
            name=f"{label}, {mode}"
            x=[p["context"]/1024 for p in group]
            color=axes[0,0].plot(x,[p["decode_tps"] for p in group],marker=".",label=name)[0].get_color()
            axes[0,1].plot(x,[p["prefill_tps"] for p in group],color=color,linestyle="--",marker=".",label=name)
            axes[1,0].plot(x,[p["decode_kv_pool_mib"] if p["decode_kv_pool_mib"] is not None else math.nan for p in group],color=color,marker=".",label=name)
            if spec_type != "none":
                axes[1,1].plot(x,[p["acceptance_pct"] for p in group],color=color,marker=".",label=name)
            rows.extend(dict(series=label,**p) for p in group)
    if not args.throughput_only:
        for ax,title,unit in zip(axes.flat,("Accepted decode throughput","Prefill throughput","Effective decode KV grant","Draft acceptance"),
                ("tokens/s","tokens/s","MiB","percent")):
            ax.set_title(title);ax.set_ylabel(unit);ax.set_ylim(bottom=0);ax.grid(alpha=.25);ax.legend(fontsize=8)
        for ax in axes[1]:ax.set_xlabel("Context capacity (Ki tokens)")
        figure.suptitle("Adaptive KV streaming: accepted uninstrumented non-UVM controls")
        figure.tight_layout()
    if args.title: figure.suptitle(args.title)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(args.output,dpi=180)
    csv_path=args.csv_output or args.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    with csv_path.open("w",newline="") as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)


if __name__=="__main__":
    main()
