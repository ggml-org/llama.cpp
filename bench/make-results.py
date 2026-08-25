#!/usr/bin/env python3
"""Generate bench/RESULTS.md from the append-only JSONL logs.

Regenerate rather than hand-edit: the logs are the source of truth, and the tables
should never drift from them.
"""
import json, statistics, collections, math, pathlib, datetime

HERE = pathlib.Path(__file__).resolve().parent

def load(name):
    p = HERE / name
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out

def test_name(r):
    return f"pp{r['n_prompt']}" if r.get("n_prompt") else f"tg{r['n_gen']}"

def depth_table(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["model_label"], test_name(r), r["n_depth"], r["build"])].append(r["avg_ts"])
    keys = sorted({(m, t, d) for m, t, d, _ in by},
                  key=lambda k: (k[0], k[1].startswith("tg"), k[2]))
    lines = ["| model | test | depth | mainline t/s | fork t/s | delta | |",
             "|---|---|---:|---:|---:|---:|---|"]
    for m, t, d in keys:
        a, b = by.get((m, t, d, "mainline"), []), by.get((m, t, d, "fork"), [])
        if not a or not b:
            continue
        ma, mb = statistics.mean(a), statistics.mean(b)
        sa = statistics.stdev(a) if len(a) > 1 else 0.0
        sb = statistics.stdev(b) if len(b) > 1 else 0.0
        se = math.sqrt(sa**2/len(a) + sb**2/len(b))
        sig = (mb-ma)/se if se else float("inf")
        note = "" if abs(sig) >= 2 else " *noise*"
        lines.append(f"| {m} | {t} | {d} | {ma:.1f} ± {sa:.1f} | {mb:.1f} ± {sb:.1f} | "
                     f"**{100*(mb/ma-1):+.1f}%**{note} | {sig:+.1f}σ |")
    return "\n".join(lines)

def gates_table(rows):
    lines = ["| model | quant | gate | off t/s | on t/s | delta |", "|---|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['quant']} | `{r['gate']}` | {r['off']:.1f} | "
                     f"{r['on']:.1f} | **{r['delta_pct']:+.1f}%** |")
    return "\n".join(lines)

def prefill_table(rows):
    if not rows:
        return "*Not yet measured.*"
    lines = ["| model | ubatch | test | t/s |", "|---|---:|---|---:|"]
    for r in sorted(rows, key=lambda r: (r.get("model_type", ""), r.get("n_ubatch", 0), r.get("n_prompt", 0))):
        lines.append(f'| {r.get("model_type","?")} | {r.get("n_ubatch","?")} | pp{r.get("n_prompt","?")} | '
                     f'{r.get("avg_ts",0):.1f} ± {r.get("stddev_ts",0):.1f} |')
    return "\n".join(lines)


def spec_table(rows):
    lines = ["| policy | workload | t/s | draft acceptance |", "|---|---|---:|---:|"]
    for r in rows:
        acc = f"{r['accept_pct']:.0f}%" if r.get("accept_pct") else "—"
        lines.append(f"| {r['policy']} | {r['workload']} | **{r['predicted_per_second']:.2f}** | {acc} |")
    return "\n".join(lines)

def main():
    depth  = load("results.jsonl")
    gates  = load("results-gates.jsonl")
    fp4    = load("results-fp4-spec.jsonl")
    prefill = load("results-prefill.jsonl")
    head   = load("results-fp4-headline.jsonl")
    sdepth = load("results-spec-depth.jsonl")

    pwr = {r.get("power_dpm_state", "unknown") for r in depth} or {"unknown"}
    doc = f"""# Benchmark results

Generated from the JSONL logs in this directory by `make-results.py`. Do not hand-edit —
regenerate, so the tables cannot drift from the data.

Generated {datetime.date.today().isoformat()}. Hardware: single Radeon 8060S (Strix Halo APU,
gfx1151, RDNA 3.5), RADV / Mesa 26.0.8.

## How to read these numbers

**The fork-vs-mainline and gate-on-vs-off figures are ratios**, each measured within a single
session on one power profile, with arms interleaved in palindrome order. The ratios hold. The
**absolute** t/s figures in those tables were taken under the power profile in force at the time
(`power_dpm_state` recorded per row; the earlier set predates that stamp) and are not directly
comparable to the headline section, which was re-measured after the machine moved to a higher power
setting. Compare deltas across tables, not absolutes.

Every generation figure carries its context depth. These models declare 262144 context and token
generation at depth is roughly a third of its depth-0 value, so a bare t/s number is not a claim.

## 1. Fork vs pinned upstream, prefill and generation against depth

Upstream pinned at `95b8e33e1`, the exact commit this fork merged, so the delta is our changes and
not upstream drift. `-ub 512 -fa 1`, 3 internal repetitions, palindrome-ordered arms, warmup
discarded. Sub-2σ marked as noise.

{depth_table(depth)}

## 2. Vulkan gate ablations

Each gate measured on/off in one binary, `pp2048 -ub 2048`. Three models across three quant
families. Transcribed from `WORKLOG.local.md` where the raw JSONL was lost with tmpfs on reboot.

{gates_table(gates)}

## 3. Prefill, absolute, current power profile

{prefill_table(prefill)}

## 4. Speculative decoding, FP4 stack

Target `Qwen3.8-27B-ROCmFP4-FAST` (13.9 GB, `Q4_0_ROCMFP4_FAST`), draft
`Qwen3.8-27B-DFlash2-Q4_0_ROCMFP4_FAST` (987 MB), DFlash2, greedy, 300 tokens, depth 0.
Fork only — upstream cannot load ROCmFPx.

{spec_table([r for r in fp4 if r.get('predicted_per_second')])}

The acceptance column is the mechanism: fixed n=7 collapses to 18% acceptance, fixed n=3 sits at
97% and is therefore *under*-drafting, and adaptive holds 96% while drafting longer. The same
`n_max = 7` that destroys the fixed arm is safe under adaptive.
"""

    if head:
        doc += f"""
### Headline, re-measured on the higher power profile

{spec_table([r for r in head if r.get('predicted_per_second')])}
"""

    if sdepth:
        doc += """
## 4. Adaptive drafting against context depth

Qwen3.8-27B UD-Q4_K_XL target. MTP uses the target's own nextn layers; DFlash2 uses the z-lab Q8_0
sidecar. Both arms adaptive, `n_max 7`, `n_min 3`.

| policy | workload | depth | prefill t/s | generation t/s | acceptance |
|---|---|---:|---:|---:|---:|
"""
        for r in sdepth:
            acc = 100.0*(r["draft_n_accepted"] or 0)/r["draft_n"] if r.get("draft_n") else 0
            doc += (f"| {r['policy']} | {r['workload']} | {r['depth']} | "
                    f"{r['prompt_per_second'] or 0:.1f} | {r['predicted_per_second'] or 0:.2f} | {acc:.0f}% |\n")
        doc += "\n*Incomplete: the collecting run was interrupted.*\n"

    (HERE / "RESULTS.md").write_text(doc)
    print(f"wrote {HERE/'RESULTS.md'} ({len(doc.splitlines())} lines)")

main()
