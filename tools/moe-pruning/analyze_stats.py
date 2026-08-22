#!/usr/bin/env python3
"""
analyze_stats.py  --  Summarize expert_stats.json and model size projections.
Usage: python analyze_stats.py [stats_file] [--keep 0.5]
"""
import json, statistics, argparse

parser = argparse.ArgumentParser()
parser.add_argument("stats", nargs="?", default="expert_stats_reap.json")
parser.add_argument("--keep", type=float, default=0.5, help="Fraction of experts to keep (default 0.5)")
args = parser.parse_args()

with open(args.stats) as f:
    data = json.load(f)

layers = sorted(data.keys(), key=int)
n_layers = len(layers)
keep_ratio = args.keep

# Detect which scoring field is available (new REAP vs old importance_score)
sample_layer = data[layers[0]]
if "reap" in sample_layer:
    score_field = "reap"
    score_label = "REAP (gate_weight × ||expert_out||₂)"
elif "importance_score" in sample_layer:
    score_field = "importance_score"
    score_label = "importance_score (freq × avg_gate_weight)  [legacy, no EAN]"
else:
    raise ValueError(f"No recognised score field in stats. Keys: {list(sample_layer.keys())}")

# ── Model architecture constants (Nemotron-3-Nano-30B-A3B) ──────────────────
N_EXPERTS        = 128
N_EXPERT_USED    = 6       # top-k per token
N_MOE_LAYERS     = 23
N_TOTAL_LAYERS   = 53
# Approximate parameter counts (bf16, billions)
PARAMS_TOTAL_B        = 30.0
PARAMS_MOE_EXPERTS_B  = 22.0   # bulk of MoE weight is in expert FFNs
PARAMS_NON_MOE_B      = PARAMS_TOTAL_B - PARAMS_MOE_EXPERTS_B

# ── Header ──────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"  Expert Stats Analysis  |  file: {args.stats}")
print("=" * 70)

# ── Profiling completeness ───────────────────────────────────────────────────
sample_tokens = list(data.values())[0]["total_tokens"]
# Each token activates N_EXPERT_USED experts, sum(activation_counts) = total*top_k
# Approximate samples: total_tokens / avg_tokens_per_sample
# We don't know avg, but can infer: total_tokens / (total_tokens / ctx) ≈ ctx chunks
# Better: just report tokens and note the user knows sample count
print(f"\n── Profiling progress ──────────────────────────────────────────────────")
print(f"  MoE layers profiled    : {n_layers} / {N_MOE_LAYERS}")
print(f"  Tokens processed       : {sample_tokens:,}  (per layer)")
act_sum = sum(data[layers[0]]["activation_counts"])
assert abs(act_sum / sample_tokens - N_EXPERT_USED) < 0.01, "unexpected top-k"
print(f"  top-k confirmed        : {N_EXPERT_USED}  (sum activations / tokens = {act_sum/sample_tokens:.1f})")

# ── Per-layer importance score stats ────────────────────────────────────────
print(f"\n── Per-layer score distribution  [{score_label}]")
print(f"  {'Layer':>5}  {'Min':>9}  {'Max':>9}  {'Range':>9}  {'CV%':>6}  {'Never':>5}")
global_cvs = []
for k in layers:
    d = data[k]
    s = d[score_field]
    mn, mx = min(s), max(s)
    cv = statistics.stdev(s) / statistics.mean(s) * 100
    global_cvs.append(cv)
    print(f"  {k:>5}  {mn:>9.5f}  {mx:>9.5f}  {mx-mn:>9.5f}  {cv:>6.3f}%  {d['never_activated']:>5}")

print(f"\n  Mean CV across layers  : {statistics.mean(global_cvs):.3f}%")
print(f"  (CV < 1% = near-uniform; load-balancing is working as designed)")

# ── Capacity loss sweep across pruning levels ────────────────────────────────
# Paper (observer.py): REAP[i] = mean(ean_norm * softmax_router_weight) over tokens
#   routed to expert i, averaged via OnlineStatsTracker weighted by expert_frequency.
# Our implementation (llama.cpp): same formula but routing weights are the top-k
#   gate weights (post-softmax within top-k), not the full softmax over all 128.
# Impact: our weights are slightly higher than the paper's (renormalized to top-k
#   only), but relative expert ranking within a layer should be preserved.
#
# IMPORTANT CAVEAT for this model (Nemotron-3-Nano-30B-A3B):
#   The model was trained with a strong load-balancing auxiliary loss, so all 128
#   experts have nearly identical activation frequency (~4.69%) AND nearly identical
#   REAP scores (Gini ~0.015, top/bottom ratio ~1.1-1.35x). The score distribution
#   is a smooth monotone curve with NO natural elbow or gap.
#
#   This means:
#   - REAP ranking beats random pruning by only ~1pp in mass terms at keep=33%
#   - The cut point boundary (rank 42 vs 43) has near-zero gap in most layers
#   - REAP paper results on Qwen3-30B-A3B likely had higher Gini (less tight
#     load-balancing or more expert specialization in pre-training)
#   - For this model, actual quality loss must be measured via eval, not predicted
#     from REAP score variance
#
# Metrics reported:
# - kept_mass%: REAP mass in the KEPT experts as % of total (> keep_ratio% = good)
# - vs_random%: how much more mass the REAP-selected set retains vs a random set
#               of the same size (= kept_mass% - keep_ratio%). Positive = REAP wins.
# - Rel.gap:    score gap at cut / layer score range. Near 0 = no natural cut point.
# - Gini:       inequality of score distribution. ~0.015 here = near-uniform.

def gini(scores):
    """Gini coefficient of a list of non-negative values."""
    n = len(scores)
    s = sorted(scores)
    total = sum(s)
    if total == 0:
        return 0.0
    cumsum = 0.0
    for i, v in enumerate(s):
        cumsum += (2 * (i + 1) - n - 1) * v
    return cumsum / (n * total)

def layer_stats(scores, n_keep):
    """Return capacity metrics for a single layer at a given keep count."""
    n = len(scores)
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    total  = sum(scores)
    kept_mass   = sum(scores[i] for i in ranked[:n_keep])
    kept_frac   = kept_mass / total if total > 0 else 0.0     # fraction of REAP mass kept
    random_frac = n_keep / n                                   # uniform expectation
    vs_random   = kept_frac - random_frac                     # positive = REAP beats random
    score_range = scores[ranked[0]] - scores[ranked[-1]]
    gap         = scores[ranked[n_keep - 1]] - (scores[ranked[n_keep]] if n_keep < n else 0)
    rel_gap     = gap / score_range if score_range > 0 else 0.0
    return kept_frac * 100, vs_random * 100, rel_gap

# Sweep over a range of keep ratios
sweep_ratios = [0.10, 0.20, 0.25, 0.33, 0.40, 0.50, 0.60, 0.75]
if keep_ratio not in sweep_ratios:
    sweep_ratios.append(keep_ratio)
sweep_ratios = sorted(set(sweep_ratios))

# Per-layer Gini (fixed, independent of keep ratio)
layer_ginis = {k: gini(data[k][score_field]) for k in layers}
mean_gini = statistics.mean(layer_ginis.values())
worst_gini_layer = max(layer_ginis, key=lambda k: layer_ginis[k])

print(f"\n── Score distribution inequality (Gini coefficient) ────────────────────")
print(f"  Gini measures how non-uniform REAP scores are within each layer.")
print(f"  Gini=0: all experts identical. Gini=1: one expert dominates.")
print(f"  With load-balanced MoE, Gini is small — but any Gini > 0 means")
print(f"  REAP ranking beats random pruning.")
print(f"")
print(f"  {'Layer':>5}  {'Gini':>8}  {'Score range':>13}  {'Max/Min ratio':>14}")
print(f"  {'-'*5}  {'-'*8}  {'-'*13}  {'-'*14}")
for k in layers:
    s = data[k][score_field]
    mn, mx = min(s), max(s)
    g = layer_ginis[k]
    ratio_mm = mx / mn if mn > 0 else float('inf')
    print(f"  {k:>5}  {g:>8.5f}  {mx-mn:>13.5f}  {ratio_mm:>13.3f}x")
print(f"")
print(f"  Mean Gini : {mean_gini:.5f}  (worst layer: {worst_gini_layer})")

print(f"\n── Capacity retention sweep ─────────────────────────────────────────────")
print(f"  Kept mass%  = REAP mass in KEPT experts as % of total (higher = better)")
print(f"  vs.rand%    = Kept mass% minus uniform baseline (keep_ratio%)")
print(f"                Positive = REAP beats random. Magnitude = advantage in pp.")
print(f"  Rel.gap     = score gap at cut / layer score range (higher = cleaner cut)")
print(f"  WARNING: near-zero rel.gap and small vs.rand mean eval is the only ground truth.")
print(f"")
print(f"  {'Keep':>5}  {'Experts':>7}  {'Kept mass%':>11}  {'vs.rand%':>9}  {'Rel.gap avg':>12}  {'Worst layer':>11}")
print(f"  {'-'*5}  {'-'*7}  {'-'*11}  {'-'*9}  {'-'*12}  {'-'*11}")

sweep_results = {}
for ratio in sweep_ratios:
    nk = max(1, round(N_EXPERTS * ratio))
    mass_fracs, excesses, rel_gaps = [], [], []
    worst_excess, worst_layer_id = -999.0, None
    for k in layers:
        scores = data[k][score_field]
        mf, exc, rg = layer_stats(scores, nk)
        mass_fracs.append(mf)
        excesses.append(exc)
        rel_gaps.append(rg)
        if exc > worst_excess:
            worst_excess = exc
            worst_layer_id = k
    avg_mf  = statistics.mean(mass_fracs)
    avg_exc = statistics.mean(excesses)
    avg_rg  = statistics.mean(rel_gaps)
    marker  = " <--" if abs(ratio - keep_ratio) < 1e-9 else ""
    print(f"  {ratio:>5.0%}  {nk:>7d}  {avg_mf:>10.2f}%  {avg_exc:>+9.2f}%  {avg_rg:>11.4f}  layer {worst_layer_id:>3}{marker}")
    sweep_results[ratio] = {
        "n_keep": nk, "avg_kept_mass": avg_mf, "avg_vs_random": avg_exc,
        "avg_rel_gap": avg_rg, "worst_layer_id": worst_layer_id, "worst_vs_random": worst_excess,
    }

print(f"")
print(f"  vs.rand% quantifies REAP's advantage over random pruning in REAP-mass terms.")
print(f"  For this model it is small (+0.7 to +1.5pp) due to tight load-balancing.")
print(f"  Rel.gap near zero means scores are smooth with no natural cut — any threshold")
print(f"  is as defensible as another. Actual quality delta requires empirical eval.")

# ── Expert keep/prune detail at selected keep_ratio ──────────────────────────
n_keep   = max(1, round(N_EXPERTS * keep_ratio))
n_prune  = N_EXPERTS - n_keep

print(f"\n── Expert pruning detail at keep_ratio={keep_ratio:.0%}  ({n_keep} keep / {n_prune} prune per layer) ──")
print(f"  {'Layer':>5}  {'Kept mass%':>11}  {'vs.rand%':>9}  {'Rel.gap':>9}  {'Min kept':>10}  {'Max pruned':>11}")
print(f"  {'-'*5}  {'-'*11}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*11}")

layer_results = {}
for k in layers:
    scores = data[k][score_field]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    mf, exc, rg = layer_stats(scores, n_keep)
    min_kept   = scores[ranked[n_keep - 1]]
    max_pruned = scores[ranked[n_keep]] if n_prune > 0 else 0
    layer_results[k] = {"mass_frac": mf, "excess": exc, "rel_gap": rg,
                        "min_kept": min_kept, "max_pruned": max_pruned}
    print(f"  {k:>5}  {mf:>10.2f}%  {exc:>+9.2f}%  {rg:>9.4f}  {min_kept:>10.5f}  {max_pruned:>11.5f}")

avg_mf  = statistics.mean(r["mass_frac"] for r in layer_results.values())
avg_exc = statistics.mean(r["excess"]    for r in layer_results.values())
avg_rg  = statistics.mean(r["rel_gap"]   for r in layer_results.values())
print(f"  {'AVG':>5}  {avg_mf:>10.2f}%  {avg_exc:>+9.2f}%  {avg_rg:>9.4f}")

# ── Model size projections ───────────────────────────────────────────────────
print(f"\n── Model size projections ──────────────────────────────────────────────")

def model_size(keep):
    expert_params = PARAMS_MOE_EXPERTS_B * keep
    return PARAMS_NON_MOE_B + expert_params

original_b   = model_size(1.0)
pruned_b     = model_size(keep_ratio)
reduction_pct = (1 - pruned_b / original_b) * 100

# GGUF sizes at common quant levels (rough: 1B params ≈ quant_bpw/8 GB)
quants = [("Q8_0", 8.0), ("Q5_K_M", 5.5), ("Q4_K_M", 4.5), ("Q3_K_M", 3.35), ("Q2_K", 2.63)]

print(f"  {'':20}  {'Original':>10}  {'Pruned':>10}  {'Saved':>8}")
print(f"  {'Parameters (B)':20}  {original_b:>10.1f}  {pruned_b:>10.1f}  {original_b-pruned_b:>8.1f}B")
print(f"  {'Reduction':20}  {'':>10}  {reduction_pct:>9.1f}%")
print()
print(f"  Estimated GGUF sizes:")
print(f"  {'Quant':10}  {'Original':>10}  {'Pruned':>10}  {'Fits in':>12}")
for name, bpw in quants:
    orig_gb  = original_b * bpw / 8
    prune_gb = pruned_b   * bpw / 8
    # VRAM fit (16GB GPU)
    fits = "16GB GPU" if prune_gb <= 15.5 else ("32GB GPU" if prune_gb <= 31 else "CPU/RAM")
    print(f"  {name:10}  {orig_gb:>9.1f}G  {prune_gb:>9.1f}G  {fits:>12}")

# ── Active params per token (inference cost) ─────────────────────────────────
print(f"\n── Inference cost (active params per token) ────────────────────────────")
# Active params = non-moe + (n_expert_used/n_experts_kept * moe_expert_params)
# After pruning: router still picks top-k but from n_keep pool
# Active expert params per token = (N_EXPERT_USED / n_keep) * (PARAMS_MOE_EXPERTS_B * keep_ratio)
# But actually active params = N_EXPERT_USED * (params per single expert)
params_per_expert_orig   = PARAMS_MOE_EXPERTS_B / N_EXPERTS          # B per expert
params_per_expert_pruned = (PARAMS_MOE_EXPERTS_B * keep_ratio) / n_keep  # same, just fewer experts

active_orig   = PARAMS_NON_MOE_B + N_EXPERT_USED * params_per_expert_orig   * N_MOE_LAYERS / N_TOTAL_LAYERS
active_pruned = PARAMS_NON_MOE_B + N_EXPERT_USED * params_per_expert_pruned * N_MOE_LAYERS / N_TOTAL_LAYERS

print(f"  Original  : {active_orig:.2f}B active params/token  (same expert size, more choice)")
print(f"  Pruned    : {active_pruned:.2f}B active params/token  (same — top-k still fires {N_EXPERT_USED} experts)")
print(f"  Note: active params per token are IDENTICAL — pruning only reduces")
print(f"        model file size and memory footprint, not per-token compute.")

# ── Consistently low-importance experts ──────────────────────────────────────
print(f"\n── Experts consistently ranked low across all layers ───────────────────")
bottom_n = max(1, round(N_EXPERTS * 0.10))  # bottom 10%
low_count = {}
for k in layers:
    scores = data[k][score_field]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i])
    for eid in ranked[:bottom_n]:
        low_count[eid] = low_count.get(eid, 0) + 1

consistent = sorted(low_count.items(), key=lambda x: -x[1])
consistent = [(eid, cnt) for eid, cnt in consistent if cnt >= 3]
print(f"  (bottom 10% in >= 3 layers — most dispensable experts globally)")
print(f"  Expert ID : layers in bottom 10%")
for eid, cnt in consistent[:20]:
    bar = "█" * cnt
    print(f"  Expert {eid:>3} : {cnt:>2}/{n_layers}  {bar}")

print()
print("=" * 70)
