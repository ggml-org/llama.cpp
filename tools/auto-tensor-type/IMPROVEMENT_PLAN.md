# auto-tensor-type — improvement plan

Context: on a 40-layer Qwen3.5 hybrid MoE, at **matched actual BPW**, auto-tensor-type recipes give
**+33–64% higher end-to-end KL divergence** than llama.cpp's hand-tuned recipes across 2.25–4.75 bpw.
It starves the small-but-critical tensors the hand recipes protect (attn_k/v → Q8_0, shared experts →
Q8_0/Q5_K) and over-spends token_embd. This plan fixes that.

---

## 0. Reconciled diagnosis (what's actually wrong)

Three mechanisms, in order of how much they explain the gap:

1. **The cost metric is local and can't see downstream amplification.** The per-choice cost is
   `relative_L2 = ‖ΔW·x‖²/‖y‖²` on one MUL_MAT's output rows. This is the *input-side* Fisher
   (`E[x xᵀ]` is literally the imatrix) with **no output-side leverage term** — it has no idea that a
   K/V perturbation propagates through the attention softmax across all layers, that shared experts
   touch every token, or that ffn_down/attn_output write into the residual stream and compound. The
   true objective is `ΔWᵀ (E[x xᵀ] ⊗ G_role) ΔW`; the tool sets the leverage `G_role ≡ 1/‖y‖²` for
   every role. **This is the root cause.**

2. **Fused-QKV dilution.** In the fused-attention layers Q/K/V are one `attn_qkv` tensor. `compute_avg_kld`
   averages relative-L2 over all output rows — Q has `n_head` rows, K/V only `n_kv_head` (often 8:1), so
   the mean is mechanically Q-dominated. `split_fused_roles` then hands K/V **5.6% each** of that
   already-Q-dominated number. K/V sensitivity is invisible to the optimizer for these layers.

3. **Element-weighting — subtle, and NOT the whole story.** The DP minimizes `Σ relative_L2 × n_elements`.
   The natural reading ("small tensors get starved") is only half-right: the Lagrangian relaxation
   `argmin_q [L2(q) + λ·bpw(q)]` is **size-independent** (the `n_elements` factor cancels), so a small
   tensor with genuinely high *measured* L2 would still win bits. Element-weighting is therefore a
   secondary lever, not the primary bug — but it *is* a useful lever for pricing big tensors' bits (see 2.2).
   The starvation we observed is mostly (1)+(2) making the *measured* L2 of attn_k/v/shexp too small.

**Honest goal.** The hand rules in `llama-quant.cpp` encode years of tuning but only for *classic* ftypes
and know nothing about this branch's new trained quants (Q3_PT, IQ2_TQ, IQ3_TQ, Q4_DPT, Q2_KPT…). Frame
success as: **parity with hand recipes on classic roles + correct placement of the new low-bit types +
guaranteed 100% tensor coverage** — not "beat hand-tuning with a local metric" (which is capped).

---

## 1. Foundation — correctness (ship first)

- **1.1 Sharded-model loading — DONE.** auto-tensor-type read only the first GGUF shard, so on a split
  model it saw 23 of 40 layers and emitted a recipe covering 44% of the model; the rest silently fell
  back to the base ftype (this invalidated our first comparison — auto IQ2_XXS came out at 3.38 bpw not
  2.25). Fixed via `open_model_shards()` (reads `split.count`/`split.no`, `llama_split_prefix/path`);
  Phase-1 now enumerates every shard. Verified: 40 layers / 472 tensors on the split model, no merge needed.

- **1.2 100%-coverage guard (do next, cheap, high value).** After emitting a recipe, dump the intended
  per-tensor type and assert every quantizable tensor is matched by some pattern; **fail loudly** if any
  would fall back to the base ftype. This is the exact confound that hid 1.1 for hours. A `--verify-coverage`
  that re-reads the output GGUF and diffs against the recipe closes it permanently.

- **1.3 Fused-QKV: measure Q/K/V sub-ranges separately.** The captured fused output already contains Q, K,
  V as contiguous row-ranges (boundaries from `head_dim`, `n_head`, `n_kv_head`). In `build_cost_matrix`,
  split the ref/quant output, compute `L2_Q/L2_K/L2_V` each over its own rows, and cost the (single-type)
  fused tensor by **`max(L2_Q, L2_K, L2_V)`** (or a high percentile) instead of the row-mean. Replace the
  `fused·fraction` dilution in `split_fused_roles` with the component's own measured L2. (llama.cpp
  quantizes attn_qkv as one tensor → one type, so the goal is a K/V-aware *shared* decision, not per-Q/K/V.)

## 2. Quick high-ROI wins (cheap; reuse the cost-matrix cache, sweep in Phase-4)

The ~40-min cost is Phase 2+3 and is already `--cost-matrix-cache`'d. The DP (Phase 4) is milliseconds, so
every knob below is **free to sweep** once one cache exists. Rank accordingly.

- **2.1 Per-role floors seeded from the hand pins (highest leverage / lowest effort).** In the DP item
  builder, forbid choices below a per-(role,bucket) floor derived from `llama_tensor_get_type_impl`:
  attn_k/v ≥ Q6_K (Q8_0 when n_expert≥8), shared experts ≥ Q5_K, ffn_down first-`n/8`/`use_more_bits`
  layers +1 tier, attn_output ≥ Q5_K for MoE, output ≥ Q6_K. Uses the existing `1e30` forbidden-sentinel;
  the DP's overshoot fallbacks handle infeasible-at-low-target gracefully. This alone should close most of
  the gap. Generalize later to a **deviation-penalty** objective `error + λ·max(0, bpw_ref − bpw_chosen)·ne`
  (λ→∞ = hand recipe, λ=0 = today), so the DP starts from the hand prior and only deviates on strong evidence.

- **2.2 Element-weight exponent.** `cost = relative_L2 × n_elements^γ`, γ≈0.5, as `--element-gamma`
  (keep the *bit* cost at true `n_elements·bpw` so the budget stays exact). Effective bit-price becomes
  `λ·ne^(1-γ)`: big tensors priced higher, small sensitive tensors can afford more bits. Tune **after** 1.3
  lands (else it compensates for the fused bug and over-corrects).

- **2.3 token_embd floor fix.** It's floored to Q5_K and booked at `closest_bpw(target)`; hand recipes use
  IQ4_XS/Q4_K for non-tied embeddings. Keep the high floor only for *tied* embeddings. Frees budget the DP
  redeploys to amplified roles.

- **2.4 importance_alpha sweep.** Already implemented, defaulted off. Cheap to try, but it targets
  low-energy *write* tensors (ffn_down/attn_output) — it will **not** fix the attn_k/v (high-energy *read*)
  starvation. Don't oversell; 2.1 is the real fix.

## 3. Architecture pre-calibrator (the strategic fix)

Both the metric-redesign and the pre-calibrator analyses converge on the **same construct**, which is the
right long-term answer and directly implements the "measure real per-role sensitivity" idea.

**Measure a per-(role,bucket) end-to-end sensitivity `s_rb` and fold it into the objective.**

- **Method — calibrated noise injection (no re-quantization).** For each role, on the already-loaded
  fp16 model in Phase-2 scope: (a) one clean forward → reference logits at a strided ~64–128 positions;
  (b) install an eval callback that, at every MUL_MAT whose `src[0]` is that role, adds row-wise Gaussian
  noise so the injected per-row relative-L2 equals ε (paired RNG across roles so common-mode cancels);
  (c) `s_rb = logitKLD(ref, perturbed) / ε²`. KL is locally quadratic (Fisher), so `logitKLD ≈ s·ε²` and
  `s_rb` is the amplification the local metric can't see. Probe at two ε (expect 4× scaling) as a validity check.
- **Cost:** ~1 reference + ~12 roles × 2 ε ≈ 25–35 sub-second forward passes on the resident model
  (~1–2 min vs Phase-3's ~40 min). **Cacheable** — `s_rb` is independent of target-bpw and the quant list,
  so it belongs in the cost-matrix cache (bump to v3) and sweeps stay free.
- **Integration:** mirror `apply_importance_alpha`. With `ρ_rb = (s_rb / n_elements_rb) / geomean`, multiply
  each cell by `clamp(ρ_rb, 1/C, C)^γ`. The DP's existing `× n_elements` then makes the **effective weight**
  `n_elements^(1-γ) · (s_rb/geomean)^γ` — i.e. γ interpolates from today's element-weighting (γ=0) to the
  additive-model-correct `Σ s·relative_L2` (γ=1). No DP change. New flag `--arch-calib-gamma`, default 0
  (byte-identical unless requested).
- **Guards:** positive-control self-test (perturb `output`; if logitKLD==0 the injection isn't propagating →
  abort, fall back to element-weighting); the two-ε quadratic check; `ρ` clamp + γ<1; and a built-in
  end-to-end validator that predicts `Σ s·relative_L2` for the chosen assignment and compares to a measured
  perturbation pass **before** spending 40 min quantizing.
- **Later:** a backward-pass Fisher estimator (same `s_rb` slot, ~1–2 backward passes instead of N forward
  passes) once ggml autodiff is wired; and post-nonlinearity *local* metrics (post-attention error for K/V,
  post-SwiGLU error for gate/up) that improve the per-cell shape orthogonally to `s_rb`.

## 4. Validation harness (gate every change on this)

- **Primary metric: end-to-end KLD vs the BF16/F16 reference** (`llama-perplexity --kl-divergence`). Never
  grade on the tool's own relative-L2 (circular).
- **Matched *actual* BPW** measured from the output GGUF, not `--target-bpw` — a recipe 0.05 bpw heavier
  "wins" trivially.
- **100%-coverage assertion** (see 1.2): any tensor that fell back to the base ftype → discard the run.
- **Report the tail** (Max KLD, p99-token KLD), not just the mean — starvation shows up as tail blowups.
- **Three architectures always**: dense (Qwen3.5/Llama-3-8B), classic n_expert=8 MoE, hybrid MoE. A knob
  tuned only on the hybrid is overfit; the other two are regression gates.
- Script it under `scripts/`: `(model, imatrix, recipe-gen, bpw-list) → quantize → measure actual bpw →
  verify coverage → KLD → CSV(recipe, target_bpw, actual_bpw, mean/p99/max_kld, coverage_ok)`.

## Sequencing

1. 1.2 coverage guard + 1.3 fused-QKV fix (correct the *inputs*), on top of the shipped 1.1.
2. Build the §4 harness; establish the baseline gap on 3 models.
3. 2.1 per-role floors → measure. Expect most of the gap to close here.
4. 2.2 element-γ + 2.3 token_embd → sweep on the harness (cache reuse = free).
5. 3. arch pre-calibrator behind `--arch-calib-gamma`; sweep γ∈{0,.25,.5,.75,1}; ship the default that wins
   on all three architectures. Promote 2.1 floors into the deviation-penalty prior.
6. Deprioritize: full per-cell logit-KLD and Fisher/autodiff until the harness shows a residual gap the
   prior + `s_rb` can't close.

Shelve as expensive/overfit-prone until proven needed: full brute-force end-to-end per-(cell,type) probing,
and any per-architecture sensitivity table not cached-and-validated on a held-out model of that family.

---

## Results & status (2026-07-08 implementation pass)

Measured on Ornith-1.0-35B (40-layer Qwen3.5 hybrid MoE), end-to-end KLD vs BF16 base, matched actual BPW.

**SHIPPED & validated:**
- **Sharded loading** (`open_model_shards`): reads all shards; 40 layers / 472 tensors on the split model.
- **Coverage guard**: aborts if any quantizable tensor is unmatched by the recipe.
- **`--element-gamma G`** (DP cost = `relative_L2 * n_elements^G`; bit cost keeps true n_elements; DP-time,
  sweeps free off the cost cache). **This beats hand-tuning at every BPW tested:**

  | BPW | Bartowski mean/median | auto G=0.25 mean/median | Δ |
  | --- | --- | --- | --- |
  | 2.91 | 0.150 / 0.068 | 0.127 / 0.059 | −15% / −12% |
  | 3.74 | 0.060 / 0.028 | 0.0465 / 0.021 | −23% / −25% |
  | 4.34 | 0.030 / 0.0146 | 0.0257 / 0.012 | −15% / −18% |

  At 3.74 the curve plateaus by G≈0.1 (G=0.1 == G=0.0 = 0.0478; G=0.1 has the better tail, max 4.16 vs
  0.25's 5.48). **Recommended default G≈0.1–0.25.** (One caveat: tail/Max KLD is noisier at low G — the
  fused-QKV fix and pre-calibrator should recover it.)

**FIXED & working — `--arch-calib-gamma` (pre-calibrator), weight-level perturbation:**
The original in-graph output perturbation was abandoned (writing an intermediate MUL_MAT output buffer
via `ggml_backend_tensor_set` in the eval callback doesn't propagate — ggml_backend_sched buffer reuse
defeats it; logitKLD came out ε-independent and only `hits=1`). Replaced with **weight-level
perturbation**, which measures the right quantity anyway:
- CB_COLLECT pass records `ggml_tensor*` handles for every target-role weight (from `t->src[0]`).
- Per role: add deterministic per-(weight,row) seeded Gaussian noise scaled to ε·RMS(row) directly into
  each weight buffer, one clean forward, logit-KLD vs reference, then restore by subtracting the identical
  seeded noise. Merged model is **BF16** so perturb/restore bf16↔f32 convert (`weight_to_f32`/`f32_to_weight`).
- **Perf fix (this session):** the per-element RNG was single-threaded and choked on MoE experts
  (`ffn_gate_exps` ≈ 10^10 elements → >10 min/role). Parallelized the row loop with OpenMP + per-row RNG
  seeding (`row_seed(base, name, row)`) so perturb and restore reproduce identical noise independently per
  row. Had to link OpenMP into the target (`find_package(OpenMP)` in CMakeLists — the pragmas were being
  silently ignored as `-Wunknown-pragmas`). Now all 18 roles probe in a few minutes.
- **Guards:** noise-floor check (clean-vs-clean logitKLD must be 0 — it is, forward is deterministic);
  `output` positive control (KLD scales super-linearly with ε, ratio ≈ 2.75 at eps=0.02); role-spread
  validity (disable if max/min < 2.0). On Ornith: spread 2.09–2.72, s_role differentiates
  ssm_out/ffn_gate_exps/attn_qkv (high ≈ 21–26) from attn_k/attn_gate (low ≈ 12–14). Cached to `<cache>.sens`.
- **Known footgun:** on a cost-matrix cache *hit* with `--arch-calib-gamma>0` but a missing `.sens`, the
  probe does NOT re-run (the model isn't loaded on the cache-hit path) — it logs an error and proceeds with
  arch-calib off. So the cache-building run must itself enable arch-calib. (Candidate hardening: hard-abort
  instead of silently continuing.)

**FIXED & working — fused-QKV separate Q/K/V measurement (§1.3):**
`component_rel_l2()` + `compute_fused_qkv_kld()` split the captured fused `attn_qkv` output into contiguous
Q/K/V feature-ranges (`qsize=n_head·head_dim`, `ksize=vsize=n_head_kv·head_dim`) and cost the tensor by
`max(L2_Q, L2_K, L2_V)` instead of the Q-dominated row mean. Geometry read from
`<arch>.attention.head_count/head_count_kv/key_length`; guarded by `qkv_q+2·qkv_kv==ref_ne0` (falls back to
plain rel-L2 if geometry mismatches, e.g. value_length≠key_length). On Ornith (n_head=16, n_head_kv=2,
head_dim=256 → 4096+512+512=5120) the guard fires for the 30 fused-attention layers; the 10 full-attention
layers use separate attn_q/k/v roles.

**RESULT — arch-calib is a tail-regularizer, not a mean-improver (Ornith, ~3.76 bpw, KLD vs BF16):**

| recipe | (elem-γ, arch-γ) | bpw | Mean KLD | Median | Max KLD |
| --- | --- | --- | --- | --- | --- |
| baseline (element-weighted DP) | (1, 0)    | 3.762 | 0.0988 | 0.0537 | 3.99 |
| **element-γ heuristic**         | (0.25, 0) | 3.760 | **0.0465** | 0.0212 | 5.48 |
| arch-calib half                 | (1, 0.5)  | 3.762 | 0.1064 | 0.0568 | 4.28 |
| arch-calib full ≈ Σ s·L2        | (1, 1.0)  | 3.761 | 0.0706 | 0.0355 | 8.04 |
| **combined**                    | (0.25, 0.5) | 3.759 | 0.0478 | 0.0219 | **3.95** |

Readings:
1. **The pre-calibrator measures real signal.** At (γ_elem=1, γ_arch=1) the DP's `× n_elements` cancels the
   `1/n_elements` in `ρ=s_role/n_elem`, so the objective becomes ≈ `Σ s_role·relative_L2` — the
   additive-model-correct cost. It beats the naive element-weighted baseline (0.0706 vs 0.0988, **−29%**),
   confirming `s_role` captures downstream amplification the local L2 can't see.
2. **But it does not beat the cheap element-γ knob on mean.** element-γ=0.25 alone (0.0465) is the best mean
   recipe; arch-calib-full (0.0706) loses to it, and the γ_arch=0.5 half-measure (0.106) is worst-of-both
   (keeps partial n_elem weighting *and* adds the arch push).
3. **arch-calib's real value is the tail.** element-γ=0.25 alone regresses the tail (Max 5.48 > baseline
   3.99). Layering arch-calib 0.5 on top (**combined**) keeps mean statistically tied (0.0478 vs 0.0465,
   Δ≈1.3σ, within noise) at *lower* bpw (3.759 < 3.760) while cutting Max KLD **5.48 → 3.95 (−28%)** — the
   best tail of any recipe, baseline included. arch-calib-full over-concentrates (Max 8.04) — moderate γ is key.

**Takeaway (MoE/Ornith):** element-γ≈0.25 for best mean; **element-γ=0.25 + arch-calib-γ=0.5** when
worst-case token KLD matters (near-free mean, −28% tail). On MoE the pre-calibrator's earned role is tail
control. **But see the dense result below — on dense/hybrid archs arch-calib becomes the decisive MEAN win.**

---

## Dense/hybrid result — arch-calib is a MEAN-KLD win that beats hand-tuning (Qwen3.5-9B, 2026-07-09)

Validated on **Qwen3.5-9B** (dense hybrid: attention + SSM + dense FFN, no MoE; MTP head on blk.32).
KLD vs BF16 on the r/LocalLLaMA comparison's exact eval (cmhamiche gist, `-c 512`), imatrix from
`combined_all_small.txt`. Head-to-head vs the best public GGUFs, all measured on one consistent base.

**Headline: at ≥5 bpw our element-γ recipes beat the field** — vs Unsloth **UD-Q4_K_XL (the thread's SOTA)**
0.01485 vs 0.01705 (**−12.9%**) and Bartowski Q4_K_M −15.6% at matched bpw. **We lost only at the dense
~4.6–4.8 bpw IQ4_XS sweet spot** — until arch-calib:

| recipe @ ~4.78 bpw | (elem-γ, arch-γ, floor) | Mean KLD | vs Bartowski |
| --- | --- | --- | --- |
| **OURS floor + arch-calib** | (1.0, **0.5**, IQ4_XS) | **0.02464** | **−7.1%** |
| OURS floor + arch-calib | (1.0, 1.0, IQ4_XS) | 0.02484 | −6.4% |
| Bartowski IQ4_XS (hand pins) | — | 0.02653 | — |
| OURS floor, no arch | (1.0, 0, IQ4_XS) | 0.02653 | tie |
| OURS floor, no arch | (1.5, 0, IQ4_XS) | 0.02727 | +2.8% |
| OURS full quants, arch only | (1.0, 1.0, none) | 0.03057 | +15% |

Findings that overturn the MoE-only reading:
1. **arch-calib is the entire margin.** floor-only ties Bartowski (0.02653); adding arch-calib-γ=0.5 →
   0.02464. Optimum at γ≈0.5 (γ=1.0 slightly over-concentrates). The floor is still required — arch-calib on
   the full quant set (no floor) still makes bad low-bit drops (0.03057).
2. **Mechanism: the probe found an architecture-specific sensitivity hand-tuning misses.** On this hybrid,
   `ssm_out` sensitivity `s=39` — ~10× everything else (attn_qkv 3.8, attn_v 3.6, ffn ~2.2, output 1.5,
   `attn_q` 1.1 = least; spread 53.8). The winning recipe bumped `ssm_out`→Q6_K on every layer and
   `attn_v`→Q6_K (floor-only left `ssm_out` at ~IQ4_XS). **Bartowski's transformer-era hand rules don't
   special-case the SSM output projection; the data-driven end-to-end probe does.** This is the pre-calibrator
   earning its keep — arch-specific critical tensors that generic hand-tuning can't know about.
3. **The element-γ knob is really an activation-sparsity correction, not a bpw knob.** γ<1 helps MoE
   (over-priced sparse experts, Ornith won at 2.9–4.3 bpw); on dense FFN (used every token) γ<1 *hurts* and
   the optimum is γ=1.0 (γ>1 craters small tensors, U-shaped). Detect MoE-vs-dense from roles and set the
   default accordingly.
4. **Not an imatrix effect.** Equal-imatrix shootout: `combined_all_small` beats Bartowski's own imatrix
   (plain IQ4_XS 0.02749 vs 0.02832) — our calibration is the *better* one; the win is purely recipe.

**Method boundary, revised:** wins on MoE (all tiers) and on dense at ≥5 bpw *and* — with floor + arch-calib —
now also at the dense IQ4_XS sweet spot. **Recommended dense default: floor at the uniform-bpw type +
element-γ=1.0 + arch-calib-γ≈0.5.**

**SHIPPED — architecture-aware auto-config (default on; `--no-auto-config` to disable).** The tool detects
MoE vs dense from the roles (`_exps`/`gate_inp`/`shexp`) and, for any knob not set explicitly, applies the
winning strategy automatically: MoE → element-γ=0.25; dense → element-γ=1.0 + floor the quant set at the
target's uniform type (`dense_floor_bpw`) + arch-calib-γ=0.5. Plus an **imatrix-less auto-pin**: any tensor
with no imatrix entry (an MTP/`nextn` head a forward pass never exercises) is pinned to IQ4_XS at the recipe
top so llama-quantize never aborts — no more manual `blk.N=q4_K` patch. (`--element-gamma` clamp lifted to
[0,2] for the dense-side γ.) **Acceptance:** on Qwen3.5-9B with zero strategy flags and no hand-patching the
auto path reproduces the hand-tuned win — **0.024595 @ 4.775 bpw, beats Bartowski (0.02653) by −7.3%**,
matching the manual floor+arch recipe (0.02464). Remaining: commit; regression-test the MoE auto path
(Ornith) still lands element-γ=0.25; consider the deviation-penalty prior (§2.1) for an even tighter floor.
