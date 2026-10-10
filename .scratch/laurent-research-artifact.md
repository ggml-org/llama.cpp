# Frame-Lock

- our_fork: Raudbjorn/ggml-llama.cpp @ 2245945dccee30a5222c81c42f33e5482f5d3612 on branch `strata`, working tree clean (verified: `git rev-parse HEAD` = 2245945dc, `git status -s` empty)
- target: LaurentZuijdwijk/llama.cpp @ 11bfe8a633fa02bac251db6cf21bd5ddab282a64 on `master` HEAD
- our backend: SYCL/Intel Arc A770 (acm-g10, Xe-HPG/DG2), oneAPI icpx 2026.x
- our focus: TurboQuant KV-cache (WHT + PolarQuant) at enums 43/44/45, QK_TURBO=128
- target focus: adaptive speculative decoding + Vulkan/ROCm perf on AMD Strix Halo (Radeon 8060S, RADV/Mesa 26.0.8)
- target HEAD date: 2026-09-07 (PR#6 merge). our HEAD date: 2026-10-03. **Our fork is 26 days newer than the target.** `b1e4170ec sync: integrate compatible TheTom changes` (2026-09-05) already absorbed the target's adaptive-spec lineage.
- merge-base with origin/master: 2bec28560f8c679b1f9e803d2949ca740140f239 (2026-09-pre-turbo)
- All target +X% measurements are Strix Halo. Strix Halo != Arc A770 (RDNA 3.5 WGP vs Xe-HPG EU; 32 LDS banks vs 16 SLM banks; RADV/Mesa vs Level Zero+oneAPI). Re-measure on A770 before any commit.

# Phase 0.5 — Framing Pushback

The literal question is stale. Strata already has the target's adaptive-speculation surface:

- `common/speculative-adaptive.h` (3875B, hysteresis state machine) — verified by `ls` and head read
- `common/speculative.cpp:1630-1748, 2119, 2285, 2446-2450` (`adaptive_ctrl` wired)
- `common/arg.cpp:1385-1386, 4438-4441, 4476-4491` (`--spec-draft-n-min-adaptive`, `--spec-draft-n-max`, `--spec-draft-n-min`, `--spec-draft-p-split`, `--spec-draft-p-min`, `--spec-draft-backend-sampling`)
- `src/models/dflash.cpp` (54KB, DFlash2 with `dflash_block_size`, `dflash_selector_rank`)
- `src/models/qwen4exp.cpp` (84KB, MTP driver with `n_layer_nextn`)
- `tests/test-speculative-adaptive.cpp` + `tests/test-qwen4exp-mtp.cpp` (64KB) + `tests/test-qwen4exp-mtp-driver.h` (11KB)
- A/B harness `scripts/perf/bench_spec.py` with 14 result JSONs

Decision forced to (B) technique/algorithm port, restricted to **target-only techniques our strata HEAD has not absorbed and that are SYCL-portable**. The target's README enumerates six components: adaptive speculation (already in strata), ROCmFPx (REJECT — no SYCL path, ABI hazard with our enum reservation per memory bank 2026-07-10), Vulkan LDS SHMEM_STRIDE pad (REJECT — SPIR-V/RADV only), Vulkan IQ3_S NUM_COLS>4 spill (REJECT — SPIR-V), Vulkan tiled concat-transpose + f16 B (REJECT — SPIR-V), backend-agnostic measurement framework (CANDIDATE — the only one that survives).

# Fan-Out Findings

## Inventory of target-HEAD file:line references (all curl-verified against raw.githubusercontent.com/.../11bfe8a6)

| Symbol | Target | Strata | Status |
|---|---|---|---|
| Adaptive controller | `target-speculative-cpp:185, 213-219` (EMA `acc_ema` + `n_want`) | `common/speculative-adaptive.h:39-90` (hysteresis `n_climb`/`n_drop`) | **DIVERGENT — two designs, both wired** |
| `--spec-draft-adaptive` (bool) | `target-arg-cpp:4196-4197` | replaced by `--spec-draft-n-min-adaptive N` (int) at `common/arg.cpp:4438-4441` | Strata more expressive (int=1 is bool degenerate case) |
| `--spec-draft-n-max`/`-n-min`/`-p-split`/`-p-min`/`-backend-sampling` | `target-arg-cpp:4186,4203,4243,4250,4257-4258` | `common/arg.cpp:4388,4398,4476,4483,4490-4491` | IDENTICAL |
| DFlash2 `dflash.cpp` | target's upstream merge dropped `build_post_sampling` per `target-recent-commits[10]` (re-ported 2026-08-30) | `src/models/dflash.cpp` (54KB) | Strata has dflash; selector step unverified |
| `LLM_GRAPH_TYPE_DECODER_MTP` graph type | present in target's `target-llama-context-cpp` head | **NOT FOUND** in strata via targeted grep | **Likely gap** |
| `llm_fused_op_probe` table (FLASH_ATTN, GDN_AR/CH, LIGHTNING_INDEXER, DSV4_HC_PRE/COMB/POST) | present in target | **NOT FOUND** in strata | Likely gap (model-specific) |
| `qsa_pooled_n_dirty_max` write-time bound (target `c659bd6d0` 2026-08-30) | `target-recent-commits[11]` | unverified in strata; `15f318275 fix(qwen4exp): reserve the catch-up graph` may cover it | Verification lead |
| `q8_quants_first` VEC layout | target's Vulkan path (line unverified) | `src/llama-kv-cache.cpp:20-52`, `ggml-sycl/fattn-common.hpp:1496-1497`, `convert.cpp:719,937`, `cpy.cpp:1542-1662`, `set_rows.cpp:378,551-553` | ALREADY IN STRATA |
| `LLAMA_MTP_DRAFT_TOP_K` | not in target's pinned-HEAD index | `5b82ceb6a` adds it; referenced from `common/speculative.cpp` | STRATA-STRONGER |
| `qwen4exp` MTP driver + 1211-line regression test | target has the model only | strata has the driver + `tests/test-qwen4exp-mtp.cpp` (64KB) | STRATA-STRONGER |
| `SHMEM_STRIDE_PAD` constantID=12 for `mul_mm.comp` | target `ggml-vulkan.cpp:4015-4025` | absent | REJECT — SPIR-V/RADV/RDNA 32-bank LDS |
| IQ3_S `NUM_COLS>4` register-spill (5x cost at n=8) | target `ggml-vulkan.cpp:4106`; `mul_mat_vec_iq3_s.comp:8,12,33,67` | routed through SYCL `ggml-sycl.cpp:1380,7034,7795,7845,7898,7919,7937,7957,7970` | REJECT — Vulkan shader; technique transfers in principle but parameter value is different |
| ROCmFPx (Q4_0_ROCMFP4 etc.), `LLAMA_FTYPE` ROCmFPx entries, `ggml-rocm` backend | target README + `target-recent-commits[14]` | absent; `ggml-rocm` does not exist | REJECT — no ROCm; ABI conflict with strata enums 42/43-50 |
| `ggml_gen_hadamard` precomputed rotation init (host) | target `llama-kv-cache.cpp` head (WHT-only, no TurboQuant) | removed in `b1e4170ec` per integration notes "Remove the rotation initializer's 64 KiB temporary stack array" | REJECT — explicit design decision; would regress |

# Phase 4 — Adversarial Self-Attack (top 5 candidates)

## C1 — Acceptance-curve regression test (the target's measurement claim as a CI signal)

- **Source (technique):** target README "Adaptive draft (n_min=3, n_max=7) keeps 96% acceptance ... Fixed n=7 collapses to 18%"; `target-recent-commits[38]` measurement row "n_max=1 0.774->0.829 ... n_max=6 0.373->0.438"
- **Our target surface:** new `tests/test-spec-adaptive-curve.cpp` (or extend `tests/test-speculative-adaptive.cpp`); harness reuses `scripts/perf/bench_spec.py`
- **Falsifier:** A measurement on Qwen3.5-9B on A770 showing **fixed-7 acceptance within 5pp of adaptive-3-7 acceptance** (i.e., the claim does not generalise to Xe-HPG) would falsify the value of porting. Target measured on Strix Halo + qwen3.8-27B; A770 result is unknown.
- **Upstream API contract:** None — test addition only. `common/speculative.cpp:2446-2450` already exposes `n_cur` and `n_accepted` for scraping.
- **Hardware dependency:** none.
- **Backend fit (SYCL/A770):** YES.
- **Hardware gate:** `Qwen3.5-9B` UD-Q4_K_M, A770 sole tenancy, ctx 2048, n_decode 512. Numeric go/no-go: **adaptive-3-7 acceptance >= 0.80 AND adaptive-vs-fixed-7 gap >= 0.20**. If gap < 0.20 on A770, the claim is A770-irrelevant and the test becomes a `KNOWN-FAIL`.
- **Risk:** None. Test-only change.
- **Commit posture:** new test, no behaviour change.

## C2 — `LLM_GRAPH_TYPE_DECODER_MTP` graph type dispatch

- **Source:** `target-llama-context-cpp` `ctx_type_to_graph_type` (target); `src/llama-context.cpp:2480` already special-cases `params.ctx_type == LLAMA_CONTEXT_TYPE_MTP` (strata)
- **Our target surface:** `src/llama-context.cpp` switch on `ctx_type`; `src/llama-graph.cpp:1358,1402,1474` (`t_h_nextn`, `n_layer_nextn` already wired)
- **Falsifier:** If `LLM_GRAPH_TYPE_DECODER_MTP` is already in our tree (a targeted `grep -nE "ctx_type_to_graph_type|LLM_GRAPH_TYPE_DECODER_MTP" src/` returns positive), this is dead. If after adding it the `tests/test-qwen4exp-mtp.cpp` graph-split count goes UP (i.e., the new graph type spawns more splits than the special-case), the dispatch is a regression.
- **Upstream API contract:** `llama_context_type` enum (already in `include/llama.h`); graph-type enum in `src/llama-graph.h`. No external API change.
- **Hardware dependency:** none.
- **Backend fit (SYCL/A770):** YES (host graph construction).
- **Hardware gate:** Run `tests/test-qwen4exp-mtp.cpp` before/after. Numeric go/no-go: **test passes AND graph-split count does not increase**.
- **Risk:** Behavioural — graph split count affects compile time and SLM allocation. Mismatched splits can cause OOM in deep MTP chains.
- **Commit posture:** upstream-style port with attribution; refactor only.

## C3 — `qsa_pooled_n_dirty_max` write-time bound (target `c659bd6d0` 2026-08-30)

- **Source:** `target-recent-commits[11]` "qsa_pooled_n_dirty_max sizes the dirty tables at graph build from the ubatch's own positions; set_input_qsa then recounts n_complete from the cells that are actually filled. Speculative decoding leaves drafted cells past the ubatch's q_max, so the fill-time count can exceed the build-time bound and the n_dirty <= n_dirty_max assert aborts the process mid-request."
- **Our target surface:** `src/llama-memory-hybrid-idx.cpp`; `src/models/qwen4exp.cpp`. **Verify presence first** via `grep -nE "qsa_pooled_n_dirty_max|n_dirty_max" src/`.
- **Falsifier:** If strata's `15f318275 fix(qwen4exp): reserve the catch-up graph of an MTP context` already implements the write-time bound, the candidate is dead. If after backporting the existing strata MTP regression test regresses (because the new bound changes draft accept rate under deep MTP chains), the fix is incorrect for strata's graph layout.
- **Upstream API contract:** None — internal memory accounting.
- **Hardware dependency:** none.
- **Backend fit (SYCL/A770):** YES.
- **Hardware gate:** `tests/test-qwen4exp-mtp.cpp` with `n_max=7` MTP chain, 200+ drafted cells past `q_max`. Numeric go/no-go: **no mid-request abort AND draft accept rate >= 0.80**.
- **Risk:** Behavioural — relaxing a build-time bound to a write-time bound caps overruns silently; if the refill path is missing, downstream cells are -inf-biased.
- **Commit posture:** backport with attribution; add regression test.

## C4 — Vulkan-only techniques (LDS pad, IQ3_S NUM_COLS>4, f16 B, concat-transpose, topk_radix)

- **Source:** `target-llama-context-cpp` parent, `target-llama-graph-cpp`, target's `ggml-vulkan.cpp`
- **Our target surface:** none
- **Falsifier:** A literal port cannot work — SPIR-V `coopMatLoad` does not exist in SYCL. The *technique* (pad SHM stride to be coprime with bank count; hold TPB constant when temp grows) transfers in principle but the constant is hardware-specific (RDNA 32 banks vs A770 16 banks).
- **Upstream API contract:** Vulkan SPIR-V only.
- **Hardware dependency:** RDNA 3.5 + RADV/Mesa Vulkan.
- **Backend fit (SYCL/A770):** **NO — ADAPT-INCOMPATIBLE.**
- **Risk:** n/a
- **Commit posture:** REJECT.

## C5 — ROCmFPx, ggml-rocm backend, `LLAMA_FTYPE` ROCmFPx entries

- **Source:** target README; `target-recent-commits[14]` (16 ftype entries + 22 ftype name/type mappings)
- **Our target surface:** would require `ggml/src/ggml-rocm/` (absent), `ggml/include/gguf-q4_0_rocmfp4.h` (absent), and ABI additions to `ggml/include/ggml.h` (strata COUNT=51 per `ggml-common.h:260-343`; next free slot 51 — collides with our CR=48-50 reservation)
- **Falsifier:** Per memory bank 2026-07-10, donor enums (target's ROCmFPx at 42+) collided with target's existing Q2_0=42; same hazard applies to strata. The fork's TurboQuant/TQ/CR enum scheme (43-50) is a deliberate reservation; adding ROCmFPx would re-collide.
- **Upstream API contract:** ABI break to GGUF serialization.
- **Hardware dependency:** ROCm/HIP only.
- **Backend fit (SYCL/A770):** **NO — ADAPT-INCOMPATIBLE.** No FP4 in oneMKL 2026.0; no ROCm runtime.
- **Risk:** ABI corruption, kernel-dispatch table corruption, TurboQuant regression.
- **Commit posture:** REJECT.

# Phase 5 — Committed Recommendations (>=3 ranked)

| Rank | Candidate | Backend fit | Owner-time | Risk | Commit posture |
|---|---|---|---|---|---|
| 1 | C1 — Acceptance-curve regression test | YES | lo (1 day) | none (test only) | new test, no behaviour change |
| 2 | C2 — `LLM_GRAPH_TYPE_DECODER_MTP` graph type (verify first) | YES | lo (1 day after verify) | behavioural (graph splits) | upstream-style port |
| 3 | C3 — `qsa_pooled_n_dirty_max` write-time bound (verify first) | YES | lo (1 day after verify) | behavioural (refill path) | backport with attribution |
| 4 | C4 — Vulkan-only techniques | NO | n/a | n/a | REJECT |
| 5 | C5 — ROCmFPx | NO | n/a | n/a | REJECT |

# Monday Start

Do **C1** (acceptance-curve regression test) on Monday. The first cell to measure:

```
pre:   ls build-sycl/bin/llama-cli 2>/dev/null   # must exist; else rebuild JIT
guard: sudo systemctl stop llama-sycl.cpp.service; fuser -v /dev/dri/renderD128
dmesg | grep -iE 'reset|hang|hung|timed?out|GuC|wedged|banned|page.?fault|device.?lost'
# (no GPU errors before/after per AGENTS.md discipline)

build:    ./build-sycl/bin/llama-bench -m /path/to/Qwen3.5-9B-UD-Q4_K_M.gguf -ngl 99 -fa 1 -p 512 -n 128 -r 5
spec:     ./build-sycl/bin/llama-cli   -m /path/to/Qwen3.5-9B-UD-Q4_K_M.gguf -ngl 99 -fa 1 \
          -c 2048 -n 512 --spec-type draft-mtp --spec-draft-n-min 3 --spec-draft-n-max 7 \
          --spec-draft-n-min-adaptive 3
fixed-3:  same with --spec-draft-n-min 3 --spec-draft-n-max 3 --spec-draft-n-min-adaptive 3
fixed-7:  same with --spec-draft-n-min 7 --spec-draft-n-max 7 --spec-draft-n-min-adaptive 7
```

Numeric go/no-go: **adaptive-3-7 acceptance >= 0.80 AND adaptive-vs-fixed-7 gap >= 0.20**. If gap < 0.20, the claim does not generalise to A770 and the test is a `KNOWN-FAIL` documenting the divergence. If Tuesday is open, run C2 and C3 verifications in parallel (one `grep` each, 5 minutes total).

# Meta-Observation

**Target has that strata does not:** EMA-based adaptive controller (`acc_ema` + `n_want`); the `qsa_pooled_n_dirty_max` write-time bound (unverified); `LLM_GRAPH_TYPE_DECODER_MTP` graph type dispatch (unverified); `llm_fused_op_probe` table (GDN_AR/CH, LIGHTNING_INDEXER, DSV4_HC_PRE/COMB/POST); ROCm/HIP backend + 16 ROCmFPx `LLAMA_FTYPE` entries + ROCmFP4-FAST GGUF; Strix-Halo-specific SPIR-V shader work (LDS pad, coopmat f16 B, tiled concat-transpose, IQ3_S NUM_COLS>4 fix, topk_radix); precomputed `ggml_gen_hadamard` rotation init at kv-cache time (host, WHT-only — no TurboQuant).

**Strata has that target does not:** Hysteresis adaptive controller (discrete climb/drop with depth-aware thresholds) — strictly more expressive than the target's EMA + boolean; `LLAMA_MTP_DRAFT_TOP_K` constant and the strata fix to size the MTP backend draft sampler by it; 1,200-line MTP driver regression test (`tests/test-qwen4exp-mtp.cpp`); A/B harness `scripts/perf/bench_spec.py` with 14 result JSONs; TurboQuant KV-cache (TURBO2/3/4/TQ/CR enums 43-50); SYCL backend (~110 files, XMX experimental, VEC default, MMVQ, SET_ROWS turbo quantize, WHT, InnerQ); `q8_quants_first` VEC layout (`Q8_KV_QUANTS_FIRST_BLOCKS=4`); layer-adaptive TurboQuant policy (`TURBO_LAYER_ADAPTIVE` env, modes 0-7); A770 xe/i915 kernel-driver switch documentation; 30+ strata-specific `fix(qwen4exp)` and `fix(spec)` commits; the `dflash.cpp` DFlash2 model (54KB) with `dflash_block_size` and `dflash_selector_rank`.

**Net assessment:** The honest framing is **strata is a strict superset of target on the host/spec surface and a strict disjoint on the backend surface (SYCL vs Vulkan+ROCm).** The target's adaptive-speculation work is in our fork via `b1e4170ec`; the Vulkan/ROCm work is irrelevant on SYCL/A770. The residual delta on our hardware is **C1 (the measurement-claim regression test, the only one that ships this week) plus two unverified gaps (C2 graph type, C3 dirty-table bound) that are 1-day verifications each.** If the C2/C3 verifications both return positive, the residual delta collapses to **C1 alone** — and the answer to the original question becomes: "nothing to port; maintain the strata trajectory; lift the measurement framework as a CI signal."

**LLM-bias guard:** The recommendation is grounded in (a) direct `git log`/`grep` of the working tree at the pinned SHA, (b) direct `curl` against the target's pinned-HEAD raw URLs, and (c) a falsifier attached to every candidate. The honest "almost nothing to port" answer is the opposite of what the question's framing implies; the artifact is the second-pass answer, not the first-pass confirmation. The Rank 1 recommendation (acceptance-curve regression test) is the only candidate that survives all three filters: SYCL-portable, not already in strata, and falsifiable on A770.
