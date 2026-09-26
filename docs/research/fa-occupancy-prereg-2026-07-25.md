# FA occupancy investigation - pre-registered predictions

Recorded 2026-07-25, before any measurement, so the experiment can fail honestly.

Base: `perf/fa-occupancy` at `64f5307b7` (equal to `master`; the post-P5 campaign state).
Device: Intel Arc A770, DG2/acm-g10, i915, Level Zero 1.15.38646.
Build: Release JIT, icpx 2026.0, `GGML_SYCL=ON`, `GGML_SYCL_TARGET=INTEL`, `GGML_SYCL_DEVICE_ARCH` omitted.

## Claim under test

`ggml/src/ggml-sycl/ggml-sycl.cpp:158` derives the flash-attention occupancy
governor by dividing a work-item count by an execution-unit count:

```cpp
info.devices[gpu_index].max_wg_per_cu = info.max_work_group_sizes[gpu_index] / prop.get_max_compute_units();
```

`clinfo` on this host reports `Max compute units 512` and `Max work group size 1024`,
so the expression evaluates to `1024 / 512 = 2`. `fattn-common.hpp:1248` consumes it
as `max_blocks_per_sm`.

The same two lines are present verbatim in upstream `ggml-org/llama.cpp` master.

## Derived prediction for Phase 0

Both fleet models are 32 query heads / 8 KV heads (GQA 4:1), D=128, 32 layers.
`GGML_SYCL_TARGET=INTEL` compiles `GGML_SYCL_WARP_SIZE=16`, and the VEC path uses
`nthreads = 128`, so `block_dim = (16, 8, 1) = 128` work-items.

Tracing `launch_fattn` for a single-token decode at any depth >= 4096:

| Quantity | Predicted value | Source |
|---|---:|---|
| `nsm` | 32 | `512 / 16` (`ggml-sycl.cpp:152`) |
| `max_wg_per_cu` | 2 | `1024 / 512` (`ggml-sycl.cpp:158`) |
| `ntiles_total` | 32 | `ntiles_x(1) * ntiles_z_gqa(4) * K->ne[2](8) * Q->ne[3](1)` |
| `parallel_blocks` | 2 | clamped by `max_blocks_per_sm`; search loop breaks at 100% "efficiency" |
| `blocks_total` | 64 | `1 * 2 * 32` |
| `work_items_total` | 8192 | `64 * 128` |

A770 residency is 32 Xe-cores x 16 XVE x 8 hardware threads x SIMD16 = 65 536
work-items, so 8 192 work-items is approximately **12.5 % occupancy**.

The search loop at `fattn-common.hpp:1276` is predicted to exit after its first
iteration: at `parallel_blocks_test = 2`, `nblocks_total = 64` equals
`blocks_per_wave = nsm * max_blocks_per_sm = 64`, giving `nwaves = 1` and
`efficiency_percent = 100`. The next iteration then satisfies
`efficiency_percent_best >= 95 && nwaves > nwaves_best` and breaks.

## Decision rule

- Observed `parallel_blocks == 2` and `blocks_total == 64` -> diagnosis confirmed; proceed to Phase 1.
- Any other value -> the diagnosis is wrong. Halt, re-open the analysis, and re-rank the program.
  Phases 1 and 2b lose their justification; Phases 3-6 survive on independent evidence
  but their order changes.

## Phase 1 predictions (recorded now, measured later)

With `max_wg_per_cu` corrected to 8:

1. `parallel_blocks` rises to 8, `blocks_total` to 256, `work_items_total` to 32 768.
2. d16384 f16 tg128 improves; effective requested-KV bandwidth moves off the
   124 GB/s recorded in P5.4.
3. **The q8_0-vs-f16 gap narrows.** This is the falsifiable core. If the gap does not
   narrow, the dequant-ALU-wall hypothesis is correct and a genuine split-KV q8_0
   kernel rewrite becomes justified rather than premature.
4. Depth 0 is neutral, because `parallel_blocks` is clamped by `ntiles_KQ` which is
   small at shallow depth. Any depth-0 regression below -2 % fails the shallow guard.

## Phase 0 result - CONFIRMED

Measured 2026-07-25 on the build above. Mistral-7B-Instruct v0.1 Q4_K_M,
`llama-bench -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 32 -d 16384`, `GGML_SYCL_FA_PROFILE=1`:

```
GGML_SYCL_FA_PROFILE: route=VEC layout=canonical launches=1056
  conversion_us=0 conversion_bytes=0 stage1_us=3650250 combine_us=76821
  gqa=4 repeated_packed_kv_bytes=111286419456
  parallel_blocks=2 ntiles_total=32 blocks_total=64 work_items_total=8192
  max_wg_per_cu=2 nsm=32
```

Every one of the six predicted fields matched exactly. The diagnosis is confirmed:
8 192 of ~65 536 resident work-items, approximately **12.5 % occupancy**.

Two findings beyond the prediction:

**1. The 64-work-group ceiling is universal across decode routes, not specific to VEC.**

| Route | KV | `parallel_blocks` | `ntiles_total` | `blocks_total` | `work_items_total` |
|---|---|---:|---:|---:|---:|
| VEC | q8_0 | 2 | 32 | 64 | 8192 |
| TILE | f16 | 8 | 8 | 64 | 8192 |
| VEC | f16 (forced) | 2 | 32 | 64 | 8192 |
| TILE | q8_0 prefill | 2 | 4096 | 8192 | 1048576 |

All three decode configurations converge on exactly `blocks_per_wave = nsm * max_wg_per_cu = 64`.
TILE reaches it through `ncols2 = 4` GQA packing (`ntiles_total` 32 -> 8) combined with an
8-way KV split; VEC reaches it through 32 tiles at 2-way split. The efficiency-search loop
at `fattn-common.hpp:1276` demonstrably works - it grew TILE's `parallel_blocks` from 2 to 8
to reach the target - which confirms 64 is precisely the ceiling it is aiming at.

This **corrects** an earlier claim that TILE outperforms VEC at depth because it has a
less-starved grid. It does not; the grids are identical in size. TILE's deep-context
advantage comes from work distribution instead: each work-group walks one eighth of the
KV rather than one half, and four query heads share each K/V load.

Prefill is unaffected - it launches 8 192 work-groups and is not occupancy-limited.

**2. The combine kernel is cheap, which bounds the risk of raising `parallel_blocks`.**

Over 1 056 decode launches: `stage1_us = 3 650 250` versus `combine_us = 76 821`. Combine is
**2.1 %** of flash-attention time. Raising `parallel_blocks` from 2 to 8 scales combine by
about 4x, to roughly 8 % of FA time, while giving stage 1 four times the memory-level
parallelism. The trade is heavily favourable unless stage 1 fails to improve.

### Measurement caveat

The render node was not under sole tenancy during these runs (`codium-insiders` and
`qoder` held `/dev/dri/renderD128`). Launch geometry is a pure function of device
properties and tensor shapes and is unaffected by contention, so the Phase 0 conclusion
stands. The throughput figures observed alongside it (q8_0 VEC 6.88 t/s, f16 forced-VEC
13.99 t/s, f16 TILE 14.83 t/s at d16384) are **not** promotion evidence and must not be
compared against the campaign baseline below. Phase 1 requires sole tenancy.

## Phase 1 result - prediction 2 confirmed, prediction 4 initially FAILED

### First attempt (`d87f024e4`) was rejected by the gate

A partial paired campaign under verified sole tenancy completed three cells before
aborting on a tenancy violation. It rejected the change:

| cell | metric | wg=2 | wg=16 | delta |
|---|---|---:|---:|---:|
| d0 f16 | pp512 | 512.90 | 435.57 | **-15.08 %** |
| d0 f16 | tg128 | 23.94 | 23.85 | -0.39 % |
| d0 q8_0 | pp512 | 537.74 | 519.01 | **-3.48 %** |
| d0 q8_0 | tg128 | 24.58 | 24.57 | -0.05 % |
| d4096 f16 | pp512 | 294.80 | 302.84 | +2.73 % |
| d4096 f16 | tg128 | 20.37 | 21.17 | +3.93 % |

Prediction 4 was stated only for tg128, where it held (-0.39 %, -0.05 %). Prefill was
never predicted, and it blew the -2 % protected-cell guard.

(Both this sentence and the heading above originally said "prediction 3". That was a
mis-numbering: `-0.39 %` and `-0.05 %` are the depth-0 `tg128` deltas and the "-2 %
protected-cell guard" is prediction 4's wording, so prediction 4 is what was being
evaluated here. Prediction 3, the q8_0-vs-f16 gap claim, is not evaluable from these three
cells at all - it needs a q8_0-against-f16 comparison at depth, not a wg=2-against-wg=16
delta - and is settled further down under "Prediction 3's falsifier did not trigger".)

Two lessons, both against the earlier analysis:

1. **The indicative contended numbers were badly inflated.** +19.5 % at d4096 became
   +3.93 % under gate conditions, roughly a 5x overstatement. A starved kernel suffers
   disproportionately from contention, so the wg=2 baseline was penalised more than the
   candidate. The +71.1 % headline does not survive.
2. **The first regression hypothesis was wrong.** It blamed the `stream_k` branch. Geometry
   showed `stream_k=0` for every workload measured; prefill uses the split-k branch too.

### Root cause: the occupancy limit was a floor, not a cap

`parallel_blocks` was initialized to `max_blocks_per_sm` and the efficiency search only
ever grew it. Harmless while the value was 2; wrong once it reflected real occupancy.

Split-K manufactures parallelism when the tile count cannot fill the device, at the cost of
multiplying `dst_tmp` scratch and widening the combine reduction over every output element.
Decode needs it (`ntiles_total` = 32 against a 512 `blocks_per_wave`). Prefill does not
(`ntiles_total` = 4096, already saturated) yet inherited the same floor:

| path | wg=2 | wg=16 (before fix) |
|---|---|---|
| prefill | 2 splits, 8192 blocks | 8 splits, **32768 blocks** |

### Fix (`53f390a91`): grow from one split, bounded by occupancy

| path | wg=2 | wg=16 |
|---|---|---|
| decode | 2 splits, 64 blocks, 8192 items | 16 splits, 512 blocks, **65536 items** |
| prefill | 1 split, 4096 blocks | 1 split, 4096 blocks |

Prefill is now **invariant** to the occupancy constant - identical grid at both values - and
splits less than the original baseline. The regression is removed structurally, not tuned
away. Correctness: `0 GATE-FAIL, 0 XPASS, 0 xfail, 0 SKIP`.

### Indicative re-measurement of the fix

Alternating arms, arm order reversed between passes, three passes. The render node was
shared with a user `llama-server` running Qwen3-Coder-30B at `-ngl 99`, so these are
**not** promotion evidence:

| metric | KV | depth | wg=2 | wg=16 | delta |
|---|---|---:|---:|---:|---:|
| pp512 | f16 | 0 | 724.65 | 696.27 | -3.92 % |
| pp512 | q8_0 | 0 | 681.92 | 719.61 | +5.53 % |
| tg32 | f16 | 16384 | 6.43 | 15.04 | +133.90 % |
| tg32 | q8_0 | 16384 | 8.74 | 14.16 | +62.01 % |

The prefill deltas straddle zero and the raw passes show a 16 % spread (f16 wg=16:
717.74, 619.40, 696.27), consistent with contention rather than a real effect. **The
geometry invariance, not this measurement, is the evidence that prefill is fixed.**

Decode direction is robust but magnitude is not: wg=2 f16 fell to 6.43 here against 13.93
when measured earlier under lighter load, which again shows contention punishing the
starved arm hardest and inflating the delta.

### Still outstanding

A paired campaign on a genuinely idle device. `scripts/perf` harness plus
`scratchpad/e1-campaign.sh` (stops and restores the CPU llama.cpp unit via a trap, verified
working) will run it. Required before promotion, before the deep-cell numbers can be
trusted, and before the upstream issue and PR.

## Phase 1 PROMOTION EVIDENCE - paired campaign under sole tenancy

Run 2026-07-25 against `53f390a91` on an idle A770 (`renderD128`; `renderD129` is the AMD
Raphael iGPU that drives the desktop). Environment-only A/B on a single binary, so there is
no build-identity confound. Six launches per arm, sample zero discarded, arms alternated,
five retained pairs per cell.

Mistral-7B-Instruct v0.1 Q4_K_M, `GGML_SYCL_MAX_WG_PER_CU` 2 versus 16:

| depth | KV | pp512 delta | tg128 delta | tg128 wg=2 | tg128 wg=16 |
|---:|---|---:|---:|---:|---:|
| 0 | f16/f16 | +0.47 % | +1.62 % | 23.78 | 24.16 |
| 0 | q8_0/q8_0 | +0.57 % | +0.08 % | 24.51 | 24.53 |
| 4096 | f16/f16 | +0.08 % | **+45.76 %** | 14.82 | 21.59 |
| 4096 | q8_0/q8_0 | +0.05 % | **+18.42 %** | 17.87 | 21.16 |
| 8192 | f16/f16 | -0.03 % | **+84.84 %** | 10.76 | 19.89 |
| 8192 | q8_0/q8_0 | -0.02 % | **+38.53 %** | 13.94 | 19.31 |
| 16384 | f16/f16 | +0.03 % | **+145.96 %** | 6.95 | 17.10 |
| 16384 | q8_0/q8_0 | +0.03 % | **+68.50 %** | 9.84 | 16.58 |

Meta-Llama-3.1-8B-Instruct-heretic Q4_K_M, independent confirmation:

| depth | KV | pp512 delta | tg128 delta | tg128 wg=2 | tg128 wg=16 |
|---:|---|---:|---:|---:|---:|
| 0 | f16/f16 | -0.44 % | +1.66 % | 22.98 | 23.36 |
| 0 | q8_0/q8_0 | -0.75 % | -0.11 % | 23.67 | 23.65 |
| 4096 | f16/f16 | +0.03 % | **+44.13 %** | 14.44 | 20.82 |

Findings:

1. **Prefill is neutral.** Every `pp512` delta lies within +/-0.6 %, against -15.08 % before
   the split-K fix. The geometry invariance argument is now backed by measurement.
2. **The shallow guard passes.** Depth 0 is neutral to slightly positive on both metrics,
   which is what the `ntiles_KQ` clamp predicts.
3. **Decode gains grow with depth**, consistent with an occupancy limit whose cost rises as
   each work-group's serial KV walk lengthens.
4. Baseline reproduces the post-P5 held campaign: q8_0 d8192 `tg128` 13.94 here against
   14.12 there.

### A framing correction

The long-standing "q8_0 KV decode is 30.82 % behind f16 VEC" figure comes from a
**forced-VEC** comparison. Under default routing f16 goes to TILE and q8_0 to VEC, and at
the wg=2 baseline q8_0 VEC (13.94 at d8192) actually *beats* f16 TILE (10.76). The
occupancy fix lifts f16 TILE by 84.84 % and closes the gap to near parity, 19.89 against
19.31. The deep-context problem was therefore worse for the f16 default route than for
q8_0, the opposite of the framing carried through the earlier research.

### Prediction 3's falsifier did not trigger - the split-KV q8_0 rewrite loses its justification

Prediction 3 in [Phase 1 predictions](#phase-1-predictions-recorded-now-measured-later) is
the only one of the four that gates a follow-up work item, so it needs an explicit verdict.
It was stated as *"The q8_0-vs-f16 gap narrows"*, with the falsifier attached: *"If the gap
does not narrow, the dequant-ALU-wall hypothesis is correct and a genuine split-KV q8_0
kernel rewrite becomes justified rather than premature."*

Reading it as a bare confirm/falsify is what makes it confusing, so state the two halves
separately:

- **The falsifier did not trigger, and that is the load-bearing half.** The rewrite was
  contingent on the gap *failing* to narrow. Under default routing the deep-context gap at
  d8192 now sits at near parity (f16 TILE 19.89 against q8_0 VEC 19.31), so the condition
  that would have justified the rewrite is not met. The dequant-ALU-wall hypothesis does
  not hold, the rewrite is **not justified** by this evidence, and none is scheduled or in
  progress.
- **The prediction as literally worded is not confirmed, because its premise was wrong.**
  It assumed q8_0 lagged f16 and would catch up. The framing correction directly above
  shows the opposite: the "30.82 % behind" figure came from a forced-VEC comparison, and
  under default routing q8_0 VEC (13.94 at d8192) already *led* f16 TILE (10.76) at the
  wg=2 baseline. What actually happened is that the fix lifted f16 by 84.84 % and closed
  the gap **from the other side**. Calling that a confirmation of "q8_0 catches up" would
  record the wrong mechanism.

Both halves point the same way for the work item; only the second one keeps the reason
straight for a future reader.

This closure is conditional on the same split-K budget constraint recorded under Phase 2b
below: split-K is not a free lever, there is a measured optimum near `wg=16` on this
device, and any future change that consumes split capacity - including a hypothetical
split-KV rewrite revisited under different evidence - has to pay for it out of that same
budget rather than assuming headroom exists.

### Promotion gates

| Gate | Result |
|---|---|
| >= +3 % paired median tg128 | met at every depth >= 4096, up to +145.96 % |
| Positive paired lower bound | yes on all improving cells |
| No protected cell below -2 % | worst across both models is -0.75 % |
| Correctness | `0 GATE-FAIL, 0 XPASS, 0 xfail, 0 SKIP` |
| New i915/xe fault, hang, reset | none |

**Recommendation: promote.** `max_wg_per_cu = 16` with split-K growing from one becomes the
default; `GGML_SYCL_MAX_WG_PER_CU` stays as a documented diagnostic override, following the
P5 pattern for `GGML_SYCL_MMV_Y` and `GGML_SYCL_MMVQ_NUM_SUBGROUPS`.

### Coverage gap - closed 2026-08-13

The five outstanding llama31-8b-heretic cells (q8_0 d4096, and both KV types at d8192 and
d16384) were filled at merged-master source `80d52e708`, same env A/B
(`GGML_SYCL_MAX_WG_PER_CU` 2 vs 16), six launches/arm, sample-zero discard, five retained
pairs, zero new i915/xe faults:

| depth | KV | pp512 delta | tg128 delta |
|---:|---|---:|---:|
| 4096 | q8_0/q8_0 | +0.13 % | **+21.13 %** |
| 8192 | f16/f16 | -0.02 % | **+77.43 %** |
| 8192 | q8_0/q8_0 | +0.20 % | **+40.11 %** |
| 16384 | f16/f16 | +0.13 % | **+135.78 %** |
| 16384 | q8_0/q8_0 | +0.01 % | **+72.92 %** |

All five clear the promotion gates decisively (prefill neutral, positive paired lower
bound, correctness `0 GATE-FAIL`); none contradicts the promoted fix. The prior "tracks its
mistral counterpart within about 2 percentage points" estimate, based only on the three
completed cells, does not hold precisely at the deepest new cells - d16384 f16 is +135.78 %
here against mistral's +145.96 % (about 10 points apart), d8192 f16 is +77.43 % against
+84.84 % (about 7 points apart) - while all three new q8_0 cells stay within ~4.5 points
(d4096 +21.13 against +18.42, d8192 +40.11 against +38.53, d16384 +72.92 against +68.50).
Direction and order of magnitude match at every cell; only the tight numeric estimate was
optimistic. The llama31 matrix is now complete across d0/d4096/d8192/d16384 x f16/q8_0.

#### Raw artifacts

The complete run is committed under
[`fa-occupancy-campaign-2026-08-13-llama31-coverage/`](fa-occupancy-campaign-2026-08-13-llama31-coverage/):
one directory per cell (60 sample JSONs, 12 per cell = 6 launches x 2 arms, with rep 0
discarded to leave the five retained pairs), per-cell `product.json` / `product.md` /
`provenance.json` and `dmesg.before.txt` / `dmesg.after.txt`, plus campaign-level
`dmesg.fault.delta.txt` (empty, hence `new=0`), `correctness.txt` and `campaign-runner.txt`.
Every cell's `provenance.json` records `repository_commit`
`80d52e708cacbb0e89c269f6237ca4bb378c4cf6`, and every `product.json` carries
`all_cells_valid: true` with `dmesg_before_hits = dmesg_after_hits = 0`.

This is a distinct campaign from the 2026-07-25 Phase 1 evidence in
`fa-occupancy-campaign-2026-07-25/`, which was run against `53f390a91` and whose `llama31/`
directory holds only the three cells completed at that time. The two directories are not
interchangeable, and the `sole-tenancy-violation.txt` under
`fa-occupancy-campaign-2026-07-25/llama31/` (PID 3561640) belongs to that July run, not to
these cells.

Sole tenancy here is enforced fail-closed rather than recorded positively:
`check_sole_tenancy()` in `scripts/bench-a770-fork-unique.py` runs a non-mutating
`fuser /dev/dri/renderD128` probe before every individual leg, and on any holder aborts the
campaign with exit 70 after writing `sole-tenancy-violation.txt` into the output directory.
The 2026-08-13 directory contains no such file and the runner completed all five cells, so
no leg started with a holder on the device. There is no per-leg positive holder record; the
evidence is the absence of the violation artifact plus a clean exit.

One further limit on the A/B: `GGML_SYCL_MAX_WG_PER_CU` is read through a bare `getenv` in
`ggml/src/ggml-sycl/ggml-sycl.cpp` and is logged only when the value is invalid, so it never
appears in the backend's "Running with Environment Variables" block. The harness env-log
assertion is therefore vacuous for this knob, and each `product.md` says so verbatim: `not
validated from backend logs; key not emitted`. Engagement rests on the per-sample recorded
process environment (`active_env`) plus effect size - a no-op override cannot move tg128 by
+21 % to +136 % while leaving pp512 inside +/-0.2 %.

### Note on tenancy

Three separate holder classes interrupted the 2026-07-25 Phase 1 campaign: editor and IDE
GPU processes, a user `llama-server` at `-ngl 99`, and KDE `kioworker` thumbnail helpers
spawned by open Dolphin windows. The thumbnailers were the most disruptive because they reappear whenever a
directory is browsed. A clean campaign needs the desktop file manager closed, not just
inference servers stopped.

## Phase 2b result - GQA packing REFUTED on the corrected baseline

The packed q8 GQA kernel (`2d53f3b25`, reverted `0bb42498e` at +7.59 % d16384 /
-1.84 % d4096) was predicted to be a victim of the occupancy cap: packing divides
`ntiles_total` by `gqa_ratio`, and on the old baseline `parallel_blocks` had no room to
absorb that, so a 4x K-traffic saving was cancelled by a 4x occupancy loss.

The geometry half of that prediction is correct. Restoring the kernel and measuring, q8_0
KV with the quants-first layout the packed path requires (`GGML_SYCL_FA_Q8_GQA_DIRECT=1`
additionally requires `GGML_SYCL_Q8_KV_QUANTS_FIRST=1`):

| config | `ntiles_total` | `parallel_blocks` | `blocks_total` | `work_items_total` |
|---|---:|---:|---:|---:|
| packing off, wg=16 | 32 | 16 | 512 | 65536 |
| packing on, wg=16 | 8 | 16 (capped) | 128 | 16384 |
| packing on, wg=64 | 8 | 64 | 512 | 65536 |

At wg=64 the packed kernel regains the full grid while still reading a quarter of the K
traffic. That configuration was not reachable before the occupancy fix.

**The performance prediction is refuted.** Absolute numbers swung roughly 2x between passes
from a competing user workload, so each pass is normalized against its own `base2` arm:

depth 8192, ratio to `base2`:

| config | pass 1 | pass 2 | pass 3 |
|---|---:|---:|---:|
| packing off, wg=16 | 1.298 | 1.321 | **1.345** |
| packing on, wg=16 | 1.121 | 1.173 | 1.156 |
| packing on, wg=64 | 0.876 | 0.974 | 0.946 |

depth 16384, ratio to `base2`:

| config | pass 1 | pass 2 | pass 3 |
|---|---:|---:|---:|
| packing off, wg=16 | 1.689 | 1.617 | **1.630** |
| packing on, wg=16 | 1.382 | 1.327 | 1.315 |
| packing on, wg=64 | 1.073 | 1.062 | 1.068 |

Ordering is consistent across all six passes: **packing loses at every occupancy setting,
and restoring its grid makes it worse rather than better.**

The likely reason is in Phase 0's own data. Combine was 2.1 % of flash-attention time at 2
splits. It scales with the split count, so at 64 splits it is a 64-wide reduction over
every output element backed by 32x the `dst_tmp` scratch. Split-K overhead overtakes the
K-traffic saving well before the grid is refilled. The same ceiling explains why wg=32
regressed against wg=16 in the Phase 1 sweep.

Consequences:

- The packed GQA kernel stays reverted, now for a measured reason on a corrected baseline
  rather than a confounded one. The working-tree restore was discarded; the candidate
  remains in history at `2d53f3b25`.
- Split-K is not a free lever. There is an optimum near wg=16 on this device and pushing
  past it costs more in combine than it returns in parallelism. Any future change that
  consumes split capacity - GQA packing, or a split-KV rewrite - has to pay for it out of
  that same budget.
- Correctness with the packed path enabled was clean (`0 GATE-FAIL`), so this is a
  performance verdict, not a correctness one.

## Baseline to beat

Post-P5 phase-7 held numbers, Mistral-7B-Instruct v0.1 Q4_K_M, q8_0/q8_0 KV:

| Depth | tg128 tok/s | Effective KV GB/s |
|---:|---:|---:|
| 0 | 24.60 | - |
| 4096 | 17.93 | 20.46 |
| 8192 | 14.12 | 32.22 |
| 16384 | 9.93 | 45.32 |
