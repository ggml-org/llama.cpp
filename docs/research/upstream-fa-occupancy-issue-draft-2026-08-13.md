# Draft upstream issue: SYCL flash-attention occupancy governor derives max_wg_per_cu from an unrelated limit

Status: **draft only, not filed.** Written for a human to review, edit, and decide
whether/where to file. Per repo policy this fork never pushes or opens issues/PRs against
`ggml-org/llama.cpp`.

## Source and verification

The bug and its consumer were re-verified live against upstream `ggml-org/llama.cpp`
`master` at commit `1d2869c6e54d5003f3927a79efbca0fefa034a6d` (fetched 2026-08-13 via
`raw.githubusercontent.com`), not assumed from a stale local mirror. Line numbers below are
for that commit and will drift on subsequent upstream commits - re-check before filing.

Internal evidence trail: `docs/research/fa-occupancy-prereg-2026-07-25.md` (pre-registered
prediction, Phase 0/1/2b measurement, promotion gate). This fork's fix landed via PR #35
(squashed; original branch commit `53f390a91`, "fix(sycl): grow flash-attention split-K
from one, not from the occupancy cap").

---

## Suggested issue title

`SYCL: flash-attention occupancy governor derives max_wg_per_cu by dividing max work-group size by compute-unit count, yielding 2 on Xe-HPG (Arc A-series); raising it to 16 measured up to +146% decode throughput in a downstream fork's paired A/B`

Scope note on the numbers: the governor's value of `2` on this device is established by
arithmetic from `clinfo`. `16` is the best of `{2, 4, 8, 16, 32}` as measured on an A770,
**not** an independently established figure for the device's true concurrent-block capacity
- no such measurement was made, which is also why the suggested fix below is deliberately
not "hardcode 16". Everything quantified in this report is the `2` versus `16` comparison.

## Suggested issue body

### Summary

`ggml_sycl_init()` derives the maximum number of concurrently resident flash-attention
work-groups per compute unit by dividing the device's max work-group size by its compute
unit count:

```cpp
// ggml/src/ggml-sycl/ggml-sycl.cpp:159-160 (upstream master, 1d2869c6e)
info.max_work_group_sizes[i] = prop.get_max_work_group_size();
info.devices[i].max_wg_per_cu = info.max_work_group_sizes[i] / prop.get_max_compute_units();
```

On Intel Arc A-series (Xe-HPG, e.g. A770: `Max compute units = 512`, `Max work group size =
1024` per `clinfo`), this evaluates to `1024 / 512 = 2`. That is not a measurement of actual
concurrent-block capacity - it conflates "maximum work-items per work-group" with "maximum
work-groups resident per compute unit," two unrelated hardware limits. Whether `2` happens
to match the device's real capacity is not something the expression could establish either
way; the measurements below show it is at least well below the value that performs best.

`launch_fattn()` in `ggml/src/ggml-sycl/fattn-common.hpp` (declared at line 895, shared by
every flash-attention route) consumes this value as `max_blocks_per_sm` (line numbers below
are the same upstream commit):

```cpp
// fattn-common.hpp:1048-1049
int max_blocks_per_sm = ggml_sycl_info().devices[id].max_wg_per_cu;
int parallel_blocks = max_blocks_per_sm;
```

`max_blocks_per_sm` reaches the launch geometry two ways, and it is worth keeping them
apart:

- **Resident-per-wave capacity.** `blocks_per_wave = nsm * max_blocks_per_sm`
  (`fattn-common.hpp:1074`) is the search's model of how many work-groups the device can
  hold at once. On A770 that is `32 * 2 = 64`.
- **The launched grid**, which is *not* that expression. The non-`stream_k` branch sets
  `blocks_num.x = ntiles_x`, `blocks_num.y = parallel_blocks`,
  `blocks_num.z = ntiles_z_gqa * K->ne[2] * Q->ne[3]` (`fattn-common.hpp:1094-1096`), so the
  work-group count is `ntiles_total * parallel_blocks`.

`parallel_blocks` is **initialized to** the undercounted value, capped only downward by the
tensor (`parallel_blocks = std::min(parallel_blocks, ntiles_KQ)`, line 1068), and the
efficiency search that follows (lines 1077-1092) only ever grows it. The search floor is
therefore wrong, not just its ceiling.

The two quantities then coincide for decode, by construction rather than by accident:
`efficiency_percent` reaches 100 as soon as `ntiles_total * parallel_blocks` is a multiple
of `blocks_per_wave`, and the `efficiency_percent_best >= 95 && nwaves > nwaves_best` guard
breaks the loop before it grows past a single wave. So for any decode shape whose
`ntiles_total` divides `blocks_per_wave`, the launched grid settles at exactly one wave:
`ntiles_total * parallel_blocks == blocks_per_wave == nsm * max_blocks_per_sm == 64`
work-groups of 128 work-items, i.e. 8,192 of the device's ~65,536 resident work-item
capacity, roughly **12.5% occupancy**. Instrumented on the fork, three independent decode
configurations land on that same 64: VEC q8_0 (`parallel_blocks` 2 x `ntiles_total` 32),
VEC f16 forced (2 x 32), and TILE f16 (8 x 8, reaching it through `ncols2 = 4` GQA packing
plus an 8-way KV split). Those launch records were captured without sole tenancy on the
render node, which does not affect them: launch geometry is a pure function of device
properties and tensor shapes, so only the throughput figures measured alongside were
discarded. The ceiling does not rise with context depth, so the wasted fraction of the
device grows as the per-work-group serial KV walk lengthens.

### Impact (measured on a downstream fork, same code path)

A fork of this project (`Raudbjorn/ggml-llama.cpp`) carrying this exact upstream code
independently discovered the geometry, confirmed the launch-parameter values by
instrumentation before making any change (pre-registered prediction, matched to six decimal
fields exactly), and measured the fix under sole-tenancy paired A/B (six launches/arm,
sample-zero discarded, five retained pairs, `GGML_SYCL_MAX_WG_PER_CU` env override):

Mistral-7B-Instruct Q4_K_M, Arc A770, `tg128` paired median delta from raising the governor
from 2 to 16 (16 was chosen as the best of `{2,4,8,16,32}` measured on this device; see
"Suggested fix" below for why the fix is not simply "hardcode 16"):

| depth | KV | tg128 delta |
|---:|---|---:|
| 0 | f16 | +1.62% |
| 0 | q8_0 | +0.08% |
| 4096 | f16 | +45.76% |
| 4096 | q8_0 | +18.42% |
| 8192 | f16 | +84.84% |
| 8192 | q8_0 | +38.53% |
| 16384 | f16 | **+145.96%** |
| 16384 | q8_0 | **+68.50%** |

Independently confirmed on Meta-Llama-3.1-8B-Instruct (heretic) Q4_K_M with matching-sign,
similar-magnitude deltas at every measured depth.

Prefill (`pp512`) is neutral within +/-0.6% at every depth on both models - but only with
both fixes below applied. The governor is **not** decode-only: raising it on its own
regressed prefill by -15.08% (`pp512`, d0 f16) in this fork's first attempt, which is what
motivated the floor fix. See "Prefill note" below. Correctness harness: `0 GATE-FAIL` on
both the unpatched and patched builds.

**Caveat on these numbers:** the fork carries other changes unrelated to this bug (a
custom KV cache row layout, additional flash-attention routing knobs). The *launch
geometry* (`parallel_blocks`, `blocks_total`, `work_items_total`, and their arithmetic
dependence on `max_wg_per_cu`) is a pure function of device properties and tensor shapes,
identical on vanilla upstream. The *throughput* numbers above should be treated as strong
evidence of the mechanism and its order of magnitude, not as a plain-upstream A/B - a
vanilla-upstream measurement is straightforward to reproduce (see below) but was not
separately run for this draft.

### Minimal reproduction (architecture-only, no build required)

1. On any Xe-HPG device (Arc A-series discrete or the built-in Arc iGPU family), run
   `clinfo` and note `Max compute units` and `Max work group size`.
2. Compute `max_work_group_size / max_compute_units`. On A770 this is `1024 / 512 = 2`.
3. This value becomes `max_blocks_per_sm` in `fattn-common.hpp`'s non-`stream_k` branch,
   where it is both the initial `parallel_blocks` (the search floor) and, via
   `blocks_per_wave = nsm * max_blocks_per_sm`, the wave size the search targets. For a GQA
   model with `ntiles_total` in the low tens (typical for single-token decode at GQA ratios
   of 4-8 and D=128), the search stops at one wave, so the launched grid
   `ntiles_total * parallel_blocks` equals `nsm * max_blocks_per_sm` - computable from
   `clinfo` output alone, no GPU run needed to establish that the ceiling exists.
4. To observe the throughput effect directly. Upstream exposes no environment override for
   this constant (`max_wg_per_cu` is only ever assigned at `ggml-sycl.cpp:160`), so each arm
   is a separate build: baseline is upstream as-is, and the candidate replaces line 160 with
   a fixed value (`info.devices[i].max_wg_per_cu = 16;`) plus the floor fix from item 2 of
   "Suggested fix" below. Build JIT (leave `-DGGML_SYCL_DEVICE_ARCH` unset) and run both KV
   types on a GQA model, e.g. Mistral-7B-Instruct-v0.1 Q4_K_M (4:1 GQA, D=128) or
   Meta-Llama-3.1-8B-Instruct Q4_K_M:

   ```bash
   # run against each of the two builds; f16 KV
   llama-bench -m <gqa-model>.gguf -ngl 99 -fa on -ctk f16 -ctv f16 \
       -p 512 -n 128 -d 0,4096,8192,16384

   # run against each of the two builds; q8_0 KV
   llama-bench -m <gqa-model>.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
       -p 512 -n 128 -d 0,4096,8192,16384
   ```

   Compare `tg128` for the decode effect and `pp512` for the prefill invariance claim. The
   fork's own runs used an added `GGML_SYCL_MAX_WG_PER_CU` env override to keep both arms on
   a single binary; that knob is fork-only and does not exist upstream.
5. To see the prefill regression that makes the floor bug an *independent* problem, a third
   arm is required: line 160 changed but **without** the floor fix. The two builds in step 4
   are "before both fixes" and "after both fixes", so they show prefill as neutral and never
   expose the -15.08% `pp512` described under "Prefill note". Only the formula-only build
   reproduces it.

### Suggested fix (illustrative, not a ready-to-merge patch)

Two independent problems, both need fixing:

1. **Wrong formula.** Replace the divide with either a hardware constant appropriate to
   the architecture family, or (more portably) a small runtime probe of actual concurrent
   occupancy, rather than deriving it from an unrelated work-group-size limit.
2. **Floor-not-cap bug**, independent of (1): initialize `parallel_blocks = 1` in the
   non-`stream_k` branch instead of `max_blocks_per_sm`, so the existing efficiency search
   loop (`fattn-common.hpp:1077-1092`) can find low values too, and only use the corrected
   `max_wg_per_cu` as the loop's upper bound. This fork's fix (branch commit `53f390a91`,
   squash-merged as part of a larger PR) did exactly this and is what "prefill invariant"
   above refers to - fixing (1) alone without (2) still leaves prefill's split-K
   multiplied unnecessarily at high `max_wg_per_cu` values, since prefill's `ntiles_total`
   already saturates the device and does not benefit from more splits.

### Prefill note

Prefill goes through the **same non-`stream_k` branch as decode**. At this commit every
`launch_fattn` call site passes `stream_k = false` - `fattn-vec.hpp:601` and `:611`, and the
six sites in `fattn-tile.hpp` at `:1089`, `:1100`, `:1111`, `:1124`, `:1135` and `:1145` -
so the `stream_k` branch at `fattn-common.hpp:1051-1063` is unreached, and `nblocks_stream_k`
is not what governs prefill geometry. (Worth stating explicitly because the obvious first
hypothesis, that the prefill regression comes from `nblocks_stream_k = max_blocks_per_sm *
nsm`, is wrong; this fork tested it against instrumented launch records, which reported
`stream_k = 0` for every workload measured, and had to discard it.)

The actual prefill mechanism is the same init-floor bug. Prefill's `ntiles_total` already
saturates the device, so it needs no split-K at all, but `parallel_blocks` is still
initialized to `max_blocks_per_sm` and clamped only by `std::min(parallel_blocks, ntiles_KQ)`
at line 1068. Raising the governor from 2 to 16 therefore raised prefill's floor from 2 to
`min(16, ntiles_KQ)`, and the fork measured that as 8 splits for the cell it instrumented
(the search only grows `parallel_blocks`, so landing on 8 implies `ntiles_KQ = 8` for that
shape; it is derived from the measurement, not separately recorded):

| path | `max_wg_per_cu` = 2 | `max_wg_per_cu` = 16, formula fix only |
|---|---|---|
| prefill (`ntiles_total` = 4096, `ntiles_KQ` = 8) | 2 splits, 8,192 blocks | 8 splits, **32,768 blocks** |

The multiplier is `min(16, ntiles_KQ) / min(2, ntiles_KQ)`, so it is 4x only for that
shape's `ntiles_KQ = 8`; a shape with `ntiles_KQ >= 16` would see 8x. That inflated grid is
what produced the -15.08% `pp512` noted above. Applying the floor fix alone - independent of
the formula - makes prefill invariant to the governor (1 split, 4,096 blocks at both 2 and
16), which is why the promoted fix changed both.

### Environment (fork's measurement environment, for completeness)

- Device: Intel Arc A770, DG2/Xe-HPG (`acm-g10`), driver Level Zero 1.15.38646, i915 kernel driver.
- Toolchain: Intel oneAPI DPC++/C++ 2026.0.0, Release build, JIT (no `GGML_SYCL_DEVICE_ARCH`).
- `GGML_SYCL_TARGET=INTEL`.

---

## What to check before filing

- [ ] Re-fetch `ggml/src/ggml-sycl/ggml-sycl.cpp`, `fattn-common.hpp`, `fattn-vec.hpp` and
      `fattn-tile.hpp` at the upstream commit current at filing time; line numbers above
      will have drifted, and the `stream_k = false` claim needs re-checking at every
      `launch_fattn` call site rather than assumed.
- [ ] If the title is used standalone, keep the "in a downstream fork's paired A/B"
      attribution on the +146% figure; the caveat that qualifies it lives in a body section
      a tracker title does not carry.
- [ ] Decide whether to include the fork's throughput table (attributed as
      fork-measured, not upstream-measured) or ask upstream maintainers to reproduce
      independently.
- [ ] Consider filing as two smaller issues (formula bug; floor-not-cap bug) if that suits
      upstream's triage process better than one combined report.
- [ ] Optionally attach the fork's `docs/research/fa-occupancy-prereg-2026-07-25.md` as
      supporting evidence (public repo, linkable).
