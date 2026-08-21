# ROCmFPx port — status, measurements, and TODOs

Working notes for the `dev` branch. Written 2026-08-21.

## Branch state

`dev` = updated `master` + four merges, in this order:

| Merge | What |
|---|---|
| `vulkan/iq3s-mmv-register-spill` | IQ3_S mat-vec for NUM_COLS > 4 (5x at n=8) + k=16*256 sweep cases |
| `feature/rocmfpx-tensors` | ROCmFPx quant types, CPU + Vulkan |
| `feature/rocmfpx-tensors` (2nd) | Vulkan integer-dot (MMVQ/MMQ) paths for ROCmFPx |
| `pr-27342` | DFlash2 speculative decoding (Jian Chen) |

On top of those, one direct commit: **Vulkan batched-decode fixes for the ROCmFPx
mat-vec kernels** (2026-08-21). See "Batched decode" below.

Nothing is pushed. `git reset --hard <merge>` undoes any of them.

## ROCmFPx types

Ported from https://github.com/ciru-ai/ROCmFPX. That fork shares **no git history**
with ggml-org/llama.cpp (different root commit) and sits on an older base, so it can
only ever be hand-ported. A whole-tree merge would delete ~819 newer upstream files
including all of `tools/ui`. Do not attempt it.

Six GGUF types, kept at the fork's IDs so existing GGUFs load:

| Type | bpw | Vulkan float | Vulkan int-dot |
|---|---|---|---|
| `Q4_0_ROCMFP4` (100) | 4.50, dual UE4M3 scale | yes | yes |
| `Q4_0_ROCMFP4_FAST` (101) | 4.25, single scale | yes | yes |
| `Q6_0_ROCMFPX` (102) | 6.50 | yes | no |
| `Q8_0_ROCMFPX` (103) | 8.25 | yes | mat-vec only |
| `Q3_0_ROCMFPX` (104) | 3.50 | yes | no |
| `Q2_0_ROCMFPX` (107) | 2.50 | CPU only | no |

IDs 105/106 are deliberately left free — TurboQuant claims them upstream in that fork.

### Two upstream-fork bugs fixed here (we diverge on purpose)

1. **Vulkan fp6 layout.** The fork declares `block_rocmfpx_fp6` as 32 unpacked int8 +
   2 scale bytes and *writes decoded int8*, but the CPU encoder writes 24 bytes of
   6-bit packed codes. Reimplemented against the real layout; verified bit-exact vs
   `rocmfpx_dequantize_row_fp6` over 640k values.
2. **CPU fp6 vec_dot.** `ggml_rocmfpx_decode_fp6_cpu` decoded the `sign|0` code as
   `-0` instead of `-32`, disagreeing with `rocmfpx_decode_fp6_code`. Cost ~2% error
   on every fp6 mat-vec.

### Validation against the fork

Built the fork at `68f23f34c` and used it as an oracle:

- Quantizer is **bit-identical** (same SHA256) for all 8 types/recipes tested:
  `Q4_0_ROCMFP4`, `_FAST`, `_STRIX`, `Q8_0_ROCMFPX`, `Q6_0_ROCMFPX`, `_AGENT`,
  `Q3_0_ROCMFPX`, `Q2_0_ROCMFPX`.
- Same model produces character-identical output on both builds.
- `test-backend-ops` full sweep green, 857 ROCmFPx cases, both with the default
  heuristic and `GGML_VK_FORCE_MMVQ=1`.

## Measurements (Radeon 8060S / RADV STRIX_HALO, idle GPU)

### Quant comparison, Qwen3.8-27B

| | size | pp512 | tg128 | wikitext-2 PPL (40 chunks) |
|---|---|---|---|---|
| ROCmFPX-MQ-Q4 (`Q4_0_ROCMFP4`) | 14.63 GiB | 296.8 | **12.56** | 5.9842 +/- 0.071 |
| unsloth UD-Q4_K_XL | 16.69 GiB | 274.5 | 11.71 | **5.8965 +/- 0.070** |

FP4 is 12% smaller, ~7% faster at batch 1, ~1.5% worse PPL. The PPL error bars
overlap, so treat the quality gap as "small but probably real", not precise.

### Batched decode — was the FP4 weak spot, fixed 2026-08-21

FP4 used to *lose badly* under speculative decoding (16.8 t/s vs unsloth's 22.3 at
n-max 3) even though it accepts about as often (52.7% vs 55.2%, mean accepted len
2.58 vs 2.65). Verification runs the main model at batch 3-8, and both ROCmFPx
mat-vec paths were mis-tuned for exactly that range. Two shader fixes:

**1. `mul_mat_vecq.comp`: `K_PER_ITER` 8 -> 32 for `ROCMFP4`/`_FAST`.**
Each MMVQ call handled a quarter of a block and re-decoded *both* UE4M3 scales
(a byte load plus a shared-LUT lookup each) every time. One whole block per call
amortises the scales 4x and lets the 8 B dwords load as two `dwordx4` instead of
two scattered dwords. Needs the matching whole-block `mmvq_dot_product` in
`mul_mat_vecq_funcs.glsl` and a `K_PER_ITER == 32` branch in the `QUANT_R == 2`
B-cache load.

**2. `dequant_funcs.glsl`: branch-free, scale-hoisted `dequantize4` for fp6 and fp3.**
Both were gathering *each weight* through its own bit window with a branch, and
doing one shared-LUT scale lookup *per weight*. Every caller passes `iqs % 4 == 0`
(`mul_mat_vec.comp` steps `col` by 8, `copy_from_quant.comp` steps `j` by 4), so
four codes are exactly three whole bytes (fp6) or twelve bits inside two bytes
(fp3), and all four share one scale because the halves split at element 16. The
branch was also stopping the compiler hoisting the decode out of the `NUM_COLS`
loop, which is why the old cost scaled *linearly* with batch size.

**fp6 mattered for an "FP4" model because the GGUF is mixed.**
`Qwen3.8-27B-ROCMFPX-MQ-Q4.gguf` is 502 `Q4_0_ROCMFP4` tensors *plus*
`output.weight` (0.96 GiB, 248320x5120) as `Q6_0_ROCMFPX`. At batch 4 that single
tensor cost ~22 ms/step, which was the entire unexplained gap after fix 1 landed.
Check the type mix before blaming the FP4 kernels.

### Kernel level, m=4096 k=14336 (us/run, lower is better)

`test-backend-ops perf`, before -> after the 2026-08-21 fixes:

| type | n=1 | n=4 | n=8 |
|---|---|---|---|
| `q4_0_rocmfp4` | 105 -> 82 | 189 -> 118 | 312 -> **173** |
| `q4_0_rocmfp4_fast` | 92 -> 80 | 177 -> 118 | 295 -> **174** |
| `q6_0_rocmfpx` | 551 -> 226 | 1308 -> 307 | 2236 -> **402** |
| `q3_0_rocmfpx` | 321 -> 275 | 1045 -> 374 | 1942 -> **463** |
| `q8_0_rocmfpx` (untouched) | 271 | 279 | 309 |
| `q4_K` (control) | 75-86 | 157-160 | 262-268 |
| `q6_K` (control) | 211 | 275 | 410 |
| `mxfp4` (control) | 100 | 160-178 | 277-292 |

Dual-scale used to cost 13-18% vs the single-scale `_FAST` layout. After the fix
the two are within noise of each other (173 vs 174 at n=8), because the second
scale is now decoded once per block instead of once per 8 weights. fp6 and fp3 no
longer scale linearly with batch.

### End to end, Qwen3.8-27B ROCMFPX-MQ-Q4

`llama-batched-bench -npp 128 -ntg 32`, TG t/s, before -> after:

| B | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| FP4 | 12.6 -> **13.3** | 20.6 -> **24.1** | 30.2 -> **41.0** | 36.3 -> **60.0** |
| unsloth UD-Q4_K_XL | 11.6 | 21.0 | 34.5 | 44.8 |

DFlash2 spec decoding, prose prompt, `--temp 0`, same Q4_K_M draft:

| | n-max 3 | n-max 7 |
|---|---|---|
| FP4 before | 16.8 | 15.2 |
| FP4 after | **24.6** | **24.5** |
| unsloth | 22.3 | 18.5 |

FP4 now wins at every batch size. Note that n-max 7 is no longer penalised for
FP4 — the verification batch got cheap enough that the acceptance loss at 7 is
paid back. On prose that does *not* hold for the K-quant models, which measure
22.3 t/s at n-max 3 against 18.5 at 7. See "spec-draft-n-max is content-dependent"
below; on structured output the picture may differ for them too.

`test-backend-ops` 17920/17920 green, and again with `GGML_VK_FORCE_MMVQ=1`.
Generation output stays coherent.

**Rejected experiment.** Replacing the shared-memory UE4M3 LUT with ~8 VALU ops
made things *worse* (n=1 105 -> 132 us). The float mat-vec path calls
`ue4m3_to_fp32` four times per 8 weights, so the LUT was never the problem —
calling it that often was. The fix is amortisation, not arithmetic.

### Integer-dot (MMVQ) contribution

MMVQ is **on by default** (`mmvq_mode = 0` -> heuristic). No flag needed;
`GGML_VK_DISABLE_MMVQ` / `GGML_VK_FORCE_MMVQ` only override. The heuristic returns
true for n > 1 unconditionally, and on AMD it also returns true at n = 1 for every
ROCmFPx type once k >= 2048 — so the FP4 tensors in a 27B run take the MMVQ path at
*every* batch size, including plain single-token generation. That is why the
`K_PER_ITER` fix above helped batch 1 as well (105 -> 82 us).

Measured before the 2026-08-21 kernel fixes:

| | gain |
|---|---|
| batch 1 | +0.6% |
| batch 8 (synthetic) | +4.6% |
| real dflash spec workload | +4.3% |
| pp512 | none |

Re-checked after the fixes, FP4 batched-bench at B=8: 46.2 t/s with MMVQ vs
45.7 without. Still a win, still small — the fixes moved the cost elsewhere.

MMQ for prefill is **neutral on this device** — a control with stock `q8_0` (which has
had MMQ upstream forever) behaves identically, because KHR_coopmat matmul already wins
here. Kept anyway; it should help devices without coopmat.

### `spec-draft-n-max` is content-dependent

Acceptance drives everything, and acceptance depends on what is being generated.
Deeper drafting only pays when acceptance is high enough to fill the extra slots.

**Trust the table below, not the old one.** An earlier version of this section
carried a three-row table (prose 20.4 / 14.6, HTML 21.7 / 19.1, digits 29.3 / 41.0)
under a "DFlash2 tuning" heading. Its provenance is not recoverable and it is
probably MTP, not DFlash2 — it contradicts the 2026-08-20 session note, which has
DFlash2 at n-max 7 reaching ~33 t/s and beating MTP-4's 26.5 on the same box. It
has been removed rather than corrected.

Measured 2026-08-21 on the post-fix build (b10558), DFlash2 draft
`incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M`, `--temp 0`, t/s:

| model | content | n-max 3 | n-max 7 |
|---|---|---|---|
| ROCMFPX-MQ-Q4 | prose | 24.6 | 24.5 |
| unsloth UD-Q4_K_XL | prose | **22.3** | 18.5 |
| ROCMFPX-MQ-Q4 | HTML | - | **40.7** |
| unsloth UD-Q4_K_XL | HTML | - | 32.9 |

The two HTML rows are *not* a controlled benchmark — they are single interactive
runs through the server UI on different prompts (1057 and 1030 tokens), recorded
because the effect is far larger than the noise. **The unsloth n-max 3 HTML cell is
still unmeasured**, so the 40.7-vs-32.9 gap is an upper bound on FP4's advantage
there: the K-quant preset may simply be mis-tuned at 7 for that content.

What this means for the presets:

- ROCmFPx is flat between 3 and 7 on prose, so 7 costs nothing and wins on anything
  structured. `spec-draft-n-max = 7` in the ROCmFPx preset is deliberate.
- The K-quant `reasoning-*` presets pay 17% at 7 on prose (22.3 -> 18.5). They also
  use 7. Whether to drop them to 3 depends on the unmeasured HTML cell above — if
  those presets are mostly used for code and structured output, 7 may still be
  right for them too.

### Where the 2026-08-21 fixes actually help

The kernel work helps at every batch size, but the size of the win tracks how many
tokens a verification step carries, so it compounds with acceptance:

| B | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| FP4 batched-bench TG gain | +6% | +17% | +36% | +65% |

Prose spec decoding, where mean accepted length is ~2.6, gained 46% (16.8 -> 24.6).
Structured output accepts far more per step, sits nearer the B=8 column, and is
where the fixes pay best. So: a real win everywhere, a large one on code and markup.

## TODOs

### Closing the FP4 batched gap — DONE 2026-08-21

Items 1-4 below were the original ranked list. The gap is closed; kept here with
outcomes so the reasoning is not re-run.

1. ~~Try `Q4_0_ROCMFP4_FAST` instead of dual-scale.~~ **Moot.** The 13-18%
   dual-scale tax was the per-8-weight scale decode, not the layout. After the
   `K_PER_ITER` fix the two are within noise (173 vs 174 us at n=8), so `_FAST` no
   longer buys speed — only the 0.25 bpw. Requantising is now purely a size/quality
   question.
2. ~~Optimise the dual-scale path.~~ **Done**, but not the way this item guessed.
   The two accumulators and two multiplies were never the cost; decoding both UE4M3
   scales four times per block was.
3. ~~Check the NUM_COLS > 4 path for FP4.~~ **Not the IQ3_S bug.** No register
   spilling anywhere in the FP4 mat-vec (`Spilled VGPRs: 0`, 48 VGPRs at every
   NUM_COLS). The perf sweep "returning no data at n=8" was a stale
   `test-backend-ops` binary, not a missing case.
4. ~~Profile which pipeline actually runs at n=4.~~ **MMVQ**, at every batch size
   including 1 — see the MMVQ section above.

Still open on the perf side:

- **Measure unsloth UD-Q4_K_XL on HTML at n-max 3.** It is the one cell that decides
  whether the three K-quant `reasoning-*` presets should drop to 3, and whether the
  40.7-vs-32.9 HTML gap is FP4 winning or the K-quant preset being mis-tuned. Use a
  fixed prompt and run both settings back to back; the two numbers on record came
  from separate interactive sessions.
- fp6/fp3 float mat-vec is fixed but still ~1.5x `q6_K`/`q3_K` at n=1 (226 vs 211,
  275 vs 109). fp3 in particular still pays a shared-LUT codebook lookup per weight.
  Low priority: no model here leans on fp3.
- The same `K_PER_ITER` amortisation probably helps `Q8_0_ROCMFPX` and the upstream
  `mxfp4` / legacy types, which still decode their scale once per 8 weights. mxfp4
  at n=8 (277-292 us) is now *slower* than FP4 (173), which is suspicious for a
  simpler format and worth one experiment. Would be an upstream-able change.

### Feature gaps in the port

5. `Q3_0_ROCMFPX` / `Q6_0_ROCMFPX` integer-dot paths. fp3 needs a bit-window gather
   that does not fit the current `cache_b` layout; the fork's fp6 int path assumes
   the broken unpacked layout this tree fixes, so it must be written fresh. Less
   urgent than it was — the float path for both is now within ~1.5x of the matching
   K-quant instead of 5x.
6. `Q2_0_ROCMFPX` has no Vulkan kernels at all (CPU only).
7. HIP/CUDA and OpenCL kernels not ported. The HIP sources ship in `ggml/rocmfp4/`
   and `ggml/rocmfpx/` but are not built.
8. Flash-attention with ROCmFPx KV types not ported. `ggml_vk_fa_type_needs_shmem`
   and `ggml_vk_fa_scalar_uses_mmq` deliberately exclude them.

### Possible next merge

9. **TurboQuant** (types 105/106, KV-cache formats at 3.5 / 4.5 bpw). IDs already
   reserved. Smaller surface than ROCmFPx (~200 refs / 22 files) but it is really a
   flash-attention project: 90 refs in `ggml-quants.c`, 63 across `flash_attn*.glsl`,
   33 in `llama-kv-cache.cpp`. **Design question to settle first:** TurboQuant ships
   its own FWHT rotation, but upstream now has its own Hadamard attention-rotation
   path (`attn_rot_k`/`attn_rot_v`) that postdates the fork — which one wins?
10. "rotorquant" does not exist anywhere in that fork or its remote branches. Need a
    pointer before it can be evaluated.

## Gotchas

- **Benchmarking.** This APU gives the *first* run of any set a ~15% boost-clock
  advantage. Always interleave configs and discard a warm-up. Several early results
  this session were pure run-order artefacts. Check
  `/sys/class/drm/card*/device/gpu_busy_percent` before trusting anything — a
  forgotten `llama-server` at 97% GPU invalidated a whole round of measurements.
- **DFlash2 sidecars need `pr-27342`.** Without that merge they fail with
  `wrong number of tensors; expected 81, got 58`. That error means the branch is
  missing, not that the GGUF is bad.
- **Known-bad model:** `kingjones777/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF` emits `////`
  garbage on **both CPU and Vulkan**, and does the same on the fork's own build. The
  GGUF is broken; do not use it to judge the port. `lmcoleman/Qwen3.8-27B-ROCmFPX-GGUF`
  works fine.
- Upstream renamed `llama-cli`; it is now `llama cli` (subcommand of `llama`).
- `-md` is not accepted by `llama completion`, only by `llama cli` / `serve`.
- **`GGML_VK_PERF_LOGGER=1` cannot profile speculative decoding.** With a draft
  model loaded it aborts on `GGML_ASSERT(ctx->compute_ctx.expired())` in
  `ggml-vulkan.cpp` after two graphs. `GGML_VK_PERF_LOGGER_CONCURRENT=1` does not
  help. Use `llama-batched-bench -npl 1,2,4,8` as the stand-in for the verification
  batch, and `GGML_VK_PIPELINE_STATS=<pipeline name>` for per-shader VGPR /
  instruction / inverse-throughput counts (needs a run that actually creates that
  pipeline; names are the `ggml_vk_create_pipeline` strings, e.g.
  `mul_mat_vec_rocmfp4_q8_1_f32`).
- **`RADV_DEBUG=shaderstats` prints almost nothing here** because llama.cpp compiles
  pipelines lazily. `GGML_VK_PIPELINE_STATS` is the tool that works.
- **`test-backend-ops -p` is a regex over the full `vars()` string**, so
  `-p "type_a=q4_0_rocmfp4,m=4096"` matches nothing — `type_b=f32` sits in between.
  Use `-p "type_a=(q4_0_rocmfp4|q4_K),type_b=f32,m=4096,n=8,"`.
- **The `m=4096 k=14336` perf case is MALL-resident.** The A matrix is ~33 MB against
  this APU's 32 MB Infinity Cache, so it reports effective bandwidth above the 256
  GB/s DRAM peak and understates any change that is really about memory. Confirm
  every kernel win with `llama-batched-bench` on a real model.
- **A stale `test-backend-ops` binary silently drops types.** `all_types` gained the
  ROCmFPx entries late; a binary older than `tests/test-backend-ops.cpp` reports
  "OK" with zero cases for `-p rocmfp`. Check timestamps before believing an empty
  sweep.

## Repro commands

```sh
# correctness
test-backend-ops -b Vulkan0
GGML_VK_FORCE_MMVQ=1 test-backend-ops -b Vulkan0

# perf, interleaved
llama-bench -m MODEL -p 512 -n 128 -r 5
llama-batched-bench -m MODEL -ngl 99 -fa on -npp 128 -ntg 32 -npl 1,2,4,8

# per-type mat-vec sweep (run twice, discard the first pass)
test-backend-ops perf -o MUL_MAT -b Vulkan0 \
  -p "type_a=(q4_0_rocmfp4|q6_0_rocmfpx|q4_K|q6_K),type_b=f32,"

# per-shader register / instruction counts
GGML_VK_PIPELINE_STATS=mul_mat_vec_rocmfp4_q8_1_f32 \
  test-backend-ops perf -o MUL_MAT -b Vulkan0 -p "type_a=q4_0_rocmfp4,"

# tensor type mix of a GGUF (a "FP4" model is usually mixed)
python3 -c "import sys; sys.path.insert(0,'gguf-py'); from gguf import GGUFReader
from collections import Counter
r=GGUFReader('MODEL.gguf'); c=Counter()
[c.update([t.tensor_type.name]) for t in r.tensors]; print(c)"

# spec decoding + acceptance rate
llama cli -m MAIN -md DRAFT --spec-type draft-dflash --spec-draft-n-max 3 \
  -ngl 99 -fa on -n 256 --temp 0 -no-cnv -st -v -p PROMPT 2>&1 \
  | grep -oE 'accepted [0-9]+/[0-9]+'

# perplexity
llama-perplexity -m MODEL -f wiki.test.raw -ngl 99 -fa on --chunks 40 -c 2048
```
