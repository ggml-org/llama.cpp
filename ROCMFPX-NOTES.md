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

### But FP4 loses under speculative decoding

Same prompt, DFlash2 draft, `--temp 0`:

| | no spec | spec n-max 3 | spec n-max 7 |
|---|---|---|---|
| FP4 | **12.56** | 16.2 | 10.9 |
| unsloth | 11.71 | **20.5** | **14.5** |

Not an acceptance problem — FP4 accepts slightly *more* (49.8% vs 46.7% at n-max 3).
The cause is batch behaviour: verification runs at batch 3-8, where the FP4 kernels
lose to the K-quant kernels.

### Kernel level, m=4096 k=14336 (GFLOPS)

| type | n=1 | n=2 | n=4 |
|---|---|---|---|
| `q4_0_rocmfp4` | 382.5 | 562.7 | 887.6 |
| `q4_0_rocmfp4_fast` | 431.1 | 666.5 | 966.7 |
| `q8_0_rocmfpx` | 283.4 | 523.3 | 881.9 |
| `q4_K` | 459.6 | 762.2 | - |
| `q5_K` | 337.7 | 583.9 | 859.1 |

**Dual-scale costs 13-18%** vs the single-scale `_FAST` layout, at every batch size.
`q4_K` still beats even `_FAST` at n=1/n=2.

### Integer-dot (MMVQ) contribution

MMVQ is **on by default** (`mmvq_mode = 0` -> heuristic, true whenever n > 1). No flag
needed; `GGML_VK_DISABLE_MMVQ` / `GGML_VK_FORCE_MMVQ` only override.

| | gain |
|---|---|
| batch 1 | +0.6% |
| batch 8 (synthetic) | +4.6% |
| real dflash spec workload | +4.3% |
| pp512 | none |

MMQ for prefill is **neutral on this device** — a control with stock `q8_0` (which has
had MMQ upstream forever) behaves identically, because KHR_coopmat matmul already wins
here. Kept anyway; it should help devices without coopmat.

### DFlash2 tuning — `spec-draft-n-max` is content-dependent

Acceptance drives everything, and acceptance depends on what is being generated:

| Content | acceptance | tok/step | n-max 3 | n-max 7 |
|---|---|---|---|---|
| Prose / reasoning | 29.5% | 3.02 | **20.4** | 14.6 |
| HTML | - | - | **21.7** | 19.1 |
| Number sequence | 77.1% | 6.40 | 29.3 | **41.0** |

`/models/models.ini` uses `spec-draft-n-max = 7` for the three `reasoning-*` presets.
Those are prose workloads, where 7 costs ~40% versus 3-4. The MTP presets in the same
file already use 3.

## TODOs

### Closing the FP4 batched gap (most promising first)

1. **Try `Q4_0_ROCMFP4_FAST` instead of dual-scale.** Free 13-18% at kernel level and
   0.25 bpw smaller. Needs a re-quantized model to confirm end-to-end, and a PPL run
   to price the quality loss. Cheapest experiment with the largest expected payoff.
2. **Optimise the dual-scale path.** Two int32 accumulators and two scale multiplies
   per 32-element block, where K-quants use one. See `mmq_dot_product` /
   `mmvq_dot_product` under `DATA_A_ROCMFP4` in `mul_mmq_funcs.glsl` /
   `mul_mat_vecq_funcs.glsl`.
3. **Check the NUM_COLS > 4 path for FP4.** This is exactly the shape of the IQ3_S
   register-spill fix already on this branch (`TPB = NUM_COLS <= 4 ? 8 : 16` in
   `mul_mat_vec_iq3_s.comp`). The perf sweep above returned no data at n=8/16, which
   is itself worth chasing. Strong candidate for the same class of win.
4. **Profile which pipeline actually runs at n=4.** Verification batch is neither the
   n=1 mat-vec regime nor the large-N coopmat prefill regime; possibly neither path
   is tuned for it.

### Feature gaps in the port

5. `Q3_0_ROCMFPX` / `Q6_0_ROCMFPX` integer-dot paths. fp3 needs a bit-window gather
   that does not fit the current `cache_b` layout; the fork's fp6 int path assumes
   the broken unpacked layout this tree fixes, so it must be written fresh.
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

## Repro commands

```sh
# correctness
test-backend-ops -b Vulkan0
GGML_VK_FORCE_MMVQ=1 test-backend-ops -b Vulkan0

# perf, interleaved
llama-bench -m MODEL -p 512 -n 128 -r 5
llama-batched-bench -m MODEL -ngl 99 -npp 128 -ntg 64 -npl 1,2,4,8

# spec decoding + acceptance rate
llama cli -m MAIN -md DRAFT --spec-type draft-dflash --spec-draft-n-max 3 \
  -ngl 99 -fa on -n 256 --temp 0 -no-cnv -st -v -p PROMPT 2>&1 \
  | grep -oE 'accepted [0-9]+/[0-9]+'

# perplexity
llama-perplexity -m MODEL -f wiki.test.raw -ngl 99 -fa on --chunks 40 -c 2048
```
