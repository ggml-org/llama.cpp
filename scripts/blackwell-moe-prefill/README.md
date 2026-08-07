# Blackwell MoE prefill experiments

This directory keeps the experiment commands and decision gates for the GPT-OSS
MXFP4 prefill investigation. The core backend changes remain disabled unless the
corresponding build or conversion option is selected.

This is a research branch, not one proposed upstream patch. It also contains
Qwen NVFP4 prefill, CUTLASS decode tests, native CUDA ceilings, and Attention
experiments. The measured results are in [`RESULTS.md`](RESULTS.md).

The parts worth splitting for review are the fused W13 model layout, shared MoE
scheduling and epilogues, the optional SM120 CUTLASS W13/W2 path, and Qwen
NVFP4 prefill as a follow-up. The relaxed Attention ceiling, CUTLASS decode,
large sweep matrix, and one-off probes should stay in the research branch.

## Baseline pipeline profile

Build the CUDA backend with NVTX support:

```bash
cmake -S . -B build-sm120 \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_MOE_PROFILE=ON \
  -DGGML_CUDA_GRAPHS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-sm120 -j 25
```

Re-run the configure command before the incremental build so CMake discovers
the added CUDA source files.

With CMake versions that do not provide `CUDA::nvtx3`, pass the root containing
`nvtx3/nvToolsExt.h` explicitly:

```bash
-DGGML_CUDA_NVTX_INCLUDE_DIR=/path/to/nvtx/include
```

Run a profile. The executable also requires `GGML_CUDA_MOE_PROFILE=1` at
runtime, so a profiling build has no NVTX activity by default.

```bash
bash profile_llama.sh baseline \
  build-sm120/bin/llama-bench \
  /models/gpt-oss-120b-MXFP4.gguf \
  results
```

The profile contains a range for every `ffn_moe_*` graph node and nested ranges
for `ids_helper`, activation quantization, and the grouped GEMM launch. A fused
top-k routing launch is recorded as `ffn_moe.routing`.

## Fused W13 GGUF layout

Create a second GGUF from the same source weights:

```bash
python convert_hf_to_gguf.py /models/gpt-oss-120b \
  --outfile /models/gpt-oss-120b-fused-w13.gguf \
  --outtype auto \
  --fuse-gate-up-exps
```

The flag stores gate followed by up in `blk.*.ffn_gate_up_exps`. The loader still
accepts the old separate gate and up tensors. Profile both GGUF files with the
same binary and arguments. Use `compare_logits.py` on raw float32 logits produced
by the debug or perplexity tools before accepting a performance result.

## Native grouped MMQ

The backend path has no FlashInfer or CUTLASS dependency. It reuses the CUDA
MXFP4 MMA primitives already present in ggml-cuda. The standalone MMQ cases cover
the 2880-row W2 and 5760-row fused W13 shapes:

```bash
bash bench_llama_mmq.sh \
  build-sm120/bin/test-backend-ops \
  results
```

Compare only runs from the same GPU, clocks, power mode, and software versions.

## Direct CUTLASS grouped GEMM

The optional CUTLASS path uses the same fused GPT-OSS graph and expert plan as
the native experiments. It replaces W13 and W2 with the SM120 block-scaled
FP8xFP4 grouped GEMM. The original MXFP4 tensor storage is compacted in place;
only the swizzled scale-factor buffers remain as extra model memory.

The best pp8192 result from each experiment is shown together here:

| Version | Best ubatch | Prefill | Share of vLLM |
| --- | ---: | ---: | ---: |
| Existing llama.cpp | 2048 | 11,738.58 tok/s | 31.9% |
| Canonical persistent | 2048 | 14,875 tok/s | 40.4% |
| Strict native CUDA | 8192 | 17,708 tok/s | 48.1% |
| Native CUDA full ceiling | 8192 | 23,301 tok/s | 63.3% |
| Direct CUTLASS | 8192 | 24,762.74 tok/s | 67.3% |
| Optimized CUTLASS support | 8192 | 25,713.80 tok/s | 69.8% |
| vLLM FlashInfer CUTLASS | 8192 | 36,819 tok/s | 100% |

The native full ceiling changes the Attention evaluation order. The direct
CUTLASS path changes the MoE intermediate format, while its optimized support
version is bitwise identical to the earlier direct CUTLASS result. `RESULTS.md`
contains the correctness and profiling details.

Configure against the CUTLASS revision that added the SM120 small-N grouped
block-scaled kernels. The CUTLASS translation unit is compiled for `120f`; the
rest of ggml-cuda keeps the architectures selected by the normal build.

```bash
git -C /path/to/cutlass checkout b46b16d003484063bca4ed365e44095c4c6ed633
cmake -S . -B build-sm120-cutlass \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_CUTLASS_MOE=ON \
  -DGGML_CUDA_CUTLASS_PATH=/path/to/cutlass \
  -DGGML_CUDA_GRAPHS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-sm120-cutlass -j 25
```

The CUTLASS translation unit is compiled directly for `compute_120f` and
`sm_120f`, which requires CUDA 12.9 or newer. Its object target disables CMake's
automatic CUDA architecture flags because older CMake releases reject the `f`
suffix. The configure step verifies the CUTLASS commit before compilation. The
rest of the CUDA backend keeps the normal architecture list.

Run the isolated GEMM, W13-fused, complete MoE, and full prefill ceiling
variants in separate processes:

```bash
bash bench_cutlass_moe.sh \
  build-sm120-cutlass/bin/llama-bench \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

Before loading the full model, run all six grouped-kernel configurations on the
backend MoE graph. The script exits on the first failed configuration:

```bash
bash validate_cutlass_kernels.sh \
  build-sm120-cutlass/bin/test-backend-ops \
  results
```

After correctness passes, collect the 512, 2048, and 8192-token component
timings for the 36 independent W13/W2 combinations:

```bash
bash sweep_cutlass_kernels.sh \
  build-sm120-cutlass/bin/test-backend-ops \
  results
```

`GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=none|w13|full` selects how much of the
surrounding pipeline is fused. `none` scatters both grouped-GEMM outputs back
to the existing graph layout. `w13` also fuses bias, SwiGLU, and A2 MXFP8
quantization. `full` additionally performs W2 bias, routing weight, and expert
reduction without materializing the routed output.

W13 and W2 select their grouped GEMM independently:

```text
GGML_CUDA_MOE_MMQ_CUTLASS_W13_TILE_N=0|32|64|128
GGML_CUDA_MOE_MMQ_CUTLASS_W2_TILE_N=0|32|64|128
GGML_CUDA_MOE_MMQ_CUTLASS_W13_SWAP_AB=0|1
GGML_CUDA_MOE_MMQ_CUTLASS_W2_SWAP_AB=0|1
```

Zero selects 32, 64, or 128 from the average routed rows per expert. The
default swaps A and B so TileN follows the routed-token dimension. Each stage
has six explicit tile and swap combinations. All variants use
`KernelScheduleAuto`, GPU-resident problem metadata, and the CUDA pool for
CUTLASS workspace. PDL is disabled by default.

The validation and benchmark scripts require the CUTLASS configuration line in
stderr. A graph mismatch or silent fallback therefore fails the run instead of
recording native MMQ numbers as CUTLASS results.

The packed values replace the original device tensor contents. Unsupported
graphs can fall back before the first repack. After repacking, the same weights
cannot be consumed by an MMQ fallback. Use a separate process for each variant.
CUTLASS repacking uses the dedicated weight stream by default; W13 waits for its
own transform while the W2 transform can continue in parallel. Set
`GGML_CUDA_MOE_MMQ_REPACK_ASYNC=0` to serialize the first-use transform.

`GGML_CUDA_MOE_MMQ_CUTLASS_DECODE=1` enables the experimental one to eight
token CUTLASS paths. GPT-OSS uses its fused MXFP4 graph, while Qwen NVFP4
dispatches each expert `MUL_MAT_ID` through an NVFP4 x NVFP4 grouped GEMM. In
both cases each routed row is a separate GEMM problem. GPT-OSS decode builds a
direct identity route plan in one kernel instead of sorting the one to 32
routed rows with the prefill histogram and prefix-sum scheduler.

The GPT-OSS call chain, storage lifetime, memory budget, and remote acceptance
gates are recorded in
[`gpt_oss_mxfp4_decode.md`](gpt_oss_mxfp4_decode.md). Use
`validate_cutlass_decode.sh` for backend correctness,
`validate_gpt_oss_mxfp4_decode_logits.sh` for separate-process full-model
logits, `bench_gpt_oss_mxfp4_decode.sh` for throughput, and
`profile_gpt_oss_mxfp4_decode_nsys.sh` for the native/CUTLASS Nsys comparison.

The Qwen path is limited to 256 experts, top-8 routing, aligned expert shapes,
and SM120. Separate gate/up with N=512 and fused W13 with N=1024 are both
accepted. It repacks the canonical `block_nvfp4` values and UE4M3 scales into
CUTLASS layouts, dynamically quantizes A with 16-value scale vectors, and
scatters BF16 results back into the existing graph. Per-expert global scales,
standard SwiGLU, routing weights, reduction, and the shared expert remain on
the existing llama.cpp path. CUDA graphs must be disabled. The path is accepted
only when stderr contains `MoE CUTLASS NVFP4 decode dispatch`.

`GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_FUSED=1` enables the strict M=1 Qwen oracle.
It quantizes the hidden state once, joins gate and up into one routed W13 GEMM,
quantizes the SwiGLU output in the W13 epilogue, runs one routed W2 GEMM, and
applies the down scale, routing weight, and expert reduction in one final
kernel. Gate and up use a persistent paired CUTLASS cache; W2 uses the existing
in-place CUTLASS layout. The matcher accepts only the 2048/512, 256-expert,
top-8 NVFP4 graph with standard SwiGLU. Other graphs use the native path. A run
is accepted only when stderr contains `MoE CUTLASS NVFP4 fused decode dispatch`.
If a matching M=1 expert GEMM reaches the per-op dispatcher, the strict oracle
aborts instead of recording a partially fused result.

The oracle keeps the canonical gate/up tensors and adds about 288 MiB of paired
W13 values and scales per matched MoE layer. The in-place W2 layout adds about
16 MiB of scale storage per layer. This memory cost is intentional for the
performance ceiling experiment and is not an upstream cache design.

Two accuracy experiments isolate the remaining numerical differences:

```text
LLAMA_CUDA_MOE_NVFP4_INPUT_SCALE=1
GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_OUTPUT_F32=1
```

The first applies the stored per-expert activation scales before W13 and W2 and
restores the scale after each GEMM. The second keeps CUTLASS output in F32
instead of rounding it to BF16 before scattering it into the graph. Both are
off by default.

## Qwen NVFP4 prefill

The full Qwen prefill path is selected by `GGML_CUDA_MOE_MMQ_BACKEND=cutlass`.
It is limited to SM120, 256 experts, top-8 routing, hidden size 2048, expert
size 512, standard SwiGLU, NVFP4 expert weights, and at least 256 tokens. The
matcher replaces only the routed expert branch. A shared expert and its gate
remain ordinary graph nodes and consume the fused routed result.

The path performs these steps on one CUDA stream:

1. Copy strided expert IDs and routing weights to compact buffers.
2. Build one histogram and prefix-sum schedule for W13 and W2.
3. Quantize each token once to expert-sorted NVFP4 A1 storage.
4. Run one grouped W13 GEMM over a persistent paired gate/up layout.
5. Apply gate/up scales and SwiGLU while producing NVFP4 A2 storage.
6. Run grouped W2 and apply the down scale, routing weight, and top-8 reduction.

The paired W13 cache and copied W2 cache preserve the canonical GGUF tensors,
so MMVQ and unsupported graphs can still use the original weights. The cache
cost is about 432 MiB per routed MoE layer for this shape. Transient buffers at
8192 tokens require about another 480 MiB and are returned to the CUDA pool
after the fused call. CUDA graph capture occurs only after warmup has populated
the persistent weight cache; subsequent captures reuse the same device
addresses.

Graphs that use `LLAMA_CUDA_MOE_NVFP4_INPUT_SCALE=1`, fused gate/up tensors,
other shapes, short batches, non-CUDA buffers, or a different activation fall
back without changing the canonical weights.

The call chain and storage contracts are recorded in
[`qwen_nvfp4_prefill.md`](qwen_nvfp4_prefill.md).

Run backend correctness tests in eager and CUDA graph modes:

```bash
bash validate_qwen_nvfp4_prefill.sh \
  build-sm120-cutlass/bin/test-backend-ops \
  results
```

Compare full-model prefill logits against the native path:

```bash
bash validate_qwen_nvfp4_prefill_logits.sh \
  build-sm120-cutlass/bin/llama-debug \
  /models/Qwen3.6-35B-A3B-NVFP4.gguf \
  results
```

Measure native eager, CUTLASS eager, and CUTLASS with CUDA graphs:

```bash
bash bench_qwen_nvfp4_prefill.sh \
  build-sm120-cutlass/bin/llama-bench \
  /models/Qwen3.6-35B-A3B-NVFP4.gguf \
  results
```

Collect native and CUTLASS Nsys profiles with stage-level NVTX ranges:

```bash
bash profile_qwen_nvfp4_prefill_nsys.sh \
  build-sm120-cutlass/bin/llama-bench \
  /models/Qwen3.6-35B-A3B-NVFP4.gguf \
  results
```

Validate the one, two, four, and eight-token backend graphs against the native
path before measuring the full model:

```bash
bash validate_cutlass_decode.sh \
  build-sm120-cutlass/bin/test-backend-ops \
  results
```

Validate Qwen NVFP4 W13 and W2 shapes separately. The script covers M=1, 2, 4,
and 8 for each individual GEMM, then checks complete W13, SwiGLU, W2, routing,
and reduction blocks at M=1 and 8. Both BF16 and F32 per-op outputs are tested,
followed by the fused M=1 oracle:

```bash
bash validate_cutlass_nvfp4_decode.sh \
  build-sm120-cutlass/bin/test-backend-ops \
  results
```

Validate the M=1 MoE path with single-token full-model logits. The default
prompt is `Hello`; set `DECODE_LOGITS_PROMPT` if it is not one token for the
model tokenizer. The script records native, calibrated native, dynamic CUTLASS,
calibrated CUTLASS, BF16 output, and F32 output in separate processes. It writes
NMSE, NRMSE, absolute-error percentiles, cosine similarity, KL divergence, and
top-k overlap to `summary.md`. Set `VLLM_LOGITS_FILE` to a raw float32 logits
file to add the same comparisons against vLLM:

```bash
bash validate_cutlass_nvfp4_decode_logits.sh \
  build-sm120-cutlass/bin/llama-debug \
  /models/Qwen3.6-35B-A3B-NVFP4.gguf \
  results
```

Compare native MMVQ and the calibrated/dynamic CUTLASS variants with the same
generation workload. Set `DECODE_FULL_MATRIX=1` to add the dynamic F32 case.
The runner writes the throughput comparison to `summary.md`:

```bash
bash bench_cutlass_decode.sh \
  build-sm120-cutlass/bin/llama-bench \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

Use the same A/B runner for Qwen NVFP4:

```bash
bash bench_cutlass_decode.sh \
  build-sm120-cutlass/bin/llama-bench \
  /models/Qwen3.6-35B-A3B-NVFP4.gguf \
  results
```

The decode Nsys runner profiles native, dynamic BF16 CUTLASS, fused M=1 BF16,
calibrated BF16 CUTLASS, and calibrated F32 CUTLASS in separate processes. Set
`DECODE_NSYS_FULL_MATRIX=1` to add calibrated native and dynamic F32. It keeps
the one-token llama-bench warmup so the timed CUTLASS run does not include
weight repacking. `summary.md` reports the transform separately and normalizes
the remaining kernel totals by the warmup and measured decode steps.

```bash
bash profile_cutlass_decode_nsys.sh \
  build-sm120-cutlass/bin/llama-bench \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

Passing the Qwen NVFP4 GGUF to the same script produces the native MMVQ versus
CUTLASS comparison and separates scheduling, activation quantization, grouped
GEMM, one-time weight repacking, and the remaining graph epilogues.

The GPT-OSS CUTLASS path writes BF16 intermediates and uses MXFP8 activations,
so its logits are not expected to be bitwise identical to the native MXFP4
path. Record the full metrics before timing:

```bash
bash validate_cutlass_moe.sh \
  build-sm120-cutlass/bin/llama-debug \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

For Nsys, export the CUTLASS mode before calling `profile_llama.sh`:

```bash
GGML_CUDA_MOE_MMQ_BACKEND=cutlass \
GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=full \
bash profile_llama.sh cutlass-full \
  build-sm120-cutlass/bin/llama-bench \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

Use the CUTLASS matrix profiler to compare the native TMA path, the earlier
W13-32/W2-64 choice, fixed TileN 32/64/128, and swapped versus unswapped
operands under the same pp8192 workload:

```bash
bash profile_cutlass_moe_nsys.sh \
  build-sm120-cutlass/bin/llama-bench \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

Each process performs one warmup pass and one measured pass. The generated
`summary.md` normalizes the NVTX totals to one 36-layer pass and separates
input quantization, W13, the W13 epilogue, W2, and final reduction. Raw reports
and a machine-readable `cutlass-stages.csv` are retained. Restrict the matrix
with `PREFILL_CUTLASS_NSYS_CASES`; use `all` to select every defined case.

### CUTLASS scope and optimized CUDA support

The `full` setting replaces the complete MoE graph sequence, but CUTLASS itself
only runs the W13 and W2 grouped GEMMs. Expert scheduling, input expansion and
quantization, the W13 activation, and W2 finalization are ggml-cuda kernels.
The current path is:

```text
shared expert plan
  -> MXFP8 input quantization and route expansion
  -> CUTLASS W13
  -> bias, SwiGLU, and A2 quantization
  -> CUTLASS W2
  -> bias, routing weight, and expert reduction
```

The optimized support kernels are selected by default for the CUTLASS backend.
Each stage can fall back to the earlier CUTLASS support path independently:

```text
GGML_CUDA_MOE_MMQ_CUTLASS_PREFIX_DISABLE=1
GGML_CUDA_MOE_MMQ_CUTLASS_CTA_QUANT_DISABLE=1
GGML_CUDA_MOE_MMQ_CUTLASS_CTA_ACTIVATION_DISABLE=1
```

The W13 activation CTA can cover 1, 4, or 8 routed rows. Four is the default;
the other shapes remain available for the SM120 sweep:

```text
GGML_CUDA_MOE_MMQ_CUTLASS_ACTIVATION_ROWS=1|4|8
```

The final pp8192 result below uses one routed row per CTA.

The prefix scheduler uses block histograms, a device prefix sum, and a stable
scatter. The input kernel gives one 256-thread CTA to each token and quantizes
the broadcast hidden state once before writing its routed rows. The W13
activation kernel gives one 256-thread CTA to a small routed-row tile and
processes two 32-value scale groups per warp.

Run the old support path, each cumulative stage, and the complete path through
the same backend correctness cases:

```bash
bash validate_cutlass_support.sh \
  build-sm120-cutlass/bin/test-backend-ops \
  results
```

The validation script sets `GGML_CUDA_MOE_MMQ_CUTLASS_VALIDATE_SUPPORT=1`.
This compares the new route maps, MXFP8 values, and scale buffers byte for byte
with the earlier CUTLASS support kernels before each GEMM. It synchronizes the
CUDA stream and must not be enabled during timing runs.

The final pp8192 support-kernel profile is:

| Component | Earlier range | Optimized range | Optimized kernels | vLLM named kernels |
| --- | ---: | ---: | ---: | ---: |
| Histogram, prefix sum, and permutation | 8.649 ms | 6.578 ms | 0.345 ms | about 0.924 ms |
| Route expansion and input MXFP8 quantization | 15.588 ms | 9.526 ms | 3.945 ms | about 4.012 ms |
| W13 bias, SwiGLU, and A2 quantization | 58.301 ms | 12.973 ms | 12.129 ms | 13.384 ms |

The range column includes launches and gaps. The kernel column contains only
GPU kernel time. The optimized path is bitwise identical to the earlier
CUTLASS support path. In an unprofiled same-build A/B run it improved pp8192
from 21,608 to 25,714 tok/s. The complete MoE range fell from 218.819 to
165.725 ms.

The support compute kernels are no longer the main gap. The remaining work is
launch integration, stable W13/W2 tile selection, W2 finalization, and the
Attention and RoPE path. See `RESULTS.md` for the full Nsys split and the vLLM
comparison.

## Fused graph and persistent grouped MMQ

`bench_native_moe.sh` runs three modes against the fused-W13 GGUF:

```bash
bash bench_native_moe.sh \
  build-sm120/bin/llama-bench \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

- `baseline` disables the fused MoE graph path.
- `fused-generic` shares the expert plan and fuses the W13 and W2 epilogues,
  but keeps the existing grouped MMQ launcher.
- `persistent` also enables the SM120 compact persistent scheduler. The kernel
  reads the canonical GGUF MXFP4 weights directly.

Timed repetitions measure steady-state prefill. Set `GGML_CUDA_MOE_MMQ_DISABLE=1` or
`GGML_CUDA_MOE_MMQ_PERSISTENT_DISABLE=1` for manual A/B runs.

Validate final logits before profiling the persistent path:

```bash
bash validate_native_moe.sh \
  build-sm120/bin/llama-debug \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

This compares both fused-generic and persistent logits against the disabled
path. The backend test requires the candidate and disabled CUDA outputs to be
bitwise identical. Backend CPU-reference correctness is opt-in
because the GPT-OSS expert tensors require several GiB across the CPU and CUDA
backends:

```bash
GGML_CUDA_MOE_MMQ_TEST=1 build-sm120/bin/test-backend-ops \
  test -b CUDA0 -o MOE_MMQ
```

## Performance acceptance gates

Run every private integration variant through `profile_llama.sh`. Record the
measured values in the schema accepted by `evaluate.py`. The evaluator enforces
the fused-W13 and grouped-MMQ decision gates and reports the full MoE pipeline target.

```bash
python evaluate.py measurements.json
```

The minimum input is:

```json
{
  "baseline": {"pp8192_ms": 703.874},
  "fused_w13": {"pp8192_ms": 680.0},
  "grouped_mmq": {
    "m2048": {"generic_ms": 10.0, "persistent_ms": 5.0},
    "m8192": {"generic_ms": 30.0, "persistent_ms": 10.0},
    "full_model_gemm_ms": 120.0
  },
  "moe_pipeline": {"measured_ms": 155.0}
}
```

Missing measurements are reported but do not make the evaluator fail. A present
measurement that misses its decision gate returns a non-zero exit code.

## Composable CUDA components

`bench_moe_components.sh` runs each pipeline component independently and then
in combination:

```bash
MOE_COMPONENT_VALIDATE=1 bash bench_moe_components.sh \
  build-sm120/bin/test-backend-ops \
  build-sm120/bin/llama-bench \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

The cases separate shared expert planning, staged or fused activation,
activation plus A2 quantization, staged or fused finalization, and persistent
W13/W2 scheduling. Set `MOE_COMPONENT_CASES` to a comma-separated subset of
the labels in the script for shorter runs.

The scheduler has independent W13 and W2 controls:

```text
GGML_CUDA_MOE_MMQ_W13_TILE_ROWS=0|32|64|128
GGML_CUDA_MOE_MMQ_W2_TILE_ROWS=0|32|64|128
GGML_CUDA_MOE_MMQ_W13_CTA_MULTIPLIER=1..8
GGML_CUDA_MOE_MMQ_W2_CTA_MULTIPLIER=1..8
GGML_CUDA_MOE_MMQ_W13_OUTPUT_TILE_MAJOR=0|1
GGML_CUDA_MOE_MMQ_W2_OUTPUT_TILE_MAJOR=0|1
```

Run the isolated scheduler matrix with:

```bash
bash sweep_moe_schedule.sh build-sm120/bin/test-backend-ops results
```

The full-model ubatch sweep keeps pp8192 fixed and compares the disabled,
persistent, and persistent plus w13-epilogue-quant paths:

```bash
bash sweep_moe_ubatch.sh \
  build-sm120/bin/llama-bench \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

`bench_moe_padding.sh` compares the native 2880 shapes with synthetic 2944
weights and activations. It measures the aligned kernel shapes without model
repack time or the memory cost of a second packed-weight copy:

```bash
MOE_PADDING_VALIDATE=1 bash bench_moe_padding.sh \
  build-sm120/bin/test-backend-ops results
```

Measure the one-time CUDA repack separately. This tool reports transform cost,
memory traffic, and output size without invoking an MMQ consumer. Use
`bench_moe_weight_layouts.sh` below to measure the matching persistent loaders.

```bash
bash bench_moe_repack.sh /path/to/llama.cpp results
```

## Contiguous KQ mask

The host KQ mask builder has a fast path for one causal sequence with contiguous
positions and contiguous KV cells. All other layouts use the existing builder.
Compare the full-ubatch path with:

```bash
KQ_MASK_VALIDATE=1 bash bench_kq_mask.sh \
  build-sm120/bin/llama-bench \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

Validate the host and CUDA-generated masks against the original full-model
logits before timing:

```bash
bash validate_kq_mask.sh \
  build-sm120/bin/llama-debug \
  /models/gpt-oss-120b-fused-w13.gguf \
    results
```

Measure the F16 fill and diagonal-mask kernels independently at the full KQ
matrix sizes. Their summed time estimates the CUDA mask materialization cost:

```bash
bash bench_kq_mask_cuda.sh \
  build-sm120/bin/test-backend-ops \
  results
```

`LLAMA_KQ_MASK_CONTIGUOUS_DISABLE=1` restores the existing implementation.
`LLAMA_KQ_MASK_CONTIGUOUS_VALIDATE=1` builds both masks and requires byte-identical
results before the graph executes. `LLAMA_KQ_MASK_CONTIGUOUS_LOG=1` records whether
each mask used the fast path.

`LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1` replaces the full KQ mask with a one-element graph
marker. FlashAttention derives the upper causal bound and the standard
sliding-window lower bound from the query tile. It is limited to a single
causal sequence starting at position zero with no existing KV prefix or ALiBi.
Unsupported SWA types and batches keep the host input path.
`LLAMA_CUDA_FATTN_DIRECT_CAUSAL_LOG=1` records whether the marker path was selected.

`LLAMA_CUDA_FATTN_Q_ROPE=1` attaches the Q RoPE parameters to that direct-causal
FlashAttention op and rotates Q while loading its tile. K remains rotated once
when it is written to the KV cache. `GGML_CUDA_FATTN_CAUSAL_TILES=1` gives each
output tile to one CTA and removes Stream-K fixup. The SM120 schedule also
enables exact causal tile ownership and can be selected with:

```text
GGML_CUDA_FATTN_SM120_CAUSAL_SCHEDULE=1
```

Run the cumulative Attention cases and validate their logits with:

```bash
bash bench_attention_stages.sh \
  build-sm120/bin/llama-bench \
  /models/gpt-oss-120b-fused-w13.gguf \
  results

bash validate_attention_stages.sh \
  build-sm120/bin/llama-debug \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

## Repacked persistent MMQ

The persistent MXFP4 kernel can consume five independently selectable weight
layouts:

```text
GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=canonical|interleaved|split|tma|tma-inplace
```

`interleaved` stores one 512-value K tile as contiguous quant data followed by
its E8M0 scales. `split` stores quant data and scales in separate aligned planes.
Both layouts pad the K block count to the kernel's 512-value iteration size.
`tma` stores each 512-value row segment as 256 data bytes followed by 16 E8M0
scales. CUDA tensor maps load 128 output rows cooperatively, or 96 output rows
when `GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1` reserves producer warps.

`tma-inplace` rewrites each loaded CUDA weight tensor in place. Full 512-value K
tiles use the TMA record layout, while the final partial tile remains in compact
canonical MXFP4 blocks. For the GPT-OSS 2880 K dimension this layout has exactly
the same byte count as the original tensor and does not keep a second copy of
the expert weights. It is a strict experiment: CUDA graphs must be disabled,
`GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1` must be set, and every later consumer of the
mutated tensor must use the in-place TMA path. Unsupported dispatches abort
instead of reading the changed data as canonical MXFP4.

Two reusable packed workspaces avoid per-layer allocation and host
synchronization. A repeated single-layer backend test reuses the packed tensor,
while a full model repacks each layer into the same two workspaces. This keeps
the experiment within the memory available after loading GPT-OSS-120B:

```bash
bash bench_moe_weight_layouts.sh build-sm120/bin/test-backend-ops results
```

The CUDA ceiling experiment first requires byte-identical backend results, then
compares canonical, cooperative TMA, and warp-specialized TMA at 2048 and 8192
tokens. Its full-model cases include repack cost and enable direct-causal
FlashAttention so `ubatch=8192` is measured directly:

```bash
bash bench_moe_cuda_ceiling.sh \
  build-sm120/bin/test-backend-ops \
  build-sm120/bin/llama-bench \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

`validate_native_moe.sh` also checks both TMA modes against disabled full-model
logits with zero tolerance. The TMA path requires CUDA 12.8 and SM120; all
unsupported shapes and devices leave the original graph intact.
The experiment scripts set `GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1`, which turns an
unsupported TMA dispatch into an error instead of recording fallback numbers.

Collect the same NCU sections for the cooperative and warp-specialized TMA
consumers with:

```bash
bash profile_moe_tma_ncu.sh build-sm120/bin/test-backend-ops results
```

The script selects W13 and W2 by their matched launch order and records uniform
and skewed expert distributions separately.

## Native CUDA vLLM ceiling experiment

`bench_moe_vllm_ceiling.sh` evaluates the full native CUDA ceiling path as
independent components and as combined pipelines:

```bash
bash bench_moe_vllm_ceiling.sh \
  build-sm120/bin/llama-bench \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

The experiment keeps the canonical GGUF tensor as the source of truth and
streams packed W13 and W2 tiles through a configurable cache. The asynchronous
mode uses a separate CUDA stream and waits only when each grouped GEMM consumes
its packed tensor. `GGML_CUDA_MOE_MMQ_REPACK_CACHE_ENTRIES` controls the number
of in-flight packed tensors and therefore the memory/overlap tradeoff.

The TMA kernels use the logical K dimension for the last packed tile. For
GPT-OSS this skips the three K64 MMA fragments that cover the 2880-to-3072
storage tail. Set `GGML_CUDA_MOE_MMQ_TMA_TAIL_DISABLE=1` to retain all padded
MMA work. The `tma-full-k`/`tma-fp4` and `fp4-all-full-k`/`fp4-all` pairs
isolate this difference in the generic and full epilogue pipelines.

`GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue` replaces the staged W13 output,
bias, SwiGLU, and A2 quantization launches with a paired W13 TMA kernel. Gate
and up MMAs consume each activation tile together instead of loading the same
activation tile twice. The W2 epilogue alternatives are:

```text
GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted
GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-atomic
```

`tma-weighted` writes routed expert rows and uses the deterministic reduction
kernel. `tma-atomic` accumulates directly into the final hidden state and is a
performance ceiling with non-deterministic floating-point accumulation order.

The mixed-activation ceiling path uses MXFP4 weights with MXFP8 activations and
the SM120 mixed-format MMA instruction:

```text
GGML_CUDA_MOE_MMQ_ACTIVATION_FORMAT=mxfp8
```

This changes activation precision and must be evaluated against its own logits
tolerance and model-quality gate. It is not a bitwise-compatible replacement
for the existing two-stage MXFP4 activation path.

Validate the deterministic FP4 pipeline exactly, and the atomic and MXFP8
pipelines with explicit tolerances, before accepting timing results:

```bash
bash validate_moe_vllm_ceiling.sh \
  build-sm120/bin/llama-debug \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

All ceiling modes require CUDA 12.8, SM120, fused W13 tensors, and prefill-sized
batches. Unsupported devices, layouts, shapes, and decode workloads retain the
existing llama.cpp CUDA path for non-mutating layouts. The in-place layout is a
strict experiment and aborts if a later dispatch cannot consume the rewritten
weights. The ceiling benchmark sets the strict TMA flag so an unsupported
dispatch fails instead of producing a fallback measurement.

Collect W13/W2 kernel counters for the FP4 and mixed MXFP8 variants with:

```bash
bash profile_moe_vllm_ceiling_ncu.sh \
  build-sm120/bin/test-backend-ops \
  results
```

The default matrix includes full-K and tail-elided FP4 kernels with deterministic
W2 epilogue. Add `w2-fp4-atomic` or `w2-mxfp8-atomic` to
`NCU_MOE_CEILING_CASES` when comparing the direct-reduction ceiling.

## Combined Blackwell prefill ceiling

The combined benchmark keeps each stage in a separate process so static
environment selection cannot leak between cases. It reports the original graph,
the current canonical persistent MoE path, Attention-only changes, in-place
TMA MoE, and the full deterministic combination at ubatch 2048 and 8192:

```bash
bash bench_blackwell_prefill_ceiling.sh \
  build-sm120/bin/llama-bench \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

`GGML_CUDA_ADD_RMS_NORM_FUSION=1` fuses residual add, RMS normalization, and
the normalization weight multiply while preserving the residual tensor. The
combined validation first exercises this subgraph in `test-backend-ops`, then
requires the direct-causal, in-place TMA, and strict tuned variants to match the
disabled path. The full Attention ceiling is reported as metrics only because
its changed floating-point evaluation order is not bitwise compatible:

```bash
bash validate_blackwell_prefill_ceiling.sh \
  build-sm120/bin/test-backend-ops \
  build-sm120/bin/llama-debug \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

## Final remote run

Run the final correctness, performance, and profiling suite with one command:

```bash
bash run_blackwell_prefill_final.sh \
  build-sm120/bin/test-backend-ops \
  build-sm120/bin/llama-debug \
  build-sm120/bin/llama-bench \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

The suite runs the CUDA backend tests, an 8192-token bitwise comparison for the
strict tuned path, a separate ceiling-quality measurement, the ubatch 2048 and
8192 performance matrix, and three Nsys profiles. The generated `summary.md`
links the child result directories. Override the main controls with:

```text
FINAL_PREFILL_THREADS=25
FINAL_PREFILL_REPETITIONS=3
FINAL_PREFILL_VALIDATE_TOKENS=8192
FINAL_PREFILL_CEILING_VALIDATE_TOKENS=1024
FINAL_PREFILL_MATRIX_CASES=baseline,sweet-spot,direct-tma-inplace-tuned-norm,full-ceiling-tuned
FINAL_PREFILL_MATRIX_UBATCHES=2048,8192
FINAL_PREFILL_RUN_NSYS=0|1
```

Run only the Nsys decomposition with:

```bash
bash profile_blackwell_prefill_nsys.sh \
  build-sm120/bin/llama-bench \
  /models/gpt-oss-120b-fused-w13.gguf \
  results
```

Each case is captured in a separate process with CUDA graphs disabled, one
unwarmed pp8192 repetition, CUDA and NVTX tracing, and verbose path-selection
logs. The default cases are the disabled baseline, the bitwise strict tuned
path, and the non-bitwise full ceiling. `summary.md` reports profiled throughput,
kernel-name component totals, the top CUDA kernels, and the top MoE NVTX ranges.
The cold trace includes the one-time in-place weight transform; the summary also
reports summed steady kernel time with that transform removed.
The raw result directory also contains the `.nsys-rep`, CUDA kernel, CUDA API,
NVTX, GPU, build, and command-configuration records. Profiled throughput includes
Nsys overhead; use the performance matrix for acceptance numbers.

## Nsight Compute MMQ counters

Build with `GGML_CUDA_MOE_PROFILE=ON`, then collect W13 and W2 independently:

```bash
bash profile_moe_ncu.sh build-sm120/bin/test-backend-ops results
```

The default matrix covers 2048 and 8192 tokens, uniform and skewed expert
distributions, and canonical, interleaved, and split weights. It records
SpeedOfLight, compute and memory workload, occupancy, scheduler, warp-state,
launch, and instruction sections. Narrow a run with `NCU_MOE_LAYOUTS`,
`NCU_MOE_STAGES`, `NCU_MOE_TOKENS`, and `NCU_MOE_DISTRIBUTIONS`; pass
architecture-specific counters through `NCU_EXTRA_METRICS` after checking the
installed Nsight Compute metric names.

Set `GGML_CUDA_MOE_MMQ_LOG_DISTRIBUTION=1` to log active experts, the minimum
and maximum nonempty expert populations, and scheduled row-tile fill for W13
and W2. This diagnostic synchronizes the CUDA stream once per fused MoE call
and must remain disabled during timing runs.
