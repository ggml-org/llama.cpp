# Blackwell MoE prefill experiments

This directory keeps the experiment commands and decision gates for the GPT-OSS
MXFP4 prefill investigation. The core backend changes remain disabled unless the
corresponding build or conversion option is selected.

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
