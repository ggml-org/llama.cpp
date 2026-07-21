# RDNA2 multi-GPU benchmarks

This document records performance for the RDNA2 tensor-split changes on the
[`exp-gpu-sampling`](https://github.com/edwinbrowwn/llama.cpp-rdna2/tree/exp-gpu-sampling)
branch. See [`FIXES.md`](FIXES.md) for implementation details, build flags,
runtime flags, and known limitations.

## Test system

| Component | Configuration |
|---|---|
| GPUs | 4 × AMD Radeon Pro V620 32 GB (`gfx1030`) |
| GPU memory available | 30,704 MiB per device |
| CPU | AMD Ryzen Threadripper PRO 3945WX, 12 cores |
| Backend | ROCm/HIP with RCCL |
| ROCm path | `/opt/rocm/core-7.14` |
| Split mode | Tensor, equal `1/1/1/1` split |
| Flash attention | Enabled |
| KV cache | F16 K and V |
| Batch / microbatch | 2048 / 256 |
| Context | 262,144 tokens |

The tested models were:

- **Qwen3.6-27B MTP**, mixed `Q4_K_M` with a separate BF16 output head,
  27.3B parameters and an 18.49 GB GGUF.
- **Qwen3.6-35B-A3B MTP**, Unsloth Dynamic `Q4_K_M`, 35.5B total / 3B
  active parameters, 22.65 GB GGUF.
- **Qwen3.5-122B-A10B MTP**, Unsloth Dynamic `Q4_K_M`, 124.6B total / 10B
  active parameters, 78.25 GB GGUF split across three files.

## Build and runtime configuration

Build this branch with:

```bash
./scripts/build-rdna2-rocm.sh
```

Important build options used by the helper:

| CMake option | Purpose |
|---|---|
| `GGML_HIP=ON` | Build the HIP backend. |
| `GGML_HIP_RCCL=ON` | Build and link RCCL collectives; required by vocabulary-parallel output. |
| `GGML_HIP_GRAPHS=ON` | Compile HIP graph support; this is the setting that actually enables graphs. |
| `GGML_HIP_ROCWMMA_FATTN=OFF` | Keep the newer-architecture rocWMMA attention path off for V620/RDNA2. |
| `GGML_HIP_NO_VMM=ON` | Avoid the VMM path used by newer GPU configurations. |
| `GGML_NATIVE=ON` | Optimize host code for this machine. |
| `AMDGPU_TARGETS=gfx1030` | Compile HIP kernels for the V620/RDNA2 target. |
| `CMAKE_HIP_ARCHITECTURES=gfx1030` | Pass the same architecture to CMake's HIP toolchain. |

The benchmark environment was:

```bash
export GGML_CUDA_ALLREDUCE=nccl
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_NO_SCRATCH_RECLAIM=1
```

### Runtime environment quick reference

These are the settings used by this branch's tested ROCm path, not an exhaustive
list of upstream debugging variables.

| Setting | What it does | Guidance |
|---|---|---|
| `GGML_CUDA_ALLREDUCE=nccl` | Uses RCCL/NCCL for tensor-parallel collectives. | Recommended for four V620s and **required** by `GGML_TP_VOCAB_OUTPUT`. |
| `GGML_CUDA_ALLREDUCE=internal` | Uses the experimental internal collective when supported. | Mainly useful for two-device testing; not compatible with vocabulary-parallel output. |
| `GGML_CUDA_ALLREDUCE=none` | Disables the backend collective and lets the meta backend use its generic butterfly fallback. | Debug/A-B option; slower here and not compatible with vocabulary-parallel output. |
| `GGML_HIP_GRAPHS=1` as a shell variable | No runtime effect in the current implementation. | Harmless but redundant in older command examples; use the CMake option instead. |
| `HSA_OVERRIDE_GFX_VERSION=10.3.0` | Presents the V620 as the tested `gfx1030` target. | Retained for reproducibility; it may be unnecessary on a native `gfx1030` runtime. |
| `HSA_NO_SCRATCH_RECLAIM=1` | Keeps HIP scratch allocations instead of reclaiming them between work. | Improves stability/consistency at the cost of retaining more GPU memory. |
| `GGML_TP_SHARDED_OUTPUT=1` | Splits validated Qwen 35B/122B output heads along the embedding dimension, then FP32-all-reduces full logits. | Supports raw and MTP paths. Do not combine with `GGML_TP_VOCAB_OUTPUT`. |
| `GGML_TP_VOCAB_OUTPUT=1` | Splits an eligible output head along vocabulary and exchanges compact per-row TOP_K results. | Fast path supports raw/MTP CPU sampling with finite `top_k <= 256`; grammar, penalties, and dense-logit callers materialize full sharded logits on demand. Requires RCCL. |

`GGML_CUDA_ALLREDUCE` may be left unset, in which case Linux currently tries
RCCL first. Set it explicitly to `nccl` for reproducible runs and because the
vocabulary-parallel eligibility check requires that exact selection.
Unrecognized values behave like `none` after a warning.

### Output-head modes and compatibility

| Output mode | Environment | What happens | Raw decode | MTP / parallel slots |
|---|---|---|---:|---|
| Mirrored (default) | Leave both TP output flags unset | Every rank computes the complete output head; no logits collective is needed. | Baseline | Supported |
| Embedding-sharded | `GGML_TP_SHARDED_OUTPUT=1` | Every rank computes a partial contribution for the full vocabulary; FP32 all-reduce produces complete mirrored logits. | Moderate model-dependent gain | Supported on the validated Qwen 35B/122B paths |
| Vocabulary-sharded | `GGML_TP_VOCAB_OUTPUT=1` and `GGML_CUDA_ALLREDUCE=nccl` | Every rank computes complete logits for its vocabulary slice; deterministic per-row TOP_K results are exchanged and merged. | **About 15% over mirrored** on the tested 27B model | MTP is supported when its head shares `model.output`; `--parallel 4` is tested |

The two sharding flags are alternatives, not layers. They split different axes
of the same output matrix, so setting both is an error. The measured 15% result
is **vocabulary-sharded versus mirrored**, not vocabulary-sharded versus
`GGML_TP_SHARDED_OUTPUT`. Combining both would require a separate 2-D tensor-
parallel design with subgroup reductions and is future work.

RCCL was confirmed active by the absence of the internal all-reduce and
meta-backend butterfly fallback warnings.

## Non-blocking GPU jobs

Long builds, server loads, profiler runs, and benchmarks should be launched
through the tracked job runner instead of a foreground SSH command:

```bash
job_id=$(./scripts/rdna2-job.sh start tg128 --timeout 1800 -- \
  ./scripts/run-my-benchmark.sh)
echo "$job_id"
```

The launch returns immediately. Inspect it later with short commands:

```bash
./scripts/rdna2-job.sh status "$job_id"
./scripts/rdna2-job.sh logs "$job_id" 80
./scripts/rdna2-job.sh result "$job_id"
./scripts/rdna2-job.sh stop "$job_id"
```

Jobs live under `~/llama-jobs/<job-id>/`, are serialized with a GPU `flock`,
run in their own process group, and are bounded by `timeout` plus a forced-kill
grace period. The runner records status, PID/PGID, command, logs, exit code,
and timestamps. A ROCm illegal-memory signature changes the terminal state to
`reset-required`. Benchmark wrappers can write `summary.json`, `result.json`,
or `result.jsonl` through the `LLAMA_JOB_DIR` environment variable.

## Prompt-processing scaling

These tests used `llama-server`, `--parallel 1`, and one 262k context slot.
Context checkpoints and prompt caching were disabled:

```text
--ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots
```

Each request generated one token. Prompt text consisted of a unique prefix and
a repeated string that tokenizes to exactly one token per repetition. The
reported prompt-token count includes the chat template. This isolates prompt
evaluation throughput and avoids measuring tokenization or output generation.

Each row is one fresh request. `Time` is calculated from llama.cpp's reported
prompt-token count and prompt tokens/second rather than HTTP wall time.

### Qwen3.6 35B-A3B

| Target | Actual prompt tokens | Time | Prompt processing |
|---:|---:|---:|---:|
| 8k | 8,181 | 5.159s | **1,585.65 t/s** |
| 64k | 65,526 | 47.236s | **1,387.21 t/s** |
| 128k | 131,063 | 130.914s | **1,001.14 t/s** |

### Qwen3.5 122B-A10B

| Target | Actual prompt tokens | Time | Prompt processing |
|---:|---:|---:|---:|
| 8k | 8,181 | 7.897s | **1,036.01 t/s** |
| 64k | 65,526 | 91.422s | **716.74 t/s** |
| 128k | 131,063 | 543.223s | **241.27 t/s** |

Both models completed all three context lengths without a GPU fault.

The sharp 122B decline at 128k is reproducible for this configuration and is
likely dominated by long-context attention/KV traffic rather than model-weight
bandwidth alone. The repeated synthetic prompt is suitable for measuring
compute throughput but is not representative of prompt quality or cache reuse.

## Standard `llama-bench`

The standard benchmark used pp512 and tg128, three measured repetitions,
four-GPU RCCL tensor split, flash attention, full GPU offload, F16 KV, batch
2048, and microbatch 256.

```bash
GGML_CUDA_ALLREDUCE=nccl \
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
HSA_NO_SCRATCH_RECLAIM=1 \
./build/bin/llama-bench \
  -m /path/to/model.gguf \
  -ngl 99 \
  -sm tensor \
  -ts 1/1/1/1 \
  -fa on \
  -b 2048 \
  -ub 256 \
  -p 512 \
  -n 128 \
  -r 3
```

`llama-bench` requires `/` between tensor-split values. A comma-separated value
is interpreted as a parameter sweep rather than one four-GPU split.

| Model | pp512 | tg128 raw decode |
|---|---:|---:|
| Qwen3.6 35B-A3B Q4_K_M | **1,607.43 ± 18.64 t/s** | **63.17 ± 4.11 t/s** |
| Qwen3.5 122B-A10B Q4_K_M | **955.39 ± 8.65 t/s** | **40.69 ± 1.92 t/s** |

The standard tg128 result does **not** use MTP speculative decoding. End-to-end
`llama-server` generation with MTP is faster when draft acceptance is healthy.

## Content-sensitive MTP generation

The following short server tests generated exactly 1,024 tokens at temperature
0.6 with MTP enabled. They use real code and prose prompts rather than the
synthetic `llama-bench` workload.

| Model | Prompt | Decode | Draft acceptance | Wall time |
|---|---|---:|---:|---:|
| Qwen3.6 35B-A3B | Python implementation and tests | **76.42 t/s** | 65.86% | 14.39s |
| Qwen3.6 35B-A3B | Historical prose essay | **69.67 t/s** | 55.70% | 14.82s |
| Qwen3.5 122B-A10B | Python implementation and tests | **54.61 t/s** | 60.77% | 19.77s |
| Qwen3.5 122B-A10B | Historical prose essay | **57.75 t/s** | 68.36% | 17.92s |

MTP throughput follows draft acceptance more closely than content category. In
these samples the 35B model accepted more coding drafts, while the 122B model
accepted more prose drafts.

## RCCL impact

A separate deterministic 122B comparison measured:

| Four-GPU all-reduce | Decode |
|---|---:|
| Meta-backend butterfly | 50.72 t/s |
| RCCL | **55.61 t/s** |

RCCL improved 122B decode by approximately **9.6%**. An 8.8k prompt improved
from approximately 615 t/s to 750 t/s in the same comparison.

## HIP decode profile and partial TOP_K optimization

A `rocprofv3` kernel profile of a 64-token 35B MTP request identified the
current decode priorities:

| Kernel family | Approximate kernel time |
|---|---:|
| RCCL all-reduce | **46.9%** |
| Q4_K output/projection matvecs | **14.7%** |
| Quantized model and MoE matvecs | remaining majority |
| Sampling sort/selection | about 2% before optimization |
| Flash attention | below 1% for short-context decode |

The model performs exactly **80 all-reduces per raw decode step**: two 2,048
float (8 KiB) reductions for each of 40 layers. This makes small-collective
latency the largest remaining tensor-parallel optimization target.

Commit `b60551777` replaces HIP's full-vocabulary sort for `TOP_K` values up to
256 with `rocprim::topk_pairs`, followed by a compact 256-element block radix
sort. The feature is detected with `__has_include`; older ROCm releases without
`device_topk.hpp` retain the existing argsort fallback.

A fixed-seed 35B server A/B test measured:

| TOP_K implementation | Average | Median |
|---|---:|---:|
| Full-vocabulary argsort | 86.04 t/s | 86.51 t/s |
| rocPRIM partial TOP_K | **86.68 t/s** | **87.04 t/s** |

The end-to-end gain is approximately **0.6-0.8%**, consistent with the sampler's
profile share. Draft acceptance was identical. A separate 256-token greedy test
produced byte-identical output, and focused 248,320-vocabulary tests pass for
`k=20` and `k=256`.

Experiments with a custom four-GPU direct-P2P collective were not retained:
peer payload reads work, but reliable cross-root synchronization on these
RDNA2 cards cost more than RCCL. Forced RCCL Ring/Tree and LL/Simple protocols
also did not improve over RCCL's automatic selection. Future collective work
should therefore target fused/persistent small-message collectives or graph
architecture that removes/reduces the 80 host-submitted boundaries.

## Experimental sharded LM head

Set the following before model load to enable embedding-axis sharding of
validated Qwen3.5/Qwen3.6 35B-A3B and 122B-A10B main/MTP output heads:

```bash
export GGML_TP_SHARDED_OUTPUT=1
```

Each rank computes a full-vocabulary partial projection from one embedding
slice. A forced-FP32 all-reduce produces mirrored complete logits before scale,
LoRA, bias, or sampling. Existing CPU/backend sampler semantics remain intact.
Unsupported architectures, model sizes, tied heads, invalid quant-block splits,
and zero-width rotated splits retain mirrored placement.

### Decode results

| Workload | Mirrored | Sharded | Change |
|---|---:|---:|---:|
| 35B MTP, five-run mean | 86.59 t/s | **92.41 t/s** | **+6.72%** |
| 35B raw tg128 | 62.93 t/s | **65.61 t/s** | **+4.25%** |
| 122B MTP, five-run mean | 49.02 t/s | **65.51 t/s** | **+33.65% observed** |

The 122B mirrored process degraded substantially across its five runs while the
sharded process remained stable. The direction and large benefit are clear, but
the exact 33.65% figure includes thermal/run-order effects and should not be
interpreted as a topology-independent estimate.

Five paired 35B runs had identical draft/accepted counts in every pair. A
separate deterministic 64-token test produced the same output SHA-256 in both
modes.

### Prompt processing

Clean-reset, isolated process results:

| Prompt | Mirrored | Sharded | Change |
|---|---:|---:|---:|
| 64k (65,533 actual) | 1,361.57 t/s | 1,361.57 t/s | parity |
| 128k (131,071 actual) | **1,029.30 t/s** | 1,020.43 t/s | -0.86% |

Both modes completed 64k and 128k without truncation or GPU faults after a clean
GPU reset. The feature is decode-focused; a sub-1% 128k prefill regression is
the measured tradeoff on this system.

### Validation and limits

- Feature selection is immutable after model placement and exact-string gated.
- Output reductions are FP32 even above the generic RCCL BF16 crossover.
- Main and shared MTP heads use tensor identity and validated split rotation.
- Terminal raw-logit/CPU-verification graphs include an explicit synchronization
  alias so final partial logits cannot escape unreduced.
- Focused split-validation unit tests cover equal/rotated/skewed/zero-width and
  non-divisible layouts.
- The implementation remains experimental and opt-in. Generic butterfly is a
  correct slower fallback when RCCL does not service a reduction.

## Experimental vocabulary-parallel output

`GGML_TP_VOCAB_OUTPUT=1` requests the more aggressive output-head experiment.
Eligibility is determined once from model structure before weight placement;
there is no model architecture or size allowlist. It is mutually exclusive with
`GGML_TP_SHARDED_OUTPUT` and currently requires tensor split plus RCCL:

```bash
export GGML_TP_VOCAB_OUTPUT=1
export GGML_CUDA_ALLREDUCE=nccl
```

For the tested 27B model, the separate, untied output head is BF16. It is split
on its vocabulary axis; with the 248,320-token vocabulary and four equal ranks,
every GPU stores and computes 62,080 logits per output row. Each rank converts
its local logits to deterministic 64-bit keys, selects 256 candidates per row
with rocPRIM, and performs one compact RCCL all-gather. Rank 0 merges the 1,024
candidates for each row on the host. The existing CPU sampler then operates on
the exact global top-256.

The key ordering is descending logit followed by ascending global token ID.
NaNs are treated as negative infinity and signed zero is canonicalized to
positive zero. This makes shard-boundary ties deterministic. Synthetic tests
cover random logits, ties, all-equal inputs, infinities, NaNs, and signed zero
for `k=20/256` and one to four rows.

Two reverse-order five-repetition `llama-bench` comparisons measured:

| Output path | Raw tg128 |
|---|---:|
| Mirrored BF16 head | 29.14 and 29.11 t/s |
| Vocabulary-parallel compact head | **33.51 and 33.63 t/s** |

The retained raw gain is approximately **15-16%**. Greedy 64/512-token server
pairs produced identical responses. Reproduce the reverse-order comparison with:

```bash
MODEL=/path/to/model.gguf ./scripts/rdna2-vocab-ab.sh
```

The script preserves all CSV/stderr outputs under a timestamped directory in
`~/llama-jobs` (override with `OUT_DIR`).

A balanced six-run 512-token MTP comparison measured:

| MTP output path | Mean decode | Draft acceptance |
|---|---:|---:|
| Mirrored head | 37.77 t/s | 329 / 544 (60.48%) |
| Vocabulary-parallel compact head | **51.88 t/s** | 329 / 544 (60.48%) |

The MTP gain is approximately **37.3%**, with identical output, accepted/generated
counts, and mean draft length. A 6,010-token prompt completed at 770.9 prompt
tokens/s and 51.45 decode tokens/s. Four concurrent MTP slots also completed
without compact-selection or decode errors. A 6,017-token JSON-schema grammar
request used the dense fallback and completed at 769.3 prompt tokens/s and 45.5
decode tokens/s. Reproduce the balanced MTP test with:

```bash
MODEL=/path/to/model.gguf ./scripts/rdna2-vocab-mtp-ab.sh
```

At load time the feature activates only for a separate, untied, canonical 2-D
`output.weight` whose vocabulary axis matches the tokenizer, whose output layer
is assigned to the tensor-parallel meta device, whose slices are 128-aligned and
at least 256 entries wide, and which has no output bias. If any predicate fails,
the model logs the reason once and retains mirrored placement. This allows
additional architectures and sizes without claiming support based on model
names.

The fast compact path has these boundaries:

- finite CPU `top_k` in the range 1-256 must run before active probability samplers;
- backend sampler offload is unavailable; `--spec-draft-backend-sampling`
  automatically falls back to the compact CPU sampler;
- MTP is admitted only when its shared head is absent or aliases `model.output`;
  models with a distinct MTP head receive an explicit context error;
- no generic collective fallback: failure to obtain the RCCL compact candidate
  provider is an explicit decode error.

When grammar, reasoning budget, logit bias, active penalties, mirostat, infill,
adaptive-p, or a dense-logit API needs the complete vocabulary, the context
materializes the split logits from the GPUs on demand for the current output
rows. This preserves exact sampling and avoids dense `n_batch * n_vocab` host
buffers, but that request gives back some of the compact path's speed. The
common server sampler supports this fallback; direct low-level sampler users
should request dense logits explicitly.

## Notes and caveats

- Results are specific to four V620 cards, this PCIe topology, ROCm version,
  quantization, model build, and branch.
- Prompt processing used one slot to make 128k requests possible. With
  `--parallel 4`, non-unified contexts may divide the configured context among
  four slots.
- Standard `llama-bench` reports raw decode and does not exercise MTP.
- Server MTP results depend on prompt content, temperature, and draft
  acceptance.
- On AMD recurrent/hybrid models, restored prompt state is never sent through
  tile flash attention because of the page fault tracked in upstream issue
  #20176. If the missing suffix is at most one eighth of the full prompt, cached
  HIP graphs are invalidated and vector flash attention evaluates that suffix.
  Larger suffixes skip checkpoint restore and use a clean tile-FA reprocess.
  Generation returns to normal kernel selection in either case. This favors
  stability while selecting the cheaper known-safe prompt path. Checkpoints
  remain conservatively disabled for other AMD model types.
- Prompt caching was disabled for scaling tests. Production multi-turn behavior
  can differ substantially when prefixes are reused.
- After any ROCm illegal-memory fault, reset the GPUs or reboot before trusting
  later measurements.