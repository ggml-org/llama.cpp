# DeepSeek V4 Flash local performance notes

Last updated: 2026-08-01

This document records the local setup, evidence, current changes, and next tuning ideas for DeepSeek V4 Flash. It is intended to make later profiling sessions reproducible without reconstructing the investigation from chat history.

## Scope and baseline

This is private local tuning work on branch `kuba/ds4-r9700-tuning`. The source baseline when the first optimization was added was:

```text
8f5ab832ca7d8a7b4f23687693fb8b0ecbc227e7
cohere2 moe template parser: enforce JSON schema for text responses if a response schema is provided (#26018)
```

Model:

```text
/fstorage/models/DSV4F-UD-IQ_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf
```

The model is a three-shard Unsloth dynamic IQ1_S GGUF. The performance issue discussed here is the model's LID attention `TOP_K` operation, not the unrelated sampler `--top-k` setting.

Hardware visible to ROCm:

- `ROCm0`: AMD Radeon AI PRO R9700, gfx1201, 32624 MiB
- `ROCm1`: AMD Radeon AI PRO R9700, gfx1201, 32624 MiB
- `ROCm2`: AMD Radeon AI PRO R9700, gfx1201, 32624 MiB
- `ROCm3`: Ryzen 9 9950X iGPU, gfx1036, shared system memory
- CPU: Ryzen 9 9950X, 16 cores / 32 threads

The display was moved to the iGPU and Hyprland was restarted, which released the residual dGPU display allocation. GPU 0 has the fastest CPU connection according to the machine topology. The tested HIP order is `ROCm0,ROCm1,ROCm2`.

The existing private server launcher was moved to `../llama.cpp.scripts/run-d4f.sh`. It is currently a reference rather than a directly runnable script because its `ROOT` calculation now resolves to the sibling scripts directory. Its important HIP settings were:

```text
--device ROCm0,ROCm1,ROCm2
--split-mode layer
-fa on
-b 4096
-ub 512
--cache-type-k f16
--cache-type-v f16
--fit on
--fit-target 2048
--fit-ctx 4096
--threads -1
```

It targeted a maximum HIP context of 650752 with fitting enabled. That very large context has not been validated after the code change in this note. Small contexts should continue to be used while measuring decode behavior.

## Builds

The active HIP build was configured as Release with these relevant options:

```text
GGML_HIP=ON
GGML_CUDA_GRAPHS=ON
GGML_HIP_GRAPHS=ON
GGML_HIP_MMQ_MFMA=ON
GGML_HIP_NO_VMM=ON
GGML_HIP_ROCWMMA_FATTN=OFF
```

No omitted build flag explained the original CPU load. The decisive limitation was an explicit backend capability check in the source: HIP only accepted `GGML_OP_TOP_K` when the input row had at most 1024 columns.

Vulkan was previously usable with the three dGPUs and provided an important comparison because it already has GPU `TOP_K` pipelines. The archived launcher used `Vulkan1,Vulkan2,Vulkan0`, but HIP is the current optimization target. Future comparisons must use identical prompt length, generated token count, cache types, batch sizes, and GPU split.

## Original symptom

When used through the Pi editor and local server, observed performance was approximately:

```text
prompt processing: 220 tokens/s
token generation:    11 tokens/s
```

During generation the CPU was saturated while aggregate dGPU utilization was often around 20 percent. Other models on the same machine did not show this behavior.

DeepSeek V4 constructs an attention indexer score whose first dimension grows with the live LID context. Each relevant layer then calls `ggml_top_k()` in `src/models/deepseek4.cpp::build_lid_top_k()`. Once that first dimension exceeded 1024, the HIP backend reported the operation as unsupported. The scheduler therefore transferred the operation to CPU. This caused device/host synchronization and transfers once per affected layer during every decode step.

This also explains why the problem became worse with context length and why adding GPU capacity or changing ordinary launch flags did not solve it.

## Pre-change profiler evidence

The raw rocprof output is retained in:

```text
/tmp/dsv4-rocp-2k
/tmp/dsv4-rocp
```

The profiled commands used one sequence, full layer offload, three dGPUs, flash attention, f16 K/V cache, batch 4096, ubatch 512, four generation threads, and 32 prompt threads.

At a 2K prompt and 256 generated tokens:

```text
profiled generation rate:       17.59 tokens/s
generation wall time:           14.552 s
sum of traced GPU kernel time:    8.716 s
unaccounted host/sync interval:   5.836 s
GPU lightning-indexer time:       0.146 s
```

At an 8K prompt and 256 generated tokens:

```text
profiled generation rate:       15.62 tokens/s
generation wall time:           16.394 s
sum of traced GPU kernel time:    9.084 s
unaccounted host/sync interval:   7.310 s
GPU lightning-indexer time:       0.172 s
```

There was no wide GPU `TOP_K` kernel in the 8K trace. About 1.47 s of the 1.84 s increase from the 2K run to the 8K run appeared outside traced GPU kernels. This made the host fallback the highest-confidence first target.

The sum of kernel durations is not an exact critical-path measurement on multiple GPUs because kernels can overlap. The comparison is still useful because the growing wall-time gap, CPU saturation, backend capability check, and absence of a GPU top-k dispatch all point to the same cause.

## First optimization: HIP wide TOP_K

Modified files:

```text
ggml/src/ggml-cuda/top-k.cu
ggml/src/ggml-cuda/ggml-cuda.cu
```

The implementation now behaves as follows:

1. HIP rows with at most 1024 columns keep the existing shared-memory bitonic path.
2. Wider HIP rows initialize row-local indices on the device.
3. hipCUB `DeviceSegmentedRadixSort::SortPairsDescending` sorts the rows on the active HIP stream.
4. Only the first `k` sorted indices from each row are copied to the destination.
5. Rows are processed in chunks targeting approximately 64 MiB for keys and the two index buffers, excluding rocPRIM temporary storage.
6. `GGML_OP_TOP_K` is now advertised as supported for all widths on HIP. Wide `GGML_OP_ARGSORT` remains unchanged and can still fall back to CPU.

This intentionally reuses ROCm's hipCUB/rocPRIM infrastructure. It does not change model graph semantics, the NVIDIA path, or non-HIP backends.

The current implementation is a full radix sort even though only the largest `k` indices are needed. It is a correctness-first way to remove the expensive CPU boundary, not the theoretical endpoint for top-k performance.

## Post-change results

Direct `llama-batched-bench` measurements after the change:

| Prompt | Generated | Context allocation | Prompt rate | Generation rate |
|---:|---:|---:|---:|---:|
| 2048 | 128 | 4096 | 420.46 tokens/s | 23.28 tokens/s |
| 8192 | 128 | 16384 | 381.68 tokens/s | 22.83 tokens/s |

Representative 8K command:

```sh
build-hip/bin/llama-batched-bench \
    -m /fstorage/models/DSV4F-UD-IQ_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf \
    --device ROCm0,ROCm1,ROCm2 \
    -ngl all \
    --split-mode layer \
    -c 16384 \
    -fa on \
    -b 4096 \
    -ub 512 \
    -ctk f16 \
    -ctv f16 \
    -t 4 \
    -tb 32 \
    --poll 0 \
    -npp 8192 \
    -ntg 128 \
    -npl 1
```

The old figures were collected under rocprof with 256 generated tokens, while the new figures are unprofiled with 128 generated tokens. They are therefore not a strict before/after benchmark. The much flatter 2K-to-8K decode rate and the increase to roughly 23 tokens/s are nevertheless consistent with removal of the context-dependent CPU fallback. A controlled profiled A/B should be done before attributing an exact percentage improvement.

## Validation completed

- HIP Release build completed for `test-backend-ops`, `llama-batched-bench`, and `llama-bench`.
- Focused GPU correctness tests passed at the 1024/1025 dispatch boundary and at 2K, 8K, and 16K widths.
- Tests included one and multiple rows, several `k` values, tied values such as repeated masked scores, and HIP graph execution.
- The final focused runs passed 56 cases on an R9700.
- `git diff --check` passed.
- NVIDIA, Vulkan, and CPU code paths were not modified.

Useful focused test commands:

```sh
build-hip/bin/test-backend-ops test -o TOP_K -b ROCm0 -p 'ne=\[(1024|1025|8192|16384),'
build-hip/bin/test-backend-ops test -o TOP_K -b ROCm0 -p 'ne=\[8203,'
```

## RDNA4 quantized matrix tuning

The next experiment tunes the quantized matrix kernels for this model and machine. The committed comparison build is `build-hip-sparse-attn`; the isolated experimental build is `build-hip-quant-tune`. The current source changes are not committed.

The three MoE weight shapes that dominate this model are:

| Type | Operation | Experts | Used | Output rows | K | Decode N | PP N |
|---|---|---:|---:|---:|---:|---:|---:|
| IQ1_S | gate/up | 256 | 6 | 2048 | 4096 | 1 | 128 |
| IQ2_XXS | gate/up | 256 | 6 | 2048 | 4096 | 1 | 128 |
| IQ3_XXS | down | 256 | 6 | 4096 | 2048 | 1 | 128 |

For decode, `ggml/src/ggml-cuda/mmvq.cu` now uses two wavefronts for IQ1_S on RDNA4. IQ2_XXS and IQ3_XXS retain one. The exact-shape microbench results were:

| Type | Original | Candidate | Result | Change |
|---|---:|---:|---:|---:|
| IQ1_S | 1 wave, 31.62 us | 2 waves, 27.37 us | retained | -13.4% |
| IQ1_S | 1 wave, 31.62 us | 4 waves, 30.24 us | rejected | -4.4% |
| IQ2_XXS | 1 wave, 57.35 us | 2 waves, 57.30 us | rejected as neutral | -0.1% |
| IQ3_XXS | 1 wave, 59.32 us | 2 waves, 62.98 us | rejected | +6.2% |

Q8_0 previously shared the RDNA4 eight-wavefront setting used by the other simple dot-product types. DSV4 has enough Q8_0 output-row parallelism that one wavefront per block is faster in aggregate. Representative timings in microseconds were:

| M | K | Replicas | 8 waves | 2 waves | 1 wave |
|---:|---:|---:|---:|---:|---:|
| 32768 | 1024 | 1 | 66.03 | 34.33 | 29.75 |
| 8192 | 4096 | 1 | 31.68 | 34.13 | 31.58 |
| 4096 | 2048 | 1 | 14.32 | 22.29 | 20.30 |
| 1024 | 4096 | 8 | 27.51 | 31.39 | 28.29 |
| 1024 | 4096 | 1 | 9.75 | 17.81 | 10.82 |
| 512 | 4096 | 1 | 19.34 | 8.05 | 9.42 |
| 256 | 4096 | 1 | 17.83 | 7.35 | 8.92 |
| 4096 | 8192 | 1 | 28.16 | 31.49 | 26.24 |

One wavefront is not the fastest for every shape, but weighting these ratios by dispatch counts and durations from the saved 8K decode trace favors it. A future shape-dependent Q8_0 dispatcher could combine the best widths, but the current compile-time MMVQ specialization does not include matrix dimensions and the added complexity is not yet justified.

For prompt processing, `ggml/src/ggml-cuda/mmq-config-rdna4.cuh` specializes only the non-fallback J=128 cases for IQ1_S, IQ2_XXS, and IQ3_XXS. The retained configuration is 512 threads, I=256, and launch-bound occupancy 1. Exact DSV4 MoE timings were:

| Configuration | IQ1_S | IQ2_XXS | IQ3_XXS |
|---|---:|---:|---:|
| 128 threads, I=64, occupancy 2 | 6.641 ms | 6.883 ms | 6.968 ms |
| 256 threads, I=128, occupancy 2, original | 6.454 ms | 6.678 ms | 6.873 ms |
| 512 threads, I=256, occupancy 2 | 3.982 ms | 4.669 ms | 4.942 ms |
| 512 threads, I=256, occupancy 1, retained | 3.899 ms | 4.515 ms | 4.857 ms |

The focused performance harness spends most of its wall time allocating and initializing all 256 expert matrices on the CPU. This explains the high CPU and low sampled GPU utilization visible during these tests. Its reported `us/run` timing excludes initialization: it surrounds the synchronous `ggml_backend_graph_compute()` loop and runs that loop for at least one second. The loaded-model A/B below is the authoritative whole-graph result.

Three repeated 2K prompt plus 128-token decode runs produced:

| Build | Average PP | Average TG |
|---|---:|---:|
| committed comparison | 424.85 tokens/s | 23.28 tokens/s |
| quant tuned | 439.25 tokens/s | 24.80 tokens/s |
| change | +3.39% | +6.51% |

A single longer-context pair using an 8K prompt, 256 generated tokens, and a 16K allocation produced:

| Build | PP | TG |
|---|---:|---:|
| committed comparison | 374.39 tokens/s | 22.86 tokens/s |
| quant tuned | 384.14 tokens/s | 24.34 tokens/s |
| change | +2.60% | +6.47% |

Small CPU-reference correctness cases passed for all three IQ types on both the N=1 MMVQ path and the N=128 MMQ path. The Q8_0 N=1 path also passed. The matching server is available at `build-hip-quant-tune/bin/llama-server`.

## Recommended next measurements

Do these before choosing the next implementation target:

1. Repeat the exact 2K and 8K rocprof runs with 256 generated tokens on the patched build. Compare wall time, summed kernel time, CPU utilization, and the new rocPRIM kernels against the saved traces.
2. Run an unprofiled controlled A/B if an old binary is available. Keep model, context allocation, prompt, generated token count, device order, graph setting, and clocks identical.
3. Measure top-k shapes directly for decode and prompt processing. Important cases are `ncols` 2048, 8192, 16384, and larger; `k` near the model's LID top-k; `nrows=1` for decode; and realistic multi-row prompt batches.
4. Recheck the Pi/server path using a fixed prompt and a sufficiently long generation. Server scheduling, prompt caching, sampling, and client timing make its reported rates less controlled than `llama-batched-bench`.
5. Record GPU clocks, power state, and per-device utilization during the benchmark so topology or throttling does not get mistaken for a kernel regression.

## Candidate follow-up optimizations

Priorities should be revised after the patched profile.

### 1. Replace full sort with selection

The hipCUB path sorts every key and writes a full keys buffer plus two full index buffers. A dedicated large top-k implementation could avoid most of that work. Possible designs include:

- block-local candidate selection followed by one or more merge stages;
- radix selection to find a cutoff followed by compaction of the winning indices;
- a specialized path for the model's common `k` and decode row count while retaining hipCUB as the general fallback.

Any selection implementation must preserve valid behavior for ties, `-INFINITY` mask values, arbitrary row widths, multiple rows, stream capture, and all tested `k` values. It should be justified by a microbenchmark before adding complexity.

### 2. Reduce temporary work in the hipCUB path

Even without a new algorithm, investigate:

- avoiding repeated generation of offsets when the row geometry is stable;
- reducing or reusing temporary index/key storage through the existing pool;
- a cheaper single-row decode path using `DeviceRadixSort` rather than segmented sort;
- whether sorting fewer radix bits is valid for the actual non-negative score representation plus `-INFINITY` masking;
- chunk sizing for prompt processing and its effect on peak VRAM.

Do not assume these matter. Profile the current rocPRIM kernels first.

### 3. Avoid constructing a dense top-k mask

`src/models/deepseek4.cpp::build_top_k_mask()` currently fills a dense mask with `-INFINITY`, writes zeros at selected rows, reshapes it, and adds the original mask. At long context this creates additional context-sized memory traffic per layer.

A more ambitious path would pass selected indices directly into an attention implementation or fuse mask construction with the consumer. This could be more valuable than optimizing radix sort further, but it crosses model graph and attention backend boundaries and should only be attempted after profiling confirms the mask operations are dominant.

### 4. Fuse lightning indexer and top-k

The lightning indexer writes all scores to global memory, after which top-k reads them again. A fused kernel could keep partial candidates closer to the producer and reduce launches and memory traffic. This is substantially more invasive, especially for prompt batches and multi-GPU graph scheduling. Treat it as a later option, not the default next patch.

### 5. Revisit multi-GPU placement

With the CPU fallback removed, device order and inter-device transfers may become more visible. Check which GPU owns the first and last layers, where logits and graph inputs reside, and whether the fastest CPU-connected GPU should remain first. Fully offloaded decode should be less sensitive to CPU connectivity than the old fallback was, so repeat rather than reuse the pre-fix topology conclusion.

Compare layer and tensor split only with the model fully resident. Tensor split can add synchronization and transfer costs that are especially visible at batch size one.

### 6. Recompare HIP and Vulkan

Vulkan already kept wide top-k on GPU, while HIP previously did not. Re-run identical short benchmarks after this patch. HIP now has a fairer comparison and is expected to benefit from its optimized model-specific kernels, but the result should be measured rather than assumed.

### 7. Retune runtime parameters after code profiling

Once the next kernel bottleneck is known, sweep only a small set of likely-impactful settings:

- `-ub 256`, `512`, and `1024` for prompt processing;
- generation threads after the host fallback is gone, including `-t 1`, `2`, and `4`;
- HIP graphs on and off if rocPRIM capture or graph resets appear expensive;
- small context allocations for decode sanity checks before testing the maximum f16 cache target;
- GPU order and explicit tensor split only if traces show transfer imbalance.

Avoid broad parameter sweeps until the patched trace identifies a sensitivity. The previous primary limitation was architectural backend fallback, not ordinary launch tuning.

## Current interpretation

The poor original result was not an inherent 11 tokens/s limit of DeepSeek V4 Flash on three R9700s. The main demonstrated problem was incomplete HIP support for the model's context-width top-k operation. Keeping that operation on-device raised the controlled decode benchmark to roughly 23 tokens/s and largely removed the decline between 2K and 8K context.

There is still likely room for improvement. The next most efficient step is a post-change trace, followed by either a partial-selection top-k kernel or dense-mask/attention work depending on where the new critical path appears.

## Sparse indexed FlashAttention experiment

An experimental build is in `build-hip-sparse-attn`. The baseline remains in `build-hip`. Both target `gfx1201` with HIP, FlashAttention, HIP graphs, and the MFMA MMQ path enabled.

The ROCm backend compiles much of its code from `ggml/src/ggml-cuda` with the HIP compiler. Changes in that directory can therefore be HIP-specific even though the path contains `cuda`.

The experiment passes the DSV4 lightning-indexer indices to FlashAttention and reads only selected K/V rows. DSV4 uses 512 CSA indices plus 256 padded raw/SWA indices, for 768 total indices. The upstream draft only reached its sparse MMA path on NVIDIA for this shape. RDNA4 normally selects the dense tile kernel for 512-wide attention, so the experiment also:

- selects the MMA/WMMA implementation only when sparse indices are attached to a 512-wide attention operation;
- uses a 16-column RDNA4 specialization because AMD WMMA does not emit device code for the draft's 8-column specialization;
- enables device code for the sparse 512-wide specialization while leaving dense RDNA4 dispatch unchanged.

Focused ROCm0 correctness tests passed against the CPU reference for 8K and 32K KV with the model's 768-index shape, including one-token and two-token query batches.

### Attention microbenchmarks

Decode, 128 GQA heads, 768 selected indices:

| KV rows | sparse us | dense us | operation speedup |
|--------:|----------:|---------:|------------------:|
| 4K | 101.36 | 137.07 | 1.35x |
| 8K | 101.03 | 259.57 | 2.57x |
| 16K | 101.72 | 508.79 | 5.00x |
| 32K | 102.78 | 1007.61 | 9.80x |
| 64K | 104.69 | 2015.48 | 19.25x |
| 128K | 107.79 | 4019.92 | 37.29x |

Prefill with 512 query tokens:

| KV rows | sparse ms | dense ms | operation speedup |
|--------:|----------:|---------:|------------------:|
| 8K | 31.814 | 77.438 | 2.43x |
| 16K | 33.900 | 155.691 | 4.59x |
| 32K | 39.078 | 308.532 | 7.89x |

These are attention-operation measurements, not whole-model token rates. The displayed sparse TFLOPS from `test-backend-ops` are virtual because its FLOP counter still uses the full KV length.

### Whole-model A/B

At 8192 prompt tokens and 256 generated tokens, sparse attention was below its activation point and was neutral:

| build | PP t/s | TG t/s |
|-------|-------:|-------:|
| baseline | 380.40 | 22.87 |
| sparse | 375.17 | 22.84 |

At 32768 prompt tokens and 128 generated tokens:

| build | PP t/s | TG t/s |
|-------|-------:|-------:|
| baseline | 269.78 | 20.79 |
| sparse | 270.30 | 20.98 |

DSV4 CSA compresses tokens at 4:1, and the current graph enables sparse indices at 8192 compressed rows. It therefore begins at roughly 32768 raw tokens. At that boundary the attention operation is much faster, but it is still a small fraction of total model time and only becomes active near the end of prompt processing. The end-to-end improvement is about 0.9 percent for TG and neutral for PP at this context. Larger contexts should increase the fraction of time saved, but this needs a controlled full-model measurement.

## Follow-up triage: split mode, sparse threshold, and TOP_K

ROCm row splitting cannot currently be tested because the HIP device does not expose split buffers. Model loading fails with `device ROCm0 does not support split buffers`. Experimental tensor splitting is also unavailable because `LLM_ARCH_DEEPSEEK4` is explicitly excluded by `llm_arch_supports_sm_tensor()`. Layer splitting is therefore the only working three-GPU mode without implementing additional backend and model support.

Additional sparse-attention measurements located the crossover below the original 4K test point.

Decode with one query token:

| KV rows | forced sparse us | dense us |
|--------:|-----------------:|---------:|
| 1K | 101.03 | 46.73 |
| 1.5K | 98.70 | 60.73 |
| 2K | 98.13 | 77.78 |
| 3K | 98.03 | 106.62 |
| 4K | 99.72 | 138.43 |

Prefill with 512 query tokens:

| KV rows | forced sparse ms | dense ms |
|--------:|-----------------:|---------:|
| 2K | 29.546 | 19.163 |
| 3K | 30.677 | 28.735 |
| 4K | 31.040 | 38.302 |

Both modes cross over between 3K and 4K KV rows. Lowering the DSV4 graph threshold from 8192 to 4096 compressed rows was tested at an 18432-token prompt and then reverted. The baseline measured 326.45 PP and 21.91 TG tokens/s; the lower threshold measured 328.97 PP and 21.63 TG tokens/s. The differences were small and mixed, so earlier activation is not a significant optimization at this context.

The below-threshold tests also found that the RDNA4 dispatcher selected the sparse-only 512-wide MMA specialization whenever a TOP_K tensor existed, even when the shared sparse heuristic declined to use it. The working tree now applies the same threshold at dispatch time. Below the crossover, TOP_K attention correctly remains on the dense tile kernel instead of reaching a no-device-code specialization.

Post-TOP_K traces are in:

```text
/tmp/dsv4-rocp-topk-post
/tmp/dsv4-rocp-topk-post-8k
```

At a 2K raw prompt, compressed width remains below 1024 and the original bitonic TOP_K kernel accounts for 0.56 percent of GPU kernel time. At an 8K raw prompt, the wide rocPRIM path costs about 0.67 ms per generated token, approximately 1.2 percent of profiled TG wall time. Dense mask fill, set-rows, and add kernels together are well below one percent at this context.

Direct one-row, top-512 measurements for the wide HIP path:

| score columns | TOP_K us |
|--------------:|---------:|
| 2K | 32.79 |
| 8K | 89.00 |
| 32K | 196.79 |
| 128K | 735.27 |

There are about 21 LID TOP_K calls per generated token. A specialized selection kernel is therefore unlikely to materially improve ordinary 8K-32K operation, but can become important near the maximum context. The post-change 8K trace instead confirms that quantized matrix kernels remain the best context-independent target. Dense K/mask concatenation also deserves attention at very long context because it still scales with the complete KV length even when FlashAttention reads only sparse indices.
