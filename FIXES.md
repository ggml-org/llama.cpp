# RDNA2 ROCm tensor-split fixes

This branch contains the changes used to run hybrid Qwen MoE/MTP models on
multiple RDNA2 GPUs with the llama.cpp HIP backend.

- **Repository:** <https://github.com/edwinbrowwn/llama.cpp-rdna2>
- **Branch:** `exp-gpu-sampling`
- **Primary test system:** 4 × AMD Radeon Pro V620 32 GB (`gfx1030`), ROCm 7.x
- **Backend:** HIP/ROCm with tensor split and RCCL

The branch is experimental. The changes are useful beyond the V620, but the
current testing and tuning focus on RDNA2/gfx1030.

## Included changes

### Recurrent and prompt-state fixes

This branch includes the work from:

- [PR #24785](https://github.com/ggml-org/llama.cpp/pull/24785): recurrent
  state shrink/expand support around prompt-cache operations.
- [PR #24891](https://github.com/ggml-org/llama.cpp/pull/24891): prevents
  `n_past` overwrite and reduces `seq_rm` crash risk.

It also adds a local multi-slot guard: recurrent-state shrink/expand is skipped
when `n_parallel_user > 1`. Shrinking to one cell while several slots are live
can discard the state for sequence IDs greater than zero.

### Tensor-split GPU sampling

Upstream rejects backend sampling whenever `LLAMA_SPLIT_MODE_TENSOR` is used.
This branch removes that blanket rejection and adds the pieces needed for the
HIP/meta backend to run the sampler safely:

- Treat zero-element graph inputs as trivially mirrored. This avoids an
  `UNKNOWN` split-state assertion while reserving MTP sampling graphs.
- Enable hipCUB as the HIP equivalent of CUB for `TOP_K`, `ARGSORT`, `SUM`,
  `MEAN`, and `CUMSUM` paths.
- Remove the 1024-column HIP `TOP_K` limit when hipCUB is available. This is
  required by Qwen's 248,320-token vocabulary.
- When rocPRIM `device_topk.hpp` is available, select only the requested
  candidates and sort that compact set instead of sorting the entire
  vocabulary. Older ROCm releases retain the full argsort fallback.
- Mirror the output projection and output bias across tensor-split devices.
  Each GPU therefore has complete logits and can execute sampling locally,
  instead of trying to sample a vocabulary shard.

Mirroring the output head uses more VRAM and repeats the output projection on
each GPU, but avoids CPU sampling and the associated device/host round trips.

### Embedding-sharded output projection

An opt-in `GGML_TP_SHARDED_OUTPUT=1` path shards validated Qwen 35B/122B main
and MTP LM heads along the embedding dimension. Each device computes
full-vocabulary partial logits, followed by an FP32 all-reduce before unchanged
sampling. This avoids four redundant full LM-head projections while preserving
full-logit semantics. See [`docs/rdna2-sharded-output-plan.md`](docs/rdna2-sharded-output-plan.md)
and [`README-RDNA2.md`](README-RDNA2.md) for design, validation, and benchmark
results.

### Four-GPU RCCL collectives

llama.cpp's internal fast all-reduce currently supports only two devices in
this configuration. Without RCCL, four GPUs fall back to the generic
meta-backend butterfly implementation.

This branch includes a reproducible RCCL build helper and is intended to run
with:

```bash
GGML_CUDA_ALLREDUCE=nccl
```

The variable retains the CUDA/NCCL name because llama.cpp shares this interface
between CUDA/NCCL and HIP/RCCL.

## Build

### Requirements

- Linux
- ROCm 7.x with HIP development files
- RCCL headers and `librccl.so`
- CMake and a C++ compiler

The helper defaults to:

```text
ROCm:   /opt/rocm/core-7.14
Target: gfx1030
Build:  ./build
```

Clone and build:

```bash
git clone --branch exp-gpu-sampling \
  https://github.com/edwinbrowwn/llama.cpp-rdna2.git
cd llama.cpp-rdna2
./scripts/build-rdna2-rocm.sh
```

Override the defaults when needed:

```bash
ROCM_PATH=/opt/rocm TARGET_ARCH=gfx1030 \
  ./scripts/build-rdna2-rocm.sh
```

The helper configures these important options:

```text
-DGGML_HIP=ON
-DGGML_HIP_RCCL=ON
-DGGML_HIP_GRAPHS=ON
-DGGML_HIP_NO_VMM=ON
-DGGML_NATIVE=ON
-DAMDGPU_TARGETS=gfx1030
-DCMAKE_HIP_ARCHITECTURES=gfx1030
-DLLAMA_BUILD_SERVER=ON
-DLLAMA_BUILD_TESTS=OFF
-DCMAKE_BUILD_TYPE=Release
```

After building, the helper verifies that `libggml-hip.so` is linked to RCCL.
This can also be checked manually:

```bash
ldd ./build/bin/libggml-hip.so.0 | grep rccl
```

## Run

A four-GPU tensor-split launch should start with:

```bash
GGML_CUDA_ALLREDUCE=nccl \
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
HSA_NO_SCRATCH_RECLAIM=1 \
GGML_HIP_GRAPHS=1 \
./build/bin/llama-server \
  -m /path/to/model.gguf \
  -ngl all \
  --split-mode tensor \
  --tensor-split 1,1,1,1 \
  --flash-attn on \
  --ctx-size 262144 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --batch-size 2048 \
  --ubatch-size 256 \
  --parallel 4 \
  --spec-type draft-mtp \
  --spec-draft-ngl all \
  --spec-draft-n-max 3 \
  --spec-draft-type-k f16 \
  --spec-draft-type-v f16 \
  --spec-draft-p-min 0.0 \
  --spec-draft-p-split 0.10 \
  --spec-draft-backend-sampling \
  --jinja \
  --metrics \
  --host 0.0.0.0 \
  --port 8080
```

For a native `gfx1030` runtime, `HSA_OVERRIDE_GFX_VERSION` may not be necessary.
It is retained in the tested V620 configuration. `GGML_HIP_GRAPHS=1` is also
specified at runtime even though graph support is enabled at build time.

Successful RCCL startup should not print:

```text
internal AllReduce init failed
falling back to meta-backend butterfly
```

### MTP settings

`--spec-draft-n-max 3` performed better than `4` on the tested Qwen3.6
35B-A3B workload. Four draft tokens reduced effective throughput when draft
acceptance fell. The best value remains model- and prompt-dependent.

### Optional stability settings

ROCm context-checkpoint restoration has caused illegal-memory faults on some
hybrid/recurrent workloads. For maximum reliability, disable context
checkpoints and server prompt caching:

```bash
--ctx-checkpoints 0 \
--cache-ram 0 \
--no-cache-idle-slots
```

This does not reduce ordinary decode speed, but repeated prefixes and multi-turn
requests may require additional prompt evaluation.

After a ROCm illegal-memory fault, stop the server and reset the affected GPUs
or reboot before drawing conclusions from later tests. GPU queues can remain
in a poisoned state and cause unrelated subsequent launches to fail.

## Tested performance

Results vary with model, prompt, sampling acceptance, clocks, and PCIe topology.
These numbers are included as reference points, not guarantees.

### Qwen3.6 35B-A3B MTP, Q4_K_M, 4 × V620

- Original CPU-sampling/tensor-split setup: approximately **64–68 t/s**.
- Native tensor-split GPU sampling: typically **72–79 t/s** on short coding
  generations.
- 8,192-token generation: approximately **65.9 t/s**.
- Four concurrent 2,048-token requests: approximately **111.9 aggregate t/s**.
- Fresh 35k-token prefill after a clean GPU reset: approximately **1,094 t/s**.
- Fresh 60k-token prefill: approximately **995 t/s**.

### Qwen3.5 122B-A10B MTP, Q4_K_M, 4 × V620

Deterministic 1,024-token decode comparison:

| All-reduce path | Average decode |
|---|---:|
| Meta-backend butterfly | 50.72 t/s |
| RCCL | **55.61 t/s** |

RCCL improved decode by approximately **9.6%**.

An 8.8k-token prompt improved from approximately **615 t/s** with butterfly to
**750 t/s** with RCCL, an improvement of about **21.9%**.

## Limitations

- Vulkan tensor split is not supported by the current meta-backend multi-buffer
  implementation. Use HIP/ROCm for tensor split. Vulkan can use layer split,
  but it was slower in this setup.
- RDNA2 does not use the RDNA3/4 WMMA flash-attention path. Do not assume
  `GGML_HIP_ROCWMMA_FATTN` improves V620/gfx1030.
- The mirrored output head is a tensor-split sampling optimization and has a
  VRAM/redundant-compute cost.
- RCCL performance depends heavily on PCIe topology. Benchmark it against the
  butterfly fallback on different systems.

## Relevant commits

```text
b60551777 hip: use rocPRIM partial top-k when available
0f6b98c2c build: add reproducible RDNA2 ROCm RCCL helper
05c471731 hip: enable tensor-split backend sampling with hipCUB
edc691711 local fix: skip recurrent shrink for prompt cache when n_parallel > 1
25d3ba717 Merge branch 'pr-24891'
29bfde3db Merge branch 'pr-24785'
```

The fork keeps `origin` available for upstream llama.cpp and uses a separate
`fork` remote for this branch.