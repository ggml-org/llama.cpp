# Qwen NVFP4 prefill path

## Supported graph

The matcher accepts the routed branch used by Qwen3.6-35B-A3B-NVFP4:

```text
gate MUL_MAT_ID -> gate scale
up   MUL_MAT_ID -> up scale
                  -> SwiGLU
                  -> down MUL_MAT_ID -> down scale
                  -> routing weight -> eight route views -> reduction
```

The strict shape is hidden 2048, expert size 512, 256 experts, top-8, and at
least 256 tokens. Gate, up, and down must be separate NVFP4 tensors on the same
CUDA device. Stored per-expert weight scales are required. The shared expert is
outside the matched range and consumes the routed result normally.

Graphs with activation input scales, a fused gate/up tensor, a different GLU,
short batches, or incompatible layouts use the existing graph.

## CUDA call chain

```text
ggml_cuda_try_fuse
  ggml_cuda_moe_cutlass_nvfp4_match
  ggml_cuda_moe_cutlass_nvfp4
    ggml_cuda_moe_repack_weight_pair
    ggml_cuda_moe_repack_weight(preserve_source=true)
    moe_cutlass_stage_routes
    ggml_cuda_launch_mm_ids_prefix
    moe_cutlass_quantize_nvfp4_broadcast_cta
    ggml_cuda_moe_cutlass_gemm(W13)
    moe_cutlass_nvfp4_w13_epilogue
    ggml_cuda_moe_cutlass_gemm(W2)
    moe_cutlass_nvfp4_w2_finalize
```

The prefix scheduler produces both route-to-row and row-to-route mappings.
W13 and W2 share those mappings. A1 quantization broadcasts one quantized token
to its eight expert-sorted rows. The W13 epilogue applies the two global weight
scales, standard SwiGLU, and A2 quantization. The final kernel applies the down
scale, routing weight, and reduction in graph order.

## Storage lifetime

Persistent weight entries belong to `ggml_backend_cuda_context`. Their keys
contain the source tensor, data pointer, backend buffer, and buffer generation.
The paired W13 entry and the W2 copy own their values and scale planes. Ready
events order first use after repacking, and last-use events record consumers.
These preserved entries are not eligible for generic repack cache eviction.
The context destructor waits for both event classes before destroying the
captured graphs, events, and owned storage in that order.

The W2 cache uses `preserve_source=true`. It never overwrites the canonical
GGUF tensor, so native MMQ and MMVQ remain valid after a CUTLASS prefill. The
paired W13 cache also leaves gate and up unchanged.

Expert maps, activations, CUTLASS metadata, workspaces, and BF16 intermediates
use `ggml_cuda_pool_alloc`. These objects do not escape the fused call. Their
destructors return allocations in reverse order, matching the CUDA pool's LIFO
contract. Every producer and consumer uses `ctx.stream()`, so reuse is ordered
after the last kernel that references each allocation.

## CUDA graphs

The first direct evaluations populate the persistent weight cache. Capture then
uses existing weight pointers and stable CUDA pool addresses. CUTLASS arguments
copy device pointers into kernel parameters during `initialize`; the local
adapter object does not outlive the call. Metadata and workspace device storage
remain valid until the GEMM launch has been enqueued on the same stream.
The graph compatibility scan treats a fully matched prefill block as one fused
operation, so the original large-batch `MUL_MAT_ID` synchronization rule does
not disable capture. Any unmatched block still uses that rule.

The relevant API contracts are documented in the
[CUDA Graph programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs),
the CUDA Runtime
[stream-ordered allocator](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html)
and [event](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html) references,
the pinned CUTLASS
[`GemmUniversalAdapter`](https://github.com/NVIDIA/cutlass/blob/b46b16d003484063bca4ed365e44095c4c6ed633/include/cutlass/gemm/device/gemm_universal_adapter.h),
the CUTLASS
[grouped NVFP4 SM120 example](https://github.com/NVIDIA/cutlass/blob/b46b16d003484063bca4ed365e44095c4c6ed633/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu),
and the CUTLASS
[Blackwell block-scale layout documentation](https://github.com/NVIDIA/cutlass/blob/b46b16d003484063bca4ed365e44095c4c6ed633/media/docs/cpp/blackwell_functionality.md).

## Validation

Use the backend test first, then full-model logits, throughput, and Nsys:

```bash
bash validate_qwen_nvfp4_prefill.sh build-sm120-cutlass/bin/test-backend-ops results
bash validate_qwen_nvfp4_prefill_logits.sh build-sm120-cutlass/bin/llama-debug MODEL results
bash bench_qwen_nvfp4_prefill.sh build-sm120-cutlass/bin/llama-bench MODEL results
bash profile_qwen_nvfp4_prefill_nsys.sh build-sm120-cutlass/bin/llama-bench MODEL results
```
