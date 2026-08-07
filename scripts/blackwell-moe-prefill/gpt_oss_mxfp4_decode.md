# GPT-OSS MXFP4 CUTLASS decode

This experiment measures the CUTLASS ceiling for the GPT-OSS-120B expert
pipeline at one to eight tokens. It is deliberately strict: other models,
types, shapes, routing widths, devices, and graph layouts stay on the existing
llama.cpp path.

## Accepted graph

The CUDA graph matcher requires the complete 15-node GPT-OSS expert branch:

```text
fused W13 MXFP4 -> bias -> OAI SwiGLU -> W2 MXFP4 -> bias
    -> routing weight -> four-route reduction
```

The accepted tensor contract is:

| Item | Value |
| --- | ---: |
| Hidden size | 2880 |
| Expert size | 2880 |
| Experts | 128 |
| Experts per token | 4 |
| W13 | K=2880, N=5760 |
| W2 | K=2880, N=2880 |
| Tokens | 1, 2, 4, or 8 |
| Weight type | MXFP4 |
| Activation input/output | F32 |
| Device | SM120/SM121 with Blackwell MMA support |

The matcher also verifies the views, offsets, strides, OAI SwiGLU parameters,
bias dependencies, routing-weight dependency, reduction tree, CUDA buffers,
and non-overlapping graph lifetime. A failed check returns before either expert
weight is transformed.

## Pipeline

The selected path is:

```text
F32 hidden to MXFP8 A1, with direct route-plan initialization in the same CTA
    -> quantized once per token and copied to four routes
    -> CUTLASS grouped W13
    -> W13 bias + OAI SwiGLU + MXFP8 A2 quantization
    -> CUTLASS grouped W2
    -> W2 bias + routing weight + four-route reduction
```

Decode uses one CUTLASS problem per routed row. Sorting at most 32 rows by
expert does not improve these M=1 problems, so decode does not run the prefill
histogram and prefix-sum scheduler. The A1 quantization CTA creates the identity
forward/inverse maps while it reads the four expert IDs. W13 and W2 reuse the
same plan without another scheduling launch.

The default 32-wide CUTLASS tile and swapped A/B orientation place the routed
row dimension on the narrow tensor-core tile. W13 and W2 remain independently
selectable through their existing tile and swap variables.

## Weight storage and lifetime

CUTLASS consumes separate value and scale planes. The transform pads K from
2880 to 2944. The padded value plane still fits in the canonical GGUF tensor
allocation, so the experiment replaces the device tensor contents in place and
allocates only the scale plane.

For one layer:

| Storage | Bytes |
| --- | ---: |
| Canonical W13 tensor | 1,128,038,400 |
| Canonical W2 tensor | 564,019,200 |
| Persistent W13 scale plane | 67,829,760 |
| Persistent W2 scale plane | 34,668,544 |

The persistent scale overhead is about 97.75 MiB per layer, or 3.44 GiB for 36
layers. Repacking uses a temporary value plane with a peak size of about 1.01
GiB. Preserving a second complete W13/W2 value copy for every layer would add
roughly the size of the expert weights again and is not practical for the
120B model.

The transform runs on `moe_weight_stream`. Each cache entry records a `ready`
event after repacking and a `last_use` event after its GEMM. The compute stream
waits for `ready`; context destruction waits for both events before destroying
CUDA graphs, streams, events, and owned scale storage. This follows CUDA's
stream-ordered allocation and event lifetime rules:

- [Stream-ordered memory allocation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)
- [CUDA event management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
- [CUDA graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)

After an in-place transform, native MMVQ must not read that tensor. The MMQ
dispatcher checks the cache state and aborts if such a fallback is attempted.
Native and CUTLASS measurements therefore use separate processes. CUDA graph
capture is disabled for this experiment because the first-use transform and
asynchronous allocations are not capture-stable.

The CUTLASS adapter lifetime is contained in one launcher call. Device problem
metadata and workspace come from the CUDA pool, remain alive through
`can_implement`, `initialize`, and `run`, and are returned only after the run is
queued on the same stream. The adapter sequence follows the pinned CUTLASS
[`GemmUniversalAdapter`](https://github.com/NVIDIA/cutlass/blob/b46b16d003484063bca4ed365e44095c4c6ed633/include/cutlass/gemm/device/gemm_universal_adapter.h).

## Validation

Build and execution are intentionally remote-only. The local stopping point is
static review; GPU compilation and validation start after the Blackwell machine
is opened.

Backend correctness covers 1, 2, 4, and 8 tokens, both graph node orders, and a
skewed expert distribution:

```bash
bash validate_cutlass_decode.sh \
  build-sm120-cutlass/bin/test-backend-ops \
  results
```

The tests run the CUTLASS graph alone against the CPU reference. They do not put
a native graph and an in-place CUTLASS graph in the same backend allocation.

Full-model logits use two processes and require a one-token prompt:

```bash
bash validate_gpt_oss_mxfp4_decode_logits.sh \
  build-sm120-cutlass/bin/llama-debug \
  /models/gpt-oss-120b-mxfp4.gguf \
  results
```

Steady decode throughput:

```bash
bash bench_gpt_oss_mxfp4_decode.sh \
  build-sm120-cutlass/bin/llama-bench \
  /models/gpt-oss-120b-mxfp4.gguf \
  results
```

Nsys comparison:

```bash
bash profile_gpt_oss_mxfp4_decode_nsys.sh \
  build-sm120-cutlass/bin/llama-bench \
  /models/gpt-oss-120b-mxfp4.gguf \
  results
```

Every script checks for `schedule=direct` and 36 unique W13 tensor names in the
dispatch markers. Set `GPT_OSS_EXPERT_LAYERS` for another layer count. A graph
mismatch or silent fallback fails the run. Nsys reports the one-time weight
transform separately from per-token kernel time.

## Acceptance gates

The experiment is complete only after the remote run establishes all of the
following:

1. The SM120 CUTLASS build succeeds without changing the pinned CUTLASS API.
2. All backend tests pass at 1, 2, 4, and 8 tokens.
3. Full-model one-token logits remain within the recorded MXFP8 activation
   error envelope and retain the same top-token decision.
4. The direct route-plan marker appears in every layer and no native fallback
   occurs after repacking.
5. Throughput includes at least five steady repetitions in separate native and
   CUTLASS processes.
6. Nsys separates first-use repacking from direct-plan/A1 quantization, W13,
   W13 epilogue/A2 quantization, W2, and final reduction.

The result may show that native MMVQ is faster at M=1. That is still a valid
closure: this path measures the best full-pipeline CUTLASS alternative without
changing the production decode default.
