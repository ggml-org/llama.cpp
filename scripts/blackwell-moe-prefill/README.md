# SM120 CUTLASS MoE prefill

This branch adds an optional CUTLASS path for W4A4 expert GEMMs on Blackwell.
The path is disabled in normal builds and is selected only for the supported
MoE graphs. Other devices, tensor layouts, and small token counts continue to
use the existing CUDA backend.

The current path covers:

- GPT-OSS MXFP4 prefill with fused W13 weights, top-4 routing, and OAI SwiGLU.
- Qwen3.6-35B-A3B NVFP4 prefill with separate gate/up weights, top-8 routing,
  output scales, and standard SwiGLU.

The implementation repacks expert weights once, builds one expert-sorted route
plan per layer, quantizes activations to the CUTLASS input layout, runs grouped
W13 and W2, and finishes the two stages with CUDA epilogues.

## Build

CUTLASS is not downloaded by CMake. Point the build at revision
`b46b16d003484063bca4ed365e44095c4c6ed633`:

```bash
cmake -S . -B build-cutlass \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_CUTLASS_MOE=ON \
  -DGGML_CUDA_CUTLASS_PATH=/path/to/cutlass
cmake --build build-cutlass -j
```

The option requires CUDA 12.9 or newer and builds the CUTLASS translation unit
for `sm_120f`. The rest of the CUDA backend keeps its normal architecture list.

## GPT-OSS conversion

GPT-OSS uses a fused gate/up tensor in this path. Convert it with:

```bash
python convert_hf_to_gguf.py /path/to/gpt-oss-120b \
  --outfile /path/to/gpt-oss-120b-mxfp4-fused.gguf \
  --outtype auto \
  --fuse-gate-up-exps
```

The converter writes gate rows first and up rows second. Models converted
without the option keep the existing separate gate/up layout and fall back to
the normal backend.

## Runtime selection

Building with `GGML_CUDA_CUTLASS_MOE=ON` enables the supported prefill path.
Set `GGML_CUDA_MOE_MMQ_DISABLE=1` to force the existing CUDA implementation for
an A/B run.

The following variables are retained for kernel sweeps:

- `GGML_CUDA_MOE_MMQ_CUTLASS_W13_TILE_N=32|64|128`
- `GGML_CUDA_MOE_MMQ_CUTLASS_W2_TILE_N=32|64|128`
- `GGML_CUDA_MOE_MMQ_CUTLASS_W13_SWAP_AB=0|1`
- `GGML_CUDA_MOE_MMQ_CUTLASS_W2_SWAP_AB=0|1`
- `GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0|1`
- `GGML_CUDA_MOE_MMQ_CUTLASS_VALIDATE_SUPPORT=0|1`

## Backend tests

The CUTLASS graph tests are opt-in because their tensors are large:

```bash
GGML_CUDA_CUTLASS_MOE_TEST=1 \
  build-cutlass/bin/test-backend-ops test -b CUDA0 -o MOE_MXFP4_BLOCK

GGML_CUDA_CUTLASS_MOE_TEST=1 \
  build-cutlass/bin/test-backend-ops test -b CUDA0 -o MOE_NVFP4_BLOCK
```

They cover contiguous and strided expert IDs, the two observed view orders,
skewed routing, and an output consumed by a shared-expert add. The performance
suite includes 512, 2048, and 8192 token cases.

Measured results and Nsys summaries are in [RESULTS.md](RESULTS.md).
