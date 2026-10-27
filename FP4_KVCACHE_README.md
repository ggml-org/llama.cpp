# FP4 E2M1 KV Cache Implementation

This branch tracks the FP4 E2M1 KV cache implementation for llama.cpp targeting RTX 5090 (Blackwell sm_120).

## Full Implementation
https://github.com/YuvrajSinghBhadoria2/fp4-kvcache-llama-cpp

## Patches
- 0001-fp4-e2m1-ggml-type.patch
- 0002-fp4-e2m1-cpu-kernels.patch  
- 0003-fp4-e2m1-cuda-kernels.patch
- 0004-fp4-e2m1-flash-attn-dispatch.patch
- 0005-fp4-e2m1-llama-bench.patch
- 0006-mxfp4-conversion-kernels.patch
- 0007-mxfp4-native-mma-kernel.patch

## Performance Targets
- Fallback path (FP4→FP16→F16 MMA): 80-120 tok/s, 4× memory reduction
- Native path (MXFP4 MMA): 230+ tok/s, 4× memory reduction

## Status
Draft PR - implementation complete in external repo, patches need final review.
