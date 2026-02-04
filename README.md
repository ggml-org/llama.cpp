# IMPORTANT: this version has some bugs that seems to be affecting output correctness. Its an experimental build, use at your risk until will be fixed! 


# llama.cpp-gfx906-2602

Based on llama.cpp build 7924.



## Benchmark Results

![Benchmark Results](benchmarks.svg)

See [SCRIPT_llama_bench.sh](SCRIPT_llama_bench.sh) for llama-bench configuration and [SCRIPT_launch_server_MI50.sh](SCRIPT_launch_server_MI50.sh) for server launch settings.



## What Changed

The core modifications are implemented in ggml-cuda/gfx906 folder.
### 2602
```
  ggml/src/ggml-cuda/gfx906/
  ├── gfx906-common.cuh          - DPP warp reductions & common utilities
  ├── gfx906-config.h            - Feature toggles
  ├── attention/
  │   ├── fattn-q8.cuh           - Q8 FlashAttention kernel
  │   ├── fattn-q8.cu            - Instance launcher
  │   ├── rope.cuh               - Optimized RoPE kernel
  │   └── instances/             - Template instantiations for various head dims
  ├── fused/
  │   ├── gather-q8.cuh          - Q8 gather helpers
  │   ├── gather-q8.cu           - Q8 gather kernel
  │   ├── graph-fusion.cuh       - Graph fusion logic
  │   ├── mmq-prequantized.cuh   - Prequantized MMQ helpers
  │   ├── norm-fused-q8.cuh      - Fused norm dispatch
  │   └── norm-fused-q8.cu       - Fused norm kernels
  ├── matmul/
  │   ├── mmf.cuh                - MMF (mul-mat-fused) helpers
  │   ├── mmq.cuh                - MMQ vectorized loads
  │   ├── mmq-prefetch.cuh       - Prefetch helpers
  │   ├── mmvq-q4_0.cuh          - Warp-cooperative MMVQ Q4_0
  │   ├── mmvq-q4_1.cuh          - Warp-cooperative MMVQ Q4_1
  │   ├── mmvq-q8_0.cuh          - Warp-cooperative MMVQ Q8_0
  │   └── sgemm.cuh              - SGEMM helpers
  └── quantize/
      ├── epilogue.cuh           - DPP-based Q8_1 epilogue
      ├── q8-cache.cuh           - Q8 cross-op cache
      └── vecdotq.cuh            - MXFP4 vectorized loads
```


### 2601
```
gfx906-mmvq-q4_0.cuh Warp-cooperative Q4_0 MMVQ kernel
gfx906-mmvq-q4_1.cuh Warp-cooperative Q4_1 MMVQ kernel
gfx906-mmvq-q8_0.cuh Warp-cooperative Q8_0 MMVQ kernel
mmvq.cu              Half-warp (32 threads) dispatch for MoE small matrices
```

### 2512
```
mmq.cuh              Software pipelining for Q8_0 MMQ loads
mmq.cuh              Optimized Q8 MMQ need_check path to avoid LDS conflicts
mmq.cuh              MXFP4 load pipeline with e8m0 conversion optimization
vecdotq.cuh          Fast Q8_0 load path using memcpy
vecdotq.cuh          Software pipeline MXFP4 MMVQ for v_perm latency hiding
vecdotq.cuh          MXFP4 lookup with 2-perm + arithmetic sign
mmq.cu/mmid.cu       MoE sub-warp shuffle fix for wavefront64 (fixes gpt-oss loading problems)
```

### 2511

```
common.cuh           DPP-based warp reductions with unified shuffle XOR dispatch
fattn-common.cuh     GCN-optimized thread counts and tile configurations
fattn.cu             Q8-optimized tile kernel selection for GFX906 flash attention
mmq.cu               Integrated GFX906 vectorized loads for Q4_0/Q4_1 quantizations
gfx906/              New directory with MI50/MI60-specific kernel implementations
```


## Quick Start

Optional but sometimes required, set your paths for rocm and device libs if they are not in /opt/rocm/

```bash
export ROCM_PATH=/opt/rocm-7.1.0 #optional
export HIP_DEVICE_LIB_PATH=/opt/rocm-7.1.0/amdgcn/bitcode #optional
```

```bash
git clone https://github.com/iacopPBK/llama.cpp-gfx906.git
cd llama.cpp-gfx906
./SCRIPT_compile_MI50.sh      # edit ROCM_PATH if not using /opt/rocm
./SCRIPT_launch_server_MI50.sh # edit MODEL_PATH to your model file
./SCRIPT_llama_bench.sh # edit MODEL_PATH to your model file, performs the bench shown above

```

Tested with ROCm 7.1.1 and GFX906 GPU (MI50/MI60).



## Power Scaling

Performance scales with power limit using [SCRIPT_overclock_upp_MI50.sh](https://github.com/sibradzic/upp) for MI50 overclocking via UPP (Powerplay Table Editor). Results gathered using 2511 release.

![PP Performance](power_sweep_pp.svg)

![TG Performance](power_sweep_tg.svg)



## Special Thanks and Links
Props to these users for spending time on the repo.

[@fuutott](https://github.com/fuutott) ・ [@mircoboschi](https://github.com/mircoboschi) ・ [@skyne98](https://github.com/skyne98) ・ [@kamali-lab](https://https://github.com/kamali-lab)

---

[AMD GCN ISA](https://gpuopen.com/learn/amdgcn-assembly/) ・ [llama.cpp](https://github.com/ggml-org/llama.cpp) ・ [ROCm](https://rocm.docs.amd.com/) ・ [GFX906 DISCORD](https://discord.gg/ZEcgt3dAw) ・ [wiki-gfx906](https://github.com/skyne98/wiki-gfx906) ・ [llama-labs-gfx906](https://github.com/skyne98/llama-labs-gfx906)



<sub>Built for the GFX906 community</sub>
