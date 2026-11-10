# llama.cpp on Elbrus (e2k) — optimization results

Hardware: Elbrus E16C (8 cores, e2k-v6, @2 GHz), 62 GB RAM, ALT Linux 11, lcc 1.29.16.

## Problem
Upstream cmake classifies e2k as `GGML_SYSTEM_ARCH=UNKNOWN` -> scalar fallback for
quantized dot/gemm kernels and repack tables. Baseline Qwen3-8B: pp512 ~2 t/s.

## Fix (this branch)
1. `ggml/cmake/common.cmake`: map `CMAKE_SYSTEM_PROCESSOR=e2k` to the x86 arch path.
2. `ggml/src/ggml-cpu/arch-fallback.h`: add `__e2k__` to the x86 rename-macro branch.

lcc 1.29 natively defines `__SSE2__`/`__AVX__`/`__AVX2__` and translates x86
SSE2/AVX2 intrinsics into native e2k-v6 vector operations
(`qpmaddubsh`, `qpshufb`, `qppermb` visible in disassembly). This activates the
full x86 SIMD path: arch/x86 quants dot kernels, repack, sgemm dispatch.

## Benchmarks (llama-bench, Qwen3-8B Q4_K_M, t8)
| Build                  | pp512  | tg128 |
|------------------------|--------|-------|
| Baseline (scalar)      | 1.96   | 0.90  |
| x86-SIMD (repack on)   | 17.89  | 1.78  |
| x86-SIMD (`--repack 0`)| 14.06  | 2.14  |
| x86-SIMD + LTO         | 17.63  | 2.05  |

Speedup: pp x9.1, tg x2.4. Repack tables hurt tg on e2k -> use `--repack 0` for chat.
Qwen3-32B Q4_K_M: pp256=2.81, tg ~0.7 t/s (memory-bandwidth bound; STREAM ~56 GB/s).

## Limitations
- PGO unusable: `ecf_opt64` hangs with `-fprofile-generate` on `arch/x86/repack.cpp`.
- LTO: no significant gain.
- Multi-threaded `llama-cli` hits SIGILL inside libomp (`__omp_parallel_start`,
  nested parallel region; reproduces without these patches). Use `llama-server`,
  or `OMP_NUM_THREADS=1` for cli.

## Recommended run
```
llama-server -m Qwen3-8B-Q4_K_M.gguf -ngl 0 -t 8 --repack 0
```

## Build
```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build -j8 --target llama-server llama-cli llama-bench
```
