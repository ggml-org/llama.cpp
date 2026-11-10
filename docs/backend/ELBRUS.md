# llama.cpp on Elbrus (e2k)

This fork carries the fixes and benchmarks needed to run llama.cpp at useful
speed on Elbrus CPUs (MCST E8C/E16C, e2k-v6 ISA) built with lcc 1.29.

Test box: Elbrus E16C, 8 cores (no SMT) @ 2 GHz, 62 GB RAM, ~56 GB/s STREAM,
ALT Linux 11, lcc 1.29.16. All numbers below are `llama-bench` runs on this box.

## Why a fork is needed

Upstream cmake classifies `CMAKE_SYSTEM_PROCESSOR=e2k` as
`GGML_SYSTEM_ARCH=UNKNOWN`, which drops every quantized dot/gemm SIMD kernel
and the repack tables into scalar C fallbacks. Qwen3-8B prompt processing is
~2 t/s this way.

lcc, however, natively defines `__SSE2__`/`__AVX__`/`__AVX2__` and translates
x86 intrinsics into real e2k-v6 vector operations (`qpmaddubsh`, `qpshufb`,
`qppermb`, `qpfmas` are visible in the disassembly of the resulting binaries).
Routing e2k through the x86 path therefore activates the full SIMD kernels with
zero hand-written e2k code.

## The two patches

1. `ggml/cmake/common.cmake`: map `e2k` to the x86 arch dir.
2. `ggml/src/ggml-cpu/arch-fallback.h`: add `__e2k__` to the x86
   rename-macro branch (the branch is keyed on `__x86_64__`, which lcc does
   not define).

Plus one optional kernel patch:

3. `ggml/src/ggml-cpu/arch/x86/quants.c`: hoist the eight
   `get_scale_shuffle_k4(i)` constants out of the `dot_q4_K_q8_K` inner loop
   (guarded by `__e2k__`). lcc spills them into the loop; microbenchmark: +14%
   on the kernel.

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF
cmake --build build -j8 --target llama-server llama-cli llama-bench
```

`-DGGML_OPENMP=OFF` is strongly recommended on Elbrus, see the barrier
finding below. It also fixes the multithreaded `llama-cli` SIGILL.

## Full benchmark log

### Stage 1 - x86 SIMD path, Qwen3-8B Q4_K_M, t8

| Build                    | pp512 | tg128 |
|--------------------------|-------|-------|
| Baseline (scalar)        | 1.96  | 0.90  |
| x86-SIMD (repack on)     | 17.89 | 1.78  |
| x86-SIMD (`--repack 0`)  | 14.06 | 2.14  |
| x86-SIMD + LTO           | 17.63 | 2.05  |

pp x9.1, tg x2.4. On e2k the Q4_K->Q8_K repack tables *hurt* token generation
(-17%) while helping prompt processing (+27%): pick by workload.

Qwen3-32B Q4_K_M (dense): pp256=2.81, tg ~0.7 t/s. Memory-bound.

### Stage 2 - Qwen3.8-27B (qwen35 arch) Q4_K_M, t8, x86 build

| Config                       | pp512 | tg64 |
|------------------------------|-------|------|
| repack on (default)          | 4.07  | 0.68 |
| `--repack 0`                 | 3.58  | 0.76 |
| `--repack 0 -fa on`          | 3.58  | 0.75 |
| `-ctk/-ctv q8_0` + fa        |   -   | 0.75 |
| `-lm mlock`                  |   -   | 0.75 |
| `-b 1024 -ub 128`            | 3.31  |    - |
| `--poll 100`                 | 4.07  | 0.76 |
| `GGML_CPU_TILED_MM_FORCE=1`  | 4.05  |    - |

flash-attn / KV q8_0 / mlock / ubatch / poll / tiled-mm: no measurable effect
on this backend at these sizes.

clang-19 (ALT) also compiles the x86 path via lcc's intrinsic headers and even
fixes the cli SIGILL, but its e2k codegen is 4x slower (pp512=0.95, tg64=0.49).
lcc 1.29 is the right compiler for Elbrus.

### Stage 3 - the real bottleneck: libomp sleep barriers

Thread scaling of dense 27B generation was 0.16/0.31/0.55/0.69/0.75 t/s for
t=1..8 (x4.7) while an 8-thread kernel harness on the same file reached x7.3.
The kernels were not the problem: production `ggml_vec_dot_q4_K_q8_K` streams
a real GGUF mmap at 12.4 G weights/s/core. The gap was lcc's libomp port:
its barriers block on condvars, and with one barrier per graph node the futex
wake latency (~90-167 us) eats most of the per-node work slice at t=8.

| config                        | tg64 | pp512 |
|-------------------------------|------|-------|
| libomp default                | 0.73 | 3.54  |
| `OMP_WAIT_POLICY=active`      | 1.09 | 3.54  |
| + `OMP_PROC_BIND=spread`      | 1.11 | 3.54  |
| `GGML_OPENMP=OFF` (ggml pool) | 1.12 | 3.47  |

**+52-54% generation, no env needed.** The ggml thread pool uses spin
barriers, which is the right trade on a box with no SMT and no other load.
Bonus: `llama-cli -t 8` stopped crashing with SIGILL (the crash was inside
`__omp_parallel_start`).

### Stage 4 - model choice: MoE wins big on Elbrus

Measured dense-vs-MoE, same box, same build, t8:

| model                         | size   | tg64       | pp512  |
|-------------------------------|--------|------------|--------|
| Qwen3-30B-A3B Q4_K_S (MoE)    | 16.3G  | 6.81/6.90  | 21.26  |
| Qwen3.8-27B Q4_K_M (dense)    | 15.3G  | 1.12       | 4.07   |

x6 faster generation, x5 prompt throughput at 30B-class quality. Reason: tg
throughput tracks *bytes touched per token*. MoE touches only its routed
experts (~3B active params), the dense model touches all its weights.

Quality smoke (4 prompts, thinking off): 30B-A3B 4/4 correct; 27B dense 2/4
(wrong on bat-and-ball trick question). Recommendation: run MoE.

### Stage 5 - MoE tuning matrix (Qwen3-30B-A3B Q4_K_S, noomp build, t8)

| config                         | tg64 | pp512       | pp2048+tg128 |
|--------------------------------|------|-------------|--------------|
| default                        | 6.92 | 21.26       | 7.20         |
| `-lm mlock`                    | 6.77 | -           | -            |
| `--cpu-strict 1`               | 6.91 | -           | -            |
| t7 / t6 / t4                   | 5.92 / 5.33 / 3.76 | - | -        |
| `-fa on`                       | 6.92 | 21.28       | 8.89         |
| `-fa on -ctk/-ctv q8_0`        | 7.24 | 15.08       | 6.03         |
| `-b 4096` / `-b 1024 -ub 256`  | -    | 21.28 / 21.12 | -          |
| model copied to tmpfs          | 6.79 | 21.07       | -            |

MoE generation is compute-bound, not memory-bound: it touches ~934 MB/token
(x6.9 = 6.4 GB/s out of 56 GB/s STREAM), and tg scales near-linearly with
threads. tmpfs, mlock and cpu-strict change nothing - the page cache is
already optimal. flash-attn is neutral at short prompts, +23% on the mixed
pp2048+tg128 profile; fa + q8_0 KV gives +5% tg64 but hurts long-prompt
mixed throughput (6.03 vs 8.89), so it is not the default.

### Stage 6 - speculative decoding: rejected for MoE targets

MoE tg costs ~934 MB/token of per-unique-token weight traffic, so verifying
d draft tokens should cost about the same as generating 1. Tested with
llama-cli (t8, temp 0, n=160, 300-token prompt):

| config                             | tg (t/s) |
|------------------------------------|----------|
| baseline                           | 5.4      |
| EAGLE3 draft (290 MB f16), d=4     | 2.3      |
| EAGLE3 draft, d=8                  | 1.7      |
| EAGLE3 draft, d=16                 | 1.1      |
| ngram-mod (d=8)                    | 5.3      |

The dense f16 EAGLE3 draft (converted from
`Tengyunw/qwen3_30b_moe_eagle3` with `conversion/convert_hf_to_gguf.py
--model-name eagle3`) runs on lcc scalar-ish dense gemm and costs more per
step than the MoE target verifies; deeper drafts scale that cost. ngram-mod
is neutral on fresh prompts. Do not enable spec decoding for MoE targets on
this backend. Draft kept at ~/models/eagle3-qwen3-30b-a3b-f16.gguf.

### Stage 7 - q5_K/q6_K dot kernel hoist patches: rejected

The same lcc shuffle-rematerialization fix as the q4_K kernel, applied to
`dot_q5_K_q8_K` (8 hoisted k4 shuffles + full unroll) and `dot_q6_K_q8_K`
(8 hoisted 16B shuffles + full unroll), both guarded by `__e2k__`. These
quants are lm_head (~244 MB) and ffn_down experts (~307 MB) of the ~934
MB/token traffic. Qwen3-30B-A3B, t8:

| build         | tg64        | pp512        |
|---------------|-------------|--------------|
| without patch | 6.92        | 21.26        |
| with patch    | 6.68 ± 0.12 | 20.85 ± 0.04 |

Greedy output was bit-identical, tg is ~3% slower: at nc=1 most of that
traffic goes through the gemv paths, not these dot kernels, and the extra
register pressure just makes lcc's allocator worse. Patch reverted; a kernel
microbenchmark win does not transfer when the kernel is not the hot path.

## Limitations found so far

- PGO: `ecf_opt64` hangs with `-fprofile-generate` on
  `ggml/src/ggml-cpu/arch/x86/repack.cpp` (78+ min at 100% CPU). Do not use
  PGO builds on lcc.
- LTO: no measurable gain.
- `-march=elbrus-v7` codegen is identical for this workload (the chip is v64);
  AVX-VNNI intrinsic (`_mm256_dpbusd_epi32`) is not provided by lcc.
- `perf`, `gdb` frame unwinding, `eu-stack` are unavailable/broken on this
  ALT kernel (ptrace paranoid=2, no e2k unwinders); profile with microbench
  harnesses and `-o md` bench matrices instead.
- `--moe-cache-mib` is GPU-only (expert staging for hybrid inference); there
  is no CPU-side expert cache in upstream yet.

## Recommended run (E16C, 8 cores)

```sh
# daily driver: Qwen3-30B-A3B-Instruct-2507-Q4_K_S, ~6.9 t/s generation
llama-server -m ~/models/Qwen3-30B-A3B-Q4_K_S.gguf -t 8 --port 8080

# dense 27B: chat mode favors --repack 0, RAG/long prompts favor repack on
llama-server -m ~/models/Qwen3.8-27B-Q4_K_M.gguf -t 8 --repack 0
llama-server -m ~/models/Qwen3.8-27B-Q4_K_M.gguf -t 8
```
