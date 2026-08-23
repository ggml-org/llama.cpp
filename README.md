# llama.cpp — ROCmFPx + DFlash2 fork

> A personal fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
> with two goals: **be the fastest Vulkan path on AMD Strix Halo**, and **carry the
> newest inference features** — new quant families, new speculative-decoding
> methods — while they are still in flight upstream.
>
> Concretely that means AMD-native FP4/FPx quantisation, every speculative
> decoding method llama.cpp supports (MTP, DFlash2, DSpark, ngram) benchmarked
> against each other on this hardware, and Vulkan kernel fixes that make the two
> work well *together*. Everything below the divider is the upstream README,
> unchanged.
>
> Vulkan is the target deliberately: it is the path that works on a stock Mesa
> install with no ROCm toolchain, and it is where Strix Halo is least tuned.

## What is in here

| | Where it came from |
|---|---|
| **ROCmFPx quant types** — `Q4_0_ROCMFP4`, `_FAST`, `Q2/Q3/Q6/Q8_0_ROCMFPX`, with CPU codecs and Vulkan dequant / mat-vec / matmul / integer-dot kernels | Hand-ported from [ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX). Two decode bugs in that source are fixed here — see [ROCMFPX-NOTES.md](ROCMFPX-NOTES.md) |
| **DFlash2 speculative decoding** | [PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342) by Jian Chen, not yet merged upstream. Carried unmodified, original authorship intact |
| **Vulkan batched mat-vec fixes** | New here. IQ3_S register spill at `NUM_COLS > 4` (5x at n=8), and the ROCmFPx batch 3-8 rework described below. Both are upstreamable |
| **Vulkan LDS stride fix** — +13.7% prefill on *any* quant, not just ours | New here. A four-way LDS bank conflict in the coopmat matmul that affects every non-Intel device. Upstreamable, and the single biggest win in this fork — see [PROGRESS.md](PROGRESS.md) |

## Against mainline llama.cpp

The table below is this fork versus upstream `ggml-org/llama.cpp` master
`95b8e33e1`, built from a clean worktree with the same CMake flags, run
interleaved on an idle GPU. **Both columns run stock third-party K-quants** — no
ROCmFPx, no fork-specific model format — so this measures the Vulkan work alone.

| model | metric | mainline | this fork | |
|---|---|---|---|---|
| unsloth `Qwen3.8-27B-UD-Q4_K_XL` (dense) | pp512 | 270.1 | **307.2** | **+13.7%** |
| | pp2048 | 260.2 | **291.7** | **+12.1%** |
| | tg64 | 11.72 | 11.71 | — |
| bartowski `Ornith-1.5-35B-A3B-Q4_K_M` (MoE) | pp512 | 1034.9 | 1049.9 | +1.5% |
| | pp2048 | 845.3 | **872.5** | **+3.2%** |
| | tg64 | 65.74 | 65.74 | — |

MoE gains less because a 3B-active model is far less matmul-bound to begin with.

Generation is unchanged **in this particular comparison**, which is the expected
result and not a disappointing one: single-stream decode of a stock K-quant is
already at 80% of theoretical memory bandwidth on this APU, so no kernel can move
it. The generation work here is aimed where the headroom actually is — at the
batch sizes speculative decoding runs at.

### Speculative decoding: present upstream, but broken where it counts

DFlash2, MTP, DSpark and Eagle3 all landed in upstream master. Having the method
is not the same as having it work. Speculation verifies several drafted tokens in
one batch, so it lives at batch 3–8 — and that is exactly the range where the
Vulkan mat-vec kernels fall apart:

IQ3_S mat-vec, m=4096 k=14336, µs/run against upstream `95b8e33e1` — lower is
better. `n` is the batch width, i.e. how many drafted tokens are verified at once:

| n | mainline | this fork | |
|---|---|---|---|
| 1 | 98.3 | 98.0 | parity |
| 2 | 110.7 | 114.0 | −3% |
| 4 | 167.4 | 172.9 | −3% |
| **8** | **1542.0** | **285.0** | **5.4x** |

The rest, which mainline has no equivalent for:

| | mainline | this fork |
|---|---|---|
| **ROCmFPx with DFlash2** | cannot load the model at all | 6 quant types, CPU + Vulkan |
| ROCmFPx mat-vec at batch 3–8 | n/a | 312 → **173** µs (fp4), 2236 → **402** (fp6) |
| GDN output projection, multi-slot | one GEMV per sequence | one batch-wide GEMV, **+8.6%** at B=8 |

The IQ3_S spill is the clearest case. `IQ3_S` is a stock upstream quant type and
the bug is in a stock upstream shader: it assigns 8 invocations per 256-weight
superblock, so at `NUM_COLS > 4` each invocation keeps 32 floats of B live per
column and overflows the 256-VGPR budget. Single-token generation never reaches
that width, which is why it went unnoticed — **speculative decoding is in that
range on every step.** So on mainline, DFlash2 plus an IQ3_S model is a feature
that nominally works and is 5x slower than it should be.

The 3% at n=2 and n=4 is a real cost, not noise: the fix moves `dscale` inside the
inner loop for every `NUM_COLS`, so the narrow cases pay slightly for the wide
ones. Trading 3% at n=2 for 5.4x at n=8 is the right side of that bargain when
the whole point is verifying batches of drafts.

ROCmFPx is the other half: mainline cannot load these models at all, so the
question of drafting with them does not arise. Making the two work *together* —
FP4 weights at spec-decoding batch widths — is what most of this fork is.

The cause is one constant. `mul_mm.comp` stages its shared tiles at
`SHMEM_STRIDE = BK/2 + PAD` in dword units, which map 1:1 onto the 32 LDS banks,
and reads B back column-major — so `gcd(SHMEM_STRIDE, 32)` is the bank-conflict
factor. Upstream sets `PAD` only for Intel-on-Windows; everything else takes the
shader default of 4, giving stride 20 and a four-way conflict on every B load.
Stride must stay *even* (an odd stride costs 3x by breaking RADV's wide
`ds_read_b64/b128` path) so 6 is the fix. It is worth 1.17x on `q4_0` and 1.18x
on `q8_0` at the kernel level.

## Why it exists

Speculative decoding verifies several tokens in one batch, so a decoder that is
fast at batch 1 and slow at batch 8 throws the speedup away. That is exactly what
the ROCmFPx kernels did: FP4 beat a K-quant of the same model at batch 1 and lost
badly at batch 8. Two shader changes fixed it — amortising the UE4M3 scale decode
over a whole 32-weight block, and replacing the per-weight bit-window gather in the
fp6/fp3 paths with a branch-free one.

Vulkan mat-vec, m=4096 k=14336, n=8, µs/run — lower is better:

| | before | after |
|---|---|---|
| `q4_0_rocmfp4` | 312 | **173** |
| `q6_0_rocmfpx` | 2236 | **402** |
| `q3_0_rocmfpx` | 1942 | **463** |

## Measured results

Qwen3.8-27B-ROCmFP4_FAST (13.55 GiB, 4.26 bpw, 505 of 506 weight tensors on the
ROCMFP4_FAST path), Radeon 8060S (Strix Halo APU), RADV / Mesa 26.0.3, idle GPU,
build `eacc56f4e`. Greedy sampling, 300 generated tokens, 186 measured cells at
0.29% median spread.

### Bare decode is against the memory wall

| model | size | tg128 | achieved | of 256 GB/s peak |
|---|---|---|---|---|
| ROCmFP4_FAST | 13.55 GiB | 14.05 ± 0.09 | 204.4 GB/s | 79.8% |
| Q3_K_M | 12.56 GiB | 15.18 ± 0.08 | 204.7 GB/s | 80.0% |

Two decoders with nothing in common — an FP4 codebook with UE4M3 scales, and Q3_K
superblocks — reach the same bandwidth to three significant figures. **Single-stream
decode has no headroom left in these shaders.** Every remaining gain has to come
from draft acceptance, which multiplies effective bandwidth instead of competing
for it. That makes this a draft-quality problem, not a kernel problem.

### Speculative decoding, tokens/s

Short context (~350 tokens):

| config | prose | code | JSON |
|---|---|---|---|
| bare | 13.90 | 13.90 | 13.89 |
| MTP n=3 | **26.18** | 29.83 | 34.91 |
| DFlash2 · z-lab Q8_0 n=7 | 21.96 | **35.24** | **41.54** |

Long context (~31K tokens of real C source):

| config | verbatim reproduction | prose about the code |
|---|---|---|
| bare | 12.20 | 12.19 |
| MTP n=4 | **36.07** (97.5% acc) | **20.54** |
| DFlash2 · Q8_0 n=7 | 31.79 | 15.69 |
| ngram-simple (no draft) | 25.41 | 11.72 (below bare) |

**The ranking inverts with context length.** DFlash2 wins at short context — 41.5 t/s
on JSON, 3.0× bare. By 31K, MTP takes both tasks, reversing a 41.5-vs-35.3 deficit.
The cause is structural: a DFlash2 sidecar keeps its own KV cache over the full
context and re-runs up to seven times per verification step, so its cost scales with
context. MTP's nextn layer reuses the target's state and never pays that.

Content dominates configuration: identical weights at identical context span
36.07 to 20.54 t/s purely on whether the output is quotable.

### What to run

| situation | method |
|---|---|
| short context, structured output | `--spec-type draft-dflash`, z-lab Q8_0 sidecar, `--spec-draft-n-max 7` |
| long context, any task | `--spec-type draft-mtp` at n-max 3–4 — no sidecar, no second KV cache |
| quote-heavy long context | `ngram-simple` alone, never layered under a draft model (costs 13%) |
| avoid | `ngram-cache` (slower than bare), `Q2_0_ROCMFPX` (no Vulkan kernel), DSpark on this target |

Do not requantize the z-lab Q8_0 sidecar. Our `Q8_0_ROCMFPX` scores 53.5% acceptance
against z-lab's 60.2% at the same bpw and identical tensor routing — it even lands
below our own FP4, which is impossible as a precision effect. The cause is the block
scale: `Q8_0` stores an fp16 scale, ROCmFPx stores a UE4M3 byte. At 8 bits per
weight the codes are not the problem, the coarse scale is.

## What this does not claim

- **One machine.** Every number above is from a single Radeon 8060S. Nothing here
  has been tested on discrete AMD, NVIDIA, Intel, or Apple hardware.
- **FP4 is not free.** ROCmFPX-MQ-Q4 measures 5.9842 wikitext-2 perplexity against
  5.8965 for UD-Q4_K_XL — about 1.5% worse, with overlapping error bars.
- **Earlier numbers in this file were wrong about which file they measured.** A
  previous revision quoted ~13.3 t/s bare and 24.6 t/s DFlash2-prose as "FP4".
  Those were measured through a `:Q3_K_M` tag that resolves to a stock unsloth
  k-quant sitting beside the FP4 builds in the same repository — a file containing
  zero ROCmFPx tensors. The table above is the actual FP4 file. Always check the
  tensor type mix of a "FP4" GGUF before attributing a number to FP4.
- **Only the JSON column is a clean draft comparison.** Greedy decoding is not
  bitwise-reproducible across verification batch shapes on Vulkan: reduction order
  shifts, and on high-entropy content a small logit difference flips an argmax.
  Output hashing showed 7 distinct outputs across 8 drafts on prose and 4 on code,
  but 1 on JSON. Speculation stays distribution-preserving; it is simply not
  bit-identical, so prose and code acceptance figures are confounded by the drafts
  generating different text of different difficulty.
- **Incomplete port.** No HIP/CUDA or OpenCL FPx kernels, no Vulkan kernels for
  `Q2_0_ROCMFPX`. Flash-attention with FPx KV is in the tree but uncommitted, and
  measurement says it is not worth much: at depth 20,000 `q8_0` KV is +2.1% over
  f16 and `q4_0_rocmfp4` is −1.8%, because per-access dequantization costs more
  than the bandwidth it saves.
- **The table above is a narrow comparison, by design.** It is one stock quant,
  one stream, no speculation — chosen so the Vulkan work is the only variable.
  It is not the whole picture: the batching and quant work below moves generation
  too, on configurations that table deliberately excludes.
- **Speculative decoding is no longer a differentiator.** DFlash2, MTP, DSpark
  and Eagle3 are all in upstream master now — [PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342)
  landed. What this fork adds on top is making them *fast at the batch sizes they
  actually run at*.
- **Not upstream, and not a stable base.** Branches here are rebuilt against
  `ggml-org/master` by hand. As of this writing the fork is 51 commits behind
  upstream master, so the comparison above also carries whatever upstream changed
  in those 51 commits.

Correctness is checked with `test-backend-ops` against the CPU backend —
17920/17920 on Vulkan, and again with `GGML_VK_FORCE_MMVQ=1`.

## Try it

```sh
cmake -B build -DGGML_VULKAN=ON && cmake --build build -j
build/bin/llama cli \
  -hf  lmcoleman/Qwen3.8-27B-ROCmFPX-GGUF:Q4 \
  -hfd incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M \
  --spec-type draft-dflash --spec-draft-n-max 7 -ngl 99 -fa on
```

`--spec-draft-n-max 7` is the DFlash2 value for short, structured output. Past
roughly 30K tokens of context, switch to `--spec-type draft-mtp` at n-max 3–4:
it needs no sidecar and no second KV cache, and it wins on both quotable and
non-quotable output at that length. MTP peaks at n=3–4 and decays — by n=7 it is
below its own n=3 result on every content class.

[ROCMFPX-NOTES.md](ROCMFPX-NOTES.md) has the full measurements, the open work, and
the profiling traps.

---

*Everything below is the upstream llama.cpp README. Its badges and links point at
`ggml-org/llama.cpp`, not at this fork.*

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3A0cc4m%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [compile times](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-compile-times.md) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
