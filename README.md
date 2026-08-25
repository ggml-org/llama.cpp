# llama.cpp - adaptive speculation (MTP / DFlash2)

A fast fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The
headline feature is **adaptive speculative decoding** with MTP or DFlash2: draft
length follows measured acceptance instead of a fixed `n`. The other job is the
fastest Vulkan path on AMD Strix Halo: stock Mesa, no ROCm toolchain.

**Qwen3.8-27B on AMD Strix Halo (Radeon 8060S): 65 t/s generation, 440 t/s prefill.**

## The numbers

**Generation, 65 t/s.** Adaptive DFlash2, FP4 target + FP4 sidecar, Qwen3.8-27B,
greedy, 300 tokens:

| | structured output | prose |
|---|---|---|
| bare decode | 14.0 t/s | 14.1 t/s |
| fixed draft n=3 | 41.6 | 25.4 |
| fixed draft n=7 | 20.2 | 24.8 |
| **adaptive draft, `n_max 7` `n_min 3`** | **65.6 t/s — 4.7×** | **26.1** |

Draft acceptance is what moves: fixed n=7 collapses to 18 %, fixed n=3 sits at 95 % and is
*under*-drafting, adaptive holds 96 % while drafting longer. The same `n_max` that destroys the
fixed arm is safe under adaptive. MTP uses the target's own nextn layers and needs no sidecar;
DFlash2 uses one. Adaptive works on both.

**Prefill, 440 t/s.** Same 27B, `pp512`, Q4_K or FP4 (they tie). Against pinned
upstream `95b8e33e1`, stock K-quants, no ROCmFPx:

| context depth | 0 | 4 K | 16 K | 32 K | 64 K |
|---|---|---|---|---|---|
| Qwen3.8-27B UD-Q4\_K\_XL | **+13.0 %** | +11.6 % | +9.1 % | +8.2 % | +4.6 % |
| Ornith-1.5-35B-A3B Q4\_K\_M | **+12.6 %** | +10.0 % | +5.9 % | +5.1 % | noise |

The gain decays with depth — as context grows, attention takes a larger share of prefill and the
matmul and concat paths this fork tunes take a smaller one. Generation on stock K-quants is flat
within ±1 %; the fork's generation story is speculative decoding and ROCmFPx, neither of which a
plain `llama-bench` run can see.

**Individual Vulkan gates**, measured on/off in one binary at `pp2048`:

| gate | Qwen3.8-27B dense | Ornith MoE | HauhauCS Q6\_K MoE |
|---|---|---|---|
| tiled concat-transpose | +3.3 % | **+45.1 %** | **+49.3 %** |
| f16 B for `mul_mat_id` | — | +6.9 % | +9.1 % |
| f16 B for `mul_mat` | +4.5 % | — | +5.8 % |
| LDS stride pad | +7.3 % | — | — |

Single Radeon 8060S (Strix Halo APU), RADV / Mesa 26.0.8, idle GPU, arms interleaved in palindrome
order against upstream `ggml-org/llama.cpp` `95b8e33e1` — the exact commit this fork merged, so the
delta is this fork's changes and not upstream drift.

**Two caveats that matter.** The speculative table was taken on a short high-power burst (79 °C,
115 W) that this chassis cannot sustain all day; on the everyday power profile the same
configuration reads **55.0 t/s** on structured output. And every generation figure needs its context
depth attached — these models declare 262144 context and generation at depth is roughly a third of
its depth-0 value.

Structured data, methodology and charts: [`bench/`](bench/) —
[`RESULTS.md`](bench/RESULTS.md), [`charts.html`](bench/charts.html),
[`README.md`](bench/README.md).

## Why this fork exists

Two problems that don't have a one-line answer:

1. **Speculative decoding lives at batch 3–8.** Every method — DFlash2, MTP,
   DSpark, ngram — verifies several drafted tokens in one batch. But the Vulkan
   mat-vec kernels fall apart in exactly that range: a stock upstream IQ3_S
   shader overflows the VGPR budget at `NUM_COLS > 4` and is **5× slower** at
   the width the method needs. The fix is a register-spill elimination in a
   shader that upstream has, but nobody noticed because single-token decode
   never reaches that width.

2. **FP4 / FPx weights beat K-quants at batch 1 and lost at batch 8.** Two
   shader changes — amortising the UE4M3 scale decode over a whole 32-weight
   block, and replacing the per-weight bit-window gather with a branch-free
   one — fixed the batching. Mainline cannot load these models at all, so the
   question of making them fast doesn't arise upstream.

The one-constant LDS stride fix is the surprise: a four-way bank conflict in
the coopmat matmul that costs 17–18 % at the kernel level and 12–14 % at the
benchmark level, on **every non-Intel device**, with **any quant**. It's
upstreamable and is the single biggest win in this fork.

## What's in the fork

| Component | Status |
|---|---|
| **Adaptive speculation** -- `--spec-draft-adaptive`, MTP or DFlash2 | New here. Draft length tracks accepted tokens; `n_max` is a ceiling. Recommended: `--spec-draft-adaptive --spec-draft-n-min 3` |
| **ROCmFPx quant types** — Q4\_0\_ROCMFP4, \_FAST, Q2/Q3/Q6/Q8\_0\_ROCMFPX, CPU codecs + Vulkan dequant / mat-vec / matmul / integer-dot kernels | Hand-ported from ciru-ai/ROCmFPX. Two decode bugs fixed here — see `ROCMFPX-NOTES.md` |
| **Vulkan batched mat-vec fixes** — IQ3\_S register spill at `NUM_COLS > 4` (5× at n=8), ROCmFPx batch 3–8 rework | New here. Both upstreamable |
| **Vulkan LDS stride fix** — `SHMEM_STRIDE` pad, +7.3 % prefill on any quant, +10–20 % at the kernel | New here. One constant. Upstreamable. Now driver-gated, see below |
| **Vulkan prefill gates** — tiled concat-transpose, f16 B operand for quantized matmul and matmul\_id | Ported from [Nathanw1014/llama.cpp](https://github.com/Nathanw1014/llama.cpp/tree/strix-halo-vulkan). Re-measured here; see `WORKLOG.local.md` |

## Against mainline — Vulkan only, stock K-quants

Same CMake flags, clean worktree, interleaved on an idle GPU. No ROCmFPx,
no fork-specific model format — this measures the Vulkan work alone.

`pp2048` at ubatch 512 with flash attention on, three repetitions per cell, arms in palindrome
order (fork · mainline · mainline · fork) so clock drift cancels between them rather than being
charged to one, and a discarded warmup per process because this APU gives the first run of a set a
15–20 % boost clock.

| model | metric | depth 0 | 4 K | 16 K | 32 K | 64 K |
|---|---|---|---|---|---|---|
| unsloth Qwen3.8-27B-UD-Q4\_K\_XL (dense) | pp2048 | **+13.0 %** | +11.6 % | +9.1 % | +8.2 % | +4.6 % |
| | tg64 | −0.7 % | −0.7 % | −0.4 % | −0.6 % | −0.5 % |
| bartowski Ornith-1.5-35B-A3B-Q4\_K\_M (MoE) | pp2048 | **+12.6 %** | +10.0 % | +5.9 % | +5.1 % | noise |
| | tg64 | noise | +1.5 % | +0.7 % | noise | noise |

Absolute t/s and the per-cell spread are in [`bench/RESULTS.md`](bench/RESULTS.md); "noise" marks a
cell under 2σ.

**The gain decays with depth.** This fork's Vulkan work is in the matmul and concat paths; as
context grows, attention takes a larger share of prefill and those paths take a smaller one. A
single headline percentage would hide that, which is why depth is an axis here and not a footnote.

**Generation is flat on stock K-quants** — a consistent −0.6 % on the dense model, small positive on
the MoE. That is the honest result for this configuration: no speculation, no ROCmFPx, so none of
the paths this fork actually optimises for generation are in play. See the speculative table above
for the case that is.

The MoE now gains as much as the dense model at short context, where an earlier measurement of this
same pair showed only +3.2 %. The difference is the tiled concat-transpose kernel, worth **+45 %**
on its own for delta-net MoE prefill — the generic concat walks the transposed conv-state with a
40960-byte stride, so every read lands on the same one of 16 memory channels.
Generation is unchanged: single-stream decode of a stock K-quant is already at
80 % of theoretical memory bandwidth on this APU — no kernel can move it.
The generation headroom is at the batch widths speculative decoding runs at,
which is where this fork works.

**The fix in one sentence.** `mul_mm.comp` stages shared tiles at
`SHMEM_STRIDE = BK/2 + PAD` in dword units (1:1 with the 32 LDS banks) and
reads B back column-major, so `gcd(SHMEM_STRIDE, 32)` is the conflict factor.
Upstream sets `PAD` only for Intel-on-Windows; everything else takes the
shader default of 4, giving stride 20 → four-way conflict on every B load.
Stride must stay even (an odd stride breaks RADV's wide `ds_read_b64/b128`
path and costs 3×), so any pad with `gcd(BK/2 + pad, 32) == 2` is the fix.
Worth 1.17× on q4\_0 and 1.18× on q8\_0 at the kernel level.

**Why it is driver-gated.** The coopmat path hands `SHMEM_STRIDE` to
`coopMatLoad` as its Stride operand, and the spec wants that 16-byte aligned for
16×16 f16 tiles. Stride bytes are `(BK/2 + pad) * 4`, so only `pad % 4 == 0` is
in contract — every conflict-free pad is out of it. RADV before 25.3 lowers
`coopMatLoad` to `ds_read_b128`, which the contract entitles it to, and the
misaligned rows then pay runtime splits: pp512 collapses by more than 2×. RADV
25.3 and later lowers to `ds_read_b64`, which the stride always suits, and the
bank spread wins. So the fix applies **only on RADV ≥ 25.3**; everything else
keeps the spec-aligned pad 4. `GGML_VK_SHMEM_PAD=N` overrides both, for probing.

## Speculative decoding at the batch widths that matter

DFlash2, MTP, DSpark, and Eagle3 all landed upstream. Having the method is not
the same as having it work. Speculation verifies several drafted tokens in one
batch, so it lives at **batch 3–8** — exactly the range where the Vulkan
mat-vec kernels were broken:

**IQ3\_S mat-vec, m=4096 k=14336, µs/run — lower is better.**
n is the batch width (how many drafted tokens are verified at once).

| n | mainline | this fork | |
|---|---|---|---|
| 1 | 98.3 | 98.0 | parity |
| 2 | 110.7 | 114.0 | −3 % |
| 4 | 167.4 | 172.9 | −3 % |
| **8** | **1542.0** | **285.0** | **5.4×** |

The 3 % at n=2 and n=4 is a real cost, not noise: the fix moves `dscale`
inside the inner loop for every `NUM_COLS`, so the narrow cases pay slightly
for the wide ones. Trading 3 % at n=2 for 5.4× at n=8 is the right side of
that bargain when the whole point is verifying batches of drafts.

**The rest, which mainline has no equivalent for:**

| | mainline | this fork |
|---|---|---|
| ROCmFPx with DFlash2 | cannot load the model at all | 6 quant types, CPU + Vulkan |
| ROCmFPx mat-vec at batch 3–8 | n/a | 312 → 173 µs (fp4), 2236 → 402 (fp6) |
| GDN output projection, multi-slot | one GEMV per sequence | one batch-wide GEMV, +8.6 % at B=8 |

## Measured results

### Bare decode is at the memory wall

| model | size | tg128 | achieved | of 256 GB/s peak |
|---|---|---|---|---|
| ROCmFP4\_FAST | 13.55 GiB | 14.05 ± 0.09 | 204.4 GB/s | 79.8 % |
| Q3\_K\_M | 12.56 GiB | 15.18 ± 0.08 | 204.7 GB/s | 80.0 % |

Two decoders with nothing in common — an FP4 codebook with UE4M3 scales, and
Q3\_K superblocks — reach the same bandwidth to three significant figures.
Single-stream decode has no headroom left. Every remaining gain has to come
from draft acceptance, which multiplies effective bandwidth instead of
competing for it. This is a draft-quality problem, not a kernel problem.

### Speculative decoding, tokens/s

**Short context (~350 tokens):**

| config | prose | code | JSON |
|---|---|---|---|
| bare | 13.90 | 13.90 | 13.89 |
| MTP n=3 | 26.18 | 29.83 | 34.91 |
| DFlash2 · z-lab Q8\_0 n=7 | 21.96 | 35.24 | **41.54** |

**Long context (~31 K tokens of real C source):**

| config | verbatim reproduction | prose about the code |
|---|---|---|
| bare | 12.20 | 12.19 |
| MTP n=4 | **36.07** (97.5 % acc) | 20.54 |
| DFlash2 · Q8\_0 n=7 | 31.79 | 15.69 |
| ngram-simple (no draft) | 25.41 | 11.72 (below bare) |

The ranking inverts with context length. DFlash2 wins at short context —
41.5 t/s on JSON, **3.0× bare**. By 31 K, MTP takes both tasks. The cause is
structural: a DFlash2 sidecar keeps its own KV cache over the full context and
re-runs up to seven times per verification step, so its cost scales with
context. MTP's nextn layer reuses the target's state and never pays that.

Content dominates configuration: identical weights at identical context span
36.07 to 20.54 t/s purely on whether the output is quotable.

### What to run

| situation | method |
|---|---|
| short context, structured output | `--spec-type draft-dflash --spec-draft-adaptive --spec-draft-n-min 3`, z-lab Q8\_0 sidecar, `--spec-draft-n-max 7` |
| long context, any task | `--spec-type draft-mtp --spec-draft-adaptive --spec-draft-n-min 3` at n-max 3–4 — no sidecar, no second KV cache |
| quote-heavy long context | ngram-simple alone, never layered under a draft model (costs 13 %) |
| **avoid** | ngram-cache (slower than bare), Q2\_0\_ROCMFPX (no Vulkan kernel), DSpark on this target |

Do not requantise the z-lab Q8\_0 sidecar. Our Q8\_0\_ROCMFPX scores 53.5 %
acceptance against z-lab's 60.2 % at the same bpw and identical tensor
routing — it even lands below our own FP4, which is impossible as a
precision effect. The cause is the block scale: Q8\_0 stores an fp16 scale,
ROCmFPx stores a UE4M3 byte. At 8 bits per weight the codes are not the
problem, the coarse scale is.

## Quick start

```bash
cmake -B build -DGGML_VULKAN=ON && cmake --build build -j

# Short context, structured output - DFlash2, adaptive
build/bin/llama cli \
  -hf  lmcoleman/Qwen3.8-27B-ROCmFPX-GGUF:Q4 \
  -hfd incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M \
  --spec-type draft-dflash --spec-draft-adaptive --spec-draft-n-min 3 \
  --spec-draft-n-max 7 -ngl 99 -fa on

# Long context - MTP, adaptive (no sidecar needed)
build/bin/llama cli \
  -hf  lmcoleman/Qwen3.8-27B-ROCmFPX-GGUF:Q4 \
  --spec-type draft-mtp --spec-draft-adaptive --spec-draft-n-min 3 \
  --spec-draft-n-max 4 -ngl 99 -fa on


---

*Everything below is the upstream llama.cpp README. Its badges and links point at
`ggml-org/llama.cpp`, not at this fork.*

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
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

Fork-specific, on top of upstream llama.cpp:

- [ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX) - the ROCmFPx quant formats and reference
  codecs. Hand-ported here (that tree shares no git history with llama.cpp, so it cannot be merged).
- [Jian Chen](https://github.com/ggml-org/llama.cpp/pull/27342) - DFlash2 speculative decoding,
  PR #27342. This fork carried it before it merged upstream.
- [Nathanw1014/llama.cpp](https://github.com/Nathanw1014/llama.cpp/tree/strix-halo-vulkan) - the
  Strix Halo Vulkan branch this fork's prefill gates and several correctness fixes were ported from.
  Every one was re-measured here before adopting, and two of their defaults are disabled on this
  hardware because we could not reproduce the gain.

Upstream llama.cpp acknowledgements:

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
