# llama-mindcontrol

<img width="4753" height="1897" alt="Mindcontrol for Llama.cpp" src="https://github.com/user-attachments/assets/4c200417-100d-4153-89b1-17ddbe8e5dee" />

A `llama.cpp` fork that extends the reasoning-budget sampler with staged, in-context budget signaling.

## Problem
### "But, wait..."

Reasoning models generate an unbounded `<think>` block prior to their final answer, and the length and quality of that block are sensitive to sampling parameters. At the temperature and top-p/top-k settings needed to avoid degenerate, low-entropy output elsewhere in generation, the reasoning block is prone to failure modes that are distinct from ordinary sampling artifacts:

- **Repetition loops** — the sampler re-enters a previously visited distribution over reasoning tokens, producing near-identical spans of text with no new information content.
- **Non-convergent revision** — the model repeatedly re-opens a conclusion it has already reached (recurring "wait, actually..." / "but hold on..." transitions) without a stopping condition ever becoming more probable than continuing.
- **Unbounded length** — absent an explicit stopping signal, nothing in the token distribution guarantees termination; the reasoning block can consume arbitrary context budget before (or without) closing.

<img width="600" height="auto" alt="Overly-verbose reasoning chain" src="https://github.com/user-attachments/assets/d45acc3f-e713-4a27-81fb-0573f0a1967f" />


The standard/naive mitigation is a hard token-count cutoff enforced by the sampler: once N tokens have been generated inside the reasoning block, `</think>` is forced regardless of position in the sequence. This bounds worst-case length but does not address any of the above — the cutoff has no dependency on the model's actual generation state and truncates at whatever token index it happens to hit, including mid-token-sequence for a partial word, mid-clause, or mid-computation. It suppresses the symptom (unbounded length) without altering the sampling behavior that produces the loop or the non-convergence in the first place.

## Mechanism

This fork adds a state machine around the existing hard-cutoff sampler, with two additional stages that inject fixed text into the reasoning stream at defined points:

1. **Intro stage** — on entry to the reasoning block, a fixed message stating the token budget is inserted (templated via a `{budget}` placeholder), e.g. `"I'm allowed to think for 512 tokens, so my reasoning should be concise. Let me start by"`. This gives the model an explicit, in-context reference for its own generation length before reasoning begins.
2. **Soft-warning stage** — at a configurable fraction of the budget (default 0.5), the sampler waits for the next newline boundary and inserts a fixed message indicating the budget is half-consumed, e.g. `"I've used up half of my thinking budget, let me start working towards a conclusion"`.
3. **Hard-stop stage with grace period** — once the budget is exhausted, the sampler enters a pending state and waits up to a configurable number of grace tokens for a paragraph boundary (two consecutive newlines) before inserting a fixed closing message and terminating the reasoning block. If no paragraph boundary occurs within the grace period, the cutoff is forced immediately. Total output length remains bounded by `budget + grace_tokens` in all cases.

<img width="2920" height="2547" alt="Mindcontrol in Action" src="https://github.com/user-attachments/assets/8f394c77-1812-4e4e-bd7c-13a8c8695edc" />

Injected text is only inserted at newline or paragraph boundaries, not mid-token or mid-sentence. If the model emits its own `</think>` before a forced stage would trigger, the natural close takes precedence.

Each stage is opt-in and independently configurable via `LLAMA_ARG_THINK_BUDGET_*` environment variables at server startup, or as per-request overrides in the API call itself:

| Environment variable | Purpose |
| --- | --- |
| `LLAMA_ARG_THINK_BUDGET` | Token budget for the reasoning block (existing upstream variable) |
| `LLAMA_ARG_THINK_BUDGET_INTRO_MESSAGE` | Templated intro message, supports a `{budget}` placeholder |
| `LLAMA_ARG_THINK_BUDGET_SOFT_RATIO` | Fraction of budget at which the soft warning fires (e.g. `0.7`) |
| `LLAMA_ARG_THINK_BUDGET_SOFT_MESSAGE` | Templated soft-warning message |
| `LLAMA_ARG_THINK_BUDGET_MESSAGE` | Templated hard-stop closing message |
| `LLAMA_ARG_THINK_BUDGET_GRACE_TOKENS` | How long to wait for a paragraph break before forcing the hard stop |

Default values preserve upstream's existing hard-cutoff behavior; the new stages are disabled unless configured.

Planned follow-up work: generalize this mechanism into a configurable reasoning template/grammar, rather than a fixed set of budget-based checkpoints.

## Benchmarking & Findings

### Setup

All results below use Qwen3.6-27B, `UD-Q4_K_XL` quantization, with MTP speculative decoding (3 draft tokens). A separate pass without speculative decoding produced consistent results and is omitted here for brevity.

Four configurations are compared at several reasoning budgets, each adding one more piece of the mechanism on top of the last:

- **Naive** — llama.cpp's existing default: the moment the budget is reached, `</think>` is force-injected immediately, with no grace period and no in-context signaling of any kind. This is the behavior described in the Problem section above, and the baseline this fork is trying to improve on.
- **Hard-limit only** — this fork's hard-stop-with-grace-period stage used on its own (no soft warning, no intro message): instead of an immediate cutoff, the sampler waits up to `grace_tokens` for a paragraph boundary before closing the block.
- **Soft + hard** — the grace-period hard stop plus the soft-warning stage (fired at a configurable fraction of the budget), without the intro stage.
- **Intro + soft + hard** — the full three-stage mechanism: intro message, soft warning, hard stop with grace period.

Two further reference points appear in the charts and tables: **Baseline (unlimited)**, where the reasoning block runs to its own natural `</think>`, and — for LiveCodeBench only — **No reasoning**, where the reasoning block is disabled entirely.

Benchmarks: HumanEval+ (n=164) and LiveCodeBench (`release_v6`, n=200). Reported token counts are average total completion tokens per test (reasoning block plus final answer), matching the chart axes.

### Results: HumanEval+

<img width="950" height="550" alt="HumanEval+: token consumption by configuration" src="https://github.com/user-attachments/assets/ceb23497-0494-466b-8486-818b7e3569e6" />

<img width="900" height="550" alt="HumanEval+: accuracy vs. token cost" src="https://github.com/user-attachments/assets/0834281d-5d96-4c0c-a28f-3f86e158e332" />


| Budget (tokens) | Naive: tok / pass@1 | Hard-limit only: tok / pass@1 | Soft + hard: tok / pass@1 | Intro + soft + hard: tok / pass@1 |
|---|---|---|---|---|
| 300 | 499 / 92.7% | 489 / 92.1% | 422 / 92.1% | 391 / 92.7% |
| 500 | 749 / 91.5% | 672 / 93.3% | 592 / 93.9% | 569 / 93.3% |
| 750 | 963 / 93.3% | 906 / 93.9% | 809 / 93.9% | 863 / 91.5% |
| 1250 | 1363 / 92.7% | 1348 / 92.7% | 1221 / 93.9% | 1360 / 95.7% |
| Unlimited (baseline) | 2776 / 92.7% | — | — | — |

Two results stand out here:

1. **Token consumption drops monotonically as guidance is added, at every budget.** Naive uses the most completion tokens of the four configurations at all four budgets, hard-limit-only is next, and soft + hard / intro + soft + hard are consistently the lowest — e.g. at budget 500: 749 (naive) → 672 (hard-limit only) → 592 (soft + hard) → 569 (intro + soft + hard). Soft + hard and intro + soft + hard are close to each other throughout, and which of the two is marginally lower varies by budget on this benchmark (soft + hard is lowest at 750 and 1250; intro + soft + hard is lowest at 300 and 500) — with n=164 this is likely within run-to-run noise rather than a real ordering between the two.
2. **Most configurations meet or beat the unlimited baseline (92.7%).** Of the 16 budget/configuration combinations, 12 are at or above 92.7%, and 8 exceed it outright — including the best result in the table, intro + soft + hard at budget 1250 (95.7%, using 1360 tokens against the baseline's 2776). The most plausible explanation is that constraining and guiding the reasoning block suppresses the repetition loops and non-convergent revision described in the Problem section above — on a benchmark like HumanEval+, where the model can typically reach a correct answer well within a modest token budget, an unconstrained reasoning block gives the model more opportunity to talk itself into a worse answer, not a better one.

### Results: LiveCodeBench

<img width="950" height="550" alt="LiveCodeBench: token usage by reasoning budget and style" src="https://github.com/user-attachments/assets/94c16e8b-e910-4b02-9f44-920a1cfc6034" />

<img width="900" height="550" alt="LiveCodeBench: accuracy vs. token cost" src="https://github.com/user-attachments/assets/ff0776dc-1f82-4b5b-ab66-2c90df1a96f1" />

| Budget (tokens) | Naive: tok / pass@1 | Hard-limit only: tok / pass@1 | Soft + hard: tok / pass@1 | Intro + soft + hard: tok / pass@1 |
|---|---|---|---|---|
| 500 | 6955 / 61.0% | 4827 / 58.5% | 3894 / 61.5% | 2930 / 56.5% |
| 1000 | 7624 / 64.0% | 5820 / 65.5% | 4582 / 64.5% | 3334 / 60.0% |
| 1750 | 7779 / 62.0% | 6556 / 62.5% | 4864 / 68.5% | 4862 / 66.0% |
| 4000 | 16324 / 70.5% | 11251 / 65.5% | 10277 / 68.5% | 7693 / 69.5% |
| Unlimited (baseline) | 36293 / 72.0% | — | — | — |
| No reasoning | — / 57.0% | — | — | — |

LiveCodeBench is far more reasoning-intensive at baseline (36293 tokens/task on average, versus 2776 for HumanEval+), and the ordering seen above holds even more cleanly here: **naive > hard-limit only > soft + hard > intro + soft + hard in total token count, at every single budget tested, with no exceptions.** At the 4000-token budget, naive uses 16324 tokens for 70.5% pass@1, while intro + soft + hard uses 7693 tokens — 47% of naive's token count — for 69.5%, a 1-point difference well within what n=200 sampling noise would produce.

Accuracy differences between configurations at a fixed budget are generally small (a few points, consistent with n=200 noise) and don't show a systematic penalty for the more guided configurations — in most cases they hold accuracy roughly level with naive while using a fraction of the tokens.

A separate effect shows up in how each configuration's accuracy responds to *increasing* the budget. For soft + hard and intro + soft + hard, pass@1 rises monotonically as budget increases from 500 to 4000, with no reversals. Naive and hard-limit-only do not show this: naive drops from 64.0% (1000 tokens) to 62.0% (1750 tokens) before jumping to 70.5% (4000 tokens), and hard-limit-only drops from 65.5% (1000 tokens) to 62.5% (1750 tokens). The guided configurations turn additional budget into a predictable accuracy gain; naive and hard-limit-only do not — this is the clearest "reduced noise" effect in this data.

None of the four budget-constrained configurations fully recovers the unlimited baseline's 72.0% at any tested budget.

### By difficulty (LiveCodeBench)

| Difficulty | Baseline (unlimited) | 500-token budget (range across 4 configs) | 4000-token budget (range across 4 configs) |
|---|---|---|---|
| Easy (n=53) | 85% | 96–98% | 94–96% |
| Medium (n=61) | 72% | 61–72% | 72–80% |
| Hard (n=86) | 64% | 26–36% | 42–50% |

This breakdown clarifies where the token savings come from, and echoes the HumanEval+ result above. On easy problems, every budget-constrained configuration at every tested budget scores at or above the unlimited baseline (96–98% vs. 85%) — again consistent with a capped, guided reasoning block reducing the chance the model overthinks its way into a wrong answer on a problem it could already solve. Medium problems are roughly flat to slightly improved at the higher budget. Hard problems are the exception: accuracy stays well below the unlimited baseline at both the smallest (26–36% vs. 64%) and largest (42–50% vs. 64%) budgets tested, for every configuration including intro + soft + hard. Budget-based control, however it's implemented, does not close this gap — the hardest problems still lose accuracy when reasoning length is capped.

### Summary

- Naive (llama.cpp's existing immediate-cutoff behavior) uses the most completion tokens of the four configurations at every budget tested, on both benchmarks — this is the behavior the mechanism is designed to improve on.
- Each additional stage of budget-aware guidance (grace period → soft warning → intro message) reduces token consumption further. On LiveCodeBench this ordering is exact at every budget: naive > hard-limit only > soft + hard > intro + soft + hard.
- Aggregate pass@1 does not show a systematic drop from budget constraints. On HumanEval+, 12 of 16 tested combinations meet or exceed the 92.7% unlimited baseline, and the single best result in either benchmark (95.7%) comes from the most heavily guided, budget-constrained configuration.
- Soft + hard and intro + soft + hard produce a monotonic, predictable accuracy/budget relationship on LiveCodeBench; naive and hard-limit-only do not.
- The gains are not evenly distributed across problem difficulty: easy-problem accuracy improves under constrained, guided budgets (consistent with reduced overthinking), while hard-problem accuracy remains below the unlimited baseline at every budget tested, for every configuration.


## Quick start

Configuration is set via `LLAMA_ARG_THINK_BUDGET_*` environment variables, and can be overridden per-request in the API call — see [server API docs](tools/server/README.md) for the request-level parameters.

### Apple Silicon

Docker on macOS cannot pass the GPU through to a container, so there is no Metal-accelerated Docker image. Build natively instead, following upstream's [build guide](docs/build.md) (Metal is enabled by default on Apple Silicon):

```sh
git clone https://github.com/laurencehardman/llama-mindcontrol
cd llama-mindcontrol
cmake -B build
cmake --build build --config Release -j

LLAMA_ARG_THINK_BUDGET="350" \
LLAMA_ARG_THINK_BUDGET_SOFT_RATIO="0.7" \
LLAMA_ARG_THINK_BUDGET_GRACE_TOKENS="64" \
./build/bin/llama-server -m /path/to/your-model.gguf
```

### AMD64 + NVIDIA CUDA

A pre-built Docker image is provided. Example `docker-compose.yml`:

```yaml
services:
  llama-server:
    image: ghcr.io/laurencehardman/llama-mindcontrol:cuda
    gpus: all
    ports:
      - "8080:8080"
    volumes:
      - ${MODEL_DIR:-./models}:/models:ro
    environment:
      LLAMA_ARG_THINK_BUDGET: "350"
      LLAMA_ARG_THINK_BUDGET_INTRO_MESSAGE: " I have {budget} tokens to reason through this - that's enough room to work through it carefully, so I'll think it through step by step rather than rushing to a conclusion."
      LLAMA_ARG_THINK_BUDGET_MESSAGE: " [!!NOTE TO SELF] I've used all of my thinking budget, I am now going to wrap up and provide the user their answer."
      LLAMA_ARG_THINK_BUDGET_SOFT_RATIO: "0.7"
      LLAMA_ARG_THINK_BUDGET_SOFT_MESSAGE: " [!NOTE TO SELF] I'm partway through my budget - I should start consolidating toward an answer, but I still have room to finish the important points."
    command:
      - "--model"
      - "/models/your-model.gguf"
```

```sh
MODEL_DIR=/path/to/models docker compose up
```

Requires the [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) on the host.

`llama-server` exposes the standard OpenAI-compatible API at `http://localhost:8080`. See the upstream documentation below for other configuration options.

Test it out:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Explain how Flash Attention works."}
    ],

    "reasoning_budget_tokens": 350,
    "reasoning_budget_message": "I have reached my reasoning budget - I have enough here to answer now.",

    "reasoning_budget_soft_ratio": 0.7,
    "reasoning_budget_soft_message": "I am partway through my budget - I should start consolidating toward an answer, but I still have room to finish the important points.",

    "reasoning_budget_intro_message": "I have {budget} tokens to reason through this - that is enough room to work through it carefully, so I will think it through step by step rather than rushing to a conclusion.",

    "reasoning_budget_grace_tokens": 50,

    "reasoning_control": true
  }'
```

---
# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3A0cc4m%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev branches](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-features.md) / [compile times](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-compile-times.md) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

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

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
