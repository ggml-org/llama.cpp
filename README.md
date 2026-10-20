# Adaptive KV Streaming for llama.cpp V2

Quantized Qwen3.8-27B and Gemma 4 31B weights can fit on a 16 GB GPU, but that leaves limited space for a long conversation's KV cache. As context grows, the cache can exceed the remaining VRAM. Unified Memory allows oversubscription, but repeated movement between RAM and VRAM can sharply reduce decoding speed.

Adaptive KV Streaming V2 manages that movement explicitly. It keeps the complete encoded full-attention KV history in system RAM, retains part of it in VRAM, and prefetches the rest through a shared ring buffer as attention needs it. The same GPU space is reused across layers, allowing full attention to cover history beyond what fits in VRAM without discarding earlier tokens. Attention still runs on the GPU; UVM is optional.

To make long-context generation faster, V2 also shares memory between prefill, decode and drafting, plus vision on Qwen. Qwen's embedded MTP streams an auxiliary history; Gemma's separate MTP assistant borrows target KV without creating another history. DFlash2 provides another Qwen option, with different memory costs and acceptance rates.

On an RTX 5070 Ti with IQ4_XS weights and Q8_0 K / Q4_0 V, stock llama.cpp with UVM measured 2.5 tok/s at 128 Ki, while V2 target-only measured 26.1 tok/s. MTP=3 increased that to 59.7 tok/s at the same context and still delivered 22.3 tok/s at the model's native 262,144-position capacity. Stock was slightly faster at short contexts; streaming's benefit becomes clear as the cache puts pressure on VRAM.

![IQ4_XS Q8/Q4 throughput: four V2 modes through 256 Ki and stock UVM without streaming through 128 Ki](media/adaptive-kv-four-mode-throughput.png)

The colored lines are the 128-point V2 sweep with UVM disabled and independently maximum-fit arenas. The gray line is a fresh 16-point stock-UVM sweep, without KV streaming, an arena or speculation, ending at 128 Ki. Both use the same IQ4_XS model, Q8/Q4 KV, corpus, context-matched prompts, 256 output tokens and b/ub=256. These are separate recorded sweeps, not repeated statistical A/B trials; results depend on the workload and hardware. [V2 details](#q8q4-four-mode-context-scaling) · [V2 CSV](media/adaptive-kv-four-mode-comparison.csv) · [Stock-UVM CSV](media/stock-uvm-iq4-throughput.csv) · [V2 project discussion](https://www.reddit.com/r/LocalLLM/comments/1wt03zh/adaptive_kvcache_streaming_v2_full_context_mtp/).

Gemma 4 31B now uses the same streaming infrastructure while keeping its sliding-window caches resident. In the new 60-point Q3_K_M/Q8-Q4 sweep, MTP=3 measures 91.2 tok/s at 8 Ki and 28.3 tok/s at 120 Ki, versus 39.6 and 10.3 target-only. These are separately fitted arenas, not equal-memory comparisons or a stock-UVM baseline.

![Gemma 4 target-only and MTP1-3 prefill/decode through 120 Ki](media/gemma4-120ki-throughput.png)

[Gemma measurements and scope](#gemma-4-through-120-ki) cover all four modes at every 8 Ki step. The shared prefill path also reuses bounded KV conversion and overlaps transfers with attention; [repeated A/B measurements](#common-prefill-improvements) separate those gains from the context sweeps.

The implementation supports **one serial Qwen3.8-27B or Gemma 4 31B request on one CUDA GPU**. Supported combinations and tested limits are listed below. Ordinary upstream modes remain available without adaptive execution. The Qwen figure above is its earlier recorded cohort, not a rerun with the newer prefill implementation.

[Supported features](#supported-features) · [Quickstart](#quickstart) · [Performance](#performance-and-reproduction) · [How it works](#how-it-works) · [Developer guide](#developer-guide) · [Upstream README](#upstream-llamacpp-readme)

## Supported features

### Models and execution modes

The reference targets are Unsloth Qwen3.8-27B `UD-IQ4_XS` and Gemma 4 31B `Q3_K_M`, with a matching `Q8_0` Gemma assistant. Weight encoding and KV-cache encoding are separate settings. Other weight encodings use llama.cpp's readers but need their own memory-fit and inference qualification; earlier Qwen experiments also used Q3_K_XL, Q4_K_S and NVFP4.

| Target / drafter | Context capacity | Draft length | Draft KV history | Image input |
| --- | --- | --- | --- | --- |
| Qwen3.8-27B, target-only | Up to 262,144 positions | None | None | Matching F16 mmproj supported |
| Qwen3.8-27B, embedded MTP | Up to 262,144 positions | 1-3 predictions; target verification through TG4 | Full history, streamed as an auxiliary KV layer | Supported with the matching mmproj and shared arena |
| Qwen3.8-27B + IncoAI DFlash2 sidecar | Up to 262,144 positions | 1-7 predictions; target verification through TG8 | Bounded 2,048-position window plus execution headroom | Not supported with the vision arena |
| Gemma 4 31B, target-only | 8-120 Ki full-workload sweep; native metadata capacity 262,144 | None | Ten full-attention histories streamed; fifty bounded SWA histories resident | Not qualified |
| Gemma 4 31B + matching MTP assistant | 8-120 Ki full-workload sweep, all lengths 1-3 | 1-3 predictions; target verification TG1-TG4 | Read-only target KV; no assistant KV append/catch-up | Not qualified |
| Other Gemma variants, Muse Glimmer and other architectures | Not qualified here | Not qualified | Need model topology, admission and numerical qualification | Not qualified |

TG1 means one query token in an attention invocation; TG4/TG8 mean four/eight queries. These are not request-concurrency settings. Context capacity includes the prompt, image positions, generated tokens and speculative headroom; it is not a promise of a 262,144-token prompt followed by additional generation.

Qwen embedded MTP needs a target GGUF containing its MTP tensors; its architecture is represented by `qwen35` in the source/GGUF metadata. Gemma requires the separate matching assistant, passed with `--model-draft`; its roughly 475.81 MiB of GPU weights remain **outside the arena**. DFlash2 requires its own matching Qwen sidecar, not an arbitrary draft model and not a Gemma pairing. See the pinned [Gemma artifacts](benchmarks/data/gemma4-31b-artifacts.json) and [Qwen DFlash2 artifacts](benchmarks/data/dflash2-qwen27b-artifacts.json).

Gemma's 120 Ki endpoint is the completed qualification scope, not full-256-Ki prefill qualification. Native controls are separate fitting cases: the 16 Ki native-MTP attempt ran out of VRAM while loading assistant weights, before inference. Long streamed lifecycle tests also cover a 98,304-token prompt and greater-than-4-GiB saved state; that is not long-context native numerical equivalence. See the [current Gemma ledger](GEMMA4_ADAPTIVE_KV_ROADMAP.md).

### KV-cache formats

| K and V choices | Combination support | Qualification / caveat |
| --- | --- | --- |
| `f32`, `f16`, `bf16`, `q8_0`, `q5_0`, `q5_1`, `q4_0`, `q4_1` | All 64 ordered pairs, subject to compiled path, geometry and memory admission | Qwen controls plus Gemma D512 span/tile/progressive native-bit oracles; this is not a full-model performance sweep of all 64 pairs |
| Q8_0/Q4_0, Q8_0/Q8_0, Q5_1/Q5_1 | Qwen target-only/MTP3/DFlash7 and Gemma target-only/MTP1-3 | Qwen's 45-point capacity matrix; Gemma's fitting native/streamed server controls across all four modes |
| Gemma long-context measurements | Q8_0 K / Q4_0 V | Full 60-point performance sweep through 120 Ki, plus lifecycle/recovery qualification; other pairs need their own long-context fit |
| `iq4_nl` | Not admitted by current CUDA streamed attention | A writer exists, but the current stock CUDA Flash Attention selector has no reader; an all-quants build cannot add that missing reader |
| Image requests | Q8_0 K / Q4_0 V is the image-inference reference | Other pairs pass shared-parent vision/MTP handoff checks, not complete image-inference qualification |

Adaptive MTP and DFlash inherit the target's K/V formats. **Do not pass draft-cache overrides**, even identical values: `--spec-draft-type-k/v` and their aliases are rejected for adaptive execution. Native non-streaming drafting retains independent cache settings.

F32 describes storage, not fully FP32 attention: the CUDA reader follows stock attention's FP16 input rounding. Cache-format speed measurements are not model-quality measurements.

### Platforms and operational limits

| Platform / backend | Implemented or tested here | What this does not establish |
| --- | --- | --- |
| Linux + CUDA, RTX 5070 Ti / SM120 | Qwen text/MTP/DFlash2/vision and Gemma text/shared-KV MTP, cache/recovery, numerical and performance checks | Other cards, drivers and workloads can have different memory limits; Gemma multimodal is not qualified |
| CUDA SM61/SM70/SM75/SM86 | Gemma D512 affected-unit functional compilation: 19 logical units, 72 actual device images; changed prefill adapters rechecked | Not a full legacy server link or physical Pascal/Volta/Turing/Ampere runtime/performance test |
| Other CUDA targets, including SM89 | Earlier Qwen selected-target/forward-JIT evidence is recorded separately | Forward-JIT on the 5070 Ti does not reproduce older-card resource/performance limits |
| Windows + CUDA | The CUDA path and DLL export fixes are in the source | The recorded Linux matrix is not Windows/MSVC qualification |
| CPU | Buffer views, leases, allocators, planning, transitions and host tests | Not a GPU streaming adapter |
| OpenCL / SYCL | Current builds pass; this machine lacks a compatible GPU/runtime for the Gemma qualification | Compile-only evidence is not streamed inference qualification |
| Vulkan / Meta composition | Actual common memory/view/completion/metadata tests | Not complete Gemma or Qwen streamed-attention inference |
| HIP / ROCm | Shared CUDA/HIP buffer-view implementation | No end-to-end AMD streaming qualification |

For SM61/SM70 builds, use a toolkit that still compiles those targets, such as CUDA 12.9; CUDA 13 cannot compile them. For older-target evidence and exact device-check commands, see [CUDA qualification](docs/build.md#adaptive-kv-cuda-qualification-and-startup-errors). Memory API portability does not make CUDA kernels portable.

| Setting | Current behavior |
| --- | --- |
| Requests / GPUs | One serial request, one CUDA GPU; no parallel slots or multi-GPU split |
| Attention | Flash Attention and GPU KV offload required |
| Batch / microbatch | 256/256 is the full text-matrix reference; other settings need their own fit test and backend admission |
| Vision | Qwen only: image-only, matching F16 projector on GPU, shared arena, no LoRA/embeddings/separate drafter |
| Image batching | Multiple images are supported within the admitted vision-batch and context limits; there is no universal image-count or resolution guarantee |
| UVM | Optional for model weights; the shared KV/phase arena stays device-local |
| Other GPU applications | Managed weights may migrate under pressure; the device-local arena itself is not automatically pageable |

The arena bounds its participating allocations, **not total process memory**. Model weights, host KV, persistent recurrent data and driver/native-executable allocations also need memory. Startup admission is not a guarantee against every later OOM.

## Quickstart

### Build

Build this fork rather than an upstream release binary. From a checkout of the Gemma-enabled `feature/gemma4-adaptive-kv-stream` branch/revision (the older V2 branch alone does not include these Gemma changes):

```sh
cmake -S . -B build-v2 -DGGML_CUDA=ON -DGGML_CUDA_FA_QUANTS=all -DCMAKE_BUILD_TYPE=Release
cmake --build build-v2 --target llama-server --parallel 2
```

Set `-DCMAKE_CUDA_ARCHITECTURES=...` for your GPU if needed. Increase build parallelism only when host RAM permits: the wide CUDA translation units can be memory-heavy, especially for legacy targets. Selected-pair builds can use, for example, `-DGGML_CUDA_FA_QUANTS="f16-f16;q8_0-q4_0;q8_0-q8_0;q5_1-q5_1"`. F32 storage uses its F16 counterpart in that list. The older `GGML_CUDA_FA_ALL_QUANTS=ON` remains a deprecated alias.

### Find a usable arena size

First free the GPU from competing workloads yourself. The [arena finder](benchmarks/find-max-arena.py) starts and stops only its own server children; it does not stop containers, download models or modify production.

A text-only Gemma MTP example (use your own model paths):

```sh
python3 benchmarks/find-max-arena.py \
  --server build-v2/bin/llama-server --model /path/to/gemma-4-31B-it-Q3_K_M.gguf \
  --model-draft /path/to/mtp-gemma-4-31B-it-Q8_0.gguf \
  --context-size 8192 --cache-type-k q8_0 --cache-type-v q4_0 \
  --spec-mode mtp --draft-length 3 \
  --output benchmarks/results/fit-gemma-mtp
```

Selected fields from an [actual RTX 5070 Ti run](benchmarks/data/gemma4-finder-example.json) of this configuration are below. It completed all 7,921 prompt tokens and 256 outputs. This measured fit is not a portable recommendation; refit for your GPU, context and workload. No starting guess or image is required.

```json
{
  "arena_mib": 1032,
  "initial_arena_mib": 2240,
  "next_mib_rejected": 1033,
  "context_size": 8192,
  "cache_type_k": "q8_0",
  "cache_type_v": "q4_0",
  "spec_mode": "mtp",
  "draft_length": 3,
  "prompt_tokens": 7921,
  "decode_tokens": 256,
  "projector_configured": false,
  "vision_execution_validated": false,
  "uvm": false,
  "reserve_mib": 0
}
```

Here, `arena_mib` is the result to use when launching the server; `initial_arena_mib` was only the search's starting guess. `next_mib_rejected` records the tested 1 MiB-larger candidate that did not fit, or is `null` if no such neighbor was tested.

| Fit configuration | Options to use |
| --- | --- |
| Target-only | `--spec-mode none` |
| Qwen embedded MTP | `--spec-mode mtp --draft-length 1`, `2` or `3`; no separate assistant |
| Gemma shared-KV MTP | The same MTP options plus `--model-draft /path/to/mtp-gemma-4-31B-it-Q8_0.gguf` |
| DFlash2 | `--spec-mode dflash --draft-length 7 --model-draft /path/to/sidecar.gguf` |
| Different K/V pair | `--cache-type-k q5_1 --cache-type-v q5_1` |
| Different batching | `--batch-size 512 --ubatch-size 256` or the settings you intend to serve |
| Deferred vision configuration, with or without MTP | `--mmproj /path/to/mmproj.gguf`; no image is required to find the arena allocation ceiling |
| Optional image-workload validation | Add `--image /path/to/photo.png`; repeat `--image` for a batch |

The largest allocatable parent is distinct from the minimum space needed by a vision batch. Projector weights and encoder workspace borrow regions inside that parent; they are not another full allocation added alongside it. With `--mmproj` alone, the finder configures the deferred projector and runs a text workload to find the allocation ceiling. It does not encode an image or qualify every possible vision batch.

Optionally supply images to additionally validate their vision-phase footprint and handoff. A literal prompt file needs one `<__media__>` marker per image, in image order. Without that file, the finder uses a short image-description prompt:

```sh
python3 benchmarks/find-max-arena.py \
  --server build-v2/bin/llama-server --model /path/to/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --context-size 262144 --cache-type-k q8_0 --cache-type-v q4_0 \
  --spec-mode mtp --draft-length 3 \
  --mmproj /path/to/mmproj-Qwen3.8-27B-F16.gguf \
  --image /path/to/first.png --image /path/to/second.png \
  --prompt-file /path/to/image-request.txt --image-max-tokens 4096 \
  --output benchmarks/results/fit-vision-mtp
```

For this optional Qwen image-validation example, `result.json` also records the supplied images and whether the vision path ran. Its vision-specific fields on success look like this (paths are schematic; no image-fit measurement is implied):

```json
{
  "context_size": 262144,
  "spec_mode": "mtp",
  "draft_length": 3,
  "projector_configured": true,
  "vision_execution_validated": true,
  "images": ["/path/to/first.png", "/path/to/second.png"],
  "prompt_file": "/path/to/image-request.txt",
  "decode_tokens": 256,
  "uvm": false
}
```

With `--mmproj` but no `--image`, `projector_configured` would be `true` and `vision_execution_validated` would be `false`.

Both runs print the selected launch command at the end:

```text
Selected command (POSIX shell):
env ... /path/to/build-v2/bin/llama-server ... --shared-device-memory-mib 1032 ...
```

That display abbreviates the measured Gemma example; the actual output and `command.txt` contain the complete command, including environment cleanup. Each output directory contains:

| File | Contents |
| --- | --- |
| `result.json` | Selected size, tested configuration, validation scope and launch command |
| `command.txt` | Complete POSIX-shell launch command, ready to copy |
| `probes.jsonl` | Results of the arena search probes |
| `arena-<MiB>.log` | Server output for each attempted arena size |

Every probe disables UVM and runs inference, including the configured output length, instead of accepting startup alone. Without `--image`, it uses the checked-in corpus and fills context minus output tokens and 15 positions of headroom by default, including when `--mmproj` is configured. Use `--prompt-tokens N` or `--prefill-text FILE` for another text workload. With images, it additionally exercises the supplied image batch and prompt, **not every possible image or a full-length text history**. JSON distinguishes `projector_configured` from `vision_execution_validated`; image-dependent minimum grants and external driver allocations still need runtime admission.

The tool determines the arena size for you. You do not need to pass `--initial-arena-mib`; its optional value only changes the initial search guess (default: 2,240 MiB), not the final size or a safety reserve. The tool searches in MiB steps without an arbitrary reserve, preserves probe logs, and writes `result.json`, `probes.jsonl` and a ready-to-run POSIX-shell command in `command.txt`. The command preserves the probe's removal of inherited `LLAMA_ARG_*` settings, UVM and graph/rollback debug overrides; JSON also records those environment changes for other shells. Unsupported kernels, incomplete responses and other non-OOM failures stop the search rather than being hidden as memory limits. A larger image batch or another process can change the result; there is no universal safe arena value. Fitting near full context can take considerable time.

### Launch examples

These examples use Bash. On Windows, adapt the argument grouping and executable path to your shell; the recorded qualification was on Linux. Substitute your own model paths and the arena found for **each** configuration.

```bash
KV_SERVER=./build-v2/bin/llama-server
KV_MODEL=/path/to/Qwen3.8-27B-UD-IQ4_XS.gguf
KV_MMPROJ=/path/to/mmproj-Qwen3.8-27B-F16.gguf
KV_DFLASH=/path/to/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
KV_ARENA=2240  # Example only; use your fit result.
KV_COMMON=(--model "$KV_MODEL" --ctx-size 262144 --parallel 1
           --batch-size 256 --ubatch-size 256 --n-gpu-layers 999
           --flash-attn on --fit off)
```

**Target-only text:**

```bash
"$KV_SERVER" "${KV_COMMON[@]}" --shared-device-memory-mib "$KV_ARENA" \
  --cache-type-k q8_0 --cache-type-v q4_0 --spec-type none --no-mmproj
```

**Another cache format:**

```bash
"$KV_SERVER" "${KV_COMMON[@]}" --shared-device-memory-mib "$KV_ARENA" \
  --cache-type-k q5_1 --cache-type-v q5_1 --spec-type none --no-mmproj
```

**Embedded MTP:**

```bash
"$KV_SERVER" "${KV_COMMON[@]}" --shared-device-memory-mib "$KV_ARENA" \
  --cache-type-k q8_0 --cache-type-v q4_0 --no-mmproj \
  --spec-type draft-mtp --spec-draft-n-max 3
```

**DFlash2:** its extra GPU weights usually leave a smaller arena than embedded MTP. The 1,500 MiB value below is an example, not the result of the MTP fit above.

```bash
"$KV_SERVER" "${KV_COMMON[@]}" --shared-device-memory-mib 1500 \
  --cache-type-k q8_0 --cache-type-v q4_0 --no-mmproj \
  --model-draft "$KV_DFLASH" --spec-draft-ngl 999 \
  --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-n-min 0 --spec-draft-p-min 0
```

**Vision without speculation:**

```bash
"$KV_SERVER" "${KV_COMMON[@]}" --shared-device-memory-mib "$KV_ARENA" \
  --cache-type-k q8_0 --cache-type-v q4_0 --mmproj "$KV_MMPROJ" --spec-type none
```

**Vision with embedded MTP:**

```bash
"$KV_SERVER" "${KV_COMMON[@]}" --shared-device-memory-mib "$KV_ARENA" \
  --cache-type-k q8_0 --cache-type-v q4_0 --mmproj "$KV_MMPROJ" \
  --spec-type draft-mtp --spec-draft-n-max 3
```

**Gemma target-only and separate shared-KV MTP:** use different fitted budgets for each mode. These 8 Ki sizes are measured examples on the reference card, not portable defaults.

```bash
GEMMA_MODEL=/path/to/gemma-4-31B-it-Q3_K_M.gguf
GEMMA_MTP=/path/to/mtp-gemma-4-31B-it-Q8_0.gguf
GEMMA_COMMON=(--model "$GEMMA_MODEL" --ctx-size 8192 --parallel 1
              --batch-size 256 --ubatch-size 256 --n-gpu-layers 999
              --flash-attn on --fit off --offline --no-context-shift
              --cache-type-k q8_0 --cache-type-v q4_0 --no-mmproj)

# Target-only: no assistant is loaded.
"$KV_SERVER" "${GEMMA_COMMON[@]}" --shared-device-memory-mib 1510 --spec-type none

# Run separately: matching Q8_0 assistant weights stay outside the 1032 MiB arena.
"$KV_SERVER" "${GEMMA_COMMON[@]}" --shared-device-memory-mib 1032 \
  --model-draft "$GEMMA_MTP" --spec-draft-ngl 999 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0
```

For Gemma lengths 1 or 2, change `--spec-draft-n-max` and refit. For 120 Ki, use `--ctx-size 122880` and repeat the finder at that capacity; do not reuse the 8 Ki result blindly. Leave `--swa-full` off: the shared arena requires bounded SWA. Common startup derives target verification width automatically; Gemma does not use Qwen recurrent snapshots. Do not attach DFlash or mmproj to this qualified Gemma text profile.

Only run one of these commands at a time. `--kv-stream-arena-mib` is an alias for `--shared-device-memory-mib`; the legacy fixed KV-pool option is not sufficient for shared vision/phase memory. The embedded auxiliary-layer count is derived from model metadata; do not add an auxiliary history for Gemma's borrowed caches or DFlash's independent sliding-window cache.

**Optional managed weights:** prefix a launch with `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`. The model weights become managed pages while the arena remains physical VRAM. This can help a multi-purpose GPU, but neither guarantees availability for another application nor eliminates migration slowdowns. Do not enable it during physical maximum-fit testing.

### Common startup and fit errors

| Error | What to check |
| --- | --- |
| Native/streamed attention unavailable for K/V or query width | Compiled FA pairs, GPU family, tensor geometry and supported model; a larger arena cannot add a missing kernel |
| Shared arena quota insufficient | The admitted execution needs more arena space; use the reported minimum when available |
| Device allocation OOM | Weights, arena, driver/executables and other applications must fit together; reduce the arena and rerun the full workload |
| Draft-cache override rejected | Remove draft K/V flags and their environment equivalents; adaptive drafting inherits target formats |
| Gemma bounded-cache / combined quota rejected | Leave full-SWA disabled and use the required persistent-local plus graph/attention minimum; a quote is not a guarantee against external driver OOM |
| Arena finder port unavailable | Select a free `--port`; live listeners are never contacted. Completed POSIX probe sockets can be reused without mistaking them for another service |
| Vision request rejected | Matching projector, image count/dimensions/token limits, context room and parent capacity; do not silently treat every image error as VRAM OOM |

## Performance and reproduction

### Gemma 4 through 120 Ki

The final 60-point sweep covers every 8 Ki step from 8 to 120 Ki at target-only and MTP lengths 1-3. It uses the Q3_K_M target, matching Q8_0 assistant when enabled, Q8/Q4 KV, b/ub=256, UVM off and 256 outputs after a fresh capacity-minus-271-token prompt. Every point has its own full-workload maximum-fit arena and an actual rejected 1 MiB-larger neighbor.

![Gemma 4 all-mode decode, prefill, physical arena and draft acceptance](media/gemma4-120ki-qualification.png)

[All 60 points (CSV)](media/gemma4-120ki.csv) · [Profile, hashes and qualification scope](media/gemma4-120ki.json)

| Capacity | Target-only decode | MTP1 | MTP2 | MTP3 |
| --- | ---: | ---: | ---: | ---: |
| 8 Ki | 39.61 | 67.68 | 81.50 | 91.16 |
| 64 Ki | 25.79 | 34.27 | 45.69 | 56.04 |
| 120 Ki | 10.33 | 16.97 | 22.70 | 28.26 |

Rates are tokens/s for this corpus and machine, with one measurement per mode/capacity. Independent maximum-fit arenas are 1,502-1,510 MiB target-only and 1,024-1,032 MiB with the assistant; its weights reduce the space left for the arena. This is not an equal-memory comparison or a statistical speedup claim. All 15,360 output IDs and counters are checked for completeness; native equivalence is established separately at fitting contexts and by exact operator/model controls. A successful 120 Ki workload is not a full-256-Ki qualification. The earlier stopped 128 Ki cohort is preserved separately, not mixed into this graph.

### Common prefill improvements

Two separate repeated A/B studies use fixed 1 GiB arenas, Q8/Q4, b/ub=256, UVM off and 256 outputs. Each has 52 fresh requests, three paired trials at 8/64 Ki for target-only/MTP3, plus single 120 Ki target-only controls. They compare successive adaptive implementations, **not stock UVM**; do not multiply different cohorts into an unmeasured combined speedup.

| Repeated 64 Ki prefill improvement | Gemma target-only / MTP3 | Qwen target-only / MTP3 |
| --- | ---: | ---: |
| Bounded conversion reuse vs pre-conversion adaptive | +32.44% / +32.12% | +5.14% / +5.23% |
| Two-wave pipeline vs conversion-only | +3.64% / +3.65% | +1.25% / +1.82% |

Paired prompts, output IDs and draft counts match; decode median differences stay within 0.17% for conversion and 0.15% for the pipeline. At 8 Ki, conversion gains are smaller and the incremental pipeline change is negligible. [Conversion CSV](benchmarks/data/prefill-conversion-v1.csv) / [profile](benchmarks/data/prefill-conversion-v1.json), [pipeline CSV](benchmarks/data/prefill-pipeline-v1.csv) / [profile](benchmarks/data/prefill-pipeline-v1.json).

Matched Nsight windows confirm the mechanism: at 96 Ki, H2D time overlapping observed GPU kernels increases from 3.2% to 86.6% for Gemma and 2.9% to 62.1% for Qwen, with reader/converter GPU interval unions roughly stable. These are clipped time-interval intersections, not SM occupancy, bandwidth or hardware counters. Instrumented throughput is not the speedup measurement; NCU counters remain unavailable on this machine.

### Stock UVM without KV streaming

The introduction adds a measured stock baseline using the same IQ4_XS weights, Q8_0 K / Q4_0 V and article corpus as the four-mode V2 sweep. Stock is pristine upstream revision `10a60cf303566e10d6a7a2774c17d2085503d87b`, the upstream base used by this V2 implementation. UVM is enabled for stock; there is no adaptive KV pool, phase arena, MTP, DFlash or host-rollback override. This is not the older Q3-weight measurement.

All 16 stock points from 8 through 128 Ki completed with 256 output tokens, one slot, b/ub=256, eight CPU threads and prompt length equal to context capacity minus 271. All stock output token IDs match the corresponding V2 target-only runs. The baseline ends at its last measured point; no stock performance is extrapolated to 256 Ki.

| At 128 Ki capacity | Stock + UVM | V2 streamed target-only |
| --- | ---: | ---: |
| Prefill | 837.1 tok/s | 934.6 tok/s |
| Decode | 2.48 tok/s | 26.15 tok/s |

The measured decode ratio at that point is about 10.6x; it is not a general speedup claim. At short contexts, stock decode is roughly 2-3% faster in this sweep. UVM migration can make performance uneven under pressure, so the plotted points are kept as measured rather than smoothed. [Stock-UVM measurements and configuration (CSV)](media/stock-uvm-iq4-throughput.csv).

### Q8/Q4 four-mode context scaling

The completed four-mode sweep measures target-only, MTP=3, DFlash2=3 and DFlash2=7 at every 8 Ki capacity from 8 Ki through 256 Ki: 32 points per mode, 128 total. It uses IQ4_XS weights, Q8/Q4 KV, the checked-in article corpus, one slot, b/ub=256, 256 generated tokens and 15 positions of headroom. Each mode/context independently uses its maximum-fit arena, without UVM or a safety reserve.

![Four-mode full-context comparison: decode, prefill, effective KV grant and draft acceptance](media/adaptive-kv-four-mode-comparison.png)

[All 128 plot points (CSV)](media/adaptive-kv-four-mode-comparison.csv). This is the earlier completed Q8/Q4 sweep, recorded before cache-format generalization. The later five-capacity Q8/Q4 regression checks in the matrix below retain identical arenas and output IDs, with throughput differences below 0.7%. The figures use their own recorded measurements; datasets are not mixed to fill missing modes or context points.

### Full-context comparison: KV formats and drafting modes

The recorded Qwen 45-point matrix uses the same Unsloth IQ4_XS target on an RTX 5070 Ti, context capacities 8/64/128/192/256 Ki, b/ub=256, one slot, 256 generated tokens and the [checked-in Wikipedia corpus](benchmarks/data/online-articles-262144-words.txt). Prompt length is capacity minus 271 positions. Each format, mode and context independently uses its maximum-fit arena, with UVM disabled and no safety reserve. This is separate from Gemma's Q8/Q4 60-point sweep and the newer fixed-budget prefill A/B cohorts.

![Decode/prefill, effective KV grant and draft acceptance](media/kv-quant-comparison.png)

[All 45 plot points (CSV)](media/kv-quant-comparison.csv)

| KV format at 256 Ki capacity | Target-only decode | MTP=3 decode | DFlash=7 decode |
| --- | ---: | ---: | ---: |
| Q8_0 / Q4_0 | 9.60 tok/s | 22.27 tok/s | 21.80 tok/s |
| Q8_0 / Q8_0 | 6.66 tok/s | 14.53 tok/s | 18.04 tok/s |
| Q5_1 / Q5_1 | 10.56 tok/s | 23.27 tok/s | 27.71 tok/s |

These are single-run measurements for this model, prompt and machine. Acceptance and memory pressure both affect speculative speed; DFlash is not always faster than MTP. Different K/V formats can change outputs, and this graph does not evaluate model quality.

All 15 Q8/Q4 comparisons with the pre-generalization build choose identical arenas and retain all 256 output token IDs. Decode differences are -0.67% to +0.57%; prefill differences are -0.11% to +0.20%. No material regression is observed; these small differences are not claimed as speed gains.

### Draft-length comparison at a fixed budget

This earlier experiment holds the parent at 2,240 MiB and compares embedded MTP lengths 0-3 with Q8/Q4. It is historical data, not another run of the maximum-fit matrix above.

![Fixed-budget MTP draft-length prefill and decode comparison](media/adaptive-kv-stream-v2-throughput-combined.png)

[Detailed plot: pool size and acceptance](media/adaptive-kv-stream-v2-mtp-sweep.png). Fixed-arena and independently maximum-fit curves answer different questions: the latter includes differences in available resident KV space. Neither equal arena size nor equal KV grant means equal total VRAM across different draft models.

### Reproduce text sweeps

Use a separate output directory for every run. The fixed-span script accepts a model path and configurable cache types:

```sh
python3 benchmarks/run-fixed-span-sweep.py \
  --server build-v2/bin/llama-server --model /path/to/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --contexts 8192,65536,131072,196608,262144 --prompt-headroom 15 \
  --cache-type-k q8_0 --cache-type-v q4_0 --mtp-lengths 0,3 \
  --decode-tokens 256 --arena-mib 2240 --auto-max-arena \
  --no-manage-production --output benchmarks/results/text-q8-q4
```

For the completed Gemma grid, use the separate assistant and explicitly stop at 120 Ki:

```sh
python3 benchmarks/run-fixed-span-sweep.py \
  --server build-v2/bin/llama-server --model /path/to/gemma-4-31B-it-Q3_K_M.gguf \
  --model-draft /path/to/mtp-gemma-4-31B-it-Q8_0.gguf \
  --min-context 8192 --max-context 122880 --context-step 8192 \
  --mtp-lengths 0,1,2,3 --cache-type-k q8_0 --cache-type-v q4_0 \
  --batch-size 256 --ubatch-size 256 --decode-tokens 256 --prompt-headroom 15 \
  --auto-max-arena --request-timeout 3600 --no-manage-production \
  --output benchmarks/results/gemma-final-120ki
```

The external assistant is not loaded for length 0. Omit `--auto-max-arena` to keep a supplied arena fixed. The general script's default endpoint is still 256 Ki, so **keep Gemma's explicit 120 Ki endpoint** for this qualification. `--no-manage-production` prevents the script from touching the local `llm-llmster` container; free the GPU yourself and use a fresh output directory.

The DFlash control harness verifies the pinned target, sidecar and publisher config before running. It is deliberately an artifact-specific reproduction tool, unlike the general arena finder:

```sh
python3 benchmarks/run_dflash_controls.py \
  --server build-v2/bin/llama-server \
  --target /path/to/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --draft /path/to/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --draft-config /path/to/DFlash2/config.json \
  --contexts 8192,65536,131072,196608,262144 --depths 3,7 \
  --prompt-tokens 262144 --prompt-headroom 15 --decode-tokens 256 \
  --target-k q8_0 --target-v q4_0 --arena-mib 1500 --auto-max-arena \
  --output benchmarks/results/dflash-q8-q4

python3 benchmarks/plot_dflash_controls.py \
  --series target-and-MTP benchmarks/results/text-q8-q4 \
  --series DFlash2 benchmarks/results/dflash-q8-q4 \
  --output benchmarks/results/speculation-comparison.png
```

For the dense four-mode sweep, pass all 32 capacities to both commands: in Bash, use `--contexts "$(seq -s, 8192 8192 262144)"`, retaining MTP lengths `0,3` and DFlash depths `3,7`. The plotting command combines both result directories without mixing in an earlier-build baseline.

Repeat with the other cache pairs and label each series accordingly. The control harness also supports `--spec-mode none --depths 0` and `--spec-mode mtp --depths 3`. It does not manage production containers. Its `--lifecycle` checks are separate from throughput measurement. The corpus is Wikipedia text under CC BY-SA 4.0 with article attribution, separate from the code license.

Correctness qualification compares matching formats and execution modes against native attention, including logits, hidden rows, KV bytes and recovery state. TG1 and multi-row stock kernels can themselves differ numerically; cross-mode output equality is not the numerical oracle. See [KV-format qualification](KV_QUANTIZATION_ROADMAP.md) and [the upstream attention port](UPSTREAM_ATTENTION_PORT_ROADMAP.md).

## How it works

### 1. Adaptive KV streaming

The two targets have different cache topologies, but use the same ordered-span reader and ownership contracts:

| Target | Full histories streamed from host | Bounded companion state |
| --- | --- | --- |
| Qwen3.8-27B | 16 full-attention blocks, head dimensions 256/256, four KV heads | 48 recurrent/linear-attention blocks |
| Gemma 4 31B | Ten full-attention blocks, dimensions 512/512, four KV heads | Fifty native SWA caches, dimensions 256/256, sixteen KV heads |

Full attention reads the growing history. The streaming pool manages those histories, not one independent full cache for every model block. Gemma's five-local/one-full pattern keeps native local masks and sliding-window semantics; only its full-attention group streams.

The complete encoded full-attention K/V history is authoritative in pinned host RAM. Gemma's local SWA caches do not get a second full host history. A logical page contains 256 token positions; its byte size follows the K/V types and head geometry. GPU space is divided into resident pages and one shared ring, not a separate large streaming allocation per layer.

```mermaid
flowchart LR
    H["Pinned host RAM: complete KV history"] -->|H2D| P["Resident pages"]
    H -->|H2D into free slots| R["Shared ring"]
    P --> A["Attention over ordered spans"]
    R --> A
    W["GPU KV producer"] -->|D2H changed rows| H
    A -->|Last reader completed| F["Slots reusable for later layers"]
    F --> R
```

When the history fits, attention reads resident data. Under pressure, the policy can stream selected layers first, then adjust placement as context grows. It trades resident pages for ring slots when earlier copies need more room to meet consumption deadlines. Newly written rows are published; unchanged historical pages do not need to be copied back to the host every token.

An attention input has at most the resident span, the current ring span and a wrap-around span in the optimized layout. The kernel follows logical token order rather than physical address order. Prefetch lookahead uses available slots and feedback, not a fixed three-layer window.

```mermaid
sequenceDiagram
    participant P as KV producer
    participant H as Host history
    participant C as Copy queue
    participant A as Attention queue
    P-->>H: Publish changed rows D2H
    par Current layer
        A->>A: Compute ready portions of L
    and Future work
        H-->>C: Prefetch stable pages of later layers
        C-->>A: Ready fence for transfer span
    end
    A->>A: Consume the next required spans
    A-->>C: Consumed fence after final reader
    Note over A,C: Reuse slots promptly and wait only for real dependencies
```

Ready fences prevent reads before copies finish; consumed fences prevent overwriting live data. Transfer-span/batch fences avoid one host/event operation for every page. Stock-compatible split/resume work can consume ready portions without inventing a different reduction order. Lookahead follows the next actual consumer, including a transition to MTP, rather than blindly prefetching another target round.

The implementation preserves stock traversal, quant conversion and final-reduction order where qualified. This avoids the extra rounding from independently reducing arbitrary chunks and merging their softmax results in a new order. Strict prefill uses bounded gathering/compact mask descriptors; attention still reads all required history. The tests qualify numerical behavior, not universal bit identity for every device and prompt.

For eligible quantized wide prefill (at least 128 query rows), one 2,048-token FP16 K/V stage reuses conversion across query tiles. Its additional quote is `2048 * KV_heads * (K_dimension + V_dimension) * sizeof(fp16)`: 8 MiB for the Qwen full layers and 16 MiB for Gemma's full layers. Existing converters and native attention plans/splits/checkpoints/reduction order are retained. Nonquantized caches, small queries and tile fallbacks keep native conversion execution; format support does not imply every profile benefits from preparation.

The prefill pipeline uses two alternating halves of the **existing encoded ring**, plus that single decoded stage. Transfer and ready-attention grouping are both bounded to a ceiling-half wave (seven/six pages for thirteen slots). While compute consumes one wave, DMA can fill the other. Conversion and attention remain ordered on the compute queue; this is not a claim that all three stages execute concurrently. Existing last-reader fences still protect slot reuse, including cancellation and failed submissions. No new GPU buffer or CUDA arithmetic is introduced, and decode scheduling is unchanged.

### 2. Phase arena

A native execution graph needs tensor workspace in addition to weights and KV. Keeping peak prefill, target decode, drafter and vision scratch independently allocated wastes space when they cannot run simultaneously. The arena gives those phases different layouts inside one bounded parent.

```mermaid
flowchart LR
    P["Text prefill: larger compute workspace + KV working set"] --> D["Text decode: smaller compute workspace + larger KV grant"]
    D --> P
    D --> M["Draft execution: borrow phase scratch"]
    M --> D
    P --> V["Vision: projector weights + encoder workspace"]
    V --> P
```

| Allocation | Lifetime / placement |
| --- | --- |
| Target model weights | Outside the shared arena; optionally managed by UVM |
| Gemma assistant / Qwen DFlash weights | Outside the shared arena; their presence reduces the maximum parent that fits |
| Compute workspace | Leased by the active target/draft phase |
| Gemma bounded SWA | Persistent fixed suffix inside the parent, preserved across prefill/decode/draft handoffs |
| Quantized wide-prefill preparation stage | Fixed 8/16 MiB for the reference full-layer geometries, inside prefill scratch; reused across layers/chunks and reclaimed for decode |
| Resident KV + ring | Uses the phase's remaining admitted KV grant |
| Vision weights + encoder workspace | Temporary separate grants inside the parent, returned after encoding |
| Complete full-attention KV history | Pinned host RAM, not the device arena |
| Recurrent/rollback state and staging | Host-backed and bounded device resources; not a second full-context draft KV allocation |
| Driver/native graph-executable bookkeeping | Also needs memory; not all of it is covered by the arena |

The main gain is reuse of graph **tensor workspace**, not an assumption that destroying a CUDA graph handle returns a fixed 1 GiB. A handoff drains affected work, invalidates stale executable bindings, releases changed leases, commits a new layout, then rebinds consumers. Reusing the same parent address does not make old tensor pointers valid.

The preparation stage raises the admitted prefill minimum by a fixed amount; startup does not silently negotiate a slower native-only profile for an undersized arena. The arena's declared total is unchanged when it fits, and decode recovers that space. Gemma's Q8/Q4, window-1024, ubatch-256, single-sequence SWA storage is 1,280 rows across fifty local roots: 425,984,000 bytes (406.25 MiB). This remains resident; the full history of the ten global layers is in host RAM. Other cache formats/batches have different quotes.

### 3. Vision and mmproj

For the matching separate Qwen vision encoder, image encoding is a distinct phase before text consumes the resulting embeddings. Existing target/MTP history survives the temporary change of device layout.

```mermaid
sequenceDiagram
    participant S as Server
    participant H as Host state
    participant T as Target/MTP
    participant V as Vision encoder
    S->>T: Stop submissions and drain
    T->>H: Preserve authoritative KV and pending state
    T->>T: Retire ring lease and device bindings
    S->>V: Lend arena regions for projector and scratch
    loop Compatible image batches
        V->>H: Encode and retain image embeddings
    end
    V->>V: Unload projector and release grants
    S->>T: Restore text layout
    H->>T: Supply image embeddings for text prefill
    T->>T: Decode and refill required KV pages lazily
```

Multiple images produce separate embedding chunks in prompt order; compatible encoder work may be batched. Larger image dimensions, token limits or image batches change the encoder requirement. The encoder grants must fit; they are not silently allowed to overwrite live text memory.

The server retains the source needed to reload projector weights. The image-aware MTP path keeps raw image embeddings distinct from shifted target hidden rows, and tracks physical KV rows separately from M-RoPE positions. Checkpoint and RAM prompt-cache restoration also preserve pending MTP inputs.

The recorded no-MTP native-capacity image runs used short backgrounds, not full-262K text histories. With a 2,240 MiB parent, repeated MTP=3/512px image requests settled at 15,836 MiB on the 16,303 MiB reference card; actual vision grants were about 884.62 MiB of weights plus 30.02 MiB of compute. These are sampled, workload-specific measurements, not an instantaneous peak guarantee. See [vision qualification and measurement instructions](tools/server/tests/README.md#adaptive-kv-vision-qualification) and [the consumer roadmap](DEVICE_MEMORY_CONSUMERS_ROADMAP.md#stage-75c-image-aware-embedded-mtp-qualification).

### 4. MTP integrated with KV streaming

#### Qwen: an auxiliary KV layer

The tested model has one dense MTP attention block. Its separate logical KV history has the same head geometry and inherited encoding as the target's full-attention histories. The common placement policy therefore counts it as a **17th attention KV layer**, while keeping its host identity and lifetime distinct.

A drafting phase acquires its resident portion plus a complete nonresident suffix in the shared ring. Catch-up and sequential predictions reuse that lease; new tail rows update it without uploading the unchanged prefix again. After the final MTP reader, the ring is released for target verification. It is not permanently carved into a target half and an MTP half.

```mermaid
sequenceDiagram
    participant T as Target
    participant R as Shared KV ring
    participant M as Embedded MTP
    T->>R: Retire final target readers and schedule MTP KV
    M->>R: Acquire complete MTP spans
    M->>M: Catch up on verified target inputs
    loop Draft predictions
        M->>R: Read retained spans and publish changed tail
        M->>M: Predict next candidate
    end
    M->>R: Release after last reader and prefetch first target layer
    T->>T: Verify candidate block
    alt All candidates accepted
        T->>T: Keep resulting state
    else Rejection
        T->>T: Restore selected recurrent state and truncate/replay suffix
    end
```

| Stock-style separate allocation | Shared implementation |
| --- | --- |
| Full-context draft GPU KV (about 416 MiB for Q8/Q4 at native capacity) | Separate host history; GPU pages compete within the shared 17-layer pool |
| Independent peak draft graph workspace | Borrowed phase workspace rather than simultaneous peak ownership |
| Large device rollback candidates for recurrent blocks | Host-backed candidates with bounded device staging; restore the chosen accepted prefix |
| Uncoordinated lookahead across target/draft phases | Common next-consumer scheduling and lease readiness |

This is **less extra VRAM, not zero extra VRAM**. MTP weights, active workspace, resident pages, device staging and transfers still have a cost. A depth-three rollback staging measurement is about 18.7 MiB of device memory plus 449 MiB of pinned host snapshots; these figures are not additive savings claims or a bound on all MTP memory. The current maximum-fit MTP arena is smaller than target-only in the performance graph.

Recurrent rollback must restore more than KV positions: recurrent state, candidate hidden inputs and cache frontiers must agree with the accepted prefix. Numerical/lifecycle tests force rejection paths instead of relying on a high benchmark acceptance rate to leave them untested. See [MTP architecture and qualification](MTP_ADAPTIVE_KV_ROADMAP.md).

#### Gemma: a read-only assistant over target KV

The matching Gemma assistant has four blocks: three local and one full. Its local blocks read the target's final local source and its full block reads the final full-attention source. The adapter derives/validates the source roles and geometry; it does not allocate four new histories or perform an independent KV catch-up. Three proposals mean three passes through all four assistant blocks, at a fixed query position within the round.

```mermaid
flowchart LR
    T["Gemma target"] --> S["Persistent native SWA in arena"]
    T --> H["Authoritative full KV in host RAM"]
    H --> R["Resident pages and shared ring"]
    S --> D["Read-only MTP assistant"]
    R --> D
```

Full-cache reads use progressive waves when a whole source cannot fit. Immutable ring coverage can be reused across predictions only while source identity, generation, logical rows and slot contents still match. Local reads retain checked native views. The serial owner lends the same parent workspace to the TG1 assistant and target verification; it reserves up to `draft_length + 1` target queries independently of Gemma's recurrent depth, which is zero. Weights remain separate from the arena.

Target verification is the only KV writer. Acceptance, correction hidden rows, local-window rollback, cancellation and saved-state restoration follow the stock driver and are tested against matching native execution. A rewind beyond available local history rejects transactionally rather than pretending the missing window was recomputed. Next-consumer hints follow actual source/role identity; borrowed reads do not train/reset the target feedback tuner, and target diagnostics do not substitute assistant transfer counts. Further Gemma decode optimization is deferred; the evaluated G5c async prototype was removed because its measured gain was negligible.

### 5. DFlash2: a separate bounded Qwen drafter

DFlash2 is not another full-history auxiliary target layer. The tested sidecar has five attention layers with a bounded sliding window, consumes target features, and produces a candidate block for verification. Its bounded cache and compute workspace participate in phase ownership, while its GPU weights still reduce the space available to target KV.

```mermaid
flowchart LR
    T["Target execution"] --> F["Bounded recent target features"]
    F --> D["DFlash2 windowed drafter"]
    D --> B["Candidate block"]
    B --> V["Target verification through TG8"]
    V --> A{"Accepted prefix"}
    A -->|Keep| T
    A -->|Rejection| R["Restore selected recurrent state; repair suffix"]
    R --> T
```

The Q8/Q4 draft cache is about 18.3 MiB at ubatch 256, rather than a second 262K-position GPU KV history. Its roughly 1,080 MiB of GPU weights remain outside the arena. Depth-seven target rollback uses about 43.6 MiB of bounded device staging inside the shared parent. Both the cache and workspace depend on the admitted configuration, including K/V formats and batching.

Two staging slots and completion dependencies allow recurrent state publication to overlap subsequent work without overwriting an in-flight source. The common ownership layer also coordinates target-feature publication, draft execution, verification and rejection recovery. A high draft length is not automatically faster: target KV grant, transfer volume, draft compute and acceptance all matter. See [DFlash2 design/qualification](DFLASH2_ADAPTIVE_KV_ROADMAP.md).

Gemma's shared-cache assistant above is not a DFlash pairing. Other Gemma variants and Muse Glimmer still need their own model/topology/admission qualification; generic memory APIs alone do not admit them.

## Developer guide

These are experimental fork contracts, not a stable upstream API. The policy/ownership layers are backend-neutral; actual transfer, KV-write, attention and executable-invalidation work still needs backend implementation.

### Architecture and API map

```mermaid
flowchart TD
    C["Server / model-specific execution order"] --> P["Phase plans and consumers"]
    P --> A["Arena planner, views and leases"]
    P --> K["Logical KV formats, placement and span plans"]
    A --> B["Backend-native allocation / view scope"]
    K --> X["Copy, publication and attention contracts"]
    X --> CUDA["CUDA implementation"]
    X -. "port and qualify" .-> OTHER["Other accelerator implementation"]
```

| Layer | Entry points / source | Responsibility |
| --- | --- | --- |
| Buffer ownership | [ggml-backend.h](ggml/include/ggml-backend.h): `ggml_backend_buffer_retain/free/view` | Reference-counted storage and bounded, independently resettable views |
| Tensor/graph allocators | [ggml-alloc.h](ggml/include/ggml-alloc.h): `ggml_tallocr_new_range`, `ggml_gallocr_set_buffer_range` | Place tensors and stock graph allocations inside caller-owned ranges |
| Arena and leases | [ggml-backend-memory.h](ggml/src/ggml-backend-memory.h) | Transactional region maps, retained leases, generations and submission gate |
| Scheduler borrowing | `ggml_backend_sched_attach_memory_lease/detach_memory_lease` | Attach complete leased workspace; detach safely before address reuse |
| Resource declarations | [llama-memory-requirements.h](src/llama-memory-requirements.h), [plans](src/llama-memory-plan.h) | Domain, alignment, size, access and content-lifetime requirements |
| Phase changes | [llama-memory-transition.h](src/llama-memory-transition.h) | Prepare, drain, invalidate, release, commit, bind, activate and recovery |
| Completion | [llama-memory-completion.h](src/llama-memory-completion.h) | Backend event or explicit synchronous fallback; safe producer/consumer dependencies |
| KV metadata | [ggml-kv-stream.h](ggml/src/ggml-kv-stream.h), [policy](src/llama-kv-stream-policy.h) | Encoded layout, validated spans and placement decisions |
| Backend execution | [copy ops](ggml/src/ggml-kv-stream-copy.h), [attention ops](ggml/src/ggml-kv-stream-device.h), [buffer-local hooks](ggml/src/ggml-backend-execution.h) | Backend-specific movement and execution within common ownership rules |
| MTP / DFlash consumers | [complete-layer lease](src/llama-kv-stream-layer-lease.h), [bounded KV](src/llama-memory-bounded-kv.h) | Full-history auxiliary leases versus bounded-window drafter storage |
| Gemma split caches | [iSWA cache](src/llama-kv-cache-iswa.h), [native cache](src/llama-kv-cache.h) | Stream the full group, quote/bind persistent local storage, reject unavailable rollback windows |
| Read-only consumers | [cache read bindings](src/llama-memory-phase.h), [session](src/llama-kv-stream-session.h) | Retained semantic/physical source bindings and progressive attention without append |

### Ownership, views and bounded allocation

A view changes the logical range/reset scope without copying storage. It retains its parent, so releasing the caller's parent handle does not destroy memory still used by a live child. This matters for OpenCL/SYCL-style buffers with per-buffer tensor metadata as well as pointer-addressed CUDA buffers.

```cpp
ggml_backend_buffer_t ggml_backend_buffer_view(
    ggml_backend_buffer_t parent, size_t offset, size_t size);

bool ggml_tallocr_new_range(
    ggml_tallocr * allocator, ggml_backend_buffer_t buffer,
    size_t offset, size_t size);

bool ggml_gallocr_set_buffer_range(
    ggml_gallocr_t allocator, int buffer_id, ggml_backend_buffer_t buffer,
    size_t offset, size_t size);
```

A bounded tallocr places persistent tensors within that range; gallocr still plans normal graph tensor reuse inside its external workspace. These do not turn tensors into relocatable virtual addresses. Growing a tensor, moving preserved contents or remapping pages requires an explicit higher-level policy.

Offsets, alignment, sizes and overflow are checked before access. Stateful graph workspace must use an isolated complete view rather than reset a shared parent. Retain/free are lifetime operations, not GPU synchronization: retaining an already-dead raw handle cannot resurrect it, and a live lease does not mean queued kernels are finished.

### Borrowing an arena region as graph workspace

Representative setup for one backend; production consumers must also follow the phase lifecycle below:

```cpp
bool attach_workspace(ggml_backend_sched_t scheduler, ggml_backend_t backend,
                      ggml_backend_buffer_type_t type, size_t bytes) {
    auto arena = ggml_backend_memory_arena_new(type, bytes);
    if (!arena) return false;
    ggml_backend_memory_region region{};
    const bool planned =
        ggml_backend_memory_arena_begin(arena, GGML_BACKEND_MEMORY_PLAN_NONE) &&
        ggml_backend_memory_arena_reserve(
            arena, 1, bytes, ggml_backend_buft_get_alignment(type),
            GGML_BACKEND_MEMORY_REGION_NONE, &region) &&
        ggml_backend_memory_arena_commit(arena);
    auto lease = planned ? ggml_backend_memory_arena_acquire(arena, 1) : nullptr;
    const bool attached = lease &&
        ggml_backend_sched_attach_memory_lease(scheduler, backend, lease);
    ggml_backend_memory_lease_free(lease);
    ggml_backend_memory_arena_free(arena);
    return attached;
}
```

The scheduler retains the attached lease; that lease retains the arena/storage. The caller can therefore release its handles after attachment. Later, `ggml_backend_sched_detach_memory_lease(scheduler, backend)` synchronizes/detaches the workspace and invalidates its graph tensor addresses. Never directly free the borrowed buffer returned by `ggml_backend_memory_lease_buffer()` as though it were a new owned allocation.

### Plans and phase transitions

Resources declare their memory domain/allocation class, minimum/preferred size, alignment, read/write access and whether contents are discardable, preserved or reconstructible. A plan is an ordered execution description, not an automatic LRU policy or an unrestricted graph of concurrent consumers.

```mermaid
flowchart LR
    PREP["Prepare and validate"] --> Q["Quiesce submissions"]
    Q --> D["Drain affected work"]
    D --> I["Invalidate old executables"]
    I --> R["Release changed leases"]
    R --> C["Commit region map"]
    C --> B["Bind and activate"]
    B --> RUN["Resume execution"]
```

Implement `llama_memory_consumer::prepare()` without mutating active state or submitting work. Its `llama_memory_preparation` then supplies `quiesce`, `drain`, `invalidate`, `release`, `bind` and `activate`. Recovery is an explicit opt-in contract; it must preserve/reconstruct consumer contents after partial failure, not merely restore pointer values.

Arena commits are transactional metadata changes with view materialization. A generation/revision change means cached bindings and native graph executables may need rebuilding even if the physical parent address is unchanged. Persistent bindings can survive only when the new plan preserves their contract. Check completion before releasing or repurposing asynchronously used memory.

### KV formats, spans and backend hooks

Common layout code derives encoded row sizes, page offsets and transfer extents from actual GGML types. A span plan represents logically ordered coverage and retains its physical source leases. The common adaptive policy proposes placement; it does not call CUDA or move bytes.

```cpp
llama_kv_stream_policy_result llama_kv_stream_policy_step(
    const llama_kv_stream_policy_config & config,
    const llama_kv_stream_policy_state & previous,
    const llama_kv_stream_policy_observation & observation,
    llama_kv_stream_policy_decision & output);

ggml_kv_stream_result ggml_kv_stream_span_plan_make(
    const ggml_kv_stream_shape & shape,
    const ggml_kv_stream_span_source * spans, size_t count,
    size_t active_tokens, size_t query_tokens,
    ggml_kv_stream_span_plan_t & output);
```

The backend's versioned copy table supplies operations such as `enqueue_span`, `acquire_span`, `release_span`, producer fencing, resource preparation and sparse feedback polling. The attention table advertises real K/V/geometry/query-width support and workspace requirements before allocation. A fallback with different workspace requirements must be admitted before launch, not silently chosen after a failed reservation.

```cpp
// Selected callbacks; consult the headers for the full versioned contracts.
bool (*enqueue_span)(void *, size_t first_slot, const void * k, const void * v,
                     size_t live_tokens, size_t padded_tokens);
bool (*acquire_span)(void *, size_t first_slot, size_t count);
bool (*release_span)(void *, size_t first_slot, size_t count);

bool (*spans)(ggml_backend_t backend, const ggml_tensor * attention,
              ggml_kv_stream_span_plan_t spans, ggml_backend_buffer_t workspace);
```

Buffer-local hooks redirect only the admitted KV-write/attention work into the owner; other graph operations remain ordinary backend execution. Host publication, consumer-ready fences, final-reader fences and cancellation have separate meanings. Do not replace them with a raw pointer whose lifetime is only implied by the next API call.

Qwen embedded MTP uses the logical-format, publication and complete-layer span contracts; Gemma instead borrows progressive read-only target sources. DFlash declares a bounded window and ordered feature/verification dependencies. Model-specific catch-up, candidate acceptance and recurrent rollback belong above the backend; they should not be reimplemented independently in every accelerator adapter.

### Gemma bounded storage and read-source contracts

These are internal C++ contracts, not additional public CLI flags. Use common model/context startup for ordinary MTP so hidden outputs, warmup retirement, assistant loading and serial workspace measurement occur in the qualified order.

| Operation | Contract |
| --- | --- |
| `llama_kv_cache_iswa::bounded_storage_requirements(...)` | Quote actual local roots, encoding, window/headroom, padding and buffer class before allocation; returns layout/identity, not storage or attention admission |
| `llama_memory_params::defer_bounded_cache` | Construct allocation-free local metadata first; normal execution is closed until the persistent lease is bound |
| `llama_kv_cache::bind_bounded_storage(lease)` | Validate exact physical quota/layout, materialize checked native tensors and retain storage. Same-lease binding preserves contents; no second persistent SWA allocation is needed |
| `llama_memory_bounded_kv::read(identity)` | Acquire a retained physical read pin. `contains_layer` revalidates root/storage bounds; `bind_layer_views` creates caller-owned metadata without initializing or clearing source bytes |
| `attach/arm/release_shared_bounded_reader(...)` | Own and validate native local-view metadata, then hold the source pin through all queued native readers. The caller completes readers before releasing it |
| `attach/arm/release_shared_stream_reader(...)` | Own full-read proxy roots over a target session. Arm after preparing the parent's serial phase; release before target mutation. Writes reject; proxy clear does not erase source KV |
| `session.read_source(...)`, `prefetch_read(...)`, `read_attention(...)` | Snapshot a completed source, validate its binding/revision/geometry, then queue/adopt or synchronously read progressive spans. No append or complete-layer residency requirement |
| `llama_context_params::kv_stream_verify_max` | Optional target query bound 1-4 for serial streamed Gemma; zero keeps the inferred profile. Common MTP startup derives D+1. Do not derive this from `n_rs_seq`, which is zero for Gemma |
| `model.sequence_stats(role)` / `runtime_diagnostics(...)` | Read-role transfer statistics stay separate; default runtime diagnostics align target append counts with target feedback timing |

Keep the target context alive until its assistant, graphs and borrowed views are destroyed. The physical parent outlives the local roots, proxy views and every queued reader. Semantic generations/frontiers come from the actual target-owned source, not a clone counter; a pointer or unchanged length does not prove unchanged content. Release graph bindings before reusing their workspace, and complete pending copies/native reads before releasing their source pins. After failed work, retain cleanup ownership until drain succeeds; reconstruct from authoritative state before reopening admission. No removed G5c ordered-reader/event-bank API is part of this contract.

For direct C API callers, select `swa_full=false` for bounded Gemma: the C API default is not the bounded profile. Full-SWA remains valid for ordinary native contexts, but shared bounded storage rejects it instead of shrinking it silently. Local metadata/cell state remains native; both cache groups must pass removal/history-availability prevalidation before either is mutated. See [Gemma qualification](GEMMA4_ADAPTIVE_KV_ROADMAP.md) for exact-fit/one-byte-short, rebind, window, state/restart and all-acceptance-count tests.

### Porting and tests

| Contribution | Required work |
| --- | --- |
| New backend | Native allocation/views and reset scope; host registration or suitable transfer storage; copy/completion semantics; encoded writes; stock-order span/resume attention; capability/workspace reporting |
| New model | Describe full-history, recurrent, sliding-window and aliased caches; derive the actual access order and speculative feature dependencies; implement preservation/recovery |
| New format or kernel path | Use actual encoded strides and bounds, retain arithmetic-order qualification, quote all scratch, and keep unsupported paths rejected |
| Optimization | Preserve ownership/cancellation correctness and compare matching execution modes plus representative throughput/memory measurements |

Use TDD for exact-fit/one-byte-short bounds, alignment/overflow, nested lifetimes, reset isolation, held leases, failed transitions, stale graphs, pending copies, cancellation and recovery. Then qualify actual backend execution and matched native/streamed numerical behavior, followed by long-context performance. Host memory tests alone do not prove GPU execution or race freedom.

See [memory infrastructure](DEVICE_MEMORY_INFRASTRUCTURE.md), [consumer roadmap and qualification](DEVICE_MEMORY_CONSUMERS_ROADMAP.md), [MTP roadmap](MTP_ADAPTIVE_KV_ROADMAP.md), [DFlash roadmap](DFLASH2_ADAPTIVE_KV_ROADMAP.md) and [KV-format qualification](KV_QUANTIZATION_ROADMAP.md) for the detailed contracts and ledgers. The CUDA [copy adapter](ggml/src/ggml-cuda/kv-stream-copy.cu) and [attention adapter](ggml/src/ggml-cuda/fattn.cu) are concrete porting references, not portable kernels.

---

## Upstream llama.cpp README

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

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

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
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
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
- [Device memory infrastructure](DEVICE_MEMORY_INFRASTRUCTURE.md)
- [Adaptive KV streaming and MTP integration](MTP_ADAPTIVE_KV_ROADMAP.md)
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
