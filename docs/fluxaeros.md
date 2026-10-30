# FluxAeros demand-paged MoE

This branch adds a VRAM expert pool, demand-ordered I/O, a registered host victim cache and optional automatic cache warmup. Selected experts and stored quantized weights are not pruned or requantized. It is a release candidate for the configuration below, not a claim of portability or production readiness on every llama.cpp backend.

## Tested envelope

- Linux x86-64, NVIDIA RTX 3060 12 GB, CUDA `sm_86`, 16 GB system RAM, local NVMe.
- The `qwen4exp` Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS model with 48 trunk layers and 512 experts per layer. Its actual router selects 10 experts, not 8.
- All routed expert tensors must be in the first GGUF shard. The second shard is still required for per-layer token embeddings.
- One paged model context per process, one server slot, context 512, 5,000 VRAM slots, `-ub 64` with the existing effective eight-token paging chunks.
- Process envelope: `MemoryMax=12G`, `MemorySwapMax=0`. The host victim cache defaults to 8 GiB when the DMA/CUDA Graph path is available. It is not an additional 8 GiB outside that limit.
- `MOE_HOT_K=0` and `MOE_PREKICK=0`. No background disk warmer competes with demand reads.

Other GPUs, operating systems, backends, models, larger contexts and parallel paged contexts are not certified by these results. The current custom CUDA sources target `sm_86`; a CPU-only build of this fork is not part of the supported release configuration. Do not expose the server to an untrusted network without a separate authentication and deployment review.

## Build all targets

Install a C++17 compiler, CMake, a working CUDA toolkit/driver, the liburing development package, and the normal llama.cpp build dependencies. Python 3.10+ and NumPy are needed for the packer and its tests; it imports the in-tree `gguf-py` reader.

```bash
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build build -j12
ctest --test-dir build --output-on-failure -j2
python3 -m unittest discover -s tests -p test_moe_pack.py
```

This builds `llama-cli`, `llama-server`, completion/benchmark tools, configured examples and tests. Some existing tests fetch external model metadata. A network failure or timeout in those tests must be reported separately, not counted as a successful test. Do not regenerate reference snapshots to hide failures.

## Prepare an exact expert pack

Obtain the model separately under its own license. Do not distribute its weights with the program. Set local paths yourself:

```bash
export MODEL=/path/to/model-00001-of-00002.gguf
export PACK=/path/to/expert-pack
python3 tools/moe_pack.py --model "$MODEL" --output "$PACK" --dry-run
python3 tools/moe_pack.py --model "$MODEL" --output "$PACK"
python3 tools/moe_pack.py --model "$MODEL" --output "$PACK" --verify-all
```

The packer copies each expert's original gate, up and down bytes, writes a compatible `native_experts.txt`, and records a pack SHA-256 in `expert-pack.json`. It does not dequantize or quantize. Existing output files are never overwritten. The completion index is published last. Interrupted temporary files are not a valid pack.

For the tested model, `experts.bin` requires 35,454,976,000 bytes (about 33.02 GiB) in addition to the GGUF files. `--verify` checks the first, middle and last expert in every layer; `--verify-all` checks every expert and reads both the GGUF and the pack. The sampled check is not a full verification. An existing compatible pack can be verified without repacking or allocating another 33 GiB.

The optimized demand worker deliberately uses buffered `io_uring` reads so the kernel page cache remains useful. The historical option name `--moe-direct-io` does not mean every demand read bypasses the page cache.

## Run

```bash
export MOE_HOT_K=0 MOE_PREKICK=0 MOE_SEED_BUDGET_MS=0
systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 \
    build/bin/llama-server -m "$MODEL" -ngl 99 \
    --moe-direct-io --moe-pack-dir "$PACK" --moe-slots 5000 \
    -ub 64 -np 1 -c 512 -t 6 --host 127.0.0.1 --port 8080
```

For the CLI, substitute `build/bin/llama-cli` and use `-st -p "Your prompt" -n 128` instead of the server address options. Keep the same model, paging, memory and thread settings when comparing performance.

`MOE_DROP_MMAP`, `MOE_SPLIT`, `MOE_PLE_STAGING`, `MOE_DMA` and the supported victim-cache path default on for paging. `MOE_VICTIM_CACHE=0` or `MOE_CACHE_MIB=0` disables the victim cache. `MOE_CACHE_MIB=N` changes its requested size; it does not raise the process memory limit. Decrease it when other contexts or models need host memory.

For diagnostics, set `MOE_IO_DIAG=1 MOE_DIAG=1`. These report demand bytes, cache hits/saves and actual CUDA Graph capture/replay. `MOE_VERIFY_ROUTER=/local/file` records raw router logits for exact comparisons. Traces contain workload-dependent data and should not be bundled in public archives.

Demand read/DMA failure is fatal rather than substituting zero expert weights. This protects output correctness; it is not automatic request recovery. Shut down the process, repair the storage/pack problem and restart. The single-context ownership guard rejects a second paged context rather than sharing global worker state unsafely.

## Automatic cache warmup

Warmup is opt-in and completes before requests or CLI generation begin:

```bash
build/bin/llama-server ... --auto-warm
build/bin/llama-cli ... --cache-warmup 128 \
    --cache-warmup-prompt "Explain the kinds of programming tasks I usually ask about."
```

- `--auto-warm` generates up to 256 tokens and replays the same sequence once.
- `--cache-warmup N` sets the budget; `0` disables it, and the maximum accepted value is 8192. `LLAMA_ARG_CACHE_WARMUP` is the environment equivalent.
- `--cache-warmup-prompt TEXT` selects a representative prompt. The default is a general explanatory prompt, not the benchmark prompt.
- The budget is additionally bounded by the context space remaining after the prompt. An EOG token ends generation early.
- Warmup clears KV/recurrent state, resets sampler RNG state and request performance counters, but keeps reusable expert caches. It requires a decoder-only generation context.
- `--no-warmup` controls the pre-existing empty graph warmup, not this separate option. Use `--cache-warmup 0` to disable cache warmup.

Startup cost is real and reported separately. On the tested workload, the 256-token generation plus replay took about 42 seconds. Warming one topic cannot guarantee cache hits for unrelated topics, nor can an 8 GiB cache hold the entire model. Do not report its cost as a free cold-start speedup.

## Measurements and interpretation

Measurements are from October 30, 2026. The controlled protocol drops the expert-pack page cache before startup, performs a 64-token warmup, then issues two identical 256-token greedy completions with a 22-token prompt and prompt caching disabled. Both requests remain in the same process. Physical MiB/generated-token includes the request's prompt reads and divides by 256 generated tokens.

| Configuration | TG Run 1 / Run 2, tok/s | PP Run 1 / Run 2, tok/s | Physical MiB/generated-token |
| --- | ---: | ---: | ---: |
| Cache-off control | 13.87 / 14.27 | 14.89 / 17.95 | 44.66 / 39.67 |
| Committed 8 GiB victim cache, `be9328717` repeat | 16.29 / 22.03 | 13.85 / 19.35 | 35.37 / 12.43 |
| Release lifecycle/warmup code, auto-warm disabled, traced | 16.32 / 21.43 | 13.89 / 18.62 | 35.38 / 12.47 |
| Same code, auto-warm enabled, traced | 16.80 / 21.85 | 16.18 / 18.29 | 30.54 / 12.88 |

20+ tok/s is a warm repeated-workload result, not a cold-start guarantee. Prefill has not improved consistently across both runs. Keep paired values and repeat measurements rather than selecting only the fastest run.

CUPTI main-stream profiling of the committed victim-cache path measured approximately:

| Phase | Main-stream span | Resolve/copy family | Other GPU compute | Gaps |
| --- | ---: | ---: | ---: | ---: |
| Run 1 decode | 60.26 ms/token | 34.89 | 24.59 | 0.78 |
| Run 2 decode | 44.30 ms/token | 18.97 | 24.54 | 0.79 |
| Paired prompt processing | 55.25 ms/token | 46.32 | 8.12 | 0.81 |

These are GPU main-stream spans, not complete request wall times. D2H and H2D overlap the main stream; do not add their duration again. Fetch/promotion stalls remain the main target for further work.

An additional prefill-capture ablation measured 16.19 / 21.86 TG and 13.43 / 18.77 PP. It did not achieve stable prefill CUDA Graph replay or a repeatable improvement, so that experimental change was removed rather than enabled by default.

## Release validation

- All configured build targets complete, including `llama-cli` and `llama-server`.
- 60 CTests pass. `test-quant-type-selection` previously timed out fetching external Hugging Face metadata and is excluded from the final local pass; it is not claimed as passing.
- Six packer tests pass, including mixed layer types, exact round trips, corruption/truncation detection and refusal to overwrite existing packs.
- Full comparison of the existing model pack verifies all 24,576 experts against the original GGUF bytes. No second full-size pack was needed.
- Both server warmup arms match 576 golden output tokens each and all 28,080 original router records. The enabled trace contains exactly one extra contiguous block of 24,768 warmup records.
- Seeded CLI generation (seed 1234, temperature 0.8, 32 output tokens, backend sampling enabled) is identical with cache warmup off/on after excluding the displayed performance counters. Both CLI processes exit cleanly.
- Native CUDA tests cover all-hit recency, D2H victim preservation, H2D slot reuse, repeated graph execution, eviction invalidation and completion-flag resets.
- Controlled server runs have zero cgroup swap/OOM kills, clean shutdown and sampled memory below 10.6 GiB. Startup reclaim can reach the 12 GiB cap.

The native binaries were built with GCC 16.1.1, CUDA 13.3, glibc 2.43 and driver 610.43.02. They are local-build artifacts, not a portable binary certification for older Linux distributions or different CPUs. Rebuild from the source archive on a different system. No model weights or MTP sidecar are included.

## MTP status

The available one-layer `qwen4exp` MTP sidecar loads and drafts tokens. It is about 2.58 GiB and must be provided separately. With the draft on CPU and the host victim cache reduced to 6 GiB, a 16-token smoke test accepted 6 of 8 drafted tokens (75%) but did not match the golden continuation: the first difference was generated token 8. The matched non-MTP control reproduced the golden tokens.

MTP is therefore not enabled or certified for the exact-output paged release. Acceptance rate alone is not evidence of either speed or exact routing. Its metadata declares a different, unspecified `other` license; no sidecar is included in release artifacts.

## Release checklist

- Build all configured targets and record compiler/CUDA/device versions.
- Run C/C++ and packer tests; retain failures and external-test limitations.
- Check golden tokens and full router traces on the exact binary, including CUDA Graph replay.
- Test cache warmup on/off, sampler reset, CLI execution and clean startup/shutdown.
- Keep the 12 GiB memory and zero-swap envelope; check cgroup OOM counters.
- Verify the expert pack against the user's GGUF before deployment.
- Include source, LICENSE, setup documentation, binary dependencies and SHA-256 checksums; exclude model weights, private prompts and raw routing traces.
- Leave MTP and speculative disk prefetch disabled. Do not label untested platforms or workloads stable.
- A local commit/archive is not publication. Publishing, tagging and uploading remain separate human decisions.
