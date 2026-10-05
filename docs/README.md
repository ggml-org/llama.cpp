# Documentation

Docs for the Raudbjorn llama.cpp fork (TurboQuant+ on Intel Arc). Folders hold upstream's docs and
the fork's own; [development/upstream-merge.md](development/upstream-merge.md) lists which is which.

## build/ - building and installing

- [build.md](build/build.md) - build from source, all shipped backends
- [build-profiling.md](build/build-profiling.md) - profile the build itself
- [install.md](build/install.md) - pre-built packages
- [docker.md](build/docker.md) - container images
- [android.md](build/android.md) - Android builds
- [xcframework.md](build/xcframework.md) - Apple XCFramework

## backend/ - backend guides

- [SYCL.md](backend/SYCL.md) - SYCL on Intel GPUs, the fork's primary backend
- [OPENVINO.md](backend/OPENVINO.md) - OpenVINO backend
- [MOE-CACHE.md](backend/MOE-CACHE.md) - MoE expert cache
- [ops.md](ops.md) - op support per backend, generated from [ops/](ops/) by `scripts/create_ops_docs.py`.
  The tables come from `test-backend-ops support --output csv` per backend. The Vulkan, SYCL and
  OpenVINO columns were measured on the Arc A770, OpenVINO with `GGML_OPENVINO_DEVICE=GPU`, whose
  table is narrower than the CPU device's.

## turboquant/ - TurboQuant+ KV and weight quantization

- [KV-cache-quantization.md](turboquant/KV-cache-quantization.md) - turbo KV types, asymmetric K/V and layer-adaptive modes
- [quality-benchmarks.md](turboquant/quality-benchmarks.md) - quality validation method and results
- [ppl-results/](turboquant/ppl-results/) - per-model perplexity and capacity sweeps

## user/ - running models

- [models.md](user/models.md) - obtain and quantize models
- [multi-gpu.md](user/multi-gpu.md) - split a model across GPUs
- [preset.md](user/preset.md) - INI presets
- [completions.md](user/completions.md) - shell completions

## features/ - feature guides

- [speculative.md](features/speculative.md) - speculative decoding and MTP drafting
- [function-calling.md](features/function-calling.md) - tool calls
- [llguidance.md](features/llguidance.md) - LLGuidance constrained decoding
- [multimodal.md](features/multimodal.md) - multimodal models, per-model notes in [multimodal/](features/multimodal/)

## development/ - working on the code

- [upstream-merge.md](development/upstream-merge.md) - merge upstream into this fork
- [turboquant-upstream-merge-notes.md](development/turboquant-upstream-merge-notes.md) - notes from earlier upstream catch-up merges
- [HOWTO-add-model.md](development/HOWTO-add-model.md) - add a model architecture
- [debugging-tests.md](development/debugging-tests.md) - debug tests
- [parsing.md](development/parsing.md) and [autoparser.md](development/autoparser.md) - model output parsing
- [token_generation_performance_tips.md](development/token_generation_performance_tips.md) - token generation performance
- [release.md](development/release.md) - upstream's release process

## Other

- [SDK.md](SDK.md) - the `lib` branch, a library-only tree for embedding
- [research/](research/README.md) - dated research and measurement corpus
