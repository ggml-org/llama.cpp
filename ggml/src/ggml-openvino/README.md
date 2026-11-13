# Compiled model cache

`GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR` exports compiled CPU/GPU graphs with their weights. It bypasses the plugin-level `GGML_OPENVINO_CACHE_DIR` and uses `OPTIMIZE_SPEED`, so weightless caching is disabled.

One directory can hold blobs for different models and compilation settings. Run each intended workload once to export its dynamic graph:

```sh
GGML_OPENVINO_DEVICE=GPU \
GGML_OPENVINO_NATIVE_SOFTPLUS=1 \
GGML_OPENVINO_DISABLE_KV_SLICE=1 \
GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all \
GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR=/path/to/qwen-cache \
./build/ReleaseOV/bin/llama-bench -m /path/to/model.gguf -r 1
```

`GGML_OPENVINO_SPILL_DIR` remains optional for this first run. Wait for the `model cache WROTE` message and completion of the workload before stopping it. Compatible prefill and decode graphs share one blob and manifest. A graph with different ports or incompatible shapes gets an exact entry instead; interrupted exports are not cache hits.

On later runs, supply the same compilation settings and enable `GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY=1`:

```sh
GGML_OPENVINO_DEVICE=GPU \
GGML_OPENVINO_NATIVE_SOFTPLUS=1 \
GGML_OPENVINO_DISABLE_KV_SLICE=1 \
GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all \
GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR=/path/to/qwen-cache \
GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY=1 \
./build/ReleaseOV/bin/llama-bench -m /path/to/model.gguf -r 1
```

Cache-only mode allocates backend address space without filling weight pages. On Windows, this also uses system commit capacity. The model-buffer size in the loader log is this virtual size. Weight uploads only record source identity; they do not read or requantize the weights. Graph conversion and compilation are skipped. Runtime buffers are still allocated and populated normally.

Cache-only mode uses the settings provided by the current process. Keep these values exactly the same, including set versus unset: `GGML_OPENVINO_REQUANT_KQUANT`, `GGML_OPENVINO_NATIVE_SOFTPLUS`, `GGML_OPENVINO_DISABLE_KV_SLICE`, `GGML_OPENVINO_MANUAL_GQA_ATTN`, `GGML_OPENVINO_STATEFUL_EXECUTION`, `GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT`, `GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS`, `GGML_OPENVINO_REDUCE_COMPILE_MEM`, `GGML_OPENVINO_MEMORY_OPTIMIZE`, `GGML_OPENVINO_PROFILING`, and `GGML_OPENVINO_DEBUG_NODE`. On GPU, also repeat `GGML_OPENVINO_MOE_OP=0` if used. `GGML_OPENVINO_SPILL_DIR` is optional on the first run and ignored in cache-only mode; host-weight release is disabled in cache-only mode.

A missing or incompatible graph fails with an error instead of compiling with absent weights. The fingerprint uses the dynamic graph's topology, ports, model parameters, weights, settings, and OpenVINO version; changing only dynamic token or KV sizes does not require a new entry. A different workload can still require another graph; populate it first without cache-only mode.

## Restrictions

- Cache-only mode requires Linux or Windows and mmap loading (`--load-mode mmap`, or the default when all selected devices support mmap). Do not use tensor validation or mlock when trying to avoid weight reads.
- On Windows, the backend commits virtual memory for its buffers without touching weight pages. Large models can still reach the system commit limit.
- The model must execute entirely on OpenVINO, with dynamic CPU/GPU graphs and in-process caching enabled. Static/NPU execution and CPU fallback are unsupported in cache-only mode.
- GGUF metadata, tokenizer data, tensor descriptors, and graph construction are still needed. The llama.cpp loader is unchanged: depending on its prefetch settings, it may request pages with `MAP_POPULATE` or read-ahead on Linux, or `PrefetchVirtualMemory` on Windows. Non-mmap loading also reads the payload before the backend sees it.
- File identity, size, modification/change timestamps, tensor offsets, graph structure, settings, and OpenVINO version identify cache entries. Linux uses device/inode and Windows uses volume serial/file index. Replacing, copying, or modifying a GGUF invalidates its entries. This avoids reading weight bytes and ties the cache to the local source files. Keep those files unchanged throughout loading and inference.
- Use the same target device and compatible OpenVINO/plugin installation. Import support depends on the plugin; the tested CPU plugin cannot import MoE graphs containing `GatherMatmulCompressed`. GPU MoE and CPU dense graph imports were tested.
- Blobs contain weights and can approach model size for each compiled graph. Import still reads those blobs and initializes the device.
