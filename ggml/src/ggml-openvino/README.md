# Qwen MTP

The default Qwen3.5/Qwen3.6 GPU path is dynamic stateless execution with in-place fused recurrent ops and stable MTP sampling. Build against the matching OpenVINO runtime/GPU plugin with independent `RecurrentConvCacheUpdate` and `RecurrentGatedDeltaNetUpdate` ops. The native Update ops return computation results and update writable cache inputs in place, including rollback snapshots where required. No recurrent/sampling opt-in flag is needed on supported graphs. See the [development documentation index](dev_doc/README.md) for the implementation and the MTP sampling graph.

Use `--spec-type draft-mtp --spec-draft-n-max 2` with a GGUF that contains MTP weights. Parallel execution uses `-np` and `--kv-unified`. CPU snapshot execution, NPU execution, and forced static MTP are not supported.

Stateless recurrent execution uses the same reset, reorder, and partial cache-writeback paths at every slot capacity, including one slot without MTP.

The matched stateless GPU path uses `RecurrentConvCacheUpdate` and
`RecurrentGatedDeltaNetUpdate`. Each cache is a writable Parameter backed by one
stable caller-owned USM allocation. The ops return only convolution or attention;
there are no recurrent cache Results. Required source rows are staged before
writes, preserving resets, cyclic reorders and rollback. Cache readers must be
exclusive, including Reshape aliases.
`GGML_OPENVINO_DISABLE_FUSED_RECURRENCE=1` retains the ordinary cache decomposition.
Unmatched snapshot GDN uses ordinary GDN token prefixes with functional writeback.
Build both repos against the matching single-output op spec. Translation revision
15 requires regenerating compiled-model blobs once without cache-only mode.

Cache-only MTP serving still requires the separate OpenVINO dynamic output-binding fix (PR #39354). Removing recurrent cache Results does not eliminate that independent trigger in imported MTP graphs.

Set `GGML_OPENVINO_ENABLE_KV_4D=1` to enable the upstream stateless in-place
attention KV update pattern where eligible. It binds full-capacity cache tensors
and lets the GPU fusion expose the live prefix to attention. The current upstream
eligibility check supports one physical stream (including unified KV); explicit
split multi-slot KV retains the ordinary full-cache path. This option does not
change recurrent cache handling. Unified KV is recommended for multi-slot GPU
execution because it enables this in-place attention update.

The recommended Qwen3.6-35B-A3B Q4_K_M setup uses GPU with `GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all` and `GGML_OPENVINO_NATIVE_SOFTPLUS=1`. Leave `GGML_OPENVINO_KV_SLICE` at its default (`1`). See [attention KV layout](dev_doc/OPENVINO-RECURRENT-CACHE.md#attention-kv-layout) for eligibility.

`q4_asym64_all` converts Q4_K/Q5_K/Q6_K/Q8_0/Q5_1 weights, including dense Q8_0 projections, to asymmetric INT4 with group size 64. `token_embd.weight` and `output.weight` retain symmetric channel-wise INT8. Q8_0/Q5_1 rows whose width is not divisible by 64 also use channel-wise INT8. This follows the primary/backup precision scheme of NNCF `INT4_ASYM`, `ratio=1.0`, `group_size=64`, `backup_mode=INT8_SYM` for these supported source types; it requantizes GGUF values rather than the original floating-point model and leaves floating-point weights unchanged. Earlier versions kept dense Q8_0 at INT8/group-32. Repopulate compiled-model caches without cache-only mode after this change; the cache fingerprint invalidates the old precision policy.

See the [implementation guide](dev_doc/OPENVINO-RECURRENT-CACHE.md) for buffer ownership and rollback.

# Compiled model cache

`GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR` exports compiled CPU/GPU graphs with their weights. It bypasses the plugin-level `GGML_OPENVINO_CACHE_DIR` and uses `OPTIMIZE_SPEED`, so weightless caching is disabled.

One directory can hold blobs for different models and compilation settings. Run each intended workload once to export its dynamic graph:

```sh
GGML_OPENVINO_DEVICE=GPU \
GGML_OPENVINO_NATIVE_SOFTPLUS=1 \
GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all \
GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR=/path/to/qwen-cache \
./build/ReleaseOV/bin/llama-bench -m /path/to/model.gguf -r 1
```

`GGML_OPENVINO_SPILL_DIR` remains optional for this first run. Wait for the `model cache WROTE` message and completion of the workload before stopping it. Compatible prefill and decode graphs share one blob and manifest. A graph with different ports or incompatible shapes gets an exact entry instead; interrupted exports are not cache hits.

On later runs, supply the same compilation settings and enable `GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY=1`:

```sh
GGML_OPENVINO_DEVICE=GPU \
GGML_OPENVINO_NATIVE_SOFTPLUS=1 \
GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all \
GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR=/path/to/qwen-cache \
GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY=1 \
./build/ReleaseOV/bin/llama-bench -m /path/to/model.gguf -r 1
```

Cache-only mode allocates backend address space without filling weight pages. On Windows, this also uses system commit capacity. The model-buffer size in the loader log is this virtual size. Weight uploads only record source identity; they do not read or requantize the weights. Graph conversion and compilation are skipped. Runtime buffers are still allocated and populated normally.

Cache-only mode uses the settings provided by the current process. Keep these values exactly the same, including set versus unset: `GGML_OPENVINO_REQUANT_KQUANT`, `GGML_OPENVINO_EXACT_ZERO_POINT`, `GGML_OPENVINO_NATIVE_SOFTPLUS`, `GGML_OPENVINO_MANUAL_GQA_ATTN`, `GGML_OPENVINO_STATEFUL_EXECUTION`, `GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT`, `GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS`, `GGML_OPENVINO_REDUCE_COMPILE_MEM`, `GGML_OPENVINO_MEMORY_OPTIMIZE`, `GGML_OPENVINO_PROFILING`, `GGML_OPENVINO_DISABLE_SHAPE_FROM_MASK`, and `GGML_OPENVINO_DEBUG_NODE`, `GGML_OPENVINO_ENABLE_KV_4D`, and `GGML_OPENVINO_DISABLE_FUSED_RECURRENCE`. On GPU, also repeat `GGML_OPENVINO_MOE_OP=0` if used. `GGML_OPENVINO_SPILL_DIR` is optional on the first run and ignored in cache-only mode; host-weight release is disabled in cache-only mode.

`GGML_OPENVINO_KV_SLICE` is a runtime binding option for dynamic graphs and can differ between cache population and cache-only runs. The same dynamic compiled graph accepts full-capacity and live-prefix attention K/V tensors. Slicing is enabled by default for contiguous ordinary K/V roots with one allocated stream, one active sequence at offset zero, and no sliding-window attention. Recurrent convolution/GDN caches and their rollback snapshots always retain their full stored row bank. Other layouts use full bindings. Repopulate caches made before this option was removed from the fingerprint once without cache-only mode.

A missing or incompatible graph fails with an error instead of compiling with absent weights. The fingerprint uses the dynamic graph's topology, ports, model parameters, weights, settings, and OpenVINO version; changing only dynamic token or KV sizes does not require a new entry. A different workload can still require another graph; populate it first without cache-only mode.

## MTP sampling and active slots

Qwen3.5/Qwen3.6 MTP uses a pure top-k backend sampler. Dynamic stateless execution retains its sampling branches for every configured slot and output row, including inactive branches. One packed `sampling_rows` input selects logits rows at runtime. Graph node names and candidate-output ports stay fixed, so changing the active subset of four slots reuses the same compiled draft model instead of creating up to 15 variants.

Active candidate IDs and scores are written to the normal GGML outputs. Inactive branches read row 0 (the padding row when there are no logits) and write to private scratch tensors (device USM on GPU), without changing sampler state or touching inactive GGML buffers. The top-k computations remain separate per branch; this trades some extra sampling work for a stable graph interface. Other sampler chains retain their existing branch-selection behavior.

This is one draft model per compatible graph configuration, not one compiled blob for the whole server. The trunk and other graph stages still have their own entries. Different sampler configurations, recurrent capacities, rollback modes or incompatible KV layouts can require additional models. Cache-only mode still requires initial preparation; changing just the active subset no longer requires exhaustive subset priming.

The generic recurrent-cache translation invalidates blobs made with the former single-slot shortcuts. Run once without `GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY` to populate compatible entries before enabling cache-only mode again.

## Restrictions

- Cache-only mode requires Linux or Windows and mmap loading (`--load-mode mmap`, or the default when all selected devices support mmap). Do not use tensor validation or mlock when trying to avoid weight reads.
- On Windows, the backend commits virtual memory for its buffers without touching weight pages. Large models can still reach the system commit limit.
- The model must execute entirely on OpenVINO, with dynamic CPU/GPU graphs and in-process caching enabled. Static/NPU execution and CPU fallback are unsupported in cache-only mode.
- GGUF metadata, tokenizer data, tensor descriptors, and graph construction are still needed. The llama.cpp loader is unchanged: depending on its prefetch settings, it may request pages with `MAP_POPULATE` or read-ahead on Linux, or `PrefetchVirtualMemory` on Windows. Non-mmap loading also reads the payload before the backend sees it.
- File identity, size, modification/change timestamps, tensor offsets, graph structure, settings, and OpenVINO version identify cache entries. Linux uses device/inode and Windows uses volume serial/file index. Replacing, copying, or modifying a GGUF invalidates its entries. This avoids reading weight bytes and ties the cache to the local source files. Keep those files unchanged throughout loading and inference.
- Use the same target device and compatible OpenVINO/plugin installation. Import support depends on the plugin; the tested CPU plugin cannot import MoE graphs containing `GatherMatmulCompressed`. GPU MoE and CPU dense graph imports were tested.
- Blobs contain weights and can approach model size for each compiled graph. Import still reads those blobs and initializes the device.
