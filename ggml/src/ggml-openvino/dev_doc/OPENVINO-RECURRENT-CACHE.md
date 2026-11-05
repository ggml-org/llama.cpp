# Current Qwen GPU implementation

As of 2026-10-29, the default Qwen3.5/Qwen3.6 path is dynamic stateless execution with `RecurrentConvCacheUpdate`, `RecurrentGatedDeltaNetUpdate`, and stable pure-top-k MTP translation. It supports one or multiple slots and rollback. The implementation applies to `qwen35` and `qwen35moe`; the reference model is the UD Qwen3.6-35B-A3B MTP GGUF.

Build against the matching OpenVINO runtime/GPU plugin with the independent Update ops. The tested runtime is `qwen-mtp-inplace-primary`, commit `9299cc3d78`; llama.cpp stable sampling is `71c0bbc4d`. Configure `GGML_OPENVINO_DEVICE=GPU`, native softplus, and the documented requantization policy. The current cache translation revision is 14.

## Buffer ownership and operation boundaries

| State | Storage and lifetime | Update path |
|---|---|---|
| Attention K/V | Full caller-owned USM allocation, sized by context and KV layout | Native stateless KV fusion when eligible; SDPA consumes the live prefix |
| Convolution `cache_r` | Full caller-owned USM row bank, sized by slot/snapshot capacity | Fused reset, original-row gather/reorder, convolution and ordered history/snapshot writes |
| GDN `cache_s` | Full caller-owned USM row bank, sized by slot/snapshot capacity | Fused reset, original-row gather/reorder, GDN and ordered final/snapshot writes |
| Runtime controls | Reused USM host storage on GPU | Row maps, offsets and write controls remain runtime inputs |
| Protected source rows | Private device scratch | Staged before writes to protect cycles, duplicate sources and overlaps |
| Inactive sampling results | Private device-USM scratch | Discarded without touching inactive GGML output buffers |

Attention context capacity does not set recurrent-cache size. Cache input Parameter and cache Result bind the same stable allocation on the primary path. There are no rotation banks or pointer swaps.

The GPU plugin uses two stages: first preserve required original rows in scratch while applying logical reset and precision rounding; then compute and perform ordered writes. This avoids overwriting a source before another destination has read it. Identity reorder sources need no scratch writes. In-place execution removes the spare full output bank, not every cache read or internal memory transaction.

`RecurrentConvCacheUpdate` has nine required inputs, including depthwise weights. Output 0 is convolution `[1,B,T,C]`; output 1 is the full updated cache. It does not expose an intermediate history/QKV tensor. `RecurrentGatedDeltaNetUpdate` emits attention and the full updated state cache; the cache retains the newest `min(T,snapshot_count)` states for MTP, newest first.

## Generic fallback and restrictions

Exact compatible layouts/storage and exclusive cache readers permit mutation, including eligible optimized Reshape aliases. Shared original-cache readers, incompatible layouts, unrelated overlaps or unsupported execution paths retain immutable-input semantics using a private output and required Result copy-back. Restore or clear state after a failed aliased inference before retrying. GPU replay is disabled for these ops because stage selection depends on bindings.

`GGML_OPENVINO_DISABLE_FUSED_RECURRENCE=1` retains the ordinary cache decomposition for diagnosis. Naive graphs skip full-model recurrence fusion setup. Unmatched/naive snapshot GDN uses the independent Update op with a temporary cache bank; ordinary GDN remains unchanged. Snapshot/rollback (speculative decoding) on OpenVINO requires a patched runtime/GPU plugin. Unpatched packages reject snapshot GDN with a warning and allow scheduler fallback to another backend. Rotation, the former three-op cache solution, the history-only/separate-convolution mode and single-slot no-rollback shortcuts are removed. Stateful and NPU MTP are not the supported default; this validation does not claim those paths.

## Attention KV layout

`GGML_OPENVINO_ENABLE_KV_4D=1` enables eligible native stateless in-place attention updates. The current eligibility covers one physical stream, including unified KV. Explicit split multi-slot KV retains the full-cache path and can be much slower even with only one active slot. Use `--kv-unified` for multi-slot OV GPU serving.

Leave `GGML_OPENVINO_KV_SLICE` at its default (`1`). Live-prefix slicing applies to eligible ordinary attention K/V roots, not recurrent caches. In-place KV fusion can bind the full allocation while exposing a live prefix internally. KV-slice on/off is a runtime binding choice and does not require different compiled blobs; incompatible graph layouts/capacities can still do so.

## Stable MTP sampling

The [graph walkthrough](MTP-SAMPLING-GRAPH.md) describes the fixed sampling interface. The backend keeps pure top-k branches for all configured slots, packs runtime row indices, and uses scratch for inactive outputs. It does not change the original cgraph or execute inactive stateful sampler chains. Only the small sampling branches stay at configured capacity; the expensive draft layer processes active rows.

One compatible draft graph is shared across active subsets. Trunk and genuinely different graph stages/configurations remain separate. Prepare a compatible revision-14 cache once, then use cache-only mode. Exhaustive active-subset priming is no longer required, though first-use model import and runtime preparation still have a cost.

## Validation

Native OV tests cover shape/attribute serialization, reset, cycles, duplicates, overlapping writes, mixed precision, changing shapes, aliased/distinct buffers, exclusive Reshapes, shared readers, host/device USM and export/import. Convolution is checked against a host reference; GDN uses an independent cache mapping around ordinary GDN. See `docs/dev/qwen35-llamacpp-recurrent-cache.md` and `recurrent_cache_update.cpp` in the OV checkout for contracts and tests.

## Debugging and source map

- `ggml-decoder.cpp`: dynamic dimensions, control inputs, output names and `cgraph_ov.txt` dump.
- `utils.cpp`: compute-branch filtering, compiled reuse and input/output binding.
- `openvino/recurrent_cache_fusion.cpp`: structural pattern matching and native Update op translation.
- `openvino/translate_session.cpp`: runtime sampling aliases and model construction.
- OV `recurrent_cache_update` implementation: mutation eligibility, source staging and GPU computation.

`GGML_OPENVINO_DUMP_CGRAPH=1` dumps the raw graph before filtering. It includes COMPUTE/OUTPUT flags and byte offsets and overwrites `cgraph_ov.txt` on each call. Enable dumps only for diagnosis; use the captured examples for a static view. GPU profiles and OpenCL copy traces add overhead, and lack of an explicit copy call does not prove zero traffic inside a kernel.
