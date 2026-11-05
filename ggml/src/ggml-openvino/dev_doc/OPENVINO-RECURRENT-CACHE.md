# Current Qwen GPU implementation

On the input-inplace development branch, the default Qwen3.5/Qwen3.6 path is dynamic stateless execution with `RecurrentConvCacheUpdate`, `RecurrentGatedDeltaNetUpdate`, and stable pure-top-k MTP translation. It supports one or multiple slots and rollback. The implementation applies to `qwen35` and `qwen35moe`; the reference model is the UD Qwen3.6-35B-A3B MTP GGUF.

Build against the matching OpenVINO runtime/GPU plugin with single-output Update ops. Configure `GGML_OPENVINO_DEVICE=GPU`, native softplus, and the documented requantization policy. The cache translation revision is 15. Historical measurements retain their original source revisions.

The separate dynamic output-binding fix in OpenVINO PR #39354 remains required for cache-only MTP serving. Native cache-op export/import tests pass without it, but Qwen3.6 imported MTP graphs still trigger the missing-input-memory error without the fix.

## Buffer ownership and operation boundaries

| State | Storage and lifetime | Update path |
|---|---|---|
| Attention K/V | Full caller-owned USM allocation, sized by context and KV layout | Native stateless KV fusion when eligible; SDPA consumes the live prefix |
| Convolution `cache_r` | Full caller-owned USM row bank, sized by slot/snapshot capacity | Fused reset, original-row gather/reorder, convolution and ordered history/snapshot writes |
| GDN `cache_s` | Full caller-owned USM row bank, sized by slot/snapshot capacity | Fused reset, original-row gather/reorder, GDN and ordered final/snapshot writes |
| Runtime controls | Reused USM host storage on GPU | Row maps, offsets and write controls remain runtime inputs |
| Protected source rows | Private device scratch | Staged before writes to protect cycles, duplicate sources and overlaps |
| Inactive sampling results | Private device-USM scratch | Discarded without touching inactive GGML output buffers |

Attention context capacity does not set recurrent-cache size. Each cache input Parameter binds a stable writable allocation; recurrent cache Results are omitted. There are no rotation banks or pointer swaps.

The GPU plugin uses two stages: first preserve required original rows in scratch while applying logical reset and precision rounding; then compute and perform ordered writes. This avoids overwriting a source before another destination has read it. Identity reorder sources need no scratch writes. In-place execution removes the spare full output bank, not every cache read or internal memory transaction.

`RecurrentConvCacheUpdate` has nine inputs, including depthwise weights, and returns convolution `[1,B,T,C]`. `RecurrentGatedDeltaNetUpdate` returns attention and writes the newest `min(T,snapshot_count)` states into its cache input, newest first. Neither op returns a cache tensor.

## Generic fallback and restrictions

Writable cache inputs must be GPU RemoteTensors with exclusive graph readers, optionally through Reshape aliases. The GPU plugin rejects shared cache readers and computed cache banks. Other inputs and the computation output must not overlap the cache. The frontend skips fusion for unmatched cache reads and retains functional cache writeback. Restore or clear state after a failed inference before retrying.

`GGML_OPENVINO_DISABLE_FUSED_RECURRENCE=1` retains ordinary cache decomposition. Naive graphs skip full-model recurrence fusion. Unmatched snapshot GDN uses ordinary GDN token prefixes to preserve state outputs. The fused path requires the matching native runtime; dynamic GPU fallback can use ordinary GDN. Stateful and NPU MTP are outside this change.

## Attention KV layout

`GGML_OPENVINO_ENABLE_KV_4D=1` enables eligible native stateless in-place attention updates. The current eligibility covers one physical stream, including unified KV. Explicit split multi-slot KV retains the full-cache path and can be much slower even with only one active slot. Use `--kv-unified` for multi-slot OV GPU serving.

Leave `GGML_OPENVINO_KV_SLICE` at its default (`1`). Live-prefix slicing applies to eligible ordinary attention K/V roots, not recurrent caches. In-place KV fusion can bind the full allocation while exposing a live prefix internally. KV-slice on/off is a runtime binding choice and does not require different compiled blobs; incompatible graph layouts/capacities can still do so.

## Stable MTP sampling

The [graph walkthrough](MTP-SAMPLING-GRAPH.md) describes the fixed sampling interface. The backend keeps pure top-k branches for all configured slots, packs runtime row indices, and uses scratch for inactive outputs. It does not change the original cgraph or execute inactive stateful sampler chains. Only the small sampling branches stay at configured capacity; the expensive draft layer processes active rows.

One compatible draft graph is shared across active subsets. Trunk and genuinely different graph stages/configurations remain separate. Prepare a compatible revision-15 cache once, then use cache-only mode. Exhaustive active-subset priming is no longer required, though first-use model import and runtime preparation still have a cost.

## Validation

Native OV tests cover shape/attribute serialization, reset, cycles, duplicates, overlapping writes, mixed precision, changing shapes, writable cache input rebinding, exclusive Reshapes, shared readers, host/device USM and export/import. Convolution is checked against a host reference; GDN uses an independent cache mapping around ordinary GDN. See `docs/dev/qwen35-llamacpp-recurrent-cache.md` and `recurrent_cache_update.cpp` in the OV checkout for contracts and tests.

## Debugging and source map

- `ggml-decoder.cpp`: dynamic dimensions, control inputs, output names and `cgraph_ov.txt` dump.
- `utils.cpp`: compute-branch filtering, compiled reuse and input/output binding.
- `openvino/recurrent_cache_fusion.cpp`: structural pattern matching and native Update op translation.
- `openvino/translate_session.cpp`: runtime sampling aliases and model construction.
- OV `recurrent_cache_update` implementation: writable-input validation, source staging and GPU computation.

`GGML_OPENVINO_DUMP_CGRAPH=1` dumps the raw graph before filtering. It includes COMPUTE/OUTPUT flags and byte offsets and overwrites `cgraph_ov.txt` on each call. Enable dumps only for diagnosis; use the captured examples for a static view. GPU profiles and OpenCL copy traces add overhead, and lack of an explicit copy call does not prove zero traffic inside a kernel.
