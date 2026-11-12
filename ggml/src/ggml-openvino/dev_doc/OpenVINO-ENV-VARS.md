# OpenVINO backend environment variables for development

The variables for users, and the rules for values, are in [docs/backend/OPENVINO.md](../../../../docs/backend/OPENVINO.md#ggml-openvino-backend-runtime-configurations). This file lists the other `GGML_OPENVINO_*` variables. They are for debugging, comparisons and experiments, and they can change or be removed.

## Debug output

| Variable | Device | Type | Default | Description |
|---|---|---|---|---|
| `GGML_OPENVINO_DUMP_CGRAPH` | all | Boolean | `0` | Dump the GGML compute graph to `cgraph_ov.txt`. |
| `GGML_OPENVINO_DUMP_IR` | all | Boolean | `0` | Write the OpenVINO IR files, with timestamps. |
| `GGML_OPENVINO_DEBUG_INPUT` | all | Boolean | `0` | Print the input tensors. |
| `GGML_OPENVINO_DEBUG_OUTPUT` | all | Boolean | `0` | Print the output tensors. |
| `GGML_OPENVINO_DEBUG_NODE` | all | String | `not set` | Add the named graph nodes as outputs of the compiled model. Separate several names with commas. |
| `GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS` | all | Boolean | `0` | Print the tensor address map once. |
| `GGML_OPENVINO_LOG_UNSUPPORTED_OPS` | all | Boolean | `0` | Log each op that the backend does not support, with the tensor details and the reason. |
| `GGML_OPENVINO_DISABLE_CACHE` | all | Boolean | `0` | Turn the in-process cache of compiled models and decoders off. |
| `GGML_OPENVINO_NPUW_DUMP_SUBS` | NPU | String | `not set` | Dump NPUW subgraphs: `YES` (all), `MIN` (unique function bodies), `last`, or indices such as `0,2`. |
| `GGML_OPENVINO_NPUW_DUMP_SUBS_DIR` | NPU | String | `not set` | Directory for the NPUW subgraph dumps. |
| `GGML_OPENVINO_NPUW_DUMP_SUBS_ON_FAIL` | NPU | String | `not set` | Passed to OpenVINO unchanged as `NPUW_DUMP_SUBS_ON_FAIL`. |
| `GGML_OPENVINO_NPUW_DUMP_IO` | NPU | String | `not set` | Passed to OpenVINO unchanged as `NPUW_DUMP_IO`. |

```bash
# Dump the GGML graph and the OpenVINO IR of a run
GGML_OPENVINO_DUMP_CGRAPH=1 GGML_OPENVINO_DUMP_IR=1 \
    ./build/ReleaseOV/bin/llama-cli -m model.gguf -c 512

# Log unsupported ops. The lines have the WARN level, which the default log verbosity shows (-lv 2 or higher).
GGML_OPENVINO_LOG_UNSUPPORTED_OPS=1 ./build/ReleaseOV/bin/llama-cli -m model.gguf -c 512

# NPUW dumps. They need an OpenVINO developer build (NPU_PLUGIN_DEVELOPER_BUILD).
export GGML_OPENVINO_DEVICE=NPU
export GGML_OPENVINO_NPUW_DUMP_SUBS=YES
export GGML_OPENVINO_NPUW_DUMP_SUBS_DIR=/tmp/npuw_dump   # optional
./build/ReleaseOV/bin/llama-cli -m model.gguf -c 512
```

## Switches for default optimizations

Each variable turns off an optimization that is on by default, or sets its strength. Use them to find the cause of a wrong result or of a speed change.

| Variable | Device | Type | Default | Description |
|---|---|---|---|---|
| `GGML_OPENVINO_KV_SLICE` | all | Boolean | `1` | Bind the KV cache, and size the KQ masks, at the attended length instead of the full context. CPU and GPU: a view over the cache. NPU: the prefill graph is compiled at the attended length, rounded up to a power of two, and replaced when that size changes. With `0`, `llama-bench -d` fails on the NPU. |
| `GGML_OPENVINO_NPU_BLOB_PREBUILD` | NPU | Boolean | `1` | Decode attends the smallest `1024 * 2^k` rows that cover the next position, with one decode graph for each window. All windows are built, or read from the cache, together with the first prefill and decode pair. `0`: one decode graph attends the whole context. Use the same value in the run that writes the cache and in the runs that read it. |
| `GGML_OPENVINO_NPU_FAST_MASK` | NPU | Boolean | `1` | Write the prefill attention mask directly into the input tensor, without an intermediate copy. The result is identical. |
| `GGML_OPENVINO_NPU_L0_HOST_TENSORS` | NPU | Boolean | `1` | Keep the KV cache and the compute buffers in Level Zero host memory, so the plugin reads them without an extra copy. Falls back to plain host memory if the allocation is not available. |
| `GGML_OPENVINO_GDN_Q_SCALE` | NPU | Float | `64` | Factor of the GDN q pre-scaling, which keeps the GDN output above the FP16 flush-to-zero floor of the NPU. `0` or a negative value turns it off. Values below `64` are raised to `64`. |
| `GGML_OPENVINO_MOE_OP` | GPU | Boolean | `1` | `0` keeps the unfused GatherMatmul path for mixture-of-experts layers. |
| `GGML_OPENVINO_DISABLE_SHAPE_FROM_MASK` | CPU, GPU | Boolean | `0` | By default the model reads the sequence count and the attended KV length from the shape of the KQ mask, so the GPU plugin can select its fused attention kernel. `1` passes them as separate inputs. |
| `GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT` | CPU, GPU | Boolean | `0` | Stateful execution only. By default the sequence axis of the KV state is moved so that new tokens are appended in place. `1` keeps the original layout. |

```bash
# Find out whether an optimization causes a wrong result: turn it off and compare the output
GGML_OPENVINO_DEVICE=NPU GGML_OPENVINO_NPU_FAST_MASK=0 \
    ./build/ReleaseOV/bin/llama-completion -m model.gguf -c 512 -n 32 --temp 0 -no-cnv -p "The capital of France is"
```

## Experimental options

These options are off by default. They are not validated for general use: some help on one model or device and give a wrong or slower result on another.

| Variable | Device | Type | Default | Description |
|---|---|---|---|---|
| `GGML_OPENVINO_NPU_DECODE_UNFOLD` | NPU | Boolean | `0` | Set `NPUW_UNFOLD_IREQS=YES` on the decode graph only. An explicit `GGML_OPENVINO_NPUW_UNFOLD_IREQS` or `GGML_OPENVINO_NPU_CONFIG` value takes precedence. |
| `GGML_OPENVINO_NPU_PARALLEL_COMPILE` | NPU | Boolean | `0` | Compile the prefill and decode graphs in parallel. The load is shorter and the host memory peak during compilation is higher. |
| `GGML_OPENVINO_NPU_KV_ON_DEMAND` | NPU | Boolean | `0` | Keep the ggml KV buffer non-resident and bind storage for each layer that grows with the rows in use, so a large `-c` costs memory only for the context that is reached. K is stored head-major and V transposed, as OpenVINO GenAI does. `llama-perplexity` fails with it. |
| `GGML_OPENVINO_NPU_REQUANT_POLICY` | NPU | String | `group-128` | Layout of requantized weights: `group-128` (`Q4_0_128`) or `channel-wise` (`Q4_0_C`, lower accuracy). |
| `GGML_OPENVINO_GDN_DECOMPOSE_CHUNK` | all | Integer | `16` | Tokens in one chunk when `GGML_OPENVINO_GDN_DECOMPOSE` is on. A power of 2 from 2 to 128; other values use `16`. |
| `GGML_OPENVINO_GDN_NATIVE_STATE` | NPU | Boolean | `0` | Keep the `GATED_DELTA_NET` state in the layout of the OpenVINO op instead of the GGML layout. Every `GATED_DELTA_NET` must then run on the OpenVINO backend; the load aborts if one would run on the CPU backend. Saved states and `GGML_OPENVINO_DEBUG_NODE` dumps hold the OpenVINO layout. `test-backend-ops` fails with it. |
| `GGML_OPENVINO_KV_SCATTER_ELEMENTS` | all | Boolean | `0` | Use `ScatterElementsUpdate` instead of `ScatterUpdate` for single-row KV writes in stateless execution. |
| `GGML_OPENVINO_ENABLE_KV_4D` | GPU | Boolean | `0` | Bind the KV cache as `[1, ctx, n_heads_kv, head_size]`, so the GPU plugin can update and read it in place. Needs an OpenVINO build newer than 2026.4.1; do not use it with 2026.4.1. |
| `GGML_OPENVINO_MEMORY_OPTIMIZE` | all | Boolean | `0` | Turn on `GGML_OPENVINO_REDUCE_COMPILE_MEM` and, on GPU, `GGML_OPENVINO_RELEASE_WEIGHTS`, unless they are set. |
| `GGML_OPENVINO_REDUCE_COMPILE_MEM` | all | Boolean | value of `GGML_OPENVINO_MEMORY_OPTIMIZE` | Create fewer weight nodes during compilation, to lower the host memory in use. |
| `GGML_OPENVINO_RELEASE_WEIGHTS` | GPU | Boolean | value of `GGML_OPENVINO_MEMORY_OPTIMIZE` | Release the host weight buffers when the compiled model holds its own copy. Needs stable graph shapes; do not use it for workloads that compile again. |
| `GGML_OPENVINO_SPILL_DIR` | all | String | `not set` | Directory for a disk-backed weight buffer, so its pages can be reclaimed under memory pressure. It must be on real storage, not on tmpfs (`/tmp` on many systems). It is not part of the key of the compiled model cache and is ignored in cache-only mode. |
| `GGML_OPENVINO_SELF_CONTAINED_BLOB` | NPU | Boolean | `0` | Requires `GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR`. Write the prefill and decode graphs as blobs with the weights inside; each blob takes about the size of the model on disk. Decode then attends the whole context. `GGML_OPENVINO_NPU_WEIGHTLESS_BANK` is the recommended cache layout for the NPU. |
| `GGML_OPENVINO_NPU_SHARED_BANK` | NPU | Boolean | `0` | With `GGML_OPENVINO_SELF_CONTAINED_BLOB=1`: one weights bank for the prefill and decode blobs. With OpenVINO 2026.4.1 the blobs cannot be read back, so decode is compiled again. |
| `GGML_OPENVINO_LAZY_HOST_THRESHOLD_MB` | NPU | Integer | `2048` | With `GGML_OPENVINO_SELF_CONTAINED_MMAP=1`: buffers of at least this size get demand-zero host memory instead of Level Zero host memory. |

```bash
# NPU cache with the weights inside the blobs (experimental). Keep all other NPU settings the same in both runs.
export GGML_OPENVINO_DEVICE=NPU
export GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR=/tmp/ov_cache
export GGML_OPENVINO_SELF_CONTAINED_BLOB=1
./build/ReleaseOV/bin/llama-cli -m model.gguf -c 512 -lv 4   # first run: log shows "model cache WROTE"

export GGML_OPENVINO_SELF_CONTAINED_MMAP=1
./build/ReleaseOV/bin/llama-cli -m model.gguf -c 512 -lv 4   # later runs: log shows "model cache HIT"
```

## NPU plugin properties

The backend sets the properties below for the NPU. `GGML_OPENVINO_NPU_CONFIG` can change them and add others.

- **Backend default**: the value that the backend passes to OpenVINO when the user sets nothing.
- **OpenVINO default**: the value that OpenVINO uses when nobody sets the property.
- Values are those of OpenVINO, for example `YES` and `NO`.

| Property | Type | Backend default | OpenVINO default | Description |
|---|---|---|---|---|
| `NPU_USE_NPUW` | Boolean | `YES` | `NO` | Use NPUW, the OpenVINO layer that splits a model into parts and compiles them for the NPU. |
| `NPUW_DEVICES` | String | `NPU` | `NPU,CPU` | Devices that NPUW tries for each part, in order. With `NPU` only, no part runs on the CPU. |
| `NPUW_FOLD` | Boolean | `YES` | `NO` | Compile the repeating blocks of the model once and reuse them. |
| `NPUW_FUNCALL_FOR_ALL` | Boolean | `YES` | `NO` | Make every part of the model a function, not only the repeating blocks. |
| `NPUW_FUNCALL_ASYNC` | Boolean | `YES` | `NO` | Run a function and the preparation of the next one as a pipeline. |
| `NPUW_WEIGHTS_BANK` | String | `shared` | empty | Name of the weights bank. Graphs with the same name share one copy of the weights. |
| `NPUW_DQ` | Boolean | `YES` | `NO` | Apply the dynamic quantization transformations in the plugin. |
| `NPUW_DQ_FULL` | Boolean | `NO` | `YES` | Apply the full dynamic quantization pipeline in the plugin. |
| `NPU_COMPILER_DYNAMIC_QUANTIZATION` | Boolean | `YES` | `NO` | Dynamic quantization in the NPU compiler. |
| `NPU_COMPILER_TYPE` | String | `DRIVER` | `PREFER_PLUGIN` | The NPU compiler: `DRIVER` (in the NPU driver), `PLUGIN` (in OpenVINO) or `PREFER_PLUGIN` (`PLUGIN` when it is available, `DRIVER` if not). Models with `GATED_DELTA_NET` layers (Qwen3.5) compile with `PLUGIN` unless the user sets this property. |
| `NPUW_CACHE_DIR` | String | value of `GGML_OPENVINO_CACHE_DIR` | empty | Directory for the compiled parts. |

Other properties that are useful with `GGML_OPENVINO_NPU_CONFIG`. The backend does not set them.

| Property | Type | Backend default | OpenVINO default | Description |
|---|---|---|---|---|
| `COMPILATION_NUM_THREADS` | Integer | `not set` | number of CPU cores | Threads of the NPU compiler. A lower value lowers the host memory peak when a large model compiles. |
| `NPUW_UNFOLD_IREQS` | Boolean | `not set` | `NO` | One infer request for each part, also for repeating blocks. |
| `LOG_LEVEL` | String | `not set` | `LOG_ERROR` | Log level of the NPU plugin, for example `LOG_INFO`. |

```bash
export GGML_OPENVINO_DEVICE=NPU
export GGML_OPENVINO_NPU_CONFIG="NPU_COMPILER_TYPE=PLUGIN,COMPILATION_NUM_THREADS=1,LOG_LEVEL=LOG_INFO"
```

```cpp
// List the properties of the installed NPU plugin with their current values
ov::Core core;
for (const auto & name : core.get_property("NPU", ov::supported_properties)) {
    std::cout << name << " = " << core.get_property("NPU", name).as<std::string>() << "\n";
}
```

- All NPU properties: [OpenVINO NPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html).
- `NPUW_*` options, with type and default: `src/plugins/intel_npu/src/al/include/intel_npu/config/npuw_option_defs.inc` in the OpenVINO source.
- Graphs without a KV cache compile without NPUW: the backend removes `NPU_USE_NPUW` and the `NPUW_*` keys for them.

## GPU and CPU plugin properties

The backend sets no GPU or CPU property, except `CACHE_DIR` and `CACHE_MODE=OPTIMIZE_SIZE` when `GGML_OPENVINO_CACHE_DIR` is set.

```bash
# CPU in FP16, to compare with the GPU and the NPU, which compute in FP16
GGML_OPENVINO_DEVICE=CPU GGML_OPENVINO_CPU_CONFIG="INFERENCE_PRECISION_HINT=f16" \
    ./build/ReleaseOV/bin/llama-cli -m model.gguf -c 512
```

All properties: [OpenVINO GPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html), [OpenVINO CPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html).

## Older variable names

Each of these variables sets one NPU property. They still work. `GGML_OPENVINO_NPU_CONFIG` sets the same properties and takes precedence when both are set.

| Variable | Device | Type | Default | Description |
|---|---|---|---|---|
| `GGML_OPENVINO_NPU_COMPILE_CONFIG` | NPU | String | `not set` | Sets `NPU_COMPILATION_MODE_PARAMS`. |
| `GGML_OPENVINO_NPU_COMPILER_TYPE` | NPU | String | `not set` | Sets `NPU_COMPILER_TYPE`. |
| `GGML_OPENVINO_NPUW_FUNCALL_FOR_ALL` | NPU | String | `not set` | Sets `NPUW_FUNCALL_FOR_ALL`. |
| `GGML_OPENVINO_NPUW_UNFOLD_IREQS` | NPU | String | `not set` | Sets `NPUW_UNFOLD_IREQS`. |
| `GGML_OPENVINO_COMPILATION_NUM_THREADS` | NPU | String | `not set` | Sets `COMPILATION_NUM_THREADS`. |

```bash
# These two lines do the same. Use the first form.
export GGML_OPENVINO_NPU_CONFIG="NPUW_UNFOLD_IREQS=YES"
export GGML_OPENVINO_NPUW_UNFOLD_IREQS=YES
```
