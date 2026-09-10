# llama.cpp for MACA

The MACA backend provides GPU acceleration on MetaX devices. It compiles the
existing `ggml-cuda` implementation through the MACA CUDA compatibility bridge
and produces a distinct `ggml-maca` backend library and `MACA<n>` device names.

## Design

The backend shares CUDA-family scheduling and kernels where the compatibility
bridge provides equivalent semantics. MACA-specific guards prevent the virtual
CUDA compute capability from selecting NVIDIA-only MMA, PTX, VMM, peer-copy,
and collective paths.

The initial implementation uses these matrix multiplication routes:

| Workload | Route |
| --- | --- |
| F16/F32 matrix multiplication | CUDA-compatible path backed by mcBLAS |
| Quantized matrix multiplication, including prefill and decode | dequantization followed by mcBLAS |
| Unvalidated fused or indexed operations | conservative fallback |

## Requirements

- A MetaX GPU with a compatible driver
- Linux
- A MACA SDK containing `tools/cu-bridge`
- CMake and Ninja

The backend has been validated with MACA 3.2.x. SDK layouts and wrapper
behavior can differ between releases.

## Build

The helper script configures the virtual CUDA toolchain, detects the target
device when possible, builds `ggml-maca` conservatively, and then builds the
host tools in parallel:

```bash
export MACA_PATH=/opt/maca
bash scripts/maca/build.sh
```

When automatic target detection is unavailable, set one of the SDK target
variables before building:

```bash
export CUCC_TARGETS_FROM_DEVICE=xcore1000
bash scripts/maca/build.sh
```

The `CMAKE_CUDA_ARCHITECTURES` value used by the compatibility toolchain is a
placeholder. The actual device target is selected by `CUCC_TARGETS` or
`CUCC_TARGETS_FROM_DEVICE`.

### Build variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MACA_PATH` | `/opt/maca` | MACA SDK root |
| `MACA_CU_BRIDGE` | `$MACA_PATH/tools/cu-bridge` | Compatibility bridge root |
| `CUBRIDGE_HOME` | `$HOME` | Parent directory where the SDK creates `cu-bridge`; `WCUDA_HOME`, when set, takes precedence |
| `MACA_VIRTUAL_ROOT` | `$CUBRIDGE_HOME/cu-bridge` | Must match the SDK-generated toolchain location |
| `MACA_BUILD_DIR` | `build-maca` | CMake build directory |
| `MACA_BUILD_JOBS` | `8` | Parallel jobs for host targets |
| `CUCC_TARGETS` | unset | Explicit multi-target architecture list |
| `CUCC_TARGETS_FROM_DEVICE` | auto-detected | Device architecture selected by the SDK |
| `MACA_GRAPHS` | `OFF` | Experimental graph capture and replay in the helper build |

To keep SDK-generated files in a workspace, set `CUBRIDGE_HOME` to an existing writable directory, unset `WCUDA_HOME`, and leave `MACA_VIRTUAL_ROOT` and `MACA_VIRTUAL_CUDA` unset. When changing the toolchain location, use a new `MACA_BUILD_DIR` to avoid stale compiler paths in the CMake cache.

## Verify the backend

List the detected devices:

```bash
export MACA_PATH=/opt/maca
export MACA_BUILD_DIR="$PWD/build-maca"
export LD_LIBRARY_PATH="$MACA_BUILD_DIR/bin:$MACA_PATH/lib:${MACA_CU_BRIDGE:-$MACA_PATH/tools/cu-bridge}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
./build-maca/bin/llama-cli --list-devices
```

Run the Q8_0 matrix multiplication coverage and a model benchmark:

```bash
bash scripts/maca/validate.sh /path/to/model.gguf
```

Each run creates a unique model-named subdirectory under `${MACA_TEST_OUTPUT:-$MACA_BUILD_DIR/validation}` and prints its location. It contains `devices.log`, `q8-mul-mat.log`, `benchmark.jsonl`, and `benchmark.log`. Set `MACA_DEVICE` to select one device for both operator tests and the benchmark (default: `MACA0`).

The operator test always covers Q8_0 x F32 `MUL_MAT`, regardless of the supplied GGUF format. The benchmark uses the supplied model. Testing a Q4 or K-quant model does not establish standalone operator coverage for that format. An unknown device, a failed test, or an empty test summary causes the script to fail.

Individual backend operation coverage can also be run directly:

```bash
./build-maca/bin/test-backend-ops \
  -b MACA0 \
  -o MUL_MAT \
  -p 'type_a=q8_0,type_b=f32' \
  -j 1
```

## Run a model

Interactive inference:

```bash
./build-maca/bin/llama-cli \
  -m /path/to/model.gguf \
  -ngl 99 \
  -c 4096 \
  --simple-io
```

OpenAI-compatible HTTP inference:

```bash
./build-maca/bin/llama-server \
  -m /path/to/model.gguf \
  -ngl 99 \
  --host 127.0.0.1 \
  --port 8080
```

For multi-device layer splitting:

```bash
./build-maca/bin/llama-bench \
  -m /path/to/model.gguf \
  -ngl 99 \
  --device MACA0/MACA1/MACA2/MACA3 \
  --split-mode layer \
  --tensor-split 1/1/1/1 \
  -p 512 -n 512
```

Small models may not benefit from multi-device splitting because transfer and
synchronization overhead can exceed the compute saved on each device.

## Validation matrix

Earlier development versions were reported tested with the configurations below. These historical results are not a claim that every configuration was rerun on the current source revision; attach revision-specific logs when reporting validation.

| Hardware | Model | Formats | Coverage |
| --- | --- | --- | --- |
| MetaX C500, one device | Qwen2.5-3B-Instruct | F16, Q8_0, Q4_0 | backend ops, benchmark, CLI, server |
| MetaX C500, one device | Llama-3.2-3B-Instruct | F16, Q8_0, Q4_0 | backend ops, benchmark, inference |
| MetaX C600-A, one device | Qwen2.5-3B-Instruct | F16, Q8_0, Q4_0 | backend ops, benchmark, inference |
| MetaX C600-A, one device | Llama-3.2-3B-Instruct | F16, Q8_0, Q4_0 | backend ops, benchmark, inference |
| MetaX C600-A, four devices | Qwen2.5-3B-Instruct | F16, Q8_0, Q4_0 | per-device execution and layer split |
| MetaX C600-A, four devices | Llama-3.2-3B-Instruct | F16, Q8_0, Q4_0 | per-device execution and layer split |

The current post-rebase C500 run passed 49 Q8_0 x F32 `MUL_MAT` cases. Model benchmarks and interactive smoke tests also completed for Llama-3.2-3B-Instruct Q8_0 and Qwen2.5-3B-Instruct Q4_0, Q2_K, and Q3_K_M. C600-A and multi-device results for this revision remain to be refreshed. Interactive generation is a smoke test, not a full numerical accuracy test.

## Current limitations

The initial backend keeps features without dedicated correctness and stability
coverage disabled or on conservative fallbacks:

- Flash Attention is disabled.
- NCCL is disabled.
- virtual memory management is disabled.
- peer copy is disabled.
- MMVQ, quantized fusion, and optimized `MUL_MAT_ID` paths are disabled;
  quantized matrix multiplication uses the dequantization plus mcBLAS fallback.
- graph capture and replay are experimental and disabled by default; set
  `MACA_GRAPHS=ON` when using the helper build, or configure CMake with
  `-DGGML_MACA_GRAPHS=ON`, for targeted testing.

Dedicated MMVQ and integer dot-product kernels remain candidates for follow-up
optimization after targeted correctness and performance validation.
