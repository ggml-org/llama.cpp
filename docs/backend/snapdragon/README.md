# Snapdragon-based devices

## Setup

The cross-compilation toolchain images are provided by the
[Qualcomm Snapdragon Toolchain registry](https://github.com/snapdragon-toolchain).
These Docker images include the Android NDK, OpenCL SDK, Hexagon SDK, CMake, and the necessary cross-compilers:

* **Android toolchain**: `ghcr.io/snapdragon-toolchain/arm64-android:v0.7`
* **Linux toolchain**: `ghcr.io/snapdragon-toolchain/arm64-linux:v0.7`

The unified build utility (`scripts/snapdragon/build.py`) automatically pulls
and orchestrates these containers to perform target compilation.
You only need to ensure that Docker (or Docker Desktop on macOS/Windows) is running on your host machine.
Specific setup, build, and installation details for Linux and Windows on Snapdragon platforms are documented in:
* [Linux on Snapdragon guide](linux.md)
* [Windows on Snapdragon guide](windows.md)

## How to Build

### Using build.py script (Recommended)

The easiest way to build llama.cpp is by using the `scripts/snapdragon/build.py` script. It automatically copies the CMake presets,
launches the correct compilation Docker container, builds the libraries and tools,
installs them, and optionally pushes them to your ADB device.

Build and deploy for Android target (accepts `android` or `adb` alias):
```
$ ./scripts/snapdragon/build.py --target adb --push
```

Build and deploy for Linux target (accepts `linux` or `lnx` alias):
```
$ ./scripts/snapdragon/build.py --target linux:user@host --push
```

### Manual CMake Build

Alternatively, you can build llama.cpp manually by entering the cross-compilation Docker container and running the CMake commands:

```bash
# Start the cross-compilation container manually:
~/src/llama.cpp$ docker run -it --rm -u $(id -u):$(id -g) --volume $(pwd):/workspace --platform linux/amd64 ghcr.io/snapdragon-toolchain/arm64-android:v0.7

# Inside the container, build the project using presets:
[d]/workspace> cp docs/backend/snapdragon/CMakeUserPresets.json .

[d]/workspace> cmake --preset arm64-android-snapdragon-release -B build-snapdragon
Preset CMake variables:
  ANDROID_ABI="arm64-v8a"
  ...
  CMAKE_TOOLCHAIN_FILE="/opt/android-ndk-r28b/build/cmake/android.toolchain.cmake"
  GGML_HEXAGON="ON"
  GGML_OPENCL="ON"
  GGML_OPENMP="OFF"
  HEXAGON_SDK_ROOT="/opt/hexagon/6.6.0.0"
...
-- Including OpenCL backend
-- Including Hexagon backend
...
-- Build files have been written to: /workspace/build-snapdragon

[d]/workspace> cmake --build build-snapdragon
...
[144/356] Performing build step for 'htp-v73'
[1/16] Generating htp_iface_skel.c, htp_iface_stub.c, htp_iface.h
[2/16] Building C object CMakeFiles/ggml-htp-v73.dir/hvx-sigmoid.c.obj
[3/16] Building C object CMakeFiles/ggml-htp-v73.dir/htp-dma.c.obj
[4/16] Building C object CMakeFiles/ggml-htp-v73.dir/worker-pool.c.obj
...
-- Installing: /workspace/build-snapdragon/ggml/src/ggml-hexagon/libggml-htp-v73.so
-- Installing: /workspace/build-snapdragon/ggml/src/ggml-hexagon/libggml-htp-v75.so
...
```

To generate an installable "package" simply use cmake --install:

```
[d]/workspace> cmake --install build-snapdragon --prefix pkg-android/llama.cpp
-- Install configuration: "Release"
-- Installing: /workspace/pkg-android/llama.cpp/lib/libggml-cpu.so
-- Installing: /workspace/pkg-android/llama.cpp/lib/libggml-opencl.so
-- Installing: /workspace/pkg-android/llama.cpp/lib/libggml-hexagon.so
-- Installing: /workspace/pkg-android/llama.cpp/lib/libggml-htp-v73.so
-- Installing: /workspace/pkg-android/llama.cpp/lib/libggml-htp-v75.so
-- Installing: /workspace/pkg-android/llama.cpp/lib/libggml-htp-v79.so
-- Installing: /workspace/pkg-android/llama.cpp/lib/libggml-htp-v81.so
-- Installing: /workspace/pkg-android/llama.cpp/lib/libggml.so
...
-- Installing: /workspace/pkg-android/llama.cpp/bin/llama-bench
-- Installing: /workspace/pkg-android/llama.cpp/bin/llama-cli
...
```

## How to Install

### Android

For this step, your device needs to be configured for on-device development.
Please see https://developer.android.com/studio/debug/dev-options for details.

Once ADB is enabled, use `adb push` to install `pkg-android` on the device.
**Note that the toolchain Docker image doesn't have ADB and doesn't set up the ADB bridge. Please use native ADB on the host.**

```
~/src/llama.cpp$ adb push pkg-android/llama.cpp /data/local/tmp/
pkg-android/llama.cpp/bin/: 67 files pushed, 0 skipped. 190.2 MB/s (919095042 bytes in 4.607s)
pkg-android/llama.cpp/include/: 19 files pushed, 0 skipped. 20.5 MB/s (255173 bytes in 0.012s)
pkg-android/llama.cpp/lib/: 16 files pushed, 0 skipped. 144.4 MB/s (43801382 bytes in 0.289s)
102 files pushed, 0 skipped. 186.9 MB/s (963151597 bytes in 4.914s)
```

At this point, you should also install some models:

```
~/src/llama.cpp$ wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf
...
2025-10-11 12:04:52 (10.7 MB/s) - ‘Llama-3.2-1B-Instruct-Q4_0.gguf’ saved [773025920/773025920]

~/src/llama.cpp$ adb push Llama-3.2-1B-Instruct-Q4_0.gguf /data/local/tmp/gguf
Llama-3.2-1B-Instruct-Q4_0.gguf: 1 file pushed, 0 skipped. 38.3 MB/s (773025920 bytes in 19.250s)
```

### Windows

All artifacts are already installed in the `pkg-wos` folder.
To run, you can use the `scripts/snapdragon/run.py` runner script (see details below).

## How to Run

The easiest way to run llama.cpp cli tools is using the provided `scripts/snapdragon/run.py` wrapper script. This script automatically
maps CLI options to environment variables, resolves executable paths, and runs the command locally, via ADB, or remotely via SSH on the
target device.

llama.cpp supports three backends on Snapdragon-based devices: CPU, Adreno GPU (GPUOpenCL), and Hexagon NPU.
You can select which backend(s) to run the model on using the `--device` option of the tool (or `--devices` option in `run.py`).

Hexagon NPU behaves as a "GPU" device when it comes to `-ngl` and other offload-related options.

Here are some examples of running various llama.cpp tools.

Generating a completion with Gemma on Android (relying on default `HTP0:0` device and default thread count `-t 6`):

```
~/src/llama.cpp$ ./scripts/snapdragon/run.py --target adb -- llama-completion -m models/gemma-2-2b-it-Q4_0.gguf -f prompts/sample_prompt_1024.txt --jinja -st
...
ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev 1
ggml-hex: Hexagon Arch version v79
ggml-hex: allocating new session: HTP0:0
...
load_tensors: offloading output layer to GPU
load_tensors: offloaded 27/27 layers to GPU
load_tensors:          CPU model buffer size =   300.00 MiB
load_tensors:      HTP0:0 model buffer size  =  1400.26 MiB
...
llama_perf_context_print: prompt eval time =     320.00 ms /  1024 tokens (    0.31 ms per token,  3200.00 tokens per second)
llama_perf_context_print:        eval time =     2100.00 ms /   100 runs   (   21.00 ms per token,    47.62 tokens per second)
```

Simple question for Llama-3.2-1B:

```
~/src/llama.cpp$ ./scripts/snapdragon/run.py --target android --devices HTP0 -- llama-cli -m Llama-3.2-1B-Instruct-Q4_0.gguf -p "what is the most popular cookie in the world?"
...
ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev 1
ggml-hex: Hexagon Arch version v79
ggml-hex: allocating new session: HTP0
ggml-hex: new session: HTP0 : session-id 0 domain-id 3 uri file:///libggml-htp-v79.so?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0 handle 0xb4000072c7955e50
...
load_tensors: offloading output layer to GPU
load_tensors: offloaded 17/17 layers to GPU
load_tensors:          CPU model buffer size =   225.49 MiB
load_tensors:         HTP0 model buffer size =   504.26 MiB
...
I hope this helps you understand the world's most popular cookies! [end of text]
...
llama_perf_sampler_print:    sampling time =      30.08 ms /   487 runs   (    0.06 ms per token, 16191.77 tokens per second)
llama_perf_context_print:        load time =     617.94 ms
llama_perf_context_print: prompt eval time =      80.76 ms /    11 tokens (    7.34 ms per token,   136.21 tokens per second)
llama_perf_context_print:        eval time =    9210.59 ms /   475 runs   (   19.39 ms per token,    51.57 tokens per second)
llama_perf_context_print:       total time =    9454.92 ms /   486 tokens
llama_perf_context_print:    graphs reused =        473
llama_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - HTP0 (Hexagon)     |  2048 = 2048 + (   0 =     0 +       0 +       0) +           0 |
llama_memory_breakdown_print: |   - Host               |                  439 =   225 +     136 +      77                |
```

Op test for MUL_MAT:

```
~/src/llama.cpp$ ./scripts/snapdragon/run.py --target adb --devices HTP0:0 -- test-backend-ops -b HTP0:0 -o MUL_MAT
...
Backend 2/3: HTP0:0
Device description: Hexagon
Device memory: 2048 MB (2048 MB free)
MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],v=0,o=1): OK
MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=2,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],v=0,o=1): OK
MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=3,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],v=0,o=1): OK
```

Llama benchmark:

```
~/src/llama.cpp$ ./scripts/snapdragon/run.py --target adb --devices HTP0 -- llama-bench -p 128 -n 64 -m Llama-3.2-1B-Instruct-Q4_0.gguf
...
ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev 1
ggml-hex: Hexagon Arch version v79
ggml-hex: allocating new session: HTP0
ggml-hex: new session: HTP0 : session-id 0 domain-id 3 uri file:///libggml-htp-v79.so?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0 handle 0xb400007d4b231090
| model          |       size | params | backend    | ngl | threads | n_batch | mmap |  test |           t/s |
| ---------------| ---------: | -----: | ---------- | --: | ------: | ------: | ---: | ----: | ------------: |
| llama 1B Q4_0  | 729.75 MiB | 1.24 B | HTP        |  99 |       4 |     128 |    0 | pp128 | 169.42 ± 1.75 |
| llama 1B Q4_0  | 729.75 MiB | 1.24 B | HTP        |  99 |       4 |     128 |    0 |  tg64 |  51.54 ± 1.13 |
```

## Multi-Device Execution Modes

The Hexagon backend supports multiple execution and partitioning modes to accommodate different model sizes, memory
constraints, and single- or multi-NPU hardware topologies:

### 1. Single-Device Mode with Dynamic Buffer Mapping

Runs the model on a single NPU session (e.g. `HTP0` or `HTP0:0`).

A single NPU session provides ~3.5GB of available virtual address space. For models larger than 3.5GB, the backend
automatically maps and unmaps weight buffers during graph execution. This allows large models to run on a single NPU
without manual configuration:

```bash
./scripts/snapdragon/run.py --target adb --devices HTP0:0 -- \
    llama-cli -m models/Llama-3.2-3B-Instruct-Q4_0.gguf -ngl 99 -p "Hello"
```

### 2. Layer-Split Mode across Virtual Sessions (`HTP0,HTP1,...` or `HTP0:0,HTP0:1,...`)

Partitions model layers at load time across multiple virtual sessions hosted on a single physical NPU.

Each virtual session acts as an independent backend device from llama.cpp's perspective (similar to multiple GPUs).
Because layers are permanently distributed across sessions, each session's allocated weights remain within its private 3.5GB
address space window, eliminating runtime buffer re-mapping overhead.

Here is an example of running the GPT-OSS-20B model on a Snapdragon device using 4 virtual sessions on a single NPU:

```bash
./scripts/snapdragon/run.py --target adb \
    --devices HTP0:0,HTP0:1,HTP0:2,HTP0:3 -- \
    llama-cli --load-mode none -m /data/local/tmp/gguf/gpt-oss-20b-Q4_0.gguf -t 4 \
    --ctx-size 8192 --batch-size 128 -ctk q8_0 -ctv q8_0 -fa on -ngl 99 -no-cnv -f surfing.txt
```

Log output snippet:

```
...
llama_model_loader: - type  f32:  289 tensors
llama_model_loader: - type q4_0:   96 tensors
llama_model_loader: - type q8_0:    2 tensors
llama_model_loader: - type mxfp4:  72 tensors
...
load_tensors: offloaded 25/25 layers to GPU
load_tensors:          CPU model buffer size =  1182.09 MiB
load_tensors:       HTP0:1 model buffer size =  2512.58 MiB
load_tensors:       HTP0:3 model buffer size =  2093.83 MiB
load_tensors:       HTP0:0 model buffer size =  2931.34 MiB
load_tensors:       HTP0:2 model buffer size =  2512.58 MiB
...
llama_perf_context_print: prompt eval time =    3843.67 ms /   197 tokens ( 19.51 ms per token, 51.25 tokens per second)
llama_perf_context_print:        eval time =    1686.13 ms /    31 runs   ( 54.39 ms per token, 18.39 tokens per second)
llama_perf_context_print:       total time =    6266.30 ms /   228 tokens
llama_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - HTP0:0 (Hexagon)   |  2048 = 2048 + (   0 =     0 +       0 +       0) +           0 |
llama_memory_breakdown_print: |   - HTP0:1 (Hexagon)   |  2048 = 2048 + (   0 =     0 +       0 +       0) +           0 |
llama_memory_breakdown_print: |   - HTP0:2 (Hexagon)   |  2048 = 2048 + (   0 =     0 +       0 +       0) +           0 |
llama_memory_breakdown_print: |   - HTP0:3 (Hexagon)   |  2048 = 2048 + (   0 =     0 +       0 +       0) +           0 |
llama_memory_breakdown_print: |   - Host               |                 1476 =  1208 +     105 +     162                |
```

### 3. Tensor-Split Mode across Physical Devices (`HTP0:0,HTP1:0,...`)

Distributes model tensors across distinct physical NPU hardware cores using llama.cpp's tensor parallelism
(`--split-mode tensor`).

Tensors are partitioned across physical NPUs for parallel execution (proportions are distributed equally by default without
needing an explicit `--tensor-split` option):

```bash
./scripts/snapdragon/run.py --target adb \
    --devices HTP0:0,HTP1:0 -- \
    llama-cli -m models/Llama-3.2-3B-Instruct-Q4_0.gguf --split-mode tensor -ngl 99 -p "Hello"
```

### 4. Row-Split Multi-Device Mode via Device Grouping (`HTP0[0-1]`)

Groups multiple physical NPU cores into a single logical device using bracket notation (`HTP0[0-1]` or `HTP0[0,1]`).

Unlike host-level tensor-splitting, row-splitting is executed entirely inside the Hexagon backend:

```bash
./scripts/snapdragon/run.py --target adb \
    --devices 'HTP0[0-1]' -- \
    llama-cli -m models/Llama-3.2-3B-Instruct-Q4_0.gguf -ngl 99 -p "Hello"
```

You can also combine row-splitting with layer-splitting across multiple grouped devices (e.g. `--devices 'HTP0[0-1],HTP1[2-3]'`
on 4 physical NPUs, or `--devices 'HTP0[0-1:0],HTP1[0-1:1]'` on 2 physical NPUs using virtual sessions 0 and 1).

## Environment variables

- `GGML_HEXAGON_DEVICES` (default: not set, defaults to HTP0 session)
  Controls which NPU devices and sessions to allocate. Configurable via `--devices` in `run.py`:
  - `N` (single integer): Allocates `N` virtual sessions named `HTP0`, `HTP1`, ..., `HTP<N-1>` on physical NPU 0.
  - `HTP<phys>:<virt>,...`: Comma-separated list of individual devices specifying physical and virtual index:
    - `HTP0:0,HTP0:1`: Two virtual sessions on physical NPU 0 (layer-split on single NPU).
    - `HTP0:0,HTP1:0`: One session on physical NPU 0 and one on physical NPU 1 (tensor-split across physical cores).
  - `HTP<name>[<phys_spec>]`: Device grouping syntax for row-split multi-device execution:
    - `HTP0[0-1]`: A single logical device `HTP0` that groups physical cores 0 and 1.
    - `HTP0[0-1],HTP1[2-3]`: Two layer-split devices across 4 physical NPUs (cores 0-1 and 2-3).
    - `HTP0[0-1:0],HTP1[0-1:1]`: Two layer-split devices across 2 physical NPUs using virtual sessions 0 and 1.

- `GGML_HEXAGON_NDEV` (deprecated)
  Replaced by `GGML_HEXAGON_DEVICES`. Controls the number of virtual sessions to allocate on physical NPU `0`.
  Allocates sessions named `HTP0`, `HTP1`, etc.

- `GGML_HEXAGON_NHVX=0`
  Controls the number of HVX hardware threads to use. The default is all (actual number varies depending on the hardware version).

- `GGML_HEXAGON_HOSTBUF=1` (default: 0, disabled)
  Enables allocating host buffers for debugging. By default, host buffers are disabled.

- `GGML_HEXAGON_VERBOSE=1`
  Enables verbose logging of Ops from the backend. Example output:

  ```
  ggml-hex: HTP0 graph-compute n_nodes 2
  ggml-hex: HTP0 matmul : blk.27.ffn_up.weight x ffn_norm-27 -> ffn_up-27 : 3072:8192 x 3072:1 -> 8192:1 : q4_0 x f32 -> f32 : HTP0 x HTP0 -> HTP0 : flags 0x1
  ggml-hex: HTP0 matmul : blk.27.ffn_gate.weight x ffn_norm-27 -> ffn_gate-27 : 3072:8192 x 3072:1 -> 8192:1 : q4_0 x f32 -> f32 : HTP0 x HTP0 -> HTP0 : flags 0x3
  ggml-hex: HTP0 graph-compute n_nodes 1
  ggml-hex: HTP0 matmul : blk.27.ffn_down.weight x ffn_gate_par-27 -> ffn_out-27 : 8192:3072 x 8192:1 -> 3072:1 : q4_0 x f32 -> f32 : HTP0 x HTP0 -> HTP0 : flags 0x0
  ggml-hex: HTP0 get-tensor result_output : data 0x7592487000 offset 0 size 513024
  ```

- `GGML_HEXAGON_PROFILE=1`
  Enables Op profiling (configurable via `--hex-profile` in `run.py`):

  - `1`: Basic profile with per-op `usecs` and `cycles` counters
  - `2`: Extended profile with per-op `usecs`, `cycles` and default PMU counter data
  - `0x1,...,0x8`: Extended profile with per-op `usecs`, `cycles` and custom PMU counter data

  The logging output can be saved to a file or piped directly into the post-processing script:

  ```bash
  ./scripts/snapdragon/run.py --target adb --hex-profile 1 -- llama-cli ... |& \
      ./scripts/snapdragon/ggml-hexagon-profile.py -
  ```

- `GGML_HEXAGON_OPFILTER=regex`
  Filters (disables) Ops matching the regex pattern (configurable via `--hex-opfilter` in `run.py`):

  ```bash
  # Disable Flash Attention on Hexagon (falls back to CPU or GPU)
  ./scripts/snapdragon/run.py --target adb --hex-opfilter "FLASH_ATTN_EXT" -- llama-cli ...

  # Disable ADD and SUB on Hexagon (fall back to CPU or GPU)
  ./scripts/snapdragon/run.py --target adb --hex-opfilter "ADD|SUB" -- llama-cli ...
  ```
