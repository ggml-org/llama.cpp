# Snapdragon-based devices

## Setup

### Android

The easiest way to build llama.cpp for a Snapdragon-based Android device is using the toolchain Docker image (see github.com/snapdragon-toolchain).
This image includes Android NDK, OpenCL SDK, Hexagon SDK, CMake, etc.

This method works on Linux, macOS, and Windows. macOS and Windows users should install Docker Desktop.

```
~/src/llama.cpp$ docker run -it -u $(id -u):$(id -g) --volume $(pwd):/workspace --platform linux/amd64 ghcr.io/snapdragon-toolchain/arm64-android:v0.3
[d]/> cd /workspace
```

Note: The rest of the **Android** build process assumes that you're running inside the toolchain container.

### Windows On Snapdragon

Native Windows 11 arm64 builds has the following tools dependencies:
- MS Visual Studio 2026 (Community Edition or Pro)
  - MSVC arm64 standard and runtime libraries
  - UCRT and Driver Kit
- LLVM core libraries and Clang compiler (winget)
- CMake, Git, Python (winget)
- Hexagon SDK Community Edition 6.4 or later (see windows.md)
- OpenCL SDK 2.3 or later (see windows.md)

Note: The rest of the **Windows** build process assumes that you're running natively in Powershell.
Adapt below build commands accordingly.

## How to Build

Let's build llama.cpp with CPU, OpenCL, and Hexagon backends via CMake presets:

```
[d]/workspace> cp docs/backend/snapdragon/CMakeUserPresets.json .

[d]/workspace> cmake --preset arm64-android-snapdragon-release -B build-snapdragon
Preset CMake variables:
  ANDROID_ABI="arm64-v8a"
  ...
  CMAKE_TOOLCHAIN_FILE="/opt/android-ndk-r28b/build/cmake/android.toolchain.cmake"
  GGML_HEXAGON="ON"
  GGML_OPENCL="ON"
  GGML_OPENMP="OFF"
  HEXAGON_SDK_ROOT="/opt/hexagon/6.4.0.2"
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
[d]/workspace> cmake --install build-snapdragon --prefix pkg-snapdragon/llama.cpp
-- Install configuration: "Release"
-- Installing: /workspace/pkg-snapdragon/llama.cpp/lib/libggml-cpu.so
-- Installing: /workspace/pkg-snapdragon/llama.cpp/lib/libggml-opencl.so
-- Installing: /workspace/pkg-snapdragon/llama.cpp/lib/libggml-hexagon.so
-- Installing: /workspace/pkg-snapdragon/llama.cpp/lib/libggml-htp-v73.so
-- Installing: /workspace/pkg-snapdragon/llama.cpp/lib/libggml-htp-v75.so
-- Installing: /workspace/pkg-snapdragon/llama.cpp/lib/libggml-htp-v79.so
-- Installing: /workspace/pkg-snapdragon/llama.cpp/lib/libggml-htp-v81.so
-- Installing: /workspace/pkg-snapdragon/llama.cpp/lib/libggml.so
...
-- Installing: /workspace/pkg-snapdragon/llama.cpp/bin/llama-bench
-- Installing: /workspace/pkg-snapdragon/llama.cpp/bin/llama-cli
...
```

## How to Install

### Android

For this step, your device needs to be configured for on-device development.
Please see https://developer.android.com/studio/debug/dev-options for details.

Once ADB is enabled, use `adb push` to install `pkg-snapdragon` on the device.
**Note that the toolchain Docker image doesn't have ADB and doesn't set up the ADB bridge. Please use native ADB on the host.**

```
~/src/llama.cpp$ adb push pkg-snapdragon/llama.cpp /data/local/tmp/
pkg-snapdragon/llama.cpp/bin/: 67 files pushed, 0 skipped. 190.2 MB/s (919095042 bytes in 4.607s)
pkg-snapdragon/llama.cpp/include/: 19 files pushed, 0 skipped. 20.5 MB/s (255173 bytes in 0.012s)
pkg-snapdragon/llama.cpp/lib/: 16 files pushed, 0 skipped. 144.4 MB/s (43801382 bytes in 0.289s)
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

All artifacts are already installed in the `pkg-snapdragon` folder.
To run, adapt below instructions to use Powershell scrits in `scripts/snapdragon/windows`.

## How to Run

The easiest way to run llama.cpp cli tools is using provided wrapper scripts that properly set up all required environment variables.

llama.cpp supports three backends on Snapdragon-based devices: CPU, Adreno GPU (GPUOpenCL), and Hexagon NPU (HTP0-4).
You can select which backend to run the model on using the `D=` variable, which maps to the `--device` option.

Hexagon NPU behaves as a "GPU" device when it comes to `-ngl` and other offload-related options.

Here are some examples of running various llama.cpp tools via ADB.

Simple question for Llama-3.2-1B

```
~/src/llama.cpp$ M=Llama-3.2-1B-Instruct-Q4_0.gguf D=HTP0 ./scripts/snapdragon/adb/run-completion.sh -p "what is the most popular cookie in the world?"
...
ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev 1
ggml-hex: Hexagon Arch version v79
ggml-hex: allocating new session: HTP0
ggml-hex: new session: HTP0 : session-id 0 domain-id 3 uri file:///libggml-htp-v79.so?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0 handle 0xb4000072c7955e50
...
load_tensors: offloading output layer to GPU
load_tensors: offloaded 17/17 layers to GPU
load_tensors:          CPU model buffer size =   225.49 MiB
load_tensors:         HTP0 model buffer size =     0.26 MiB
load_tensors:  HTP0-REPACK model buffer size =   504.00 MiB
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
llama_memory_breakdown_print: |   - HTP0-REPACK        |                  504 =   504 +       0 +       0                |
```

Summary request for OLMoE-1B-7B. This is a large model that requires two HTP sessions/devices

```
~/src/llama.cpp$ M=OLMoE-1B-7B-0125-Instruct-Q4_0.gguf NDEV=2 D=HTP0,HTP1 ./scripts/snapdragon/adb/run-completion.sh -f surfing.txt
...
ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev 1
ggml-hex: Hexagon Arch version v81
ggml-hex: allocating new session: HTP0
ggml-hex: allocating new session: HTP1
...
load_tensors: offloading output layer to GPU
load_tensors: offloaded 17/17 layers to GPU
load_tensors:          CPU model buffer size =   143.86 MiB
load_tensors:         HTP1 model buffer size =     0.23 MiB
load_tensors:  HTP1-REPACK model buffer size =  1575.00 MiB
load_tensors:         HTP0 model buffer size =     0.28 MiB
load_tensors:  HTP0-REPACK model buffer size =  2025.00 MiB
...
llama_context:        CPU  output buffer size =     0.19 MiB
llama_kv_cache:       HTP1 KV buffer size =   238.00 MiB
llama_kv_cache:       HTP0 KV buffer size =   306.00 MiB
llama_kv_cache: size =  544.00 MiB (  8192 cells,  16 layers,  1/1 seqs), K (q8_0):  272.00 MiB, V (q8_0):  272.00 MiB
llama_context:       HTP0 compute buffer size =    15.00 MiB
llama_context:       HTP1 compute buffer size =    15.00 MiB
llama_context:        CPU compute buffer size =    24.56 MiB
...
llama_perf_context_print: prompt eval time =    1730.57 ms /   212 tokens (    8.16 ms per token,   122.50 tokens per second)
llama_perf_context_print:        eval time =    5624.75 ms /   257 runs   (   21.89 ms per token,    45.69 tokens per second)
llama_perf_context_print:       total time =    7377.33 ms /   469 tokens
llama_perf_context_print:    graphs reused =        255
llama_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - HTP0 (Hexagon)     |  2048 = 2048 + (   0 =     0 +       0 +       0) +           0 |
llama_memory_breakdown_print: |   - HTP1 (Hexagon)     |  2048 = 2048 + (   0 =     0 +       0 +       0) +           0 |
llama_memory_breakdown_print: |   - Host               |                  742 =   144 +     544 +      54                |
llama_memory_breakdown_print: |   - HTP1-REPACK        |                 1575 =  1575 +       0 +       0                |
llama_memory_breakdown_print: |   - HTP0-REPACK        |                 2025 =  2025 +       0 +       0                |
```

Op test for MUL_MAT

```
~/src/llama.cpp$ HB=0 ./scripts/snapdragon/adb/run-tool.sh test-backend-ops -b HTP0 -o MUL_MAT
...
Backend 2/3: HTP0
Device description: Hexagon
Device memory: 2048 MB (2048 MB free)
MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],v=0,o=1): OK
MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=2,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],v=0,o=1): OK
MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=3,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],v=0,o=1): OK

~/src/llama.cpp-hexagon$ M=Llama-3.2-1B-Instruct-Q4_0.gguf ./scripts/snapdragon/adb/run-bench.sh -p 128 -n 64
...
ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev 1
ggml-hex: Hexagon Arch version v79
ggml-hex: allocating new session: HTP0
ggml-hex: new session: HTP0 : session-id 0 domain-id 3 uri file:///libggml-htp-v79.so?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0 handle 0xb400007d4b231090
| model          |       size | params | backend    | ngl | threads | n_batch | mmap |  test |           t/s |
| ---------------| ---------: | -----: | ---------- | --: | ------: | ------: | ---: | ----: | ------------: |
| llama 1B Q4_0  | 729.75 MiB | 1.24 B | HTP        |  99 |       4 |     128 |    0 | pp128 | 169.42 ± 1.75 |
| llama 1B Q4_0  | 729.75 MiB | 1.24 B | HTP        |  99 |       4 |     128 |    0 |  tg64 |  51.54 ± 1.13 |

build: 6a8cf8914 (6733)
```

## Environment variables

- `GGML_HEXAGON_NDEV=1`
  Controls the number of devices/sessions to allocate. The default is 1.
  Most quantized models under 4B fit into a single session; an 8B model needs two, and a 20B model needs four.

- `GGML_HEXAGON_NHVX=0`
  Controls the number of HVX hardware threads to use. The default is all (actual number varies depending on the hardware version).

- `GGML_HEXAGON_HOSTBUF=1`
  Controls whether the Hexagon backend allocates host buffers. By default, all buffers except for REPACK are host buffers.
  This option is required for testing Ops that require REPACK buffers (MUL_MAT and MUL_MAT_ID).

- `GGML_HEXAGON_EXPERIMENTAL=1`
  Controls whether the Hexagon backend enables experimental features.
  This option is required for enabling/testing experimental Ops (FLASH_ATTN_EXT).

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
  Generates a host-side profile for the ggml-hexagon Ops.

- `GGML_HEXAGON_OPMASK=0x0`
  Allows enabling specific stages of the processing pipeline:

  - `0x1` Enable Op Queue (i.e., queuing Ops into NPU)
  - `0x2` Enable Dynamic Quantizer (if needed for the Op)
  - `0x4` Enable Op Compute (MUL_MAT, etc.)

  Examples:

      `GGML_HEXAGON_OPMASK=0x1 llama-completion ...` - Ops are enqueued but NPU-side processing is stubbed out
      `GGML_HEXAGON_OPMASK=0x3 llama-completion ...` - NPU performs dynamic quantization and skips the rest
      `GGML_HEXAGON_OPMASK=0x7 llama-completion ...` - Full queuing and processing of Ops (default)
