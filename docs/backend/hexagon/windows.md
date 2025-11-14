# Windows on Snapdragon

## Requirements

Native Windows 11 arm64 builds has the following tools dependencies:
- MS Visual Studio 2022 or later (Community Edition or Pro)
  - MSVC arm64 standard and runtime libraries
- LLVM core libraries and Clang compiler (winget)
- CMake, Git (winget)

## Install Hexagon SDK & AddOns

Download Qualcomm's Software Center and install the SDK in default location i.e. C:\Qualcomm\ and addons inside the SDK:
  - Hexagon SDK 6.x (latest)
  - Compute_AddOn
  - Windows on Snapdragon Addon

## Supported Devices
- Snapdragon X Elite
- Snapdragon X2 Elite

## How to Build

The rest of the Windows build process assumes that you're running inside PowerShell.

First let's setup the Hexagon SDK:

```
PS ...> cd C:\Qualcomm\Hexagon_SDK\6.4.0.0\
PS C:\Qualcomm\Hexagon_SDK\6.4.0.0> .\setup_sdk_env.ps1
```

Now build llama.cpp with CPU and Hexagon backends via CMake presets:

```
PS ...> cd c:\...\workspace\llama.cpp
PS ...\workspace\llama.cpp> cp docs/backend/hexagon/CMakeUserPresets.json .

PS ...\workspace\llama.cpp> cmake --preset arm64-windows-snapdragon-release -B build-snapdragon
-- The C compiler identification is Clang 21.1.1 with GNU-like command-line
-- The CXX compiler identification is Clang 21.1.1 with GNU-like command-line
...
-- Including Hexagon backend
...
-- Build files have been written to: /workspace/llama.cpp/build-snapdragon

PS ...\workspace\llama.cpp> cmake --build build-snapdragon
...
[61/333] Performing configure step for 'htp-v81'
-- The C compiler identification is Clang 19.0.0
-- The CXX compiler identification is Clang 19.0.0
-- The ASM compiler identification is unknown
-- Found assembler: C:/Qualcomm/Hexagon_SDK/6.4.0.0/tools/HEXAGON_Tools/19.0.04/Tools/bin/hexagon-clang++.exe
...
-- Installing: C:/.../workspace/llama.cpp/build-snapdragon/ggml/src/ggml-hexagon/libggml-htp-v73.so
-- Installing: C:/.../workspace/llama.cpp/build-snapdragon/ggml/src/ggml-hexagon/libggml-htp-v81.so
...
```

To generate an installable "package" simply run cmake --install:

```
PS ...\workspace\llama.cpp> cmake --install .\build-snapdragon --prefix pkg-snapdragon
-- Install configuration: "Release"
-- Installing: C:/.../workspace/llama.cpp/pkg-snapdragon/lib/ggml-cpu.lib
-- Installing: C:/.../workspace/llama.cpp/pkg-snapdragon/bin/ggml-cpu.dll
-- Installing: C:/.../workspace/llama.cpp/pkg-snapdragon/lib/ggml-opencl.lib
-- Installing: C:/.../workspace/llama.cpp/pkg-snapdragon/bin/ggml-opencl.dll
-- Installing: C:/.../workspace/llama.cpp/pkg-snapdragon/lib/ggml-hexagon.lib
-- Installing: C:/.../workspace/llama.cpp/pkg-snapdragon/bin/ggml-hexagon.dll
-- Installing: C:/.../workspace/llama.cpp/pkg-snapdragon/lib/libggml-htp-v73.so
-- Installing: C:/.../workspace/llama.cpp/pkg-snapdragon/lib/libggml-htp-v75.so
-- Installing: C:/.../workspace/llama.cpp/pkg-snapdragon/lib/libggml-htp-v79.so
-- Installing: C:/.../workspace/llama.cpp/pkg-snapdragon/lib/libggml-htp-v81.so
-- Installing: C:/.../workspace/llama.cpp/pkg-snapdragon/lib/ggml.lib
-- Installing: C:/.../workspace/llama.cpp/pkg-snapdragon/bin/ggml.dll
...
-- Up-to-date: C:/.../workspace/llama.cpp/pkg-snapdragon/bin/convert_hf_to_gguf.py
-- Installing: C:/.../workspace/llama.cpp/pkg-snapdragon/lib/pkgconfig/llama.pc
...
```

## How to Sign Generated Artifacts

Prior to running any of the llama.cpp tools, the generated artifacts have to be signed.
Copy the assets from \\fringe\morpheus_sandiego_hetai\users\todorb\wos\* to the pkg install i.e. ...\workspace\llama.cpp\pkg-snapdragon\lib\ and then sing the artifacts:

```
PS ...\workspace\llama.cpp> cd pkg-snapdragon\lib
PS ...\workspace\llama.cpp> .\CreateCatalogAndSign.ps1 -catalogName "Test" -srcPath ".\" -certPath ".\"
```

## Setup Environment

Prior to running any of the llama.cpp tools, add the location of the MCDM driver to your environment path:
```
$env:Path += 'C:\Windows\System32\DriverStore\FileRepository\qcnspmcdm8480.inf_arm64_4531c920e899ab0c\'
```
Also add the location of the generated artifacts to your environment:
```
PS ...\workspace\llama.cpp> $env:ADSP_LIBRARY_PATH='C:\Users\HCKTest\workspace\llama.cpp\pkg-snapdragon\lib\'
```

## How to Run

The easiest way to run llama.cpp cli tools is using provided wrapper scripts that properly set up all required environment variables.

llama.cpp supports three backends on Snapdragon-based devices: CPU, Adreno GPU (GPUOpenCL), and Hexagon NPU (HTP0-4).
You can select which backend to run the model on using the `D=` variable, which maps to the `--device` option.

Hexagon NPU behaves as a "GPU" device when it comes to `-ngl` and other offload-related options.

Here are some examples of running various llama.cpp tools on Windows.

Simple question for LFM2-1.2B

```
PS ...\workspace\llama.cpp> $env:M="LFM2-1.2B-Q4_0.gguf"
PS ...\workspace\llama.cpp> $env:D="HTP0"
PS ...\workspace\llama.cpp> .\scripts\snapdragon\windows\run-cli.ps1 -no-cnv -p "what is the most popular cookie in the world?"
ggml_opencl: selected platform: 'QUALCOMM Snapdragon(TM)'
...
ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev 1
ggml-hex: Hexagon Arch version v81
ggml-hex: allocating new session: HTP0
ggml-hex: new session: HTP0 : session-id 0 domain-id 3 uri file:///libggml-htp-v81.so?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0 handle 0x9f6fbd0
...
load_tensors: offloading output layer to GPU
load_tensors: offloaded 17/17 layers to GPU
load_tensors:          CPU model buffer size =   105.24 MiB
load_tensors:         HTP0 model buffer size =     0.25 MiB
load_tensors:  HTP0-REPACK model buffer size =   555.75 MiB
...
chocolate chip cookies are widely considered the global favorite, though exact figures are challenging to pinpoint. [end of text]
...
llama_perf_sampler_print:    sampling time =       2.40 ms /   100 runs   (    0.02 ms per token, 41753.65 tokens per second)
llama_perf_context_print:        load time =    1223.06 ms
llama_perf_context_print: prompt eval time =      76.41 ms /    14 tokens (    5.46 ms per token,   183.21 tokens per second)
llama_perf_context_print:        eval time =    1296.18 ms /    85 runs   (   15.25 ms per token,    65.58 tokens per second)
llama_perf_context_print:       total time =    1378.58 ms /    99 tokens
llama_perf_context_print:    graphs reused =          0
llama_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute       unaccounted |
llama_memory_breakdown_print: |   - HTP0 (Hexagon)     |  2048 = 2048 + ( 555 =   555 +       0 +       0) +              0 |
llama_memory_breakdown_print: |   - Host         
```

Summary request for OLMoE-1B-7B on GPU. This is a large model that requires two HTP sessions/devices

```
PS ...\workspace\llama.cpp> $env:M="OLMoE-1B-7B-0125-Instruct-Q4_0.gguf"
PS ...\workspace\llama.cpp> $env:D="HTP0,HTP1"
PS ...\workspace\llama.cpp> .\scripts\snapdragon\windows\run-cli.sps1 -f surfing.txt -no-cnv
ggml_opencl: selected platform: 'QUALCOMM Snapdragon(TM)'
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

## Environment variables

Refer to Android build documentation for full list of the supported environment variables.
