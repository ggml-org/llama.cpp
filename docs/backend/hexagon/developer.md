# Hexagon backend developer details

## Backend libraries

The Hexagon backend consist of two parts:

  - `libggml-hexagon`
    This is the regular CPU-side GGML backend library, either shared or statically linked

  - `libggml-htp-vNN`
    This is the NPU-side (HTP stands for Hexagon Tensor Processor) shared library that contains the Op dispatcher and kernels.
    The correct library is selected automatically at runtime based on the HW version.

Here is an example of the build artifacts

```
~/src/llama.cpp$ ls -l pkg-adb/llama.cpp/lib/libggml*
pkg-adb/llama.cpp/lib/libggml-base.so
pkg-adb/llama.cpp/lib/libggml-cpu.so
pkg-adb/llama.cpp/lib/libggml-hexagon.so      <<< CPU library
pkg-adb/llama.cpp/lib/libggml-htp-v73.so      <<< HTP op/kernels for Hexagon v73
pkg-adb/llama.cpp/lib/libggml-htp-v75.so
pkg-adb/llama.cpp/lib/libggml-htp-v79.so
pkg-adb/llama.cpp/lib/libggml-htp-v81.so
```

## Memory buffers

Hexagon NPU backend takes advantage of the Snapdragon's unified memory model where all buffers are fully accessible by the CPU and GPU.
The NPU does have a dedicated tightly-coupled memory called VTCM but that memory is used only for intermediate data (e.g. dynamically
quantized tensors) or temporary data (chunks of the weight tensors fetched via DMA).

Please note that currently the Hexagon backend does not implement SET/GET_ROWS Ops because there is no advantage in offloading those
to the NPU at this point.

The backend does allocates non-host buffers for the tensors with datatypes that require repacking: Q4_0, Q8_0, MXFP4.
From the MMU perspective these buffers are still regular buffers (normal access by the CPU) they are marked as non-host simply to force
the repacking.

## Large model handling

Hexagon NPU session (aka Process Domain (PD) in the Hexagon docs) is limited to a memory mapping of around 3.5GB.
In llama.cpp/GGML the Hexagon session is mapped to a single GGML backend device (HTP0, HTP1, etc).

In order to map models larger than 3.5GB we need to allocate multiple devices and split the model.
For this we're taking advantage of the llama.cpp/GGML multi-GPU layer-splitting support.
Each Hexagon device behaves like a GPU from the offload and model splitting perspective.

Here is an example of running GPT-OSS-20B model on a newer Snapdragon device with 16GB of DDR.

```
M=gpt-oss-20b-Q4_0.gguf NDEV=4 D=HTP0,HTP1,HTP2,HTP3 P=surfing.txt scripts/snapdragon/adb/run-cli.sh -no-cnv -f surfing.txt -n 32
...
LD_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib
ADSP_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib
GGML_HEXAGON_NDEV=4 ./bin/llama-cli --no-mmap -m /data/local/tmp/llama.cpp/../gguf/gpt-oss-20b-Q4_0.gguf
      -t 4 --ctx-size 8192 --batch-size 128 -ctk q8_0 -ctv q8_0 -fa on -ngl 99 --device HTP0,HTP1,HTP2,HTP3 -no-cnv -f surfing.txt
...
llama_model_loader: - type  f32:  289 tensors
llama_model_loader: - type q4_0:   96 tensors
llama_model_loader: - type q8_0:    2 tensors
llama_model_loader: - type mxfp4:  72 tensors
...
load_tensors: offloaded 25/25 layers to GPU
load_tensors:          CPU model buffer size =  1182.09 MiB
load_tensors:         HTP1 model buffer size =     6.64 MiB
load_tensors:  HTP1-REPACK model buffer size =  2505.94 MiB
load_tensors:         HTP3 model buffer size =     5.55 MiB
load_tensors:  HTP3-REPACK model buffer size =  2088.28 MiB
load_tensors:         HTP0 model buffer size =     7.75 MiB
load_tensors:  HTP0-REPACK model buffer size =  2923.59 MiB
load_tensors:         HTP2 model buffer size =     6.64 MiB
load_tensors:  HTP2-REPACK model buffer size =  2505.94 MiB
...
llama_context: n_ctx_per_seq (8192) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
llama_context:        CPU  output buffer size =     0.77 MiB
llama_kv_cache_iswa: creating non-SWA KV cache, size = 8192 cells
llama_kv_cache:       HTP1 KV buffer size =    25.50 MiB
llama_kv_cache:       HTP3 KV buffer size =    25.50 MiB
llama_kv_cache:       HTP0 KV buffer size =    25.50 MiB
llama_kv_cache:       HTP2 KV buffer size =    25.50 MiB
llama_kv_cache: size =  102.00 MiB (  8192 cells,  12 layers,  1/1 seqs), K (q8_0):   51.00 MiB, V (q8_0):   51.00 MiB
llama_kv_cache_iswa: creating     SWA KV cache, size = 256 cells
llama_kv_cache:       HTP1 KV buffer size =     0.80 MiB
llama_kv_cache:       HTP3 KV buffer size =     0.53 MiB
llama_kv_cache:       HTP0 KV buffer size =     1.06 MiB
llama_kv_cache:       HTP2 KV buffer size =     0.80 MiB
llama_kv_cache: size =    3.19 MiB (   256 cells,  12 layers,  1/1 seqs), K (q8_0):    1.59 MiB, V (q8_0):    1.59 MiB
llama_context:       HTP0 compute buffer size =    16.06 MiB
llama_context:       HTP1 compute buffer size =    16.06 MiB
llama_context:       HTP2 compute buffer size =    16.06 MiB
llama_context:       HTP3 compute buffer size =    16.06 MiB
llama_context:        CPU compute buffer size =    98.19 MiB
...
llama_perf_context_print: prompt eval time =    3843.67 ms /   197 tokens ( 19.51 ms per token, 51.25 tokens per second)
llama_perf_context_print:        eval time =    1686.13 ms /    31 runs   ( 54.39 ms per token, 18.39 tokens per second)
llama_perf_context_print:       total time =    6266.30 ms /   228 tokens
llama_perf_context_print:    graphs reused =         30
llama_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - HTP0 (Hexagon)     |  2048 = 2048 + (   0 =     0 +       0 +       0) +           0 |
llama_memory_breakdown_print: |   - HTP1 (Hexagon)     |  2048 = 2048 + (   0 =     0 +       0 +       0) +           0 |
llama_memory_breakdown_print: |   - HTP2 (Hexagon)     |  2048 = 2048 + (   0 =     0 +       0 +       0) +           0 |
llama_memory_breakdown_print: |   - HTP3 (Hexagon)     |  2048 = 2048 + (   0 =     0 +       0 +       0) +           0 |
llama_memory_breakdown_print: |   - Host               |                 1476 =  1208 +     105 +     162                |
llama_memory_breakdown_print: |   - HTP1-REPACK        |                 2505 =  2505 +       0 +       0                |
llama_memory_breakdown_print: |   - HTP3-REPACK        |                 2088 =  2088 +       0 +       0                |
llama_memory_breakdown_print: |   - HTP0-REPACK        |                 2923 =  2923 +       0 +       0                |
llama_memory_breakdown_print: |   - HTP2-REPACK        |                 2505 =  2505 +       0 +       0                |
```
