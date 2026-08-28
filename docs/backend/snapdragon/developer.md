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

Hexagon NPU sessions (aka Process Domains (PD) in the Hexagon SDK) are limited to a maximum memory mapping window of around 3.5GB.
In llama.cpp/GGML, each Hexagon session is mapped to a single GGML backend device (e.g., `HTP0:0`, `HTP0:1`, etc. when using
`GGML_HEXAGON_DEVICES`, or `HTP0`, `HTP1` in legacy mode).

To support running models larger than 3.5GB on a single device, the Hexagon backend dynamically maps and unmaps execution buffers
during the graph execution cycle to stay within the Process Domain window. This enables large models to run successfully on a single
NPU device.

Alternatively, users can choose to use standard llama.cpp/GGML layer-splitting mode to partition and split the model across
multiple Hexagon devices or virtual sessions (which behave like multiple GPUs from the offload and splitting perspective).

Here is an example of running GPT-OSS-20B model on a Snapdragon device using 4 virtual sessions on a single NPU (physical index 0).

```
~/src/llama.cpp$ ./scripts/snapdragon/run.py --target adb --devices HTP0:0,HTP0:1,HTP0:2,HTP0:3 -- llama-cli --load-mode none -m /data/local/tmp/gguf/gpt-oss-20b-Q4_0.gguf -t 4 --ctx-size 8192 --batch-size 128 -ctk q8_0 -ctv q8_0 -fa on -ngl 99 -no-cnv -f surfing.txt
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
llama_context: n_ctx_per_seq (8192) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
llama_context:        CPU  output buffer size =     0.77 MiB
llama_kv_cache_iswa: creating non-SWA KV cache, size = 8192 cells
llama_kv_cache:     HTP0:1 KV buffer size =    25.50 MiB
llama_kv_cache:     HTP0:3 KV buffer size =    25.50 MiB
llama_kv_cache:     HTP0:0 KV buffer size =    25.50 MiB
llama_kv_cache:     HTP0:2 KV buffer size =    25.50 MiB
llama_kv_cache: size =  102.00 MiB (  8192 cells,  12 layers,  1/1 seqs), K (q8_0):   51.00 MiB, V (q8_0):   51.00 MiB
llama_kv_cache_iswa: creating     SWA KV cache, size = 256 cells
llama_kv_cache:     HTP0:1 KV buffer size =     0.80 MiB
llama_kv_cache:     HTP0:3 KV buffer size =     0.53 MiB
llama_kv_cache:     HTP0:0 KV buffer size =     1.06 MiB
llama_kv_cache:     HTP0:2 KV buffer size =     0.80 MiB
llama_kv_cache: size =    3.19 MiB (   256 cells,  12 layers,  1/1 seqs), K (q8_0):    1.59 MiB, V (q8_0):    1.59 MiB
llama_context:     HTP0:0 compute buffer size =    16.06 MiB
llama_context:     HTP0:1 compute buffer size =    16.06 MiB
llama_context:     HTP0:2 compute buffer size =    16.06 MiB
llama_context:     HTP0:3 compute buffer size =    16.06 MiB
llama_context:        CPU compute buffer size =    98.19 MiB
...
llama_perf_context_print: prompt eval time =    3843.67 ms /   197 tokens ( 19.51 ms per token, 51.25 tokens per second)
llama_perf_context_print:        eval time =    1686.13 ms /    31 runs   ( 54.39 ms per token, 18.39 tokens per second)
llama_perf_context_print:       total time =    6266.30 ms /   228 tokens
llama_perf_context_print:    graphs reused =         30
llama_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - HTP0:0 (Hexagon)   |  2048 = 2048 + (   0 =     0 +       0 +       0) +           0 |
llama_memory_breakdown_print: |   - HTP0:1 (Hexagon)   |  2048 = 2048 + (   0 =     0 +       0 +       0) +           0 |
llama_memory_breakdown_print: |   - HTP0:2 (Hexagon)   |  2048 = 2048 + (   0 =     0 +       0 +       0) +           0 |
llama_memory_breakdown_print: |   - HTP0:3 (Hexagon)   |  2048 = 2048 + (   0 =     0 +       0 +       0) +           0 |
llama_memory_breakdown_print: |   - Host               |                 1476 =  1208 +     105 +     162                |
```
