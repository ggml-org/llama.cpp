# fit-params

llama.cpp binaries can automatically fit the projected memory use of a model to the free device memory available at runtime,
this is controlled using the CLI arguments starting with `-fit`/`--fit`.
Internally the code is calling `llama_params_fit` to adjust the `llama_model_params` and `llama_context_params` structs.
`llama-fit-params` is a simple utility that prints the CLI arguments corresponding to these adjustments to stdout.
Example usage:

``` bash
# First, run llama-fit-params and store the results in a file:
> ./build/bin/llama-fit-params --model /opt/models/qwen_3-30b3a-f16.gguf | tee args.txt
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
build: 6895 (4341dc8bc) with cc (GCC) 15.2.1 20250813 for x86_64-pc-linux-gnu
llama_params_fit_impl: projected to use 61807 MiB of device memory vs. 24077 MiB of free device memory
llama_params_fit_impl: cannot fulfill margin of 1024 MiB, need to reduce device memory by 42444 MiB
llama_params_fit_impl: context size reduced from 40960 to 4096 -> need 3456 MiB less memory in total
llama_params_fit_impl: with only dense weights in device memory there is a total surplus of 16164 MiB
llama_params_fit_impl: distributing layers across devices with overflow to next device/system memory:
llama_params_fit_impl:   - CUDA0 (NVIDIA GeForce RTX 4090): 48 layers (34 overflowing),  19187 MiB used,   1199 MiB free
llama_params_fit: successfully fit params to free device memory
llama_params_fit: fitting params to free memory took 1.15 seconds
Printing fitted CLI arguments to stdout...
-c 4096 -ngl 48 -ot blk\.14\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.15\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.16\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.17\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.18\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.19\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.20\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.21\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.22\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.23\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.24\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.25\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.26\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.27\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.28\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.29\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.30\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.31\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.32\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.33\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.34\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.35\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.36\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.37\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.38\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.39\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.40\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.41\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.42\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.43\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.44\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.45\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.46\.ffn_(up|down|gate)_(ch|)exps=CPU,blk\.47\.ffn_(up|down|gate)_(ch|)exps=CPU

# Next, use those results for a llama.cpp binary:
> cat args.txt | xargs ./build/bin/llama-server --model /opt/models/qwen_3-30b3a-f16.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
build: 6895 (4341dc8bc) with cc (GCC) 15.2.1 20250813 for x86_64-pc-linux-gnu
system info: n_threads = 16, n_threads_batch = 16, total_threads = 32

system_info: n_threads = 16 (n_threads_batch = 16) / 32 | CUDA : ARCHS = 890 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |

main: binding port with default address family
main: HTTP server is listening, hostname: 127.0.0.1, port: 8080, http threads: 31
main: loading model
srv    load_model: loading model '/opt/models/qwen_3-30b3a-f16.gguf'
llama_params_fit_impl: projected to use 19187 MiB of device memory vs. 24077 MiB of free device memory
llama_params_fit_impl: will leave 1199 >= 1024 MiB of free device memory, no changes needed
llama_params_fit: successfully fit params to free device memory
llama_params_fit: fitting params to free memory took 0.28 seconds
[...]
main: server is listening on http://127.0.0.1:8080 - starting the main loop
srv  update_slots: all slots are idle
^Csrv    operator(): operator(): cleaning up before exit...

llama_memory_breakdown_print: | memory breakdown [MiB] | total   free     self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - CUDA0 (RTX 4090)   | 24077 =  945 + (19187 = 17904 +     384 +     898) +        3945 |
llama_memory_breakdown_print: |   - Host               |                 58271 = 58259 +       0 +      12                |
```
