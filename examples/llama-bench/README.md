# llama.cpp/examples/llama-bench

Performance testing tool for llama.cpp.

## Table of contents

1. [Syntax](#syntax)
2. [Examples](#examples)
    1. [Text generation with different models](#text-generation-with-different-models)
    2. [Prompt processing with different batch sizes](#prompt-processing-with-different-batch-sizes)
    3. [Different numbers of threads](#different-numbers-of-threads)
    4. [Different numbers of layers offloaded to the GPU](#different-numbers-of-layers-offloaded-to-the-gpu)
3. [Output formats](#output-formats)
    1. [Markdown](#markdown)
    2. [CSV](#csv)
    3. [JSON](#json)
    4. [JSONL](#jsonl)
    5. [SQL](#sql)

## Syntax

```
usage: ./llama-bench [options]

options:
  -h, --help
  -m, --model <filename>                    (default: models/7B/ggml-model-q4_0.gguf)
  -p, --n-prompt <n>                        (default: 0)
  -n, --n-gen <n>                           (default: 32)
  -pg <pp,tg>                               (default: 4096,32)
  -b, --batch-size <n>                      (default: 2048)
  -ub, --ubatch-size <n>                    (default: 512)
  -ctk, --cache-type-k <t>                  (default: f16)
  -ctv, --cache-type-v <t>                  (default: f16)
  -t, --threads <n>                         (default: 8)
  -C, --cpu-mask <hex,hex>                  (default: 0x0)
  --cpu-strict <0|1>                        (default: 0)
  --poll <0...100>                          (default: 50)
  -ngl, --n-gpu-layers <n>                  (default: 99)
  -rpc, --rpc <rpc_servers>                 (default: )
  -sm, --split-mode <none|layer|row>        (default: layer)
  -mg, --main-gpu <i>                       (default: 0)
  -nkvo, --no-kv-offload <0|1>              (default: 0)
  -fa, --flash-attn <0|1>                   (default: 0)
  -mmp, --mmap <0|1>                        (default: 1)
  --numa <distribute|isolate|numactl>       (default: disabled)
  -embd, --embeddings <0|1>                 (default: 0)
  -ts, --tensor-split <ts0/ts1/..>          (default: 0)
  -r, --repetitions <n>                     (default: 5)
  --prio <0|1|2|3>                          (default: 0)
  --delay <0...N> (seconds)                 (default: 0)
  -o, --output <csv|json|jsonl|md|sql>      (default: md)
  -oe, --output-err <csv|json|jsonl|md|sql> (default: none)
  -v, --verbose                             (default: 0)

Multiple values can be given for each parameter by separating them with ',' or by specifying the parameter multiple times.
```

llama-bench can perform three types of tests:

- Prompt processing (pp): processing a prompt in batches (`-p`)
- Text generation (tg): generating a sequence of tokens (`-n`)
- Prompt processing + text generation (pg): processing a prompt followed by generating a sequence of tokens (`-pg`)

With the exception of `-r`, `-o` and `-v`, all options can be specified multiple times to run multiple tests. Each pp and tg test is run with all combinations of the specified options. To specify multiple values for an option, the values can be separated by commas (e.g. `-n 16,32`), or the option can be specified multiple times (e.g. `-n 16 -n 32`).

Each test is repeated the number of times given by `-r`, and the results are averaged. The results are given in average tokens per second (t/s) and standard deviation. Some output formats (e.g. json) also include the individual results of each repetition.

For a description of the other options, see the [main example](../main/README.md).

Note:

- When using SYCL backend, there would be hang issue in some cases. Please set `--mmp 0`.

## Examples

### Prompt processing and text generation

```sh
$ ./llama-bench -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p 0 -n 0 -pg 100,100 -pg 500,100
```

| model                          |     params | backend    | ngl |          test |         prompt t/s |         gen t/s |
| ------------------------------ | ---------: | ---------- | --: | ------------: | -----------------: | --------------: |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |   pp100+tg100 |  14303.91 ± 362.95 |  455.50 ± 11.99 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |   pp500+tg100 |  28026.49 ± 970.98 |   440.05 ± 4.57 |


### Text generation with different models

```sh
$ ./llama-bench -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -p 0 -n 128,256,512
```

| model                          |     params | backend    | ngl |          test |         prompt t/s |         gen t/s |
| ------------------------------ | ---------: | ---------- | --: | ------------: | -----------------: | --------------: |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |         tg128 |        0.00 ± 0.00 |   469.34 ± 2.16 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |         tg256 |        0.00 ± 0.00 |   459.78 ± 9.43 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |         tg512 |        0.00 ± 0.00 |  449.25 ± 11.74 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |   pp4096+tg32 |    15545.82 ± 8.35 |   385.90 ± 3.47 |
| llama 3B Q4_K - Medium         |     3.21 B | CUDA       |  99 |         tg128 |        0.00 ± 0.00 |   212.78 ± 5.12 |
| llama 3B Q4_K - Medium         |     3.21 B | CUDA       |  99 |         tg256 |        0.00 ± 0.00 |   214.56 ± 2.16 |
| llama 3B Q4_K - Medium         |     3.21 B | CUDA       |  99 |         tg512 |        0.00 ± 0.00 |   212.84 ± 1.41 |
| llama 3B Q4_K - Medium         |     3.21 B | CUDA       |  99 |   pp4096+tg32 |   8825.07 ± 100.28 |   177.25 ± 1.89 |

### Prompt processing with different batch sizes

```sh
$ ./llama-bench -n 0 -p 1024 -b 128,256,512,1024
```

| model                          |     params | backend    | ngl | n_batch |          test |         prompt t/s |         gen t/s |
| ------------------------------ | ---------: | ---------- | --: | ------: | ------------: | -----------------: | --------------: |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |     128 |        pp1024 |  17125.18 ± 731.13 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |     128 |   pp4096+tg32 |  12139.39 ± 446.63 |   378.76 ± 8.18 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |     256 |        pp1024 |  24112.17 ± 161.18 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |     256 |   pp4096+tg32 |   14508.80 ± 53.00 |   386.58 ± 0.42 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |     512 |        pp1024 |  25534.56 ± 368.03 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |     512 |   pp4096+tg32 |   15388.41 ± 13.06 |   386.30 ± 0.53 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |    1024 |        pp1024 |  25654.61 ± 772.86 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |    1024 |   pp4096+tg32 |    15487.92 ± 8.59 |   385.20 ± 0.50 |

### Different numbers of threads

```sh
$ ./llama-bench -n 0 -n 16 -p 64 -t 1,2,4,8,16,32
```

| model                          |     params | backend    | ngl | threads |          test |         prompt t/s |         gen t/s |
| ------------------------------ | ---------: | ---------- | --: | ------: | ------------: | -----------------: | --------------: |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       1 |          pp64 |  9229.99 ± 1897.41 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       1 |          tg16 |        0.00 ± 0.00 |  444.33 ± 25.11 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       1 |   pp4096+tg32 |   15357.53 ± 27.52 |   373.90 ± 7.03 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       2 |          pp64 |   10799.57 ± 33.90 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       2 |          tg16 |        0.00 ± 0.00 |  461.43 ± 10.99 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       2 |   pp4096+tg32 |   15371.18 ± 57.24 |   372.59 ± 4.02 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       4 |          pp64 |  11033.35 ± 177.05 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       4 |          tg16 |        0.00 ± 0.00 |   448.57 ± 8.66 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       4 |   pp4096+tg32 |   15371.12 ± 43.70 |   376.71 ± 0.93 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       8 |          pp64 |  11206.45 ± 187.47 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       8 |          tg16 |        0.00 ± 0.00 |   457.99 ± 6.92 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       8 |   pp4096+tg32 |  15022.14 ± 161.68 |   369.76 ± 4.71 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |      16 |          pp64 |  10397.19 ± 304.08 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |      16 |          tg16 |        0.00 ± 0.00 |   457.53 ± 7.06 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |      16 |   pp4096+tg32 |  15434.32 ± 158.08 |   372.00 ± 3.34 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |      32 |          pp64 | 10588.34 ± 1043.71 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |      32 |          tg16 |        0.00 ± 0.00 |   468.10 ± 9.16 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |      32 |   pp4096+tg32 |    15544.54 ± 4.30 |   374.14 ± 7.18 |

### Different numbers of layers offloaded to the GPU

```sh
$ ./llama-bench -ngl 10,20,30,31,32,33,34,35
```

| model                          |     params | backend    | ngl |          test |         prompt t/s |         gen t/s |
| ------------------------------ | ---------: | ---------- | --: | ------------: | -----------------: | --------------: |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  10 |          tg32 |        0.00 ± 0.00 |   107.29 ± 1.37 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  10 |   pp4096+tg32 |   8458.79 ± 154.44 |    70.84 ± 0.10 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  20 |          tg32 |        0.00 ± 0.00 |   484.02 ± 0.93 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  20 |   pp4096+tg32 |  15303.20 ± 120.74 |   372.57 ± 6.32 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  30 |          tg32 |        0.00 ± 0.00 |   473.82 ± 4.27 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  30 |   pp4096+tg32 |  15372.85 ± 239.94 |   378.99 ± 4.72 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  31 |          tg32 |        0.00 ± 0.00 |   474.76 ± 7.11 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  31 |   pp4096+tg32 |  15373.12 ± 263.84 |  377.83 ± 12.16 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  32 |          tg32 |        0.00 ± 0.00 |   482.19 ± 0.92 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  32 |   pp4096+tg32 |   15515.24 ± 15.85 |   369.73 ± 0.23 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  33 |          tg32 |        0.00 ± 0.00 |   482.07 ± 0.63 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  33 |   pp4096+tg32 |  15299.93 ± 261.50 |   373.32 ± 9.92 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  34 |          tg32 |        0.00 ± 0.00 |   482.89 ± 0.99 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  34 |   pp4096+tg32 |   15551.65 ± 14.10 |   381.00 ± 6.75 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  35 |          tg32 |        0.00 ± 0.00 |   481.55 ± 1.15 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  35 |   pp4096+tg32 |    15565.34 ± 5.96 |   385.77 ± 0.25 |

## Output formats

By default, llama-bench outputs the results in markdown format. The results can be output in other formats by using the `-o` option.

### Markdown

```sh
$ ./llama-bench -o md
```

| model                          |     params | backend    | ngl |          test |         prompt t/s |         gen t/s |
| ------------------------------ | ---------: | ---------- | --: | ------------: | -----------------: | --------------: |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |          tg32 |        0.00 ± 0.00 |  455.34 ± 13.25 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |   pp4096+tg32 |   15479.05 ± 93.15 |   383.70 ± 2.79 |

### CSV

```sh
$ ./llama-bench -o csv
```

```csv
build_commit,build_number,cpu_info,gpu_info,backends,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_mask,cpu_strict,poll,type_k,type_v,n_gpu_layers,split_mode,main_gpu,no_kv_offload,flash_attn,tensor_split,use_mmap,embeddings,n_prompt,n_gen,test_time,avg_prompt_ns,stddev_prompt_ns,avg_prompt_ts,stddev_prompt_ts,avg_gen_ns,stddev_gen_ns,avg_gen_ts,stddev_gen_ts
"fa6cb8ae","5100","AMD Ryzen 7 7800X3D 8-Core Processor           ","NVIDIA GeForce RTX 4080","CUDA","models/Llama-3.2-1B-Instruct-Q4_K_M.gguf","llama 1B Q4_K - Medium","799862912","1235814432","2048","512","8","0x0","0","50","f16","f16","99","layer","0","0","0","0.00","1","0","0","32","2025-04-18T11:21:18Z","66","58","0.000000","0.000000","71886000","7590","445.149267","0.046999"
"fa6cb8ae","5100","AMD Ryzen 7 7800X3D 8-Core Processor           ","NVIDIA GeForce RTX 4080","CUDA","models/Llama-3.2-1B-Instruct-Q4_K_M.gguf","llama 1B Q4_K - Medium","799862912","1235814432","2048","512","8","0x0","0","50","f16","f16","99","layer","0","0","0","0.00","1","0","4096","32","2025-04-18T11:21:18Z","272293733","3247466","15044.014817","180.586130","87201066","125581","366.968490","0.525734"
```

### JSON

```sh
$ ./llama-bench -o json
```

```json
[
  {
    "build_commit": "fa6cb8ae",
    "build_number": 5100,
    "cpu_info": "AMD Ryzen 7 7800X3D 8-Core Processor           ",
    "gpu_info": "NVIDIA GeForce RTX 4080",
    "backends": "CUDA",
    "model_filename": "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "model_type": "llama 1B Q4_K - Medium",
    "model_size": 799862912,
    "model_n_params": 1235814432,
    "n_batch": 2048,
    "n_ubatch": 512,
    "n_threads": 8,
    "cpu_mask": "0x0",
    "cpu_strict": false,
    "poll": 50,
    "type_k": "f16",
    "type_v": "f16",
    "n_gpu_layers": 99,
    "split_mode": "layer",
    "main_gpu": 0,
    "no_kv_offload": false,
    "flash_attn": false,
    "tensor_split": "0.00",
    "use_mmap": true,
    "embeddings": false,
    "n_prompt": 0,
    "n_gen": 32,
    "test_time": "2025-04-18T11:21:45Z",
    "avg_prompt_ns": 66,
    "stddev_prompt_ns": 58,
    "avg_prompt_ts": 0.000000,
    "stddev_prompt_ts": 0.000000,
    "avg_gen_ns": 67903233,
    "stddev_gen_ns": 498856,
    "avg_gen_ts": 471.275875,
    "stddev_gen_ts": 3.475513,
    "samples_prompt_ns": [ 100, 0, 100 ],
    "samples_prompt_ts": [ 0 ]
    "samples_gen_ns": [ 68251300, 68126600, 67331800 ],
    "samples_gen_ts": [ 468.856, 469.714, 475.258 ]
  },
  {
    "build_commit": "fa6cb8ae",
    "build_number": 5100,
    "cpu_info": "AMD Ryzen 7 7800X3D 8-Core Processor           ",
    "gpu_info": "NVIDIA GeForce RTX 4080",
    "backends": "CUDA",
    "model_filename": "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "model_type": "llama 1B Q4_K - Medium",
    "model_size": 799862912,
    "model_n_params": 1235814432,
    "n_batch": 2048,
    "n_ubatch": 512,
    "n_threads": 8,
    "cpu_mask": "0x0",
    "cpu_strict": false,
    "poll": 50,
    "type_k": "f16",
    "type_v": "f16",
    "n_gpu_layers": 99,
    "split_mode": "layer",
    "main_gpu": 0,
    "no_kv_offload": false,
    "flash_attn": false,
    "tensor_split": "0.00",
    "use_mmap": true,
    "embeddings": false,
    "n_prompt": 4096,
    "n_gen": 32,
    "test_time": "2025-04-18T11:21:46Z",
    "avg_prompt_ns": 263273600,
    "stddev_prompt_ns": 273278,
    "avg_prompt_ts": 15557.970647,
    "stddev_prompt_ts": 16.143068,
    "avg_gen_ns": 85820333,
    "stddev_gen_ns": 4372337,
    "avg_gen_ts": 373.500825,
    "stddev_gen_ts": 18.514532,
    "samples_prompt_ns": [ 263043600, 263201500, 263575700 ],
    "samples_prompt_ts": [ 15571.6, 15562.2, 15540.1 ]
    "samples_gen_ns": [ 82844300, 83776400, 90840300 ],
    "samples_gen_ts": [ 386.267, 381.969, 352.267 ]
  }
]
```


### JSONL

```sh
$ ./llama-bench -o jsonl
```

```json lines
{"build_commit": "fa6cb8ae", "build_number": 5100, "cpu_info": "AMD Ryzen 7 7800X3D 8-Core Processor           ", "gpu_info": "NVIDIA GeForce RTX 4080", "backends": "CUDA", "model_filename": "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf", "model_type": "llama 1B Q4_K - Medium", "model_size": 799862912, "model_n_params": 1235814432, "n_batch": 2048, "n_ubatch": 512, "n_threads": 8, "cpu_mask": "0x0", "cpu_strict": false, "poll": 50, "type_k": "f16", "type_v": "f16", "n_gpu_layers": 99, "split_mode": "layer", "main_gpu": 0, "no_kv_offload": false, "flash_attn": false, "tensor_split": "0.00", "use_mmap": true, "embeddings": false, "n_prompt": 0, "n_gen": 32, "test_time": "2025-04-18T11:22:14Z", "avg_prompt_ns": 100, "stddev_prompt_ns": 0, "avg_prompt_ts": 0.000000, "stddev_prompt_ts": 0.000000, "avg_gen_ns": 71156300, "stddev_gen_ns": 912152, "avg_gen_ts": 449.763857, "stddev_gen_ts": 5.808090, "samples_prompt_ns": [ 100, 100, 100 ],"samples_prompt_ts": [ 0 ]"samples_gen_ns": [ 71725200, 71639500, 70104200 ],"samples_gen_ts": [ 446.147, 446.681, 456.463 ]}
{"build_commit": "fa6cb8ae", "build_number": 5100, "cpu_info": "AMD Ryzen 7 7800X3D 8-Core Processor           ", "gpu_info": "NVIDIA GeForce RTX 4080", "backends": "CUDA", "model_filename": "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf", "model_type": "llama 1B Q4_K - Medium", "model_size": 799862912, "model_n_params": 1235814432, "n_batch": 2048, "n_ubatch": 512, "n_threads": 8, "cpu_mask": "0x0", "cpu_strict": false, "poll": 50, "type_k": "f16", "type_v": "f16", "n_gpu_layers": 99, "split_mode": "layer", "main_gpu": 0, "no_kv_offload": false, "flash_attn": false, "tensor_split": "0.00", "use_mmap": true, "embeddings": false, "n_prompt": 4096, "n_gen": 32, "test_time": "2025-04-18T11:22:14Z", "avg_prompt_ns": 267673800, "stddev_prompt_ns": 4917668, "avg_prompt_ts": 15305.627579, "stddev_prompt_ts": 279.255714, "avg_gen_ns": 83914500, "stddev_gen_ns": 1515058, "avg_gen_ts": 381.422650, "stddev_gen_ts": 6.822569, "samples_prompt_ns": [ 266315000, 273128000, 263578400 ],"samples_prompt_ts": [ 15380.3, 14996.6, 15540 ]"samples_gen_ns": [ 85644600, 83274100, 82824800 ],"samples_gen_ts": [ 373.637, 384.273, 386.358 ]}
```


### SQL

SQL output is suitable for importing into a SQLite database. The output can be piped into the `sqlite3` command line tool to add the results to a database.

```sh
$ ./llama-bench -o sql
```

```sql
CREATE TABLE IF NOT EXISTS test (
  build_commit TEXT,
  build_number INTEGER,
  cpu_info TEXT,
  gpu_info TEXT,
  backends TEXT,
  model_filename TEXT,
  model_type TEXT,
  model_size INTEGER,
  model_n_params INTEGER,
  n_batch INTEGER,
  n_ubatch INTEGER,
  n_threads INTEGER,
  cpu_mask TEXT,
  cpu_strict INTEGER,
  poll INTEGER,
  type_k TEXT,
  type_v TEXT,
  n_gpu_layers INTEGER,
  split_mode TEXT,
  main_gpu INTEGER,
  no_kv_offload INTEGER,
  flash_attn INTEGER,
  tensor_split TEXT,
  use_mmap INTEGER,
  embeddings INTEGER,
  n_prompt INTEGER,
  n_gen INTEGER,
  test_time TEXT,
  avg_prompt_ns INTEGER,
  stddev_prompt_ns INTEGER,
  avg_prompt_ts REAL,
  stddev_prompt_ts REAL,
  avg_gen_ns INTEGER,
  stddev_gen_ns INTEGER,
  avg_gen_ts REAL,
  stddev_gen_ts REAL
);

INSERT INTO test (build_commit, build_number, cpu_info, gpu_info, backends, model_filename, model_type, model_size, model_n_params, n_batch, n_ubatch, n_threads, cpu_mask, cpu_strict, poll, type_k, type_v, n_gpu_layers, split_mode, main_gpu, no_kv_offload, flash_attn, tensor_split, use_mmap, embeddings, n_prompt, n_gen, test_time, avg_prompt_ns, stddev_prompt_ns, avg_prompt_ts, stddev_prompt_ts, avg_gen_ns, stddev_gen_ns, avg_gen_ts, stddev_gen_ts) VALUES ('fa6cb8ae', '5100', 'AMD Ryzen 7 7800X3D 8-Core Processor           ', 'NVIDIA GeForce RTX 4080', 'CUDA', 'models/Llama-3.2-1B-Instruct-Q4_K_M.gguf', 'llama 1B Q4_K - Medium', '799862912', '1235814432', '2048', '512', '8', '0x0', '0', '50', 'f16', 'f16', '99', 'layer', '0', '0', '0', '0.00', '1', '0', '0', '32', '2025-04-18T11:22:37Z', '66', '58', '0.000000', '0.000000', '70741266', '2050337', '452.606173', '13.122321');
INSERT INTO test (build_commit, build_number, cpu_info, gpu_info, backends, model_filename, model_type, model_size, model_n_params, n_batch, n_ubatch, n_threads, cpu_mask, cpu_strict, poll, type_k, type_v, n_gpu_layers, split_mode, main_gpu, no_kv_offload, flash_attn, tensor_split, use_mmap, embeddings, n_prompt, n_gen, test_time, avg_prompt_ns, stddev_prompt_ns, avg_prompt_ts, stddev_prompt_ts, avg_gen_ns, stddev_gen_ns, avg_gen_ts, stddev_gen_ts) VALUES ('fa6cb8ae', '5100', 'AMD Ryzen 7 7800X3D 8-Core Processor           ', 'NVIDIA GeForce RTX 4080', 'CUDA', 'models/Llama-3.2-1B-Instruct-Q4_K_M.gguf', 'llama 1B Q4_K - Medium', '799862912', '1235814432', '2048', '512', '8', '0x0', '0', '50', 'f16', 'f16', '99', 'layer', '0', '0', '0', '0.00', '1', '0', '4096', '32', '2025-04-18T11:22:37Z', '270934866', '4466069', '15120.737903', '246.900896', '85258733', '2156168', '375.487736', '9.468350');
```
