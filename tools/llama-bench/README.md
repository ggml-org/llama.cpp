# llama.cpp/tools/llama-bench

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
usage: llama-bench [options]

options:
  -h, --help
  --numa <distribute|isolate|numactl>       numa mode (default: disabled)
  -r, --repetitions <n>                     number of times to repeat each test (default: 5)
  --prio <-1|0|1|2|3>                       process/thread priority (default: 0)
  --delay <0...N> (seconds)                 delay between each test (default: 0)
  -o, --output <csv|json|jsonl|md|sql>      output format printed to stdout (default: md)
  -oe, --output-err <csv|json|jsonl|md|sql> output format printed to stderr (default: none)
  --list-devices                            list available devices and exit
  -v, --verbose                             verbose output
  --progress                                print test progress indicators
  --no-warmup                               skip warmup runs before benchmarking
  --n-gen-warmup <n>                       generation tokens to run before timing (default: 1)
  -fitt, --fit-target <MiB>                 fit model to device memory with this margin per device in MiB (default: off)
  -fitc, --fit-ctx <n>                      minimum ctx size for --fit-target (default: 4096)
  -rpc, --rpc <rpc_servers>                 register RPC devices (comma separated)

test parameters:
  -m, --model <filename>                    (default: models/7B/ggml-model-q4_0.gguf)
  -hf, -hfr, --hf-repo <user>/<model>[:quant] Hugging Face model repository; quant is optional, case-insensitive
                                            default to Q4_K_M, or falls back to the first file in the repo if Q4_K_M doesn't exist.
                                            example: ggml-org/GLM-4.7-Flash-GGUF:Q4_K_M
                                            (default: unused)
  -hff, --hf-file <file>                    Hugging Face model file. If specified, it will override the quant in --hf-repo
                                            (default: unused)
  -hft, --hf-token <token>                  Hugging Face access token
                                            (default: value from HF_TOKEN environment variable)
  --offline                                 use cached model files and disable network access
                                            (default: disabled)
  -p, --n-prompt <n>                        (default: 512)
  -n, --n-gen <n>                           (default: 128)
  -pg <pp,tg>                               (default: )
  -d, --n-depth <n>                         (default: 0)
  -b, --batch-size <n>                      (default: 2048)
  -ub, --ubatch-size <n>                    (default: 512)
  -ctk, --cache-type-k <t>                  (default: f16)
  -ctv, --cache-type-v <t>                  (default: f16)
  -t, --threads <n>                         (default: system dependent)
  -C, --cpu-mask <hex,hex>                  (default: 0x0)
  --cpu-strict <0|1>                        (default: 0)
  --poll <0...100>                          (default: 50)
  -ngl, --n-gpu-layers <n>                  (default: -1)
  -ncmoe, --n-cpu-moe <n>                   (default: 0)
  --moe-cache <auto|on|off|0|MiB>           (default: auto)
  --repack <auto|on|off>                    weight repacking policy (default: auto)
  -nr, --no-repack                          equivalent to --repack off
  -sm, --split-mode <none|layer|row|tensor> (default: layer)
  -mg, --main-gpu <i>                       (default: 0)
  -nkvo, --no-kv-offload <0|1>              (default: 0)
  -fa, --flash-attn <on|off|auto>           (default: auto)
  -dev, --device <dev0/dev1/...>            (default: auto)
  -lm, --load-mode <auto|none|mmap|mlock|mmap+mlock|dio>
                                            (default: auto)
  -lzm, --lazy-mode <on|auto|off>           (default: auto)
  -embd, --embeddings <0|1>                 (default: 0)
  -ts, --tensor-split <ts0/ts1/..>          (default: 0)
  -ot --override-tensor <tensor name pattern>=<buffer type>;...
                                            (default: disabled)
  -nopo, --no-op-offload <0|1>              (default: 0)
  --no-host <0|1>                           (default: 0)
  --repack <0|1>                            (default: 1)

Multiple values can be given for each parameter by separating them with ','
or by specifying the parameter multiple times. Ranges can be given as
'first-last' or 'first-last+step' or 'first-last*mult'.
```

llama-bench can perform three types of tests:

- Prompt processing (pp): processing a prompt in batches (`-p`)
- Text generation (tg): generating a sequence of tokens (`-n`)
- Prompt processing + text generation (pg): processing a prompt followed by generating a sequence of tokens (`-pg`)

List-valued options can be specified multiple times to run multiple tests. Each pp and tg test is run with all combinations of those options. Multiple values can be separated by commas (e.g. `-n 16,32`), or the option can be specified multiple times (e.g. `-n 16 -n 32`). Scalar options such as `-r`, `-o`, `-v`, `--n-gen-warmup`, and `--repack` apply to all generated tests.

Each test is repeated the number of times given by `-r`, and the results are averaged. The results are given in average tokens per second (t/s) and standard deviation. Some output formats (e.g. json) also include the individual results of each repetition.

Using the `-d <n>` option, each test can be run at a specified context depth, prefilling the KV cache with `<n>` tokens.

`--n-gen-warmup <n>` controls how many generation tokens run before timing. This is useful when a benchmark needs to populate persistent runtime state such as the MoE expert cache. `--moe-cache off` provides a hard-disabled baseline. `auto` preserves repacking and may remain dormant, while `on` and a positive per-device MiB budget use canonical CPU weights and disable repacking. `--repack on` is rejected for those modes. Use `--repack off` for a matched canonical-weight baseline, for example `--moe-cache off,4096 --repack off`.

Prompt and generation inputs use deterministic synthetic token traces. Every cache or repack arm receives the same trace for a given repetition, so matched A/B tests do not measure different expert-routing inputs. The generation warmup uses a separate fixed trace and is excluded from the timed region.

llama-bench sets `GGML_CUDA_MOE_CACHE`, `GGML_CUDA_MOE_CACHE_MODE`, and `GGML_CUDA_MOE_CACHE_BUDGET_MB` for each test instance. Do not use those raw variables to select benchmark arms; use `--moe-cache` or `LLAMA_ARG_MOE_CACHE`. Other cache tuning variables, such as the reserve and admission settings, are not replaced.

For a description of the other options, see the [completion example](../completion/README.md).

> [!NOTE]
> The measurements with `llama-bench` do not include the times for tokenization and for sampling.

## Examples

### Text generation with different models

```sh
$ ./llama-bench -m models/7B/ggml-model-q4_0.gguf -m models/13B/ggml-model-q4_0.gguf -p 0 -n 128,256,512
```

| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  -1 | tg 128     |    132.19 ± 0.55 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  -1 | tg 256     |    129.37 ± 0.54 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  -1 | tg 512     |    123.83 ± 0.25 |
| llama 13B mostly Q4_0          |   6.86 GiB |    13.02 B | CUDA       |  -1 | tg 128     |     82.17 ± 0.31 |
| llama 13B mostly Q4_0          |   6.86 GiB |    13.02 B | CUDA       |  -1 | tg 256     |     80.74 ± 0.23 |
| llama 13B mostly Q4_0          |   6.86 GiB |    13.02 B | CUDA       |  -1 | tg 512     |     78.08 ± 0.07 |

### Prompt processing with different batch sizes

```sh
$ ./llama-bench -n 0 -p 1024 -b 128,256,512,1024
```

| model                          |       size |     params | backend    | ngl |    n_batch | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | ---------- | ---------------: |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  -1 |        128 | pp 1024    |   1436.51 ± 3.66 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  -1 |        256 | pp 1024    |  1932.43 ± 23.48 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  -1 |        512 | pp 1024    |  2254.45 ± 15.59 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  -1 |       1024 | pp 1024    |  2498.61 ± 13.58 |

### Different numbers of threads

```sh
$ ./llama-bench -n 0 -n 16 -p 64 -t 1,2,4,8,16,32
```

| model                          |       size |     params | backend    |    threads | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ---------: | ---------- | ---------------: |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CPU        |          1 | pp 64      |      6.17 ± 0.07 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CPU        |          1 | tg 16      |      4.05 ± 0.02 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CPU        |          2 | pp 64      |     12.31 ± 0.13 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CPU        |          2 | tg 16      |      7.80 ± 0.07 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CPU        |          4 | pp 64      |     23.18 ± 0.06 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CPU        |          4 | tg 16      |     12.22 ± 0.07 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CPU        |          8 | pp 64      |     32.29 ± 1.21 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CPU        |          8 | tg 16      |     16.71 ± 0.66 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CPU        |         16 | pp 64      |     33.52 ± 0.03 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CPU        |         16 | tg 16      |     15.32 ± 0.05 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CPU        |         32 | pp 64      |     59.00 ± 1.11 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CPU        |         32 | tg 16      |     16.41 ± 0.79 |

### Different numbers of layers offloaded to the GPU

```sh
$ ./llama-bench -ngl 10,20,30,31,32,33,34,35
```

| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  10 | pp 512     |    373.36 ± 2.25 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  10 | tg 128     |     13.45 ± 0.93 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  20 | pp 512     |    472.65 ± 1.25 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  20 | tg 128     |     21.36 ± 1.94 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  30 | pp 512     |   631.87 ± 11.25 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  30 | tg 128     |     40.04 ± 1.82 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  31 | pp 512     |    657.89 ± 5.08 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  31 | tg 128     |     48.19 ± 0.81 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  32 | pp 512     |    688.26 ± 3.29 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  32 | tg 128     |     54.78 ± 0.65 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  33 | pp 512     |    704.27 ± 2.24 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  33 | tg 128     |     60.62 ± 1.76 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  34 | pp 512     |    881.34 ± 5.40 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  34 | tg 128     |     71.76 ± 0.23 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  35 | pp 512     |   2400.01 ± 7.72 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  35 | tg 128     |    131.66 ± 0.49 |

### Different prefilled context

```
$ ./llama-bench -d 0,512
```

| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen2 7B Q4_K - Medium         |   4.36 GiB |     7.62 B | CUDA       |  -1 |           pp512 |      7340.20 ± 23.45 |
| qwen2 7B Q4_K - Medium         |   4.36 GiB |     7.62 B | CUDA       |  -1 |           tg128 |        120.60 ± 0.59 |
| qwen2 7B Q4_K - Medium         |   4.36 GiB |     7.62 B | CUDA       |  -1 |    pp512 @ d512 |      6425.91 ± 18.88 |
| qwen2 7B Q4_K - Medium         |   4.36 GiB |     7.62 B | CUDA       |  -1 |    tg128 @ d512 |        116.71 ± 0.60 |

## Output formats

By default, llama-bench outputs the results in markdown format. The results can be output in other formats by using the `-o` option.

### Markdown

```sh
$ ./llama-bench -o md
```

| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  -1 | pp 512     |  2368.80 ± 93.24 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  -1 | tg 128     |    131.42 ± 0.59 |

The CSV, JSON, JSONL, and SQL printers use the same runtime-generated schema. In addition to model, hardware, test-shape, and timing data, the schema includes these configuration fields:

| field | type | meaning |
| ----- | ---- | ------- |
| `load_mode` | string | effective model load mode, such as `mmap`, `mlock`, or `dio` |
| `moe_cache` | string | selected cache policy: `auto`, `on`, `off`, or a per-device MiB budget |
| `repack` | boolean | whether the loader allows extra buffer types used for weight repacking after cache compatibility rules |
| `n_gen_warmup` | integer | effective number of untimed generation warmup tokens |

`moe_cache` records the requested policy, not proof that a cache pool engaged. `repack` records loader policy, not proof that every eligible tensor was repacked. The full field list is emitted by the selected formatter, which avoids copying a version-specific schema into this document.

### CSV

```sh
$ ./llama-bench -o csv
```

CSV prints one header row followed by one row per benchmark result. Repetition samples are summarized by the aggregate timing fields.

Historical upstream output illustrating this format (older schema and hardware; the current additional fields are listed above):

```csv
build_commit,build_number,cpu_info,gpu_info,backends,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_mask,cpu_strict,poll,type_k,type_v,n_gpu_layers,n_cpu_moe,split_mode,main_gpu,no_kv_offload,flash_attn,devices,tensor_split,tensor_buft_overrides,use_mmap,use_direct_io,embeddings,no_op_offload,no_host,repack,fit_target,fit_min_ctx,n_prompt,n_gen,n_depth,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts
"8cf427ff","5163","AMD Ryzen 7 7800X3D 8-Core Processor","NVIDIA GeForce RTX 4080","CUDA","models/Qwen2.5-7B-Instruct-Q4_K_M.gguf","qwen2 7B Q4_K - Medium","4677120000","7615616512","2048","512","8","0x0","0","50","f16","f16","-1","0","layer","0","0","-1","auto","0.00","none","1","0","0","0","0","0","0","512","0","0","2025-04-24T11:57:09Z","70285660","982040","7285.676949","100.064434"
"8cf427ff","5163","AMD Ryzen 7 7800X3D 8-Core Processor","NVIDIA GeForce RTX 4080","CUDA","models/Qwen2.5-7B-Instruct-Q4_K_M.gguf","qwen2 7B Q4_K - Medium","4677120000","7615616512","2048","512","8","0x0","0","50","f16","f16","-1","0","layer","0","0","-1","auto","0.00","none","1","0","0","0","0","0","0","0","128","0","2025-04-24T11:57:10Z","1067431600","3834831","119.915244","0.430617"
```

### JSON

```sh
$ ./llama-bench -o json
```

JSON prints an array of benchmark objects. Each object also contains `samples_ns` and `samples_ts` arrays for the individual repetitions.

Historical upstream output illustrating this format (older schema and hardware; the current additional fields are listed above):

```json
[
  {
    "build_commit": "8cf427ff",
    "build_number": 5163,
    "cpu_info": "AMD Ryzen 7 7800X3D 8-Core Processor",
    "gpu_info": "NVIDIA GeForce RTX 4080",
    "backends": "CUDA",
    "model_filename": "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "model_type": "qwen2 7B Q4_K - Medium",
    "model_size": 4677120000,
    "model_n_params": 7615616512,
    "n_batch": 2048,
    "n_ubatch": 512,
    "n_threads": 8,
    "cpu_mask": "0x0",
    "cpu_strict": false,
    "poll": 50,
    "type_k": "f16",
    "type_v": "f16",
    "n_gpu_layers": -1,
    "n_cpu_moe": 0,
    "split_mode": "layer",
    "main_gpu": 0,
    "no_kv_offload": false,
    "flash_attn": -1,
    "devices": "auto",
    "tensor_split": "0.00",
    "tensor_buft_overrides": "none",
    "use_mmap": true,
    "use_direct_io": false,
    "embeddings": false,
    "no_op_offload": 0,
    "no_host": false,
    "repack": true,
    "fit_target": 0,
    "fit_min_ctx": 0,
    "n_prompt": 512,
    "n_gen": 0,
    "n_depth": 0,
    "test_time": "2025-04-24T11:58:50Z",
    "avg_ns": 72135640,
    "stddev_ns": 1453752,
    "avg_ts": 7100.002165,
    "stddev_ts": 140.341520,
    "samples_ns": [ 74601900, 71632900, 71745200, 71952700, 70745500 ],
    "samples_ts": [ 6863.1, 7147.55, 7136.37, 7115.79, 7237.21 ]
  },
  {
    "build_commit": "8cf427ff",
    "build_number": 5163,
    "cpu_info": "AMD Ryzen 7 7800X3D 8-Core Processor",
    "gpu_info": "NVIDIA GeForce RTX 4080",
    "backends": "CUDA",
    "model_filename": "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "model_type": "qwen2 7B Q4_K - Medium",
    "model_size": 4677120000,
    "model_n_params": 7615616512,
    "n_batch": 2048,
    "n_ubatch": 512,
    "n_threads": 8,
    "cpu_mask": "0x0",
    "cpu_strict": false,
    "poll": 50,
    "type_k": "f16",
    "type_v": "f16",
    "n_gpu_layers": -1,
    "n_cpu_moe": 0,
    "split_mode": "layer",
    "main_gpu": 0,
    "no_kv_offload": false,
    "flash_attn": -1,
    "devices": "auto",
    "tensor_split": "0.00",
    "tensor_buft_overrides": "none",
    "use_mmap": true,
    "use_direct_io": false,
    "embeddings": false,
    "no_op_offload": 0,
    "no_host": false,
    "repack": true,
    "fit_target": 0,
    "fit_min_ctx": 0,
    "n_prompt": 0,
    "n_gen": 128,
    "n_depth": 0,
    "test_time": "2025-04-24T11:58:51Z",
    "avg_ns": 1076767880,
    "stddev_ns": 9449585,
    "avg_ts": 118.881588,
    "stddev_ts": 1.041811,
    "samples_ns": [ 1075361300, 1065089400, 1071761200, 1081934900, 1089692600 ],
    "samples_ts": [ 119.03, 120.178, 119.43, 118.307, 117.464 ]
  }
]
```

### JSONL

```sh
$ ./llama-bench -o jsonl
```

JSONL prints the same objects as JSON, one complete object per line.

### SQL

SQL output creates a `llama_bench` table from the current runtime schema and emits one `INSERT` statement per result. It can be piped directly into SQLite:

```sh
$ ./llama-bench -o sql | sqlite3 benchmark.sqlite
```
