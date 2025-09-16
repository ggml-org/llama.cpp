# llama.cpp/tools/llama-bench

Performance testing tool for llama.cpp.

## Table of contents

1. [Syntax](#syntax)
2. [Metrics](#metrics)
3. [Examples](#examples)
    1. [Text generation with different models](#text-generation-with-different-models)
    2. [Prompt processing with different batch sizes](#prompt-processing-with-different-batch-sizes)
    3. [Different numbers of threads](#different-numbers-of-threads)
    4. [Different numbers of layers offloaded to the GPU](#different-numbers-of-layers-offloaded-to-the-gpu)
4. [Output formats](#output-formats)
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
  --prio <0|1|2|3>                          process/thread priority (default: 0)
  --delay <0...N> (seconds)                 delay between each test (default: 0)
  -o, --output <csv|json|jsonl|md|sql>      output format printed to stdout (default: md)
  -oe, --output-err <csv|json|jsonl|md|sql> output format printed to stderr (default: none)
  -v, --verbose                             verbose output
  --progress                                print test progress indicators

test parameters:
  -m, --model <filename>                    (default: models/7B/ggml-model-q4_0.gguf)
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
  -ngl, --n-gpu-layers <n>                  (default: 99)
  -rpc, --rpc <rpc_servers>                 (default: none)
  -sm, --split-mode <none|layer|row>        (default: layer)
  -mg, --main-gpu <i>                       (default: 0)
  -nkvo, --no-kv-offload <0|1>              (default: 0)
  -fa, --flash-attn <0|1>                   (default: 0)
  -mmp, --mmap <0|1>                        (default: 1)
  -embd, --embeddings <0|1>                 (default: 0)
  -ts, --tensor-split <ts0/ts1/..>          (default: 0)
  -ot --override-tensors <tensor name pattern>=<buffer type>;...
                                            (default: disabled)
  -nopo, --no-op-offload <0|1>              (default: 0)

Multiple values can be given for each parameter by separating them with ','
or by specifying the parameter multiple times. Ranges can be given as
'first-last' or 'first-last+step' or 'first-last*mult'.
```

llama-bench can perform three types of tests:

- Prompt processing (pp): processing a prompt in batches (`-p`)
- Text generation (tg): generating a sequence of tokens (`-n`)
- Prompt processing + text generation (pg): processing a prompt followed by generating a sequence of tokens (`-pg`)

With the exception of `-r`, `-o` and `-v`, all options can be specified multiple times to run multiple tests. Each pp and tg test is run with all combinations of the specified options. To specify multiple values for an option, the values can be separated by commas (e.g. `-n 16,32`), or the option can be specified multiple times (e.g. `-n 16 -n 32`).

Each test is repeated the number of times given by `-r`, and the results are averaged. The results are given in average tokens per second (t/s) and standard deviation. Some output formats (e.g. json) also include the individual results of each repetition.

Using the `-d <n>` option, each test can be run at a specified context depth, prefilling the KV cache with `<n>` tokens.

For a description of the other options, see the [main example](../main/README.md).

## Metrics

### Time to First Token (TTFT)

$$ T_{ttft} = t_{prompt} + t^{(1)}_{gen} $$

where
* $t_{prompt}$ : total prompt processing time
* $t^{(1)}_{gen}$ : token generation time for the first token

> **_NOTE_**: This is only meaningful with a `-pg` test.

### End-to-End Request Latency (E2E)

$$ T_{e2e} = t_{prompt} + t_{gen} $$

where
* $t_{prompt}$ : total prompt processing time
* $t_{gen}$ : total token generation time

### Inter-token Latency (ITL)

$$ T_{itl} = \frac{T_{e2e} - T_{ttft}}{n\_{gen} - 1} $$

where
* $n\_{gen}$ : tokens to generate (`-n` flag)

## Examples

### Text generation with different models

```sh
$ ./llama-bench -m models/7B/ggml-model-q4_0.gguf -m models/13B/ggml-model-q4_0.gguf -p 0 -n 128,256,512
```

| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  99 | tg 128     |    132.19 ± 0.55 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  99 | tg 256     |    129.37 ± 0.54 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  99 | tg 512     |    123.83 ± 0.25 |
| llama 13B mostly Q4_0          |   6.86 GiB |    13.02 B | CUDA       |  99 | tg 128     |     82.17 ± 0.31 |
| llama 13B mostly Q4_0          |   6.86 GiB |    13.02 B | CUDA       |  99 | tg 256     |     80.74 ± 0.23 |
| llama 13B mostly Q4_0          |   6.86 GiB |    13.02 B | CUDA       |  99 | tg 512     |     78.08 ± 0.07 |

### Prompt processing with different batch sizes

```sh
$ ./llama-bench -n 0 -p 1024 -b 128,256,512,1024
```

| model                          |       size |     params | backend    | ngl |    n_batch | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | ---------- | ---------------: |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  99 |        128 | pp 1024    |   1436.51 ± 3.66 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  99 |        256 | pp 1024    |  1932.43 ± 23.48 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  99 |        512 | pp 1024    |  2254.45 ± 15.59 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  99 |       1024 | pp 1024    |  2498.61 ± 13.58 |

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
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CPU        |         32 | tg 16      |     16.41 ± 0.79 ||

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
| qwen2 7B Q4_K - Medium         |   4.36 GiB |     7.62 B | CUDA       |  99 |           pp512 |      7340.20 ± 23.45 |
| qwen2 7B Q4_K - Medium         |   4.36 GiB |     7.62 B | CUDA       |  99 |           tg128 |        120.60 ± 0.59 |
| qwen2 7B Q4_K - Medium         |   4.36 GiB |     7.62 B | CUDA       |  99 |    pp512 @ d512 |      6425.91 ± 18.88 |
| qwen2 7B Q4_K - Medium         |   4.36 GiB |     7.62 B | CUDA       |  99 |    tg128 @ d512 |        116.71 ± 0.60 |

## Output formats

By default, llama-bench outputs the results in markdown format. The results can be output in other formats by using the `-o` option.

### Markdown

```sh
$ ./llama-bench -o md
```

| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  99 | pp 512     |  2368.80 ± 93.24 |
| llama 7B mostly Q4_0           |   3.56 GiB |     6.74 B | CUDA       |  99 | tg 128     |    131.42 ± 0.59 |

### CSV

```sh
$ ./llama-bench -o csv
```

```csv
build_commit,build_number,cpu_info,gpu_info,backends,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_mask,cpu_strict,poll,type_k,type_v,n_gpu_layers,split_mode,main_gpu,no_kv_offload,flash_attn,tensor_split,tensor_buft_overrides,use_mmap,embeddings,no_op_offload,n_prompt,n_gen,n_depth,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts,avg_ttft_ms,stddev_ttft_ms,avg_e2e_ms,stddev_e2e_ms,avg_itl_ms,stddev_itl_ms
"69f7e7116","6321","OpenBLAS, CPU","","BLAS","models/granite-3.3-2b-instruct-be.IQ4_XS.gguf","granite 3B IQ4_XS - 4.25 bpw","1395281392","2533539840","2048","512","8","0x0","0","50","f16","f16","99","layer","0","0","0","0.00","none","1","0","0","512","0","0","2025-08-28T16:57:00Z","6313008195","4119608","81.102409","0.052946","0.000000","0.000000","6313.008195","4.118842","0.000000","0.000000"
"69f7e7116","6321","OpenBLAS, CPU","","BLAS","models/granite-3.3-2b-instruct-be.IQ4_XS.gguf","granite 3B IQ4_XS - 4.25 bpw","1395281392","2533539840","2048","512","8","0x0","0","50","f16","f16","99","layer","0","0","0","0.00","none","1","0","0","0","128","0","2025-08-28T16:57:38Z","5510702782","66734031","23.230240","0.280041","42.929458","0.403130","5510.702782","66.734011","43.053333","0.522383"
```

### JSON

```sh
$ ./llama-bench -o json
```

```json
[
  {
    "build_commit": "69f7e7116",
    "build_number": 6321,
    "cpu_info": "OpenBLAS, CPU",
    "gpu_info": "",
    "backends": "BLAS",
    "model_filename": "models/granite-3.3-2b-instruct-be.IQ4_XS.gguf",
    "model_type": "granite 3B IQ4_XS - 4.25 bpw",
    "model_size": 1395281392,
    "model_n_params": 2533539840,
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
    "tensor_buft_overrides": "none",
    "use_mmap": true,
    "embeddings": false,
    "no_op_offload": 0,
    "n_prompt": 512,
    "n_gen": 0,
    "n_depth": 0,
    "test_time": "2025-08-28T17:01:34Z",
    "avg_ns": 6276064173,
    "stddev_ns": 34323113,
    "avg_ts": 81.581735,
    "stddev_ts": 0.444487,
    "avg_ttft_ms": 0.000000,
    "stddev_ttft_ms": 0.000000,
    "avg_e2e_ms": 6276.064174,
    "stddev_e2e_ms": 34.322931,
    "avg_itl_ms": 0.000000,
    "stddev_itl_ms": 0.000000,
    "samples_ns": [ 6255489794, 6293138165, 6328736359, 6254136857, 6248819694 ],
    "samples_ts": [ 81.8481, 81.3585, 80.9008, 81.8658, 81.9355 ],
    "samples_ttft_ns": [  ],
    "samples_itl_ns": [  ]
  },
  {
    "build_commit": "69f7e7116",
    "build_number": 6321,
    "cpu_info": "OpenBLAS, CPU",
    "gpu_info": "",
    "backends": "BLAS",
    "model_filename": "models/granite-3.3-2b-instruct-be.IQ4_XS.gguf",
    "model_type": "granite 3B IQ4_XS - 4.25 bpw",
    "model_size": 1395281392,
    "model_n_params": 2533539840,
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
    "tensor_buft_overrides": "none",
    "use_mmap": true,
    "embeddings": false,
    "no_op_offload": 0,
    "n_prompt": 0,
    "n_gen": 128,
    "n_depth": 0,
    "test_time": "2025-08-28T17:02:12Z",
    "avg_ns": 5613693967,
    "stddev_ns": 9159226,
    "avg_ts": 22.801434,
    "stddev_ts": 0.037157,
    "avg_ttft_ms": 43.766002,
    "stddev_ttft_ms": 0.161696,
    "avg_e2e_ms": 5613.693967,
    "stddev_e2e_ms": 9.158920,
    "avg_itl_ms": 43.857701,
    "stddev_itl_ms": 0.073127,
    "samples_ns": [ 5617273869, 5609876644, 5628089934, 5607197607, 5606031783 ],
    "samples_ts": [ 22.7869, 22.8169, 22.7431, 22.8278, 22.8325 ],
    "samples_ttft_ns": [ 43584301, 43738022, 43642511, 43936642, 43928534 ],
    "samples_itl_ns": [ 4.38873e+07, 4.38279e+07, 4.3972e+07, 4.38052e+07, 4.37961e+07 ]
  }
]
```


### JSONL

```sh
$ ./llama-bench -o jsonl
```

```json lines
{"build_commit": "69f7e7116", "build_number": 6321, "cpu_info": "OpenBLAS, CPU", "gpu_info": "", "backends": "BLAS", "model_filename": "models/granite-3.3-2b-instruct-be.IQ4_XS.gguf", "model_type": "granite 3B IQ4_XS - 4.25 bpw", "model_size": 1395281392, "model_n_params": 2533539840, "n_batch": 2048, "n_ubatch": 512, "n_threads": 8, "cpu_mask": "0x0", "cpu_strict": false, "poll": 50, "type_k": "f16", "type_v": "f16", "n_gpu_layers": 99, "split_mode": "layer", "main_gpu": 0, "no_kv_offload": false, "flash_attn": false, "tensor_split": "0.00", "tensor_buft_overrides": "none", "use_mmap": true, "embeddings": false, "no_op_offload": 0, "n_prompt": 512, "n_gen": 0, "n_depth": 0, "test_time": "2025-08-28T17:05:56Z", "avg_ns": 6287855687, "stddev_ns": 24678326, "avg_ts": 81.427806, "stddev_ts": 0.318701, "avg_ttft_ms": 0.000000, "stddev_ttft_ms": 0.000000, "avg_e2e_ms": 6287.855687, "stddev_e2e_ms": 24.678263, "avg_itl_ms": 0.000000, "stddev_itl_ms": 0.000000, "samples_ns": [ 6278179171, 6280580749, 6291595583, 6327539008, 6261383925 ],"samples_ts": [ 81.5523, 81.5211, 81.3784, 80.9161, 81.7711 ],"samples_ttft_ns": [  ],"samples_itl_ns": [  ]}
{"build_commit": "69f7e7116", "build_number": 6321, "cpu_info": "OpenBLAS, CPU", "gpu_info": "", "backends": "BLAS", "model_filename": "models/granite-3.3-2b-instruct-be.IQ4_XS.gguf", "model_type": "granite 3B IQ4_XS - 4.25 bpw", "model_size": 1395281392, "model_n_params": 2533539840, "n_batch": 2048, "n_ubatch": 512, "n_threads": 8, "cpu_mask": "0x0", "cpu_strict": false, "poll": 50, "type_k": "f16", "type_v": "f16", "n_gpu_layers": 99, "split_mode": "layer", "main_gpu": 0, "no_kv_offload": false, "flash_attn": false, "tensor_split": "0.00", "tensor_buft_overrides": "none", "use_mmap": true, "embeddings": false, "no_op_offload": 0, "n_prompt": 0, "n_gen": 128, "n_depth": 0, "test_time": "2025-08-28T17:06:33Z", "avg_ns": 5498943758, "stddev_ns": 137464807, "avg_ts": 23.288635, "stddev_ts": 0.571937, "avg_ttft_ms": 43.519547, "stddev_ttft_ms": 1.625999, "avg_e2e_ms": 5498.943758, "stddev_e2e_ms": 137.464807, "avg_itl_ms": 42.956096, "stddev_itl_ms": 1.075456, "samples_ns": [ 5563859512, 5712210875, 5398301124, 5410462401, 5409884878 ],"samples_ts": [ 23.0056, 22.4081, 23.7112, 23.6579, 23.6604 ],"samples_ttft_ns": [ 46182840, 43768889, 43176480, 42210650, 42258878 ],"samples_itl_ns": [ 4.34463e+07, 4.46334e+07, 4.21663e+07, 4.22697e+07, 4.22648e+07 ]}
```


### SQL

SQL output is suitable for importing into a SQLite database. The output can be piped into the `sqlite3` command line tool to add the results to a database.

```sh
$ ./llama-bench -o sql
```

```sql
CREATE TABLE IF NOT EXISTS llama_bench (
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
  tensor_buft_overrides TEXT,
  use_mmap INTEGER,
  embeddings INTEGER,
  no_op_offload INTEGER,
  n_prompt INTEGER,
  n_gen INTEGER,
  n_depth INTEGER,
  test_time TEXT,
  avg_ns INTEGER,
  stddev_ns INTEGER,
  avg_ts REAL,
  stddev_ts REAL,
  avg_ttft_ms REAL,
  stddev_ttft_ms REAL,
  avg_e2e_ms REAL,
  stddev_e2e_ms REAL,
  avg_itl_ms REAL,
  stddev_itl_ms REAL
);

INSERT INTO llama_bench (build_commit, build_number, cpu_info, gpu_info, backends, model_filename, model_type, model_size, model_n_params, n_batch, n_ubatch, n_threads, cpu_mask, cpu_strict, poll, type_k, type_v, n_gpu_layers, split_mode, main_gpu, no_kv_offload, flash_attn, tensor_split, tensor_buft_overrides, use_mmap, embeddings, no_op_offload, n_prompt, n_gen, n_depth, test_time, avg_ns, stddev_ns, avg_ts, stddev_ts, avg_ttft_ms, stddev_ttft_ms, avg_e2e_ms, stddev_e2e_ms, avg_itl_ms, stddev_itl_ms) VALUES ('69f7e7116', '6321', 'OpenBLAS, CPU', '', 'BLAS', 'models/granite-3.3-2b-instruct-be.IQ4_XS.gguf', 'granite 3B IQ4_XS - 4.25 bpw', '1395281392', '2533539840', '2048', '512', '8', '0x0', '0', '50', 'f16', 'f16', '99', 'layer', '0', '0', '0', '0.00', 'none', '1', '0', '0', '512', '0', '0', '2025-08-28T17:08:13Z', '6326649352', '9345621', '80.927655', '0.119504', '0.000000', '0.000000', '6326.649353', '9.344944', '0.000000', '0.000000');
INSERT INTO llama_bench (build_commit, build_number, cpu_info, gpu_info, backends, model_filename, model_type, model_size, model_n_params, n_batch, n_ubatch, n_threads, cpu_mask, cpu_strict, poll, type_k, type_v, n_gpu_layers, split_mode, main_gpu, no_kv_offload, flash_attn, tensor_split, tensor_buft_overrides, use_mmap, embeddings, no_op_offload, n_prompt, n_gen, n_depth, test_time, avg_ns, stddev_ns, avg_ts, stddev_ts, avg_ttft_ms, stddev_ttft_ms, avg_e2e_ms, stddev_e2e_ms, avg_itl_ms, stddev_itl_ms) VALUES ('69f7e7116', '6321', 'OpenBLAS, CPU', '', 'BLAS', 'models/granite-3.3-2b-instruct-be.IQ4_XS.gguf', 'granite 3B IQ4_XS - 4.25 bpw', '1395281392', '2533539840', '2048', '512', '8', '0x0', '0', '50', 'f16', 'f16', '99', 'layer', '0', '0', '0', '0.00', 'none', '1', '0', '0', '0', '128', '0', '2025-08-28T17:08:51Z', '5640474378', '40328612', '22.694063', '0.163702', '44.155657', '0.086653', '5640.474378', '40.328543', '44.065502', '0.317725');
```
