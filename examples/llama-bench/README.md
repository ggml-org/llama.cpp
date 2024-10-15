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
  -p, --n-prompt <n>                        (default: 512)
  -n, --n-gen <n>                           (default: 128)
  -pg <pp,tg>                               (default: )
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
build_commit,build_number,cuda,metal,gpu_blas,blas,cpu_info,gpu_info,model_filename,model_type,model_size,model_n_params,n_batch,n_threads,f16_kv,n_gpu_layers,main_gpu,mul_mat_q,tensor_split,n_prompt,n_gen,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts
"3469684","1275","1","0","0","1","1","13th Gen Intel(R) Core(TM) i9-13900K","NVIDIA GeForce RTX 3090 Ti","models/7B/ggml-model-q4_0.gguf","llama 7B mostly Q4_0","3825065984","6738415616","512","16","1","99","0","1","0.00","512","0","2023-09-23T12:09:01Z","212155977","732372","2413.341687","8.305961"
"3469684","1275","1","0","0","1","1","13th Gen Intel(R) Core(TM) i9-13900K","NVIDIA GeForce RTX 3090 Ti","models/7B/ggml-model-q4_0.gguf","llama 7B mostly Q4_0","3825065984","6738415616","512","16","1","99","0","1","0.00","0","128","2023-09-23T12:09:02Z","969320879","2728399","132.052051","0.371342"
```

### JSON

```sh
$ ./llama-bench -o json
```

```json
[
  {
    "build_commit": "3469684",
    "build_number": 1275,
    "cuda": true,
    "metal": false,
    "gpu_blas": true,
    "blas": true,
    "cpu_info": "13th Gen Intel(R) Core(TM) i9-13900K",
    "gpu_info": "NVIDIA GeForce RTX 3090 Ti",
    "model_filename": "models/7B/ggml-model-q4_0.gguf",
    "model_type": "llama 7B mostly Q4_0",
    "model_size": 3825065984,
    "model_n_params": 6738415616,
    "n_batch": 512,
    "n_threads": 16,
    "f16_kv": true,
    "n_gpu_layers": 99,
    "main_gpu": 0,
    "mul_mat_q": true,
    "tensor_split": "0.00",
    "n_prompt": 512,
    "n_gen": 0,
    "test_time": "2023-09-23T12:09:57Z",
    "avg_ns": 212365953,
    "stddev_ns": 985423,
    "avg_ts": 2410.974041,
    "stddev_ts": 11.163766,
    "samples_ns": [ 213837238, 211635853, 212328053, 211329715, 212698907 ],
    "samples_ts": [ 2394.34, 2419.25, 2411.36, 2422.75, 2407.16 ]
  },
  {
    "build_commit": "3469684",
    "build_number": 1275,
    "cuda": true,
    "metal": false,
    "gpu_blas": true,
    "blas": true,
    "cpu_info": "13th Gen Intel(R) Core(TM) i9-13900K",
    "gpu_info": "NVIDIA GeForce RTX 3090 Ti",
    "model_filename": "models/7B/ggml-model-q4_0.gguf",
    "model_type": "llama 7B mostly Q4_0",
    "model_size": 3825065984,
    "model_n_params": 6738415616,
    "n_batch": 512,
    "n_threads": 16,
    "f16_kv": true,
    "n_gpu_layers": 99,
    "main_gpu": 0,
    "mul_mat_q": true,
    "tensor_split": "0.00",
    "n_prompt": 0,
    "n_gen": 128,
    "test_time": "2023-09-23T12:09:59Z",
    "avg_ns": 977425219,
    "stddev_ns": 9268593,
    "avg_ts": 130.965708,
    "stddev_ts": 1.238924,
    "samples_ns": [ 984472709, 974901233, 989474741, 970729355, 967548060 ],
    "samples_ts": [ 130.019, 131.295, 129.362, 131.86, 132.293 ]
  }
]
```


### JSONL

```sh
$ ./llama-bench -o jsonl
```

```json lines
{"build_commit":"3469684","build_number":1275,"cuda":true,"metal":false,"gpu_blas":true,"blas":true,"cpu_info":"13th Gen Intel(R) Core(TM) i9-13900K","gpu_info":"NVIDIA GeForce RTX 3090 Ti","model_filename":"models/7B/ggml-model-q4_0.gguf","model_type":"llama 7B mostly Q4_0","model_size":3825065984,"model_n_params":6738415616,"n_batch":512,"n_threads":16,"f16_kv":true,"n_gpu_layers":99,"main_gpu":0,"mul_mat_q":true,"tensor_split":"0.00","n_prompt":512,"n_gen":0,"test_time":"2023-09-23T12:09:57Z","avg_ns":212365953,"stddev_ns":985423,"avg_ts":2410.974041,"stddev_ts":11.163766,"samples_ns":[213837238,211635853,212328053,211329715,212698907],"samples_ts":[2394.34,2419.25,2411.36,2422.75,2407.16]}
{"build_commit":"3469684","build_number":1275,"cuda":true,"metal":false,"gpu_blas":true,"blas":true,"cpu_info":"13th Gen Intel(R) Core(TM) i9-13900K","gpu_info":"NVIDIA GeForce RTX 3090 Ti","model_filename":"models/7B/ggml-model-q4_0.gguf","model_type":"llama 7B mostly Q4_0","model_size":3825065984,"model_n_params":6738415616,"n_batch":512,"n_threads":16,"f16_kv":true,"n_gpu_layers":99,"main_gpu":0,"mul_mat_q":true,"tensor_split":"0.00","n_prompt":0,"n_gen":128,"test_time":"2023-09-23T12:09:59Z","avg_ns":977425219,"stddev_ns":9268593,"avg_ts":130.965708,"stddev_ts":1.238924,"samples_ns":[984472709,974901233,989474741,970729355,967548060],"samples_ts":[130.019,131.295,129.362,131.86,132.293]}
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
  cuda INTEGER,
  metal INTEGER,
  gpu_blas INTEGER,
  blas INTEGER,
  cpu_info TEXT,
  gpu_info TEXT,
  model_filename TEXT,
  model_type TEXT,
  model_size INTEGER,
  model_n_params INTEGER,
  n_batch INTEGER,
  n_threads INTEGER,
  f16_kv INTEGER,
  n_gpu_layers INTEGER,
  main_gpu INTEGER,
  mul_mat_q INTEGER,
  tensor_split TEXT,
  n_prompt INTEGER,
  n_gen INTEGER,
  test_time TEXT,
  avg_ns INTEGER,
  stddev_ns INTEGER,
  avg_ts REAL,
  stddev_ts REAL
);

INSERT INTO test (build_commit, build_number, cuda, metal, gpu_blas, blas, cpu_info, gpu_info, model_filename, model_type, model_size, model_n_params, n_batch, n_threads, f16_kv, n_gpu_layers, main_gpu, mul_mat_q, tensor_split, n_prompt, n_gen, test_time, avg_ns, stddev_ns, avg_ts, stddev_ts) VALUES ('3469684', '1275', '1', '0', '0', '1', '1', '13th Gen Intel(R) Core(TM) i9-13900K', 'NVIDIA GeForce RTX 3090 Ti', 'models/7B/ggml-model-q4_0.gguf', 'llama 7B mostly Q4_0', '3825065984', '6738415616', '512', '16', '1', '99', '0', '1', '0.00', '512', '0', '2023-09-23T12:10:30Z', '212693772', '743623', '2407.240204', '8.409634');
INSERT INTO test (build_commit, build_number, cuda, metal, gpu_blas, blas, cpu_info, gpu_info, model_filename, model_type, model_size, model_n_params, n_batch, n_threads, f16_kv, n_gpu_layers, main_gpu, mul_mat_q, tensor_split, n_prompt, n_gen, test_time, avg_ns, stddev_ns, avg_ts, stddev_ts) VALUES ('3469684', '1275', '1', '0', '0', '1', '1', '13th Gen Intel(R) Core(TM) i9-13900K', 'NVIDIA GeForce RTX 3090 Ti', 'models/7B/ggml-model-q4_0.gguf', 'llama 7B mostly Q4_0', '3825065984', '6738415616', '512', '16', '1', '99', '0', '1', '0.00', '0', '128', '2023-09-23T12:10:31Z', '977925003', '4037361', '130.891159', '0.537692');
```
