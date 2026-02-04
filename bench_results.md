ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/4: starting
llama-bench: benchmark 1/4: warmup prompt run
llama-bench: benchmark 1/4: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | n_ubatch | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -----: | -: | --------------: | -------------------: |
| qwen3vlmoe 30B.A3B Q4_1        |  17.87 GiB |    30.53 B | ROCm       |  99 |      12 |     2048 |   q8_0 |  1 |           pp512 |       1683.39 ± 0.00 |
llama-bench: benchmark 2/4: starting
llama-bench: benchmark 2/4: warmup generation run
llama-bench: benchmark 2/4: generation run 1/1
| qwen3vlmoe 30B.A3B Q4_1        |  17.87 GiB |    30.53 B | ROCm       |  99 |      12 |     2048 |   q8_0 |  1 |             tg1 |         18.50 ± 0.00 |
llama-bench: benchmark 3/4: starting
llama-bench: benchmark 3/4: warmup generation run
llama-bench: benchmark 3/4: generation run 1/1
| qwen3vlmoe 30B.A3B Q4_1        |  17.87 GiB |    30.53 B | ROCm       |  99 |      12 |     2048 |   q8_0 |  1 |           tg128 |        130.57 ± 0.00 |
llama-bench: benchmark 4/4: starting
llama-bench: benchmark 4/4: warmup generation run
llama-bench: benchmark 4/4: generation run 1/1
| qwen3vlmoe 30B.A3B Q4_1        |  17.87 GiB |    30.53 B | ROCm       |  99 |      12 |     2048 |   q8_0 |  1 |          tg2048 |        123.54 ± 0.00 |

build: 423bee462 (7937)
