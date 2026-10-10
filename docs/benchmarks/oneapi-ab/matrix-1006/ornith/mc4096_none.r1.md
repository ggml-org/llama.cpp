| model                          |       size |     params | backend    | ngl | moe-cache |   fit | rpk    | type_k | type_v |  fa |         lm |       fitt |        fitc |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------- | ----: | ------ | -----: | -----: | --: | ---------: | ---------: | ----------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | 4096      |     1 | off    |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |           pp512 |        284.11 ± 5.73 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | 4096      |     1 | off    |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |            tg64 |         11.99 ± 0.04 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | 4096      |     1 | off    |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |   pp512 @ d8192 |        239.71 ± 2.18 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | 4096      |     1 | off    |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |    tg64 @ d8192 |         11.36 ± 0.01 |

build: c4b9d6430 (12327)
