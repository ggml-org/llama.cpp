| model                          |       size |     params | backend    | ngl | moe-cache | type_k | type_v |  fa |         lm |       fitt |        fitc |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------- | -----: | -----: | --: | ---------: | ---------: | ----------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |           pp512 |        376.98 ± 7.92 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |            tg64 |         48.89 ± 0.72 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |   pp512 @ d8192 |        309.76 ± 3.96 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |    tg64 @ d8192 |         42.40 ± 0.19 |

build: c4b9d6430 (12327)
