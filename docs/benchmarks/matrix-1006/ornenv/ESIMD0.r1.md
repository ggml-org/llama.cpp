| model                          |       size |     params | backend    | ngl | moe-cache | type_k | type_v |  fa |         lm |       fitt |        fitc |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------- | -----: | -----: | --: | ---------: | ---------: | ----------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |           pp512 |        377.34 ± 9.28 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |            tg64 |         46.22 ± 0.41 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |   pp512 @ d8192 |        310.32 ± 2.93 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |    tg64 @ d8192 |         40.10 ± 0.29 |

build: c4b9d6430 (12327)
