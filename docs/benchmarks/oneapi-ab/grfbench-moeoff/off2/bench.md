| model                          |       size |     params | backend    | ngl | moe-cache | type_k | type_v |  fa |       fitt |        fitc |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------- | -----: | -----: | --: | ---------: | ----------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       1024 |       32768 |           pp128 |        164.68 ± 7.92 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       1024 |       32768 |           pp512 |       347.61 ± 17.26 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       1024 |       32768 |            tg64 |         44.48 ± 3.23 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       1024 |       32768 |   pp128 @ d8192 |        125.64 ± 6.76 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       1024 |       32768 |   pp512 @ d8192 |        293.43 ± 8.34 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       1024 |       32768 |    tg64 @ d8192 |         39.55 ± 1.74 |

build: 5b1eb7c34 (12321)
