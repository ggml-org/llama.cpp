| model                          |       size |     params | backend    | ngl | moe-cache |  fa |         lm |       fitt |        fitc |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------- | --: | ---------: | ---------: | ----------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   1 |       none |       1024 |       32768 |           pp512 |        376.20 ± 9.69 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   1 |       none |       1024 |       32768 |   pp512 @ d2048 |        357.61 ± 4.52 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   1 |       none |       1024 |       32768 |   pp512 @ d4096 |        349.59 ± 1.23 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   1 |       none |       1024 |       32768 |   pp512 @ d8192 |        310.02 ± 1.04 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   1 |       none |       1024 |       32768 |  pp512 @ d16384 |        270.52 ± 1.00 |

build: c4b9d6430 (12327)
