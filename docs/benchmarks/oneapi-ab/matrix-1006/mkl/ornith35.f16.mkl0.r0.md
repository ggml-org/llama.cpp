| model                          |       size |     params | backend    | ngl | moe-cache |  fa |         lm |       fitt |        fitc |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------- | --: | ---------: | ---------: | ----------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   1 |       none |       1024 |       32768 |           pp512 |       377.55 ± 10.98 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   1 |       none |       1024 |       32768 |   pp512 @ d2048 |        320.97 ± 0.22 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   1 |       none |       1024 |       32768 |   pp512 @ d4096 |        286.12 ± 0.13 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   1 |       none |       1024 |       32768 |   pp512 @ d8192 |        237.44 ± 0.14 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   1 |       none |       1024 |       32768 |  pp512 @ d16384 |        179.27 ± 0.23 |

build: c4b9d6430 (12327)
