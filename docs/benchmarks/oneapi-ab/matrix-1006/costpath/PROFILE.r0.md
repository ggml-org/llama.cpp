| model                          |       size |     params | backend    | ngl | moe-cache | type_k | type_v |  fa |         lm |       fitt |        fitc |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------- | -----: | -----: | --: | ---------: | ---------: | ----------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |             pp1 |         45.29 ± 2.39 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |             pp2 |         33.03 ± 0.46 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |             pp3 |         40.16 ± 0.30 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |             pp4 |         46.60 ± 0.87 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |             pp6 |         58.12 ± 0.38 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |             pp8 |         65.88 ± 0.36 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |            pp12 |         65.20 ± 1.66 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |            pp17 |         77.70 ± 0.94 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |            pp32 |         77.57 ± 1.40 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |            pp64 |        114.87 ± 0.87 |

build: c4b9d6430 (12327)
