| model                          |       size |     params | backend    | ngl | moe-cache |   fit | type_k | type_v |  fa |         lm |       fitt |        fitc |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------- | ----: | -----: | -----: | --: | ---------: | ---------: | ----------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | auto      |     1 |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |           pp512 |        283.39 ± 5.60 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | auto      |     1 |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |            tg64 |         12.07 ± 0.02 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | auto      |     1 |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |   pp512 @ d8192 |        239.05 ± 2.68 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | auto      |     1 |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |    tg64 @ d8192 |         11.34 ± 0.04 |

build: c4b9d6430 (12327)
