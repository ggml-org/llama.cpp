| model                          |       size |     params | backend    | ngl | moe-cache |   fit | type_k | type_v |  fa |       fitt |        fitc |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------- | ----: | -----: | -----: | --: | ---------: | ----------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | auto      |     1 |   q8_0 |   q8_0 |   1 |       1024 |       32768 |           pp512 |        273.67 ± 7.38 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | auto      |     1 |   q8_0 |   q8_0 |   1 |       1024 |       32768 |            tg64 |         13.71 ± 0.01 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | auto      |     1 |   q8_0 |   q8_0 |   1 |       1024 |       32768 |   pp512 @ d8192 |        230.24 ± 1.68 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | auto      |     1 |   q8_0 |   q8_0 |   1 |       1024 |       32768 |    tg64 @ d8192 |         12.84 ± 0.04 |

build: c4b9d6430 (12327)
