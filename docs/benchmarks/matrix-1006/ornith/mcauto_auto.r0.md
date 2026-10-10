| model                          |       size |     params | backend    | ngl | moe-cache |   fit | type_k | type_v |  fa |       fitt |        fitc |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------- | ----: | -----: | -----: | --: | ---------: | ----------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | auto      |     1 |   q8_0 |   q8_0 |   1 |       1024 |       32768 |           pp512 |        273.44 ± 7.32 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | auto      |     1 |   q8_0 |   q8_0 |   1 |       1024 |       32768 |            tg64 |         14.00 ± 0.03 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | auto      |     1 |   q8_0 |   q8_0 |   1 |       1024 |       32768 |   pp512 @ d8192 |        228.70 ± 2.52 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | auto      |     1 |   q8_0 |   q8_0 |   1 |       1024 |       32768 |    tg64 @ d8192 |         13.38 ± 0.04 |

build: c4b9d6430 (12327)
