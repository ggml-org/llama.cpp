| model                          |       size |     params | backend    | ngl | moe-cache |   fit | type_k | type_v |  fa |       fitt |        fitc |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------- | ----: | -----: | -----: | --: | ---------: | ----------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | soft      |     1 |   q8_0 |   q8_0 |   1 |       1024 |       32768 |           pp512 |        272.89 ± 7.52 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | soft      |     1 |   q8_0 |   q8_0 |   1 |       1024 |       32768 |            tg64 |         22.72 ± 0.06 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | soft      |     1 |   q8_0 |   q8_0 |   1 |       1024 |       32768 |   pp512 @ d8192 |        226.27 ± 3.59 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | soft      |     1 |   q8_0 |   q8_0 |   1 |       1024 |       32768 |    tg64 @ d8192 |         20.08 ± 0.19 |

build: 4e7400c3a (12305)
