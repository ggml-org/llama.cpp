| model                          |       size |     params | backend    | ngl | moe-cache |   fit | type_k | type_v |  fa |         lm |       fitt |        fitc |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------- | ----: | -----: | -----: | --: | ---------: | ---------: | ----------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | soft      |     1 |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |           pp512 |        347.00 ± 6.66 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | soft      |     1 |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |            tg64 |         19.55 ± 0.07 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | soft      |     1 |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |   pp512 @ d8192 |        287.19 ± 3.23 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | soft      |     1 |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |    tg64 @ d8192 |         18.26 ± 0.05 |

build: c4b9d6430 (12327)
