| model                          |       size |     params | backend    | ngl | moe-cache |   fit | rpk    | type_k | type_v |  fa |         lm |       fitt |        fitc |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------- | ----: | ------ | -----: | -----: | --: | ---------: | ---------: | ----------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | on        |     1 | off    |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |           pp512 |        283.09 ± 5.16 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | on        |     1 | off    |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |            tg64 |         12.02 ± 0.01 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | on        |     1 | off    |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |   pp512 @ d8192 |        240.24 ± 2.26 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | on        |     1 | off    |   q8_0 |   q8_0 |   1 |       none |       1024 |       32768 |    tg64 @ d8192 |         11.30 ± 0.04 |

build: c4b9d6430 (12327)
