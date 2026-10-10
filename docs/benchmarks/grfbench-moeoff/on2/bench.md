| model                          |       size |     params | backend    | ngl | moe-cache | type_k | type_v |  fa |       fitt |        fitc |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------- | -----: | -----: | --: | ---------: | ----------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       1024 |       32768 |           pp128 |        161.85 ± 3.12 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       1024 |       32768 |           pp512 |        371.89 ± 9.85 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       1024 |       32768 |            tg64 |        40.06 ± 10.08 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       1024 |       32768 |   pp128 @ d8192 |        125.97 ± 4.38 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       1024 |       32768 |   pp512 @ d8192 |        297.73 ± 5.12 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 | off       |   q8_0 |   q8_0 |   1 |       1024 |       32768 |    tg64 @ d8192 |         39.48 ± 1.49 |

build: 5b1eb7c34 (12321)
