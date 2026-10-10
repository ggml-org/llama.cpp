| model                          |       size |     params | backend    | ngl | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | --: | --------------: | -------------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |           pp512 |       1633.27 ± 1.13 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |            tg64 |         57.93 ± 0.07 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |   pp512 @ d8192 |        219.20 ± 2.32 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |    tg64 @ d8192 |         42.81 ± 0.13 |

build: c4b9d6430 (12327)
