| model                          |       size |     params | backend    | ngl | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | --: | --------------: | -------------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |           pp512 |       1623.99 ± 1.41 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |            tg64 |         35.58 ± 0.03 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |   pp512 @ d8192 |        219.87 ± 0.99 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |    tg64 @ d8192 |         29.21 ± 0.23 |

build: c4b9d6430 (12327)
