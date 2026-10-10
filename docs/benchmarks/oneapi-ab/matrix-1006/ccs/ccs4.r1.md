| model                          |       size |     params | backend    | ngl | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | --: | --------------: | -------------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |           pp512 |        623.55 ± 0.70 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |            tg64 |         30.21 ± 0.03 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |   pp512 @ d8192 |        179.37 ± 1.43 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |    tg64 @ d8192 |         18.57 ± 0.10 |

build: c4b9d6430 (12327)
