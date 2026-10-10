| model                          |       size |     params | backend    | ngl | threads | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -----: | --: | --------------: | -------------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |       6 |   q8_0 |   q8_0 |   1 |           pp512 |        532.99 ± 0.95 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |       6 |   q8_0 |   q8_0 |   1 |            tg64 |         24.09 ± 0.03 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |       6 |   q8_0 |   q8_0 |   1 |   pp512 @ d8192 |        102.79 ± 0.29 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |       6 |   q8_0 |   q8_0 |   1 |    tg64 @ d8192 |         16.02 ± 0.16 |

build: c4b9d6430 (12327)
