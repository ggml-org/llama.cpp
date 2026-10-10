| model                          |       size |     params | backend    | ngl | threads | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -----: | --: | --------------: | -------------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |       6 |   q8_0 |   q8_0 |   1 |           pp512 |        576.54 ± 0.85 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |       6 |   q8_0 |   q8_0 |   1 |            tg64 |         25.23 ± 0.14 |

build: c4b9d6430 (12327)
