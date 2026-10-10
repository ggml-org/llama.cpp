| model                          |       size |     params | backend    | ngl | threads | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -----: | --: | --------------: | -------------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |       6 |   q8_0 |   q8_0 |   1 |           pp512 |        760.68 ± 1.68 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |       6 |   q8_0 |   q8_0 |   1 |            tg64 |         29.46 ± 0.05 |

build: c4b9d6430 (12327)
