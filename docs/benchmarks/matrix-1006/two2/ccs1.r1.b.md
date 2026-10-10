| model                          |       size |     params | backend    | ngl | threads | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -----: | --: | --------------: | -------------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |       6 |   q8_0 |   q8_0 |   1 |           pp512 |        761.41 ± 1.45 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |       6 |   q8_0 |   q8_0 |   1 |            tg64 |         29.48 ± 0.04 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |       6 |   q8_0 |   q8_0 |   1 |   pp512 @ d8192 |        108.28 ± 0.30 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |       6 |   q8_0 |   q8_0 |   1 |    tg64 @ d8192 |         22.10 ± 0.12 |

build: c4b9d6430 (12327)
