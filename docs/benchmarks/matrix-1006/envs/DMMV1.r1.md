| model                          |       size |     params | backend    | ngl | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | --: | --------------: | -------------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |           pp512 |       1631.56 ± 3.20 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |            tg64 |         61.83 ± 0.14 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |   pp512 @ d8192 |        218.64 ± 1.37 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |    tg64 @ d8192 |         45.09 ± 0.39 |

build: c4b9d6430 (12327)
