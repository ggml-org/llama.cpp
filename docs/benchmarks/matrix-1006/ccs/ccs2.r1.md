| model                          |       size |     params | backend    | ngl | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | --: | --------------: | -------------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |           pp512 |       1163.08 ± 0.20 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |            tg64 |         50.72 ± 0.04 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |   pp512 @ d8192 |        206.19 ± 2.70 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |    tg64 @ d8192 |         32.67 ± 0.30 |

build: c4b9d6430 (12327)
