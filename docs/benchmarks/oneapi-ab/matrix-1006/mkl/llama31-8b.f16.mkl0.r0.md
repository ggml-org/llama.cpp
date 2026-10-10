| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   1 |           pp512 |       1631.31 ± 1.10 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   1 |   pp512 @ d2048 |        912.00 ± 0.46 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   1 |   pp512 @ d4096 |        575.35 ± 0.03 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   1 |   pp512 @ d8192 |        344.09 ± 0.27 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   1 |  pp512 @ d16384 |        192.47 ± 1.13 |

build: c4b9d6430 (12327)
