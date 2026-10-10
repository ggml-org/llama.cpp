| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   1 |           pp512 |       1630.75 ± 0.83 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   1 |   pp512 @ d2048 |        687.69 ± 1.28 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   1 |   pp512 @ d4096 |        483.96 ± 0.29 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   1 |   pp512 @ d8192 |        221.64 ± 0.43 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | SYCL       |  99 |   1 |  pp512 @ d16384 |        123.01 ± 0.15 |

build: c4b9d6430 (12327)
