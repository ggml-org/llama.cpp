| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   1 |           pp512 |       1585.34 ± 0.18 |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   1 |   pp512 @ d2048 |       1193.48 ± 0.83 |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   1 |   pp512 @ d4096 |        948.44 ± 0.43 |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   1 |   pp512 @ d8192 |        676.57 ± 0.52 |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   1 |  pp512 @ d16384 |        432.00 ± 0.44 |

build: c4b9d6430 (12327)
