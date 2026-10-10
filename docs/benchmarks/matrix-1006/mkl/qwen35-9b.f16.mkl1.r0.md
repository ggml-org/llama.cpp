| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   1 |           pp512 |       1586.88 ± 1.32 |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   1 |   pp512 @ d2048 |       1398.34 ± 1.19 |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   1 |   pp512 @ d4096 |       1262.15 ± 1.70 |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   1 |   pp512 @ d8192 |       868.68 ± 22.09 |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   1 |  pp512 @ d16384 |        622.65 ± 0.28 |

build: c4b9d6430 (12327)
