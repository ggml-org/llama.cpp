| model                          |       size |     params | backend    | ngl | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |           pp512 |       1581.42 ± 3.13 |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |   pp512 @ d2048 |       1186.84 ± 0.25 |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |   pp512 @ d4096 |        945.89 ± 1.44 |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |   pp512 @ d8192 |        674.51 ± 0.81 |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |  pp512 @ d16384 |        430.84 ± 0.31 |

build: c4b9d6430 (12327)
