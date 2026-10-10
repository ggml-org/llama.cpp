| model                          |       size |     params | backend    | ngl | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |           pp512 |       1580.32 ± 1.40 |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |   pp512 @ d2048 |       1397.57 ± 2.64 |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |   pp512 @ d4096 |       1262.90 ± 1.76 |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |   pp512 @ d8192 |       870.98 ± 14.18 |
| qwen35 9B Q4_K - Medium        |   5.23 GiB |     8.95 B | SYCL       |  99 |   q8_0 |   q8_0 |   1 |  pp512 @ d16384 |        625.22 ± 1.55 |

build: c4b9d6430 (12327)
