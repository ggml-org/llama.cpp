# Arc A770 fork-unique benchmark summary

JSONL: `/mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork/docs/research/a770-fork-unique-2026-07-09/results.jsonl`

| model | case | status | pp tok/s | tg tok/s |
|---|---|---:|---:|---:|
| llama31-8b-heretic | upstream-f16-f16 | ok | 327.67 | 24.83 |
| llama31-8b-heretic | fork-f16-f16 | ok | 326.65 | 24.85 |
| llama31-8b-heretic | upstream-q8_0-q8_0 | ok | 318.51 | 23.45 |
| llama31-8b-heretic | fork-q8_0-q8_0 | ok | 324.31 | 25.29 |
| llama31-8b-heretic | fork-xmx-default-f16-f16 | ok | 157.97 | 6.72 |
| llama31-8b-heretic | fork-xmx-default-q8_0-q8_0 | ok | 222.95 | 10.90 |
| llama31-8b-heretic | fork-xmx-default-turbo3-turbo3 | ok | 156.49 | 6.64 |
| llama31-8b-heretic | fork-default-turbo2-turbo2 | ok | 308.63 | 23.75 |
| llama31-8b-heretic | fork-pure-turbo2-turbo2 | ok | 313.06 | 24.33 |
| llama31-8b-heretic | fork-xmx-pure-turbo2-turbo2 | ok | 157.20 | 6.66 |
| llama31-8b-heretic | fork-default-turbo3-turbo3 | ok | 312.00 | 23.92 |
| llama31-8b-heretic | fork-pure-turbo3-turbo3 | ok | 311.75 | 23.98 |
| llama31-8b-heretic | fork-xmx-pure-turbo3-turbo3 | ok | 156.70 | 6.64 |
| llama31-8b-heretic | fork-default-turbo4-turbo4 | ok | 315.55 | 24.35 |
| llama31-8b-heretic | fork-pure-turbo4-turbo4 | ok | 315.10 | 24.25 |
| llama31-8b-heretic | fork-xmx-pure-turbo4-turbo4 | ok | 151.68 | 6.66 |
| llama31-8b-heretic | fork-default-q8_0-turbo3 | ok | 317.56 | 24.44 |
| llama31-8b-heretic | fork-pure-q8_0-turbo3 | ok | 317.21 | 24.31 |
| llama31-8b-heretic | fork-xmx-pure-q8_0-turbo3 | ok | 317.23 | 24.35 |
| llama31-8b-heretic | fork-nonfa-turbo3-turbo3 | ok | 222.96 | 9.16 |
| mistral-7b | upstream-f16-f16 | ok | 330.53 | 25.73 |
| mistral-7b | fork-f16-f16 | ok | 329.77 | 25.73 |
| mistral-7b | upstream-q8_0-q8_0 | ok | 320.73 | 24.21 |
| mistral-7b | fork-q8_0-q8_0 | ok | 326.48 | 26.08 |
| mistral-7b | fork-xmx-default-f16-f16 | ok | 158.48 | 6.79 |
| mistral-7b | fork-xmx-default-q8_0-q8_0 | ok | 224.15 | 11.08 |
| mistral-7b | fork-xmx-default-turbo3-turbo3 | ok | 157.31 | 6.70 |
| mistral-7b | fork-default-turbo2-turbo2 | ok | 309.31 | 24.46 |
| mistral-7b | fork-pure-turbo2-turbo2 | ok | 315.28 | 25.18 |
| mistral-7b | fork-xmx-pure-turbo2-turbo2 | ok | 157.37 | 6.72 |
| mistral-7b | fork-default-turbo3-turbo3 | ok | 314.38 | 24.72 |
| mistral-7b | fork-pure-turbo3-turbo3 | ok | 314.45 | 24.83 |
| mistral-7b | fork-xmx-pure-turbo3-turbo3 | ok | 157.28 | 6.70 |
| mistral-7b | fork-default-turbo4-turbo4 | ok | 316.98 | 25.03 |
| mistral-7b | fork-pure-turbo4-turbo4 | ok | 317.01 | 25.14 |
| mistral-7b | fork-xmx-pure-turbo4-turbo4 | ok | 157.57 | 6.71 |
| mistral-7b | fork-default-q8_0-turbo3 | ok | 320.70 | 25.17 |
| mistral-7b | fork-pure-q8_0-turbo3 | ok | 320.63 | 25.30 |
| mistral-7b | fork-xmx-pure-q8_0-turbo3 | ok | 320.78 | 25.26 |
| mistral-7b | fork-nonfa-turbo3-turbo3 | ok | 224.60 | 9.27 |
| qwen3-coder-30b-a3b | upstream-f16-f16 | ok | 58.21 | 14.77 |
| qwen3-coder-30b-a3b | fork-f16-f16 | ok | 55.01 | 14.77 |
| qwen3-coder-30b-a3b | upstream-q8_0-q8_0 | ok | 57.41 | 13.89 |
| qwen3-coder-30b-a3b | fork-q8_0-q8_0 | ok | 54.77 | 14.82 |
| qwen3-coder-30b-a3b | fork-xmx-default-f16-f16 | ok | 43.92 | 4.32 |
| qwen3-coder-30b-a3b | fork-xmx-default-q8_0-q8_0 | ok | 50.35 | 6.96 |
| qwen3-coder-30b-a3b | fork-xmx-default-turbo3-turbo3 | ok | 56.08 | 14.39 |
| qwen3-coder-30b-a3b | fork-default-turbo2-turbo2 | ok | 55.08 | 14.56 |
| qwen3-coder-30b-a3b | fork-pure-turbo2-turbo2 | ok | 55.94 | 14.35 |
| qwen3-coder-30b-a3b | fork-xmx-pure-turbo2-turbo2 | ok | 44.54 | 4.31 |
| qwen3-coder-30b-a3b | fork-default-turbo3-turbo3 | ok | 56.12 | 14.40 |
| qwen3-coder-30b-a3b | fork-pure-turbo3-turbo3 | ok | 55.97 | 14.11 |
| qwen3-coder-30b-a3b | fork-xmx-pure-turbo3-turbo3 | ok | 44.07 | 4.29 |
| qwen3-coder-30b-a3b | fork-default-turbo4-turbo4 | ok | 56.66 | 14.54 |
| qwen3-coder-30b-a3b | fork-pure-turbo4-turbo4 | ok | 54.91 | 14.34 |
| qwen3-coder-30b-a3b | fork-xmx-pure-turbo4-turbo4 | ok | 43.62 | 4.29 |
| qwen3-coder-30b-a3b | fork-default-q8_0-turbo3 | ok | 55.87 | 14.42 |
| qwen3-coder-30b-a3b | fork-pure-q8_0-turbo3 | ok | 55.88 | 14.36 |
| qwen3-coder-30b-a3b | fork-xmx-pure-q8_0-turbo3 | ok | 55.85 | 14.37 |
| qwen3-coder-30b-a3b | fork-nonfa-turbo3-turbo3 | ok | 50.42 | 5.98 |
