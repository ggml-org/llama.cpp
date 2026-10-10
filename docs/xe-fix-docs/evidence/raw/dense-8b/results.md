# Dense 8B A/B, i915 vs xe (2026-09-29)

Model: `/mnt/ssd1/models/Meta-Llama-3.1-8B-Instruct-heretic.Q4_K_M.gguf` (4.58 GiB, all
layers on device, no expert streaming). Same binary (`llama.cpp-sycl-f16-git b12305`,
build `4e7400c3a`), same command on both drivers:

```
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_ENABLE_GRAPH=1 \
llama-bench -m /mnt/ssd1/models/Meta-Llama-3.1-8B-Instruct-heretic.Q4_K_M.gguf \
  -p 512,2048 -n 128 -r 2 -fa 1 -ngl 99 -o md
```

| test | i915 (10:37, Ornith unit stopped) | xe (11:09, Ornith unit running, idle) | xe (11:46, Ornith unit stopped) |
|---|--:|--:|--:|
| pp512 | 1408.08 +/- 2.41 | 1494.10 +/- 2.23 | 1495.56 +/- 0.49 |
| pp2048 | 848.93 +/- 0.57 | 988.93 +/- 0.23 | 990.40 +/- 1.28 |
| tg128 | 60.15 +/- 0.07 | 60.53 +/- 0.14 | 60.58 +/- 0.01 |

Clean xe vs i915: pp512 +6.2 %, pp2048 +16.7 %, tg128 +0.7 % (noise). An idle
resident server costs nothing measurable.

Files: `bench-xe-contended.err` is the stderr of the 11:09 xe run. The i915 run's
stderr was written to `/tmp` before the 11:00 reboot and did not survive it; the
table values above are from the session transcript (markdown output of
`llama-bench`, `-o md`). The 11:46 run's stderr was not kept.
