# Product campaign: mistral-7b-instruct-v0.1.Q4_K_M.gguf

- bin-dir: /mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork-p5-post/build-campaign-jit/bin
- baseline label: stock
- candidate label: prefetch-d2-l2
- baseline env: {}
- candidate env: {'GGML_SYCL_Q8_PREFETCH_DISTANCE': '2'}
- candidate_enabled: True
- model shape: None
- candidate env log assertions: {'GGML_SYCL_Q8_PREFETCH_DISTANCE': {'requested_value': '2', 'backend_logs_key': False, 'candidate_samples': 24, 'candidate_samples_with_requested_value': 0, 'valid': True}}
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 968.78 | 968.23 | 3.05 | +/- 3.79 | 970.59 | 970.41 | 2.65 | +/- 3.29 | +0.14 | +0.23 | 0.15 | +/- 0.19 | n/a | n/a | n/a | 5 |
| 0 | q8_0/q8_0 | tg128 | Y | 24.54 | 24.56 | 0.05 | +/- 0.06 | 24.55 | 24.53 | 0.05 | +/- 0.07 | -0.18 | -0.13 | 0.17 | +/- 0.21 | n/a | n/a | n/a | 5 |
| 4096 | q8_0/q8_0 | pp512 | Y | 368.75 | 368.73 | 0.14 | +/- 0.17 | 369.16 | 369.13 | 0.25 | +/- 0.32 | +0.10 | +0.11 | 0.05 | +/- 0.06 | n/a | n/a | n/a | 5 |
| 4096 | q8_0/q8_0 | tg128 | Y | 17.93 | 17.93 | 0.02 | +/- 0.03 | 17.87 | 17.86 | 0.02 | +/- 0.02 | -0.38 | -0.36 | 0.13 | +/- 0.16 | n/a | n/a | n/a | 5 |
| 8192 | q8_0/q8_0 | pp512 | Y | 213.26 | 213.28 | 0.28 | +/- 0.34 | 213.56 | 213.48 | 0.23 | +/- 0.28 | +0.10 | +0.09 | 0.16 | +/- 0.19 | n/a | n/a | n/a | 5 |
| 8192 | q8_0/q8_0 | tg128 | Y | 14.17 | 14.17 | 0.01 | +/- 0.01 | 14.07 | 14.07 | 0.02 | +/- 0.03 | -0.71 | -0.66 | 0.14 | +/- 0.17 | n/a | n/a | n/a | 5 |
| 16384 | q8_0/q8_0 | pp512 | Y | 115.98 | 116.00 | 0.08 | +/- 0.10 | 115.95 | 115.95 | 0.12 | +/- 0.14 | -0.09 | -0.05 | 0.15 | +/- 0.18 | n/a | n/a | n/a | 5 |
| 16384 | q8_0/q8_0 | tg128 | Y | 9.94 | 9.94 | 0.01 | +/- 0.01 | 9.85 | 9.84 | 0.01 | +/- 0.01 | -0.96 | -0.96 | 0.09 | +/- 0.12 | n/a | n/a | n/a | 5 |
