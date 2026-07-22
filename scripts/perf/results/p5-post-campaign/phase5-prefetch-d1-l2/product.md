# Product campaign: mistral-7b-instruct-v0.1.Q4_K_M.gguf

- bin-dir: /mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork-p5-post/build-campaign-jit/bin
- baseline label: stock
- candidate label: prefetch-d1-l2
- baseline env: {}
- candidate env: {'GGML_SYCL_Q8_PREFETCH_DISTANCE': '1'}
- candidate_enabled: True
- model shape: None
- candidate env log assertions: {'GGML_SYCL_Q8_PREFETCH_DISTANCE': {'requested_value': '1', 'backend_logs_key': False, 'candidate_samples': 24, 'candidate_samples_with_requested_value': 0, 'valid': True}}
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 968.89 | 968.30 | 3.42 | +/- 4.25 | 964.19 | 963.33 | 2.89 | +/- 3.59 | -0.39 | -0.51 | 0.57 | +/- 0.71 | n/a | n/a | n/a | 5 |
| 0 | q8_0/q8_0 | tg128 | Y | 24.57 | 24.57 | 0.04 | +/- 0.05 | 24.49 | 24.51 | 0.03 | +/- 0.04 | -0.18 | -0.24 | 0.20 | +/- 0.24 | n/a | n/a | n/a | 5 |
| 4096 | q8_0/q8_0 | pp512 | Y | 368.32 | 368.53 | 0.67 | +/- 0.83 | 368.51 | 368.59 | 0.39 | +/- 0.49 | +0.00 | +0.02 | 0.13 | +/- 0.16 | n/a | n/a | n/a | 5 |
| 4096 | q8_0/q8_0 | tg128 | Y | 17.94 | 17.94 | 0.02 | +/- 0.02 | 17.89 | 17.89 | 0.02 | +/- 0.03 | -0.20 | -0.28 | 0.14 | +/- 0.17 | n/a | n/a | n/a | 5 |
| 8192 | q8_0/q8_0 | pp512 | Y | 213.18 | 213.17 | 0.29 | +/- 0.36 | 213.13 | 213.11 | 0.32 | +/- 0.40 | -0.02 | -0.03 | 0.15 | +/- 0.19 | n/a | n/a | n/a | 5 |
| 8192 | q8_0/q8_0 | tg128 | Y | 14.16 | 14.16 | 0.01 | +/- 0.01 | 14.07 | 14.08 | 0.01 | +/- 0.01 | -0.57 | -0.61 | 0.07 | +/- 0.09 | n/a | n/a | n/a | 5 |
| 16384 | q8_0/q8_0 | pp512 | Y | 115.94 | 115.99 | 0.11 | +/- 0.13 | 116.05 | 116.01 | 0.10 | +/- 0.12 | -0.03 | +0.02 | 0.09 | +/- 0.11 | n/a | n/a | n/a | 5 |
| 16384 | q8_0/q8_0 | tg128 | Y | 9.94 | 9.94 | 0.01 | +/- 0.01 | 9.84 | 9.84 | 0.01 | +/- 0.01 | -1.01 | -1.03 | 0.09 | +/- 0.11 | n/a | n/a | n/a | 5 |
