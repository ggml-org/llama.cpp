# Product campaign: mistral-7b-instruct-v0.1.Q4_K_M.gguf

- bin-dir: /mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork-p5-post/build-campaign-jit/bin
- baseline label: stock
- candidate label: prefetch-d1-l1
- baseline env: {}
- candidate env: {'GGML_SYCL_Q8_PREFETCH_DISTANCE': '1'}
- candidate_enabled: True
- model shape: None
- candidate env log assertions: {'GGML_SYCL_Q8_PREFETCH_DISTANCE': {'requested_value': '1', 'backend_logs_key': False, 'candidate_samples': 24, 'candidate_samples_with_requested_value': 0, 'valid': True}}
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 958.34 | 948.24 | 19.27 | +/- 23.93 | 954.32 | 954.22 | 11.23 | +/- 13.94 | +0.54 | +0.66 | 2.26 | +/- 2.81 | n/a | n/a | n/a | 5 |
| 0 | q8_0/q8_0 | tg128 | Y | 24.43 | 24.43 | 0.05 | +/- 0.07 | 24.47 | 24.44 | 0.07 | +/- 0.08 | +0.10 | +0.05 | 0.16 | +/- 0.20 | n/a | n/a | n/a | 5 |
| 4096 | q8_0/q8_0 | pp512 | Y | 368.37 | 368.23 | 0.75 | +/- 0.93 | 368.75 | 368.69 | 0.41 | +/- 0.51 | +0.06 | +0.12 | 0.23 | +/- 0.28 | n/a | n/a | n/a | 5 |
| 4096 | q8_0/q8_0 | tg128 | Y | 17.86 | 17.86 | 0.05 | +/- 0.07 | 17.76 | 17.76 | 0.03 | +/- 0.04 | -0.62 | -0.56 | 0.43 | +/- 0.54 | n/a | n/a | n/a | 5 |
| 8192 | q8_0/q8_0 | pp512 | Y | 213.21 | 213.22 | 0.25 | +/- 0.31 | 213.37 | 213.32 | 0.34 | +/- 0.42 | +0.14 | +0.04 | 0.23 | +/- 0.28 | n/a | n/a | n/a | 5 |
| 8192 | q8_0/q8_0 | tg128 | Y | 14.11 | 14.12 | 0.04 | +/- 0.05 | 14.01 | 13.99 | 0.04 | +/- 0.05 | -0.90 | -0.90 | 0.47 | +/- 0.59 | n/a | n/a | n/a | 5 |
| 16384 | q8_0/q8_0 | pp512 | Y | 116.01 | 115.94 | 0.14 | +/- 0.17 | 115.90 | 115.96 | 0.13 | +/- 0.16 | +0.04 | +0.02 | 0.18 | +/- 0.22 | n/a | n/a | n/a | 5 |
| 16384 | q8_0/q8_0 | tg128 | Y | 9.94 | 9.94 | 0.00 | +/- 0.01 | 9.84 | 9.84 | 0.00 | +/- 0.00 | -1.00 | -1.01 | 0.05 | +/- 0.06 | n/a | n/a | n/a | 5 |
