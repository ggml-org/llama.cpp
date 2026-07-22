# Product campaign: mistral-7b-instruct-v0.1.Q4_K_M.gguf

- bin-dir: /mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork-p5-post/build-campaign-jit/bin
- baseline label: stock
- candidate label: prefetch-d2-l1
- baseline env: {}
- candidate env: {'GGML_SYCL_Q8_PREFETCH_DISTANCE': '2'}
- candidate_enabled: True
- model shape: None
- candidate env log assertions: {'GGML_SYCL_Q8_PREFETCH_DISTANCE': {'requested_value': '2', 'backend_logs_key': False, 'candidate_samples': 24, 'candidate_samples_with_requested_value': 0, 'valid': True}}
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 966.93 | 965.64 | 3.14 | +/- 3.90 | 960.53 | 957.54 | 10.63 | +/- 13.19 | -0.39 | -0.84 | 0.97 | +/- 1.21 | n/a | n/a | n/a | 5 |
| 0 | q8_0/q8_0 | tg128 | Y | 24.60 | 24.59 | 0.03 | +/- 0.04 | 24.54 | 24.54 | 0.04 | +/- 0.04 | -0.13 | -0.22 | 0.20 | +/- 0.25 | n/a | n/a | n/a | 5 |
| 4096 | q8_0/q8_0 | pp512 | Y | 368.69 | 368.75 | 0.15 | +/- 0.18 | 368.87 | 368.81 | 0.18 | +/- 0.23 | +0.00 | +0.02 | 0.03 | +/- 0.04 | n/a | n/a | n/a | 5 |
| 4096 | q8_0/q8_0 | tg128 | Y | 17.93 | 17.93 | 0.02 | +/- 0.02 | 17.87 | 17.87 | 0.00 | +/- 0.00 | -0.34 | -0.32 | 0.09 | +/- 0.11 | n/a | n/a | n/a | 5 |
| 8192 | q8_0/q8_0 | pp512 | Y | 213.54 | 213.55 | 0.19 | +/- 0.24 | 213.64 | 213.64 | 0.23 | +/- 0.29 | +0.07 | +0.04 | 0.09 | +/- 0.12 | n/a | n/a | n/a | 5 |
| 8192 | q8_0/q8_0 | tg128 | Y | 14.19 | 14.18 | 0.01 | +/- 0.02 | 14.09 | 14.09 | 0.01 | +/- 0.01 | -0.64 | -0.63 | 0.11 | +/- 0.13 | n/a | n/a | n/a | 5 |
| 16384 | q8_0/q8_0 | pp512 | Y | 116.00 | 116.00 | 0.08 | +/- 0.10 | 115.97 | 115.96 | 0.12 | +/- 0.15 | -0.07 | -0.03 | 0.09 | +/- 0.11 | n/a | n/a | n/a | 5 |
| 16384 | q8_0/q8_0 | tg128 | Y | 9.95 | 9.95 | 0.01 | +/- 0.01 | 9.86 | 9.85 | 0.00 | +/- 0.00 | -0.95 | -0.95 | 0.04 | +/- 0.05 | n/a | n/a | n/a | 5 |
