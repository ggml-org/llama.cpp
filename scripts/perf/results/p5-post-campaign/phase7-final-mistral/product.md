# Product campaign: mistral-7b-instruct-v0.1.Q4_K_M.gguf

- bin-dir: /mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork-p5-post/build-campaign-jit/bin
- baseline label: p5-held
- candidate label: final-candidate
- baseline env: {'GGML_SYCL_FA_FORCE_VEC_STANDARD': '0'}
- candidate env: {'GGML_SYCL_FA_FORCE_VEC_STANDARD': '0'}
- candidate_enabled: True
- model shape: {'model_layers': 32, 'query_heads': 32, 'head_dim': 128}
- candidate env log assertions: {'GGML_SYCL_FA_FORCE_VEC_STANDARD': {'requested_value': '0', 'backend_logs_key': False, 'candidate_samples': 24, 'candidate_samples_with_requested_value': 0, 'valid': True}}
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 962.43 | 962.80 | 9.39 | +/- 11.65 | 968.59 | 966.64 | 4.83 | +/- 6.00 | -0.32 | +0.41 | 1.14 | +/- 1.41 | 0 | 0.000 | 0.000 | 5 |
| 0 | q8_0/q8_0 | tg128 | Y | 24.60 | 24.60 | 0.01 | +/- 0.01 | 24.60 | 24.58 | 0.03 | +/- 0.03 | +0.02 | -0.05 | 0.14 | +/- 0.17 | 0 | 0.000 | 0.000 | 5 |
| 4096 | q8_0/q8_0 | pp512 | Y | 368.87 | 368.92 | 0.23 | +/- 0.29 | 369.17 | 369.05 | 0.34 | +/- 0.42 | +0.07 | +0.04 | 0.11 | +/- 0.14 | 1140850688 | 20.458 | 20.445 | 5 |
| 4096 | q8_0/q8_0 | tg128 | Y | 17.93 | 17.93 | 0.02 | +/- 0.02 | 17.92 | 17.92 | 0.02 | +/- 0.03 | -0.06 | -0.07 | 0.11 | +/- 0.14 | 1140850688 | 20.458 | 20.445 | 5 |
| 8192 | q8_0/q8_0 | pp512 | Y | 213.60 | 213.50 | 0.21 | +/- 0.26 | 213.54 | 213.50 | 0.08 | +/- 0.10 | -0.02 | +0.00 | 0.10 | +/- 0.12 | 2281701376 | 32.215 | 32.260 | 5 |
| 8192 | q8_0/q8_0 | tg128 | Y | 14.12 | 14.12 | 0.03 | +/- 0.04 | 14.14 | 14.14 | 0.01 | +/- 0.02 | +0.03 | +0.15 | 0.29 | +/- 0.36 | 2281701376 | 32.215 | 32.260 | 5 |
| 16384 | q8_0/q8_0 | pp512 | Y | 115.98 | 115.96 | 0.09 | +/- 0.12 | 115.76 | 115.81 | 0.13 | +/- 0.17 | -0.13 | -0.13 | 0.17 | +/- 0.22 | 4563402752 | 45.319 | 45.328 | 5 |
| 16384 | q8_0/q8_0 | tg128 | Y | 9.93 | 9.93 | 0.00 | +/- 0.00 | 9.93 | 9.93 | 0.01 | +/- 0.02 | +0.01 | -0.04 | 0.14 | +/- 0.17 | 4563402752 | 45.319 | 45.328 | 5 |
