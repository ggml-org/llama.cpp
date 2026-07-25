# Product campaign: Meta-Llama-3.1-8B-Instruct-heretic.Q4_K_M.gguf

- bin-dir: /home/svnbjrn/build-sync-old/bin
- baseline label: pre-sync
- candidate label: post-sync
- baseline env: {'GGML_SYCL_DEBUG': '0'}
- candidate env: {'GGML_SYCL_DEBUG': '0'}
- candidate_enabled: True
- model shape: {'model_layers': 32, 'query_heads': 32, 'head_dim': 128}
- candidate env log assertions: {'GGML_SYCL_DEBUG': {'requested_value': '0', 'backend_logs_key': False, 'candidate_samples': 24, 'candidate_samples_with_requested_value': 0, 'valid': True}}
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | f16/f16 | pp512 | Y | 903.39 | 900.36 | 9.28 | +/- 11.52 | 900.76 | 899.59 | 7.08 | +/- 8.79 | -0.21 | -0.08 | 1.17 | +/- 1.46 | 0 | 0.000 | 0.000 | 5 |
| 0 | f16/f16 | tg128 | Y | 23.18 | 23.18 | 0.02 | +/- 0.03 | 23.25 | 23.25 | 0.02 | +/- 0.03 | +0.29 | +0.29 | 0.11 | +/- 0.14 | 0 | 0.000 | 0.000 | 5 |
| 0 | q8_0/q8_0 | pp512 | Y | 899.93 | 892.31 | 13.98 | +/- 17.36 | 899.23 | 899.50 | 1.35 | +/- 1.67 | +0.16 | +0.83 | 1.59 | +/- 1.98 | 0 | 0.000 | 0.000 | 5 |
| 0 | q8_0/q8_0 | tg128 | Y | 23.58 | 23.56 | 0.03 | +/- 0.04 | 23.54 | 23.54 | 0.03 | +/- 0.04 | -0.15 | -0.08 | 0.13 | +/- 0.16 | 0 | 0.000 | 0.000 | 5 |
| 4096 | f16/f16 | pp512 | Y | 369.54 | 369.64 | 0.26 | +/- 0.32 | 369.86 | 369.81 | 0.11 | +/- 0.13 | +0.06 | +0.05 | 0.08 | +/- 0.10 | 2147483648 | 42.646 | 42.717 | 5 |
| 4096 | f16/f16 | tg128 | Y | 19.86 | 19.86 | 0.05 | +/- 0.06 | 19.89 | 19.89 | 0.03 | +/- 0.04 | +0.10 | +0.17 | 0.18 | +/- 0.22 | 2147483648 | 42.646 | 42.717 | 5 |
| 4096 | q8_0/q8_0 | pp512 | Y | 368.34 | 368.41 | 0.27 | +/- 0.34 | 368.45 | 368.38 | 0.19 | +/- 0.24 | -0.01 | -0.01 | 0.05 | +/- 0.07 | 1140850688 | 19.749 | 19.742 | 5 |
| 4096 | q8_0/q8_0 | tg128 | Y | 17.31 | 17.32 | 0.02 | +/- 0.02 | 17.30 | 17.31 | 0.02 | +/- 0.02 | -0.05 | -0.06 | 0.15 | +/- 0.18 | 1140850688 | 19.749 | 19.742 | 5 |
