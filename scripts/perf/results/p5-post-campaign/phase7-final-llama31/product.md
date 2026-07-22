# Product campaign: Meta-Llama-3.1-8B-Instruct-heretic.Q4_K_M.gguf

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
| 0 | q8_0/q8_0 | pp512 | Y | 892.70 | 889.15 | 9.51 | +/- 11.81 | 894.50 | 893.19 | 3.09 | +/- 3.83 | +0.09 | +0.46 | 0.94 | +/- 1.16 | 0 | 0.000 | 0.000 | 5 |
| 0 | q8_0/q8_0 | tg128 | Y | 23.60 | 23.59 | 0.07 | +/- 0.09 | 23.56 | 23.58 | 0.04 | +/- 0.05 | +0.01 | -0.04 | 0.22 | +/- 0.27 | 0 | 0.000 | 0.000 | 5 |
| 4096 | q8_0/q8_0 | pp512 | Y | 368.43 | 368.52 | 0.30 | +/- 0.37 | 368.30 | 368.47 | 0.30 | +/- 0.37 | +0.00 | -0.01 | 0.13 | +/- 0.16 | 1140850688 | 19.819 | 19.808 | 5 |
| 4096 | q8_0/q8_0 | tg128 | Y | 17.37 | 17.37 | 0.02 | +/- 0.02 | 17.36 | 17.35 | 0.02 | +/- 0.02 | -0.08 | -0.11 | 0.17 | +/- 0.21 | 1140850688 | 19.819 | 19.808 | 5 |
| 8192 | q8_0/q8_0 | pp512 | Y | 213.29 | 213.30 | 0.10 | +/- 0.12 | 213.21 | 213.22 | 0.12 | +/- 0.15 | -0.03 | -0.03 | 0.05 | +/- 0.06 | 2281701376 | 31.510 | 31.518 | 5 |
| 8192 | q8_0/q8_0 | tg128 | Y | 13.81 | 13.81 | 0.01 | +/- 0.02 | 13.81 | 13.82 | 0.01 | +/- 0.02 | +0.11 | +0.06 | 0.10 | +/- 0.13 | 2281701376 | 31.510 | 31.518 | 5 |
| 16384 | q8_0/q8_0 | pp512 | Y | 115.72 | 115.73 | 0.09 | +/- 0.11 | 115.68 | 115.68 | 0.05 | +/- 0.07 | -0.08 | -0.04 | 0.08 | +/- 0.10 | 4563402752 | 44.570 | 44.538 | 5 |
| 16384 | q8_0/q8_0 | tg128 | Y | 9.77 | 9.77 | 0.00 | +/- 0.01 | 9.76 | 9.76 | 0.01 | +/- 0.01 | -0.09 | -0.04 | 0.10 | +/- 0.12 | 4563402752 | 44.570 | 44.538 | 5 |
