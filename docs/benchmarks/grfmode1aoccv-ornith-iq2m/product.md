# Product campaign: Ornith-1.5-35B-A3B-IQ2_M.gguf

- bin-dir: /home/svnbjrn/build-grf-split/bin
- baseline label: grf-off
- candidate label: grf-mode1
- baseline env: {}
- candidate env: {'GGML_SYCL_FA_LARGE_GRF': '1'}
- candidate_enabled: True
- model shape: {'model_layers': 41, 'query_heads': 16, 'head_dim': 256}
- campaign valid: True
- invalid diagnostics: none
- candidate env log assertions: GGML_SYCL_FA_LARGE_GRF=1 (validated in 8/8 candidate samples; valid=True)
- dmesg fault hits before=1 after=1 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 264.39 | 261.68 | 6.45 | +/- 16.03 | 244.24 | 251.74 | 16.83 | +/- 41.82 | -3.96 | -3.81 | 5.50 | +/- 13.66 | 0 | 0.000 | 0.000 | 3 |
| 0 | q8_0/q8_0 | tg128 | Y | 53.06 | 52.80 | 0.56 | +/- 1.39 | 52.24 | 52.20 | 0.08 | +/- 0.19 | -1.55 | -1.13 | 0.91 | +/- 2.27 | 0 | 0.000 | 0.000 | 3 |
| 8192 | q8_0/q8_0 | pp512 | Y | 264.27 | 264.89 | 2.32 | +/- 5.77 | 258.74 | 259.68 | 3.22 | +/- 7.99 | -1.59 | -1.97 | 0.67 | +/- 1.65 | 2923429888 | 132.689 | 133.381 | 3 |
| 8192 | q8_0/q8_0 | tg128 | Y | 45.39 | 45.67 | 0.53 | +/- 1.32 | 45.62 | 45.73 | 0.52 | +/- 1.29 | +0.01 | +0.13 | 0.46 | +/- 1.15 | 2923429888 | 132.689 | 133.381 | 3 |
