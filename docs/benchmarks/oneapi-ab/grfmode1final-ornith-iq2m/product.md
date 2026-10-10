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
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 259.83 | 249.48 | 18.04 | +/- 44.82 | 259.75 | 256.56 | 12.82 | +/- 31.86 | +2.95 | +2.97 | 3.06 | +/- 7.60 | 0 | 0.000 | 0.000 | 3 |
| 0 | q8_0/q8_0 | tg128 | Y | 51.90 | 52.19 | 0.73 | +/- 1.81 | 52.99 | 52.77 | 0.39 | +/- 0.96 | +2.09 | +1.13 | 2.14 | +/- 5.31 | 0 | 0.000 | 0.000 | 3 |
| 8192 | q8_0/q8_0 | pp512 | Y | 258.45 | 259.57 | 3.90 | +/- 9.68 | 258.44 | 260.30 | 3.61 | +/- 8.96 | +0.21 | +0.28 | 0.33 | +/- 0.82 | 2923429888 | 134.910 | 132.064 | 3 |
| 8192 | q8_0/q8_0 | tg128 | Y | 46.15 | 45.85 | 0.57 | +/- 1.42 | 45.17 | 45.32 | 0.33 | +/- 0.81 | -1.15 | -1.17 | 0.93 | +/- 2.31 | 2923429888 | 134.910 | 132.064 | 3 |
