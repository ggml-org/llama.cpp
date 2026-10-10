# Product campaign: Ornith-1.5-35B-A3B-IQ2_M.gguf

- bin-dir: /home/svnbjrn/build-fa-grf256/bin
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
| 0 | q8_0/q8_0 | pp512 | Y | 266.53 | 266.49 | 1.05 | +/- 2.60 | 273.46 | 272.99 | 1.68 | +/- 4.18 | +2.22 | +2.44 | 0.44 | +/- 1.10 | 0 | 0.000 | 0.000 | 3 |
| 0 | q8_0/q8_0 | tg128 | Y | 53.41 | 53.41 | 0.02 | +/- 0.04 | 53.28 | 53.34 | 0.13 | +/- 0.31 | -0.23 | -0.13 | 0.25 | +/- 0.61 | 0 | 0.000 | 0.000 | 3 |
| 8192 | q8_0/q8_0 | pp512 | Y | 265.41 | 265.27 | 0.25 | +/- 0.62 | 265.20 | 265.05 | 0.80 | +/- 1.99 | -0.08 | -0.08 | 0.22 | +/- 0.54 | 2923429888 | 135.696 | 135.741 | 3 |
| 8192 | q8_0/q8_0 | tg128 | Y | 46.42 | 46.44 | 0.04 | +/- 0.10 | 46.43 | 46.42 | 0.07 | +/- 0.18 | -0.11 | -0.04 | 0.17 | +/- 0.43 | 2923429888 | 135.696 | 135.741 | 3 |
