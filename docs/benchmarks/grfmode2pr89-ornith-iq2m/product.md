# Product campaign: Ornith-1.5-35B-A3B-IQ2_M.gguf

- bin-dir: /home/svnbjrn/build-grf-split/bin
- baseline label: grf-off
- candidate label: grf-mode2
- baseline env: {}
- candidate env: {'GGML_SYCL_FA_LARGE_GRF': '2'}
- candidate_enabled: True
- model shape: {'model_layers': 41, 'query_heads': 16, 'head_dim': 256}
- campaign valid: False
- invalid diagnostics: ['sample build_commit 9ae0f9ad9 does not match repository commit 9595c7ca1200a2b2872c5699775c073a98274063']
- candidate env log assertions: GGML_SYCL_FA_LARGE_GRF=2 (validated in 8/8 candidate samples; valid=True)
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 229.72 | 237.25 | 17.84 | +/- 44.32 | 234.70 | 236.21 | 6.51 | +/- 16.18 | +4.58 | +0.01 | 9.12 | +/- 22.66 | 0 | 0.000 | 0.000 | 3 |
| 0 | q8_0/q8_0 | tg128 | Y | 52.83 | 52.44 | 0.76 | +/- 1.89 | 51.80 | 52.10 | 0.63 | +/- 1.56 | -2.11 | -0.61 | 2.66 | +/- 6.60 | 0 | 0.000 | 0.000 | 3 |
| 8192 | q8_0/q8_0 | pp512 | Y | 257.16 | 257.33 | 0.48 | +/- 1.19 | 263.98 | 262.36 | 6.68 | +/- 16.60 | +2.73 | +1.95 | 2.49 | +/- 6.18 | 2923429888 | 134.524 | 122.799 | 3 |
| 8192 | q8_0/q8_0 | tg128 | Y | 46.02 | 45.66 | 0.62 | +/- 1.55 | 42.01 | 42.03 | 0.05 | +/- 0.13 | -8.72 | -7.94 | 1.38 | +/- 3.42 | 2923429888 | 134.524 | 122.799 | 3 |
