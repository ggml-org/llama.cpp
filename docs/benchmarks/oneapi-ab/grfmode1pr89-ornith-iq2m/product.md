# Product campaign: Ornith-1.5-35B-A3B-IQ2_M.gguf

- bin-dir: /home/svnbjrn/build-grf-split/bin
- baseline label: grf-off
- candidate label: grf-mode1
- baseline env: {}
- candidate env: {'GGML_SYCL_FA_LARGE_GRF': '1'}
- candidate_enabled: True
- model shape: {'model_layers': 41, 'query_heads': 16, 'head_dim': 256}
- campaign valid: False
- invalid diagnostics: ['sample build_commit 9ae0f9ad9 does not match repository commit 9595c7ca1200a2b2872c5699775c073a98274063']
- candidate env log assertions: GGML_SYCL_FA_LARGE_GRF=1 (validated in 8/8 candidate samples; valid=True)
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 226.81 | 229.37 | 5.57 | +/- 13.83 | 264.34 | 256.05 | 21.23 | +/- 52.75 | +17.20 | +11.82 | 11.72 | +/- 29.11 | 0 | 0.000 | 0.000 | 3 |
| 0 | q8_0/q8_0 | tg128 | Y | 51.65 | 52.03 | 0.84 | +/- 2.10 | 52.13 | 52.21 | 0.60 | +/- 1.50 | +0.00 | +0.36 | 0.86 | +/- 2.15 | 0 | 0.000 | 0.000 | 3 |
| 8192 | q8_0/q8_0 | pp512 | Y | 261.12 | 261.85 | 2.20 | +/- 5.48 | 260.95 | 260.97 | 0.08 | +/- 0.19 | -0.03 | -0.33 | 0.85 | +/- 2.12 | 2923429888 | 134.627 | 131.822 | 3 |
| 8192 | q8_0/q8_0 | tg128 | Y | 46.05 | 45.83 | 0.50 | +/- 1.24 | 45.09 | 45.46 | 0.75 | +/- 1.87 | -2.34 | -0.77 | 2.73 | +/- 6.78 | 2923429888 | 134.627 | 131.822 | 3 |
