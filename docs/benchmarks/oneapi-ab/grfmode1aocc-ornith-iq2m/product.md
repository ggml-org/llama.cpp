# Product campaign: Ornith-1.5-35B-A3B-IQ2_M.gguf

- bin-dir: /home/svnbjrn/build-grf-split/bin
- baseline label: grf-off
- candidate label: grf-mode1
- baseline env: {}
- candidate env: {'GGML_SYCL_FA_LARGE_GRF': '1'}
- candidate_enabled: True
- model shape: {'model_layers': 41, 'query_heads': 16, 'head_dim': 256}
- campaign valid: False
- invalid diagnostics: ['sample build_commit b23424e30 does not match repository commit 5b97abd57472e316c5c206516a97517af03a5850']
- candidate env log assertions: GGML_SYCL_FA_LARGE_GRF=1 (validated in 8/8 candidate samples; valid=True)
- dmesg fault hits before=1 after=1 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 246.93 | 246.95 | 11.16 | +/- 27.72 | 271.14 | 267.53 | 12.08 | +/- 30.01 | +7.74 | +8.34 | 1.28 | +/- 3.18 | 0 | 0.000 | 0.000 | 3 |
| 0 | q8_0/q8_0 | tg128 | Y | 52.34 | 52.51 | 0.33 | +/- 0.83 | 53.24 | 52.98 | 0.52 | +/- 1.28 | +0.79 | +0.89 | 0.79 | +/- 1.96 | 0 | 0.000 | 0.000 | 3 |
| 8192 | q8_0/q8_0 | pp512 | Y | 264.28 | 264.39 | 4.47 | +/- 11.10 | 259.78 | 260.91 | 3.02 | +/- 7.51 | -2.14 | -1.29 | 2.64 | +/- 6.56 | 2923429888 | 132.735 | 133.589 | 3 |
| 8192 | q8_0/q8_0 | tg128 | Y | 45.40 | 45.41 | 0.15 | +/- 0.36 | 45.70 | 45.75 | 0.21 | +/- 0.53 | +0.65 | +0.74 | 0.50 | +/- 1.25 | 2923429888 | 132.735 | 133.589 | 3 |
