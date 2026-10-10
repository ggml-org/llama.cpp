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
- candidate env log assertions: GGML_SYCL_FA_LARGE_GRF=1 (not validated from backend logs; key not emitted)
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 267.34 | 267.65 | 1.06 | +/- 2.62 | 274.44 | 274.46 | 0.10 | +/- 0.26 | +2.71 | +2.55 | 0.40 | +/- 1.00 | 0 | 0.000 | 0.000 | 3 |
| 0 | q8_0/q8_0 | tg128 | Y | 53.51 | 53.51 | 0.02 | +/- 0.06 | 53.51 | 53.52 | 0.05 | +/- 0.12 | +0.06 | +0.02 | 0.11 | +/- 0.28 | 0 | 0.000 | 0.000 | 3 |
| 8192 | q8_0/q8_0 | pp512 | Y | 267.07 | 267.29 | 0.98 | +/- 2.43 | 264.82 | 265.14 | 0.79 | +/- 1.96 | -0.84 | -0.80 | 0.63 | +/- 1.57 | 2923429888 | 136.118 | 136.103 | 3 |
| 8192 | q8_0/q8_0 | tg128 | Y | 46.56 | 46.55 | 0.07 | +/- 0.17 | 46.56 | 46.55 | 0.07 | +/- 0.18 | -0.01 | -0.01 | 0.30 | +/- 0.76 | 2923429888 | 136.118 | 136.103 | 3 |
