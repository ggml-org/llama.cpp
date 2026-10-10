# Product campaign: Ornith-1.5-35B-A3B-IQ2_M.gguf

- bin-dir: /home/svnbjrn/build-xe-kmd/bin
- baseline label: grf128-default
- candidate label: grf256-fa
- baseline env: {}
- candidate env: {'GGML_SYCL_FA_GRF_SIZE_BUILD': '256'}
- candidate_enabled: True
- model shape: {'model_layers': 41, 'query_heads': 16, 'head_dim': 256}
- campaign valid: True
- invalid diagnostics: none
- candidate env log assertions: GGML_SYCL_FA_GRF_SIZE_BUILD=256 (not validated from backend logs; key not emitted)
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 266.27 | 266.02 | 0.55 | +/- 1.37 | 274.91 | 274.92 | 0.20 | +/- 0.50 | +3.33 | +3.34 | 0.24 | +/- 0.58 | 0 | 0.000 | 0.000 | 3 |
| 0 | q8_0/q8_0 | tg128 | Y | 53.52 | 53.53 | 0.06 | +/- 0.14 | 53.35 | 53.37 | 0.05 | +/- 0.12 | -0.36 | -0.30 | 0.17 | +/- 0.43 | 0 | 0.000 | 0.000 | 3 |
| 2048 | q8_0/q8_0 | pp512 | Y | 298.92 | 298.97 | 0.17 | +/- 0.43 | 298.97 | 298.77 | 0.42 | +/- 1.04 | +0.01 | -0.07 | 0.20 | +/- 0.49 | 730857472 | 38.676 | 38.598 | 3 |
| 2048 | q8_0/q8_0 | tg128 | Y | 52.92 | 52.93 | 0.02 | +/- 0.04 | 52.81 | 52.82 | 0.05 | +/- 0.14 | -0.20 | -0.20 | 0.13 | +/- 0.32 | 730857472 | 38.676 | 38.598 | 3 |
| 8192 | q8_0/q8_0 | pp512 | Y | 265.02 | 265.23 | 0.53 | +/- 1.32 | 265.59 | 265.28 | 0.67 | +/- 1.66 | -0.09 | +0.02 | 0.22 | +/- 0.55 | 2923429888 | 136.200 | 141.073 | 3 |
| 8192 | q8_0/q8_0 | tg128 | Y | 46.59 | 46.59 | 0.03 | +/- 0.07 | 48.26 | 48.26 | 0.00 | +/- 0.01 | +3.58 | +3.58 | 0.07 | +/- 0.17 | 2923429888 | 136.200 | 141.073 | 3 |
