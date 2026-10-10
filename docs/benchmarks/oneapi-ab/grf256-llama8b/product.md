# Product campaign: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

- bin-dir: /home/svnbjrn/build-xe-kmd/bin
- baseline label: grf128-default
- candidate label: grf256-fa
- baseline env: {}
- candidate env: {'GGML_SYCL_FA_GRF_SIZE_BUILD': '256'}
- candidate_enabled: True
- model shape: {'model_layers': 32, 'query_heads': 32, 'head_dim': 128}
- campaign valid: True
- invalid diagnostics: none
- candidate env log assertions: GGML_SYCL_FA_GRF_SIZE_BUILD=256 (not validated from backend logs; key not emitted)
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 1003.68 | 1004.04 | 0.80 | +/- 1.98 | 1095.20 | 1098.03 | 5.58 | +/- 13.87 | +9.12 | +9.36 | 0.62 | +/- 1.53 | 0 | 0.000 | 0.000 | 3 |
| 0 | q8_0/q8_0 | tg128 | Y | 47.03 | 47.03 | 0.01 | +/- 0.03 | 47.76 | 47.75 | 0.03 | +/- 0.07 | +1.55 | +1.53 | 0.06 | +/- 0.15 | 0 | 0.000 | 0.000 | 3 |
| 8192 | q8_0/q8_0 | pp512 | Y | 210.32 | 210.76 | 0.88 | +/- 2.19 | 211.03 | 211.26 | 1.15 | +/- 2.87 | +0.33 | +0.24 | 0.92 | +/- 2.28 | 2281701376 | 83.241 | 79.471 | 3 |
| 8192 | q8_0/q8_0 | tg128 | Y | 36.48 | 36.49 | 0.01 | +/- 0.03 | 34.83 | 34.83 | 0.00 | +/- 0.01 | -4.54 | -4.54 | 0.03 | +/- 0.08 | 2281701376 | 83.241 | 79.471 | 3 |
