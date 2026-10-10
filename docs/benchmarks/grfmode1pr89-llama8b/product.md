# Product campaign: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

- bin-dir: /home/svnbjrn/build-grf-split/bin
- baseline label: grf-off
- candidate label: grf-mode1
- baseline env: {}
- candidate env: {'GGML_SYCL_FA_LARGE_GRF': '1'}
- candidate_enabled: True
- model shape: {'model_layers': 32, 'query_heads': 32, 'head_dim': 128}
- campaign valid: False
- invalid diagnostics: ['sample build_commit 9ae0f9ad9 does not match repository commit 9595c7ca1200a2b2872c5699775c073a98274063']
- candidate env log assertions: GGML_SYCL_FA_LARGE_GRF=1 (validated in 8/8 candidate samples; valid=True)
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 966.94 | 938.60 | 56.17 | +/- 139.54 | 1067.74 | 1020.92 | 107.05 | +/- 265.95 | +10.43 | +8.57 | 5.09 | +/- 12.66 | 0 | 0.000 | 0.000 | 3 |
| 0 | q8_0/q8_0 | tg128 | Y | 46.64 | 46.45 | 0.36 | +/- 0.90 | 46.09 | 46.26 | 0.31 | +/- 0.76 | -0.06 | -0.40 | 0.75 | +/- 1.85 | 0 | 0.000 | 0.000 | 3 |
| 8192 | q8_0/q8_0 | pp512 | Y | 209.76 | 209.65 | 0.28 | +/- 0.69 | 207.22 | 207.14 | 0.65 | +/- 1.62 | -1.01 | -1.20 | 0.33 | +/- 0.82 | 2281701376 | 82.043 | 81.799 | 3 |
| 8192 | q8_0/q8_0 | tg128 | Y | 35.96 | 35.96 | 0.07 | +/- 0.18 | 35.85 | 35.83 | 0.04 | +/- 0.10 | -0.30 | -0.37 | 0.31 | +/- 0.76 | 2281701376 | 82.043 | 81.799 | 3 |
