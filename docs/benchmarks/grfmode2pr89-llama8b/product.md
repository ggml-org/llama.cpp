# Product campaign: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

- bin-dir: /home/svnbjrn/build-grf-split/bin
- baseline label: grf-off
- candidate label: grf-mode2
- baseline env: {}
- candidate env: {'GGML_SYCL_FA_LARGE_GRF': '2'}
- candidate_enabled: True
- model shape: {'model_layers': 32, 'query_heads': 32, 'head_dim': 128}
- campaign valid: False
- invalid diagnostics: ['sample build_commit 9ae0f9ad9 does not match repository commit 9595c7ca1200a2b2872c5699775c073a98274063']
- candidate env log assertions: GGML_SYCL_FA_LARGE_GRF=2 (validated in 8/8 candidate samples; valid=True)
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 895.16 | 922.04 | 56.42 | +/- 140.16 | 964.23 | 999.85 | 77.48 | +/- 192.49 | +7.72 | +8.37 | 1.72 | +/- 4.28 | 0 | 0.000 | 0.000 | 3 |
| 0 | q8_0/q8_0 | tg128 | Y | 46.54 | 46.51 | 0.32 | +/- 0.79 | 46.92 | 47.07 | 0.32 | +/- 0.80 | +1.46 | +1.20 | 0.88 | +/- 2.19 | 0 | 0.000 | 0.000 | 3 |
| 8192 | q8_0/q8_0 | pp512 | Y | 209.49 | 209.78 | 0.55 | +/- 1.36 | 209.28 | 209.16 | 0.72 | +/- 1.79 | -0.51 | -0.30 | 0.39 | +/- 0.97 | 2281701376 | 81.791 | 78.147 | 3 |
| 8192 | q8_0/q8_0 | tg128 | Y | 35.85 | 35.85 | 0.05 | +/- 0.12 | 34.25 | 34.38 | 0.25 | +/- 0.63 | -4.53 | -4.10 | 0.82 | +/- 2.03 | 2281701376 | 81.791 | 78.147 | 3 |
