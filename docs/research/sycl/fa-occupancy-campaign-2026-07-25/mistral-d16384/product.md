# Product campaign: mistral-7b-instruct-v0.1.Q4_K_M.gguf

- bin-dir: /mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork-fa-occupancy/build/bin
- baseline label: wg2-baseline
- candidate label: wg16-candidate
- baseline env: {'GGML_SYCL_MAX_WG_PER_CU': '2'}
- candidate env: {'GGML_SYCL_MAX_WG_PER_CU': '16'}
- candidate_enabled: True
- model shape: {'model_layers': 32, 'query_heads': 32, 'head_dim': 128}
- campaign valid: True
- invalid diagnostics: none
- candidate env log assertions: GGML_SYCL_MAX_WG_PER_CU=16 (not validated from backend logs; key not emitted)
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16384 | f16/f16 | pp512 | Y | 97.83 | 97.83 | 0.05 | +/- 0.06 | 97.85 | 97.85 | 0.04 | +/- 0.04 | -0.01 | +0.01 | 0.06 | +/- 0.07 | 8589934592 | 59.712 | 146.867 | 5 |
| 16384 | f16/f16 | tg128 | Y | 6.95 | 6.95 | 0.01 | +/- 0.01 | 17.10 | 17.09 | 0.02 | +/- 0.02 | +145.96 | +145.91 | 0.51 | +/- 0.63 | 8589934592 | 59.712 | 146.867 | 5 |
| 16384 | q8_0/q8_0 | pp512 | Y | 97.61 | 97.56 | 0.12 | +/- 0.15 | 97.63 | 97.63 | 0.11 | +/- 0.13 | +0.02 | +0.08 | 0.21 | +/- 0.26 | 4563402752 | 44.907 | 75.669 | 5 |
| 16384 | q8_0/q8_0 | tg128 | Y | 9.84 | 9.84 | 0.04 | +/- 0.04 | 16.58 | 16.58 | 0.01 | +/- 0.01 | +68.66 | +68.50 | 0.59 | +/- 0.74 | 4563402752 | 44.907 | 75.669 | 5 |
