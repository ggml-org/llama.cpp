# Product campaign: Meta-Llama-3.1-8B-Instruct-heretic.Q4_K_M.gguf

- bin-dir: /home/svnbjrn/build-p63-80d52e708/bin
- baseline label: wg2
- candidate label: wg16
- baseline env: {'GGML_SYCL_MAX_WG_PER_CU': '2'}
- candidate env: {'GGML_SYCL_MAX_WG_PER_CU': '16'}
- candidate_enabled: True
- model shape: None
- campaign valid: True
- invalid diagnostics: none
- candidate env log assertions: GGML_SYCL_MAX_WG_PER_CU=16 (not validated from backend logs; key not emitted)
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16384 | f16/f16 | pp512 | Y | 112.45 | 112.37 | 0.19 | +/- 0.23 | 112.58 | 112.56 | 0.10 | +/- 0.13 | +0.13 | +0.17 | 0.18 | +/- 0.22 | n/a | n/a | n/a | 5 |
| 16384 | f16/f16 | tg128 | Y | 7.34 | 7.34 | 0.01 | +/- 0.01 | 17.32 | 17.32 | 0.02 | +/- 0.03 | +135.78 | +135.79 | 0.23 | +/- 0.29 | n/a | n/a | n/a | 5 |
