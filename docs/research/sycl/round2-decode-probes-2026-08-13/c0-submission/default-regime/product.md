# Product campaign: mistral-7b-instruct-v0.1.Q4_K_M.gguf

- bin-dir: /home/svnbjrn/build-p63-80d52e708/bin
- baseline label: default-batch-unset
- candidate label: default-batch64
- baseline env: {}
- candidate env: {'UR_L0_BATCH_SIZE': '64'}
- candidate_enabled: True
- model shape: None
- campaign valid: True
- invalid diagnostics: none
- candidate env log assertions: UR_L0_BATCH_SIZE=64 (not validated from backend logs; key not emitted)
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16384 | q8_0/q8_0 | pp512 | Y | 111.52 | 111.58 | 0.14 | +/- 0.17 | 111.70 | 111.75 | 0.15 | +/- 0.18 | +0.16 | +0.16 | 0.21 | +/- 0.26 | n/a | n/a | n/a | 5 |
| 16384 | q8_0/q8_0 | tg128 | Y | 19.83 | 19.78 | 0.18 | +/- 0.23 | 19.88 | 19.85 | 0.05 | +/- 0.06 | +0.30 | +0.40 | 0.92 | +/- 1.14 | n/a | n/a | n/a | 5 |
