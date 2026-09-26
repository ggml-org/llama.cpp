# Product campaign: mistral-7b-instruct-v0.1.Q4_K_M.gguf

- bin-dir: /home/svnbjrn/build-p63-80d52e708/bin
- baseline label: imm0-batch-unset
- candidate label: imm0-batch64
- baseline env: {'UR_L0_USE_IMMEDIATE_COMMANDLISTS': '0'}
- candidate env: {'UR_L0_USE_IMMEDIATE_COMMANDLISTS': '0', 'UR_L0_BATCH_SIZE': '64'}
- candidate_enabled: True
- model shape: None
- campaign valid: True
- invalid diagnostics: none
- candidate env log assertions: UR_L0_BATCH_SIZE=64 (not validated from backend logs; key not emitted); UR_L0_USE_IMMEDIATE_COMMANDLISTS=0 (not validated from backend logs; key not emitted)
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 907.25 | 914.36 | 12.79 | +/- 15.88 | 944.60 | 932.00 | 28.79 | +/- 35.74 | +2.71 | +1.96 | 4.06 | +/- 5.04 | n/a | n/a | n/a | 5 |
| 0 | q8_0/q8_0 | tg128 | Y | 28.09 | 28.07 | 0.04 | +/- 0.05 | 37.15 | 37.10 | 0.06 | +/- 0.08 | +32.18 | +32.18 | 0.30 | +/- 0.38 | n/a | n/a | n/a | 5 |
| 16384 | q8_0/q8_0 | pp512 | Y | 112.15 | 112.22 | 0.20 | +/- 0.24 | 112.19 | 112.16 | 0.12 | +/- 0.15 | -0.12 | -0.05 | 0.23 | +/- 0.28 | n/a | n/a | n/a | 5 |
| 16384 | q8_0/q8_0 | tg128 | Y | 21.02 | 21.01 | 0.04 | +/- 0.05 | 25.51 | 25.52 | 0.12 | +/- 0.15 | +21.56 | +21.49 | 0.38 | +/- 0.47 | n/a | n/a | n/a | 5 |
