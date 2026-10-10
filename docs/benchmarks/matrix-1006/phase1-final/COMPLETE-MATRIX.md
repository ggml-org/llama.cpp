# Complete matrix, b12327, Arc A770 (xe)

Baseline env (every run): `ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_ENABLE_GRAPH=1 GGML_SYCL_GRAPH_EVICTION_TIMEOUT=300 GGML_SYCL_XE_COPY_ENGINE_DEFAULT=0 GGML_SYCL_FA_LARGE_GRF=1`; ambient GGML_/SYCL_/UR_/ZE_/LLAMA_ARG vars were unset first. 'env delta' = what the arm changed on top. Per-run rows with full env, bench args, idle, load and mean act_freq: COMPLETE-MATRIX.csv.

## ccs

| arm | ccs | freq min/max/profile | env delta | extra args | n | pp512 | tg64 | pp512@8k | tg64@8k |
|---|---|---|---|---|---|---|---|---|---|
| ccs1 | 1 | 600/2400/base | (none) | - | 3 | 1636.2 | 62.3 | 219.4 | 45.3 |
| ccs2 | 2 | 600/2400/base | (none) | - | 3 | 1163.7 | 50.8 | 206.3 | 32.8 |
| ccs4 | 4 | 600/2400/base | (none) | - | 3 | 623.9 | 30.2 | 179.4 | 18.6 |

## envs

| arm | ccs | freq min/max/profile | env delta | extra args | n | pp512 | tg64 | pp512@8k | tg64@8k |
|---|---|---|---|---|---|---|---|---|---|
| ASYNC0 | 1 | 600/2400/base | GGML_SYCL_USE_ASYNC_MEM_OP=0 | - | 3 | 1633.7 | 62.1 | 219.4 | 45.1 |
| COPYENG1 | 1 | 600/2400/base | UR_L0_USE_COPY_ENGINE=1 | - | 3 | 1633.4 | 62.0 | 219.5 | 45.1 |
| DMMV1 | 1 | 600/2400/base | GGML_SYCL_PRIORITIZE_DMMV=1 | - | 3 | 1632.8 | 61.9 | 218.7 | 45.1 |
| ESIMD0 | 1 | 600/2400/base | GGML_SYCL_ENABLE_ESIMD=0 | - | 3 | 1632.1 | 59.2 | 219.3 | 43.6 |
| FFN0 | 1 | 600/2400/base | GGML_SYCL_FFN_FUSION=0 | - | 3 | 1632.6 | 61.8 | 219.2 | 45.0 |
| FUSION0 | 1 | 600/2400/base | GGML_SYCL_ENABLE_FUSION=0 | - | 3 | 1619.7 | 60.9 | 219.0 | 44.6 |
| GRAPH0 | 1 | 600/2400/base | GGML_SYCL_ENABLE_GRAPH=0 | - | 3 | 1633.0 | 58.0 | 219.4 | 42.8 |
| GRF0 | 1 | 600/2400/base | GGML_SYCL_FA_LARGE_GRF=0 | - | 3 | 1502.7 | 62.1 | 219.2 | 45.2 |
| ILCL0 | 1 | 600/2400/base | SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 | - | 3 | 1629.3 | 59.1 | 213.5 | 43.1 |
| ILCL1 | 1 | 600/2400/base | SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 | - | 3 | 1634.2 | 62.1 | 219.0 | 45.2 |
| MKLFA0 | 1 | 600/2400/base | GGML_SYCL_ENABLE_MKL_FA=0 | - | 3 | 1632.4 | 62.1 | 344.0 | 45.2 |
| MKLQT2048 | 1 | 600/2400/base | GGML_SYCL_MKL_FA_Q_TILE=2048 | - | 3 | 1633.1 | 62.1 | 218.9 | 45.2 |
| OPT0 | 1 | 600/2400/base | GGML_SYCL_ENABLE_OPT=0 | - | 3 | 1621.8 | 35.5 | 219.4 | 29.1 |
| PINNED0 | 1 | 600/2400/base | GGML_SYCL_ENABLE_HOST_PINNED_MEM=0 | - | 3 | 1634.2 | 61.4 | 219.1 | 44.8 |
| Q8GQATILE | 1 | 600/2400/base | GGML_SYCL_FA_Q8_GQA_TILE=1 | - | 3 | 1632.8 | 58.4 | 218.7 | 33.3 |
| Q8QF0 | 1 | 600/2400/base | GGML_SYCL_Q8_KV_QUANTS_FIRST=0 | - | 3 | 1630.2 | 63.3 | 221.1 | 37.9 |
| VECSTD | 1 | 600/2400/base | GGML_SYCL_FA_FORCE_VEC_STANDARD=1 | - | 3 | 1632.9 | 62.0 | 219.1 | 45.2 |
| VMM0 | 1 | 600/2400/base | GGML_SYCL_ENABLE_VMM=0 | - | 3 | 1632.8 | 62.2 | 212.9 | 45.2 |
| WG32 | 1 | 600/2400/base | GGML_SYCL_MAX_WG_PER_CU=32 | - | 3 | 1634.6 | 62.1 | 219.7 | 46.0 |
| WG8 | 1 | 600/2400/base | GGML_SYCL_MAX_WG_PER_CU=8 | - | 3 | 1632.8 | 62.1 | 219.6 | 42.3 |
| base | 1 | 600/2400/base | (none) | - | 3 | 1634.2 | 62.1 | 219.3 | 45.2 |

## il

| arm | ccs | freq min/max/profile | env delta | extra args | n | pp512 | tg64 | pp512@8k | tg64@8k |
|---|---|---|---|---|---|---|---|---|---|
| ccs1.il0 | 1 | 600/2400/base | SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 | - | 3 | 1628.5 | 59.1 | 212.9 | 43.0 |
| ccs1.il1 | 1 | 600/2400/base | SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 | - | 3 | 1632.4 | 62.0 | 219.1 | 45.2 |
| ccs2.il0 | 2 | 600/2400/base | SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 | - | 3 | 1160.7 | 48.7 | 200.4 | 31.6 |
| ccs2.il1 | 2 | 600/2400/base | SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 | - | 3 | 1162.9 | 50.6 | 206.0 | 32.7 |
| ccs4.il0 | 4 | 600/2400/base | SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 | - | 3 | 623.4 | 29.4 | 174.9 | 18.1 |
| ccs4.il1 | 4 | 600/2400/base | SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 | - | 3 | 623.8 | 30.2 | 178.4 | 18.5 |

## freq

| arm | ccs | freq min/max/profile | env delta | extra args | n | pp512 | tg64 | pp512@8k | tg64@8k |
|---|---|---|---|---|---|---|---|---|---|
| def | 1 | 600/2400/base | (none) | - | 3 | 1632.6 | 62.1 | 219.2 | 45.2 |
| floor1500 | 1 | 1500/2400/base | (none) | - | 3 | 1633.4 | 62.2 | 219.4 | 45.3 |
| pin1500 | 1 | 1500/1500/base | (none) | - | 3 | 1344.9 | 50.4 | 157.3 | 35.7 |
| pin2000 | 1 | 2000/2000/base | (none) | - | 3 | 1602.3 | 57.8 | 197.0 | 41.8 |
| pin2400 | 1 | 2400/2400/base | (none) | - | 3 | 1634.1 | 62.2 | 220.8 | 45.4 |
| powersave | 1 | 600/2400/power_saving | (none) | - | 3 | 1633.2 | 62.0 | 218.9 | 45.2 |

## ornith

| arm | ccs | freq min/max/profile | env delta | extra args | n | pp512 | tg64 | pp512@8k | tg64@8k |
|---|---|---|---|---|---|---|---|---|---|
| mc4096_none | 1 | 600/2400/base | (none) | --moe-cache 4096 --load-mode none | 3 | 284.1 | 12.0 | 239.1 | 11.4 |
| mcauto_auto | 1 | 600/2400/base | (none) | --moe-cache auto --load-mode auto | 3 | 273.7 | 13.9 | 229.4 | 13.1 |
| mcauto_none | 1 | 600/2400/base | (none) | --moe-cache auto --load-mode none | 3 | 283.1 | 12.1 | 238.7 | 11.3 |
| mcoff_auto | 1 | 600/2400/base | (none) | --moe-cache off --load-mode auto | 3 | 373.5 | 49.8 | 306.6 | 42.4 |
| mcon_none | 1 | 600/2400/base | (none) | --moe-cache on --load-mode none | 3 | 283.2 | 12.0 | 239.9 | 11.3 |
| mcsoft_none | 1 | 600/2400/base | (none) | --moe-cache soft --load-mode none | 3 | 347.4 | 19.6 | 286.5 | 18.2 |
| prod | 1 | 600/2400/base | (none) | --moe-cache off --load-mode none | 3 | 377.2 | 49.7 | 308.5 | 42.6 |
| prod_graph0 | 1 | 600/2400/base | GGML_SYCL_ENABLE_GRAPH=0 | --moe-cache off --load-mode none | 3 | 378.0 | 49.6 | 308.4 | 42.7 |
| prod_grf0 | 1 | 600/2400/base | GGML_SYCL_FA_LARGE_GRF=0 | --moe-cache off --load-mode none | 3 | 369.6 | 49.4 | 310.0 | 42.4 |
| prod_ilcl1 | 1 | 600/2400/base | SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 | --moe-cache off --load-mode none | 3 | 377.0 | 49.9 | 308.0 | 42.5 |
| prod_mkl0 | 1 | 600/2400/base | GGML_SYCL_ENABLE_MKL_FA=0 | --moe-cache off --load-mode none | 3 | 377.9 | 49.4 | 236.1 | 42.4 |
| prod_pinned0 | 1 | 600/2400/base | GGML_SYCL_ENABLE_HOST_PINNED_MEM=0 | --moe-cache off --load-mode none | 3 | 374.5 | 45.5 | 305.7 | 38.5 |

## two concurrent 8B processes (aggregate of A+B, mean of 3 rounds)

| arm | ccs | timeslice us | env delta | agg pp512 | agg tg64 |
|---|---|---|---|---|---|
| ccs1.ts1000 | 1 | 1000 | (none) | 1523.6 | 58.96 |
| ccs1.ts20000 | 1 | 20000 | (none) | 1625.0 | 62.08 |
| ccs2.ts1000 | 2 | 1000 | (none) | 1064.1 | 48.24 |
| ccs2.ts20000 | 2 | 20000 | (none) | 1153.5 | 50.57 |
| ccs4.ts1000 | 4 | 1000 | (none) | 568.3 | 28.85 |
| ccs4.ts20000 | 4 | 20000 | (none) | 619.6 | 30.14 |

## spec-type server (Ornith production flags; mean of 2 launches x 3 reps)

| arm | server args | prompt | tg t/s | draft acc/n |
|---|---|---|---|---|
| none | --spec-type none --moe-cache off --load-mode none | repeat | 46.8 | 0/0 |
| none | --spec-type none --moe-cache off --load-mode none | prose | 47.0 | 0/0 |
| ngmod | --spec-type ngram-mod --moe-cache off --load-mode none | repeat | 34.9 | 2102/3200 |
| ngmod | --spec-type ngram-mod --moe-cache off --load-mode none | prose | 36.7 | 354/1152 |
| ngmod_small | --spec-type ngram-mod --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 32 --moe-cache off --load-mode none | repeat | 31.3 | 2062/3018 |
| ngmod_small | --spec-type ngram-mod --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 32 --moe-cache off --load-mode none | prose | 39.6 | 434/992 |
| ngsimple | --spec-type ngram-simple --moe-cache off --load-mode none | repeat | 66.7 | 2112/2118 |
| ngsimple | --spec-type ngram-simple --moe-cache off --load-mode none | prose | 47.0 | 0/0 |
| ngmapk | --spec-type ngram-map-k --moe-cache off --load-mode none | repeat | 35.4 | 1134/2158 |
| ngmapk | --spec-type ngram-map-k --moe-cache off --load-mode none | prose | 47.0 | 0/0 |
| ngmapk4v | --spec-type ngram-map-k4v --moe-cache off --load-mode none | repeat | 41.9 | 580/1120 |
| ngmapk4v | --spec-type ngram-map-k4v --moe-cache off --load-mode none | prose | 47.1 | 0/0 |
| ngcache | --spec-type ngram-cache --moe-cache off --load-mode none | repeat | 44.1 | 1854/1972 |
| ngcache | --spec-type ngram-cache --moe-cache off --load-mode none | prose | 23.0 | 21/1242 |
| none_mcauto | --spec-type none --moe-cache auto --load-mode none | repeat | 11.6 | 0/0 |
| none_mcauto | --spec-type none --moe-cache auto --load-mode none | prose | 11.7 | 0/0 |
| ngmod_mcauto | --spec-type ngram-mod --moe-cache auto --load-mode none | repeat | 23.1 | 2102/3200 |
| ngmod_mcauto | --spec-type ngram-mod --moe-cache auto --load-mode none | prose | 11.9 | 276/1152 |
| ngmod_lmauto | --spec-type ngram-mod --moe-cache off --load-mode auto | repeat | 35.5 | 2102/3200 |
| ngmod_lmauto | --spec-type ngram-mod --moe-cache off --load-mode auto | prose | 37.8 | 574/1344 |
