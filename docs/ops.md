# GGML Operations

List of GGML operations and backend support status.

Legend:
- ✅ Fully supported by this backend
- 🟡 Partially supported by this backend
- ❌ Not supported by this backend

| Operation | BLAS | CPU | CUDA | Metal |
|-----------|------|------|------|------|
|                              ABS | ❌ | ✅ | 🟡 | ❌ |
|                              ACC | ❌ | ✅ | ✅ | ✅ |
|                              ADD | ❌ | ✅ | ✅ | 🟡 |
|                             ADD1 | ❌ | ✅ | ✅ | ❌ |
|                           ARANGE | ❌ | ✅ | ✅ | ✅ |
|                           ARGMAX | ❌ | ✅ | ✅ | ✅ |
|                          ARGSORT | ❌ | ✅ | ✅ | ✅ |
|                            CLAMP | ❌ | ✅ | ✅ | 🟡 |
|                           CONCAT | ❌ | ✅ | 🟡 | ✅ |
|                             CONT | ❌ | ✅ | 🟡 | ✅ |
|                       CONV_2D_DW | ❌ | ✅ | ✅ | ❌ |
|                CONV_TRANSPOSE_1D | ❌ | ✅ | ✅ | ✅ |
|                CONV_TRANSPOSE_2D | ❌ | ✅ | ✅ | ❌ |
|                              COS | ❌ | ✅ | ✅ | 🟡 |
|                      COUNT_EQUAL | ❌ | ✅ | ✅ | ❌ |
|                              CPY | ❌ | 🟡 | 🟡 | 🟡 |
|               CROSS_ENTROPY_LOSS | ❌ | ✅ | ✅ | ❌ |
|          CROSS_ENTROPY_LOSS_BACK | ❌ | ✅ | ✅ | ❌ |
|                    DIAG_MASK_INF | ❌ | ✅ | ✅ | 🟡 |
|                              DIV | ❌ | ✅ | ✅ | 🟡 |
|                              DUP | ❌ | ✅ | 🟡 | 🟡 |
|                              ELU | ❌ | ✅ | ❌ | 🟡 |
|                              EXP | ❌ | ✅ | 🟡 | ❌ |
|                   FLASH_ATTN_EXT | ❌ | ✅ | 🟡 | 🟡 |
|                GATED_LINEAR_ATTN | ❌ | ✅ | ✅ | ❌ |
|                            GEGLU | ❌ | ✅ | ✅ | 🟡 |
|                        GEGLU_ERF | ❌ | ✅ | ✅ | 🟡 |
|                      GEGLU_QUICK | ❌ | ✅ | ✅ | 🟡 |
|                             GELU | ❌ | ✅ | 🟡 | 🟡 |
|                         GELU_ERF | ❌ | ✅ | 🟡 | 🟡 |
|                       GELU_QUICK | ❌ | ✅ | 🟡 | 🟡 |
|                         GET_ROWS | ❌ | ✅ | 🟡 | ✅ |
|                    GET_ROWS_BACK | ❌ | 🟡 | 🟡 | ❌ |
|                       GROUP_NORM | ❌ | ✅ | ✅ | ✅ |
|                      HARDSIGMOID | ❌ | ✅ | 🟡 | ❌ |
|                        HARDSWISH | ❌ | ✅ | 🟡 | ❌ |
|                           IM2COL | ❌ | ✅ | ✅ | 🟡 |
|                          L2_NORM | ❌ | ✅ | ✅ | ✅ |
|                       LEAKY_RELU | ❌ | ✅ | ✅ | ✅ |
|                              LOG | ❌ | ✅ | ✅ | ❌ |
|                             MEAN | ❌ | ✅ | ✅ | ✅ |
|                              MUL | ❌ | ✅ | ✅ | 🟡 |
|                          MUL_MAT | 🟡 | 🟡 | 🟡 | 🟡 |
|                       MUL_MAT_ID | ❌ | ✅ | ✅ | ✅ |
|                              NEG | ❌ | ✅ | 🟡 | 🟡 |
|                             NORM | ❌ | ✅ | ✅ | 🟡 |
|                   OPT_STEP_ADAMW | ❌ | ✅ | ✅ | ❌ |
|                         OUT_PROD | 🟡 | 🟡 | 🟡 | ❌ |
|                              PAD | ❌ | ✅ | ✅ | ✅ |
|                   PAD_REFLECT_1D | ❌ | ✅ | ❌ | ✅ |
|                          POOL_2D | ❌ | ✅ | ✅ | ✅ |
|                            REGLU | ❌ | ✅ | ✅ | 🟡 |
|                             RELU | ❌ | ✅ | 🟡 | 🟡 |
|                           REPEAT | ❌ | ✅ | 🟡 | ✅ |
|                      REPEAT_BACK | ❌ | ✅ | ✅ | ❌ |
|                         RMS_NORM | ❌ | ✅ | ✅ | 🟡 |
|                    RMS_NORM_BACK | ❌ | ✅ | ✅ | ❌ |
|                     RMS_NORM_MUL | ❌ | ✅ | ✅ | ✅ |
|                             ROPE | ❌ | ✅ | ✅ | ✅ |
|                        ROPE_BACK | ❌ | ✅ | ✅ | ❌ |
|                        RWKV_WKV6 | ❌ | ✅ | ✅ | ✅ |
|                        RWKV_WKV7 | ❌ | ✅ | ✅ | ✅ |
|                            SCALE | ❌ | ✅ | ✅ | ✅ |
|                              SET | ❌ | ✅ | ❌ | ✅ |
|                         SET_ROWS | ❌ | 🟡 | ❌ | 🟡 |
|                              SGN | ❌ | ✅ | 🟡 | ❌ |
|                          SIGMOID | ❌ | ✅ | 🟡 | 🟡 |
|                             SILU | ❌ | ✅ | 🟡 | 🟡 |
|                        SILU_BACK | ❌ | ✅ | ✅ | ❌ |
|                              SIN | ❌ | ✅ | ✅ | 🟡 |
|                         SOFT_MAX | ❌ | ✅ | ✅ | ✅ |
|                    SOFT_MAX_BACK | ❌ | 🟡 | 🟡 | ❌ |
|                              SQR | ❌ | ✅ | ✅ | 🟡 |
|                             SQRT | ❌ | ✅ | ✅ | 🟡 |
|                         SSM_CONV | ❌ | ✅ | ✅ | ✅ |
|                         SSM_SCAN | ❌ | ✅ | ✅ | ✅ |
|                             STEP | ❌ | ✅ | 🟡 | ❌ |
|                              SUB | ❌ | ✅ | ✅ | 🟡 |
|                              SUM | ❌ | ✅ | ✅ | ❌ |
|                         SUM_ROWS | ❌ | ✅ | ✅ | ✅ |
|                           SWIGLU | ❌ | ✅ | ✅ | 🟡 |
|                             TANH | ❌ | ✅ | 🟡 | 🟡 |
|               TIMESTEP_EMBEDDING | ❌ | ✅ | ✅ | ✅ |
|                          UPSCALE | ❌ | ✅ | ✅ | 🟡 |
