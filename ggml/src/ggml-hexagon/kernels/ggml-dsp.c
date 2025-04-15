#include "ggml-dsp.h"

// =================================================================================================
//  section-1: tiny ggml-dsp, ported from original ggml
// =================================================================================================

static void   ggml_vec_dot_f32(int n, float * GGML_RESTRICT s, size_t bs, const float * GGML_RESTRICT x, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc);

static void   dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
static void   quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
static void   quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
static void   ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

static float ggml_table_f32_f16[1 << 16];

static struct ggml_compute_params g_params;

static int32 g_thread_counts = 1;

struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
        [GGML_TYPE_F32] = {
                .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
                .vec_dot_type             = GGML_TYPE_F32,
                .nrows                    = 1,
        },
        [GGML_TYPE_F16] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_F16,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q4_0] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
                .nrows                    = 2,
#else
                .nrows                    = 1,
#endif
        },
        [GGML_TYPE_Q4_1] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
                .nrows                    = 2,
#else
                .nrows                    = 1,
#endif
        },
        [GGML_TYPE_Q5_0] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_0,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q5_1] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_1,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q8_0] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
                .nrows                    = 2,
#else
                .nrows                    = 1,
#endif
        },
        [GGML_TYPE_Q8_1] = {
                .from_float               = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_1,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q2_K] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_K,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q3_K] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_K,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q4_K] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_K,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q5_K] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_K,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q6_K] = {
                .from_float               = quantize_row_q6_K,
                .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
                .vec_dot_type             = GGML_TYPE_Q8_K,
                .nrows                    = 1,
        },
};

static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {
        [GGML_TYPE_I8] = {
                .type_name                = "i8",
                .blck_size                = 1,
                .type_size                = sizeof(int8_t),
                .is_quantized             = false,
        },
        [GGML_TYPE_I16] = {
                .type_name                = "i16",
                .blck_size                = 1,
                .type_size                = sizeof(int16_t),
                .is_quantized             = false,
        },
        [GGML_TYPE_I32] = {
                .type_name                = "i32",
                .blck_size                = 1,
                .type_size                = sizeof(int32_t),
                .is_quantized             = false,
        },
        [GGML_TYPE_I64] = {
                .type_name                = "i64",
                .blck_size                = 1,
                .type_size                = sizeof(int64_t),
                .is_quantized             = false,
        },
        [GGML_TYPE_F64] = {
                .type_name                = "f64",
                .blck_size                = 1,
                .type_size                = sizeof(double),
                .is_quantized             = false,
        },
        [GGML_TYPE_F32] = {
                .type_name                = "f32",
                .blck_size                = 1,
                .type_size                = sizeof(float),
                .is_quantized             = false,
        },
        [GGML_TYPE_F16] = {
                .type_name                = "f16",
                .blck_size                = 1,
                .type_size                = sizeof(ggml_fp16_t),
                .is_quantized             = false,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q4_0] = {
                .type_name                = "q4_0",
                .blck_size                = QK4_0,
                .type_size                = sizeof(block_q4_0),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q4_1] = {
                .type_name                = "q4_1",
                .blck_size                = QK4_1,
                .type_size                = sizeof(block_q4_1),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [4] = { // GGML_TYPE_Q4_2
                .type_name                = "DEPRECATED",
                .blck_size                = 0,
                .type_size                = 0,
                .is_quantized             = false,
        },
        [5] = { // GGML_TYPE_Q4_3
                .type_name                = "DEPRECATED",
                .blck_size                = 0,
                .type_size                = 0,
                .is_quantized             = false,
        },
        [GGML_TYPE_Q5_0] = {
                .type_name                = "q5_0",
                .blck_size                = QK5_0,
                .type_size                = sizeof(block_q5_0),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q5_1] = {
                .type_name                = "q5_1",
                .blck_size                = QK5_1,
                .type_size                = sizeof(block_q5_1),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q8_0] = {
                .type_name                = "q8_0",
                .blck_size                = QK8_0,
                .type_size                = sizeof(block_q8_0),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q8_1] = {
                .type_name                = "q8_1",
                .blck_size                = QK8_1,
                .type_size                = sizeof(block_q8_1),
                .is_quantized             = true,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q2_K] = {
                .type_name                = "q2_K",
                .blck_size                = QK_K,
                .type_size                = sizeof(block_q2_K),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q3_K] = {
                .type_name                = "q3_K",
                .blck_size                = QK_K,
                .type_size                = sizeof(block_q3_K),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q4_K] = {
                .type_name                = "q4_K",
                .blck_size                = QK_K,
                .type_size                = sizeof(block_q4_K),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q5_K] = {
                .type_name                = "q5_K",
                .blck_size                = QK_K,
                .type_size                = sizeof(block_q5_K),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q6_K] = {
                .type_name                = "q6_K",
                .blck_size                = QK_K,
                .type_size                = sizeof(block_q6_K),
                .is_quantized             = true,
                .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
                .from_float_ref           = (ggml_from_float_t) quantize_row_q6_K_ref,
        },

};

void ggmlhexagon_log_internal(int level, const char *file, const char *func, int line, const char *format, ...) {
#if !GGMLHEXAGON_DEBUG
    return;
#endif
    static char s_ggmlhexagon_log_internal_buf[GGMLHEXAGON_LOGBUF_LEN];
    va_list args;
    va_start(args, format);
    int len_prefix = snprintf(s_ggmlhexagon_log_internal_buf, GGMLHEXAGON_LOGBUF_LEN, "[%s, %d]: ",
                              func, line);
    int len = vsnprintf(s_ggmlhexagon_log_internal_buf + len_prefix,
                        GGMLHEXAGON_LOGBUF_LEN - len_prefix, format, args);
    if (len < (GGMLHEXAGON_LOGBUF_LEN - len_prefix)) {
        FARF(ALWAYS, "%s\n", s_ggmlhexagon_log_internal_buf);
    }
    va_end(args);
}

void ggmlhexagon_dump_tensor_elements(const ggml_tensor * tensor) {
#if !GGMLHEXAGON_DEBUG
    return;
#endif
    float value = 0;
    char tmpbuf[GGMLHEXAGON_LOGBUF_LEN];
    size_t buflen = 0;
    if (tensor->type == GGML_TYPE_F32) {
        memset(tmpbuf, 0, GGMLHEXAGON_LOG_LEVEL_DEBUG);
        for (int h = 0; h < tensor->ne[3]; h++) {
            for (int i = 0; i < tensor->ne[2]; i++) {
                for (int j = 0; j < tensor->ne[1]; j++) {
                    for (int k = 0; k < tensor->ne[0]; k++) {
                        value = ((float *) tensor->data)[h * tensor->ne[2] + i * tensor->ne[1] +
                                                         j * tensor->ne[0] + k];
                        buflen += snprintf(tmpbuf + buflen, GGMLHEXAGON_LOGBUF_LEN - buflen, "%-4.2f\t", value);
                    }
                    buflen += snprintf(tmpbuf + buflen, GGMLHEXAGON_LOGBUF_LEN - buflen, "\n");
                }
            }
        }
        GGMLHEXAGON_LOG_DEBUG("\n%s\n", tmpbuf);
    }

    GGMLHEXAGON_LOG_DEBUG("\n");
}

void ggmlhexagon_dump_tensor(const ggml_tensor * tensor, int dump_tensor_data) {
    GGMLHEXAGON_LOG_DEBUG("ne = %5d x %5d x %5d x %5d , nb = (%5zi, %5zi, %5zi, %5zi)\n",
         tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
         tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);

    if ((1 == dump_tensor_data) && (ggml_nbytes(tensor) < 320)) {
        ggmlhexagon_dump_tensor_elements(tensor);
    }
}

static const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type) {
    return &type_traits_cpu[type];
}

void ggml_vec_dot_f32(int n, float * GGML_RESTRICT s, size_t bs, const float * GGML_RESTRICT x,
                             size_t bx, const float *GGML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float) (x[i] * y[i]);
    }
    *s = sumf;
}

inline void ggml_vec_mul_f32 (const int n, float * z, const float * x, const float * y) {
    for (int i = 0; i < n; ++i) z[i]  = x[i]*y[i];
}

inline void ggml_vec_div_f32 (const int n, float * z, const float * x, const float * y) {
    for (int i = 0; i < n; ++i) z[i]  = x[i]/y[i];
}

inline void ggml_vec_sub_f32 (const int n, float * z, const float * x, const float * y) {
    for (int i = 0; i < n; ++i) z[i]  = x[i] - y[i];
}

const struct ggml_type_traits * ggml_get_type_traits(enum ggml_type type) {
    return &type_traits[type];
}

int64_t ggml_blck_size(enum ggml_type type) {
    return type_traits[type].blck_size;
}

size_t ggml_type_size(enum ggml_type type) {
    return type_traits[type].type_size;
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    assert(ne % ggml_blck_size(type) == 0);
    return ggml_type_size(type)*ne/ggml_blck_size(type);
}

size_t ggml_nbytes(const struct ggml_tensor * tensor) {
    size_t nbytes;
    const size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    else {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    return nbytes;
}

size_t ggml_nbytes_pad(const struct ggml_tensor * tensor) {
    return GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

double ggml_type_sizef(enum ggml_type type) {
    return ((double)(type_traits[type].type_size))/type_traits[type].blck_size;
}

const char * ggml_type_name(enum ggml_type type) {
    return type < GGML_TYPE_COUNT ? type_traits[type].type_name : "NONE";
}

bool ggml_is_quantized(enum ggml_type type) {
    return type_traits[type].is_quantized;
}

bool ggml_is_empty(const struct ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] == 0) {
            return true;
        }
    }
    return false;
}

bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return ggml_is_empty(t0) ? ggml_is_empty(t1) :
           (t1->ne[0]%t0->ne[0] == 0) &&
           (t1->ne[1]%t0->ne[1] == 0) &&
           (t1->ne[2]%t0->ne[2] == 0) &&
           (t1->ne[3]%t0->ne[3] == 0);
}

bool ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");
    return
            (t0->ne[0] == t1->ne[0]) &&
            (t0->ne[1] == t1->ne[1]) &&
            (t0->ne[2] == t1->ne[2]) &&
            (t0->ne[3] == t1->ne[3]);
}

int64_t ggml_nrows(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

bool ggml_is_transposed(const struct ggml_tensor * tensor) {
    return tensor->nb[0] > tensor->nb[1];
}

static bool ggml_is_contiguous_n(const struct ggml_tensor * tensor, int n) {
    size_t next_nb = ggml_type_size(tensor->type);
    if (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        return false;
    }
    next_nb *= tensor->ne[0]/ggml_blck_size(tensor->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] != 1) {
            if (i > n) {
                if (tensor->nb[i] != next_nb) {
                    return false;
                }
                next_nb *= tensor->ne[i];
            } else {
                // this dimension does not need to be contiguous
                next_nb = tensor->ne[i]*tensor->nb[i];
            }
        }
    }
    return true;
}

 int64_t ggml_nelements(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

static bool ggml_is_contiguous_0(const struct ggml_tensor * tensor) {
    return ggml_is_contiguous_n(tensor, 0);
}

 bool ggml_is_contiguous(const struct ggml_tensor * tensor) {
    return ggml_is_contiguous_0(tensor);
}

inline static void ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) {
    for (int i = 0; i < n; ++i) z[i]  = x[i] + y[i];
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    GGMLHEXAGON_LOG_DEBUG("enter ggml_abort");
    abort();
}

// FP16 <-> FP32
static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
                            (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

 inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
        const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

 inline float ggml_lookup_fp16_to_fp32(ggml_fp16_t f) {
    uint16_t s;
    memcpy(&s, &f, sizeof(uint16_t));
    return ggml_table_f32_f16[s];
}

static inline void ggml_init(void) {
    for (int i = 0; i < (1 << 16); ++i) {
        union {
            uint16_t u16;
            ggml_fp16_t fp16;
        } u = {i};
        ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
    }

    //FIXME:HVX multithreading should be utilized in hexagon-kernels
    g_params.ith = 0;
    g_params.nth = 1;
    //FIXME:hardcode buffer size
    g_params.wsize = 512 * 1024 * 1024;
    g_params.wdata = (char*)malloc(g_params.wsize);
    GGML_ASSERT(NULL != g_params.wdata);
}

static inline void ggml_deinit(void) {
    free(g_params.wdata);
    g_params.wdata = NULL;
    g_params.wsize = 0;
}

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static float make_qx_quants(int n, int nmax, const float * GGML_RESTRICT x, int8_t * GGML_RESTRICT L, int rmse_type,
                            const float * GGML_RESTRICT qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 0; i < n; ++i) {
#else
    for (int i = 0; i < n; ++i) {
#endif
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

static void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT ql = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    int8_t L[QK_K];
    float   scales[QK_K/16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {

            const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        if (max_abs_scale < GROUP_MAX_EPS) {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = GGML_FP32_TO_FP16(0.f);
            x += QK_K;
            continue;
        }

        float iscale = -128.f/max_scale;
        y[i].d = GGML_FP32_TO_FP16(1/iscale);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
        }

        for (int j = 0; j < QK_K/16; ++j) {
            float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-32, MIN(31, l));
                L[16*j + ii] = l + 32;
            }
        }

        uint8_t * GGML_RESTRICT ql = y[i].ql;
        uint8_t * GGML_RESTRICT qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;
    }
}

static void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q6_K * GGML_RESTRICT y = vy;
    quantize_row_q6_K_ref(x, y, k);
}

static void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;

}

static inline uint64 hexagon_perf_get_time_us(void) {
    unsigned long long count;
    asm volatile (" %0 = c31:30 " : "=r"(count));
    return (uint64)(count) * 10ull / 192ull;
}

void ggml_time_init(void) {
}

int64_t ggml_time_ms(void) {
    return hexagon_perf_get_time_us() * 1000;
}

int64_t ggml_time_us(void) {
    return hexagon_perf_get_time_us();
}

// =================================================================================================
//  ggml-hexagon kernel helper function
// =================================================================================================
int ggmlop_get_thread_counts(void) {
    return g_thread_counts;
}

struct ggml_compute_params * ggmlop_get_params(void) {
     return &g_params;
}

int ggml_get_params_size(void) {
     return g_params.wsize;
}

char * ggml_get_params_data(void) {
     return g_params.wdata;
}

// =================================================================================================
//  implementation of ggml-hexagon kernel skel function
// =================================================================================================
int ggmlop_dsp_open(const char*uri, remote_handle64* handle) {
    void *tptr = NULL;
    GGMLHEXAGON_LOG_DEBUG("uri %s", uri);
    tptr = (void *)malloc(1);
    *handle = (remote_handle64)tptr;
    assert(*handle);

    ggml_init();

    GGMLHEXAGON_LOG_DEBUG("api_version = 0x%x", qurt_api_version());
    GGMLHEXAGON_LOG_DEBUG("hvx units = 0x%d", qurt_hvx_get_units());
    qurt_arch_version_t  vers;
    qurt_sysenv_get_arch_version(&vers);
    GGMLHEXAGON_LOG_DEBUG("arch_version=0x%x", vers.arch_version);
    qurt_sysenv_app_heap_t aheap;
    qurt_sysenv_get_app_heap(&aheap);
    GGMLHEXAGON_LOG_DEBUG("aheap.heap_base=0x%x, aheap.heap_limit=0x%x", aheap.heap_base, aheap.heap_limit);
    qurt_sysenv_max_hthreads_t mhwt;
    qurt_sysenv_get_max_hw_threads(&mhwt);
    GGMLHEXAGON_LOG_DEBUG("max hardware threads=%d", mhwt.max_hthreads);

    return 0;
}

int ggmlop_dsp_close(remote_handle64 handle) {
    if (handle)
        free((void*)handle);

    ggml_deinit();

    return 0;
}

AEEResult ggmlop_dsp_setclocks(remote_handle64 handle, int32 power_level, int32 latency, int32 dcvs_enabled, int32 thread_counts) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    HAP_power_request_t request;
    memset(&request, 0, sizeof(HAP_power_request_t));
    request.type = HAP_power_set_apptype;
    request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

    g_thread_counts = thread_counts;

    void * ggmop_ctx = (void*)(handle);
    int retval = HAP_power_set(ggmop_ctx, &request);
    if (retval)  {
        GGMLHEXAGON_LOG_DEBUG("failed first power vote");
        return AEE_EFAILED;
    }

    //configure clocks & DCVS mode
    memset(&request, 0, sizeof(HAP_power_request_t));
    request.type = HAP_power_set_DCVS_v2;
    request.dcvs_v2.dcvs_enable = TRUE;
    request.dcvs_v2.dcvs_params.target_corner = (HAP_dcvs_voltage_corner_t)power_level;
    if (dcvs_enabled) {
        request.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_DISABLE;
        request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_DISABLE;
    } else {
        request.dcvs_v2.dcvs_params.min_corner = request.dcvs_v2.dcvs_params.target_corner;
        request.dcvs_v2.dcvs_params.max_corner = request.dcvs_v2.dcvs_params.target_corner;
    }
    request.dcvs_v2.dcvs_option     = HAP_DCVS_V2_PERFORMANCE_MODE;
    request.dcvs_v2.set_dcvs_params = TRUE;
    request.dcvs_v2.set_latency     = TRUE;
    request.dcvs_v2.latency         = latency;
    retval = HAP_power_set(ggmop_ctx, &request);
    if (retval) {
        GGMLHEXAGON_LOG_DEBUG("failed to vote for performance mode");
        return AEE_EFAILED;
    }

    memset(&request, 0, sizeof(HAP_power_request_t));
    request.type = HAP_power_set_HVX;
    request.hvx.power_up = TRUE;
    retval = HAP_power_set(ggmop_ctx, &request);
    if (retval) {
        GGMLHEXAGON_LOG_DEBUG("failed to vote for HVX power");
        return AEE_EFAILED;
    }
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return AEE_SUCCESS;
}


// =================================================================================================
//  implementation of ggml-hexagon kernel, it's better to put every kernel to a single file
// =================================================================================================
int ggmlop_dsp_softmax(remote_handle64 h, const dsptensor * src0, const dsptensor * src1, dsptensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );


    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );

    return 0;
}

int ggmlop_dsp_rmsnorm(remote_handle64 h, const dsptensor * src0, const dsptensor * src1, dsptensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );

    return 0;
}

int ggmlop_dsp_pool2d(remote_handle64 h, const dsptensor * src0, const dsptensor * src1, dsptensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );


    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );


    return 0;
}
