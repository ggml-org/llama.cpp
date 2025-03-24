#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include "HAP_farf.h"
#include "ggmlop_ap_skel.h"

#define ggml_tensor dsptensor

int ggmlop_dsp_open(const char*uri, remote_handle64* handle) {
    void *tptr = NULL;
    FARF(HIGH, "uri %s", uri);
    tptr = (void *)malloc(1);
    *handle = (remote_handle64)tptr;
    assert(*handle);
    return 0;
}

int ggmlop_dsp_close(remote_handle64 handle) {
    if (handle)
        free((void*)handle);
    return 0;
}

static void ggml_dump_tensor(struct ggml_tensor * tensor) {
    FARF(HIGH, "ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi, %5zi)\n",
         tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
         tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
}

static void ggml_abort(const char * file, int line, const char * fmt, ...) {
    //abort();
    return;
}

#define GGML_MAX_DIMS       4
#define GGML_UNUSED(x)      (void)(x)
#define GGML_PAD(x, n)      (((x) + (n) - 1) & ~((n) - 1))
#define GGML_ABORT(...)     ggml_abort(__FILE__, __LINE__, __VA_ARGS__)
#define GGML_ASSERT(x)      if (!(x)) GGML_ABORT("GGML_ASSERT(%s) failed", #x)
#define MIN(a, b)           ((a) < (b) ? (a) : (b))
#define MAX(a, b)           ((a) > (b) ? (a) : (b))
#define GGML_RESTRICT

#define static_assert(a, b) do { } while (0)

#define GGML_TENSOR_LOCALS_1(type, prefix, pointer, array) \
    const type prefix##0 = (pointer)->array[0]; \
    GGML_UNUSED(prefix##0);
#define GGML_TENSOR_LOCALS_2(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_1    (type, prefix, pointer, array) \
    const type prefix##1 = (pointer)->array[1]; \
    GGML_UNUSED(prefix##1);
#define GGML_TENSOR_LOCALS_3(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_2    (type, prefix, pointer, array) \
    const type prefix##2 = (pointer)->array[2]; \
    GGML_UNUSED(prefix##2);
#define GGML_TENSOR_LOCALS(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_3  (type, prefix, pointer, array) \
    const type prefix##3 = (pointer)->array[3]; \
    GGML_UNUSED(prefix##3);

#define GGML_TENSOR_UNARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define GGML_TENSOR_BINARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define GGML_TENSOR_BINARY_OP_LOCALS01 \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // GGML_TYPE_Q4_0_4_8 = 32,
    // GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    // GGML_TYPE_IQ4_NL_4_4 = 36,
    // GGML_TYPE_IQ4_NL_4_8 = 37,
    // GGML_TYPE_IQ4_NL_8_8 = 38,
    GGML_TYPE_COUNT   = 39,
};

static bool ggml_is_empty(const struct ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            return true;
        }
    }
    return false;
}

static bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return ggml_is_empty(t0) ? ggml_is_empty(t1) :
           (t1->ne[0]%t0->ne[0] == 0) &&
           (t1->ne[1]%t0->ne[1] == 0) &&
           (t1->ne[2]%t0->ne[2] == 0) &&
           (t1->ne[3]%t0->ne[3] == 0);
}

static bool ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return
            (t0->ne[0] == t1->ne[0]) &&
            (t0->ne[1] == t1->ne[1]) &&
            (t0->ne[2] == t1->ne[2]) &&
            (t0->ne[3] == t1->ne[3]);
}

static int64_t ggml_nrows(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

static bool ggml_is_contiguous_n(const struct ggml_tensor * tensor, int n) {
    size_t next_nb = sizeof(float);//ggml_type_size(tensor->type);
    if (tensor->ne[0] != 1/*ggml_blck_size(tensor->type)*/ && tensor->nb[0] != next_nb) {
        return false;
    }
    next_nb *= tensor->ne[0]/1/*ggml_blck_size(tensor->type)*/;
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

static bool ggml_is_contiguous_0(const struct ggml_tensor * tensor) {
    return ggml_is_contiguous_n(tensor, 0);
}

static bool ggml_is_contiguous(const struct ggml_tensor * tensor) {
    return ggml_is_contiguous_0(tensor);
}

inline static void ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] + y[i]; }

static void ggml_compute_forward_add_f32(
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {

    ggml_dump_tensor(src0);
    ggml_dump_tensor(src1);
#if 1
    float * a = (float*)src0->data;
    float * b = (float*)src1->data;
    float * c = (float*)dst->data;
    //TODO: Hexagon SIMD
    for (size_t idx = 0; idx < src0->data_len; idx++) {
        *c = *a + *b;
        a++;
        b++;
        c++;
    }
    return;
#endif
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    const int ith = 0;
    const int nth = 1;

    const int nr  = ggml_nrows(src0);

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (nb10 == sizeof(float)) {
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;
            const int64_t nr0 = ne00 / ne10;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
            float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

            for (int64_t r = 0; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
                vDSP_vadd(src0_ptr + r*ne10, 1, src1_ptr, 1, dst_ptr + r*ne10, 1, ne10);
#else
                ggml_vec_add_f32(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
#endif
            }
        }
    } else {
        // src1 is not contiguous
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const int64_t i10 = i0 % ne10;
                float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10);

                dst_ptr[i0] = src0_ptr[i0] + *src1_ptr;
            }
        }
    }
}

int ggmlop_dsp_add(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst)
{
    FARF(HIGH, "===============     DSP: ggmlop_dsp_add ");
    switch (src0->type) {
        case GGML_TYPE_F32:
        {
            if (src1->type == GGML_TYPE_F32) {
                ggml_compute_forward_add_f32(src0, src1, dst);
            } else {
                GGML_ABORT("fatal error");
            }
        } break;
        case GGML_TYPE_F16:
        {
            if (src1->type == GGML_TYPE_F16) {
                //ggml_compute_forward_add_f16_f16(dst);
            }
            else if (src1->type == GGML_TYPE_F32) {
                //ggml_compute_forward_add_f16_f32(dst);
            }
            else {
                GGML_ABORT("fatal error");
            }
        } break;
        case GGML_TYPE_BF16:
        {
            if (src1->type == GGML_TYPE_BF16) {
                //ggml_compute_forward_add_bf16_bf16(dst);
            }
            else if (src1->type == GGML_TYPE_F32) {
                //ggml_compute_forward_add_bf16_f32(dst);
            }
            else {
                GGML_ABORT("fatal error");
            }
        } break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        {
            //ggml_compute_forward_add_q_f32(dst);
        } break;
        default:
        {
            GGML_ABORT("fatal error");
        }
    }

    return 0;
}


int ggmlop_dsp_mulmat(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    FARF(HIGH, "===============     DSP: ggmlop_dsp_mulmat ");

    GGML_TENSOR_BINARY_OP_LOCALS

    ggml_dump_tensor(src0);
    ggml_dump_tensor(src1);

    const int vec_dot_type = 0;
    int64_t const vec_dot_num_rows = 1;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb10 == sizeof(float));

    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int64_t nr0 = ne0;
    const int64_t nr1 = ne1 * ne2 * ne3;

    int chunk_size = 16;
    int nth = 1;

    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    if (nchunk0 * nchunk1 < nth * 4) {
        nchunk0 = nr0 > nr1 ? nth : 1;
        nchunk1 = nr0 > nr1 ? 1 : nth;
    }

    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    int current_chunk = 0;

    const int64_t ith0 = current_chunk % nchunk0;
    const int64_t ith1 = current_chunk / nchunk0;

    const int64_t ir0_start = dr0 * ith0;
    const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

    const int64_t ir1_start = dr1 * ith1;
    const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

    int64_t num_rows_per_vec_dot = vec_dot_num_rows;

    const int src1_cont = ggml_is_contiguous(src1);
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const void * wdata = src1->data;
    const size_t row_size = sizeof(float) * ne10;
    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1;
                    ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char *)src0->data + (0 + i02 * nb02 + i03 * nb03);

                const char * src1_col = (const char *)wdata +
                                       (src1_cont || src1->type != vec_dot_type
                                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float *)((char *) dst->data +
                                            (i1 * nb1 + i2 * nb2 + i3 * nb3));


                for (int64_t ir0 = iir0;
                        ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {

                    float sumf = 0.0;
                    const float * GGML_RESTRICT x = (float*)src0_row + ir0 * nb01;
                    const float * GGML_RESTRICT y = (float*)src1_col;
                    float * GGML_RESTRICT s = &tmp[ir0 - iir0];
                    for (int i = 0; i < ne00; i++) {
                        sumf += x[i] * y[i];
                    }
                    *s = sumf;

                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16),
                           (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }

    return 0;
}
