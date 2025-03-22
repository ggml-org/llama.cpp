/* ggml op functions, running on Hexagon cDSP as libggmlop_skel.so
 *
 * currently I didn't find a general approach to compile/build this hexagon-kernel file, a manual build approach can works fine in my local dev envs. I'm working on this build issue.
 *
 */

#if 0
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "HAP_farf.h"
#include "ggmlop.h"

#define GGML_ASSERT(x)  do { } while(0)
#define MIN(a, b)       ((a) < (b) ? (a) : (b))
#define GGML_RESTRICT

int ggmlop_open(const char * uri, remote_handle64 * handle) {
    void * tptr = NULL;
    FARF(HIGH, "uri %s", uri);
    tptr = (void *)malloc(1);
    *handle = (remote_handle64)tptr;
    assert(*handle);
    return 0;
}

int ggmlop_close(remote_handle64 handle) {
    if (handle)
        free((void*)handle);
    return 0;
}

int ggmlop_add(remote_handle64 h, const dsptensor * src0, const dsptensor * src1, dsptensor * dst) {
    FARF(HIGH, "===============     DSP: ggmlop_add ");
    for (size_t idx = 0; idx < src0->dataLen; idx++) {
        dst->data[idx] = src0->data[idx] + src1->data[idx];
    }

    return 0;
}

static void ggmldsp_dump_tensor(struct dsptensor * src0) {
    FARF(HIGH, "ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi, %5zi)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
}

static int ggmldsp_is_contiguous(const struct dsptensor * tensor) {
    int n = 0;
    size_t next_nb = sizeof(float);
    if (tensor->ne[0] != 1 && tensor->nb[0] != next_nb) {
        return 0;
    }
    next_nb *= tensor->ne[0];
    for (int i = 1; i < 4; i++) {
        if (tensor->ne[i] != 1) {
            if (i > n) {
                if (tensor->nb[i] != next_nb) {
                    return 0;
                }
                next_nb *= tensor->ne[i];
            } else {
                next_nb = tensor->ne[i] * tensor->nb[i];
            }
        }
    }
    return 1;
}

//FIXME: unknown issue on cDSP
int ggmlop_mulmat(remote_handle64 h, const struct dsptensor * src00, const struct dsptensor * src10, dsptensor * dst) {
    FARF(HIGH, "===============     DSP: ggmlop_mulmat ");

    dsptensor * src0 = (dsptensor*)src00;
    dsptensor * src1 = (dsptensor*)src10;
    const int64_t ne00 = src0->ne[0];
    (void) (ne00);
    const int64_t ne01 = (src0)->ne[1];
    (void) (ne01);
    const int64_t ne02 = (src0)->ne[2];
    (void) (ne02);
    const int64_t ne03 = (src0)->ne[3];
    (void) (ne03);
    const size_t nb00 = (src0)->nb[0];
    (void) (nb00);
    const size_t nb01 = (src0)->nb[1];
    (void) (nb01);
    const size_t nb02 = (src0)->nb[2];
    (void) (nb02);
    const size_t nb03 = (src0)->nb[3];
    (void) (nb03);
    const int64_t ne10 = (src1)->ne[0];
    (void) (ne10);
    const int64_t ne11 = (src1)->ne[1];
    (void) (ne11);
    const int64_t ne12 = (src1)->ne[2];
    (void) (ne12);
    const int64_t ne13 = (src1)->ne[3];
    (void) (ne13);
    const size_t nb10 = (src1)->nb[0];
    (void) (nb10);
    const size_t nb11 = (src1)->nb[1];
    (void) (nb11);
    const size_t nb12 = (src1)->nb[2];
    (void) (nb12);
    const size_t nb13 = (src1)->nb[3];
    (void) (nb13);
    const int64_t ne0 = (dst)->ne[0];
    (void) (ne0);
    const int64_t ne1 = (dst)->ne[1];
    (void) (ne1);
    const int64_t ne2 = (dst)->ne[2];
    (void) (ne2);
    const int64_t ne3 = (dst)->ne[3];
    (void) (ne3);
    const size_t nb0 = (dst)->nb[0];
    (void) (nb0);
    const size_t nb1 = (dst)->nb[1];
    (void) (nb1);
    const size_t nb2 = (dst)->nb[2];
    (void) (nb2);
    const size_t nb3 = (dst)->nb[3];
    (void) (nb3);

    ggmldsp_dump_tensor(src0);
    ggmldsp_dump_tensor(src1);

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

    const int src1_cont = ggmldsp_is_contiguous(src1);
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
#endif
