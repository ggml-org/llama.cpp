#include "ggml.h"
#include "ggml-cpu.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <windows.h>
typedef volatile LONG atomic_int;
static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
#else
#include <stdatomic.h>
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

struct ggml_context * make_ctx(void) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    return ggml_init(params);
}

char g_userdata[] = "ggml";
atomic_int g_custom1_count = 0;
atomic_int g_custom2_count = 0;
atomic_int g_custom3_count = 0;

void custom1(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    // check that the userdata is correct
    GGML_ASSERT(userdata == NULL);
    GGML_ASSERT(ggml_are_same_shape(dst, a));

    atomic_fetch_add(&g_custom1_count, 1);

    const float * a_data = ggml_get_data_f32(a);
    float * dst_data = ggml_get_data_f32(dst);

    // this assumes that the tensors are contiguous
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(a));

    // parallelize by elements
    const int ne = (int)ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = MIN(ie0 + dr, ne);

    for (int i = ie0; i < ie1; ++i) {
        dst_data[i] = a_data[i] * 2;
    }
}

void custom2(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata) {
    // check that the userdata is correct
    GGML_ASSERT(userdata == g_userdata);
    GGML_ASSERT(strcmp(userdata, "ggml") == 0);
    GGML_ASSERT(ggml_are_same_shape(dst, a));
    GGML_ASSERT(ggml_are_same_shape(dst, b));

    atomic_fetch_add(&g_custom2_count, 1);

    const float * a_data = ggml_get_data_f32(a);
    const float * b_data = ggml_get_data_f32(b);
    float * dst_data = ggml_get_data_f32(dst);

    // parallelize by rows
    const int nr = (int)ggml_nrows(dst);
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // number of columns
    const int nc = (int)dst->ne[0];

    // this assumes that the tensors are contiguous
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));

    for (int ir = ir0; ir < ir1; ++ir) {
        for (int ic = 0; ic < nc; ++ic) {
            const int i = ir * nc + ic;
            dst_data[i] = a_data[i] + b_data[i];
        }
    }
}

void custom3(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, const struct ggml_tensor * c, int ith, int nth, void * userdata) {
    // check that the userdata is correct
    GGML_ASSERT(userdata == g_userdata);
    GGML_ASSERT(strcmp(userdata, "ggml") == 0);
    GGML_ASSERT(ggml_are_same_shape(dst, a));
    GGML_ASSERT(ggml_are_same_shape(dst, b));
    GGML_ASSERT(ggml_are_same_shape(dst, c));

    atomic_fetch_add(&g_custom3_count, 1);

    const float * a_data = ggml_get_data_f32(a);
    const float * b_data = ggml_get_data_f32(b);
    const float * c_data = ggml_get_data_f32(c);
    float * dst_data = ggml_get_data_f32(dst);

    // dont parallelize
    GGML_ASSERT(ith == 0);

    // number of elements
    const int ne = (int)ggml_nelements(dst);

    // this assumes that the tensors are contiguous
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(c));

    for (int i = 0; i < ne; ++i) {
        dst_data[i] = a_data[i] + b_data[i] + c_data[i];
    }
}

void custom(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    struct ggml_tensor * src0 = dst->src[0];
    struct ggml_tensor * src1 = dst->src[1];
    struct ggml_tensor * src2 = dst->src[2];
    struct ggml_tensor * src3 = dst->src[3];
    struct ggml_tensor * src4 = dst->src[4];

    int32_t * dst_data = (int32_t *) ggml_get_data(dst);
    const float * src0_data = ggml_get_data_f32(src0);
    const float * src1_data = ggml_get_data_f32(src1);
    const float * src2_data = ggml_get_data_f32(src2);
    const float * src3_data = ggml_get_data_f32(src3);
    const float * src4_data = ggml_get_data_f32(src4);

    // check that the userdata is correct
    GGML_ASSERT(userdata == g_userdata);
    GGML_ASSERT(strcmp(userdata, "ggml") == 0);

    // check that the tensors are contiguous
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(src2));
    GGML_ASSERT(ggml_is_contiguous(src3));
    GGML_ASSERT(ggml_is_contiguous(src4));

    // check that the shapes are the same
    GGML_ASSERT(ggml_are_same_shape(dst, src0));
    GGML_ASSERT(ggml_are_same_shape(dst, src1));
    GGML_ASSERT(ggml_are_same_shape(dst, src2));
    GGML_ASSERT(ggml_are_same_shape(dst, src3));
    GGML_ASSERT(ggml_are_same_shape(dst, src4));


    for (int i = ith; i < ggml_nelements(dst); i += nth) {
        dst_data[i] = src0_data[i] + src1_data[i] * src2_data[i] - src3_data[i] * src4_data[i];
    }
}

int main(int argc, const char** argv) {

    float buf1_f32[1024];
    for (int i = 0; i < 1024; ++i) {
        buf1_f32[i] = (float)(i + 1);
    }
    float buf2_f32[1024];
    for (int i = 0; i < 1024; ++i) {
        buf2_f32[i] = (float)(i + 1) * 2;
    }
    float buf3_f32[1024];
    for (int i = 0; i < 1024; ++i) {
        buf3_f32[i] = (float)(i + 1) * 3;
    }

    // map_custom1
    // 2 tasks, no userdata, parallelized by elements
    {
        struct ggml_context * ctx = make_ctx();
        struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);
        memcpy(t->data, buf1_f32, ggml_nbytes(t));

        struct ggml_tensor * m1 = ggml_map_custom1(ctx, t, custom1, 2, NULL);

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, m1);

        ggml_graph_compute_with_ctx(ctx, graph, 4);

        const float * output = ggml_get_data_f32(m1);

        for (int i = 0; i < ggml_nelements(m1); ++i) {
            GGML_ASSERT(output[i] == buf1_f32[i] * 2);
        }
        GGML_ASSERT(g_custom1_count == 2);

        ggml_free(ctx);
    }

    // map_custom2
    // max tasks (4), userdata, parallelized by rows
    {
        struct ggml_context * ctx = make_ctx();
        struct ggml_tensor * t1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);
        struct ggml_tensor * t2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);
        memcpy(t1->data, buf1_f32, ggml_nbytes(t1));
        memcpy(t2->data, buf2_f32, ggml_nbytes(t2));

        struct ggml_tensor * m2 = ggml_map_custom2(ctx, t1, t2, custom2, GGML_N_TASKS_MAX, g_userdata);

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, m2);

        ggml_graph_compute_with_ctx(ctx, graph, 4);

        const float * output = ggml_get_data_f32(m2);

        for (int i = 0; i < ggml_nelements(m2); ++i) {
            GGML_ASSERT(output[i] == buf1_f32[i] + buf2_f32[i]);
        }

        GGML_ASSERT(g_custom2_count == 4);

        ggml_free(ctx);
    }

    // map_custom3
    // 1 task, userdata, not parallelized
    {
        struct ggml_context * ctx = make_ctx();
        struct ggml_tensor * t1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);
        struct ggml_tensor * t2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);
        struct ggml_tensor * t3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);

        memcpy(t1->data, buf1_f32, ggml_nbytes(t1));
        memcpy(t2->data, buf2_f32, ggml_nbytes(t2));
        memcpy(t3->data, buf3_f32, ggml_nbytes(t3));

        struct ggml_tensor * m3 = ggml_map_custom3(ctx, t1, t2, t3, custom3, 1, g_userdata);

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, m3);

        ggml_graph_compute_with_ctx(ctx, graph, 4);

        const float * output = ggml_get_data_f32(m3);

        for (int i = 0; i < ggml_nelements(m3); ++i) {
            GGML_ASSERT(output[i] == buf1_f32[i] + buf2_f32[i] + buf3_f32[i]);
        }

        GGML_ASSERT(g_custom3_count == 1);

        ggml_free(ctx);
    }

    // custom
    {
        struct ggml_context * ctx = make_ctx();
        struct ggml_tensor * t1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);
        struct ggml_tensor * t2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);
        struct ggml_tensor * t3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);
        struct ggml_tensor * t4 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);
        struct ggml_tensor * t5 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);
        memcpy(t1->data, buf1_f32, ggml_nbytes(t1));
        memcpy(t2->data, buf2_f32, ggml_nbytes(t2));
        memcpy(t3->data, buf3_f32, ggml_nbytes(t3));
        memcpy(t4->data, buf1_f32, ggml_nbytes(t4));
        memcpy(t5->data, buf2_f32, ggml_nbytes(t5));

        struct ggml_tensor * args[] = {
            t1, t2, t3, t4, t5,
        };

        struct ggml_tensor * m4 = ggml_custom_4d(ctx, GGML_TYPE_I32, 10, 2, 1, 1, args, sizeof(args)/sizeof(args[0]), custom, GGML_N_TASKS_MAX, g_userdata);

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, m4);

        ggml_graph_compute_with_ctx(ctx, graph, 4);

        const int32_t * output = (const int32_t *) ggml_get_data(m4);

        for (int i = 0; i < ggml_nelements(m4); ++i) {
            GGML_ASSERT(output[i] == buf1_f32[i] + buf2_f32[i] * buf3_f32[i] - buf1_f32[i] * buf2_f32[i]);
        }

        ggml_free(ctx);
    }

    return 0;
}
