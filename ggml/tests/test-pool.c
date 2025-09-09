#include "ggml.h"
#include "ggml-cpu.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

struct ggml_context* make_ctx(void) {
    struct ggml_init_params params = {
        .mem_size = 2 * 1024 * 1024,
    };

    return ggml_init(params);
}

int main(int argc, const char** argv) {

    float buf_f32[1024];
    ggml_fp16_t buf_f16[1024];
    for (int i = 0; i < 1024; ++i) {
        buf_f32[i] = (float)(i + 1);
        buf_f16[i] = ggml_fp32_to_fp16(buf_f32[i]);
    }

    // avg pool 1d - Float 32
    {
        struct ggml_context * ctx = make_ctx();
        struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);
        memcpy(t->data, buf_f32, ggml_nbytes(t));

        struct ggml_tensor * t_pooled = ggml_pool_1d(ctx, t, GGML_OP_POOL_AVG, 3, 3, 0);
        GGML_ASSERT(t_pooled->ne[0] == 3);
        GGML_ASSERT(t_pooled->ne[1] == 2);
        GGML_ASSERT(t_pooled->ne[2] == 1);

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, t_pooled);

        ggml_graph_compute_with_ctx(ctx, graph, 4);

        const float * output = ggml_get_data_f32(t_pooled);

        GGML_ASSERT(output[0] == 2);
        GGML_ASSERT(output[1] == 5);
        GGML_ASSERT(output[2] == 8);
        GGML_ASSERT(output[3] == 12);
        GGML_ASSERT(output[4] == 15);
        GGML_ASSERT(output[5] == 18);

        ggml_free(ctx);
    }

    // avg pool 1d - Float 16
    {
        struct ggml_context * ctx = make_ctx();
        struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 10, 2);
        memcpy(t->data, buf_f16, ggml_nbytes(t));

        struct ggml_tensor * t_pooled = ggml_pool_1d(ctx, t, GGML_OP_POOL_AVG, 3, 3, 0);
        GGML_ASSERT(t_pooled->ne[0] == 3);
        GGML_ASSERT(t_pooled->ne[1] == 2);
        GGML_ASSERT(t_pooled->ne[2] == 1);

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, t_pooled);

        ggml_graph_compute_with_ctx(ctx, graph, 4);

        const float * output = ggml_get_data_f32(t_pooled);

        GGML_ASSERT(output[0] == 2);
        GGML_ASSERT(output[1] == 5);
        GGML_ASSERT(output[2] == 8);
        GGML_ASSERT(output[3] == 12);
        GGML_ASSERT(output[4] == 15);
        GGML_ASSERT(output[5] == 18);

        ggml_free(ctx);
    }

    // max pool 1d - Float 32
    {
        struct ggml_context * ctx = make_ctx();
        struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);
        memcpy(t->data, buf_f32, ggml_nbytes(t));

        struct ggml_tensor * t_pooled = ggml_pool_1d(ctx, t, GGML_OP_POOL_MAX, 3, 3, 0);
        GGML_ASSERT(t_pooled->ne[0] == 3);
        GGML_ASSERT(t_pooled->ne[1] == 2);
        GGML_ASSERT(t_pooled->ne[2] == 1);

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, t_pooled);

        ggml_graph_compute_with_ctx(ctx, graph, 4);

        const float * output = ggml_get_data_f32(t_pooled);
        GGML_ASSERT(output[0] == 3);
        GGML_ASSERT(output[1] == 6);
        GGML_ASSERT(output[2] == 9);
        GGML_ASSERT(output[3] == 13);
        GGML_ASSERT(output[4] == 16);
        GGML_ASSERT(output[5] == 19);

        ggml_free(ctx);
    }

    // max pool 1d - Float 16
    {
        struct ggml_context * ctx = make_ctx();
        struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 10, 2);
        memcpy(t->data, buf_f16, ggml_nbytes(t));

        struct ggml_tensor * t_pooled = ggml_pool_1d(ctx, t, GGML_OP_POOL_MAX, 3, 3, 0);
        GGML_ASSERT(t_pooled->ne[0] == 3);
        GGML_ASSERT(t_pooled->ne[1] == 2);
        GGML_ASSERT(t_pooled->ne[2] == 1);

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, t_pooled);

        ggml_graph_compute_with_ctx(ctx, graph, 4);

        const float * output = ggml_get_data_f32(t_pooled);
        GGML_ASSERT(output[0] == 3);
        GGML_ASSERT(output[1] == 6);
        GGML_ASSERT(output[2] == 9);
        GGML_ASSERT(output[3] == 13);
        GGML_ASSERT(output[4] == 16);
        GGML_ASSERT(output[5] == 19);

        ggml_free(ctx);
    }

    // avg pool 2d - Float 32
    {
        struct ggml_context * ctx = make_ctx();
        struct ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 10, 10, 2);
        memcpy(t->data, buf_f32, ggml_nbytes(t));

        struct ggml_tensor * t_pooled = ggml_pool_2d(ctx, t, GGML_OP_POOL_AVG, 3, 4, 3, 4, 0, 0);
        GGML_ASSERT(t_pooled->ne[0] == 3);
        GGML_ASSERT(t_pooled->ne[1] == 2);
        GGML_ASSERT(t_pooled->ne[2] == 2);
        GGML_ASSERT(t_pooled->ne[3] == 1);

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, t_pooled);

        ggml_graph_compute_with_ctx(ctx, graph, 4);

        const float * output = ggml_get_data_f32(t_pooled);
        GGML_ASSERT(output[0] == 17);
        GGML_ASSERT(output[1] == 20);
        GGML_ASSERT(output[2] == 23);
        GGML_ASSERT(output[3] == 57);
        GGML_ASSERT(output[4] == 60);
        GGML_ASSERT(output[5] == 63);
        GGML_ASSERT(output[6] == 117);
        GGML_ASSERT(output[7] == 120);
        GGML_ASSERT(output[8] == 123);
        GGML_ASSERT(output[9] == 157);
        GGML_ASSERT(output[10] == 160);
        GGML_ASSERT(output[11] == 163);


        ggml_free(ctx);
    }

    // avg pool 2d - Float 16
    {
        struct ggml_context * ctx = make_ctx();
        struct ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 10, 10, 2);
        memcpy(t->data, buf_f16, ggml_nbytes(t));

        struct ggml_tensor * t_pooled = ggml_pool_2d(ctx, t, GGML_OP_POOL_AVG, 3, 4, 3, 4, 0, 0);
        GGML_ASSERT(t_pooled->ne[0] == 3);
        GGML_ASSERT(t_pooled->ne[1] == 2);
        GGML_ASSERT(t_pooled->ne[2] == 2);
        GGML_ASSERT(t_pooled->ne[3] == 1);

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, t_pooled);

        ggml_graph_compute_with_ctx(ctx, graph, 4);

        const float * output = ggml_get_data_f32(t_pooled);
        GGML_ASSERT(output[0] == 17);
        GGML_ASSERT(output[1] == 20);
        GGML_ASSERT(output[2] == 23);
        GGML_ASSERT(output[3] == 57);
        GGML_ASSERT(output[4] == 60);
        GGML_ASSERT(output[5] == 63);
        GGML_ASSERT(output[6] == 117);
        GGML_ASSERT(output[7] == 120);
        GGML_ASSERT(output[8] == 123);
        GGML_ASSERT(output[9] == 157);
        GGML_ASSERT(output[10] == 160);
        GGML_ASSERT(output[11] == 163);


        ggml_free(ctx);
    }

    // max pool 2d - Float 32
    {
        struct ggml_context * ctx = make_ctx();
        struct ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 10, 10, 2);
        memcpy(t->data, buf_f32, ggml_nbytes(t));

        struct ggml_tensor * t_pooled = ggml_pool_2d(ctx, t, GGML_OP_POOL_MAX, 3, 4, 3, 4, 0, 0);
        GGML_ASSERT(t_pooled->ne[0] == 3);
        GGML_ASSERT(t_pooled->ne[1] == 2);
        GGML_ASSERT(t_pooled->ne[2] == 2);
        GGML_ASSERT(t_pooled->ne[3] == 1);

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, t_pooled);

        ggml_graph_compute_with_ctx(ctx, graph, 4);

        const float * output = ggml_get_data_f32(t_pooled);
        GGML_ASSERT(output[0] == 33);
        GGML_ASSERT(output[1] == 36);
        GGML_ASSERT(output[2] == 39);
        GGML_ASSERT(output[3] == 73);
        GGML_ASSERT(output[4] == 76);
        GGML_ASSERT(output[5] == 79);
        GGML_ASSERT(output[6] == 133);
        GGML_ASSERT(output[7] == 136);
        GGML_ASSERT(output[8] == 139);
        GGML_ASSERT(output[9] == 173);
        GGML_ASSERT(output[10] == 176);
        GGML_ASSERT(output[11] == 179);

        ggml_free(ctx);
    }

    // max pool 2d - Float 16
    {
        struct ggml_context * ctx = make_ctx();
        struct ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 10, 10, 2);
        memcpy(t->data, buf_f16, ggml_nbytes(t));

        struct ggml_tensor * t_pooled = ggml_pool_2d(ctx, t, GGML_OP_POOL_MAX, 3, 4, 3, 4, 0, 0);
        GGML_ASSERT(t_pooled->ne[0] == 3);
        GGML_ASSERT(t_pooled->ne[1] == 2);
        GGML_ASSERT(t_pooled->ne[2] == 2);
        GGML_ASSERT(t_pooled->ne[3] == 1);

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, t_pooled);

        ggml_graph_compute_with_ctx(ctx, graph, 4);

        const float * output = ggml_get_data_f32(t_pooled);
        GGML_ASSERT(output[0] == 33);
        GGML_ASSERT(output[1] == 36);
        GGML_ASSERT(output[2] == 39);
        GGML_ASSERT(output[3] == 73);
        GGML_ASSERT(output[4] == 76);
        GGML_ASSERT(output[5] == 79);
        GGML_ASSERT(output[6] == 133);
        GGML_ASSERT(output[7] == 136);
        GGML_ASSERT(output[8] == 139);
        GGML_ASSERT(output[9] == 173);
        GGML_ASSERT(output[10] == 176);
        GGML_ASSERT(output[11] == 179);

        ggml_free(ctx);
    }

    return 0;
}
