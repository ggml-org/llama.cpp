#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <string.h>
#include <stdio.h>
#include <stdlib.h>


static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct ggml_context* make_ctx(void) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 2 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc.  =*/ false
    };
    return ggml_init(params);
}

void check_tensor(struct ggml_tensor * t, float * expected_t_d, int ne0, int ne1, int ne2) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    GGML_ASSERT(t->ne[0] == ne0);
    GGML_ASSERT(t->ne[1] == ne1);
    GGML_ASSERT(t->ne[2] == ne2);
    for (int i2 = 0; i2 < ne2; ++i2) {
        for (int i1 = 0; i1 < ne1; ++i1) {
            for (int i0 = 0; i0 < ne0; ++i0) {
                float expected = *(expected_t_d + i2 * ne1 * ne0 + i1 * ne0 + i0);
                float actual = ggml_get_data_f32(t)[i2 * ne1 * ne0 + i1 * ne0 + i0];
                if (expected != actual) {
                    printf("expected %.1f, got %.1f at (%d,%d,%d)\n", expected, actual, i0, i1, i2);
                }
                GGML_ASSERT(expected == actual);
            }
        }
    }
}

void test_pad_reflect_1d(bool use_gpu) {
    ggml_backend_t backend = NULL;
    struct ggml_init_params params;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
    struct ggml_tallocr tallocr;
    ggml_gallocr_t gallocr;

    // initialize the backend
#ifdef GGML_USE_CUDA
    if (use_gpu) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        backend = ggml_backend_cuda_init(0);
        if (!backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (use_gpu) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        backend = ggml_backend_metal_init();
        if (!backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#endif

    if (!backend) {
        fprintf(stderr, "%s: using CPU backend\n", __func__);
        backend = ggml_backend_cpu_init();
    }

    // Test cases for different padding configurations
    {
        params = ggml_init_params{
            /*.mem_size   =*/ 16*1024*1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc.  =*/ true
        };

        ggml_log_set(ggml_log_callback_default, nullptr);

        ctx = ggml_init(params);
        buffer = ggml_backend_alloc_buffer(backend, 16*1024*1024);
        tallocr = ggml_tallocr_new(buffer);
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        // Create a simple 1D input tensor [1, 2, 3, 4]
        struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        ggml_tallocr_alloc(&tallocr, t);

        // load data to buffer
        if(ggml_backend_is_cpu(backend)) {
            memcpy(t->data, input_data, ggml_nbytes(t));
        } else {
            ggml_backend_tensor_set(t, input_data, 0, ggml_nbytes(t));
        }

        // Test case 1: pad left=1, right=1
        // Expected: [2, 1, 2, 3, 4, 3]
        float expected_1[] = {2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f};
        struct ggml_tensor * out_1 = ggml_pad_reflect_1d(ctx, t, 1, 1);

        // Test case 2: pad left=2, right=1
        // Expected: [3, 2, 1, 2, 3, 4, 3]
        float expected_2[] = {3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f};
        struct ggml_tensor * out_2 = ggml_pad_reflect_1d(ctx, t, 2, 1);

        // Test case 3: pad left=1, right=2
        // Expected: [2, 1, 2, 3, 4, 3, 2]
        float expected_3[] = {2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 2.0f};
        struct ggml_tensor * out_3 = ggml_pad_reflect_1d(ctx, t, 1, 2);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out_1);
        ggml_build_forward_expand(gf, out_2);
        ggml_build_forward_expand(gf, out_3);

        ggml_gallocr_alloc_graph(gallocr, gf);

        ggml_backend_graph_compute(backend, gf);

        check_tensor(out_1, expected_1, 6, 1, 1);
        check_tensor(out_2, expected_2, 7, 1, 1);
        check_tensor(out_3, expected_3, 7, 1, 1);

        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
        ggml_gallocr_free(gallocr);
    }

    {
        params = ggml_init_params{
            /*.mem_size   =*/ 16*1024*1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc.  =*/ true
        };

        ggml_log_set(ggml_log_callback_default, nullptr);

        ctx = ggml_init(params);
        buffer = ggml_backend_alloc_buffer(backend, 16*1024*1024);
        tallocr = ggml_tallocr_new(buffer);
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        // Create a 2D input tensor (5 columns × 4 rows)
        struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 4);
        float input_data[] = {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,  // row 1
            6.0f, 7.0f, 8.0f, 9.0f, 10.0f, // row 2
            11.0f, 12.0f, 13.0f, 14.0f, 15.0f, // row 3
            16.0f, 17.0f, 18.0f, 19.0f, 20.0f  // row 4
        };
        ggml_tallocr_alloc(&tallocr, t);

        // load data to buffer
        if(ggml_backend_is_cpu(backend)) {
            memcpy(t->data, input_data, ggml_nbytes(t));
        } else {
            ggml_backend_tensor_set(t, input_data, 0, ggml_nbytes(t));
        }

        // Test case 4: pad left=3, right=2 on a 2D tensor
        // Each row should be padded independently
        float expected_4[] = {
            4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 4.0f, 3.0f,  // row 1
            9.0f, 8.0f, 7.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 9.0f, 8.0f, // row 2
            14.0f, 13.0f, 12.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 14.0f, 13.0f, // row 3
            19.0f, 18.0f, 17.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 19.0f, 18.0f  // row 4
        };
        struct ggml_tensor * out_4 = ggml_pad_reflect_1d(ctx, t, 3, 2);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out_4);

        ggml_gallocr_alloc_graph(gallocr, gf);

        ggml_backend_graph_compute(backend, gf);

        check_tensor(out_4, expected_4, 10, 4, 1);

        ggml_free(ctx);
        ggml_gallocr_free(gallocr);
        ggml_backend_buffer_free(buffer);
    }

    ggml_backend_free(backend);
}

int main(int argc, const char * argv[]) {
    bool use_gpu = false;
    if (argc > 1) {
        use_gpu = strcmp(argv[1], "--gpu") == 0;
    }
    test_pad_reflect_1d(use_gpu);
    return 0;
}
