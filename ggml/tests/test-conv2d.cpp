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

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct test_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void load_model(test_model & model, bool use_gpu = false) {
    // create data
    int KW = 3, KH = 3, IC = 10, OC = 10;
    int IW = 8, IH = 6, N = 1;

    // Initialize adata
    std::vector<float> adata(KW * KH * IC * OC);
    for (int i = 0; i < KW * KH * IC * OC; i++) {
        adata[i] = 2.5f;
    }

    // Convert adata to fp16 format
    std::vector<ggml_fp16_t> hadata(KW * KH * IC * OC);
    ggml_fp32_to_fp16_row(adata.data(), hadata.data(), KW * KH * IC * OC);

    // Initialize bdata
    std::vector<float> bdata(IW * IH * IC * N);
    for (int i = 0; i < IW * IH * IC * N; i++) {
        bdata[i] = 1.5f;
    }

    size_t buffer_size = 0;
    {
        buffer_size += KW * KH * IC * OC * ggml_type_size(GGML_TYPE_F16); // tensor a
        buffer_size += IW * IH * IC * N  * ggml_type_size(GGML_TYPE_F32); // tensor b
        buffer_size += 1024; // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %0.2f MB\n", __func__, (buffer_size/ 1024.f/ 1024.f));

    int num_tensors = 2;
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    ggml_log_set(ggml_log_callback_default, nullptr);

    // initialize the backend
#ifdef GGML_USE_CUDA
    if (use_gpu) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init(0);
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (use_gpu) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#endif

    if(!model.backend) {
        // fallback to CPU backend
        model.backend = ggml_backend_cpu_init();
    }

    model.buffer = ggml_backend_alloc_buffer(model.backend, buffer_size);

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16,  KW, KH, IC, OC);
    model.b = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32, IW, IH, IC, N);

    // create a allocator
    struct ggml_tallocr alloc = ggml_tallocr_new(model.buffer);

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.a);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.a->data, hadata.data(), ggml_nbytes(model.a));
    } else {
        ggml_backend_tensor_set(model.a, hadata.data(), 0, ggml_nbytes(model.a));
    }

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.b);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.b->data, bdata.data(), ggml_nbytes(model.b));
    } else {
        ggml_backend_tensor_set(model.b, bdata.data(), 0, ggml_nbytes(model.b));
    }
}

struct ggml_cgraph * build_graph(const test_model& model) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);

    int s0 = 1;
    int s1 = 1;
    int p0 = 1;
    int p1 = 1;
    int d0 = 1;
    int d1 = 1;

    // split conv2d in fundamental methods for test unit
    struct ggml_tensor* im2col_0 = ggml_im2col(ctx0, model.a, model.b, s0, s1, p0, p1, d0, d1, true, GGML_TYPE_F16);
    ggml_set_name(im2col_0, "im2col_res");
    ggml_build_forward_expand(gf, im2col_0);

    // recalculate for avoid fragmentation
    struct ggml_tensor* conv2d_res = ggml_conv_2d(ctx0, model.a, model.b, s0, s1, p0, p1, d0, d1);
    ggml_set_name(conv2d_res, "conv2d_res");
    ggml_build_forward_expand(gf, conv2d_res);

    ggml_free(ctx0);
    return gf;
}

struct ggml_cgraph * compute_graph(const test_model & model, ggml_gallocr_t allocr) {
    struct ggml_cgraph * gf = build_graph(model);

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);
    int n_threads = 1;

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    ggml_backend_graph_compute(model.backend, gf);

    //ggml_graph_print(gf);

    return gf;
}

int main(void)
{
    ggml_time_init();

    test_model model;
    load_model(model, true);

    ggml_gallocr_t allocr = NULL;

    {
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        //create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph(model);

        // compute the required memory
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);
        fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0f/1024.0f);
    }

    struct ggml_cgraph * gf_res = compute_graph(model, allocr);

    struct ggml_tensor * im2col_res = NULL;
    struct ggml_tensor * conv2d_res = NULL;

    for(int i = 0; i < ggml_graph_n_nodes(gf_res); ++i) {
        if(strcmp(ggml_get_name(ggml_graph_node(gf_res, i)), "im2col_res") == 0) {
            im2col_res = ggml_graph_node(gf_res, i);
        } else if(strcmp(ggml_get_name(ggml_graph_node(gf_res, i)), "conv2d_res") == 0) {
            conv2d_res = ggml_graph_node(gf_res, i);
        }
    }

    std::vector<uint16_t> im2col_data(ggml_nelements(im2col_res));
    std::vector<float> conv2d_data(ggml_nelements(conv2d_res));

    ggml_backend_tensor_get(im2col_res, im2col_data.data(), 0, ggml_nbytes(im2col_res));
    ggml_backend_tensor_get(conv2d_res, conv2d_data.data(), 0, ggml_nbytes(conv2d_res));

    const int n_conv2d_test = 480;
    const int n_im2col_test = 4320;

    float expected_conv2d [n_conv2d_test] = {
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        150.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 225.00f, 150.00f };

    uint16_t expected_im2col[n_conv2d_test] = {
            0, 0, 0, 0, 15872, 15872, 0, 15872,
            15872, 0, 0, 0, 0, 15872, 15872, 0,
            15872, 15872, 0, 0, 0, 0, 15872, 15872,
            0, 15872, 15872, 0, 0, 0, 0, 15872,
            15872, 0, 15872, 15872, 0, 0, 0, 0,
            15872, 15872, 0, 15872, 15872, 0, 0, 0,
            0, 15872, 15872, 0, 15872, 15872, 0, 0,
            0, 0, 15872, 15872, 0, 15872, 15872, 0,
            0, 0, 0, 15872, 15872, 0, 15872, 15872,
            0, 0, 0, 0, 15872, 15872, 0, 15872,
            15872, 0, 0, 0, 0, 15872, 15872, 0,
            15872, 15872, 0, 0, 0, 15872, 15872, 15872,
            15872, 15872, 15872, 0, 0, 0, 15872, 15872,
            15872, 15872, 15872, 15872, 0, 0, 0, 15872,
            15872, 15872, 15872, 15872, 15872, 0, 0, 0,
            15872, 15872, 15872, 15872, 15872, 15872, 0, 0,
            0, 15872, 15872, 15872, 15872, 15872, 15872, 0,
            0, 0, 15872, 15872, 15872, 15872, 15872, 15872,
            0, 0, 0, 15872, 15872, 15872, 15872, 15872,
            15872, 0, 0, 0, 15872, 15872, 15872, 15872,
            15872, 15872, 0, 0, 0, 15872, 15872, 15872,
            15872, 15872, 15872, 0, 0, 0, 15872, 15872,
            15872, 15872, 15872, 15872, 0, 0, 0, 15872,
            15872, 15872, 15872, 15872, 15872, 0, 0, 0,
            15872, 15872, 15872, 15872, 15872, 15872, 0, 0,
            0, 15872, 15872, 15872, 15872, 15872, 15872, 0,
            0, 0, 15872, 15872, 15872, 15872, 15872, 15872,
            0, 0, 0, 15872, 15872, 15872, 15872, 15872,
            15872, 0, 0, 0, 15872, 15872, 15872, 15872,
            15872, 15872, 0, 0, 0, 15872, 15872, 15872,
            15872, 15872, 15872, 0, 0, 0, 15872, 15872,
            15872, 15872, 15872, 15872, 0, 0, 0, 15872,
            15872, 15872, 15872, 15872, 15872, 0, 0, 0,
            15872, 15872, 15872, 15872, 15872, 15872, 0, 0,
            0, 15872, 15872, 15872, 15872, 15872, 15872, 0,
            0, 0, 15872, 15872, 15872, 15872, 15872, 15872,
            0, 0, 0, 15872, 15872, 15872, 15872, 15872,
            15872, 0, 0, 0, 15872, 15872, 15872, 15872,
            15872, 15872, 0, 0, 0, 15872, 15872, 15872,
            15872, 15872, 15872, 0, 0, 0, 15872, 15872,
            15872, 15872, 15872, 15872, 0, 0, 0, 15872,
            15872, 15872, 15872, 15872, 15872, 0, 0, 0,
            15872, 15872, 15872, 15872, 15872, 15872, 0, 0,
            0, 15872, 15872, 15872, 15872, 15872, 15872, 0,
            0, 0, 15872, 15872, 15872, 15872, 15872, 15872,
            0, 0, 0, 15872, 15872, 15872, 15872, 15872,
            15872, 0, 0, 0, 15872, 15872, 15872, 15872,
            15872, 15872, 0, 0, 0, 15872, 15872, 15872,
            15872, 15872, 15872, 0, 0, 0, 15872, 15872,
            15872, 15872, 15872, 15872, 0, 0, 0, 15872,
            15872, 15872, 15872, 15872, 15872, 0, 0, 0,
            15872, 15872, 15872, 15872, 15872, 15872, 0, 0,
            0, 15872, 15872, 15872, 15872, 15872, 15872, 0,
            0, 0, 15872, 15872, 15872, 15872, 15872, 15872,
            0, 0, 0, 15872, 15872, 15872, 15872, 15872,
            15872, 0, 0, 0, 15872, 15872, 15872, 15872,
            15872, 15872, 0, 0, 0, 15872, 15872, 15872,
            15872, 15872, 15872, 0, 0, 0, 15872, 15872,
            15872, 15872, 15872, 15872, 0, 0, 0, 15872,
            15872, 15872, 15872, 15872, 15872, 0, 0, 0
    };

    printf("\nPerforming test:\n");

    bool passed = true;
    for(int i = 0; i < n_conv2d_test; i++) {
        if(
            im2col_data[i] != expected_im2col[i]) {
            passed = false;
            break;
        }
    }

    printf("ggml_im2col (%d): %s\n", (int) ggml_nelements(im2col_res), passed && (ggml_nelements(im2col_res) == n_im2col_test) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");

    passed = true;
    for(int i = 0; i < n_conv2d_test; i++) {
        if(conv2d_data[i] != expected_conv2d[i]) {
            passed = false;
            break;
        }
    }

    printf("ggml_conv2d (%d): %s\n", (int) ggml_nelements(conv2d_res), passed && (ggml_nelements(conv2d_res) == n_conv2d_test) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");

    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(allocr);
    return 0;
}
