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
    struct ggml_tensor * weight;
    struct ggml_tensor * input;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void load_model(test_model & model, bool use_gpu = false) {
    // create data
    int K = 3, IC = 2, OC = 2;
    int IL = 6, N = 1;

    // Initialize adata
    float weight_data[6] = {10.0f, 20.0f, 30.0f, 0.1f, 0.2f, 0.3f};

    // Convert adata to fp16 format
    std::vector<ggml_fp16_t> h_weight_data(K * IC);
    ggml_fp32_to_fp16_row(weight_data, h_weight_data.data(), K * IC);

    // Initialize input data, 2 channels, 6 timesteps, 1 batch
    float input_data[12] = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    };

    size_t buffer_size = 0;
    {
        buffer_size += K * IC * ggml_type_size(GGML_TYPE_F16); // tensor weight
        buffer_size += IL * IC * N * ggml_type_size(GGML_TYPE_F32); // tensor input
        buffer_size += 1024; // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %0.2f MB\n", __func__, (buffer_size/ 1024.f/ 1024.f));

    ggml_log_set(ggml_log_callback_default, nullptr);

    int num_tensors = 2;
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

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
    // A Pytorch grouped Conv1d weight parameter is of shape (out_channels, input_channels/groups, kernel_size)
    model.weight = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F16,  K, 1, IC);
    model.input = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, IL, IC, N);

    // create a allocator
    ggml_tallocr alloc = ggml_tallocr_new(model.buffer);

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.weight);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.weight->data, h_weight_data.data(), ggml_nbytes(model.weight));
    } else {
        ggml_backend_tensor_set(model.weight, h_weight_data.data(), 0, ggml_nbytes(model.weight));
    }

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.input);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.input->data, input_data, ggml_nbytes(model.input));
    } else {
        ggml_backend_tensor_set(model.input, input_data, 0, ggml_nbytes(model.input));
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
    int p0 = 1;
    int d0 = 1;

    struct ggml_tensor* conv1d_dw_res = ggml_conv_1d_dw(ctx0, model.weight, model.input, s0, p0, d0);
    ggml_set_name(conv1d_dw_res, "conv1d_dw_res");
    ggml_build_forward_expand(gf, conv1d_dw_res);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

struct ggml_cgraph* compute_graph(const test_model & model, ggml_gallocr_t allocr) {
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

    struct ggml_tensor * conv1d_dw_res = NULL;

    for(int i = 0; i < ggml_graph_n_nodes(gf_res); i++) {
        if(strcmp(ggml_get_name(ggml_graph_node(gf_res, i)), "conv1d_dw_res") == 0) {
            conv1d_dw_res = ggml_graph_node(gf_res, i);
        }
    }

    std::vector<float> conv2d_data(ggml_nelements(conv1d_dw_res));

    ggml_backend_tensor_get(conv1d_dw_res, conv2d_data.data(), 0, ggml_nbytes(conv1d_dw_res));

    const int n_conv1d_dw_test = 12;

    float expected_conv1d_dw[n_conv1d_dw_test] = {
        50.0f, 60.0f, 60.0f, 60.0f, 60.0f, 30.0f, 0.50f,  0.60f,  0.60f,  0.60f,  0.60f,  0.30f
    };

    printf("\nPerforming test:\n");

    bool passed = true;
    passed = true;
    for(int i = 0; i < n_conv1d_dw_test; i++) {
        if(std::abs(conv2d_data[i] - expected_conv1d_dw[i]) > 1e-4) {
            passed = false;
            break;
        }
    }

    printf("ggml_conv1d (%d): %s\n", (int) ggml_nelements(conv1d_dw_res), passed && (ggml_nelements(conv1d_dw_res) == n_conv1d_dw_test) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");
    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(allocr);
    return 0;
}
