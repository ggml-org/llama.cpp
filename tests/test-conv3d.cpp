#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
//#include <cuda_runtime.h>
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

void load_model(test_model & model, int ic, int oc, int iw, int ih, int id,
                                    int kw, int kh, int kd,
                                    bool use_fp16, bool use_gpu);
struct ggml_cgraph * build_graph_0(const test_model& model, const int64_t ic, const int64_t n, const int64_t oc);
struct ggml_cgraph * build_graph_1(const test_model& model, const int64_t ic, const int64_t n, const int64_t oc);
typedef struct ggml_cgraph* (*build_graph_t)(const test_model& model,
     const int64_t i0,  const int64_t i1, const int64_t i2);

std::vector<float> compute_graph(const test_model & model, ggml_gallocr_t allocr,
            build_graph_t build_graph, int iters,
            const int64_t ic, const int64_t n, const int64_t oc, double *t);


void load_model(test_model & model, int ic, int oc, int iw, int ih, int id,
                                    int kw = 3, int kh = 3, int kd = 3,
                                    bool use_fp16 = true, bool use_gpu = false ) {
    // create data
    int KW = kw, KH = kh, KD = kd;
    int IC = ic, OC = oc;
    int IW = iw, IH = ih, ID = id, N = 1;
    srand(time(NULL));

    // printf(" input: IC = %d, OC = %d, IW = %d, IH = %d \n ", IC, OC, IW, IH);

    // Initialize adata
    std::vector<float> adata(KW * KH * KD * IC * OC);
    for (int i = 0; i < KW * KH * KD * IC * OC; i++) {
        // adata[i] = 2.f;
        // adata[i] = (float)(i%KW)-1.f;
        // adata[i] = (rand() % 255) / 255.0;
        float r = -1.f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1.f-(-1.f))));
        adata[i] = r;
    }

    // Convert adata to fp16 format
    std::vector<ggml_fp16_t> hadata(KW * KH * KD * IC * OC);
    ggml_fp32_to_fp16_row(adata.data(), hadata.data(), KW * KH * KD * IC * OC);

    // Initialize bdata
    std::vector<float> bdata(IW * IH * ID * IC * N);
    for (int i = 0; i < IW * IH * ID * IC * N; i++) {
        // bdata[i] = (float)(i%IW)/10.f;
        // bdata[i] = 1.5f;
        // bdata[i] = (rand() % 255) / 255.0;
        float r = -1.f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1.f-(-1.f))));
        bdata[i] = r;
    }

    size_t buffer_size = 0;
    {   if(use_fp16)
            buffer_size += KW * KH * KD * IC * OC * ggml_type_size(GGML_TYPE_F16); // tensor a
        else
            buffer_size += KW * KH * KD * IC * OC * ggml_type_size(GGML_TYPE_F32); // tensor a
        buffer_size += IW * IH * ID * IC * N  * ggml_type_size(GGML_TYPE_F32); // tensor b
        buffer_size += 1024; // overhead
    }

    // printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    // printf("%s: backend buffer size = %0.2f MB\n", __func__, (buffer_size/ 1024.f/ 1024.f));

    int num_tensors = 2;
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    // initialize the backend
#ifdef GGML_USE_CUDA
    if (use_gpu) {
        // fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init(0);
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#else
    GGML_UNUSED(use_gpu);
#endif

#ifdef GGML_USE_METAL
    if (use_gpu) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#else
    GGML_UNUSED(use_gpu);
#endif

    if(!model.backend) {
        // fallback to CPU backend
        model.backend = ggml_backend_cpu_init();
    }

    model.buffer = ggml_backend_alloc_buffer(model.backend, buffer_size);

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    if(use_fp16)
        model.a = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16,  KW, KH, KD, IC*OC);
    else
        model.a = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32,  KW, KH, KD, IC*OC);
    model.b = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32, IW, IH, ID, IC*N);

    // create a allocator
    struct ggml_tallocr alloc = ggml_tallocr_new(model.buffer);

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.a);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        if(use_fp16)
            memcpy(model.a->data, hadata.data(), ggml_nbytes(model.a));
        else
            memcpy(model.a->data, adata.data(), ggml_nbytes(model.a));
    } else {
        if(use_fp16)
            ggml_backend_tensor_set(model.a, hadata.data(), 0, ggml_nbytes(model.a));
        else
            ggml_backend_tensor_set(model.a,  adata.data(), 0, ggml_nbytes(model.a));
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

struct ggml_cgraph * build_graph_0(const test_model& model, const int64_t ic, const int64_t n, const int64_t oc) {

    GGML_UNUSED(n);
    GGML_UNUSED(oc);

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

    // int s0 = 2;
    // int s1 = 1;
    // int s2 = 1;
    // int p0 = 2;
    // int p1 = 0;
    // int p2 = 1;
    // int d0 = 1;
    // int d1 = 1;
    // int d2 = 2;

    int s0 = 1;
    int s1 = 1;
    int s2 = 1;
    int p0 = 1;
    int p1 = 1;
    int p2 = 1;

    int d0 = 1;
    int d1 = 1;
    int d2 = 1;

    // recalculate for avoid fragmentation
    struct ggml_tensor* conv2d_res = ggml_conv_3d(ctx0, model.a, model.b, ic, s0, s1, s2, p0, p1, p2, d0, d1, d2);
    ggml_set_name(conv2d_res, "conv2d_res");
    ggml_build_forward_expand(gf, conv2d_res);
    // int64_t *ne = conv2d_res->ne;
    // printf("conv2d: (%zu, %zu, %zu, %zu) \n", ne[0], ne[1], ne[2], ne[3]);


    // struct ggml_tensor* wino_res = ggml_conv_2d_3x3(ctx0, model.a, model.b);
    // ggml_set_name(wino_res, "wino_res");
    // ggml_build_forward_expand(gf, wino_res);
    // ne = wino_res->ne;
    // printf("wino: (%zu, %zu, %zu, %zu) \n", ne[0], ne[1], ne[2], ne[3]);
    ggml_free(ctx0);
    return gf;
}

struct ggml_cgraph * build_graph_1(const test_model& model, const int64_t ic, const int64_t n, const int64_t oc) {
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
    int s2 = 1;
    int p0 = 1;
    int p1 = 1;
    int p2 = 1;
    int d0 = 1;
    int d1 = 1;
    int d2 = 1;

    // int s0 = 2;
    // int s1 = 1;
    // int s2 = 1;
    // int p0 = 2;
    // int p1 = 0;
    // int p2 = 1;
    // int d0 = 1;
    // int d1 = 1;
    // int d2 = 2;

    // recalculate for avoid fragmentation
    // struct ggml_tensor* conv2d_res = ggml_conv_2d(ctx0, model.a, model.b, s0, s1, p0, p1, d0, d1);
    // ggml_set_name(conv2d_res, "conv2d_res");
    // ggml_build_forward_expand(gf, conv2d_res);
    // int64_t *ne = conv2d_res->ne;
    // printf("conv2d: (%zu, %zu, %zu, %zu) \n", ne[0], ne[1], ne[2], ne[3]);


    // struct ggml_tensor* wino_res = ggml_conv_2d_implicitgemm(ctx0, model.a, model.b, s0, s1, p0, p1, d0, d1);
    struct ggml_tensor* wino_res = ggml_conv_3d_direct(ctx0, model.a, model.b,
                                       s0, s1, s2, p0, p1, p2, d0, d1, d2,
                                       ic, n, oc);
    ggml_set_name(wino_res, "wino_res");
    ggml_build_forward_expand(gf, wino_res);
    // int64_t *ne = wino_res->ne;
    // printf("wino: (%zu, %zu, %zu, %zu) \n", ne[0], ne[1], ne[2], ne[3]);
    ggml_free(ctx0);
    return gf;
}




std::vector<float> compute_graph(const test_model & model, ggml_gallocr_t allocr,
            build_graph_t build_graph, int iters,
            const int64_t ic, const int64_t n, const int64_t oc, double *t) {

    struct ggml_cgraph * gf = build_graph(model, ic, n, oc);


    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);
    int n_threads = 1;

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    ggml_backend_graph_compute(model.backend, gf);

    ggml_backend_synchronize(model.backend);

    int64_t start_time = ggml_time_us();

    for(int iter=0; iter<iters; iter++){
        ggml_backend_graph_compute(model.backend, gf);
        ggml_backend_synchronize(model.backend);
    }

    // ggml_backend_synchronize(model.backend);
    int64_t end_time = ggml_time_us();
    double time_us = end_time - start_time;

    time_us = time_us/iters;
    //ggml_graph_print(gf);

    struct ggml_tensor *res = NULL;

    for(int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if(strcmp(ggml_get_name(ggml_graph_node(gf, i)), "wino_res") == 0) {
            res = ggml_graph_node(gf, i);
        } else if(strcmp(ggml_get_name(ggml_graph_node(gf, i)), "conv2d_res") == 0) {
            res = ggml_graph_node(gf, i);
        }
    }

    std::vector<float> data(ggml_nelements(res));
    ggml_backend_tensor_get(res, data.data(), 0, ggml_nbytes(res));

    *t = time_us/1000;
    return data;

}


int main(void)
{
    ggml_time_init();
    std::vector<std::tuple<int, int, int, int, int, int, int, int>> configs = {
        // std::make_tuple(1,2,16,32,4,3,3,3),
        // std::make_tuple(320,1280,26,38,8,3,3,3),
        // std::make_tuple(1280,1280,26,38,8,3,3,3),
        // std::make_tuple(320,1280,52,76,8,3,3,3),
        // std::make_tuple(1280,1280,52,76,8,3,3,3),
        // std::make_tuple(320,1280,104,152,8,3,3,3),
        // std::make_tuple(1280,1280,104,152,8,3,3,3),
        // std::make_tuple(320,1280,208,304,4,3,3,3),
        // std::make_tuple(1024,2048,30,52,3,3,3,3),
        // std::make_tuple(1024,2048,52,76,4,3,3,3),
        // std::make_tuple(1024,2048,52,76,6,3,3,3),
        // std::make_tuple(48,3072,64,64,9,2,2,1),
        // std::make_tuple(48,3072,64,64,17,2,2,1),
        // std::make_tuple(48,3072,64,64,33,2,2,1),
        std::make_tuple(320,320,104,158,8,3,3,3),
    };

    int k = 0;

    for (auto c : configs){
        test_model model;
        load_model(model, std::get<0>(c), std::get<1>(c), std::get<2>(c),
            std::get<3>(c), std::get<4>(c), std::get<5>(c), std::get<6>(c), std::get<7>(c), true, true);

        ggml_gallocr_t allocr = NULL;
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        //create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph_0(model, std::get<0>(c), 0, 0);

        // compute the required memory
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size0 = ggml_gallocr_get_buffer_size(allocr, 0);
        // fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0f/1024.0f);


        int iterations = 20;

        double run_time0;
        std::vector<float> im2col_data = compute_graph(model, allocr, build_graph_0, iterations,
            std::get<0>(c), 1, std::get<1>(c), &run_time0);

        ggml_gallocr_free(allocr);

        allocr = NULL;

        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        //create the worst case graph for memory usage estimation
        gf = build_graph_1(model, std::get<0>(c), 1, std::get<1>(c));

        // compute the required memory
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size1 = ggml_gallocr_get_buffer_size(allocr, 0);

        double run_time1;
        std::vector<float> conv2d_data = compute_graph(model, allocr, build_graph_1, iterations,
            std::get<0>(c), 1, std::get<1>(c), &run_time1);

        if(k==0) {
            k = 1;
            fprintf(stderr, "| (IC, OC, IW, IH, ID, KW, KH, KD) | im2col+GEMM TIME | im2col+GEMM VRAM | implicit GEMM TIME | implicit GEMM VRAM \n");
            fprintf(stderr, "| --- | --- | --- | --- | --- \n");
        }

        fprintf(stderr, " | (%d, %d, %d, %d, %d, %d, %d, %d) | %.2f ms | %.2f MB | %.2f ms | %.2f MB\n",
                std::get<0>(c), std::get<1>(c), std::get<2>(c),
                std::get<3>(c), std::get<4>(c), std::get<5>(c),
                std::get<6>(c), std::get<7>(c),
                run_time0, mem_size0/1024.0f/1024.0f,
                run_time1, mem_size1/1024.0f/1024.0f);


        // for(int i = 0; i < conv2d_data.size(); i++) {
        //     float diff = fabs(im2col_data[i] - conv2d_data[i]);
        //     // if(diff > 0.5) {
        //         printf("(%7.3f, %7.3f, %f, %d) \n",
        //           im2col_data[i], conv2d_data[i],
        //            diff,  i);
        //         // break;
        //     // }
        // }

        ggml_free(model.ctx);
        ggml_backend_buffer_free(model.buffer);
        ggml_backend_free(model.backend);
        ggml_gallocr_free(allocr);

    }
    return 0;
}
