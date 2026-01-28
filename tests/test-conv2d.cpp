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

void load_model(test_model &, int,  int, int, int, int, int, bool);
struct ggml_cgraph * build_graph_0(const test_model&);
struct ggml_cgraph * build_graph_1(const test_model&);

void load_model(test_model & model, int ic, int oc, int iw, int ih, int kw = 3, int kh = 3, bool use_gpu = false ) {
    // create data
    int KW = kw, KH = kh, IC = ic, OC = oc;
    int IW = iw, IH = ih, N = 2;
    // srand(time(NULL));

    // printf(" input: IC = %d, OC = %d, IW = %d, IH = %d \n ", IC, OC, IW, IH);

    // Initialize adata
    std::vector<float> adata(KW * KH * IC * OC);
    for (int i = 0; i < KW * KH * IC * OC; i++) {
        // adata[i] = 2.f;
        // adata[i] = (float)(i%KW)-1.f;
        // adata[i] = (float)((i+1)%KW+1)/10.0;
        // adata[i] = (float)(i%100);
        // adata[i] = (rand() % 255) / 255.0;
        float r = -1.f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1.f-(-1.f))));
        adata[i] = r;
    }

    // Convert adata to fp16 format
    std::vector<ggml_fp16_t> hadata(KW * KH * IC * OC);
    ggml_fp32_to_fp16_row(adata.data(), hadata.data(), KW * KH * IC * OC);

    // Initialize bdata
    std::vector<float> bdata(IW * IH * IC * N);
    for (int i = 0; i < IW * IH * IC * N; i++) {
        // bdata[i] = (float)(i%IW)/10.f;
        // bdata[i] = 1.5f;
        // bdata[i] = (rand() % 255) / 255.0;
        float r = -1.f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1.f-(-1.f))));
        bdata[i] = r;
    }

    size_t buffer_size = 0;
    {
        // buffer_size += KW * KH * IC * OC * ggml_type_size(GGML_TYPE_F32); // tensor a
        buffer_size += KW * KH * IC * OC * ggml_type_size(GGML_TYPE_F16); // tensor a
        buffer_size += IW * IH * IC * N  * ggml_type_size(GGML_TYPE_F32); // tensor b
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
    model.a = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16,  KW, KH, IC, OC);
    // model.a = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32,  KW, KH, IC, OC);
    model.b = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32, IW, IH, IC, N);

    // create a allocator
    struct ggml_tallocr alloc = ggml_tallocr_new(model.buffer);

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.a);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.a->data, hadata.data(), ggml_nbytes(model.a));
        // memcpy(model.a->data, adata.data(), ggml_nbytes(model.a));
    } else {
        ggml_backend_tensor_set(model.a, hadata.data(), 0, ggml_nbytes(model.a));
        // ggml_backend_tensor_set(model.a, adata.data(), 0, ggml_nbytes(model.a));
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

typedef struct ggml_cgraph* (*build_graph_t)(const test_model& model);

struct ggml_cgraph * build_graph_0(const test_model& model) {
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

    // int s0 = 1;
    // int s1 = 1;
    // int p0 = 1;
    // int p1 = 1;
    // int d0 = 1;
    // int d1 = 1;

    int s0 = 1;
    int s1 = 5;
    int p0 = 5;
    int p1 = 2;
    int d0 = 2;
    int d1 = 4;

    // recalculate for avoid fragmentation
    struct ggml_tensor* conv2d_res = ggml_conv_2d(ctx0, model.a, model.b, s0, s1, p0, p1, d0, d1);
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

struct ggml_cgraph * build_graph_1(const test_model& model) {
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

    // int s0 = 1;
    // int s1 = 1;
    // int p0 = 1;
    // int p1 = 1;
    // int d0 = 1;
    // int d1 = 1;


    // int s0 = 3;
    // int s1 = 5;
    // int p0 = 5;
    // int p1 = 5;
    // int d0 = 2;
    // int d1 = 4;

    int s0 = 1;
    int s1 = 5;
    int p0 = 5;
    int p1 = 2;
    int d0 = 2;
    int d1 = 4;


    // recalculate for avoid fragmentation
    // struct ggml_tensor* conv2d_res = ggml_conv_2d(ctx0, model.a, model.b, s0, s1, p0, p1, d0, d1);
    // ggml_set_name(conv2d_res, "conv2d_res");
    // ggml_build_forward_expand(gf, conv2d_res);
    // int64_t *ne = conv2d_res->ne;
    // printf("conv2d: (%zu, %zu, %zu, %zu) \n", ne[0], ne[1], ne[2], ne[3]);


    // struct ggml_tensor* wino_res = ggml_conv_2d_implicitgemm(ctx0, model.a, model.b, s0, s1, p0, p1, d0, d1);
    struct ggml_tensor* wino_res = ggml_conv_2d_direct(ctx0, model.a, model.b, s0, s1, p0, p1, d0, d1);
    ggml_set_name(wino_res, "wino_res");
    ggml_build_forward_expand(gf, wino_res);
    // ne = wino_res->ne;
    // printf("wino: (%zu, %zu, %zu, %zu) \n", ne[0], ne[1], ne[2], ne[3]);
    ggml_free(ctx0);
    return gf;
}

std::vector<float> compute_graph(const test_model &, ggml_gallocr_t,
            build_graph_t, int, double *);


std::vector<float> compute_graph(const test_model & model, ggml_gallocr_t allocr,
            build_graph_t build_graph, int iters, double *t) {
    struct ggml_cgraph * gf = build_graph(model);


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

static std::vector<std::tuple<int, int, int, int, int, int>> configs = {
        // std::make_tuple(64,64,48,64,3,3),
        // std::make_tuple(320,320,104,152,3,3),
        // std::make_tuple(640,640,52,76,3,3),
        // std::make_tuple(640,640,104,152,3,3),
        // std::make_tuple(960,320,104,152,3,3),
        // std::make_tuple(1280,1280,26,38,3,3),
        // std::make_tuple(1920,640,32,32,3,3)
        // std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(1,1,1,133,1,1),
        // std::make_tuple(32,12,141,133,3,3),
        // std::make_tuple(32,6,141,133,3,3),
        // std::make_tuple(32,12,141,121,3,3),
        // std::make_tuple(32,9,141,121,3,3),
        // std::make_tuple(320,8,16,16,3,3),  //working
        // std::make_tuple(320,9,16,16,3,3),  //working
        // std::make_tuple(320,12,16,16,3,3),  //working
        //  std::make_tuple(256,12,16,16,3,3),  //working
        //  std::make_tuple(32,12,16,16,3,3),  //not working
        // std::make_tuple(16,12,16,16,3,3),  //not working
        // std::make_tuple(32,12,16,16,3,3),  //not working
        // std::make_tuple(48,12,16,16,3,3),  // not working
        // std::make_tuple(96,12,16,16,3,3),  //not working
        // std::make_tuple(64,12,16,16,3,3),  //working
        // std::make_tuple(64,12,141,133,3,3),  //working
        // std::make_tuple(32,12,141,133,3,3),  //working
        // std::make_tuple(1280,1280,16,16,3,3),
        // std::make_tuple(32,8,24,24,3,3),
        // std::make_tuple(640,640,64,64,3,3),
        // std::make_tuple(320,640,32,32,3,3),
        // std::make_tuple(4,320,96,128,3,3),
        // std::make_tuple(320,4,96,128,3,3),
        // std::make_tuple(4,320,64,96,3,3),
        // std::make_tuple(320,4,64,96,3,3),
        // std::make_tuple(640,640,96,128,3,3),
        // std::make_tuple(1280,1280,26,38,1,1),
        // std::make_tuple(256,128,768,1024,3,3),
        // std::make_tuple(128,3,768,1024,3,3),
        // std::make_tuple(256,128,768,1024,1,1),
        // std::make_tuple(512,256,384,512,1,1),
        // std::make_tuple(1280,640,52,76,3,3),
        // std::make_tuple(1920,1280,26,38,3,3),
        // std::make_tuple(2560,1280,26,38,3,3),
        // std::make_tuple(320,1280,26,38,3,3),
        // std::make_tuple(512,512,104,152,3,3),
        // std::make_tuple(512,512,208,304,3,3),
        // std::make_tuple(512,256,416,608,3,3),
        // std::make_tuple(256,128,832,1216,3,3),
        // std::make_tuple(256,256,832,1216,3,3),
        // std::make_tuple(32,64,58,58,3,3)
        // std::make_tuple(320,256,1024,1920)
    };

static std::vector<std::tuple<int, int, int, int, int, int>> configs_sdxl_512 = {
        //512x512
        std::make_tuple(4,320,64,64,3,3),
        std::make_tuple(320,320,64,64,3,3),
        std::make_tuple(320,320,64,64,3,3),
        std::make_tuple(320,320,64,64,3,3),
        std::make_tuple(320,320,64,64,3,3),
        std::make_tuple(320,320,64,64,3,3),
        std::make_tuple(320,640,32,32,3,3),
        std::make_tuple(640,640,32,32,3,3),
        std::make_tuple(320,640,32,32,3,3),
        std::make_tuple(640,640,32,32,3,3),
        std::make_tuple(640,640,32,32,3,3),
        std::make_tuple(640,640,32,32,3,3),
        std::make_tuple(640,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(640,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(2560,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(2560,1280,16,16,3,3),
        std::make_tuple(2560,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(2560,1280,16,16,3,3),
        std::make_tuple(1920,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(1920,1280,16,16,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(1920,640,32,32,3,3),
        std::make_tuple(640,640,32,32,3,3),
        std::make_tuple(1920,640,32,32,3,3),
        std::make_tuple(1280,640,32,32,3,3),
        std::make_tuple(640,640,32,32,3,3),
        std::make_tuple(1280,640,32,32,3,3),
        std::make_tuple(960,640,32,32,3,3),
        std::make_tuple(640,640,32,32,3,3),
        std::make_tuple(960,640,32,32,3,3),
        std::make_tuple(640,640,64,64,3,3),
        std::make_tuple(960,320,64,64,3,3),
        std::make_tuple(320,320,64,64,3,3),
        std::make_tuple(960,320,64,64,3,3),
        std::make_tuple(640,320,64,64,3,3),
        std::make_tuple(320,320,64,64,3,3),
        std::make_tuple(640,320,64,64,3,3),
        std::make_tuple(640,320,64,64,3,3),
        std::make_tuple(320,320,64,64,3,3),
        std::make_tuple(640,320,64,64,3,3),
        std::make_tuple(320,4,64,64,3,3),
        std::make_tuple(4,320,64,64,3,3),
        std::make_tuple(320,320,64,64,3,3),
        std::make_tuple(320,320,64,64,3,3),
        std::make_tuple(320,320,64,64,3,3),
        std::make_tuple(320,320,64,64,3,3),
        std::make_tuple(320,320,64,64,3,3),
        std::make_tuple(320,640,32,32,3,3),
        std::make_tuple(640,640,32,32,3,3),
        std::make_tuple(320,640,32,32,3,3),
        std::make_tuple(640,640,32,32,3,3),
        std::make_tuple(640,640,32,32,3,3),
        std::make_tuple(640,640,32,32,3,3),
        std::make_tuple(640,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(640,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(2560,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(2560,1280,16,16,3,3),
        std::make_tuple(2560,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(2560,1280,16,16,3,3),
        std::make_tuple(1920,1280,16,16,3,3),
        std::make_tuple(1280,1280,16,16,3,3),
        std::make_tuple(1920,1280,16,16,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(1920,640,32,32,3,3),
        std::make_tuple(640,640,32,32,3,3),
        std::make_tuple(1920,640,32,32,3,3),
        std::make_tuple(1280,640,32,32,3,3),
        std::make_tuple(640,640,32,32,3,3),
        std::make_tuple(1280,640,32,32,3,3),
        std::make_tuple(960,640,32,32,3,3),
        std::make_tuple(640,640,32,32,3,3),
        std::make_tuple(960,640,32,32,3,3),
        std::make_tuple(640,640,64,64,3,3),
        std::make_tuple(960,320,64,64,3,3),
        std::make_tuple(320,320,64,64,3,3),
        std::make_tuple(960,320,64,64,3,3),
        std::make_tuple(640,320,64,64,3,3),
        std::make_tuple(320,320,64,64,3,3),
        std::make_tuple(640,320,64,64,3,3),
        std::make_tuple(640,320,64,64,3,3),
        std::make_tuple(320,320,64,64,3,3),
        std::make_tuple(640,320,64,64,3,3),
        std::make_tuple(320,4,64,64,3,3)
        };

static std::vector<std::tuple<int, int, int, int, int, int>> configs_sdxl_768 = {
        //768x768
        std::make_tuple(4,320,96,96,3,3),
        std::make_tuple(320,320,96,96,3,3),
        std::make_tuple(320,320,96,96,3,3),
        std::make_tuple(320,320,96,96,3,3),
        std::make_tuple(320,320,96,96,3,3),
        std::make_tuple(320,320,96,96,3,3),
        std::make_tuple(320,640,48,48,3,3),
        std::make_tuple(640,640,48,48,3,3),
        std::make_tuple(320,640,48,48,3,3),
        std::make_tuple(640,640,48,48,3,3),
        std::make_tuple(640,640,48,48,3,3),
        std::make_tuple(640,640,48,48,3,3),
        std::make_tuple(640,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(640,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(2560,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(2560,1280,24,24,3,3),
        std::make_tuple(2560,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(2560,1280,24,24,3,3),
        std::make_tuple(1920,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(1920,1280,24,24,3,3),
        std::make_tuple(1280,1280,48,48,3,3),
        std::make_tuple(1920,640,48,48,3,3),
        std::make_tuple(640,640,48,48,3,3),
        std::make_tuple(1920,640,48,48,3,3),
        std::make_tuple(1280,640,48,48,3,3),
        std::make_tuple(640,640,48,48,3,3),
        std::make_tuple(1280,640,48,48,3,3),
        std::make_tuple(960,640,48,48,3,3),
        std::make_tuple(640,640,48,48,3,3),
        std::make_tuple(960,640,48,48,3,3),
        std::make_tuple(640,640,96,96,3,3),
        std::make_tuple(960,320,96,96,3,3),
        std::make_tuple(320,320,96,96,3,3),
        std::make_tuple(960,320,96,96,3,3),
        std::make_tuple(640,320,96,96,3,3),
        std::make_tuple(320,320,96,96,3,3),
        std::make_tuple(640,320,96,96,3,3),
        std::make_tuple(640,320,96,96,3,3),
        std::make_tuple(320,320,96,96,3,3),
        std::make_tuple(640,320,96,96,3,3),
        std::make_tuple(320,4,96,96,3,3),
        std::make_tuple(4,320,96,96,3,3),
        std::make_tuple(320,320,96,96,3,3),
        std::make_tuple(320,320,96,96,3,3),
        std::make_tuple(320,320,96,96,3,3),
        std::make_tuple(320,320,96,96,3,3),
        std::make_tuple(320,320,96,96,3,3),
        std::make_tuple(320,640,48,48,3,3),
        std::make_tuple(640,640,48,48,3,3),
        std::make_tuple(320,640,48,48,3,3),
        std::make_tuple(640,640,48,48,3,3),
        std::make_tuple(640,640,48,48,3,3),
        std::make_tuple(640,640,48,48,3,3),
        std::make_tuple(640,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(640,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(2560,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(2560,1280,24,24,3,3),
        std::make_tuple(2560,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(2560,1280,24,24,3,3),
        std::make_tuple(1920,1280,24,24,3,3),
        std::make_tuple(1280,1280,24,24,3,3),
        std::make_tuple(1920,1280,24,24,3,3),
        std::make_tuple(1280,1280,48,48,3,3),
        std::make_tuple(1920,640,48,48,3,3),
        std::make_tuple(640,640,48,48,3,3),
        std::make_tuple(1920,640,48,48,3,3),
        std::make_tuple(1280,640,48,48,3,3),
        std::make_tuple(640,640,48,48,3,3),
        std::make_tuple(1280,640,48,48,3,3),
        std::make_tuple(960,640,48,48,3,3),
        std::make_tuple(640,640,48,48,3,3),
        std::make_tuple(960,640,48,48,3,3),
        std::make_tuple(640,640,96,96,3,3),
        std::make_tuple(960,320,96,96,3,3),
        std::make_tuple(320,320,96,96,3,3),
        std::make_tuple(960,320,96,96,3,3),
        std::make_tuple(640,320,96,96,3,3),
        std::make_tuple(320,320,96,96,3,3),
        std::make_tuple(640,320,96,96,3,3),
        std::make_tuple(640,320,96,96,3,3),
        std::make_tuple(320,320,96,96,3,3),
        std::make_tuple(640,320,96,96,3,3),
        std::make_tuple(320,4,96,96,3,3),
        };

static std::vector<std::tuple<int, int, int, int, int, int>> configs_sdxl_1024 = {
        //1024x1024
        std::make_tuple(4,320,128,128,3,3),
        std::make_tuple(320,320,128,128,3,3),
        std::make_tuple(320,320,128,128,3,3),
        std::make_tuple(320,320,128,128,3,3),
        std::make_tuple(320,320,128,128,3,3),
        std::make_tuple(320,320,128,128,3,3),
        std::make_tuple(320,640,64,64,3,3),
        std::make_tuple(640,640,64,64,3,3),
        std::make_tuple(320,640,64,64,3,3),
        std::make_tuple(640,640,64,64,3,3),
        std::make_tuple(640,640,64,64,3,3),
        std::make_tuple(640,640,64,64,3,3),
        std::make_tuple(640,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(640,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(2560,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(2560,1280,32,32,3,3),
        std::make_tuple(2560,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(2560,1280,32,32,3,3),
        std::make_tuple(1920,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(1920,1280,32,32,3,3),
        std::make_tuple(1280,1280,64,64,3,3),
        std::make_tuple(1920,640,64,64,3,3),
        std::make_tuple(640,640,64,64,3,3),
        std::make_tuple(1920,640,64,64,3,3),
        std::make_tuple(1280,640,64,64,3,3),
        std::make_tuple(640,640,64,64,3,3),
        std::make_tuple(1280,640,64,64,3,3),
        std::make_tuple(960,640,64,64,3,3),
        std::make_tuple(640,640,64,64,3,3),
        std::make_tuple(960,640,64,64,3,3),
        std::make_tuple(640,640,128,128,3,3),
        std::make_tuple(960,320,128,128,3,3),
        std::make_tuple(320,320,128,128,3,3),
        std::make_tuple(960,320,128,128,3,3),
        std::make_tuple(640,320,128,128,3,3),
        std::make_tuple(320,320,128,128,3,3),
        std::make_tuple(640,320,128,128,3,3),
        std::make_tuple(640,320,128,128,3,3),
        std::make_tuple(320,320,128,128,3,3),
        std::make_tuple(640,320,128,128,3,3),
        std::make_tuple(320,4,128,128,3,3),
        std::make_tuple(4,320,128,128,3,3),
        std::make_tuple(320,320,128,128,3,3),
        std::make_tuple(320,320,128,128,3,3),
        std::make_tuple(320,320,128,128,3,3),
        std::make_tuple(320,320,128,128,3,3),
        std::make_tuple(320,320,128,128,3,3),
        std::make_tuple(320,640,64,64,3,3),
        std::make_tuple(640,640,64,64,3,3),
        std::make_tuple(320,640,64,64,3,3),
        std::make_tuple(640,640,64,64,3,3),
        std::make_tuple(640,640,64,64,3,3),
        std::make_tuple(640,640,64,64,3,3),
        std::make_tuple(640,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(640,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(2560,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(2560,1280,32,32,3,3),
        std::make_tuple(2560,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(2560,1280,32,32,3,3),
        std::make_tuple(1920,1280,32,32,3,3),
        std::make_tuple(1280,1280,32,32,3,3),
        std::make_tuple(1920,1280,32,32,3,3),
        std::make_tuple(1280,1280,64,64,3,3),
        std::make_tuple(1920,640,64,64,3,3),
        std::make_tuple(640,640,64,64,3,3),
        std::make_tuple(1920,640,64,64,3,3),
        std::make_tuple(1280,640,64,64,3,3),
        std::make_tuple(640,640,64,64,3,3),
        std::make_tuple(1280,640,64,64,3,3),
        std::make_tuple(960,640,64,64,3,3),
        std::make_tuple(640,640,64,64,3,3),
        std::make_tuple(960,640,64,64,3,3),
        std::make_tuple(640,640,128,128,3,3),
        std::make_tuple(960,320,128,128,3,3),
        std::make_tuple(320,320,128,128,3,3),
        std::make_tuple(960,320,128,128,3,3),
        std::make_tuple(640,320,128,128,3,3),
        std::make_tuple(320,320,128,128,3,3),
        std::make_tuple(640,320,128,128,3,3),
        std::make_tuple(640,320,128,128,3,3),
        std::make_tuple(320,320,128,128,3,3),
        std::make_tuple(640,320,128,128,3,3),
        std::make_tuple(320,4,128,128,3,3)
    };


int main(void)
{
    ggml_time_init();

    double time_iter0 = 0.0, time_iter1 = 0.0;

    int k = 0;

    // for (auto c : configs_sdxl_1024){
    for (auto c : configs){
        test_model model;
        load_model(model, std::get<0>(c), std::get<1>(c), std::get<2>(c),
            std::get<3>(c), std::get<4>(c), std::get<5>(c), true);
            // std::get<3>(c), std::get<4>(c), std::get<5>(c), false);

        ggml_gallocr_t allocr = NULL;
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        //create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph_0(model);

        // compute the required memory
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size0 = ggml_gallocr_get_buffer_size(allocr, 0);
        // fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0f/1024.0f);


        int iterations = 0;

        double run_time0;
        std::vector<float> im2col_data = compute_graph(model, allocr, build_graph_0, iterations, &run_time0);

        ggml_gallocr_free(allocr);

        allocr = NULL;

        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        //create the worst case graph for memory usage estimation
        gf = build_graph_1(model);

        // compute the required memory
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size1 = ggml_gallocr_get_buffer_size(allocr, 0);
            // fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0f/1024.0f);



        double run_time1;
        // std::vector<float> wino_data = compute_graph(model, allocr, build_graph_1, iterations, &run_time1);
        std::vector<float> conv2d_data = compute_graph(model, allocr, build_graph_1, iterations, &run_time1);

        if(k==0) {
            k = 1;
            fprintf(stdout, "| (IC, OC, IW, IH, KW, KH) | im2col+GEMM TIME | im2col+GEMM VRAM | implicit GEMM TIME | implicit GEMM VRAM \n");
            fprintf(stdout, "| --- | --- | --- | --- | --- \n");
        }

        time_iter0 += run_time0;
        time_iter1 += run_time1;


        fprintf(stdout, " | (%d, %d, %d, %d, %d, %d) | %.2f ms | %.2f MB | %.2f ms | %.2f MB\n",
                std::get<0>(c), std::get<1>(c), std::get<2>(c), std::get<3>(c), std::get<4>(c), std::get<5>(c),
                run_time0, mem_size0/1024.0f/1024.0f,
                run_time1, mem_size1/1024.0f/1024.0f);

       // int i = 2048;
        // for(int i = 0; i < ggml_nelements(wino_res); i++) {
        // for(int i = 0; i < 26*38; i++) {
        for(int i = 0; i < conv2d_data.size(); i++) {
            float diff = fabs(im2col_data[i] - conv2d_data[i]);
            // if(diff > 0.5) {
                //   printf("(%7.3f, %7.3f, %.2f, %d) \n",
                    printf("(%f, %f, %f, %d) \n",
                  im2col_data[i], conv2d_data[i],
                  diff, i);
                // break;
            // }
        }

        ggml_free(model.ctx);
        ggml_backend_buffer_free(model.buffer);
        ggml_backend_free(model.backend);
        ggml_gallocr_free(allocr);

    }
    printf("| 1 unet iter takes| %.2f ms |   | %.2f ms | \n", time_iter0, time_iter1);

    // printf("\nPerforming test:\n");
    return 0;
}
