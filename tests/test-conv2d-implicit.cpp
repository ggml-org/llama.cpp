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



void load_model(test_model & model, int ic, int oc, int iw, int ih, bool use_gpu = false ) {
    // create data
    int KW = 3, KH = 3, IC = ic, OC = oc;
    int IW = iw, IH = ih, N = 1;

    // printf(" input: IC = %d, OC = %d, IW = %d, IH = %d \n ", IC, OC, IW, IH);

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
#endif

#ifdef GGML_USE_METAL
    if (use_gpu) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        ggml_backend_metal_log_set_callback(ggml_log_callback_default, nullptr);
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

    int s0 = 1;
    int s1 = 1;
    int p0 = 1;
    int p1 = 1;
    int d0 = 1;
    int d1 = 1;

       

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

    int s0 = 1;
    int s1 = 1;
    int p0 = 1;
    int p1 = 1;
    int d0 = 1;
    int d1 = 1;

       

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

struct ggml_cgraph * build_graph_2(const test_model& model) {
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



    // recalculate for avoid fragmentation
    // struct ggml_tensor* conv2d_res = ggml_conv_2d(ctx0, model.a, model.b, s0, s1, p0, p1, d0, d1);
    // ggml_set_name(conv2d_res, "conv2d_res");
    // ggml_build_forward_expand(gf, conv2d_res);
    // int64_t *ne = conv2d_res->ne;
    // printf("conv2d: (%zu, %zu, %zu, %zu) \n", ne[0], ne[1], ne[2], ne[3]);


    struct ggml_tensor* wino_res = ggml_conv_2d_implicitgemm(ctx0, model.a, model.b, s0, s1, p0, p1, d0, d1, 0);
    // struct ggml_tensor* wino_res = ggml_conv_2d_direct(ctx0, model.a, model.b, s0, s1, p0, p1, d0, d1);
    ggml_set_name(wino_res, "wino_res");
    ggml_build_forward_expand(gf, wino_res);
    // ne = wino_res->ne;
    // printf("wino: (%zu, %zu, %zu, %zu) \n", ne[0], ne[1], ne[2], ne[3]);
    ggml_free(ctx0);
    return gf;
}




std::vector<float> compute_graph(const test_model & model, ggml_gallocr_t allocr,
            build_graph_t build_graph, int iters, double *t) {
    struct ggml_cgraph * gf = build_graph(model);


    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);
    int n_threads = 1;

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads);
    }
#endif

   

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
    // printf(" Taking %f ms\n ", time_us/1000);
   
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
    std::vector<std::tuple<int, int, int, int>> configs = {
        // std::make_tuple(64,64,48,64),
        // std::make_tuple(320,320,104,152),
        // std::make_tuple(640,640,52,76),
        // std::make_tuple(640,640,104,152),
        // std::make_tuple(960,320,104,152),
        std::make_tuple(160,1280,26,38),
        // std::make_tuple(1280,640,52,76),
        // std::make_tuple(1920,1280,26,38),
        // std::make_tuple(2560,1280,26,38),
        // std::make_tuple(512,512,104,152),
        // std::make_tuple(512,512,208,304),
        // std::make_tuple(512,256,416,608),
        // std::make_tuple(256,128,832,1216),
        // std::make_tuple(256,256,832,1216),
        // std::make_tuple(320,256,1024,1920)
    };

    int k = 0;

    for (auto c : configs){
        test_model model;
        load_model(model, std::get<0>(c), std::get<1>(c), std::get<2>(c), std::get<3>(c), true);

        ggml_gallocr_t allocr = NULL;
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        //create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph_0(model);

        // compute the required memory
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size0 = ggml_gallocr_get_buffer_size(allocr, 0);
        // fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0f/1024.0f);
       

        struct ggml_cgraph * gf_res_0 = NULL;    
        int iterations = 0;

        double run_time0;
        std::vector<float> conv2d_data = compute_graph(model, allocr, build_graph_0, iterations, &run_time0);

        ggml_gallocr_free(allocr);

        allocr = NULL;
        
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        //create the worst case graph for memory usage estimation
        gf = build_graph_1(model);

        // compute the required memory
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size1 = ggml_gallocr_get_buffer_size(allocr, 0);
            // fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0f/1024.0f);
        

        struct ggml_cgraph * gf_res_1 = NULL;    

        double run_time1;
        std::vector<float> wino_data = compute_graph(model, allocr, build_graph_1, iterations, &run_time1);


        ggml_gallocr_free(allocr);

        allocr = NULL;

        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        //create the worst case graph for memory usage estimation
        gf = build_graph_2(model);

        // compute the required memory
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size2 = ggml_gallocr_get_buffer_size(allocr, 0);
            // fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0f/1024.0f);


        struct ggml_cgraph * gf_res_2 = NULL;

        double run_time2;
        wino_data = compute_graph(model, allocr, build_graph_2, iterations, &run_time2);


        if(k==0) { 
            k = 1;
            fprintf(stderr, "| (IC, OC, IW, IH) | im2col+GEMM TIME | im2col+GEMM VRAM | direct TIME |  direct VRAM | implicit GEMM TIME | implicit GEMM VRAM \n");
            fprintf(stderr, "| --- | --- | --- |  --- | --- | --- | --- \n");
        }

        fprintf(stderr, " | (%d, %d, %d, %d) | %.2f ms | %.2f MB | %.2f ms | %.2f MB | %.2f ms | %.2f MB\n", 
                std::get<0>(c), std::get<1>(c), std::get<2>(c), std::get<3>(c),
                run_time0, mem_size0/1024.0f/1024.0f,
                run_time1, mem_size1/1024.0f/1024.0f,
                run_time2, mem_size2/1024.0f/1024.0f);


        // for(int i = 0; i < ggml_nelements(wino_res); i++) {
        for(int i = 0; i < 26*38; i++) {
            float diff = fabs(conv2d_data[i] - wino_data[i]);
            // if(diff > 1.e-4) {
                  printf("(%f, %f, %f, %d) \n",
                  conv2d_data[i],
                  wino_data[i], diff, i);
                // break;
            // }
        }

        ggml_free(model.ctx);
        ggml_backend_buffer_free(model.buffer);
        ggml_backend_free(model.backend);
        ggml_gallocr_free(allocr);

    }

    // printf("\nPerforming test:\n");    
    return 0;
}
