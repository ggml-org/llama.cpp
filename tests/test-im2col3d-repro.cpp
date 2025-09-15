// Minimal IM2COL_3D repro: compare CPU vs CUDA for a specific shape

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <stdexcept>

static void fill_uniform(std::mt19937 &rng, float *dst, size_t n, float lo=-1.0f, float hi=1.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < n; ++i) dst[i] = dist(rng);
}

static ggml_tensor * build_im2col3d_graph(ggml_context * ctx, ggml_type ktype,
                                          int IW, int IH, int ID, int N, int IC,
                                          int KW, int KH, int KD,
                                          int s0, int s1, int s2,
                                          int p0, int p1, int p2,
                                          int d0, int d1, int d2,
                                          ggml_tensor **out_a, ggml_tensor **out_b) {
    // a: [OC*IC, KD, KH, KW], choose OC=1 for simplicity
    int OC = 1;
    ggml_tensor * a = ggml_new_tensor_4d(ctx, ktype, KW, KH, KD, OC*IC);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, IW, IH, ID, N*IC);
    *out_a = a; *out_b = b;

    ggml_tensor * out = ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, GGML_TYPE_F32);
    return out;
}

static std::vector<float> run_backend(ggml_backend_t backend, ggml_type ktype,
                                      int IW, int IH, int ID, int N, int IC,
                                      int KW, int KH, int KD,
                                      int s0, int s1, int s2,
                                      int p0, int p1, int p2,
                                      int d0, int d1, int d2,
                                      const std::vector<float> &kernel_fill, const std::vector<float> &input_fill) {
    const size_t mem = ggml_tensor_overhead()*64 + ggml_graph_overhead();
    ggml_init_params ip = { mem, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a=nullptr, *b=nullptr;
    ggml_tensor * out = build_im2col3d_graph(ctx, ktype,
        IW, IH, ID, N, IC, KW, KH, KD,
        s0,s1,s2,p0,p1,p2,d0,d1,d2,
        &a, &b);

    // buffer allocate
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // set data
    // OC is 1 in this helper
    std::vector<float> a_f( (size_t)KW*KH*KD*IC , 0.0f);
    if (!kernel_fill.empty()) a_f = kernel_fill;
    if (a->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(a_f.size());
        ggml_fp32_to_fp16_row(a_f.data(), tmp.data(), tmp.size());
        ggml_backend_tensor_set(a, tmp.data(), 0, tmp.size()*sizeof(tmp[0]));
    } else {
        ggml_backend_tensor_set(a, a_f.data(), 0, a_f.size()*sizeof(float));
    }
    const size_t nB = (size_t)IW*IH*ID*N*IC;
    ggml_backend_tensor_set(b, input_fill.data(), 0, nB*sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        throw std::runtime_error("graph compute failed");
    }

    // fetch
    std::vector<float> out_f(ggml_nelements(out));
    ggml_backend_tensor_get(out, out_f.data(), 0, out_f.size()*sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out_f;
}

int main() {
    ggml_backend_load_all();

    // Shape from repro note: input [20,20,10,3]; kernel [3,3,3,3]; s/d/p=1.
    // Interpret as IW=20, IH=20, ID=10, N*IC=3 => choose N=1, IC=3; OC=1.
    const int IW=20, IH=20, ID=10, N=1, IC=3;
    const int KW=3, KH=3, KD=3;
    const int s0=1, s1=1, s2=1, p0=1, p1=1, p2=1, d0=1, d1=1, d2=1;

    std::mt19937 rng(123);
    const size_t nB = (size_t)IW*IH*ID*N*IC;
    std::vector<float> input(nB);
    fill_uniform(rng, input.data(), nB);
    // kernel not used by im2col directly, but provide something
    std::vector<float> kernel((size_t)KW*KH*KD*IC, 0.0f);

    // Run CPU
    // Initialize a CPU backend via device registry
    ggml_backend_t cpu = nullptr;
    {
        const size_t n_dev = ggml_backend_dev_count();
        for (size_t i = 0; i < n_dev; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const char * name = ggml_backend_dev_name(dev);
            if (strstr(name, "CPU")) { cpu = ggml_backend_dev_init(dev, NULL); break; }
        }
    }
    if (!cpu) { printf("[SKIP] CPU backend not available\n"); return 0; }
    auto out_cpu = run_backend(cpu, GGML_TYPE_F32, IW,IH,ID,N,IC, KW,KH,KD, s0,s1,s2, p0,p1,p2, d0,d1,d2, kernel, input);
    ggml_backend_free(cpu);

    // Find a CUDA backend
    ggml_backend_t cuda = nullptr;
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        if (strstr(name, "CUDA")) { cuda = ggml_backend_dev_init(dev, NULL); break; }
    }
    if (!cuda) {
        printf("[SKIP] CUDA backend not available\n");
        return 0;
    }
    auto out_cuda = run_backend(cuda, GGML_TYPE_F32, IW,IH,ID,N,IC, KW,KH,KD, s0,s1,s2, p0,p1,p2, d0,d1,d2, kernel, input);
    ggml_backend_free(cuda);

    // Compare
    double num=0.0, den=0.0;
    size_t n = out_cpu.size();
    size_t n_bad=0, idx0=0;
    for (size_t i = 0; i < n; ++i) {
        double a = out_cpu[i];
        double b = out_cuda[i];
        num += (a-b)*(a-b);
        den += (a*a);
        if (fabs(a-b) > 1e-3) { if (n_bad==0) idx0=i; n_bad++; }
    }
    double nmse = den > 0 ? num/den : 0.0;
    if (nmse > 1e-6) {
        printf("[FAIL] IM2COL_3D mismatch: nmse=%.6g n_bad=%zu example idx=%zu cpu=%g cuda=%g\n",
               nmse, n_bad, idx0, out_cpu[idx0], out_cuda[idx0]);
        return 1;
    }
    printf("[OK] IM2COL_3D CPU vs CUDA nmse=%.3g\n", nmse);
    return 0;
}
