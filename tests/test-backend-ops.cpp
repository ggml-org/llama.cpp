#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <vector>


static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    // static RNG initialization (revisit if n_threads stops being constant)
    static const size_t n_threads = std::thread::hardware_concurrency();
    static std::vector<std::default_random_engine> generators = []() {
        std::random_device rd;
        std::vector<std::default_random_engine> vec;
        vec.reserve(n_threads);
        //for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(1234 + i); } // fixed seed
        for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(rd()); }
        return vec;
    }();

    size_t size = ggml_nelements(tensor);
    std::vector<float> data(size);

    auto init_thread = [&](size_t ith, size_t start, size_t end) {
        std::uniform_real_distribution<float> distribution(min, max);
        for (size_t i = start; i < end; i++) {
            data[i] = distribution(generators[ith]);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (size_t i = 0; i < n_threads; i++) {
        size_t start =     i*size/n_threads;
        size_t end   = (i+1)*size/n_threads;
        threads.emplace_back(init_thread, i, start, end);
    }
    for (auto & t : threads) {
        t.join();
    }

#if 0
    const char * val_str = getenv("GGML_TEST_EPS");
    float val = 1e-9f;
    if (val_str != nullptr) {
        val = std::stof(val_str);
        printf("GGML_TEST_EPS=%e\n", val);
    }

    // test quantization with very small values that may result in nan scales due to division by zero
    if (ggml_is_quantized(tensor->type)) {
        for (int i = 0; i < 256; i++) {
            data[i] = val;
        }
    }
#endif

    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, size * sizeof(float));
    } else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        GGML_ASSERT(size % ggml_blck_size(tensor->type) == 0);
        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, size));
        std::vector<float> imatrix(tensor->ne[0], 1.0f); // dummy importance matrix
        const float * im = imatrix.data();
        if (!ggml_quantize_requires_imatrix(tensor->type)) {
            // when the imatrix is optional, we want to test both quantization with and without imatrix
            // use one of the random numbers to decide
            if (data[0] > 0.5f*(min + max)) {
                im = nullptr;
            }
        }

        ggml_quantize_chunk(tensor->type, data.data(), dataq.data(), 0, size/tensor->ne[0], tensor->ne[0], im);
        GGML_ASSERT(ggml_validate_row_data(tensor->type, dataq.data(), dataq.size()));
        // TODO: other cases
        //#pragma omp parallel for
        //for (int i = 0; i < tensor->ne[1]; i++) {
        //    ggml_quantize_chunk(tensor->type, data.data(), dataq.data(),
        //        i * tensor->ne[0], 1, tensor->ne[0], im);
        //}

        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    } else if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16 || tensor->type == GGML_TYPE_I32) {
        // This is going to create some weird integers though.
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static std::vector<float> tensor_to_float(const ggml_tensor * t) {
    std::vector<float> tv;
    tv.reserve(ggml_nelements(t));

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    ggml_type_traits_t tt = ggml_internal_get_type_traits(t->type);
    size_t bs = ggml_blck_size(t->type);
    std::vector<float> vq(ggml_blck_size(t->type));
    bool quantized = ggml_is_quantized(t->type);

    // access elements by index to avoid gaps in views
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0 += bs) {
                    size_t i = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0/bs*t->nb[0];
                    if (t->type == GGML_TYPE_F16) {
                        tv.push_back(ggml_fp16_to_fp32(*(ggml_fp16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_BF16) {
                        tv.push_back(ggml_bf16_to_fp32(*(ggml_bf16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_F32) {
                        tv.push_back(*(float *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I32) {
                        tv.push_back((float)*(int32_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I16) {
                        tv.push_back((float)*(int16_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I8) {
                        tv.push_back((float)*(int8_t *) &buf[i]);
                    } else if (quantized) {
                        tt.to_float(&buf[i], vq.data(), bs);
                        tv.insert(tv.end(), vq.begin(), vq.end());
                    } else {
                        GGML_ABORT("fatal error");
                    }
                }
            }
        }
    }

    return tv;
}

/*
static double cosine_similarity(const float * v1, const float * v2, size_t n) {
    double dot = 0.0;
    double mag1 = 0.0;
    double mag2 = 0.0;

    for (size_t i = 0; i < n; i++) {
        if (std::isnan(v1[i]) || std::isnan(v2[i])) {
            return -1.0f;
        }
        if (std::isinf(v1[i]) && std::isinf(v2[i])) {
            continue;
        }
        dot  += v1[i]*v2[i];
        mag1 += v1[i]*v1[i];
        mag2 += v2[i]*v2[i];
    }

    return dot/sqrt(mag1*mag2);
}

static float distance(const float * v1, const float * v2, size_t n) {
    double d = 0.0;

    for (size_t i = 0; i < n; i++) {
        if (std::isnan(v1[i]) || std::isnan(v2[i])) {
            return INFINITY;
        }
        if (std::isinf(v1[i]) && std::isinf(v2[i])) {
            continue;
        }
        d += (v1[i] - v2[i])*(v1[i] - v2[i]);
    }

    return sqrt(d);
}

static float vec_len(const float * v, size_t n) {
    double d = 0.0;

    for (size_t i = 0; i < n; i++) {
        if (std::isnan(v[i])) {
            return INFINITY;
        }
        if (std::isinf(v[i])) {
            continue;
        }
        d += v[i]*v[i];
    }

    return sqrt(d);
}
*/

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

// utils for printing the variables of the test cases
#define VAR_TO_STR(x) (#x "=" + var_to_str(x))

template<typename T>
static std::string var_to_str(const T & x) {
    return std::to_string(x);
}

template<typename T, size_t N>
static std::string var_to_str(const T (&x)[N]) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

template<typename T, size_t N>
static std::string var_to_str(const std::array<T, N> & x) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

//static std::string var_to_str(ggml_unary_op unary_op) {
//    return ggml_unary_op_name(unary_op);
//}

static std::string var_to_str(ggml_type type) {
    return ggml_type_name(type);
}

static std::string var_to_str(ggml_op_pool pool) {
    switch (pool) {
        case GGML_OP_POOL_AVG:  return "avg";
        case GGML_OP_POOL_MAX:  return "max";
        default:                return std::to_string(pool);
    }
}

#define VARS_TO_STR1(a) VAR_TO_STR(a)
#define VARS_TO_STR2(a, b) VAR_TO_STR(a) + "," + VAR_TO_STR(b)
#define VARS_TO_STR3(a, b, c) VAR_TO_STR(a) + "," + VARS_TO_STR2(b, c)
#define VARS_TO_STR4(a, b, c, d) VAR_TO_STR(a) + "," + VARS_TO_STR3(b, c, d)
#define VARS_TO_STR5(a, b, c, d, e) VAR_TO_STR(a) + "," + VARS_TO_STR4(b, c, d, e)
#define VARS_TO_STR6(a, b, c, d, e, f) VAR_TO_STR(a) + "," + VARS_TO_STR5(b, c, d, e, f)
#define VARS_TO_STR7(a, b, c, d, e, f, g) VAR_TO_STR(a) + "," + VARS_TO_STR6(b, c, d, e, f, g)
#define VARS_TO_STR8(a, b, c, d, e, f, g, h) VAR_TO_STR(a) + "," + VARS_TO_STR7(b, c, d, e, f, g, h)
#define VARS_TO_STR9(a, b, c, d, e, f, g, h, i) VAR_TO_STR(a) + "," + VARS_TO_STR8(b, c, d, e, f, g, h, i)
#define VARS_TO_STR10(a, b, c, d, e, f, g, h, i, j) VAR_TO_STR(a) + "," + VARS_TO_STR9(b, c, d, e, f, g, h, i, j)
#define VARS_TO_STR11(a, b, c, d, e, f, g, h, i, j, k) VAR_TO_STR(a) + "," + VARS_TO_STR10(b, c, d, e, f, g, h, i, j, k)
#define VARS_TO_STR12(a, b, c, d, e, f, g, h, i, j, k, l) VAR_TO_STR(a) + "," + VARS_TO_STR11(b, c, d, e, f, g, h, i, j, k, l)

#ifdef GGML_USE_SYCL
static bool inline _isinf(float f) {
    return (*(uint32_t *)&f & 0x7fffffff) == 0x7f800000;
}
#else
static bool inline _isinf(float f) { return std::isinf(f); }
#endif

// accept FLT_MAX as infinity
static bool isinf_or_max(float f) {
    return _isinf(f) || f == FLT_MAX || f == -FLT_MAX;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

enum test_mode {
    MODE_TEST,
    MODE_PERF,
};

struct test_case {
    virtual ~test_case() {}

    virtual std::string op_desc(ggml_tensor * t) {
        return ggml_op_desc(t);
    }

    virtual std::string vars() {
        return "";
    }

    virtual ggml_tensor * build_graph(ggml_context * ctx) = 0;

    virtual double max_nmse_err() {
        return 1e-7;
    }

    virtual void initialize_tensors(ggml_context * ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t);
        }
    }

    virtual size_t op_size(ggml_tensor * t) {
        size_t size = ggml_nbytes(t);
        // add source tensors
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (t->src[i] != NULL) {
                size += ggml_nbytes(t->src[i]);
            }
        }
        return size;
    }

    ggml_cgraph * gf = nullptr;

    static const int sentinel_size = 1024;

    test_mode mode;

    std::vector<ggml_tensor *> sentinels;

    void add_sentinel(ggml_context * ctx) {
        if (mode == MODE_PERF) {
            return;
        }
        ggml_tensor * sentinel = ::ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sentinel_size);
        ggml_format_name(sentinel, "sent_%zu", sentinels.size());
        sentinels.push_back(sentinel);
    }

    // hijack ggml_new_tensor to add sentinels after each tensor to check for overflows in the backend

    ggml_tensor * ggml_new_tensor(ggml_context * ctx, ggml_type type, int n_dims, const int64_t * ne) {
        ggml_tensor * t = ::ggml_new_tensor(ctx, type, n_dims, ne);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_1d(ggml_context * ctx, ggml_type type, int64_t ne0) {
        ggml_tensor * t = ::ggml_new_tensor_1d(ctx, type, ne0);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_2d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1) {
        ggml_tensor * t = ::ggml_new_tensor_2d(ctx, type, ne0, ne1);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_3d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
        ggml_tensor * t = ::ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_4d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        ggml_tensor * t = ::ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
        add_sentinel(ctx);
        return t;
    }

    bool eval(ggml_backend_t backend1, ggml_backend_t backend2, const char * op_name) {
        mode = MODE_TEST;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context * ctx = ggml_init(params);

        gf = ggml_new_graph(ctx);

        // pre-graph sentinel
        add_sentinel(ctx);

        ggml_tensor * out = build_graph(ctx);

        if (op_name != nullptr && op_desc(out) != op_name) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            ggml_free(ctx);
            return true;
        }

        printf("  %s(%s): ", op_desc(out).c_str(), vars().c_str());
        fflush(stdout);

        // check if the backends support the ops
        bool supported = true;
        for (ggml_backend_t backend : {backend1, backend2}) {
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
                if (!ggml_backend_supports_op(backend, t)) {
                    printf("not supported [%s] ", ggml_backend_name(backend));
                    supported = false;
                    break;
                }
            }
        }
        if (!supported) {
            printf("\n");
            ggml_free(ctx);
            return true;
        }

        // post-graph sentinel
        add_sentinel(ctx);

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend1);
        if (buf == NULL) {
            printf("failed to allocate tensors [%s] ", ggml_backend_name(backend1));
            ggml_free(ctx);
            return false;
        }

        // build graph
        ggml_build_forward_expand(gf, out);

        // add sentinels as graph nodes so that they are checked in the callback
        for (ggml_tensor * sentinel : sentinels) {
            gf->nodes[gf->n_nodes++] = sentinel;
        }

        // randomize tensors
        initialize_tensors(ctx);

        // compare
        struct callback_userdata {
            bool   ok;
            double max_err;
            ggml_backend_t backend1;
            ggml_backend_t backend2;
        };

        callback_userdata ud {
            true,
            max_nmse_err(),
            backend1,
            backend2
        };

        auto callback = [](int index, ggml_tensor * t1, ggml_tensor * t2, void * user_data) -> bool {
            callback_userdata * ud = (callback_userdata *) user_data;
            const char * bn1 = ggml_backend_name(ud->backend1);
            const char * bn2 = ggml_backend_name(ud->backend2);

            if (t1->op == GGML_OP_NONE) {
                // sentinels must be unchanged
                std::vector<uint8_t> t1_data(ggml_nbytes(t1));
                std::vector<uint8_t> t2_data(ggml_nbytes(t2));
                ggml_backend_tensor_get(t1, t1_data.data(), 0, ggml_nbytes(t1));
                ggml_backend_tensor_get(t2, t2_data.data(), 0, ggml_nbytes(t2));

                if (memcmp(t1_data.data(), t2_data.data(), ggml_nbytes(t1)) != 0) {
                    printf("sentinel mismatch: %s ", t1->name);
                    ud->ok = false;
                    return true;
                }
            }

            std::vector<float> f1 = tensor_to_float(t1);
            std::vector<float> f2 = tensor_to_float(t2);

            for (size_t i = 0; i < f1.size(); i++) {
                // check for nans
                if (std::isnan(f1[i]) || std::isnan(f2[i])) {
                    printf("[%s] NaN at index %zu (%s=%f %s=%f) ", ggml_op_desc(t1), i, bn1, f1[i], bn2, f2[i]);
                    ud->ok = false;
                    return true;
                }
                // check for infs: both must be inf of the same sign, or both must be finite
                if (isinf_or_max(f1[i]) || isinf_or_max(f2[i])) {
                    if (isinf_or_max(f1[i]) && isinf_or_max(f2[i])) {
                        if (std::signbit(f1[i]) != std::signbit(f2[i])) {
                            printf("[%s] inf sign mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                            ud->ok = false;
                            return true;
                        }
                    } else {
                        printf("[%s] inf mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                        ud->ok = false;
                        return true;
                    }
                }
            }

            double err = nmse(f1.data(), f2.data(), f1.size());
            if (err > ud->max_err) {
                printf("[%s] NMSE = %.9f > %.9f ", ggml_op_desc(t1), err, ud->max_err);
                //for (int i = 0; i < (int) f1.size(); i++) {
                //    printf("%5d %9.6f %9.6f, diff = %9.6f\n", i, f1[i], f2[i], f1[i] - f2[i]);
                //}
                //printf("\n");
                //exit(1);
                ud->ok = false;
            }
            return true;

            GGML_UNUSED(index);
        };

        const bool cmp_ok = ggml_backend_compare_graph_backend(backend1, backend2, gf, callback, &ud);

        if (!cmp_ok) {
            printf("compare failed ");
        }

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);

        if (ud.ok && cmp_ok) {
            printf("\033[1;32mOK\033[0m\n");
            return true;
        }

        printf("\033[1;31mFAIL\033[0m\n");
        return false;
    }

    bool eval_perf(ggml_backend_t backend, const char * op_name) {
        mode = MODE_PERF;

        static const size_t graph_nodes = 8192;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context * ctx = ggml_init(params);

        ggml_tensor * out = build_graph(ctx);

        if (op_name != nullptr && op_desc(out) != op_name) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            ggml_free(ctx);
            return true;
        }

        int len = printf("  %s(%s): ", op_desc(out).c_str(), vars().c_str());
        fflush(stdout);

        // check if backends support op
        if (!ggml_backend_supports_op(backend, out)) {
            printf("not supported\n");
            ggml_free(ctx);
            return true;
        }

        // align while also leaving some margin for variations in parameters
        int align = 20;
        int last = (len + align - 1) / align * align;
        if (last - len < 5) {
            last += align;
        }
        last = std::max(last, 60);
        printf("%*s", last - len, "");

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == NULL) {
            printf("failed to allocate tensors\n");
            ggml_free(ctx);
            return false;
        }

        // randomize tensors
        initialize_tensors(ctx);

        // build graph
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_nodes, false);
        ggml_build_forward_expand(gf, out);

        // warmup run
        ggml_backend_graph_compute(backend, gf);

        // duplicate the op
        size_t target_size = ggml_backend_is_cpu(backend) ? 1ULL << 33 : 1ULL << 35; // 8 GB CPU, 32 GB GPU
        int n_runs = std::min((size_t)gf->size - gf->n_nodes, target_size / op_size(out)) + 1;
        for (int i = 1; i < n_runs; i++) {
            gf->nodes[gf->n_nodes++] = out;
        }

        // calculate memory
        size_t mem = n_runs * op_size(out);
        auto tensor_op_size = [](ggml_tensor * t) {
            size_t size = ggml_nbytes(t);
            // add source tensors
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (t->src[i] != NULL) {
                    size += ggml_nbytes(t->src[i]);
                }
            }
            return size;
        };
        for (int i = 0; i < gf->n_nodes; i++) {
            if (ggml_is_view_op(gf->nodes[i]->op) || gf->nodes[i] == out) {
                continue;
            }
            mem += tensor_op_size(gf->nodes[i]);
        }

        // run
        ggml_backend_synchronize(backend);

        int64_t start_time = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        int64_t end_time = ggml_time_us();
        double time_us = end_time - start_time;

        printf("    %5d runs - %8.2f us/run - %8zu kB/run - \033[1;34m%7.2f GB/s\033[0m\n",
            n_runs,
            time_us / n_runs,
            op_size(out) / 1024,
            mem / (time_us/1e6) / 1024.0 / 1024.0 / 1024.0);

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);

        return true;
    }
};

// GGML_OP_UNARY
struct test_unary : public test_case {
    const ggml_unary_op op;
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int v; // view (1 : non-contiguous a)

    std::string vars() override {
        return VARS_TO_STR3(type, ne_a, v);
    }

    test_unary(ggml_unary_op op,
            ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {128, 10, 10, 10},
            int v = 0)
        : op(op), type(type), ne_a(ne_a), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        }
        ggml_tensor * out = ggml_unary(ctx, a, op);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            // test extended range of values to check for NaNs in GELU
            init_tensor_uniform(t, -150.f, 150.f);
        }
    }
};

// GGML_OP_GET_ROWS
struct test_get_rows : public test_case {
    const ggml_type type;
    const int n; // cols
    const int m; // rows
    const int r; // rows to get
    const int b; // batch size
    const bool v; // view (non-contiguous src1)

    std::string vars() override {
        return VARS_TO_STR6(type, n, m, r, b, v);
    }

    test_get_rows(ggml_type type = GGML_TYPE_F32, int n = 10, int m = 5, int r = 3, int b = 1, bool v = false)
        : type(type), n(n), m(m), r(r), b(b), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * in = ggml_new_tensor_3d(ctx, type, n, m, b);
        ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r, b);
        if (v) {
            rows = ggml_view_2d(ctx, rows, r/2, b, rows->nb[1], 0);
        }
        ggml_tensor * out = ggml_get_rows(ctx, in, rows);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                // rows
                std::vector<int> data(r*b);
                for (int i = 0; i < r*b; i++) {
                    data[i] = rand() % m;
                }
                ggml_backend_tensor_set(t, data.data(), 0, r * b * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_REPEAT
struct test_repeat : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4> nr;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, nr);
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) * 2;
    }

    test_repeat(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            std::array<int, 4> nr = {2, 2, 2, 2})
        : type(type), ne(ne), nr(nr) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * target = ggml_new_tensor_4d(ctx, type, ne[0]*nr[0], ne[1]*nr[1], ne[2]*nr[2], ne[3]*nr[3]);
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_repeat(ctx, src, target);
        return out;
    }
};

// GGML_OP_DUP
struct test_dup : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> permute;
    bool _use_permute;

    std::string vars() override {
        std::string v = VARS_TO_STR2(type, ne);
        if (_use_permute) v += "," + VAR_TO_STR(permute);
        return v;
    }

    test_dup(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 20, 1},
            std::array<int64_t, 4> permute = {0, 0, 0, 0})
        : type(type), ne(ne), permute(permute),
            _use_permute(permute[0] + permute[1] + permute[2] + permute[3] > 0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        if (_use_permute) {
            src = ggml_permute(ctx, src, permute[0], permute[1], permute[2], permute[3]);
        }
        ggml_tensor * out = ggml_dup(ctx, src);
        return out;
    }
};

// GGML_OP_CPY
struct test_cpy : public test_case {
    const ggml_type type_src;
    const ggml_type type_dst;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> permute;
    bool _src_use_permute;

    std::string vars() override {
        return VARS_TO_STR4(type_src, type_dst, ne, permute);
    }

    double max_nmse_err() override {
        return 1e-6;
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) + ggml_nbytes(t->src[0]);
    }

    test_cpy(ggml_type type_src = GGML_TYPE_F32, ggml_type type_dst = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 1},
            std::array<int64_t, 4> permute = {0, 0, 0, 0})
        : type_src(type_src), type_dst(type_dst), ne(ne), permute(permute),
          _src_use_permute(permute[0] + permute[1] + permute[2] + permute[3] > 0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type_src, 4, ne.data());
        if (_src_use_permute) {
            src = ggml_permute(ctx, src, permute[0], permute[1], permute[2], permute[3]);
        }
        ggml_tensor* dst = ggml_new_tensor(ctx, type_dst, 4, src->ne);
        ggml_tensor * out = ggml_cpy(ctx, src, dst);
        return out;
    }
};

// GGML_OP_CONT
struct test_cont : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_cont(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 1})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        src = ggml_transpose(ctx, src);
        ggml_tensor * out = ggml_cont(ctx, src);

        return out;
    }
};

// GGML_OP_ADD
// GGML_OP_MUL
// GGML_OP_DIV
struct test_bin_bcast : public test_case {
    using op_t = ggml_tensor * (*) (ggml_context *, ggml_tensor *, ggml_tensor *);
    op_t op;
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4> nr;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, nr);
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) * 3;
    }

    test_bin_bcast(op_t op, ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 1, 1},
            std::array<int, 4> nr = {1, 2, 1, 1})
        : op(op), type(type), ne(ne), nr(nr) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0]*nr[0], ne[1]*nr[1], ne[2]*nr[2], ne[3]*nr[3]);
        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = op(ctx, a, b);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (op == ggml_div) {
                // avoid division by zero
                init_tensor_uniform(t, 1.0f, 2.0f);
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_SCALE
struct test_scale : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float scale;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, scale);
    }

    test_scale(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            float scale = 2.0f)
        : type(type), ne(ne), scale(scale) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_scale(ctx, a, scale);
        return out;
    }
};

// GGML_OP_NORM
struct test_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float eps;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, eps);
    }

    test_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 10, 10, 10},
            float eps = 1e-6f)
        : type(type), ne(ne), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_norm(ctx, a, eps);
        return out;
    }
};

// GGML_OP_RMS_NORM
struct test_rms_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float eps;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, eps);
    }

    test_rms_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 10, 10, 10},
            float eps = 1e-6f)
        : type(type), ne(ne), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_rms_norm(ctx, a, eps);
        return out;
    }
};

// GGML_OP_SSM_CONV
struct test_ssm_conv : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const std::array<int64_t, 4> ne_b;

    std::string vars() override {
        return VARS_TO_STR3(type, ne_a, ne_b);
    }

    test_ssm_conv(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {10, 10, 10, 1},
            std::array<int64_t, 4> ne_b = {3, 3, 1, 1})
        : type(type), ne_a(ne_a), ne_b(ne_b) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a   = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_tensor * b   = ggml_new_tensor(ctx, type, 4, ne_b.data());
        ggml_tensor * out = ggml_ssm_conv(ctx, a, b);
        return out;
    }
};

// GGML_OP_SSM_SCAN
struct test_ssm_scan : public test_case {
    const ggml_type type;

    const int64_t d_state;
    const int64_t d_inner;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;

    std::string vars() override {
        return VARS_TO_STR5(type, d_state, d_inner, n_seq_tokens, n_seqs);
    }

    test_ssm_scan(ggml_type type = GGML_TYPE_F32,
            int64_t d_state = 32, int64_t d_inner = 32, int64_t n_seq_tokens = 32, int64_t n_seqs = 32)
        : type(type), d_state(d_state), d_inner(d_inner), n_seq_tokens(n_seq_tokens), n_seqs(n_seqs) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * s   = ggml_new_tensor(ctx, type, 4, std::vector<int64_t>{ d_state, d_inner,      n_seqs, 1 }.data());
        ggml_tensor * x   = ggml_new_tensor(ctx, type, 4, std::vector<int64_t>{ d_inner, n_seq_tokens, n_seqs, 1 }.data());
        ggml_tensor * dt  = ggml_new_tensor(ctx, type, 4, std::vector<int64_t>{ d_inner, n_seq_tokens, n_seqs, 1 }.data());
        ggml_tensor * A   = ggml_new_tensor(ctx, type, 4, std::vector<int64_t>{ d_state, d_inner,      1     , 1 }.data());
        ggml_tensor * B   = ggml_new_tensor(ctx, type, 4, std::vector<int64_t>{ d_state, n_seq_tokens, n_seqs, 1 }.data());
        ggml_tensor * C   = ggml_new_tensor(ctx, type, 4, std::vector<int64_t>{ d_state, n_seq_tokens, n_seqs, 1 }.data());
        ggml_tensor * out = ggml_ssm_scan(ctx, s, x, dt, A, B, C);
        return out;
    }
};

// GGML_OP_MUL_MAT
struct test_mul_mat : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const std::array<int64_t, 2> bs; // dims 3 and 4
    const std::array<int64_t, 2> nr; // repeat in dims 3 and 4

    std::string vars() override {
        return VARS_TO_STR7(type_a, type_b, m, n, k, bs, nr);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    size_t op_size(ggml_tensor * t) override {
        size_t a = ggml_nbytes(t->src[0]) * n * nr[0] * nr[1];
        size_t b = ggml_nbytes(t->src[1]) * m;
        size_t c  = ggml_nbytes(t);
        return a + b + c;

        GGML_UNUSED(t);
    }

    test_mul_mat(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int64_t m = 32, int64_t n = 32, int64_t k = 32,
            std::array<int64_t, 2> bs = {10, 10},
            std::array<int64_t, 2> nr = {2, 2})
        : type_a(type_a), type_b(type_b), m(m), n(n), k(k), bs(bs), nr(nr) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type_a, k, m, bs[0]      , bs[1]);
        ggml_tensor * b = ggml_new_tensor_4d(ctx, type_b, k, n, bs[0]*nr[0], bs[1]*nr[1]);
        ggml_tensor * out = ggml_mul_mat(ctx, a, b);
        return out;
    }
};

// GGML_OP_MUL_MAT_ID
struct test_mul_mat_id : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int n_mats;
    const int n_used;
    const bool b; // brodcast b matrix
    const int64_t m;
    const int64_t n;
    const int64_t k;

    std::string vars() override {
        return VARS_TO_STR8(type_a, type_b, n_mats, n_used, b, m, n, k);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    size_t op_size(ggml_tensor * t) override {
        size_t a = ggml_nbytes(t->src[2]) * n;
        size_t b = ggml_nbytes(t->src[1]) * m;
        size_t c  = ggml_nbytes(t);
        return a + b + c;

        GGML_UNUSED(t);
    }

    test_mul_mat_id(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int n_mats = 8, int n_used = 2, bool b = false,
            int64_t m = 32, int64_t n = 32, int64_t k = 32)
        : type_a(type_a), type_b(type_b), n_mats(n_mats), n_used(n_used), b(b),
            m(m), n(n), k(k) {
            GGML_ASSERT(n_used <= n_mats);
        }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor * as = ggml_new_tensor_3d(ctx, type_a, k, m, n_mats);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_mats, n);
        if (n_used != n_mats) {
            ids = ggml_view_2d(ctx, ids, n_used, n, ids->nb[1], 0);
        }
        ggml_tensor * b = ggml_new_tensor_3d(ctx, type_b, k, this->b ? 1 : n_used, n);
        ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                // ids
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<int32_t> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i % n_mats;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_SQR
struct test_sqr : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_sqr(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_sqr(ctx, a);
        return out;
    }
};

// GGML_OP_SQRT
struct test_sqrt : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_sqrt(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_sqrt(ctx, a);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        // fill with positive values
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, 0.0f, 100.0f);
        }
    }
};

// GGML_OP_SIN
struct test_sin : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_sin(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_sin(ctx, a);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -100.0f, 100.0f);
        }
    }
};

// GGML_OP_COS
struct test_cos : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_cos(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_cos(ctx, a);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -100.0f, 100.0f);
        }
    }
};

// GGML_OP_CLAMP
struct test_clamp : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float min;
    float max;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, min, max);
    }

    test_clamp(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            float min = -0.5f, float max = 0.5f)
        : type(type), ne(ne), min(min), max(max) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_clamp(ctx, a, min, max);
        return out;
    }
};

// GGML_OP_DIAG_MASK_INF
struct test_diag_mask_inf : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int n_past;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, n_past);
    }

    test_diag_mask_inf(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            int n_past = 5)
        : type(type), ne(ne), n_past(n_past) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_diag_mask_inf(ctx, a, n_past);
        return out;
    }
};

// GGML_OP_SOFT_MAX
struct test_soft_max : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const bool mask;
    const float scale;
    const float max_bias;

    std::string vars() override {
        return VARS_TO_STR5(type, ne, mask, scale, max_bias);
    }

    // the 1024 test with bias occasionally fails:
    // SOFT_MAX(type=f32,ne=[1024,16,1,1],mask=1,scale=1.000000,max_bias=8.000000): [SOFT_MAX] NMSE = 0.000000103 > 0.000000100 FAIL
    virtual double max_nmse_err() override {
        return 1e-6;
    }

    test_soft_max(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            bool mask = false,
            float scale = 1.0f,
            float max_bias = 0.0f)
        : type(type), ne(ne), mask(mask), scale(scale), max_bias(max_bias) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * mask = nullptr;
        if (this->mask) {
            mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne[0], ne[1]);
        }
        ggml_tensor * out = ggml_soft_max_ext(ctx, a, mask, scale, max_bias);
        return out;
    }
};


// GGML_OP_ROPE
struct test_rope : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int n_dims;
    int mode;
    int n_ctx; // used to generate positions
    float fs; // freq_scale
    float ef; // ext_factor
    float af; // attn_factor
    bool ff;
    int v; // view (1 : non-contiguous a)

    std::string vars() override {
        return VARS_TO_STR10(type, ne_a, n_dims, mode, n_ctx, fs, ef, af, ff, v);
    }

    test_rope(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {10, 10, 10, 1},
            int n_dims = 10, int mode = 0, int n_ctx = 512, float fs = 1.0f, float ef = 0.0f, float af = 0.0f, bool ff = false, int v = 0)
        : type(type), ne_a(ne_a), n_dims(n_dims), mode(mode), n_ctx(n_ctx), fs(fs), ef(ef), af(af), ff(ff), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 2; ne[1] *= 4; ne[2] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        }
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne_a[2]);
        ggml_tensor * freq = ff ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_dims/2) : nullptr;
        ggml_tensor * out = ggml_rope_ext(ctx, a, pos, freq, n_dims, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // pos
                std::vector<int> data(ne_a[2]);
                for (int i = 0; i < ne_a[2]; i++) {
                    data[i] = rand() % n_ctx;
                }
                ggml_backend_tensor_set(t, data.data(), 0, ne_a[2] * sizeof(int));
            } else {
                if (t->ne[0] == n_dims/2) {
                    // frequency factors in the range [0.9f, 1.1f]
                    init_tensor_uniform(t, 0.9f, 1.1f);
                } else {
                    init_tensor_uniform(t);
                }
            }
        }
    }
};

// GGML_OP_POOL2D
struct test_pool2d : public test_case {
    enum ggml_op_pool pool_type;
    const ggml_type type_input;
    const std::array<int64_t, 4> ne_input;
    // kernel size
    const int k0;
    const int k1;
    // stride
    const int s0;
    const int s1;
    // padding
    const int p0;
    const int p1;

    std::string vars() override {
        return VARS_TO_STR9(pool_type, type_input, ne_input, k0, k1, s0, s1, p0, p1);
    }

    test_pool2d(ggml_op_pool pool_type = GGML_OP_POOL_AVG,
            ggml_type type_input = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_input = {10, 10, 3, 1}, // [input_width, input_height, input_channels, 1]
            int k0 = 3, int k1 = 3,
            int s0 = 1, int s1 = 1,
            int p0 = 1, int p1 = 1)
        : pool_type(pool_type), type_input(type_input), ne_input(ne_input), k0(k0), k1(k1), s0(s0), s1(s1), p0(p0), p1(p1) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, type_input, 4, ne_input.data());
        ggml_tensor * out = ggml_pool_2d(ctx, input, pool_type, k0, k1, s0, s1, p0, p1);
        return out;
    }
};

// GGML_OP_CONV_TRANSPOSE_1D
struct test_conv_transpose_1d : public test_case {
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;

    const int s0; // stride
    const int p0; // padding
    const int d0; // dilation

    std::string vars() override {
        return VARS_TO_STR5(ne_input, ne_kernel, s0, p0, d0);
    }

    test_conv_transpose_1d(std::array<int64_t, 4> ne_input = {197, 32, 1, 1}, // [input_width, input_height, input_channels, 1]
                           std::array<int64_t, 4> ne_kernel = {16, 32, 32, 1}, // [kernel_width, kernel_height, input_channels, 1]
                           int s0 = 1, int p0 = 0, int d0 = 1)
        : ne_input(ne_input), ne_kernel(ne_kernel), s0(s0), p0(p0), d0(d0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_input.data());
        ggml_tensor * kernel = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_kernel.data());
        ggml_tensor * out = ggml_conv_transpose_1d(ctx, kernel, input, s0, p0, d0);
        return out;
    }
};

// GGML_OP_IM2COL
struct test_im2col : public test_case {
    const ggml_type type_input;
    const ggml_type type_kernel;
    const ggml_type dst_type;
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;
    // stride
    const int s0;
    const int s1;
    // padding
    const int p0;
    const int p1;
    // dilation
    const int d0;
    const int d1;
    // mode
    const bool is_2D;

    std::string vars() override {
        return VARS_TO_STR12(type_input, type_kernel, dst_type, ne_input, ne_kernel, s0, s1, p0, p1, d0, d1, is_2D);
    }

    test_im2col(ggml_type type_input = GGML_TYPE_F32, ggml_type type_kernel = GGML_TYPE_F16, ggml_type dst_type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_input = {10, 10, 3, 1}, // [input_width, input_height, input_channels, 1]
            std::array<int64_t, 4> ne_kernel = {3, 3, 3, 1}, // [kernel_width, kernel_height, input_channels, 1]
            int s0 = 1, int s1 = 1,
            int p0 = 1, int p1 = 1,
            int d0 = 1, int d1 = 1,
            bool is_2D = true)
        : type_input(type_input), type_kernel(type_kernel), dst_type(dst_type), ne_input(ne_input), ne_kernel(ne_kernel), s0(s0), s1(s1), p0(p0), p1(p1), d0(d0), d1(d1), is_2D(is_2D) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, type_input, 4, ne_input.data());
        ggml_tensor * kernel = ggml_new_tensor(ctx, type_kernel, 4, ne_kernel.data());
        ggml_tensor * out = ggml_im2col(ctx, kernel, input, s0, s1, p0, p1, d0, d1, is_2D, dst_type);
        return out;
    }
};

// GGML_OP_CONCAT
struct test_concat : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int64_t ne_b_d;
    const int dim;
    const int v; // view (1 << 0: non-cont a, 1 << 1: non-cont b)

    std::string vars() override {
        return VARS_TO_STR5(type, ne_a, ne_b_d, dim, v);
    }

    test_concat(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {10, 10, 10, 10},
            int64_t ne_b_d = 10,
            int dim = 2, int v = 0)
        : type(type), ne_a(ne_a), ne_b_d(ne_b_d), dim(dim), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        auto ne_b = ne_a;
        ne_b[dim] = ne_b_d;
        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 2; ne[1] *= 4; ne[2] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        }
        ggml_tensor * b;
        if (v & 2) {
            auto ne = ne_b; ne[0] *= 3; ne[1] *= 2; ne[2] *= 4;
            b = ggml_new_tensor(ctx, type, 4, ne.data());
            b = ggml_view_4d(ctx, b, ne_b[0], ne_b[1], ne_b[2], ne_b[3], b->nb[1], b->nb[2], b->nb[3], 0);
        } else {
            b = ggml_new_tensor(ctx, type, 4, ne_b.data());
        }
        ggml_tensor * out = ggml_concat(ctx, a, b, dim);
        return out;
    }
};

// GGML_OP_ARGSORT
struct test_argsort : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    ggml_sort_order order;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, order);
    }

    test_argsort(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {16, 10, 10, 10},
            ggml_sort_order order = GGML_SORT_ORDER_ASC)
        : type(type), ne(ne), order(order) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_argsort(ctx, a, order);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // indices
                std::vector<int> data(ggml_nelements(t));
                for (int i = 0; i < ggml_nelements(t); i++) {
                    data[i] = rand();
                }
                std::shuffle(data.begin(), data.end(), rng);
                ggml_backend_tensor_set(t, data.data(), 0, ne[0]*ne[1]*ne[2]*ne[3] * sizeof(int));
            } else if (t->type == GGML_TYPE_F32) {
                // initialize with unique values to avoid ties
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<float> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(float));
                }
            } else {
                GGML_ABORT("fatal error");
            }
        }
    }
};

// GGML_OP_SUM_ROWS
struct test_sum_rows : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_sum_rows(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_sum_rows(ctx, a);
        return out;
    }
};

// GGML_OP_UPSCALE
struct test_upscale : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int32_t scale_factor;
    const bool transpose;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, scale_factor, transpose);
    }

    test_upscale(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {512, 512, 3, 1},
            int32_t scale_factor = 2, bool transpose = false)
        : type(type), ne(ne), scale_factor(scale_factor), transpose(transpose) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        if (transpose) a = ggml_transpose(ctx, a);
        ggml_tensor * out = ggml_upscale(ctx, a, scale_factor);
        return out;
    }
};

// GGML_OP_UPSCALE (ext)
struct test_upscale_ext : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> ne_tgt;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, ne_tgt);
    }

    test_upscale_ext(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne     = {2, 5,  7, 11},
            std::array<int64_t, 4> ne_tgt = {5, 7, 11, 13})
        : type(type), ne(ne), ne_tgt(ne_tgt) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_upscale_ext(ctx, a, ne_tgt[0], ne_tgt[1],ne_tgt[2], ne_tgt[3]);
        return out;
    }
};

// GGML_OP_GROUP_NORM
struct test_group_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int32_t num_groups;
    const float eps;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, num_groups);
    }

    test_group_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 64, 320, 1},
            int32_t num_groups = 32,
            float eps = 1e-6f)
        : type(type), ne(ne), num_groups(num_groups), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_group_norm(ctx, a, num_groups, eps);
        return out;
    }
};

// GGML_OP_ACC
struct test_acc : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const std::array<int64_t, 4> ne_b;

    std::string vars() override {
        return VARS_TO_STR3(type, ne_a, ne_b);
    }

    test_acc(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {1024, 577, 1, 1},
            std::array<int64_t, 4> ne_b = {1024, 576, 1, 1})
        : type(type), ne_a(ne_a), ne_b(ne_b) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne_b.data());
        ggml_tensor * out = ggml_acc(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], b->nb[1]);
        return out;
    }
};

// GGML_OP_PAD
struct test_pad : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int pad_0;
    const int pad_1;

    std::string vars() override {
        return VARS_TO_STR4(type, ne_a, pad_0, pad_1);
    }

    test_pad(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {512, 512, 1, 1},
            int pad_0 = 1, int pad_1 = 1)
        : type(type), ne_a(ne_a), pad_0(pad_0), pad_1(pad_1)  {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_tensor * out = ggml_pad(ctx, a, pad_0, pad_1, 0, 0);
        return out;
    }
};

// GGML_OP_ARANGE
struct test_arange : public test_case {
    const ggml_type type;
    const float start;
    const float stop;
    const float step;

    std::string vars() override {
        return VARS_TO_STR4(type, start, stop, step);
    }

    test_arange(ggml_type type = GGML_TYPE_F32,
            float start = 0.f, float stop = 10.f, float step = 1.f)
        : type(type), start(start), stop(stop), step(step)  {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * out = ggml_arange(ctx, start, stop, step);
        return out;
    }
};

// GGML_OP_TIMESTEP_EMBEDDING
struct test_timestep_embedding : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int dim;
    const int max_period;

    std::string vars() override {
        return VARS_TO_STR4(type, ne_a, dim, max_period);
    }

    test_timestep_embedding(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {2, 1, 1, 1},
            int dim = 320, int max_period=10000)
        : type(type), ne_a(ne_a), dim(dim), max_period(max_period)  {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_tensor * out = ggml_timestep_embedding(ctx, a, dim, max_period);
        return out;
    }
};

// GGML_OP_LEAKY_RELU
struct test_leaky_relu : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const float negative_slope;

    std::string vars() override {
        return VARS_TO_STR3(type, ne_a, negative_slope);
    }

    test_leaky_relu(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {10, 10, 10, 10},
            float negative_slope = 0.1f)
        : type(type), ne_a(ne_a), negative_slope(negative_slope)  {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_tensor * out = ggml_leaky_relu(ctx, a, negative_slope, true);
        return out;
    }
};

// GGML_OP_FLASH_ATTN_EXT
struct test_flash_attn_ext : public test_case {
    const int64_t hs; // head size
    const int64_t nh; // num heads
    const int64_t kv; // kv size
    const int64_t nb; // batch size

    const bool mask; // use mask

    const float max_bias; // ALiBi
    const float logit_softcap; // Gemma 2

    const ggml_type type_KV;

    std::string vars() override {
        return VARS_TO_STR8(hs, nh, kv, nb, mask, max_bias, logit_softcap, type_KV);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    test_flash_attn_ext(int64_t hs = 128, int64_t nh = 32, int64_t kv = 96, int64_t nb = 8, bool mask = true, float max_bias = 0.0f, float logit_softcap = 0.0f, ggml_type type_KV = GGML_TYPE_F16)
        : hs(hs), nh(nh), kv(kv), nb(nb), mask(mask), max_bias(max_bias), logit_softcap(logit_softcap), type_KV(type_KV) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t hs_padded = GGML_PAD(hs, ggml_blck_size(type_KV));

        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs_padded, nb, nh, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, type_KV,       hs_padded, kv, nh, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, type_KV,       hs_padded, kv, nh, 1);
        ggml_tensor * m = mask ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, GGML_PAD(nb, GGML_KQ_MASK_PAD), 1, 1) : nullptr;
        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf(hs), max_bias, logit_softcap);
        return out;
    }
};

// GGML_OP_CROSS_ENTROPY_LOSS
struct test_cross_entropy_loss : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_cross_entropy_loss(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * logits = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * labels = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_cross_entropy_loss(ctx, logits, labels);
        return out;
    }
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
};

struct llama_hparams {
    uint32_t n_vocab;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    static constexpr uint32_t n_layer = 1;
    uint32_t n_rot;
    uint32_t n_embd_head; // dimension of values (d_v)
    uint32_t n_ff;

    float f_norm_eps;
    float f_norm_rms_eps;

    // cparams
    static constexpr uint32_t n_ctx = 512; // user-specified context size
    static constexpr uint32_t n_ctx_orig = n_ctx;

    // batch
    int32_t n_tokens;

    // llm_build_context
    static constexpr int32_t n_kv    = 32; // size of KV cache to consider (n_kv <= n_ctx
    static constexpr int32_t kv_head = 1;  // index of where we store new KV data in the cache

    uint32_t n_embd_gqa() const { // dimension of key embeddings across all k-v heads
        return n_embd_head * n_head_kv;
    }
};

// LLM base class
struct test_llm : public test_case {
    llama_hparams hp;

protected:
    test_llm(llama_hparams hp)
        : hp(std::move(hp)) {
    }

public:
    struct ggml_tensor * llm_build_norm(
            struct ggml_context * ctx,
             struct ggml_tensor * cur,
             struct ggml_tensor * mw,
             struct ggml_tensor * mb,
                  llm_norm_type   type) {
        switch (type) {
            case LLM_NORM:     cur = ggml_norm    (ctx, cur, hp.f_norm_eps); break;
            case LLM_NORM_RMS: cur = ggml_rms_norm(ctx, cur, hp.f_norm_rms_eps); break;
        }
        cur = ggml_mul(ctx, cur, mw);
        if (mb) {
            cur = ggml_add(ctx, cur, mb);
        }
        return cur;
    }

    void llm_build_kv_store(
            struct ggml_context * ctx,
             struct ggml_tensor * k_l,
             struct ggml_tensor * v_l,
             struct ggml_tensor * k_cur,
             struct ggml_tensor * v_cur) {
        // compute the transposed [n_tokens, n_embd] V matrix
        struct ggml_tensor * v_cur_t = ggml_transpose(ctx, ggml_reshape_2d(ctx, v_cur, hp.n_embd_gqa(), hp.n_tokens));

        struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, k_l, hp.n_tokens*hp.n_embd_gqa(),
                (ggml_row_size(k_l->type, hp.n_embd_gqa()))*hp.kv_head);

        struct ggml_tensor * v_cache_view = ggml_view_2d(ctx, v_l, hp.n_tokens, hp.n_embd_gqa(),
                (  hp.n_ctx)*ggml_element_size(v_l),
                (hp.kv_head)*ggml_element_size(v_l));

        // important: storing RoPE-ed version of K in the KV cache!
        ggml_cpy(ctx, k_cur,   k_cache_view);
        ggml_cpy(ctx, v_cur_t, v_cache_view);
    }

    struct ggml_tensor * llm_build_kqv(
            struct ggml_context * ctx,
             struct ggml_tensor * k_l,
             struct ggml_tensor * v_l,
             struct ggml_tensor * q_cur,
             struct ggml_tensor * kq_mask,
                        float     kq_scale) {
        struct ggml_tensor * q = ggml_permute(ctx, q_cur, 0, 2, 1, 3);

        struct ggml_tensor * k =
            ggml_view_3d(ctx, k_l,
                    hp.n_embd_head, hp.n_kv, hp.n_head_kv,
                    ggml_row_size(k_l->type, hp.n_embd_gqa()),
                    ggml_row_size(k_l->type, hp.n_embd_head),
                    0);

        struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);

        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0f);

        // split cached v into n_head heads
        struct ggml_tensor * v =
            ggml_view_3d(ctx, v_l,
                    hp.n_kv, hp.n_embd_head, hp.n_head_kv,
                    ggml_element_size(v_l)*hp.n_ctx,
                    ggml_element_size(v_l)*hp.n_ctx*hp.n_embd_head,
                    0);

        struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);

        struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);

        struct ggml_tensor * cur = ggml_cont_2d(ctx, kqv_merged, hp.n_embd_head*hp.n_head, hp.n_tokens);

        struct ggml_tensor * wo = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd);
        cur = ggml_mul_mat(ctx, wo, cur);

        return cur;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // pos
                std::vector<int> data(hp.n_tokens);
                for (int i = 0; i < hp.n_tokens; i++) {
                    data[i] = rand() % hp.n_ctx;
                }
                ggml_backend_tensor_set(t, data.data(), 0, hp.n_tokens * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// Llama
struct test_llama : public test_llm {
    static constexpr float freq_base = 10000.0f;
    static constexpr float freq_scale = 1.0f;
    static constexpr float ext_factor = 0.0f;
    static constexpr float attn_factor = 1.0f;
    static constexpr float beta_fast = 32.0f;
    static constexpr float beta_slow = 1.0f;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "LLAMA";
    }

    std::string vars() override {
        auto n_tokens = hp.n_tokens;
        return VARS_TO_STR1(n_tokens);
    }

    double max_nmse_err() override {
        return 2e-3;
    }

    test_llama(int n_tokens = 1)
        : test_llm({
            /*n_vocab        =*/ 32000,
            /*n_embd         =*/ 3200,
            /*n_head         =*/ 32,
            /*n_head_kv      =*/ 32,
            /*n_rot          =*/ 100,
            /*n_embd_head    =*/ 100,
            /*n_ff           =*/ 8640,
            /*f_norm_eps     =*/ 0.f,
            /*f_norm_rms_eps =*/ 1e-5f,
            /*n_tokens       =*/ n_tokens,
        }) {
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, hp.n_tokens);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, hp.n_tokens);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hp.n_kv, hp.n_tokens, 1);

        ggml_tensor * k_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);
        ggml_tensor * v_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);

        for (uint32_t il = 0; il < hp.n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            ggml_tensor * attn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            cur = llm_build_norm(ctx, inpL, attn_norm, nullptr, LLM_NORM_RMS);

            // self-attention
            {
                ggml_tensor * wq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd);
                ggml_tensor * wk = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd_gqa());
                ggml_tensor * wv = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd_gqa());

                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx, wq, cur);
                struct ggml_tensor * Kcur = ggml_mul_mat(ctx, wk, cur);
                struct ggml_tensor * Vcur = ggml_mul_mat(ctx, wv, cur);

                Qcur = ggml_rope_ext(
                    ctx, ggml_reshape_3d(ctx, Qcur, hp.n_embd_head, hp.n_head,    hp.n_tokens), inp_pos, nullptr,
                    hp.n_rot, 0, hp.n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );

                Kcur = ggml_rope_ext(
                    ctx, ggml_reshape_3d(ctx, Kcur, hp.n_embd_head, hp.n_head_kv, hp.n_tokens), inp_pos, nullptr,
                    hp.n_rot, 0, hp.n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );

                llm_build_kv_store(ctx, k_l, v_l, Kcur, Vcur);

                cur = llm_build_kqv(ctx, k_l, v_l, Qcur, KQ_mask, 1.0f/sqrtf(float(hp.n_embd_head)));
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx, cur, inpSA);

            // feed-forward network
            ggml_tensor * ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            cur = llm_build_norm(ctx, ffn_inp, ffn_norm, nullptr, LLM_NORM_RMS);

            ggml_tensor * ffn_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
            ggml_tensor * ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_ff,   hp.n_embd);
            ggml_tensor * ffn_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
            struct ggml_tensor * tmp = ggml_mul_mat(ctx, ffn_up, cur);
            cur = ggml_mul_mat(ctx, ffn_gate, cur);
            cur = ggml_silu(ctx, cur);
            cur = ggml_mul(ctx, cur, tmp);
            cur = ggml_mul_mat(ctx, ffn_down, cur);

            cur = ggml_add(ctx, cur, ffn_inp);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        ggml_tensor * output_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
        cur = llm_build_norm(ctx, cur, output_norm, nullptr, LLM_NORM_RMS);

        // lm_head
        ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_vocab);
        cur = ggml_mul_mat(ctx, output, cur);

        return cur;
    }
};

// Falcon
struct test_falcon : public test_llm {
    static constexpr float freq_base = 10000.0f;
    static constexpr float freq_scale = 1.0f;
    static constexpr float ext_factor = 0.0f;
    static constexpr float attn_factor = 1.0f;
    static constexpr float beta_fast = 32.0f;
    static constexpr float beta_slow = 1.0f;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "FALCON";
    }

    std::string vars() override {
        auto n_tokens = hp.n_tokens;
        return VARS_TO_STR1(n_tokens);
    }

    double max_nmse_err() override {
        return 2e-3;
    }

    test_falcon(int n_tokens = 1)
        : test_llm({
            /*n_vocab        =*/ 32000,
            /*n_embd         =*/ 3200,
            /*n_head         =*/ 50,
            /*n_head_kv      =*/ 1,
            /*n_rot          =*/ 64,
            /*n_embd_head    =*/ 64,
            /*n_ff           =*/ 8640,
            /*f_norm_eps     =*/ 1e-5f,
            /*f_norm_rms_eps =*/ 0.f,
            /*n_tokens       =*/ n_tokens,
        }) {
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, hp.n_tokens);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, hp.n_tokens);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hp.n_kv, hp.n_tokens, 1);

        ggml_tensor * k_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);
        ggml_tensor * v_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);

        for (uint32_t il = 0; il < hp.n_layer; ++il) {
            // norm
            ggml_tensor * attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            ggml_tensor * attn_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            ggml_tensor * attn_norm = llm_build_norm(ctx, inpL, attn_norm_w, attn_norm_b, LLM_NORM);

            // self-attention
            {
                cur = attn_norm;

                ggml_tensor * wqkv = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd + 2*hp.n_embd_gqa());

                cur = ggml_mul_mat(ctx, wqkv, cur);

                struct ggml_tensor * Qcur = ggml_cont(ctx, ggml_view_2d(ctx, cur, hp.n_embd,     hp.n_tokens, cur->nb[1], 0*sizeof(float)*(hp.n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx, ggml_view_2d(ctx, cur, hp.n_embd_gqa(), hp.n_tokens, cur->nb[1], 1*sizeof(float)*(hp.n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx, ggml_view_2d(ctx, cur, hp.n_embd_gqa(), hp.n_tokens, cur->nb[1], 1*sizeof(float)*(hp.n_embd + hp.n_embd_gqa())));

                Qcur = ggml_reshape_3d(ctx, Qcur, hp.n_embd_head, hp.n_head,    hp.n_tokens);
                Kcur = ggml_reshape_3d(ctx, Kcur, hp.n_embd_head, hp.n_head_kv, hp.n_tokens);

                // using mode = 2 for neox mode
                Qcur = ggml_rope_ext(
                    ctx, Qcur, inp_pos, nullptr, hp.n_rot, 2, hp.n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );

                Kcur = ggml_rope_ext(
                    ctx, Kcur, inp_pos, nullptr, hp.n_rot, 2, hp.n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );

                llm_build_kv_store(ctx, k_l, v_l, Kcur, Vcur);

                cur = llm_build_kqv(ctx, k_l, v_l, Qcur, KQ_mask, 1.0f/sqrtf(float(hp.n_embd_head)));
            }

            struct ggml_tensor * ffn_inp = cur;

            // feed forward
            {
                ggml_tensor * ffn_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
                ggml_tensor * ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_ff, hp.n_embd);
                cur = attn_norm;
                cur = ggml_mul_mat(ctx, ffn_up, cur);
                cur = ggml_gelu(ctx, cur);
                cur = ggml_mul_mat(ctx, ffn_down, cur);
            }

            cur = ggml_add(ctx, cur, ffn_inp);

            cur = ggml_add(ctx, cur, inpL);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        ggml_tensor * output_norm   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
        ggml_tensor * output_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
        cur = llm_build_norm(ctx, cur, output_norm, output_norm_b, LLM_NORM);

        // lm_head
        ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, hp.n_embd, hp.n_vocab);
        cur = ggml_mul_mat(ctx, output, cur);

        return cur;
    }
};

static bool test_backend(ggml_backend_t backend, test_mode mode, const char * op_name) {
    std::vector<std::unique_ptr<test_case>> test_cases;
    std::default_random_engine rng(0);

    const ggml_type all_types[] = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
        GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS,
    };

    const ggml_type base_types[] = {
        GGML_TYPE_F32, GGML_TYPE_F16,
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_K,
        GGML_TYPE_IQ2_XXS
    };

    const ggml_type other_types[] = {
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
        GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
        GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS,
        GGML_TYPE_BF16,
    };

    // unary ops
    for (int v : {0, 1}) {
        for (int op = 0; op < GGML_UNARY_OP_COUNT; op++) {
            test_cases.emplace_back(new test_unary((ggml_unary_op) op, GGML_TYPE_F32, { 128, 10, 10, 10 }, v));
            test_cases.emplace_back(new test_unary((ggml_unary_op) op, GGML_TYPE_F32, { 7, 13, 19, 23 }, v));
        }
    }

    test_cases.emplace_back(new test_get_rows(GGML_TYPE_F32, 1, 8, 2, 1, false));
    for (ggml_type type : all_types) {
        for (int b : {1, 7}) {
            for (bool v : {false, true}) {
                test_cases.emplace_back(new test_get_rows(type, 256, 5, 4, b, v));
            }
        }
    }
    for (int b : {1, 7}) {
        for (bool v : {false, true}) {
            test_cases.emplace_back(new test_get_rows(GGML_TYPE_I32, 256, 5, 4, b, v));
        }
    }

    for (ggml_type type_input : {GGML_TYPE_F32}) {
        for (ggml_op_pool pool_type : {GGML_OP_POOL_AVG, GGML_OP_POOL_MAX}) {
            for (int k0 : {1, 3}) {
                for (int k1 : {1, 3}) {
                    for (int s0 : {1, 2}) {
                        for (int s1 : {1, 2}) {
                            for (int p0 : {0, 1}) {
                                for (int p1 : {0, 1}) {
                                    test_cases.emplace_back(new test_pool2d(pool_type, type_input, {10, 10, 3, 1}, k0, k1, s0, s1, p0, p1));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16));
    // test cases for 1D im2col
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {3000, 128, 1, 1}, {3, 128, 1280, 1}, 1, 0, 1, 0, 1, 0, false));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, {3000, 128, 1, 1}, {3, 128, 1280, 1}, 1, 0, 1, 0, 1, 0, false));

    // sycl backend will limit task global_range < MAX_INT
    // test cases for 2D im2col with large input W and H (occurs in stable-diffusion)
    // however these cases need to alloc more memory which may fail in some devices (Intel Arc770, etc.)
    // these cases are verified (pass) in Intel(R) Data Center GPU Max 1100 (sycl backend) and NV A30 (cuda backend)
    // test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {1024, 1024, 256, 1}, {3, 3, 256, 1}, 1, 1, 1, 1, 1, 1, true));
    // test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, {1024, 1024, 256, 1}, {3, 3, 256, 1}, 1, 1, 1, 1, 1, 1, true));

    test_cases.emplace_back(new test_conv_transpose_1d());
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {2,3,2,1}, 3, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {2,3,2,1}, 2, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {2,3,2,1}, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {3,2,2,1}, 2, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {3,2,2,1}, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {3,1,2,1}, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({2,1,1,1}, {3,1,1,1}, 1, 0, 1));


    test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 1, 1, 1}));
    test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 10, 10, 10}, {2, 1, 1, 1}));
    test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 2, 1, 1}));
    test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 1, 2, 1}));
    test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 1, 1, 2}));
    test_cases.emplace_back(new test_repeat(GGML_TYPE_I32, {10, 10, 10, 10}, {2, 1, 1, 1}));
    test_cases.emplace_back(new test_repeat(GGML_TYPE_I16, {10, 10, 10, 10}, {1, 1, 1, 2}));

    test_cases.emplace_back(new test_dup(GGML_TYPE_F32));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I32));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F32, {10, 10, 5, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16, {10, 10, 5, 1}, {0, 2, 1, 3})); // dup by rows
    test_cases.emplace_back(new test_dup(GGML_TYPE_F32, {10, 10, 5, 1}, {1, 0, 2, 3}));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16, {10, 10, 5, 1}, {1, 0, 2, 3})); // dup dst not-contiguous
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16, {10, 8, 3, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16, {10, 8, 3, 1}, {1, 2, 0, 3}));

    for (ggml_type type_src : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (ggml_type type_dst : all_types) {
           test_cases.emplace_back(new test_cpy(type_src, type_dst, {256, 4, 4, 4}));
           test_cases.emplace_back(new test_cpy(type_src, type_dst, {256, 2, 3, 4}, {0, 2, 1, 3})); // cpy by rows
        }
    }
    for (ggml_type type_src : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (ggml_type type_dst : {GGML_TYPE_F16, GGML_TYPE_F32}) {
            test_cases.emplace_back(new test_cpy(type_src, type_dst, {256, 2, 3, 4}, {1, 0, 2, 3})); // cpy not-contiguous
        }
    }

    test_cases.emplace_back(new test_cont());

    auto add_test_bin_bcast = [&](ggml_type type, std::array<int64_t, 4> ne, std::array<int, 4> nr) {
        for (auto op : {ggml_add, ggml_mul, ggml_div}) {
            test_cases.emplace_back(new test_bin_bcast(op, type, ne, nr));
        }
    };

    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 8, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 1, 1}, {32, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 320, 320}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 1, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {2, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 2, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 1, 2, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 1, 1, 2});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 1, 2, 2});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 2, 2, 2});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {2, 2, 2, 2});

    // stable diffusion
    add_test_bin_bcast(GGML_TYPE_F32, {1280, 1, 1, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1280, 1, 1, 1}, {1, 16, 16, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1280, 16, 16, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1280, 1, 1, 1}, {1, 256, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 1280, 1}, {16, 16, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 16, 1280, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 1920, 1}, {16, 16, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 2560, 1}, {16, 16, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 1280, 1}, {32, 32, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 1920, 1}, {32, 32, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 640, 1}, {32, 32, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {5120, 1, 1, 1}, {1, 256, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {640, 1, 1, 1}, {1, 1, 1, 1});
    //add_test_bin_bcast(GGML_TYPE_F32, {3, 3, 2560, 1280}, {1, 1, 1, 1});
    //add_test_bin_bcast(GGML_TYPE_F32, {3, 3, 2560, 1280}, {2, 1, 1, 1});

    test_cases.emplace_back(new test_scale());

    for (float eps : {1e-6f, 1e-5f, 1e-3f, 1e-1f}) {
        test_cases.emplace_back(new test_norm(GGML_TYPE_F32, {64, 10, 10, 10}, eps));
        test_cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, {64, 10, 10, 10}, eps));
    }

    test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {4, 1536, 1, 1}, {4, 1536, 1, 1}));
    test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {8, 1536, 1, 1}, {4, 1536, 1, 1}));
    test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {4, 1536, 4, 1}, {4, 1536, 1, 1}));

    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 16, 1024, 32, 4));

#if 1
    for (ggml_type type_a : base_types) {
        for (ggml_type type_b : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 1,  1}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10,  1}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10,  1}, {2, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10, 10}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10, 10}, {2, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10, 10}, {1, 2}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10, 10}, {2, 2}));

            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 1,  1}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10,  1}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10,  1}, {2, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10, 10}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10, 10}, {2, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10, 10}, {1, 2}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10, 10}, {2, 2}));
        }
    }
#else
    // m = a rows
    // n = b rows
    // k = cols
    std::uniform_int_distribution<> dist_m(1, 128);
    std::uniform_int_distribution<> dist_n(16, 128);
    std::uniform_int_distribution<> dist_k(1, 16);
    for (int i = 0; i < 1000; i++) {
        for (ggml_type type_a : all_types) {
            for (ggml_type type_b : {GGML_TYPE_F32}) {
                int m = dist_m(rng);
                int n = dist_n(rng);
                int k = dist_k(rng) * ggml_blck_size(type_a);
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, m, n, k, { 1,  1}, {1, 1}));
            }
        }
    }
#endif

    for (ggml_type type_a : other_types) {
        for (ggml_type type_b : {GGML_TYPE_F32}) {
            if (ggml_blck_size(type_a) != 256) {
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, ggml_blck_size(type_a), {1,  1}, {1, 1}));
            }
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {1,  1}, {1, 1}));
        }
    }

    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  64, 2,  128, { 8,  1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  83, 2,  128, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  64, 2,   64, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  83, 2,   64, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  64, 45, 128, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 128, 45,  64, { 8,  1}, {4, 1}));

    // sycl backend will limit task global_range < MAX_INT
    // test case for f16-type-convert-to-fp32 kernel with large k under fp32 compute dtype (occurs in stable-diffusion)
    // however this case needs to alloc more memory which may fail in some devices (Intel Arc770, etc.)
    // this case is verified (pass) in Intel(R) Data Center GPU Max 1100 (sycl backend) and NV A30 (cuda backend)
    // test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F16, 512, 262144, 9216, {1, 1}, {1, 1}));

    for (ggml_type type_a : base_types) {
        for (ggml_type type_b : {GGML_TYPE_F32 /*, GGML_TYPE_F16 */}) {
            for (int n_mats : {4, 8}) {
                for (int n_used : {1, 2, 4}) {
                    for (bool b : {false, true}) {
                        for (int n : {1, 32}) {
                            int m = 512;
                            int k = 256;
                            test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k));
                        }
                    }
                }
            }
        }
    }

    for (ggml_type type_a : other_types) {
        for (ggml_type type_b : {GGML_TYPE_F32 /*, GGML_TYPE_F16 */}) {
            for (int n_mats : {4}) {
                for (int n_used : {2}) {
                    for (bool b : {false}) {
                        for (int n : {1}) {
                            int m = 512;
                            int k = 256;
                            test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k));
                        }
                    }
                }
            }
        }
    }

    test_cases.emplace_back(new test_sqr());
    test_cases.emplace_back(new test_sqrt());
    test_cases.emplace_back(new test_sin());
    test_cases.emplace_back(new test_cos());
    test_cases.emplace_back(new test_clamp());

    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, {10, 10,  1,  1}, 5));
    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, {10, 10, 10,  1}, 5));
    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, {10, 10, 10, 10}, 5));

#if 0
    std::uniform_int_distribution<> dist_ne1(1, 50);
    int exponent = 1;
    while (exponent < (1 << 17)) {
        std::uniform_int_distribution<> dist_ne0(exponent, 2*exponent);

        for (int n = 0; n < 10; ++n) {
            int64_t ne0 = dist_ne0(rng);
            int64_t ne1 = dist_ne1(rng);
            test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, GGML_TYPE_F32, {ne0, ne1, 1, 1}, n/2 == 0, 0.1f, ne0 < 1000 ? 4.0f : 0.0f));
        }

        exponent <<= 1;
    }
#endif
    for (bool mask : {false, true}) {
        for (float max_bias : {0.0f, 8.0f}) {
            if (!mask && max_bias > 0.0f) continue;
            for (float scale : {1.0f, 0.1f}) {
                for (int64_t ne0 : {16, 1024}) {
                    for (int64_t ne1 : {16, 1024}) {
                        test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {ne0,   ne1,   1, 1}, mask, scale, max_bias));
                        test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {ne0-1, ne1-1, 1, 1}, mask, scale, max_bias));
                    }
                }
            }
        }
    }
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {16, 2, 32, 1}, true, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {16, 2, 32, 1}, false, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {32, 2, 32, 1}, true,  0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {32, 2, 32, 1}, true,  0.1f, 8.0f));

    {
        bool all = true;

        for (float v : { 0, 1 }) {
            for (float fs : { 1.0f, 1.4245f }) {
                for (float ef : { 0.0f, 0.7465f }) {
                    for (float af : { 1.0f, 1.4245f }) {
                        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                            for (bool ff : {false, true}) { // freq_factors
                                test_cases.emplace_back(new test_rope(type, {128,  32, 10, 1}, 128, 0, 512, fs, ef, af, ff, v)); // llama 7B

                                if (all) {
                                    test_cases.emplace_back(new test_rope(type, {128,  40, 10, 1}, 128, 0, 512, fs, ef, af, ff, v)); // llama 13B
                                    test_cases.emplace_back(new test_rope(type, {128,  52, 10, 1}, 128, 0, 512, fs, ef, af, ff, v)); // llama 30B
                                    test_cases.emplace_back(new test_rope(type, {128,  64, 10, 1}, 128, 0, 512, fs, ef, af, ff, v)); // llama 65B
                                }

                                if (all) {
                                    test_cases.emplace_back(new test_rope(type, { 64,   1, 10, 1},  64, 2, 512, fs, ef, af, ff, v)); // neox (falcon 7B)
                                    test_cases.emplace_back(new test_rope(type, { 64,  71, 10, 1},  64, 2, 512, fs, ef, af, ff, v)); // neox (falcon 7B)
                                    test_cases.emplace_back(new test_rope(type, { 64,   8, 10, 1},  64, 2, 512, fs, ef, af, ff, v)); // neox (falcon 40B)
                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 10, 1},  20, 2, 512, fs, ef, af, ff, v)); // neox (stablelm)
                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 10, 1},  32, 2, 512, fs, ef, af, ff, v)); // neox (phi-2)
                                }

                                test_cases.emplace_back(new test_rope(type, { 64, 128, 10, 1},  64, 2, 512, fs, ef, af, ff, v)); // neox (falcon 40B)
                            }
                        }

                        all = false;
                    }
                }
            }
        }
    }

    for (int v : { 0, 1, 2, 3 }) {
        for (int dim : { 0, 1, 2, 3, }) {
            test_cases.emplace_back(new test_concat(GGML_TYPE_F32, {11, 12, 13, 14}, 7, dim, v));
            test_cases.emplace_back(new test_concat(GGML_TYPE_I32, {11, 12, 13, 14}, 7, dim, v));
        }
    }

    for (ggml_sort_order order : {GGML_SORT_ORDER_ASC, GGML_SORT_ORDER_DESC}) {
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {8, 1, 1, 1}, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {16, 10, 10, 10}, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {60, 10, 10, 10}, order)); // qwen
    }

    test_cases.emplace_back(new test_sum_rows());
    test_cases.emplace_back(new test_upscale());
    test_cases.emplace_back(new test_upscale(GGML_TYPE_F32, { 512, 512, 3, 1 }, 2, true));
    test_cases.emplace_back(new test_upscale_ext());
    test_cases.emplace_back(new test_group_norm());
    test_cases.emplace_back(new test_acc());
    test_cases.emplace_back(new test_pad());
    test_cases.emplace_back(new test_arange());
    test_cases.emplace_back(new test_timestep_embedding());
    test_cases.emplace_back(new test_leaky_relu());

    for (int hs : { 64, 80, 128, 256, }) {
        for (bool mask : { true, false } ) {
            for (float max_bias : { 0.0f, 8.0f }) {
                if (!mask && max_bias > 0.0f) continue;
                for (float logit_softcap : {0.0f, 10.0f}) {
                    if (hs != 128 && logit_softcap != 0.0f) continue;
                    for (int nh : { 32, }) {
                        for (int kv : { 512, 1024, }) {
                            for (int nb : { 1, 2, 4, 8, }) {
                                for (ggml_type type_KV : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
                                    test_cases.emplace_back(new test_flash_attn_ext(hs, nh, kv, nb, mask, max_bias, logit_softcap, type_KV));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    test_cases.emplace_back(new test_cross_entropy_loss());

    // these tests are disabled to save execution time, but they can be handy for debugging
#if 0
    test_cases.emplace_back(new test_llama(1));
    test_cases.emplace_back(new test_llama(2));
    test_cases.emplace_back(new test_falcon(1));
    test_cases.emplace_back(new test_falcon(2));
#endif

    // run tests
    if (mode == MODE_TEST) {
        ggml_backend_t backend_cpu = ggml_backend_cpu_init();

        size_t n_ok = 0;
        for (auto & test : test_cases) {
            if (test->eval(backend, backend_cpu, op_name)) {
                n_ok++;
            }
        }
        printf("  %zu/%zu tests passed\n", n_ok, test_cases.size());

        ggml_backend_free(backend_cpu);

        return n_ok == test_cases.size();
    }

    if (mode == MODE_PERF) {
        for (auto & test : test_cases) {
            test->eval_perf(backend, op_name);
        }
        return true;
    }

    GGML_ABORT("fatal error");
}

static void usage(char ** argv) {
    printf("Usage: %s [mode] [-o op] [-b backend]\n", argv[0]);
    printf("  valid modes are: test (compare with CPU backend for correctness) or perf (performance evaluation)\n");
    printf("  op names are as given by ggml_op_desc()\n");
}

int main(int argc, char ** argv) {
    test_mode mode = MODE_TEST;
    const char * op_name_filter = NULL;
    const char * backend_filter = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "test") == 0) {
            mode = MODE_TEST;
        } else if (strcmp(argv[i], "perf") == 0) {
            mode = MODE_PERF;
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                op_name_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                backend_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else {
            usage(argv);
            return 1;
        }
    }

    // enumerate backends
    printf("Testing %zu backends\n\n", ggml_backend_reg_get_count());

    size_t n_ok = 0;

    for (size_t i = 0; i < ggml_backend_reg_get_count(); i++) {
        printf("Backend %zu/%zu (%s)\n", i + 1, ggml_backend_reg_get_count(), ggml_backend_reg_get_name(i));

        if (backend_filter != NULL && strcmp(backend_filter, ggml_backend_reg_get_name(i)) != 0) {
            printf("  Skipping\n");
            n_ok++;
            continue;
        }

        ggml_backend_t backend = ggml_backend_reg_init_backend(i, NULL);
        GGML_ASSERT(backend != NULL);

        if (backend_filter == NULL && ggml_backend_is_cpu(backend)) {
            printf("  Skipping CPU backend\n");
            ggml_backend_free(backend);
            n_ok++;
            continue;
        }

        printf("  Backend name: %s\n", ggml_backend_name(backend));

        bool ok = test_backend(backend, mode, op_name_filter);

        printf("  Backend %s: ", ggml_backend_name(backend));
        if (ok) {
            printf("\033[1;32mOK\033[0m\n");
            n_ok++;
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }

        printf("\n");

        ggml_backend_free(backend);
    }

    printf("%zu/%zu backends passed\n", n_ok, ggml_backend_reg_get_count());

    if (n_ok != ggml_backend_reg_get_count()) {
        printf("\033[1;31mFAIL\033[0m\n");
        return 1;
    }

    ggml_quantize_free();

    printf("\033[1;32mOK\033[0m\n");
    return 0;
}
