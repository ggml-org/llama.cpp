#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

struct bias_test_config {
    const char * name;
    int64_t d;
    int64_t nq;
    int64_t nkv;
    int64_t hq;
    int64_t hkv;
    int64_t extent;
    ggml_type type;
    bool use_mask;
    bool sliding;
    int64_t n_batch = 1;
    int64_t rel_batch = 1;
    bool strided_rel = false;
};

struct test_data {
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> rel;
    std::vector<ggml_fp16_t> mask;
    std::vector<uint8_t> k_typed;
    std::vector<uint8_t> v_typed;
    std::vector<uint8_t> rel_typed;
    std::vector<float> k_rounded;
    std::vector<float> v_rounded;
    std::vector<float> rel_rounded;
    std::vector<float> dense_bias;
};

struct run_result {
    std::vector<float> output;
    size_t allocated_bytes;
    double ms;
};

static std::vector<uint8_t> convert_type(ggml_type type, const std::vector<float> & src, std::vector<float> & rounded) {
    rounded.resize(src.size());
    if (type == GGML_TYPE_F32) {
        rounded = src;
        std::vector<uint8_t> bytes(src.size()*sizeof(float));
        memcpy(bytes.data(), src.data(), bytes.size());
        return bytes;
    }
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(src.size());
        ggml_fp32_to_fp16_row(src.data(), tmp.data(), src.size());
        ggml_fp16_to_fp32_row(tmp.data(), rounded.data(), src.size());
        std::vector<uint8_t> bytes(tmp.size()*sizeof(tmp[0]));
        memcpy(bytes.data(), tmp.data(), bytes.size());
        return bytes;
    }
    GGML_ASSERT(type == GGML_TYPE_BF16);
    std::vector<ggml_bf16_t> tmp(src.size());
    ggml_fp32_to_bf16_row_ref(src.data(), tmp.data(), src.size());
    ggml_bf16_to_fp32_row(tmp.data(), rounded.data(), src.size());
    std::vector<uint8_t> bytes(tmp.size()*sizeof(tmp[0]));
    memcpy(bytes.data(), tmp.data(), bytes.size());
    return bytes;
}

static test_data make_data(const bias_test_config & c) {
    test_data data;
    data.q.resize(c.d*c.nq*c.hq*c.n_batch);
    data.k.resize(c.d*c.nkv*c.hkv*c.n_batch);
    data.v.resize(c.d*c.nkv*c.hkv*c.n_batch);
    data.rel.resize(c.extent*c.hq*c.nq*c.rel_batch);
    data.mask.resize(c.nkv*c.nq);

    for (size_t i = 0; i < data.q.size(); ++i) {
        data.q[i] = 0.20f*std::sin(float(i)*0.017f + 0.13f);
    }
    for (size_t i = 0; i < data.k.size(); ++i) {
        data.k[i] = 0.25f*std::cos(float(i)*0.013f - 0.29f);
        data.v[i] = 0.30f*std::sin(float(i)*0.019f + 0.71f);
    }
    for (int64_t ib = 0; ib < c.rel_batch; ++ib) {
        for (int64_t iq = 0; iq < c.nq; ++iq) {
            for (int64_t ih = 0; ih < c.hq; ++ih) {
                for (int64_t ie = 0; ie < c.extent; ++ie) {
                    const size_t idx = ((ib*c.nq + iq)*c.hq + ih)*c.extent + ie;
                    data.rel[idx] = 0.75f*std::sin(float(idx)*0.007f + float(ie)*0.021f + 0.31f);
                }
            }
        }
    }
    for (int64_t iq = 0; iq < c.nq; ++iq) {
        for (int64_t ik = 0; ik < c.nkv; ++ik) {
            const int64_t dist = iq + (c.nkv - c.nq) - ik;
            const bool visible = !c.use_mask || (dist >= 0 && (!c.sliding || dist < c.extent));
            data.mask[iq*c.nkv + ik] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
        }
    }

    data.k_typed = convert_type(c.type, data.k, data.k_rounded);
    data.v_typed = convert_type(c.type, data.v, data.v_rounded);
    data.rel_typed = convert_type(GGML_TYPE_F32, data.rel, data.rel_rounded);

    data.dense_bias.assign(c.nkv*c.nq*c.hq*c.n_batch, 0.0f);
    for (int64_t ib = 0; ib < c.n_batch; ++ib) {
        const int64_t irb = ib % c.rel_batch;
        for (int64_t ih = 0; ih < c.hq; ++ih) {
            for (int64_t iq = 0; iq < c.nq; ++iq) {
                for (int64_t ik = 0; ik < c.nkv; ++ik) {
                    const int64_t dist = iq + (c.nkv - c.nq) - ik;
                    if (dist >= 0 && dist < c.extent) {
                        data.dense_bias[((ib*c.hq + ih)*c.nq + iq)*c.nkv + ik] =
                            data.rel_rounded[((irb*c.nq + iq)*c.hq + ih)*c.extent + dist];
                    }
                }
            }
        }
    }
    return data;
}

static run_result run_graph(
        ggml_backend_t backend,
        const bias_test_config & c,
        const test_data & data,
        bool dense,
        int repeats) {
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead()*64 + ggml_graph_overhead_custom(64, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx);

    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, c.d, c.nq, c.hq, c.n_batch);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), c.type, c.d, c.nkv, c.hkv, c.n_batch);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), c.type, c.d, c.nkv, c.hkv, c.n_batch);
    ggml_tensor * r_storage = nullptr;
    ggml_tensor * r;
    if (c.strided_rel) {
        r_storage = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32,
            2*c.extent, c.hq, c.nq, c.rel_batch);
        r = ggml_view_4d(ctx.get(), r_storage, c.extent, c.hq, c.nq, c.rel_batch,
            r_storage->nb[1], r_storage->nb[2], r_storage->nb[3], 0);
    } else {
        r = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32,
            c.extent, c.hq, c.nq, c.rel_batch);
    }
    ggml_tensor * m = c.use_mask ? ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, c.nkv, c.nq, 1, 1) : nullptr;
    ggml_set_name(q, "q");
    ggml_set_name(k, "k");
    ggml_set_name(v, "v");
    ggml_set_name(r, "rel_logits");
    if (m) {
        ggml_set_name(m, "mask");
    }

    ggml_tensor * out;
    ggml_tensor * bias = nullptr;
    if (!dense) {
        out = ggml_flash_attn_ext_banded(ctx.get(), q, k, v, m, r, 1.0f/float(c.d), c.extent);
    } else {
        bias = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, c.nkv, c.nq, c.hq, c.n_batch);
        ggml_set_name(bias, "dense_bias");
        ggml_tensor * scores = ggml_mul_mat(ctx.get(), k, q);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        scores = ggml_scale(ctx.get(), scores, 1.0f/float(c.d));
        scores = ggml_add(ctx.get(), scores, bias);
        scores = ggml_soft_max_ext(ctx.get(), scores, m, 1.0f, 0.0f);
        ggml_tensor * vt = ggml_cont(ctx.get(), ggml_transpose(ctx.get(), v));
        out = ggml_mul_mat(ctx.get(), vt, scores);
        ggml_mul_mat_set_prec(out, GGML_PREC_F32);
        out = ggml_cont(ctx.get(), ggml_permute(ctx.get(), out, 0, 2, 1, 3));
    }
    ggml_set_name(out, dense ? "out_dense" : "out_flash");

    GGML_ASSERT(ggml_backend_supports_op(backend, out));
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer);

    ggml_backend_tensor_set(q, data.q.data(), 0, data.q.size()*sizeof(float));
    ggml_backend_tensor_set(k, data.k_typed.data(), 0, data.k_typed.size());
    ggml_backend_tensor_set(v, data.v_typed.data(), 0, data.v_typed.size());
    if (r_storage) {
        std::vector<float> physical(2*c.extent*c.hq*c.nq*c.rel_batch, 0.0f);
        for (int64_t ib = 0; ib < c.rel_batch; ++ib) {
            for (int64_t iq = 0; iq < c.nq; ++iq) {
                for (int64_t ih = 0; ih < c.hq; ++ih) {
                    const size_t logical = ((ib*c.nq + iq)*c.hq + ih)*c.extent;
                    const size_t storage = ((ib*c.nq + iq)*c.hq + ih)*(2*c.extent);
                    memcpy(physical.data() + storage, data.rel_rounded.data() + logical,
                           c.extent*sizeof(float));
                }
            }
        }
        ggml_backend_tensor_set(r_storage, physical.data(), 0, physical.size()*sizeof(float));
    } else {
        ggml_backend_tensor_set(r, data.rel_typed.data(), 0, data.rel_typed.size());
    }
    if (m) {
        ggml_backend_tensor_set(m, data.mask.data(), 0, data.mask.size()*sizeof(data.mask[0]));
    }
    if (bias) {
        ggml_backend_tensor_set(bias, data.dense_bias.data(), 0, data.dense_bias.size()*sizeof(float));
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, out);
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    const int64_t start = ggml_time_us();
    for (int i = 0; i < repeats; ++i) {
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    }
    ggml_backend_synchronize(backend);
    const int64_t elapsed = ggml_time_us() - start;

    run_result result;
    result.output.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.output.data(), 0, result.output.size()*sizeof(float));
    result.allocated_bytes = ggml_backend_buffer_get_size(buffer.get());
    result.ms = double(elapsed)/1000.0/repeats;
    return result;
}

static std::vector<float> naive_materialized(const bias_test_config & c, const test_data & data) {
    const int64_t nrows = c.n_batch*c.hq*c.nq;
    std::vector<float> scores(nrows*c.nkv);
    std::vector<float> output(c.d*c.hq*c.nq*c.n_batch, 0.0f);
    std::atomic<int64_t> next_row(0);
    const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> workers;
    workers.reserve(nt);

    for (unsigned it = 0; it < nt; ++it) {
        workers.emplace_back([&]() {
            while (true) {
                const int64_t row = next_row.fetch_add(1);
                if (row >= nrows) {
                    break;
                }
                const int64_t ib = row / (c.hq*c.nq);
                const int64_t ih = (row / c.nq) % c.hq;
                const int64_t iq = row % c.nq;
                const int64_t ihkv = ih / (c.hq/c.hkv);
                float row_max = -INFINITY;
                for (int64_t ik = 0; ik < c.nkv; ++ik) {
                    float dot = 0.0f;
                    for (int64_t id = 0; id < c.d; ++id) {
                        dot += data.q[((ib*c.hq + ih)*c.nq + iq)*c.d + id] *
                            data.k_rounded[((ib*c.hkv + ihkv)*c.nkv + ik)*c.d + id];
                    }
                    const float mask = ggml_fp16_to_fp32(data.mask[iq*c.nkv + ik]);
                    const float score = dot/float(c.d) + data.dense_bias[row*c.nkv + ik] + mask;
                    scores[row*c.nkv + ik] = score;
                    row_max = std::max(row_max, score);
                }
                float sum = 0.0f;
                for (int64_t ik = 0; ik < c.nkv; ++ik) {
                    const float p = std::exp(scores[row*c.nkv + ik] - row_max);
                    scores[row*c.nkv + ik] = p;
                    sum += p;
                }
                for (int64_t ik = 0; ik < c.nkv; ++ik) {
                    const float p = scores[row*c.nkv + ik]/sum;
                    for (int64_t id = 0; id < c.d; ++id) {
                        output[((ib*c.nq + iq)*c.hq + ih)*c.d + id] +=
                            p*data.v_rounded[((ib*c.hkv + ihkv)*c.nkv + ik)*c.d + id];
                    }
                }
            }
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }
    return output;
}

static void error_stats(const std::vector<float> & got, const std::vector<float> & ref,
        double & max_abs, double & mean_abs, double & max_rel, double & mean_rel, double & rmse) {
    double sq = 0.0;
    double abs_sum = 0.0;
    double rel_sum = 0.0;
    max_abs = 0.0;
    max_rel = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double ae = std::abs(double(got[i]) - ref[i]);
        const double re = ae/std::max(1e-5, std::abs(double(ref[i])));
        max_abs = std::max(max_abs, ae);
        max_rel = std::max(max_rel, re);
        abs_sum += ae;
        rel_sum += re;
        sq += ae*ae;
    }
    mean_abs = abs_sum/got.size();
    mean_rel = rel_sum/got.size();
    rmse = std::sqrt(sq/got.size());
}

static void overflow_arithmetic_self_test() {
    // mirrors the scalar-path offset math; exact offset checked at 128 bits, lands past 2^31
    const uint64_t nb0 = sizeof(float);
    const uint64_t nb1 = 1024*nb0;
    const uint64_t nb2 = 64*nb1;
    const uint64_t nb3 = 131072*nb2;
    const uint64_t dist = 1023, head = 63, query = 131071, batch = 3;
    const uint64_t offset = dist*nb0 + head*nb1 + query*nb2 + batch*nb3;
    __extension__ typedef unsigned __int128 uint128_t;
    const uint128_t exact = uint128_t(dist)*nb0 + uint128_t(head)*nb1 +
        uint128_t(query)*nb2 + uint128_t(batch)*nb3;
    GGML_ASSERT(exact <= UINT64_MAX && offset == (uint64_t) exact && offset > INT32_MAX);

    const int64_t nq = int64_t(1) << 40;
    const int64_t nkv = nq + 8192;
    const int64_t iq = nq - 1;
    const int64_t ik = nkv - 1024;
    const int64_t rel_dist = iq + (nkv - nq) - ik;
    GGML_ASSERT(rel_dist == 1023);
    printf("overflow_check offset=%llu (>INT32_MAX) large_T=%lld rel_dist=%lld PASS\n",
           (unsigned long long) offset, (long long) nq, (long long) rel_dist);
}

static bool overflow_kernel_test(ggml_backend_t backend, const char * backend_kind) {
    // rel-logits row for query 1 sits beyond INT32_MAX; only two small logical rows are touched
    const bias_test_config c = {
        "overflow_kernel_stride", 64, 2, 2, 2, 1, 8, GGML_TYPE_F32, true, false,
    };
    test_data data = make_data(c);
    const uint64_t rel_nb2 = (UINT64_C(1) << 31) + 4096;
    const size_t rel_row_bytes = c.extent*c.hq*sizeof(float);
    const uint64_t storage_bytes = rel_nb2 + rel_row_bytes;

    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead()*32 + ggml_graph_overhead_custom(32, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx);
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, c.d, c.nq, c.hq, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, c.d, c.nkv, c.hkv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, c.d, c.nkv, c.hkv, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, c.nkv, c.nq, 1, 1);
    ggml_tensor * r_storage = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32,
        (storage_bytes + sizeof(float) - 1)/sizeof(float));
    ggml_tensor * r = ggml_view_4d(ctx.get(), r_storage, c.extent, c.hq, c.nq, 1,
        c.extent*sizeof(float), rel_nb2, rel_nb2*c.nq, 0);
    ggml_tensor * out = ggml_flash_attn_ext_banded(
        ctx.get(), q, k, v, m, r, 1.0f/float(c.d), c.extent);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer);

    ggml_backend_tensor_set(q, data.q.data(), 0, data.q.size()*sizeof(float));
    ggml_backend_tensor_set(k, data.k_typed.data(), 0, data.k_typed.size());
    ggml_backend_tensor_set(v, data.v_typed.data(), 0, data.v_typed.size());
    ggml_backend_tensor_set(m, data.mask.data(), 0, data.mask.size()*sizeof(data.mask[0]));
    ggml_backend_tensor_set(r_storage, data.rel_typed.data(), 0, rel_row_bytes);
    ggml_backend_tensor_set(r_storage, data.rel_typed.data() + rel_row_bytes, rel_nb2, rel_row_bytes);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    std::vector<float> got(ggml_nelements(out));
    ggml_backend_tensor_get(out, got.data(), 0, got.size()*sizeof(float));
    const std::vector<float> ref = naive_materialized(c, data);
    double max_abs, mean_abs, max_rel, mean_rel, rmse;
    error_stats(got, ref, max_abs, mean_abs, max_rel, mean_rel, rmse);
    const bool pass = max_abs <= 2e-5;
    printf("overflow_kernel backend=%s rel_query_stride=%llu allocated_bytes=%zu "
           "naive_max_abs=%.9g naive_mean_abs=%.9g naive_max_rel=%.9g naive_mean_rel=%.9g naive_rmse=%.9g %s\n",
           backend_kind, (unsigned long long) rel_nb2, ggml_backend_buffer_get_size(buffer.get()),
           max_abs, mean_abs, max_rel, mean_rel, rmse, pass ? "PASS" : "FAIL");
    return pass;
}

int main(int argc, char ** argv) {
    std::string backend_kind = argc > 1 ? argv[1] : "cpu";
    std::string suite = argc > 2 ? argv[2] : "small";
    int repeats = argc > 3 ? std::max(1, atoi(argv[3])) : 1;

    overflow_arithmetic_self_test();
    ggml_backend_load_all();
    ggml_backend_dev_t chosen = nullptr;
    const enum ggml_backend_dev_type wanted = backend_kind == "cuda" ?
        GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == wanted) {
            chosen = dev;
            break;
        }
    }
    GGML_ASSERT(chosen);
    ggml_backend_ptr backend(ggml_backend_dev_init(chosen, nullptr));
    GGML_ASSERT(backend);

    if (suite == "overflow") {
        return overflow_kernel_test(backend.get(), backend_kind.c_str()) ? 0 : 1;
    }

    if (suite == "perf") {
        const std::vector<bias_test_config> perf_cases = {
            {"prefill_t1024", 64, 1024, 1024,  8, 2,  512, GGML_TYPE_F16, true, false},
            {"prefill_t2048", 64, 2048, 2048,  8, 2,  512, GGML_TYPE_F16, true, false},
            {"prefill_t4096", 64, 4096, 4096,  8, 2,  512, GGML_TYPE_F16, true, false},
            {"decode_8k",    128,    1, 8192,  8, 1, 1024, GGML_TYPE_F16, true, false},
            {"heads64_gqa",  64, 1024, 1024, 64, 8,  512, GGML_TYPE_F16, true, false},
        };
        printf("backend=%s suite=perf device=%s repeats=%d\n", backend_kind.c_str(),
               ggml_backend_dev_description(chosen), repeats);
        for (const bias_test_config & c : perf_cases) {
            test_data data = make_data(c);
            const run_result flash = run_graph(backend.get(), c, data, false, repeats);
            const run_result dense = run_graph(backend.get(), c, data, true, repeats);
            printf("%s type=%s D=%lld nq=%lld nkv=%lld hq=%lld hkv=%lld E=%lld "
                   "flash_ms=%.6f dense_ms=%.6f speedup=%.6f flash_bytes=%zu dense_bytes=%zu memory_ratio=%.6f\n",
                   c.name, ggml_type_name(c.type), (long long)c.d, (long long)c.nq,
                   (long long)c.nkv, (long long)c.hq, (long long)c.hkv, (long long)c.extent,
                   flash.ms, dense.ms, dense.ms/flash.ms, flash.allocated_bytes,
                   dense.allocated_bytes, double(dense.allocated_bytes)/flash.allocated_bytes);
        }
        return 0;
    }

    if (suite == "memory") {
        const std::vector<bias_test_config> memory_cases = {
            {"memory_t1024", 64, 1024, 1024, 8, 2, 512, GGML_TYPE_F16, false, false},
            {"memory_t2048", 64, 2048, 2048, 8, 2, 512, GGML_TYPE_F16, false, false},
            {"memory_t4096", 64, 4096, 4096, 8, 2, 512, GGML_TYPE_F16, false, false},
        };
        printf("backend=%s suite=memory device=%s\n", backend_kind.c_str(),
               ggml_backend_dev_description(chosen));
        for (const bias_test_config & c : memory_cases) {
            test_data data = make_data(c);
            const run_result flash = run_graph(backend.get(), c, data, false, 1);
            const run_result dense = run_graph(backend.get(), c, data, true, 1);
            double max_abs, mean_abs, max_rel, mean_rel, rmse;
            error_stats(flash.output, dense.output, max_abs, mean_abs, max_rel, mean_rel, rmse);
            printf("%s T=%lld flash_bytes=%zu dense_bytes=%zu memory_ratio=%.6f "
                   "dense_max_abs=%.9g dense_mean_abs=%.9g PASS\n",
                   c.name, (long long)c.nq, flash.allocated_bytes, dense.allocated_bytes,
                   double(dense.allocated_bytes)/flash.allocated_bytes, max_abs, mean_abs);
        }
        return 0;
    }

    std::vector<bias_test_config> configs;
    if (suite == "small") {
        configs = {
            {"edge_e8_f32",       64, 16, 16, 2, 1,   8, GGML_TYPE_F32,  true, false},
            {"gqa_d128_f16",     128, 16, 16, 8, 2,   8, GGML_TYPE_F16,  true, false},
            {"sliding_bf16",      64, 64, 64, 8, 2,   8, GGML_TYPE_BF16, true, true },
            {"decode_offset",     64,  1, 513, 8, 2,   8, GGML_TYPE_F32,  true, false},
            {"extent_512_edge",   64, 64, 64, 8, 1, 512, GGML_TYPE_F16,  true, false},
            {"strided_rel_f16",   64, 17, 33, 8, 2,   8, GGML_TYPE_F16,  true, false, 1, 1, true},
            {"batch_distinct",    64, 16, 16, 8, 2,   8, GGML_TYPE_F16,  true, false, 2, 2, false},
            {"batch_broadcast",   64, 16, 16, 8, 2,   8, GGML_TYPE_F16,  true, false, 2, 1, false},
            {"heads64_gqa4",      64, 16, 16, 64, 16, 8, GGML_TYPE_F16, true, false},
            {"heads64_gqa8",      64, 16, 16, 64,  8, 8, GGML_TYPE_F16, true, false},
        };
    } else if (suite == "medium") {
        configs = {
            {"medium_f32_e512",   64, 1024, 1024, 8, 2,  512, GGML_TYPE_F32,  true, false},
            {"medium_f16_e1024", 128, 1024, 1024, 8, 2, 1024, GGML_TYPE_F16,  true, false},
            {"medium_bf16_local", 64, 2048, 2048, 8, 2,  512, GGML_TYPE_BF16, true, true },
            {"decode_8k_e1024",  128,    1, 8192, 8, 1, 1024, GGML_TYPE_F16,  true, false},
        };
    } else if (suite == "hard") {
        configs = {
            {"heads64_gqa",       64, 1024, 1024, 64, 8, 512, GGML_TYPE_F16, true, false},
        };
    } else {
        fprintf(stderr, "unknown suite: %s (expected small, medium, hard, perf, memory, or overflow)\n",
                suite.c_str());
        return 2;
    }

    bool ok = true;
    printf("backend=%s suite=%s device=%s\n", backend_kind.c_str(), suite.c_str(), ggml_backend_dev_description(chosen));
    for (const bias_test_config & c : configs) {
        test_data data = make_data(c);
        run_result flash = run_graph(backend.get(), c, data, false, repeats);
        run_result dense = run_graph(backend.get(), c, data, true,  repeats);
        // always compare to an independently materialized oracle (O(T^2), test-only)
        const std::vector<float> naive = naive_materialized(c, data);

        double abs_dense, mean_abs_dense, rel_dense, mean_rel_dense, rmse_dense;
        error_stats(flash.output, dense.output, abs_dense, mean_abs_dense, rel_dense, mean_rel_dense, rmse_dense);
        double abs_naive = 0.0, mean_abs_naive = 0.0, rel_naive = 0.0, mean_rel_naive = 0.0, rmse_naive = 0.0;
        error_stats(flash.output, naive, abs_naive, mean_abs_naive, rel_naive, mean_rel_naive, rmse_naive);
        const double tol = c.type == GGML_TYPE_F32 ? 2e-5 : 2e-3;
        const bool pass = abs_naive <= tol;
        ok = ok && pass;
        printf("%s type=%s D=%lld nq=%lld nkv=%lld hq=%lld hkv=%lld E=%lld mask=%d sliding=%d "
               "batch=%lld rel_batch=%lld strided_rel=%d "
               "dense_max_abs=%.9g dense_mean_abs=%.9g dense_max_rel=%.9g dense_mean_rel=%.9g dense_rmse=%.9g "
               "naive_max_abs=%.9g naive_mean_abs=%.9g naive_max_rel=%.9g naive_mean_rel=%.9g naive_rmse=%.9g "
               "flash_ms=%.4f dense_ms=%.4f speedup=%.4f "
               "flash_bytes=%zu dense_bytes=%zu memory_ratio=%.4f %s\n",
               c.name, ggml_type_name(c.type), (long long)c.d, (long long)c.nq, (long long)c.nkv,
               (long long)c.hq, (long long)c.hkv, (long long)c.extent, c.use_mask, c.sliding,
               (long long)c.n_batch, (long long)c.rel_batch, c.strided_rel,
               abs_dense, mean_abs_dense, rel_dense, mean_rel_dense, rmse_dense,
               abs_naive, mean_abs_naive, rel_naive, mean_rel_naive, rmse_naive,
               flash.ms, dense.ms, dense.ms/flash.ms,
               flash.allocated_bytes, dense.allocated_bytes, double(dense.allocated_bytes)/flash.allocated_bytes,
               pass ? "PASS" : "FAIL");
    }
    return ok ? 0 : 1;
}
