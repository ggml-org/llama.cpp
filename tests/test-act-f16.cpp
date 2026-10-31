// CPU backend checks for F16 activations: the typed-result ops (ggml_*_cast) and F16 sources.
//
// test-backend-ops uses the CPU backend as the reference for every other backend, so the CPU
// kernels themselves are checked here against the existing F32 paths:
//   - get_rows / rms_norm / scale / set_rows / flash_attn_ext: bit-exact against the F32 result rounded to F16
//     (or against an F32 source holding the same F16 values)
//   - mul_mat / mul_mat_id: the F32 path may take a different kernel (llamafile), so these are
//     compared against a double-precision reference instead
// plus supports_op, which must reject the F16 forms that have no CPU kernel.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { n_fail++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static std::mt19937 rng(1234);

static std::vector<float> rand_f32(size_t n, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto & x : v) {
        x = d(rng);
    }
    return v;
}

// values exactly representable in F16, so F32 and F16 inputs carry the same numbers
static std::vector<float> rand_f16_values(size_t n, float lo = -1.0f, float hi = 1.0f) {
    std::vector<float> v = rand_f32(n, lo, hi);
    for (auto & x : v) {
        x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
    }
    return v;
}

// set tensor t from F32 values, converting to t->type
static void set_from_f32(ggml_tensor * t, const std::vector<float> & v) {
    GGML_ASSERT((int64_t) v.size() == ggml_nelements(t));
    std::vector<uint8_t> buf(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F32) {
        memcpy(buf.data(), v.data(), buf.size());
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(v.data(), (ggml_fp16_t *) buf.data(), v.size());
    } else {
        ggml_quantize_chunk(t->type, v.data(), buf.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
    }
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
}

static std::vector<uint8_t> get_bytes(const ggml_tensor * t) {
    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
    return buf;
}

static std::vector<float> get_f32(const ggml_tensor * t) {
    const std::vector<uint8_t> b = get_bytes(t);
    std::vector<float> v(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        memcpy(v.data(), b.data(), b.size());
    } else {
        GGML_ASSERT(t->type == GGML_TYPE_F16);
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) b.data(), v.data(), v.size());
    }
    return v;
}

// the F32 result rounded to F16 the same way the kernels do (round to nearest even)
static std::vector<uint8_t> f32_to_f16_bytes(const std::vector<float> & v) {
    std::vector<uint8_t> b(v.size()*sizeof(ggml_fp16_t));
    ggml_fp32_to_fp16_row(v.data(), (ggml_fp16_t *) b.data(), v.size());
    return b;
}

static double nmse(const std::vector<float> & a, const std::vector<double> & ref) {
    double err = 0.0, norm = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        err  += (a[i] - ref[i])*(a[i] - ref[i]);
        norm += ref[i]*ref[i];
    }
    return err/norm;
}

// one graph on the CPU backend: build() creates the tensors and returns the result,
// init() fills the inputs once they are allocated
struct cpu_graph {
    ggml_backend_t backend;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * out = nullptr;

    cpu_graph(ggml_backend_t backend, const std::function<ggml_tensor * (ggml_context *)> & build) : backend(backend) {
        ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        out = build(ctx);
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }

    ~cpu_graph() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    ggml_tensor * get(const char * name) { return ggml_get_tensor(ctx, name); }

    void compute() {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    }
};

static ggml_tensor * named(ggml_tensor * t, const char * name) {
    ggml_set_name(t, name);
    return t;
}

static void test_get_rows(ggml_backend_t be) {
    printf("get_rows_cast\n");
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0}) {
        const int n = 256, m = 10, r = 7;
        const std::vector<float> data = rand_f32(n*m);
        std::vector<int32_t> rows(r);
        for (auto & i : rows) {
            i = rng() % m;
        }

        auto run = [&](ggml_type type_dst) {
            cpu_graph g(be, [&](ggml_context * ctx) {
                ggml_tensor * a = named(ggml_new_tensor_2d(ctx, type, n, m), "a");
                ggml_tensor * b = named(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, r), "b");
                return type_dst == GGML_TYPE_F32 ? ggml_get_rows(ctx, a, b) : ggml_get_rows_cast(ctx, a, b, type_dst);
            });
            set_from_f32(g.get("a"), data);
            ggml_backend_tensor_set(g.get("b"), rows.data(), 0, r*sizeof(int32_t));
            g.compute();
            return std::make_pair(get_f32(g.out), get_bytes(g.out));
        };

        const auto ref = run(GGML_TYPE_F32);
        const auto got = run(GGML_TYPE_F16);
        CHECK(got.second == f32_to_f16_bytes(ref.first), "get_rows_cast src=%s dst=f16 differs from rounded F32 result", ggml_type_name(type));
    }
}

static void test_rms_norm(ggml_backend_t be) {
    printf("rms_norm (F16)\n");
    for (int64_t n : {64, 2048, 2051}) {
        const int64_t rows = 5;
        const std::vector<float> x = rand_f16_values(n*rows, -4.0f, 4.0f);

        auto run = [&](ggml_type type) {
            cpu_graph g(be, [&](ggml_context * ctx) {
                ggml_tensor * a = named(ggml_new_tensor_2d(ctx, type, n, rows), "a");
                return ggml_rms_norm(ctx, a, 1e-5f);
            });
            set_from_f32(g.get("a"), x);
            g.compute();
            return std::make_pair(get_f32(g.out), get_bytes(g.out));
        };

        const auto ref = run(GGML_TYPE_F32);
        const auto got = run(GGML_TYPE_F16);
        CHECK(got.second == f32_to_f16_bytes(ref.first), "rms_norm f16 n=%lld differs from rounded F32 result", (long long) n);
    }
}

static void test_scale(ggml_backend_t be) {
    printf("scale (F16)\n");
    for (float bias : {0.0f, 0.25f}) {
        const int64_t n = 2048, rows = 3;
        const std::vector<float> x = rand_f16_values(n*rows, -8.0f, 8.0f);

        auto run = [&](ggml_type type) {
            cpu_graph g(be, [&](ggml_context * ctx) {
                ggml_tensor * a = named(ggml_new_tensor_2d(ctx, type, n, rows), "a");
                return ggml_scale_bias(ctx, a, 0.0441941738f, bias);
            });
            set_from_f32(g.get("a"), x);
            g.compute();
            return std::make_pair(get_f32(g.out), get_bytes(g.out));
        };

        const auto ref = run(GGML_TYPE_F32);
        const auto got = run(GGML_TYPE_F16);
        CHECK(got.second == f32_to_f16_bytes(ref.first), "scale f16 bias=%g differs from rounded F32 result", bias);
    }
}

static void test_set_rows(ggml_backend_t be) {
    printf("set_rows (F16 source)\n");
    for (ggml_type type_dst : {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0}) {
        for (ggml_type type_idx : {GGML_TYPE_I64, GGML_TYPE_I32}) {
            const int64_t nc = 64, nr = 5, r = 3, ne2 = 2;
            const std::vector<float> src = rand_f16_values(nc*r*ne2);
            const std::vector<float> init = rand_f32(nc*nr*ne2);
            std::vector<int64_t> idx64 = {4, 0, 2, 1, 3, 0};
            std::vector<int32_t> idx32(idx64.begin(), idx64.end());

            auto run = [&](ggml_type type_src) {
                cpu_graph g(be, [&](ggml_context * ctx) {
                    ggml_tensor * dst = named(ggml_new_tensor_3d(ctx, type_dst, nc, nr, ne2), "dst");
                    ggml_tensor * s   = named(ggml_new_tensor_3d(ctx, type_src, nc, r, ne2), "src");
                    ggml_tensor * ids = named(ggml_new_tensor_2d(ctx, type_idx, r, ne2), "ids");
                    return ggml_set_rows(ctx, dst, s, ids);
                });
                set_from_f32(g.get("dst"), init);
                set_from_f32(g.get("src"), src);
                if (type_idx == GGML_TYPE_I64) {
                    ggml_backend_tensor_set(g.get("ids"), idx64.data(), 0, idx64.size()*sizeof(int64_t));
                } else {
                    ggml_backend_tensor_set(g.get("ids"), idx32.data(), 0, idx32.size()*sizeof(int32_t));
                }
                g.compute();
                return get_bytes(g.get("dst"));
            };

            CHECK(run(GGML_TYPE_F16) == run(GGML_TYPE_F32), "set_rows src=f16 dst=%s idx=%s differs from an F32 source with the same values",
                    ggml_type_name(type_dst), ggml_type_name(type_idx));
        }
    }
}

static void test_flash_attn(ggml_backend_t be) {
    printf("flash_attn_ext_cast\n");
    struct shape { int64_t hs, nh, nr, kv, nb; bool exact; };
    // exact: the F32 reference also takes the per-chunk path (nb < GGML_FA_TILE_Q and kv < 512)
    const shape shapes[] = {
        {64, 4, 2,  96,  1, true }, {64, 4, 2,  96,  7, true }, {64, 4, 2, 96, 40, true },
        {64, 4, 2, 600,  1, false}, {64, 4, 2, 128, 80, false},
    };
    for (ggml_type type_kv : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        for (const shape & s : shapes) {
            const std::vector<float> q = rand_f16_values(s.hs*s.nb*s.nh*s.nr);
            const std::vector<float> k = rand_f32(s.hs*s.kv*s.nh);
            const std::vector<float> v = rand_f32(s.hs*s.kv*s.nh);
            std::vector<float> mask(s.kv*s.nb);
            for (int64_t i1 = 0; i1 < s.nb; i1++) {
                for (int64_t i0 = 0; i0 < s.kv; i0++) {
                    mask[i1*s.kv + i0] = i0 > s.kv - s.nb + i1 ? -INFINITY : 0.0f;
                }
            }

            auto run = [&](ggml_type type_q, ggml_type type_dst) {
                cpu_graph g(be, [&](ggml_context * ctx) {
                    ggml_tensor * tq = named(ggml_new_tensor_4d(ctx, type_q,  s.hs, s.nb, s.nh*s.nr, 1), "q");
                    ggml_tensor * tk = named(ggml_new_tensor_4d(ctx, type_kv, s.hs, s.kv, s.nh, 1), "k");
                    ggml_tensor * tv = named(ggml_new_tensor_4d(ctx, type_kv, s.hs, s.kv, s.nh, 1), "v");
                    ggml_tensor * tm = named(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, s.kv, s.nb, 1, 1), "m");
                    ggml_tensor * o  = ggml_flash_attn_ext_cast(ctx, tq, tk, tv, tm, 1.0f/sqrtf(s.hs), 0.0f, 0.0f, type_dst);
                    ggml_prec_set_acc(o, GGML_PREC_F32);
                    return o;
                });
                set_from_f32(g.get("q"), q);
                set_from_f32(g.get("k"), k);
                set_from_f32(g.get("v"), v);
                set_from_f32(g.get("m"), mask);
                g.compute();
                return std::make_pair(get_f32(g.out), get_bytes(g.out));
            };

            const auto ref = run(GGML_TYPE_F32, GGML_TYPE_F32);
            const std::vector<uint8_t> ref16 = f32_to_f16_bytes(ref.first);
            std::vector<double> refd(ref.first.begin(), ref.first.end());

            const auto q16_d32 = run(GGML_TYPE_F16, GGML_TYPE_F32);
            const auto q32_d16 = run(GGML_TYPE_F32, GGML_TYPE_F16);
            const auto q16_d16 = run(GGML_TYPE_F16, GGML_TYPE_F16);

            if (s.exact) {
                CHECK(q16_d32.second == ref.second, "FA kv=%s nb=%lld kv_len=%lld: q=f16 dst=f32 not bit-exact", ggml_type_name(type_kv), (long long) s.nb, (long long) s.kv);
                CHECK(q32_d16.second == ref16,      "FA kv=%s nb=%lld kv_len=%lld: q=f32 dst=f16 not bit-exact", ggml_type_name(type_kv), (long long) s.nb, (long long) s.kv);
                CHECK(q16_d16.second == ref16,      "FA kv=%s nb=%lld kv_len=%lld: q=f16 dst=f16 not bit-exact", ggml_type_name(type_kv), (long long) s.nb, (long long) s.kv);
            } else {
                // the F32 reference takes the tiled / split-KV path, the F16 forms the per-chunk path,
                // which accumulates in F16 for an F16 V; same bound as test-backend-ops
                for (const auto * r : {&q16_d32, &q32_d16, &q16_d16}) {
                    const double e = nmse(r->first, refd);
                    CHECK(e < 5e-4, "FA kv=%s nb=%lld kv_len=%lld: nmse %.3g", ggml_type_name(type_kv), (long long) s.nb, (long long) s.kv, e);
                }
            }
        }
    }
}

static void test_mul_mat(ggml_backend_t be) {
    printf("mul_mat_cast / mul_mat_id_cast\n");
    struct shape { int64_t m, n, k; };
    for (ggml_type type_a : {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0}) {
        for (const shape & s : {shape{16, 1, 256}, shape{64, 9, 256}, shape{128, 33, 64}}) {
            const std::vector<float> a = rand_f16_values(s.k*s.m);
            const std::vector<float> b = rand_f16_values(s.k*s.n);

            // reference from the weights as stored (quantization is part of the input)
            std::vector<float> a_eff = a;
            if (type_a == GGML_TYPE_Q8_0) {
                std::vector<uint8_t> qa(ggml_row_size(type_a, s.k)*s.m);
                ggml_quantize_chunk(type_a, a.data(), qa.data(), 0, s.m, s.k, nullptr);
                ggml_get_type_traits(type_a)->to_float(qa.data(), a_eff.data(), s.k*s.m);
            }
            std::vector<double> ref(s.m*s.n);
            for (int64_t j = 0; j < s.n; j++) {
                for (int64_t i = 0; i < s.m; i++) {
                    double acc = 0.0;
                    for (int64_t l = 0; l < s.k; l++) {
                        acc += (double) a_eff[i*s.k + l] * b[j*s.k + l];
                    }
                    ref[j*s.m + i] = acc;
                }
            }

            auto run = [&](ggml_type type_b, ggml_type type_dst) {
                cpu_graph g(be, [&](ggml_context * ctx) {
                    ggml_tensor * ta = named(ggml_new_tensor_2d(ctx, type_a, s.k, s.m), "a");
                    ggml_tensor * tb = named(ggml_new_tensor_2d(ctx, type_b, s.k, s.n), "b");
                    return ggml_mul_mat_cast(ctx, ta, tb, type_dst);
                });
                if (!ggml_backend_supports_op(be, g.out)) {
                    return std::make_pair(std::vector<float>(), std::vector<uint8_t>());
                }
                set_from_f32(g.get("a"), a);
                set_from_f32(g.get("b"), b);
                g.compute();
                return std::make_pair(get_f32(g.out), get_bytes(g.out));
            };

            // an F16 activation is a valid src1 only where it is the weights' vec_dot type
            const bool b16_ok = type_a == GGML_TYPE_F16;

            const auto d32     = run(GGML_TYPE_F32, GGML_TYPE_F32);
            const auto d16     = run(GGML_TYPE_F32, GGML_TYPE_F16);
            const auto b16_d16 = run(GGML_TYPE_F16, GGML_TYPE_F16);

            // F32 accumulation over k products of [-1, 1] values; F16 output adds ~2^-22 relative error
            const double tol = type_a == GGML_TYPE_Q8_0 ? 1e-4 : 1e-5;
            CHECK(nmse(d32.first, ref) < tol, "mul_mat a=%s m=%lld n=%lld k=%lld dst=f32: nmse %.3g", ggml_type_name(type_a), (long long) s.m, (long long) s.n, (long long) s.k, nmse(d32.first, ref));
            CHECK(nmse(d16.first, ref) < tol, "mul_mat a=%s m=%lld n=%lld k=%lld dst=f16: nmse %.3g", ggml_type_name(type_a), (long long) s.m, (long long) s.n, (long long) s.k, nmse(d16.first, ref));
            CHECK(b16_ok == !b16_d16.first.empty(), "mul_mat a=%s b=f16: supports_op=%d, expected %d", ggml_type_name(type_a), (int) !b16_d16.first.empty(), (int) b16_ok);
            if (b16_ok) {
                // same vec_dot inputs either way: F32 src1 is converted to F16 first
                CHECK(b16_d16.second == d16.second, "mul_mat a=%s m=%lld n=%lld k=%lld: b=f16 and b=f32 give different F16 results",
                        ggml_type_name(type_a), (long long) s.m, (long long) s.n, (long long) s.k);
            }
        }
    }

    for (ggml_type type_a : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        const int n_mats = 4, n_used = 2;
        const int64_t m = 32, n = 8, k = 64;
        const std::vector<float> a = rand_f16_values(k*m*n_mats);
        const std::vector<float> b = rand_f16_values(k*n_used*n);
        std::vector<int32_t> ids(n_mats*n);
        for (int64_t t = 0; t < n; t++) {
            for (int e = 0; e < n_mats; e++) {
                ids[t*n_mats + e] = (t + e) % n_mats;
            }
        }

        auto run = [&](ggml_type type_dst) {
            cpu_graph g(be, [&](ggml_context * ctx) {
                ggml_tensor * ta  = named(ggml_new_tensor_3d(ctx, type_a, k, m, n_mats), "a");
                ggml_tensor * tb  = named(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, n), "b");
                ggml_tensor * tid = named(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_mats, n), "ids");
                ggml_tensor * idv = ggml_view_2d(ctx, tid, n_used, n, tid->nb[1], 0);
                return ggml_mul_mat_id_cast(ctx, ta, tb, idv, type_dst);
            });
            set_from_f32(g.get("a"), a);
            set_from_f32(g.get("b"), b);
            ggml_backend_tensor_set(g.get("ids"), ids.data(), 0, ids.size()*sizeof(int32_t));
            g.compute();
            return get_f32(g.out);
        };

        const std::vector<float> d32 = run(GGML_TYPE_F32);
        const std::vector<float> d16 = run(GGML_TYPE_F16);
        std::vector<double> refd(d32.begin(), d32.end());
        CHECK(nmse(d16, refd) < 1e-6, "mul_mat_id a=%s dst=f16 vs dst=f32: nmse %.3g", ggml_type_name(type_a), nmse(d16, refd));
    }
}

static void test_supports_op(ggml_backend_t be) {
    printf("supports_op\n");
    ggml_init_params ip = { 64*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * x32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
    ggml_tensor * x16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 4);
    ggml_tensor * w16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 8);
    ggml_tensor * w32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 8);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);

    struct expect { const char * what; ggml_tensor * t; bool supported; };
    const expect cases[] = {
        { "rms_norm f32",                     ggml_rms_norm(ctx, x32, 1e-5f),                    true  },
        { "rms_norm f16",                     ggml_rms_norm(ctx, x16, 1e-5f),                    true  },
        { "norm f16",                         ggml_norm(ctx, x16, 1e-5f),                        false },
        { "scale f16",                        ggml_scale(ctx, x16, 2.0f),                        true  },
        { "pad f16",                          ggml_pad(ctx, x16, 8, 0, 0, 0),                    false },
        { "soft_max f16",                     ggml_soft_max(ctx, x16),                           false },
        { "sum_rows f16",                     ggml_sum_rows(ctx, x16),                           true  },
        { "add f16+f16",                      ggml_add(ctx, x16, x16),                           true  },
        { "add f16+f32 -> f16",               ggml_add(ctx, x16, x32),                           true  },
        { "add f32+f16",                      ggml_add(ctx, x32, x16),                           false },
        { "mul f32+f16",                      ggml_mul(ctx, x32, x16),                           false },
        { "mul_mat_cast w16 x32 -> f16",      ggml_mul_mat_cast(ctx, w16, x32, GGML_TYPE_F16),   true  },
        { "mul_mat_cast w16 x16 -> f16",      ggml_mul_mat_cast(ctx, w16, x16, GGML_TYPE_F16),   true  },
        { "mul_mat_cast w32 x16 -> f16",      ggml_mul_mat_cast(ctx, w32, x16, GGML_TYPE_F16),   false },
        { "get_rows_cast f16 -> f16",         ggml_get_rows_cast(ctx, w16, ids, GGML_TYPE_F16),  true  },
        { "set_rows src f16",                 ggml_set_rows(ctx, w16, x16, ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 4)), true },
    };
    for (const expect & e : cases) {
        const bool got = ggml_backend_supports_op(be, e.t);
        CHECK(got == e.supported, "supports_op(%s) = %d, expected %d", e.what, (int) got, (int) e.supported);
    }

    ggml_free(ctx);
}

int main() {
    ggml_backend_t be = ggml_backend_cpu_init();
    GGML_ASSERT(be);

    test_get_rows(be);
    test_rms_norm(be);
    test_scale(be);
    test_set_rows(be);
    test_flash_attn(be);
    test_mul_mat(be);
    test_supports_op(be);

    ggml_backend_free(be);

    printf("%s (%d failure%s)\n", n_fail == 0 ? "PASS" : "FAIL", n_fail, n_fail == 1 ? "" : "s");
    return n_fail == 0 ? 0 : 1;
}
