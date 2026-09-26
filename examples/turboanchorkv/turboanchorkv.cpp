#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama-context.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

extern "C" {
void       quantize_row_turbo2_0_ref(const float * x, void * y, int64_t k);
void       dequantize_row_turbo2_0(const void * x, float * y, int64_t k);
void       quantize_row_turbo3_0_ref(const float * x, void * y, int64_t k);
void       dequantize_row_turbo3_0(const void * x, float * y, int64_t k);
void       quantize_row_turbo4_0_ref(const float * x, void * y, int64_t k);
void       dequantize_row_turbo4_0(const void * x, float * y, int64_t k);
void       turbo_cpu_fwht_inverse(float * x, int group_size);
extern int turbo3_cpu_wht_group_size;
}

namespace {

using clock_type = std::chrono::steady_clock;

struct tensor_series {
    int64_t            d = 0;
    int64_t            h = 0;
    int64_t            n = 0;
    std::vector<float> data;

    const float * row(int64_t token, int64_t head) const { return data.data() + (token * h + head) * d; }

    bool append(const ggml_tensor * t) {
        if (t->ne[3] != 1 || (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_BF16)) {
            return false;
        }
        if (d == 0) {
            d = t->ne[0];
            h = t->ne[1];
        }
        if (d != t->ne[0] || h != t->ne[1]) {
            return false;
        }

        std::vector<uint8_t> copy;
        const uint8_t *      src = nullptr;
        if (ggml_backend_buffer_is_host(t->buffer)) {
            src = static_cast<const uint8_t *>(t->data);
        } else {
            copy.resize(ggml_nbytes(t));
            ggml_backend_tensor_get(t, copy.data(), 0, copy.size());
            src = copy.data();
        }

        const size_t old = data.size();
        data.resize(old + size_t(t->ne[0] * t->ne[1] * t->ne[2]));
        size_t out = old;
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    const uint8_t * p = src + i2 * t->nb[2] + i1 * t->nb[1] + i0 * t->nb[0];
                    if (t->type == GGML_TYPE_F16) {
                        data[out++] = ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(p));
                    } else if (t->type == GGML_TYPE_BF16) {
                        data[out++] = ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(p));
                    } else {
                        data[out++] = *reinterpret_cast<const float *>(p);
                    }
                }
            }
        }
        n += t->ne[2];
        return true;
    }
};

struct capture_state {
    int           layer                = 0;
    int           rope_layer_secondary = 5;
    bool          rope_only            = false;
    std::string   q_name               = "Qcur_pos";
    std::string   k_pre_name           = "Kcur_normed";
    std::string   k_post_name          = "Kcur_pos";
    std::string   v_name               = "Vcur_normed";
    tensor_series q_post;
    tensor_series k_pre;
    tensor_series k_post;
    tensor_series v;
    tensor_series k_pre_secondary;
    tensor_series k_post_secondary;

    std::string full_name(const std::string & base) const { return base + "-" + std::to_string(layer); }

    std::string full_name_secondary(const std::string & base) const {
        return base + "-" + std::to_string(rope_layer_secondary);
    }

    tensor_series * match(const char * name) {
        if (!rope_only && full_name(q_name) == name) {
            return &q_post;
        }
        if (full_name(k_pre_name) == name) {
            return &k_pre;
        }
        if (full_name(k_post_name) == name) {
            return &k_post;
        }
        if (rope_only && rope_layer_secondary != layer && full_name_secondary(k_pre_name) == name) {
            return &k_pre_secondary;
        }
        if (rope_only && rope_layer_secondary != layer && full_name_secondary(k_post_name) == name) {
            return &k_post_secondary;
        }
        if (!rope_only && full_name(v_name) == name) {
            return &v;
        }
        return nullptr;
    }
};

bool capture_callback(ggml_tensor * t, bool ask, void * user_data) {
    auto &          state = *static_cast<capture_state *>(user_data);
    tensor_series * dst   = state.match(t->name);
    if (ask) {
        return dst != nullptr;
    }
    return dst && dst->append(t);
}

float dot(const float * a, const float * b, int64_t d) {
    float sum = 0.0f;
    for (int64_t i = 0; i < d; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float norm_sq(const float * a, int64_t d) {
    return dot(a, a, d);
}

template <class F> void parallel_heads(int64_t n_head, F fn) {
    std::atomic<int64_t>     next{ 0 };
    const unsigned           hw       = std::max(1u, std::thread::hardware_concurrency());
    const int                n_worker = int(std::min<int64_t>(n_head, hw));
    std::vector<std::thread> workers;
    workers.reserve(n_worker);
    for (int i = 0; i < n_worker; ++i) {
        workers.emplace_back([&]() {
            while (true) {
                const int64_t h = next.fetch_add(1);
                if (h >= n_head) {
                    break;
                }
                fn(h);
            }
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }
}

struct observation {
    int64_t            kv_head  = 0;
    int64_t            position = 0;
    std::vector<float> q;
    std::vector<float> alpha;
    std::vector<float> y;
};

std::vector<observation> make_observations(const tensor_series & q,
                                           const tensor_series & k,
                                           const tensor_series & v,
                                           int                   window) {
    const int64_t            group = q.h / k.h;
    const int64_t            first = std::max<int64_t>(0, k.n - window);
    std::vector<observation> result(size_t((k.n - first) * q.h));

    parallel_heads(k.h, [&](int64_t kh) {
        for (int64_t pos = first; pos < k.n; ++pos) {
            for (int64_t g = 0; g < group; ++g) {
                const int64_t qh  = kh * group + g;
                const size_t  oi  = size_t((pos - first) * q.h + qh);
                auto &        obs = result[oi];
                obs.kv_head       = kh;
                obs.position      = pos;
                obs.q.assign(q.row(pos, qh), q.row(pos, qh) + q.d);
                obs.alpha.assign(size_t(k.n), 0.0f);
                obs.y.assign(size_t(v.d), 0.0f);

                float       max_logit = -INFINITY;
                const float scale     = 1.0f / std::sqrt(float(k.d));
                for (int64_t t = 0; t <= pos; ++t) {
                    const float logit    = dot(obs.q.data(), k.row(t, kh), k.d) * scale;
                    obs.alpha[size_t(t)] = logit;
                    max_logit            = std::max(max_logit, logit);
                }
                float denom = 0.0f;
                for (int64_t t = 0; t <= pos; ++t) {
                    const float a        = std::exp(obs.alpha[size_t(t)] - max_logit);
                    obs.alpha[size_t(t)] = a;
                    denom += a;
                }
                for (int64_t t = 0; t <= pos; ++t) {
                    const float a        = obs.alpha[size_t(t)] / denom;
                    obs.alpha[size_t(t)] = a;
                    const float * vt     = v.row(t, kh);
                    for (int64_t j = 0; j < v.d; ++j) {
                        obs.y[size_t(j)] += a * vt[j];
                    }
                }
            }
        }
    });
    return result;
}

std::vector<std::vector<int32_t>> select_anchors(const std::vector<observation> & observations,
                                                 int64_t                          n_head,
                                                 int64_t                          n_token,
                                                 int                              window,
                                                 int                              anchor_budget,
                                                 std::string_view                 mode,
                                                 uint32_t                         seed) {
    std::vector<std::vector<int32_t>> anchors((size_t(n_head)));
    const int64_t                     first_window = std::max<int64_t>(0, n_token - window);

    parallel_heads(n_head, [&](int64_t h) {
        std::vector<float> score(size_t(n_token), 0.0f);
        int                n_obs = 0;
        for (const auto & obs : observations) {
            if (obs.kv_head != h) {
                continue;
            }
            ++n_obs;
            for (int64_t t = 0; t < first_window; ++t) {
                score[size_t(t)] += obs.alpha[size_t(t)];
            }
        }
        if (n_obs > 0) {
            for (float & value : score) {
                value /= n_obs;
            }
        }

        std::vector<float> pooled(score.size(), 0.0f);
        for (int64_t t = 0; t < first_window; ++t) {
            const int64_t lo = std::max<int64_t>(0, t - 3);
            const int64_t hi = std::min<int64_t>(first_window, t + 4);
            for (int64_t u = lo; u < hi; ++u) {
                pooled[size_t(t)] += score[size_t(u)];
            }
            pooled[size_t(t)] /= std::max<int64_t>(1, hi - lo);
        }

        auto & dst = anchors[size_t(h)];
        for (int64_t t = first_window; t < n_token; ++t) {
            dst.push_back(int32_t(t));
        }
        const int remaining = std::max(0, anchor_budget - int(dst.size()));
        if (mode == "uniform") {
            for (int i = 0; i < remaining; ++i) {
                dst.push_back(int32_t((double(i) + 0.5) * first_window / remaining));
            }
            std::sort(dst.begin(), dst.end());
            return;
        }
        const int scored = mode == "attention" ? int(std::floor(0.7 * remaining)) : 0;

        std::vector<int32_t> candidates((size_t(first_window)));
        std::iota(candidates.begin(), candidates.end(), 0);
        std::partial_sort(candidates.begin(), candidates.begin() + std::min<int>(scored, candidates.size()),
                          candidates.end(),
                          [&](int32_t a, int32_t b) { return pooled[size_t(a)] > pooled[size_t(b)]; });
        for (int i = 0; i < scored && i < int(candidates.size()); ++i) {
            dst.push_back(candidates[size_t(i)]);
        }

        std::vector<uint8_t> used(size_t(n_token), 0);
        for (int32_t t : dst) {
            used[size_t(t)] = 1;
        }
        std::vector<int32_t> random_pool;
        for (int64_t t = 0; t < first_window; ++t) {
            if (!used[size_t(t)]) {
                random_pool.push_back(int32_t(t));
            }
        }
        std::mt19937 rng(seed + uint32_t(h));
        std::shuffle(random_pool.begin(), random_pool.end(), rng);
        for (int32_t t : random_pool) {
            if (int(dst.size()) >= anchor_budget) {
                break;
            }
            dst.push_back(t);
        }
        std::sort(dst.begin(), dst.end());
    });
    return anchors;
}

struct projection {
    int64_t                  n_head   = 0;
    int64_t                  n_token  = 0;
    int64_t                  n_anchor = 0;
    int64_t                  d        = 0;
    std::vector<ggml_bf16_t> anchors;
    std::vector<uint16_t>    assignment;
    std::vector<float>       gamma;
    std::vector<float>       residual;
    std::vector<uint8_t>     is_anchor;

    const float * residual_row(int64_t token, int64_t head) const {
        return residual.data() + (head * n_token + token) * d;
    }
};

projection build_projection(const tensor_series & x, const std::vector<std::vector<int32_t>> & anchors) {
    GGML_ASSERT(!anchors.empty() && !anchors.front().empty());
    projection p;
    p.n_head   = x.h;
    p.n_token  = x.n;
    p.d        = x.d;
    p.n_anchor = int64_t(anchors.front().size());
    GGML_ASSERT(p.n_anchor <= UINT16_MAX);
    p.anchors.resize(size_t(p.n_head * p.n_anchor * p.d));
    p.assignment.resize(size_t(x.h * x.n));
    p.gamma.resize(size_t(x.h * x.n));
    p.residual.resize(size_t(x.h * x.n * x.d));
    p.is_anchor.resize(size_t(x.h * x.n), 0);

    parallel_heads(x.h, [&](int64_t h) {
        std::vector<float> anchor_data(size_t(anchors[size_t(h)].size() * x.d));
        std::vector<float> anchor_norm2(anchors[size_t(h)].size());
        GGML_ASSERT(int64_t(anchors[size_t(h)].size()) == p.n_anchor);
        for (size_t a = 0; a < anchors[size_t(h)].size(); ++a) {
            const int32_t pos                  = anchors[size_t(h)][a];
            p.is_anchor[size_t(h * x.n + pos)] = 1;
            float *       stored               = anchor_data.data() + a * size_t(x.d);
            const float * source               = x.row(pos, h);
            for (int64_t j = 0; j < x.d; ++j) {
                const ggml_bf16_t value = ggml_fp32_to_bf16(source[j]);
                stored[j]               = ggml_bf16_to_fp32(value);
                p.anchors[(size_t(h) * size_t(p.n_anchor) + a) * size_t(p.d) + size_t(j)] = value;
            }
            anchor_norm2[a] = norm_sq(stored, x.d);
        }

        for (int64_t t = 0; t < x.n; ++t) {
            const float * xt       = x.row(t, h);
            const float   xn2      = norm_sq(xt, x.d);
            size_t        best     = 0;
            float         best_cos = -1.0f;
            float         best_dot = 0.0f;
            for (size_t a = 0; a < anchors[size_t(h)].size(); ++a) {
                const float dprod  = dot(xt, anchor_data.data() + a * size_t(x.d), x.d);
                const float denom  = std::sqrt(std::max(1e-20f, xn2 * anchor_norm2[a]));
                const float cosine = std::fabs(dprod) / denom;
                if (cosine > best_cos) {
                    best_cos = cosine;
                    best     = a;
                    best_dot = dprod;
                }
            }
            const float  g   = ggml_bf16_to_fp32(ggml_fp32_to_bf16(best_dot / std::max(1e-20f, anchor_norm2[best])));
            const size_t ti  = size_t(h * x.n + t);
            p.assignment[ti] = uint16_t(best);
            p.gamma[ti]      = g;
            float *       r  = p.residual.data() + ti * size_t(x.d);
            const float * xa = anchor_data.data() + best * size_t(x.d);
            for (int64_t j = 0; j < x.d; ++j) {
                r[j] = xt[j] - g * xa[j];
            }
        }
    });
    return p;
}

struct rope_map {
    bool               split     = false;
    int64_t            d         = 0;
    int64_t            n         = 0;
    double             fit_error = 0.0;
    std::vector<float> cosine;
    std::vector<float> sine;

    std::pair<int64_t, int64_t> pair(int64_t p) const {
        return split ? std::make_pair(p, p + d / 2) : std::make_pair(2 * p, 2 * p + 1);
    }

    tensor_series apply(const tensor_series & x) const {
        tensor_series out = x;
        for (int64_t t = 0; t < x.n; ++t) {
            for (int64_t h = 0; h < x.h; ++h) {
                const float * src = x.row(t, h);
                float *       dst = out.data.data() + size_t((t * x.h + h) * x.d);
                for (int64_t p = 0; p < x.d / 2; ++p) {
                    const auto [i0, i1] = pair(p);
                    const float c       = cosine[size_t(t * (x.d / 2) + p)];
                    const float s       = sine[size_t(t * (x.d / 2) + p)];
                    dst[i0]             = c * src[i0] - s * src[i1];
                    dst[i1]             = s * src[i0] + c * src[i1];
                }
            }
        }
        return out;
    }

    tensor_series apply_inverse(const tensor_series & x) const {
        tensor_series out = x;
        for (int64_t t = 0; t < x.n; ++t) {
            for (int64_t h = 0; h < x.h; ++h) {
                const float * src = x.row(t, h);
                float *       dst = out.data.data() + size_t((t * x.h + h) * x.d);
                for (int64_t p = 0; p < x.d / 2; ++p) {
                    const auto [i0, i1] = pair(p);
                    const float c       = cosine[size_t(t * (x.d / 2) + p)];
                    const float s       = sine[size_t(t * (x.d / 2) + p)];
                    dst[i0]             = c * src[i0] + s * src[i1];
                    dst[i1]             = -s * src[i0] + c * src[i1];
                }
            }
        }
        return out;
    }
};

rope_map derive_rope_map(const tensor_series & before, const tensor_series & after, bool split) {
    rope_map map;
    map.split = split;
    map.d     = before.d;
    map.n     = before.n;
    map.cosine.resize(size_t(before.n * before.d / 2));
    map.sine.resize(map.cosine.size());

    double diff2 = 0.0;
    double ref2  = 0.0;
    for (int64_t t = 0; t < before.n; ++t) {
        for (int64_t p = 0; p < before.d / 2; ++p) {
            const auto [i0, i1] = map.pair(p);
            double denom        = 0.0;
            double cnum         = 0.0;
            double snum         = 0.0;
            for (int64_t h = 0; h < before.h; ++h) {
                const float * x = before.row(t, h);
                const float * y = after.row(t, h);
                denom += double(x[i0]) * x[i0] + double(x[i1]) * x[i1];
                cnum += double(x[i0]) * y[i0] + double(x[i1]) * y[i1];
                snum += double(x[i0]) * y[i1] - double(x[i1]) * y[i0];
            }
            double       c      = denom > 1e-30 ? cnum / denom : 1.0;
            double       s      = denom > 1e-30 ? snum / denom : 0.0;
            const double length = std::hypot(c, s);
            if (length > 1e-12) {
                c /= length;
                s /= length;
            }
            map.cosine[size_t(t * (before.d / 2) + p)] = float(c);
            map.sine[size_t(t * (before.d / 2) + p)]   = float(s);
        }
    }

    const tensor_series check = map.apply(before);
    for (size_t i = 0; i < check.data.size(); ++i) {
        const double delta = double(check.data[i]) - after.data[i];
        diff2 += delta * delta;
        ref2 += double(after.data[i]) * after.data[i];
    }
    map.fit_error = std::sqrt(diff2 / std::max(1e-30, ref2));
    return map;
}

projection rotate_projection_residual(const projection & input, const rope_map & rope) {
    tensor_series residual;
    residual.d = input.d;
    residual.h = input.n_head;
    residual.n = input.n_token;
    residual.data.resize(size_t(residual.d * residual.h * residual.n));
    for (int64_t t = 0; t < input.n_token; ++t) {
        for (int64_t h = 0; h < input.n_head; ++h) {
            std::memcpy(residual.data.data() + size_t((t * input.n_head + h) * input.d), input.residual_row(t, h),
                        size_t(input.d) * sizeof(float));
        }
    }
    const tensor_series rotated = rope.apply(residual);

    projection output = input;
    for (int64_t t = 0; t < input.n_token; ++t) {
        for (int64_t h = 0; h < input.n_head; ++h) {
            std::memcpy(output.residual.data() + size_t((h * input.n_token + t) * input.d), rotated.row(t, h),
                        size_t(input.d) * sizeof(float));
        }
    }
    return output;
}

double mean_nearest_cosine(const tensor_series & x, const std::vector<std::vector<int32_t>> & anchors) {
    std::vector<double> sums(size_t(x.h), 0.0);
    parallel_heads(x.h, [&](int64_t h) {
        std::vector<float> an2(anchors[size_t(h)].size());
        for (size_t a = 0; a < an2.size(); ++a) {
            an2[a] = norm_sq(x.row(anchors[size_t(h)][a], h), x.d);
        }
        for (int64_t t = 0; t < x.n; ++t) {
            const float * xt   = x.row(t, h);
            const float   xn2  = norm_sq(xt, x.d);
            float         best = 0.0f;
            for (size_t a = 0; a < an2.size(); ++a) {
                const float c = std::fabs(dot(xt, x.row(anchors[size_t(h)][a], h), x.d)) /
                                std::sqrt(std::max(1e-20f, xn2 * an2[a]));
                best = std::max(best, c);
            }
            sums[size_t(h)] += best;
        }
    });
    return std::accumulate(sums.begin(), sums.end(), 0.0) / double(x.h * x.n);
}

struct utilities {
    std::vector<float> k;
    std::vector<float> v;
};

utilities compute_utilities(const std::vector<observation> & observations,
                            const tensor_series &            v,
                            const projection &               pk,
                            const projection &               pv) {
    utilities result;
    result.k.assign(size_t(pk.n_head * pk.n_token), 0.0f);
    result.v.assign(size_t(pv.n_head * pv.n_token), 0.0f);

    parallel_heads(pk.n_head, [&](int64_t h) {
        int n_obs = 0;
        for (const auto & obs : observations) {
            if (obs.kv_head != h) {
                continue;
            }
            ++n_obs;
            for (int64_t t = 0; t < pk.n_token; ++t) {
                if (pk.is_anchor[size_t(h * pk.n_token + t)]) {
                    continue;
                }
                const float alpha2 = obs.alpha[size_t(t)] * obs.alpha[size_t(t)];
                if (alpha2 == 0.0f) {
                    continue;
                }
                const float   kr  = dot(obs.q.data(), pk.residual_row(t, h), pk.d);
                float         vy2 = 0.0f;
                const float * vt  = v.row(t, h);
                const float * rv  = pv.residual_row(t, h);
                for (int64_t j = 0; j < v.d; ++j) {
                    const float delta = vt[j] - obs.y[size_t(j)];
                    vy2 += delta * delta;
                }
                result.k[size_t(h * pk.n_token + t)] += alpha2 * kr * kr / float(pk.d) * vy2;
                result.v[size_t(h * pk.n_token + t)] += alpha2 * norm_sq(rv, pv.d);
            }
        }
        if (n_obs > 0) {
            for (int64_t t = 0; t < pk.n_token; ++t) {
                result.k[size_t(h * pk.n_token + t)] /= n_obs;
                result.v[size_t(h * pk.n_token + t)] /= n_obs;
            }
        }
    });
    return result;
}

std::vector<size_t> rank_candidates(const std::vector<float> & utility,
                                    const projection &         p,
                                    std::string_view           mode,
                                    uint32_t                   seed) {
    std::vector<size_t> order;
    order.reserve(utility.size());
    for (size_t i = 0; i < utility.size(); ++i) {
        if (!p.is_anchor[i]) {
            order.push_back(i);
        }
    }
    std::vector<float> score = utility;
    if (mode == "random") {
        std::mt19937 rng(seed);
        std::shuffle(order.begin(), order.end(), rng);
    } else if (mode == "norm") {
        for (size_t i : order) {
            const int64_t h = int64_t(i) / p.n_token;
            const int64_t t = int64_t(i) % p.n_token;
            score[i]        = norm_sq(p.residual_row(t, h), p.d);
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return score[a] > score[b]; });
    } else {
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return score[a] > score[b]; });
    }
    return order;
}

// The persistent slot map follows Giveen's AnchorKV CPU reference. The format here uses TurboQuant residual rows.
struct packed_projection {
    int64_t                  n_head        = 0;
    int64_t                  n_token       = 0;
    int64_t                  n_anchor      = 0;
    int64_t                  d             = 0;
    ggml_type                residual_type = GGML_TYPE_TURBO2_0;
    std::vector<ggml_bf16_t> anchors;
    std::vector<uint16_t>    assignment;
    std::vector<ggml_bf16_t> gamma;
    std::vector<uint32_t>    residual_slot;
    std::vector<uint8_t>     residual_data;

    size_t row_size() const { return ggml_row_size(residual_type, d); }

    size_t serialized_size() const {
        return anchors.size() * sizeof(anchors[0]) + assignment.size() * sizeof(assignment[0]) +
               gamma.size() * sizeof(gamma[0]) + residual_slot.size() * sizeof(residual_slot[0]) + residual_data.size();
    }
};

size_t packed_projection_base_size(const projection & p) {
    return size_t(p.n_head * p.n_anchor * p.d) * sizeof(ggml_bf16_t) +
           size_t(p.n_head * p.n_token) * (sizeof(uint16_t) + sizeof(ggml_bf16_t) + sizeof(uint32_t));
}

void encode_residual(ggml_type type, const float * source, int64_t d, uint8_t * destination) {
    if (type == GGML_TYPE_F16) {
        for (int64_t j = 0; j < d; ++j) {
            const ggml_fp16_t value = ggml_fp32_to_fp16(source[j]);
            std::memcpy(destination + size_t(j) * sizeof(value), &value, sizeof(value));
        }
    } else if (type == GGML_TYPE_TURBO2_0) {
        quantize_row_turbo2_0_ref(source, destination, d);
    } else if (type == GGML_TYPE_TURBO3_0) {
        quantize_row_turbo3_0_ref(source, destination, d);
    } else {
        quantize_row_turbo4_0_ref(source, destination, d);
    }
}

void decode_residual(ggml_type type, const uint8_t * source, int64_t d, float * destination) {
    if (type == GGML_TYPE_F16) {
        for (int64_t j = 0; j < d; ++j) {
            ggml_fp16_t value;
            std::memcpy(&value, source + size_t(j) * sizeof(value), sizeof(value));
            destination[j] = ggml_fp16_to_fp32(value);
        }
        return;
    }
    if (type == GGML_TYPE_TURBO2_0) {
        dequantize_row_turbo2_0(source, destination, d);
    } else if (type == GGML_TYPE_TURBO3_0) {
        dequantize_row_turbo3_0(source, destination, d);
    } else {
        dequantize_row_turbo4_0(source, destination, d);
    }
    for (int64_t off = 0; off < d; off += 128) {
        turbo_cpu_fwht_inverse(destination + off, 128);
    }
}

packed_projection pack_projection(const projection &          p,
                                  const std::vector<size_t> & order,
                                  size_t                      n_residual,
                                  ggml_type                   residual_type) {
    packed_projection packed;
    packed.n_head        = p.n_head;
    packed.n_token       = p.n_token;
    packed.n_anchor      = p.n_anchor;
    packed.d             = p.d;
    packed.residual_type = residual_type;
    packed.anchors       = p.anchors;
    packed.assignment    = p.assignment;
    packed.gamma.resize(p.gamma.size());
    for (size_t i = 0; i < p.gamma.size(); ++i) {
        packed.gamma[i] = ggml_fp32_to_bf16(p.gamma[i]);
    }

    const size_t count = std::min(n_residual, order.size());
    packed.residual_slot.assign(size_t(p.n_head * p.n_token), UINT32_MAX);
    packed.residual_data.resize(count * packed.row_size());
    for (size_t slot = 0; slot < count; ++slot) {
        GGML_ASSERT(slot < UINT32_MAX);
        packed.residual_slot[order[slot]] = uint32_t(slot);
    }
    parallel_heads(int64_t(count), [&](int64_t slot) {
        const size_t  ti = order[size_t(slot)];
        const int64_t h  = int64_t(ti) / p.n_token;
        const int64_t t  = int64_t(ti) % p.n_token;
        encode_residual(residual_type, p.residual_row(t, h), p.d,
                        packed.residual_data.data() + size_t(slot) * packed.row_size());
    });
    return packed;
}

tensor_series unpack_projection(const packed_projection & packed) {
    tensor_series out;
    out.d = packed.d;
    out.h = packed.n_head;
    out.n = packed.n_token;
    out.data.resize(size_t(out.d * out.h * out.n));

    parallel_heads(packed.n_head, [&](int64_t h) {
        std::vector<float> residual(size_t(packed.d));
        for (int64_t t = 0; t < packed.n_token; ++t) {
            const size_t   ti = size_t(h * packed.n_token + t);
            const uint16_t ai = packed.assignment[ti];
            GGML_ASSERT(ai < packed.n_anchor);
            const ggml_bf16_t * anchor = packed.anchors.data() + (size_t(h * packed.n_anchor) + ai) * size_t(packed.d);
            const float         gamma  = ggml_bf16_to_fp32(packed.gamma[ti]);
            float *             dst    = out.data.data() + size_t((t * packed.n_head + h) * packed.d);
            for (int64_t j = 0; j < packed.d; ++j) {
                dst[j] = gamma * ggml_bf16_to_fp32(anchor[j]);
            }
            const uint32_t slot = packed.residual_slot[ti];
            if (slot != UINT32_MAX) {
                GGML_ASSERT((size_t(slot) + 1) * packed.row_size() <= packed.residual_data.size());
                decode_residual(packed.residual_type, packed.residual_data.data() + size_t(slot) * packed.row_size(),
                                packed.d, residual.data());
                for (int64_t j = 0; j < packed.d; ++j) {
                    dst[j] += residual[size_t(j)];
                }
            }
        }
    });
    return out;
}

struct cache_tensor_copy {
    tensor_series        series;
    std::vector<uint8_t> raw;
};

bool cache_type_supported(const ggml_tensor * tensor) {
    return tensor && (tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16);
}

float cache_value_read(const ggml_tensor * tensor, const uint8_t * data) {
    if (tensor->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(data));
    }
    return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(data));
}

void cache_value_write(const ggml_tensor * tensor, uint8_t * data, float value) {
    if (tensor->type == GGML_TYPE_F16) {
        *reinterpret_cast<ggml_fp16_t *>(data) = ggml_fp32_to_fp16(value);
    } else {
        *reinterpret_cast<ggml_bf16_t *>(data) = ggml_fp32_to_bf16(value);
    }
}

cache_tensor_copy cache_tensor_read(const ggml_tensor *           tensor,
                                    const std::vector<uint32_t> & cells,
                                    int64_t                       n_head,
                                    int64_t                       d) {
    cache_tensor_copy result;
    result.series.d = d;
    result.series.h = n_head;
    result.series.n = int64_t(cells.size());
    result.series.data.resize(size_t(d * n_head * result.series.n));
    result.raw.resize(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, result.raw.data(), 0, result.raw.size());

    for (size_t t = 0; t < cells.size(); ++t) {
        for (int64_t h = 0; h < n_head; ++h) {
            float *      dst    = result.series.data.data() + (t * size_t(n_head) + size_t(h)) * size_t(d);
            const size_t offset = size_t(cells[t]) * tensor->nb[1] + size_t(h * d) * tensor->nb[0];
            for (int64_t j = 0; j < d; ++j) {
                dst[j] = cache_value_read(tensor, result.raw.data() + offset + size_t(j) * tensor->nb[0]);
            }
        }
    }
    return result;
}

void cache_tensor_write(ggml_tensor *                 tensor,
                        const std::vector<uint32_t> & cells,
                        const tensor_series &         series,
                        std::vector<uint8_t> &        raw) {
    for (size_t t = 0; t < cells.size(); ++t) {
        for (int64_t h = 0; h < series.h; ++h) {
            const float * src    = series.row(int64_t(t), h);
            const size_t  offset = size_t(cells[t]) * tensor->nb[1] + size_t(h * series.d) * tensor->nb[0];
            for (int64_t j = 0; j < series.d; ++j) {
                cache_value_write(tensor, raw.data() + offset + size_t(j) * tensor->nb[0], src[j]);
            }
        }
    }
    ggml_backend_tensor_set(tensor, raw.data(), 0, raw.size());
}

llama_kv_cache * resolve_kv_cache(llama_context * ctx) {
    llama_memory_t memory = ctx->get_memory();
    if (auto * cache = dynamic_cast<llama_kv_cache *>(memory)) {
        return cache;
    }
    if (auto * cache = dynamic_cast<llama_kv_cache_iswa *>(memory)) {
        return cache->get_base();
    }
    return nullptr;
}

struct roundtrip_stats {
    size_t layers          = 0;
    size_t bytes_full      = 0;
    size_t bytes_simulated = 0;
};

bool cache_roundtrip(llama_context *               ctx,
                     int                           ratio,
                     int                           window,
                     bool                          identity,
                     ggml_type                     residual_type,
                     int                           keep_edge_layers,
                     const std::vector<rope_map> * rope_maps,
                     roundtrip_stats &             stats) {
    llama_synchronize(ctx);
    llama_kv_cache * cache = resolve_kv_cache(ctx);
    if (!cache || cache->get_v_transposed()) {
        std::fprintf(stderr, "TurboAnchorKV dense round-trip requires a plain or iSWA non-transposed KV cache\n");
        return false;
    }

    const llama_kv_cells &                      cell_meta = cache->get_cells(0);
    std::vector<std::pair<llama_pos, uint32_t>> positions;
    for (uint32_t i = 0; i < cell_meta.size(); ++i) {
        if (cell_meta.seq_has(i, 0)) {
            positions.push_back({ cell_meta.pos_get(i), i });
        }
    }
    std::sort(positions.begin(), positions.end());
    std::vector<uint32_t> cells;
    cells.reserve(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        const auto & position = positions[i];
        if (position.first != llama_pos(i)) {
            std::fprintf(
                stderr,
                "TurboAnchorKV dense round-trip requires contiguous zero-based positions; got position %lld at "
                "index %zu\n",
                (long long) position.first, i);
            return false;
        }
        cells.push_back(position.second);
    }
    if (cells.empty()) {
        std::fprintf(stderr, "TurboAnchorKV dense round-trip found no sequence-0 cache cells\n");
        return false;
    }

    const llama_hparams &       hparams   = ctx->get_model().hparams;
    const std::vector<uint32_t> layer_ids = cache->get_layer_ids();
    for (size_t layer_index = 0; layer_index < layer_ids.size(); ++layer_index) {
        const uint32_t   il       = layer_ids[layer_index];
        ggml_tensor *    tensor_k = cache->get_k_storage(int32_t(il));
        ggml_tensor *    tensor_v = cache->get_v_storage(int32_t(il));
        const int64_t    n_head   = hparams.n_head_kv(il);
        const int64_t    d_k      = hparams.n_embd_head_k(il);
        const int64_t    d_v      = hparams.n_embd_head_v(il);
        const rope_map * rope     = nullptr;
        if (rope_maps) {
            for (const rope_map & candidate : *rope_maps) {
                if (candidate.d == d_k) {
                    rope = &candidate;
                    break;
                }
            }
        }
        if (!cache_type_supported(tensor_k) || !cache_type_supported(tensor_v) || d_k != d_v || d_k % 128 != 0 ||
            tensor_k->ne[0] != n_head * d_k || tensor_v->ne[0] != n_head * d_v) {
            std::fprintf(stderr, "TurboAnchorKV dense round-trip does not support layer %u cache shape or type\n", il);
            return false;
        }
        if (rope_maps && !rope) {
            std::fprintf(stderr, "TurboAnchorKV has no RoPE map for layer %u head dimension %lld\n", il,
                         (long long) d_k);
            return false;
        }
        if (rope && rope->n != int64_t(cells.size())) {
            std::fprintf(stderr, "TurboAnchorKV RoPE map mismatch at layer %u: map={%lld,%lld} cache={%lld,%zu}\n", il,
                         (long long) rope->d, (long long) rope->n, (long long) d_k, cells.size());
            return false;
        }

        cache_tensor_copy exact_k    = cache_tensor_read(tensor_k, cells, n_head, d_k);
        cache_tensor_copy exact_v    = cache_tensor_read(tensor_v, cells, n_head, d_v);
        const int         n_token    = int(cells.size());
        const size_t      bytes_full = size_t(4LL * n_token * n_head * d_k);
        const bool        edge       = layer_index < size_t(std::max(0, keep_edge_layers)) ||
                          layer_index + size_t(std::max(0, keep_edge_layers)) >= layer_ids.size();
        if (identity || edge) {
            cache_tensor_write(tensor_k, cells, exact_k.series, exact_k.raw);
            cache_tensor_write(tensor_v, cells, exact_v.series, exact_v.raw);
            ++stats.layers;
            stats.bytes_full += bytes_full;
            stats.bytes_simulated += bytes_full;
            continue;
        }
        const tensor_series source_k      = rope ? rope->apply_inverse(exact_k.series) : exact_k.series;
        const int           window_layer  = std::min(window, n_token);
        const int           anchor_budget = std::max(window_layer, n_token / 128);
        const auto          anchors = select_anchors({}, n_head, n_token, window_layer, anchor_budget, "uniform", 42);
        const projection    projection_k = build_projection(source_k, anchors);
        const projection    projection_v = build_projection(exact_v.series, anchors);
        const std::vector<float> empty_k(size_t(n_head * n_token), 0.0f);
        const std::vector<float> empty_v(size_t(n_head * n_token), 0.0f);
        const auto               order_k = rank_candidates(empty_k, projection_k, "norm", 123);
        const auto               order_v = rank_candidates(empty_v, projection_v, "norm", 456);

        const size_t bytes_base = packed_projection_base_size(projection_k) + packed_projection_base_size(projection_v);
        const size_t bytes_k    = ggml_row_size(residual_type, d_k);
        const size_t bytes_v    = ggml_row_size(residual_type, d_v);
        const size_t target     = bytes_full / size_t(ratio);
        const size_t available  = target > bytes_base ? target - bytes_base : 0;
        const size_t n_k        = std::min(size_t(std::floor(0.5 * available / bytes_k)), order_k.size());
        const size_t n_v        = std::min(size_t(std::floor(0.5 * available / bytes_v)), order_v.size());
        const packed_projection packed_k   = pack_projection(projection_k, order_k, n_k, residual_type);
        const packed_projection packed_v   = pack_projection(projection_v, order_v, n_v, residual_type);
        const size_t            bytes_used = packed_k.serialized_size() + packed_v.serialized_size();

        const tensor_series reconstructed_k_native = unpack_projection(packed_k);
        const tensor_series reconstructed_k = rope ? rope->apply(reconstructed_k_native) : reconstructed_k_native;
        const tensor_series reconstructed_v = unpack_projection(packed_v);
        cache_tensor_write(tensor_k, cells, reconstructed_k, exact_k.raw);
        cache_tensor_write(tensor_v, cells, reconstructed_v, exact_v.raw);

        ++stats.layers;
        stats.bytes_full += bytes_full;
        stats.bytes_simulated += bytes_used;
    }
    llama_synchronize(ctx);
    return true;
}

struct error_stats {
    double mean = 0.0;
    double p95  = 0.0;
    double max  = 0.0;
};

error_stats attention_error(const std::vector<observation> & observations,
                            const tensor_series &            k,
                            const tensor_series &            v) {
    std::vector<double>      errors(observations.size());
    std::atomic<size_t>      next{ 0 };
    const unsigned           n_worker = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> workers;
    workers.reserve(n_worker);
    for (unsigned i = 0; i < n_worker; ++i) {
        workers.emplace_back([&]() {
            std::vector<float> alpha(size_t(k.n));
            std::vector<float> y(size_t(v.d));
            while (true) {
                const size_t oi = next.fetch_add(1);
                if (oi >= observations.size()) {
                    break;
                }
                const auto & obs = observations[oi];
                std::fill(y.begin(), y.end(), 0.0f);
                float         max_logit = -INFINITY;
                const float   scale     = 1.0f / std::sqrt(float(k.d));
                const int64_t last      = obs.position;
                for (int64_t t = 0; t <= last; ++t) {
                    alpha[size_t(t)] = dot(obs.q.data(), k.row(t, obs.kv_head), k.d) * scale;
                    max_logit        = std::max(max_logit, alpha[size_t(t)]);
                }
                float denom = 0.0f;
                for (int64_t t = 0; t <= last; ++t) {
                    alpha[size_t(t)] = std::exp(alpha[size_t(t)] - max_logit);
                    denom += alpha[size_t(t)];
                }
                for (int64_t t = 0; t <= last; ++t) {
                    const float   a  = alpha[size_t(t)] / denom;
                    const float * vt = v.row(t, obs.kv_head);
                    for (int64_t j = 0; j < v.d; ++j) {
                        y[size_t(j)] += a * vt[j];
                    }
                }
                double diff2  = 0.0;
                double exact2 = 0.0;
                for (int64_t j = 0; j < v.d; ++j) {
                    const double diff = double(y[size_t(j)]) - obs.y[size_t(j)];
                    diff2 += diff * diff;
                    exact2 += double(obs.y[size_t(j)]) * obs.y[size_t(j)];
                }
                errors[oi] = std::sqrt(diff2 / std::max(1e-30, exact2));
            }
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }

    error_stats result;
    result.mean = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    std::sort(errors.begin(), errors.end());
    result.p95 = errors[std::min(errors.size() - 1, size_t(0.95 * errors.size()))];
    result.max = errors.back();
    return result;
}

int env_int(const char * name, int fallback) {
    const char * value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}

std::string env_string(const char * name, const char * fallback) {
    const char * value = std::getenv(name);
    return value ? value : fallback;
}

ggml_type residual_type_from_env() {
    const std::string value = env_string("TURBOANCHORKV_RESIDUAL_TYPE", "turbo2");
    if (value == "turbo3") {
        return GGML_TYPE_TURBO3_0;
    }
    if (value == "turbo4") {
        return GGML_TYPE_TURBO4_0;
    }
    if (value == "f16") {
        return GGML_TYPE_F16;
    }
    return GGML_TYPE_TURBO2_0;
}

double elapsed_seconds(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

bool decode_range(llama_context *                  ctx,
                  const std::vector<llama_token> & tokens,
                  size_t                           begin,
                  size_t                           end,
                  int32_t                          n_batch) {
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (size_t offset = begin; offset < end; offset += size_t(n_batch)) {
        common_batch_clear(batch);
        const size_t count = std::min<size_t>(n_batch, end - offset);
        for (size_t i = 0; i < count; ++i) {
            common_batch_add(batch, tokens[offset + i], llama_pos(offset + i), { 0 }, false);
        }
        if (offset + count == end) {
            batch.logits[batch.n_tokens - 1] = true;
        }
        if (llama_decode(ctx, batch)) {
            std::fprintf(stderr, "decode failed at token %zu\n", offset);
            llama_batch_free(batch);
            return false;
        }
    }
    llama_batch_free(batch);
    return true;
}

bool decode_one(llama_context * ctx, llama_batch & batch, llama_token token, llama_pos position) {
    common_batch_clear(batch);
    common_batch_add(batch, token, position, { 0 }, true);
    return llama_decode(ctx, batch) == 0;
}

struct suffix_eval {
    double                   nll = 0.0;
    std::vector<float>       logits;
    std::vector<llama_token> targets;
};

double token_nll(const float * logits, int32_t n_vocab, llama_token target) {
    float maximum = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        maximum = std::max(maximum, logits[i]);
    }
    double sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        sum += std::exp(double(logits[i] - maximum));
    }
    return double(maximum) + std::log(sum) - logits[target];
}

suffix_eval evaluate_suffix(llama_context *                  ctx,
                            const llama_vocab *              vocab,
                            const std::vector<llama_token> & tokens,
                            size_t                           first_input) {
    suffix_eval   result;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const size_t  n_eval  = tokens.size() - first_input - 1;
    result.logits.reserve(n_eval * size_t(n_vocab));
    result.targets.reserve(n_eval);
    llama_batch batch = llama_batch_init(1, 0, 1);
    for (size_t i = first_input; i + 1 < tokens.size(); ++i) {
        if (!decode_one(ctx, batch, tokens[i], llama_pos(i))) {
            std::fprintf(stderr, "suffix decode failed at token %zu\n", i);
            result.targets.clear();
            break;
        }
        const float * logits = llama_get_logits_ith(ctx, 0);
        result.nll += token_nll(logits, n_vocab, tokens[i + 1]);
        result.logits.insert(result.logits.end(), logits, logits + n_vocab);
        result.targets.push_back(tokens[i + 1]);
    }
    llama_batch_free(batch);
    return result;
}

struct distribution_drift {
    double kl_mean        = 0.0;
    double top1_agreement = 0.0;
};

distribution_drift compare_logits(const suffix_eval & baseline, const suffix_eval & compressed, int32_t n_vocab) {
    distribution_drift result;
    if (baseline.targets.empty() || baseline.targets.size() != compressed.targets.size()) {
        return result;
    }
    size_t top1_equal = 0;
    for (size_t row = 0; row < baseline.targets.size(); ++row) {
        const float * p_logits = baseline.logits.data() + row * size_t(n_vocab);
        const float * q_logits = compressed.logits.data() + row * size_t(n_vocab);
        const float   p_max    = *std::max_element(p_logits, p_logits + n_vocab);
        const float   q_max    = *std::max_element(q_logits, q_logits + n_vocab);
        double        p_sum    = 0.0;
        double        q_sum    = 0.0;
        int32_t       p_top    = 0;
        int32_t       q_top    = 0;
        for (int32_t i = 0; i < n_vocab; ++i) {
            p_sum += std::exp(double(p_logits[i] - p_max));
            q_sum += std::exp(double(q_logits[i] - q_max));
            if (p_logits[i] > p_logits[p_top]) {
                p_top = i;
            }
            if (q_logits[i] > q_logits[q_top]) {
                q_top = i;
            }
        }
        top1_equal += p_top == q_top;
        const double p_log_z = double(p_max) + std::log(p_sum);
        const double q_log_z = double(q_max) + std::log(q_sum);
        double       kl      = 0.0;
        for (int32_t i = 0; i < n_vocab; ++i) {
            const double log_p = double(p_logits[i]) - p_log_z;
            const double log_q = double(q_logits[i]) - q_log_z;
            kl += std::exp(log_p) * (log_p - log_q);
        }
        result.kl_mean += kl;
    }
    result.kl_mean /= baseline.targets.size();
    result.top1_agreement = double(top1_equal) / baseline.targets.size();
    return result;
}

std::vector<llama_token> generate_greedy(llama_context *     ctx,
                                         const llama_vocab * vocab,
                                         llama_token         final_prompt_token,
                                         llama_pos           final_prompt_position,
                                         int                 n_generate) {
    std::vector<llama_token> result;
    llama_batch              batch    = llama_batch_init(1, 0, 1);
    llama_token              input    = final_prompt_token;
    llama_pos                position = final_prompt_position;
    for (int i = 0; i < n_generate; ++i) {
        if (!decode_one(ctx, batch, input, position++)) {
            result.clear();
            break;
        }
        const float * logits  = llama_get_logits_ith(ctx, 0);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        llama_token   next    = llama_token(std::max_element(logits, logits + n_vocab) - logits);
        if (llama_vocab_is_eog(vocab, next)) {
            break;
        }
        result.push_back(next);
        input = next;
    }
    llama_batch_free(batch);
    return result;
}

bool state_save(llama_context * ctx, std::vector<uint8_t> & state) {
    state.resize(llama_state_get_size(ctx));
    const size_t written = llama_state_get_data(ctx, state.data(), state.size());
    if (written == 0 || written > state.size()) {
        return false;
    }
    state.resize(written);
    return true;
}

bool state_restore(llama_context * ctx, const std::vector<uint8_t> & state) {
    return llama_state_set_data(ctx, state.data(), state.size()) == state.size();
}

bool build_rope_maps(llama_context * ctx, capture_state * capture, int64_t n_token, std::vector<rope_map> & maps) {
    const tensor_series * before[] = { &capture->k_pre, &capture->k_pre_secondary };
    const tensor_series * after[]  = { &capture->k_post, &capture->k_post_secondary };
    for (int i = 0; i < 2; ++i) {
        if (before[i]->n == 0 && after[i]->n == 0) {
            continue;
        }
        if (before[i]->n != n_token || after[i]->n != n_token) {
            return false;
        }
        const rope_map adjacent = derive_rope_map(*before[i], *after[i], false);
        const rope_map split    = derive_rope_map(*before[i], *after[i], true);
        maps.push_back(adjacent.fit_error < split.fit_error ? adjacent : split);
        const rope_map & map = maps.back();
        std::printf("Dense round-trip RoPE: D=%lld layout=%s relative_fit=%.8g\n", (long long) map.d,
                    map.split ? "split-half" : "adjacent", map.fit_error);
    }
    ggml_backend_sched_set_eval_callback(ctx->get_sched(), nullptr, nullptr);
    capture->k_pre.data.clear();
    capture->k_pre.data.shrink_to_fit();
    capture->k_post.data.clear();
    capture->k_post.data.shrink_to_fit();
    capture->k_pre_secondary.data.clear();
    capture->k_pre_secondary.data.shrink_to_fit();
    capture->k_post_secondary.data.clear();
    capture->k_post_secondary.data.shrink_to_fit();
    return !maps.empty();
}

int run_ppl_mode(llama_context *                  ctx,
                 const llama_vocab *              vocab,
                 const std::vector<llama_token> & tokens,
                 int32_t                          n_batch,
                 capture_state *                  capture) {
    const int       n_eval_requested = env_int("TURBOANCHORKV_PPL_TOKENS", 128);
    const int       ratio            = env_int("TURBOANCHORKV_RATIO", 10);
    const int       window           = env_int("TURBOANCHORKV_WINDOW", 32);
    const bool      identity         = env_string("TURBOANCHORKV_DENSE_MODE", "compress") == "identity";
    const ggml_type residual_type    = residual_type_from_env();
    const int       keep_edge_layers = env_int("TURBOANCHORKV_KEEP_EDGE_LAYERS", 0);
    if (ratio <= 1 || tokens.size() < 4) {
        std::fprintf(stderr, "TurboAnchorKV PPL mode needs ratio > 1 and at least four tokens\n");
        return 1;
    }
    const size_t n_eval      = std::min<size_t>(std::max(2, n_eval_requested), tokens.size() - 2);
    const size_t first_input = tokens.size() - n_eval - 1;
    if (!decode_range(ctx, tokens, 0, first_input, n_batch)) {
        return 1;
    }
    std::fprintf(stderr, "TurboAnchorKV PPL prefix ready: tokens=%zu kpre=%lld kpost=%lld\n", first_input,
                 (long long) (capture ? capture->k_pre.n : 0), (long long) (capture ? capture->k_post.n : 0));

    std::vector<rope_map>         rope_maps;
    const std::vector<rope_map> * rope_maps_ptr = nullptr;
    if (env_string("TURBOANCHORKV_K_SPACE", "pre") == "pre") {
        if (!capture || !build_rope_maps(ctx, capture, int64_t(first_input), rope_maps)) {
            std::fprintf(stderr, "failed to capture pre/post-RoPE keys for dense round-trip\n");
            return 1;
        }
        rope_maps_ptr = &rope_maps;
    }

    std::vector<uint8_t> state;
    if (!state_save(ctx, state)) {
        std::fprintf(stderr, "failed to save baseline state\n");
        return 1;
    }
    if (!state_restore(ctx, state)) {
        std::fprintf(stderr, "failed to normalize baseline state through restore\n");
        return 1;
    }
    const suffix_eval baseline = evaluate_suffix(ctx, vocab, tokens, first_input);
    if (baseline.targets.empty() || !state_restore(ctx, state)) {
        std::fprintf(stderr, "failed to evaluate or restore baseline state\n");
        return 1;
    }

    roundtrip_stats stats;
    const auto      start = clock_type::now();
    if (!cache_roundtrip(ctx, ratio, window, identity, residual_type, keep_edge_layers, rope_maps_ptr, stats)) {
        return 1;
    }
    const double      roundtrip_s = elapsed_seconds(start);
    const suffix_eval compressed  = evaluate_suffix(ctx, vocab, tokens, first_input);
    if (compressed.targets.size() != baseline.targets.size()) {
        return 1;
    }
    const distribution_drift drift          = compare_logits(baseline, compressed, llama_vocab_n_tokens(vocab));
    const double             ppl_baseline   = std::exp(baseline.nll / baseline.targets.size());
    const double             ppl_compressed = std::exp(compressed.nll / compressed.targets.size());
    std::printf("TurboAnchorKV PPL evaluation\n");
    std::printf("tokens_prefix=%zu tokens_eval=%zu layers=%zu ratio_simulated=%.4f roundtrip_s=%.3f\n", first_input,
                baseline.targets.size(), stats.layers,
                double(stats.bytes_full) / std::max<size_t>(1, stats.bytes_simulated), roundtrip_s);
    std::printf("ppl_baseline=%.8f ppl_compressed=%.8f ppl_delta=%+.8f ppl_relative=%+.6f%%\n", ppl_baseline,
                ppl_compressed, ppl_compressed - ppl_baseline, 100.0 * (ppl_compressed / ppl_baseline - 1.0));
    std::printf("kl_mean=%.10f top1_agreement=%.6f\n", drift.kl_mean, drift.top1_agreement);
    return 0;
}

int run_generate_mode(llama_context *                  ctx,
                      const llama_vocab *              vocab,
                      const std::vector<llama_token> & tokens,
                      int32_t                          n_batch,
                      capture_state *                  capture) {
    const int         ratio            = env_int("TURBOANCHORKV_RATIO", 10);
    const int         window           = env_int("TURBOANCHORKV_WINDOW", 32);
    const int         n_generate       = env_int("TURBOANCHORKV_GENERATE_TOKENS", 64);
    const std::string expected         = env_string("TURBOANCHORKV_EXPECT", "");
    const bool        identity         = env_string("TURBOANCHORKV_DENSE_MODE", "compress") == "identity";
    const ggml_type   residual_type    = residual_type_from_env();
    const int         keep_edge_layers = env_int("TURBOANCHORKV_KEEP_EDGE_LAYERS", 0);
    if (ratio <= 1 || tokens.size() < 2 || n_generate < 1) {
        std::fprintf(stderr,
                     "TurboAnchorKV generation mode needs ratio > 1, at least two prompt tokens, and a positive "
                     "generation count\n");
        return 1;
    }

    const size_t final = tokens.size() - 1;
    if (!decode_range(ctx, tokens, 0, final, n_batch)) {
        return 1;
    }
    std::vector<rope_map>         rope_maps;
    const std::vector<rope_map> * rope_maps_ptr = nullptr;
    if (env_string("TURBOANCHORKV_K_SPACE", "pre") == "pre") {
        if (!capture || !build_rope_maps(ctx, capture, int64_t(final), rope_maps)) {
            std::fprintf(stderr, "failed to capture pre/post-RoPE keys for dense round-trip\n");
            return 1;
        }
        rope_maps_ptr = &rope_maps;
    }
    std::vector<uint8_t> state;
    if (!state_save(ctx, state)) {
        return 1;
    }
    if (!state_restore(ctx, state)) {
        return 1;
    }
    const auto baseline = generate_greedy(ctx, vocab, tokens[final], llama_pos(final), n_generate);
    if (!state_restore(ctx, state)) {
        return 1;
    }

    roundtrip_stats stats;
    const auto      start = clock_type::now();
    if (!cache_roundtrip(ctx, ratio, window, identity, residual_type, keep_edge_layers, rope_maps_ptr, stats)) {
        return 1;
    }
    const double      roundtrip_s     = elapsed_seconds(start);
    const auto        compressed      = generate_greedy(ctx, vocab, tokens[final], llama_pos(final), n_generate);
    const std::string text_baseline   = common_detokenize(vocab, baseline, false);
    const std::string text_compressed = common_detokenize(vocab, compressed, false);
    size_t            common          = 0;
    while (common < baseline.size() && common < compressed.size() && baseline[common] == compressed[common]) {
        ++common;
    }
    const bool expected_found = expected.empty() || text_compressed.find(expected) != std::string::npos;
    std::printf("TurboAnchorKV generation evaluation\n");
    std::printf("tokens_prompt=%zu layers=%zu ratio_simulated=%.4f roundtrip_s=%.3f\n", tokens.size(), stats.layers,
                double(stats.bytes_full) / std::max<size_t>(1, stats.bytes_simulated), roundtrip_s);
    std::printf("tokens_baseline=%zu tokens_compressed=%zu common_prefix=%zu exact_match=%s expected_match=%s\n",
                baseline.size(), compressed.size(), common, baseline == compressed ? "yes" : "no",
                expected_found ? "yes" : "no");
    std::printf("--- baseline ---\n%s\n--- compressed ---\n%s\n", text_baseline.c_str(), text_compressed.c_str());
    return expected_found ? 0 : 2;
}

bool validate_capture(const capture_state & c) {
    const bool shapes = c.q_post.n == c.k_post.n && c.k_pre.n == c.k_post.n && c.v.n == c.k_post.n &&
                        c.k_pre.d == c.k_post.d && c.v.d == c.k_post.d && c.k_pre.h == c.k_post.h &&
                        c.v.h == c.k_post.h && c.q_post.d == c.k_post.d && c.q_post.h % c.k_post.h == 0 &&
                        c.k_post.d % 128 == 0;
    if (!shapes) {
        std::fprintf(
            stderr,
            "capture mismatch: q={%lld,%lld,%lld} kpre={%lld,%lld,%lld} kpost={%lld,%lld,%lld} v={%lld,%lld,%lld}\n",
            (long long) c.q_post.d, (long long) c.q_post.h, (long long) c.q_post.n, (long long) c.k_pre.d,
            (long long) c.k_pre.h, (long long) c.k_pre.n, (long long) c.k_post.d, (long long) c.k_post.h,
            (long long) c.k_post.n, (long long) c.v.d, (long long) c.v.h, (long long) c.v.n);
    }
    return shapes;
}

bool self_test_rope() {
    tensor_series before;
    before.d = 128;
    before.h = 3;
    before.n = 4;
    before.data.resize(size_t(before.d * before.h * before.n));
    for (size_t i = 0; i < before.data.size(); ++i) {
        before.data[i] = std::sin(float(i + 1) * 0.017f) + 0.2f * std::cos(float(i + 3) * 0.031f);
    }

    rope_map expected;
    expected.split = true;
    expected.d     = before.d;
    expected.n     = before.n;
    expected.cosine.resize(size_t(expected.n * expected.d / 2));
    expected.sine.resize(expected.cosine.size());
    for (int64_t t = 0; t < expected.n; ++t) {
        for (int64_t p = 0; p < expected.d / 2; ++p) {
            const float angle                               = float(t + 1) * float(p + 1) * 0.003f;
            expected.cosine[size_t(t * expected.d / 2 + p)] = std::cos(angle);
            expected.sine[size_t(t * expected.d / 2 + p)]   = std::sin(angle);
        }
    }

    const tensor_series after    = expected.apply(before);
    const rope_map      split    = derive_rope_map(before, after, true);
    const rope_map      adjacent = derive_rope_map(before, after, false);
    return split.fit_error < 1e-6 && adjacent.fit_error > 1e-2;
}

bool self_test_uniform_anchors() {
    const auto anchors = select_anchors({}, 2, 1024, 4, 8, "uniform", 42);
    if (anchors.size() != 2) {
        return false;
    }
    for (const auto & head : anchors) {
        if (head.size() != 8 || head[4] != 1020 || head[5] != 1021 || head[6] != 1022 || head[7] != 1023) {
            return false;
        }
        const int32_t expected[] = { 127, 382, 637, 892 };
        for (int i = 0; i < 4; ++i) {
            if (head[size_t(i)] != expected[i]) {
                return false;
            }
        }
    }
    return true;
}

bool self_test_norm_ranking() {
    projection p;
    p.n_head    = 1;
    p.n_token   = 3;
    p.d         = 128;
    p.is_anchor = { 1, 0, 0 };
    p.residual.assign(size_t(p.n_token * p.d), 0.0f);
    p.residual[size_t(p.d)]     = 1.0f;
    p.residual[size_t(2 * p.d)] = 2.0f;
    const std::vector<float> utility(3, 0.0f);
    const auto               order = rank_candidates(utility, p, "norm", 42);
    return order.size() == 2 && order[0] == 2 && order[1] == 1;
}

double self_test_reconstruction_error(const tensor_series & exact, const tensor_series & reconstructed) {
    double error = 0.0;
    for (size_t i = 0; i < exact.data.size(); ++i) {
        const double delta = double(exact.data[i]) - reconstructed.data[i];
        error += delta * delta;
    }
    return std::sqrt(error / exact.data.size());
}

bool self_test_residual_codecs() {
    tensor_series exact;
    exact.d = 128;
    exact.h = 1;
    exact.n = 4;
    exact.data.resize(size_t(exact.d * exact.n));
    for (size_t i = 0; i < exact.data.size(); ++i) {
        exact.data[i] = std::sin(float(i + 1) * 0.071f) + 0.3f * std::cos(float(i + 5) * 0.037f);
    }
    const std::vector<std::vector<int32_t>> anchors = { { 0 } };
    const projection                        p       = build_projection(exact, anchors);
    const std::vector<float>                utility(size_t(p.n_token), 0.0f);
    const auto                              order    = rank_candidates(utility, p, "norm", 42);
    const packed_projection                 packed2  = pack_projection(p, order, order.size(), GGML_TYPE_TURBO2_0);
    const packed_projection                 packed4  = pack_projection(p, order, order.size(), GGML_TYPE_TURBO4_0);
    const packed_projection                 packed16 = pack_projection(p, order, order.size(), GGML_TYPE_F16);
    const tensor_series                     turbo2   = unpack_projection(packed2);
    const tensor_series                     turbo4   = unpack_projection(packed4);
    const tensor_series                     f16      = unpack_projection(packed16);
    const double                            error2   = self_test_reconstruction_error(exact, turbo2);
    const double                            error4   = self_test_reconstruction_error(exact, turbo4);
    const double                            error16  = self_test_reconstruction_error(exact, f16);
    const bool                              slots    = packed2.residual_slot[order.front()] == 0 &&
                       packed2.residual_slot[order.back()] == uint32_t(order.size() - 1) &&
                       packed2.serialized_size() == packed_projection_base_size(p) + order.size() * packed2.row_size();
    return slots && std::isfinite(error2) && std::isfinite(error4) && std::isfinite(error16) && error4 < error2 &&
           error16 < error4;
}

bool self_test_packed_slots() {
    tensor_series exact;
    exact.d = 128;
    exact.h = 2;
    exact.n = 5;
    exact.data.resize(size_t(exact.d * exact.h * exact.n));
    for (size_t i = 0; i < exact.data.size(); ++i) {
        exact.data[i] = std::sin(float(i + 3) * 0.043f) + 0.2f * std::cos(float(i + 11) * 0.019f);
    }
    const std::vector<std::vector<int32_t>> anchors = {
        { 0, 4 },
        { 0, 4 }
    };
    const projection         p = build_projection(exact, anchors);
    const std::vector<float> utility(size_t(p.n_head * p.n_token), 0.0f);
    const auto               order   = rank_candidates(utility, p, "norm", 42);
    const packed_projection  packed  = pack_projection(p, order, 3, GGML_TYPE_TURBO4_0);
    const tensor_series      decoded = unpack_projection(packed);

    size_t selected = 0;
    for (uint32_t slot : packed.residual_slot) {
        selected += slot != UINT32_MAX;
    }
    if (selected != 3 || packed.residual_slot[order[0]] != 0 || packed.residual_slot[order[1]] != 1 ||
        packed.residual_slot[order[2]] != 2 || packed.residual_slot[order[3]] != UINT32_MAX ||
        packed.serialized_size() != packed_projection_base_size(p) + 3 * packed.row_size()) {
        return false;
    }

    const size_t  selected_ti    = order[0];
    const int64_t selected_h     = int64_t(selected_ti) / p.n_token;
    const int64_t selected_t     = int64_t(selected_ti) % p.n_token;
    const size_t  omitted_ti     = order[3];
    const int64_t omitted_h      = int64_t(omitted_ti) / p.n_token;
    const int64_t omitted_t      = int64_t(omitted_ti) % p.n_token;
    double        selected_error = 0.0;
    double        omitted_error  = 0.0;
    for (int64_t j = 0; j < p.d; ++j) {
        const double selected_delta = exact.row(selected_t, selected_h)[j] - decoded.row(selected_t, selected_h)[j];
        const double omitted_delta  = exact.row(omitted_t, omitted_h)[j] - decoded.row(omitted_t, omitted_h)[j];
        selected_error += selected_delta * selected_delta;
        omitted_error += omitted_delta * omitted_delta;
    }
    return selected_error < omitted_error;
}

int run_self_tests() {
    const bool rope      = self_test_rope();
    const bool anchors   = self_test_uniform_anchors();
    const bool ranking   = self_test_norm_ranking();
    const bool residuals = self_test_residual_codecs();
    const bool slots     = self_test_packed_slots();
    std::printf("TurboAnchorKV self-test: rope=%s anchors=%s ranking=%s residuals=%s slots=%s\n", rope ? "ok" : "fail",
                anchors ? "ok" : "fail", ranking ? "ok" : "fail", residuals ? "ok" : "fail", slots ? "ok" : "fail");
    return rope && anchors && ranking && residuals && slots ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) {
        return run_self_tests();
    }

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    capture_state capture;
    capture.layer                 = env_int("TURBOANCHORKV_LAYER", 0);
    capture.rope_layer_secondary  = env_int("TURBOANCHORKV_ROPE_LAYER_SECONDARY", 5);
    capture.q_name                = env_string("TURBOANCHORKV_Q_NAME", "Qcur_pos");
    capture.k_pre_name            = env_string("TURBOANCHORKV_K_PRE_NAME", "Kcur_normed");
    capture.k_post_name           = env_string("TURBOANCHORKV_K_POST_NAME", "Kcur_pos");
    capture.v_name                = env_string("TURBOANCHORKV_V_NAME", "Vcur_normed");
    const std::string anchor_mode = env_string("TURBOANCHORKV_ANCHOR_MODE", "uniform");
    const std::string rank_mode   = env_string("TURBOANCHORKV_RANK_MODE", "norm");
    const std::string k_space     = env_string("TURBOANCHORKV_K_SPACE", "pre");
    const std::string mode        = env_string("TURBOANCHORKV_MODE", "analyze");
    const int         seed        = env_int("TURBOANCHORKV_SEED", 42);
    capture.rope_only             = mode != "analyze";
    if (mode == "analyze" || k_space == "pre") {
        params.cb_eval           = capture_callback;
        params.cb_eval_user_data = &capture;
    }
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);
    auto            init = common_init_from_params(params);
    llama_context * ctx  = init ? init->context() : nullptr;
    if (!ctx) {
        std::fprintf(stderr, "failed to initialize model\n");
        return 1;
    }

    const llama_vocab *      vocab  = llama_model_get_vocab(llama_get_model(ctx));
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (tokens.empty()) {
        std::fprintf(stderr, "prompt produced no tokens\n");
        return 1;
    }
    const int max_tokens = env_int("TURBOANCHORKV_MAX_TOKENS", 0);
    if (max_tokens > 0 && tokens.size() > size_t(max_tokens)) {
        tokens.resize(size_t(max_tokens));
    }
    turbo3_cpu_wht_group_size = 128;

    if (mode == "ppl") {
        return run_ppl_mode(ctx, vocab, tokens, params.n_batch, &capture);
    }
    if (mode == "generate") {
        return run_generate_mode(ctx, vocab, tokens, params.n_batch, &capture);
    }
    if (mode != "analyze") {
        std::fprintf(stderr, "unknown TURBOANCHORKV_MODE: %s\n", mode.c_str());
        return 1;
    }

    std::printf("TurboAnchorKV capture: layer=%d tokens=%zu Q=%s Kpre=%s Kpost=%s V=%s\n", capture.layer, tokens.size(),
                capture.q_name.c_str(), capture.k_pre_name.c_str(), capture.k_post_name.c_str(),
                capture.v_name.c_str());
    if (!decode_range(ctx, tokens, 0, tokens.size(), params.n_batch)) {
        return 1;
    }
    if (!validate_capture(capture)) {
        return 1;
    }

    const int window        = std::min<int64_t>(env_int("TURBOANCHORKV_WINDOW", 32), capture.k_post.n);
    const int anchor_budget = std::max(window, int(capture.k_post.n / 128));

    std::printf("Captured Q={D=%lld,H=%lld,S=%lld} KV={D=%lld,H=%lld,S=%lld} group=%lld W=%d k=%d\n",
                (long long) capture.q_post.d, (long long) capture.q_post.h, (long long) capture.q_post.n,
                (long long) capture.k_post.d, (long long) capture.k_post.h, (long long) capture.k_post.n,
                (long long) (capture.q_post.h / capture.k_post.h), window, anchor_budget);

    auto       start        = clock_type::now();
    const auto observations = make_observations(capture.q_post, capture.k_post, capture.v, window);
    const auto anchors      = select_anchors(observations, capture.k_post.h, capture.k_post.n, window, anchor_budget,
                                             anchor_mode, uint32_t(seed));
    std::printf("Observations and anchors: %.3f s\n", elapsed_seconds(start));

    start                 = clock_type::now();
    const double pre_cos  = mean_nearest_cosine(capture.k_pre, anchors);
    const double post_cos = mean_nearest_cosine(capture.k_post, anchors);
    std::printf("Nearest-anchor |cosine|: pre-RoPE=%.6f post-RoPE=%.6f delta=%+.6f (%.3f s)\n", pre_cos, post_cos,
                pre_cos - post_cos, elapsed_seconds(start));

    start                          = clock_type::now();
    const rope_map   rope_adjacent = derive_rope_map(capture.k_pre, capture.k_post, false);
    const rope_map   rope_split    = derive_rope_map(capture.k_pre, capture.k_post, true);
    const rope_map & rope          = rope_adjacent.fit_error < rope_split.fit_error ? rope_adjacent : rope_split;
    std::printf("Recovered RoPE: layout=%s relative_fit=%.8g (alternate=%.8g, %.3f s)\n",
                rope.split ? "split-half" : "adjacent", rope.fit_error,
                rope.split ? rope_adjacent.fit_error : rope_split.fit_error, elapsed_seconds(start));

    start                              = clock_type::now();
    const bool            use_pre_rope = k_space != "post";
    const tensor_series & k_source     = use_pre_rope ? capture.k_pre : capture.k_post;
    const projection      pk           = build_projection(k_source, anchors);
    const projection      pk_error     = use_pre_rope ? rotate_projection_residual(pk, rope) : pk;
    const projection      pv           = build_projection(capture.v, anchors);
    std::printf("Representation: K=%s anchors=%s rank=%s (projection %.3f s)\n",
                use_pre_rope ? "pre-RoPE" : "post-RoPE", anchor_mode.c_str(), rank_mode.c_str(),
                elapsed_seconds(start));

    start = clock_type::now();
    utilities utility;
    if (rank_mode == "utility") {
        utility = compute_utilities(observations, capture.v, pk_error, pv);
    } else {
        utility.k.assign(size_t(pk.n_head * pk.n_token), 0.0f);
        utility.v.assign(size_t(pv.n_head * pv.n_token), 0.0f);
    }
    const auto k_order = rank_candidates(utility.k, pk_error, rank_mode, uint32_t(seed + 81));
    const auto v_order = rank_candidates(utility.v, pv, rank_mode, uint32_t(seed + 414));
    std::printf("Residual ranking: %.3f s\n", elapsed_seconds(start));

    const int64_t   S                = capture.k_post.n;
    const int64_t   H                = capture.k_post.h;
    const int64_t   D                = capture.k_post.d;
    const ggml_type residual_type    = residual_type_from_env();
    const size_t    full_bytes       = size_t(4 * S * H * D);
    const size_t    base_bytes       = packed_projection_base_size(pk) + packed_projection_base_size(pv);
    const size_t    k_residual_bytes = ggml_row_size(residual_type, D);
    const size_t    v_residual_bytes = ggml_row_size(residual_type, capture.v.d);
    std::printf("Bytes: FullKV=%zu base=%zu (%.2fx floor) residual=%s K=%zu V=%zu\n", full_bytes, base_bytes,
                double(full_bytes) / base_bytes, ggml_type_name(residual_type), k_residual_bytes, v_residual_bytes);
    std::printf("ratio,ksplit,nk,nv,actual_ratio,mean_rel_l2,p95_rel_l2,max_rel_l2,reconstruct_s,evaluate_s\n");

    const int    ratios[] = { 5, 10, 20 };
    const double splits[] = { 0.50, 0.67, 0.75, 0.90 };
    for (int ratio : ratios) {
        const size_t target    = full_bytes / size_t(ratio);
        const size_t available = target > base_bytes ? target - base_bytes : 0;
        for (double split : splits) {
            size_t nk                             = size_t(std::floor(available * split / k_residual_bytes));
            size_t nv                             = size_t(std::floor(available * (1.0 - split) / v_residual_bytes));
            nk                                    = std::min(nk, k_order.size());
            nv                                    = std::min(nv, v_order.size());
            start                                 = clock_type::now();
            const packed_projection packed_k      = pack_projection(pk, k_order, nk, residual_type);
            const packed_projection packed_v      = pack_projection(pv, v_order, nv, residual_type);
            const size_t            used          = packed_k.serialized_size() + packed_v.serialized_size();
            const tensor_series     khat_native   = unpack_projection(packed_k);
            const tensor_series     khat          = use_pre_rope ? rope.apply(khat_native) : khat_native;
            const tensor_series     vhat          = unpack_projection(packed_v);
            const double            reconstruct_s = elapsed_seconds(start);
            start                                 = clock_type::now();
            const error_stats error               = attention_error(observations, khat, vhat);
            const double      evaluate_s          = elapsed_seconds(start);
            std::printf("%d,%.2f,%zu,%zu,%.4f,%.8f,%.8f,%.8f,%.3f,%.3f\n", ratio, split, nk, nv,
                        double(full_bytes) / used, error.mean, error.p95, error.max, reconstruct_s, evaluate_s);
        }
    }

    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
