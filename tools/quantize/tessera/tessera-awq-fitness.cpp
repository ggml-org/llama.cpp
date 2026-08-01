//
// tessera-awq-fitness.cpp
//
// C++ port of the AWQ GA fitness. Faithful numeric port of
// tools/tessera/awq-evolve.py: _ternary_reconstruct, relative_output_error,
// _evaluate_layer (+ _layer_features), _evaluate_uncached. Parity is checked
// by test_awq_fitness.cpp against a Python-generated fixture.
//
// dtype policy (matches Python/numpy defaults):
//   - Reconstruction math (scale, transformed, ternary, row_scale) is float
//     to track Python's float32 array math. Python keeps these arrays f32
//     because every numpy op there is array-or-Python-scalar, which numpy
//     keeps in the array's dtype (no upcast to f64).
//   - Reductions (median, mean, matmul accumulators, quantile, fitness) use
//     double to match numpy's default reduction dtype for f32 inputs (numpy
//     computes mean/sum/median/quantile in f64 even for f32 arrays when the
//     default dtype is used - which is what Python does here, since none of
//     these calls pass dtype=np.float32).
//

#include "tessera-awq-fitness.h"
#include "tessera-awq.h"

// BLAS: the two matmuls in ts_awq_relative_output_error dominate the GA cost
// (O(n_tokens * out_dim * in_dim) per evaluation). Use single-precision
// cblas_sgemm to match Python's numpy matmul dtype exactly (activations @
// weight.T stays float32 because numpy preserves the f32 input dtype); the
// error/ref reductions below stay in double to match numpy's default f64
// reductions for f32 inputs. cblas_dgemm would diverge from Python's f32
// matmul accumulation order.
#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#define TS_HAS_CBLAS 1
#elif defined(GGML_USE_OPENBLAS)
#include <cblas.h>
#define TS_HAS_CBLAS 1
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

// Linear-interpolation quantile, matching numpy.quantile default (method=
// 'linear'): idx = (n-1)*q; lo = floor(idx); hi = ceil(idx); result is a
// linear blend of the sorted[lo]/sorted[hi] values. Input is taken by value
// so we can sort in place.
double ts_quantile_linear(std::vector<double> vals, double q) {
    const size_t n = vals.size();
    if (n == 0) {
        return 0.0;
    }
    if (n == 1) {
        return vals[0];
    }
    std::sort(vals.begin(), vals.end());
    double pos = (double)(n - 1) * q;
    size_t lo = (size_t)std::floor(pos);
    size_t hi = (size_t)std::ceil(pos);
    if (lo >= n - 1) {
        return vals[n - 1];
    }
    if (lo == hi) {
        return vals[lo];
    }
    double frac = pos - (double)lo;
    return vals[lo] * (1.0 - frac) + vals[hi] * frac;
}

// Median = quantile(0.5). Implementing it directly avoids one sort round-trip
// for the per-layer hot path (we sort the finite-positive view once).
double ts_median_sorted(const std::vector<double> & sorted) {
    const size_t n = sorted.size();
    if (n == 0) {
        return 0.0;
    }
    if (n % 2 == 1) {
        return sorted[n / 2];
    }
    return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
}

// f32 reduction helper (numpy defaults to f64 reductions for f32 input).
inline double ts_sum_f64(const float * p, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) {
        s += (double)p[i];
    }
    return s;
}

// double-array mean (used for the per-layer error aggregations, which are
// already in f64 to match numpy).
inline double ts_mean_double(const double * p, int64_t n) {
    if (n <= 0) {
        return 0.0;
    }
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) {
        s += p[i];
    }
    return s / (double)n;
}

}  // namespace

void ts_awq_ternary_reconstruct(const float * W,
                                const ts_policy_genes & g,
                                const float * importance,
                                int64_t out_dim, int64_t in_dim,
                                float * reconstructed_out) {
    if (!W || !importance || !reconstructed_out || out_dim <= 0 || in_dim <= 0) {
        return;
    }

    const int64_t n = out_dim * in_dim;

    // safe = max(importance, 1e-8), f32 (Python: np.maximum(importance, 1e-8)).
    std::vector<float> safe(in_dim);
    for (int64_t j = 0; j < in_dim; j++) {
        float v = importance[j];
        safe[j] = std::max(v, 1e-8f);
    }

    // reference = median of the finite, strictly-positive safe entries
    // (Python: np.median(safe[isfinite(safe) & (safe > 0)])). median is f64.
    double reference = 0.0;
    {
        std::vector<double> fin_pos;
        fin_pos.reserve(in_dim);
        for (int64_t j = 0; j < in_dim; j++) {
            float v = safe[j];
            if (std::isfinite(v) && v > 0.0f) {
                fin_pos.push_back((double)v);
            }
        }
        std::sort(fin_pos.begin(), fin_pos.end());
        reference = ts_median_sorted(fin_pos);
        // Python calls float(np.median(...)) which yields 0.0 for an empty
        // selection (nan would be returned by np.median, but float(nan) is
        // still nan; we mirror Python's float() of an empty median which is
        // nan, then max(nan, 1e-8) propagates nan and the subsequent divide
        // would yield nan. To keep parity with a degenerate fixture we follow
        // Python exactly: if fin_pos is empty Python emits nan and the math
        // falls through nan_to_num -> reference 1e-8 path effectively. We
        // match that by leaving reference at the median of nothing = nan and
        // letting the max(reference, 1e-8) below propagate it via nan_to_num.
        if (fin_pos.empty()) {
            reference = std::numeric_limits<double>::quiet_NaN();
        }
    }
    double ref_clamped = std::max(reference, 1e-8);

    // relative = nan_to_num(safe / max(reference, 1e-8),
    //                        nan=1.0, posinf=256.0, neginf=1/256.0)
    // then clip to [1/256, 256]. Python keeps relative f32 here.
    std::vector<float> relative(in_dim);
    const float k_inv256 = 1.0f / 256.0f;
    for (int64_t j = 0; j < in_dim; j++) {
        float num = safe[j];
        float denom = (float)ref_clamped;
        float r;
        if (denom == 0.0f) {
            // would be inf or nan; mimic Python's nan_to_num substitution
            r = (num == 0.0f) ? 1.0f : (num > 0.0f ? 256.0f : k_inv256);
        } else {
            r = num / denom;
            if (std::isnan(r))      r = 1.0f;
            else if (std::isinf(r)) r = r > 0.0f ? 256.0f : k_inv256;
        }
        relative[j] = std::max(k_inv256, std::min(256.0f, r));
    }

    // safe = relative * reference (Python reassigns safe; f32 in, f32 out).
    // We don't reuse safe[j] below until the residual fold, so update in place.
    for (int64_t j = 0; j < in_dim; j++) {
        safe[j] = relative[j] * (float)reference;
    }

    // scale = relative ** alpha, dtype=float32 (Python forces f32 here).
    std::vector<float> scale(in_dim);
    for (int64_t j = 0; j < in_dim; j++) {
        // np.power on f32 with a Python float exponent stays f32. Match by
        // computing in float (use float powf).
        scale[j] = powf(relative[j], g.alpha);
    }

    // transformed = weight * scale[None,:]; then clip per row to
    // [-row_limit, row_limit] where row_limit = max(|transformed|) * clip.
    // We fuse transformed + clip into one buffer.
    std::vector<float> transformed(n);
    for (int64_t i = 0; i < out_dim; i++) {
        const float * wrow = W + i * in_dim;
        float * trow = transformed.data() + i * in_dim;
        float row_max_abs = 0.0f;
        for (int64_t j = 0; j < in_dim; j++) {
            float v = wrow[j] * scale[j];
            trow[j] = v;
            float a = std::fabs(v);
            if (a > row_max_abs) {
                row_max_abs = a;
            }
        }
        float row_limit = row_max_abs * g.clip;
        for (int64_t j = 0; j < in_dim; j++) {
            float v = trow[j];
            v = std::max(-row_limit, std::min(row_limit, v));
            trow[j] = v;
        }
    }

    // threshold = mean(|transformed|, axis=1) * ternary_threshold, per row.
    // ternary: >=thr -> +1, <=-thr -> -1, else 0.
    // row_scale = sum(|transformed|*|ternary|) / sum(|ternary|)
    //             (0 where denominator 0 - the np.divide where= pattern).
    // reconstructed = ternary * row_scale.
    std::vector<float> ternary(n, 0.0f);
    std::vector<float> reconstructed(n, 0.0f);
    for (int64_t i = 0; i < out_dim; i++) {
        float * trow = transformed.data() + i * in_dim;
        float * yrow = ternary.data() + i * in_dim;
        // mean(|transformed|) in f64 (numpy reduction default).
        double sum_abs = 0.0;
        for (int64_t j = 0; j < in_dim; j++) {
            sum_abs += std::fabs((double)trow[j]);
        }
        double mean_abs = sum_abs / (double)in_dim;
        float thr = (float)(mean_abs * (double)g.ternary_threshold);
        for (int64_t j = 0; j < in_dim; j++) {
            float v = trow[j];
            if (v >= thr) {
                yrow[j] = 1.0f;
            } else if (v <= -thr) {
                yrow[j] = -1.0f;
            } else {
                yrow[j] = 0.0f;
            }
        }
        // row_scale: f64 reductions, then cast back to f32 for the multiply
        // (Python divides f32 arrays with where= and gets f32 row_scale,
        // because numerator/denominator are f32; np.mean|sum on f32 default
        // to f64 but the result is written into np.zeros_like(denominator)
        // which is f32 - so the final row_scale is f32).
        double num = 0.0;
        double den = 0.0;
        for (int64_t j = 0; j < in_dim; j++) {
            float t = yrow[j];
            float at = std::fabs(t);
            num += (double)(std::fabs(trow[j]) * at);
            den += (double)at;
        }
        float row_scale = (den != 0.0) ? (float)(num / den) : 0.0f;
        float * rrow = reconstructed.data() + i * in_dim;
        for (int64_t j = 0; j < in_dim; j++) {
            rrow[j] = yrow[j] * row_scale;
        }
    }

    // Outlier residual fold.
    // residual_count = max(1, ceil(weight.size * outlier_fraction)).
    // error_score = (transformed - reconstructed)^2 * safe[None,:].
    // Select top residual_count indices by error_score and overwrite
    // reconstructed[selected] with transformed[selected].
    int64_t residual_count = (int64_t)std::ceil((double)n * (double)g.outlier_fraction);
    if (residual_count < 1) {
        residual_count = 1;
    }
    if (residual_count > n) {
        residual_count = n;
    }

    // Compute error_score in f32 (matches Python: f32 array math).
    std::vector<float> error_score(n);
    for (int64_t i = 0; i < out_dim; i++) {
        const float * trow = transformed.data() + i * in_dim;
        const float * rrow = reconstructed.data() + i * in_dim;
        const float * srow = safe.data();  // safe is (in_dim,)
        float * erow = error_score.data() + i * in_dim;
        for (int64_t j = 0; j < in_dim; j++) {
            float d = trow[j] - rrow[j];
            erow[j] = d * d * srow[j];
        }
    }

    // np.argpartition(error_score.reshape(-1), -residual_count)[-residual_count:]
    // selects the residual_count LARGEST entries (unordered). std::nth_element
    // with the (n - residual_count)-th element in ascending order partitions so
    // that the tail [first+n-residual_count, end) is the top-k.
    if (residual_count < n) {
        std::vector<int64_t> idx(n);
        for (int64_t k = 0; k < n; k++) {
            idx[k] = k;
        }
        auto cmp = [&](int64_t a, int64_t b) {
            return error_score[a] < error_score[b];
        };
        std::nth_element(idx.begin(), idx.begin() + (n - residual_count),
                         idx.end(), cmp);
        for (int64_t k = n - residual_count; k < n; k++) {
            int64_t flat = idx[k];
            reconstructed[flat] = transformed[flat];
        }
    } else {
        // residual_count == n: copy everything.
        for (int64_t k = 0; k < n; k++) {
            reconstructed[k] = transformed[k];
        }
    }

    // return reconstructed / scale[None,:]
    for (int64_t i = 0; i < out_dim; i++) {
        float * rrow = reconstructed.data() + i * in_dim;
        float * orow = reconstructed_out + i * in_dim;
        for (int64_t j = 0; j < in_dim; j++) {
            orow[j] = rrow[j] / scale[j];
        }
    }
}

// Compute C(M x N) = A(M x K) @ B(N x K)^T, row-major, single-precision.
// This is the BLAS dispatch for the two matmuls in relative_output_error;
// it dispatches to cblas_sgemm when Accelerate/OpenBLAS is linked and falls
// back to a naive scalar triple loop otherwise (the pre-BLAS path, kept so
// the file still builds without a BLAS dependency). The matmul is f32 to
// match Python's numpy matmul dtype (activations @ weight.T stays f32).
static void ts_fitness_matmul_at(const float * A, const float * B,
                                 float * C,
                                 int64_t M, int64_t K, int64_t N) {
    if (M <= 0 || N <= 0) {
        return;
    }
#if defined(TS_HAS_CBLAS)
    // C = A * B^T. A is (M x K) row-major, B is (N x K) row-major so B^T is
    // (K x N). Row-major strides: lda=K (A), ldb=K (B, read transposed),
    // ldc=N (C). op(A)=NoTrans, op(B)=Trans.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)M, (int)N, (int)K,
                1.0f, A, (int)K, B, (int)K, 0.0f, C, (int)N);
#else
    for (int64_t t = 0; t < M; t++) {
        const float * arow = A + t * K;
        float * crow = C + t * N;
        for (int64_t o = 0; o < N; o++) {
            const float * brow = B + o * K;
            float acc = 0.0f;
            for (int64_t j = 0; j < K; j++) {
                acc += arow[j] * brow[j];
            }
            crow[o] = acc;
        }
    }
#endif
}

double ts_awq_relative_output_error(const float * activations,
                                    const float * W,
                                    const float * reconstructed,
                                    int64_t n_tokens,
                                    int64_t in_dim,
                                    int64_t out_dim,
                                    const float * reference_or_null) {
    if (!activations || !W || !reconstructed || n_tokens <= 0 || in_dim <= 0 || out_dim <= 0) {
        return 0.0;
    }

    const int64_t ref_n = n_tokens * out_dim;

    // reference = A @ W^T  (n_tokens x out_dim, f32). Python computes this
    // once and caches; here we either use the caller's buffer or materialize
    // it via BLAS. Kept in float to match Python's f32 matmul output.
    std::vector<float> ref_buf;
    const float * ref;
    if (reference_or_null) {
        ref = reference_or_null;
    } else {
        ref_buf.resize(ref_n);
        ts_fitness_matmul_at(activations, W, ref_buf.data(),
                             n_tokens, in_dim, out_dim);
        ref = ref_buf.data();
    }

    // approximate = A @ R^T  (n_tokens x out_dim, f32) via BLAS.
    std::vector<float> approx_buf(ref_n);
    ts_fitness_matmul_at(activations, reconstructed, approx_buf.data(),
                         n_tokens, in_dim, out_dim);

    // Error reduction: sum((approx - ref)^2) and sum(ref^2). This is
    // O(n_tokens * out_dim), much smaller than the matmul, so a naive loop
    // is fine. Reductions are f64 to match numpy's default reduction dtype
    // for f32 inputs.
    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;
    for (int64_t i = 0; i < ref_n; i++) {
        double a = (double)approx_buf[i];
        double r = (double)ref[i];
        double d = a - r;
        sum_sq_err += d * d;
        sum_sq_ref += r * r;
    }

    double mean_err = sum_sq_err / (double)ref_n;
    double mean_ref = sum_sq_ref / (double)ref_n;
    return mean_err / (mean_ref + 1e-12);
}

// _layer_features: compute rms/kurtosis_excess/tail_excess on the fly from
// second_moment / fourth_moment / max_abs, then build the importance vector.
// importance = rms * (1 + moment_mix * kurtosis_excess + tail_guard * tail_excess)
static void ts_awq_layer_importance(const ts_awq_layer & layer,
                                    const ts_policy_genes & g,
                                    std::vector<float> & importance) {
    const int64_t in_dim = layer.in_dim;
    importance.assign(in_dim, 0.0f);
    for (int64_t j = 0; j < in_dim; j++) {
        double second = layer.second_moment ? (double)layer.second_moment[j] : 1.0;
        double fourth = layer.fourth_moment ? (double)layer.fourth_moment[j]
                                            : second * second;
        double maxabs = layer.max_abs ? (double)layer.max_abs[j]
                                      : std::sqrt(std::max(second, 0.0));
        double rms = std::sqrt(std::max(second, 1e-12));
        double kurt_excess = std::max(fourth / std::max(second * second, 1e-12) - 3.0, 0.0);
        double tail_excess = std::max(maxabs / std::max(rms, 1e-6) - 3.0, 0.0);
        double imp = rms * (1.0 + (double)g.moment_mix * kurt_excess
                              + (double)g.tail_guard * tail_excess);
        importance[j] = (float)imp;
    }
}

int ts_awq_evaluate_layer(const ts_awq_candidate & c,
                          const ts_awq_layer & layer,
                          ts_awq_score * out) {
    if (!out) {
        return -1;
    }
    if (!layer.weights || layer.out_dim <= 0 || layer.in_dim <= 0) {
        return -1;
    }

    const int64_t in_dim  = layer.in_dim;
    const int64_t out_dim = layer.out_dim;
    const int64_t n = out_dim * in_dim;

    std::vector<float> importance;
    ts_awq_layer_importance(layer, c.genes, importance);

    std::vector<float> reconstructed(n);
    ts_awq_ternary_reconstruct(layer.weights, c.genes, importance.data(),
                               out_dim, in_dim, reconstructed.data());

    // error = reconstructed - weight, used by the diagonal and tail losses.
    // Python: np.mean(np.square(error) * moment.reshape(1, -1)).
    // Reductions are f64.
    double diagonal_acc = 0.0;
    double tail_acc = 0.0;
    for (int64_t i = 0; i < out_dim; i++) {
        for (int64_t j = 0; j < in_dim; j++) {
            double e = (double)reconstructed[i * in_dim + j]
                     - (double)layer.weights[i * in_dim + j];
            double e2 = e * e;
            double second = layer.second_moment ? (double)layer.second_moment[j] : 1.0;
            double fourth = layer.fourth_moment ? (double)layer.fourth_moment[j]
                                                : second * second;
            diagonal_acc += e2 * second;
            tail_acc     += e2 * fourth;
        }
    }
    double diagonal_error = diagonal_acc / (double)n;
    double tail_error     = tail_acc / (double)n;

    // train_error: relative_output_error if train_activations, else diagonal.
    double train_error = diagonal_error;
    if (layer.train_activations && layer.n_tokens > 0) {
        train_error = ts_awq_relative_output_error(
            layer.train_activations, layer.weights, reconstructed.data(),
            layer.n_tokens, in_dim, out_dim, layer.ref_train_output);
    }
    // heldout_error: heldout activations if present, else fall back to train.
    double heldout_error = train_error;
    if (layer.heldout_activations && layer.n_tokens_h > 0) {
        heldout_error = ts_awq_relative_output_error(
            layer.heldout_activations, layer.weights, reconstructed.data(),
            layer.n_tokens_h, in_dim, out_dim, layer.ref_heldout_output);
    }

    out->mse           = (float)train_error;     // Score.train_error
    out->heldout_mse   = (float)heldout_error;   // Score.heldout_error
    out->relative_frob = (float)tail_error;      // Score.tail_error
    out->composite     = 0.0f;                   // per-layer has no fitness
    return 0;
}

int ts_awq_evaluate(const ts_awq_candidate & c,
                    const ts_awq_layer * layers, int64_t n_layers,
                    ts_awq_score * out) {
    if (!layers || n_layers < 1 || !out) {
        return -1;
    }

    std::vector<double> train_errors;
    std::vector<double> heldout_errors;
    std::vector<double> tail_errors;
    train_errors.reserve(n_layers);
    heldout_errors.reserve(n_layers);
    tail_errors.reserve(n_layers);

    for (int64_t i = 0; i < n_layers; i++) {
        ts_awq_score s;
        if (ts_awq_evaluate_layer(c, layers[i], &s) != 0) {
            return -1;
        }
        train_errors.push_back((double)s.mse);
        heldout_errors.push_back((double)s.heldout_mse);
        tail_errors.push_back((double)s.relative_frob);
    }

    double train_error   = ts_mean_double(train_errors.data(), n_layers);
    double heldout_error = ts_mean_double(heldout_errors.data(), n_layers);
    double tail_error    = ts_mean_double(tail_errors.data(), n_layers);
    double worst_layer   = ts_quantile_linear(heldout_errors, 0.9);
    double size_cost     = (double)c.genes.outlier_fraction;
    double fitness =
        train_error
        + 2.0 * heldout_error
        + 0.25 * worst_layer
        + 0.05 * tail_error
        + 0.15 * size_cost;

    out->mse           = (float)train_error;
    out->heldout_mse   = (float)heldout_error;
    out->relative_frob = (float)tail_error;
    out->composite     = (float)fitness;
    return 0;
}

// Default GA evaluator: single-layer composite. The GA maximizes composite,
// and Python's fitness is a *lower-is-better* quantity (it is a weighted
// reconstruction error). We expose the raw fitness as composite-positive
// only in the per-layer sense; the GA in tessera-awq.cpp treats higher
// composite as better, so we negate to drive the GA toward lower error.
// This matches the convention already used by ts_dispatch_awq_eval.
ts_awq_score ts_awq_default_eval(const ts_awq_candidate * cand,
                                 const ts_awq_layer * layer,
                                 void * ctx) {
    (void)ctx;
    ts_awq_score s = {};
    if (!cand || !layer) {
        return s;
    }
    if (ts_awq_evaluate_layer(*cand, *layer, &s) != 0) {
        return s;
    }
    // GA maximizes composite; fitness is lower-is-better -> negate.
    // Use the train error as the primary driver (matches ts_dispatch_awq_eval
    // which optimizes the proxy t_l^2; here we use the Python train_error).
    s.composite = -s.mse;
    return s;
}
