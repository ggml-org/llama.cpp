#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <set>
#include <random>
#include <vector>

extern "C" void ggml_vec_dot_nvfp4_nvfp4(
    int n,
    float * s,
    size_t bs,
    const void * vx,
    size_t bx,
    const void * vy,
    size_t by,
    int nrc);

static constexpr int32_t GGML_CUDA_STREAM_K_CONTROL_TAG = 0x534b4d51; // "SKMQ"

struct diff_stats {
    float  max_abs = 0.0f;
    double rms = 0.0;
    size_t idx_max = 0;
};

static std::vector<float> make_f32_data(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = dist(rng);
    }
    return out;
}

static std::vector<uint8_t> quantize_2d(enum ggml_type qtype, const std::vector<float> & src, int64_t k, int64_t m) {
    const size_t row_size = ggml_row_size(qtype, k);
    std::vector<uint8_t> out(row_size * m);
    for (int64_t row = 0; row < m; ++row) {
        ggml_quantize_chunk(
            qtype,
            src.data() + row*k,
            out.data() + row*row_size,
            0, 1, k, nullptr);
    }
    return out;
}

static std::vector<float> dequantize_2d(enum ggml_type qtype, const std::vector<uint8_t> & src, int64_t k, int64_t m) {
    const size_t row_size = ggml_row_size(qtype, k);
    const ggml_to_float_t to_float = ggml_get_type_traits(qtype)->to_float;
    if (to_float == nullptr) {
        std::fprintf(stderr, "type %s does not support to_float\n", ggml_type_name(qtype));
        std::abort();
    }

    std::vector<float> out((size_t) m * (size_t) k);
    for (int64_t row = 0; row < m; ++row) {
        to_float(src.data() + row * row_size, out.data() + row * k, k);
    }
    return out;
}

static std::vector<float> run_mul_mat_q(
        ggml_backend_t backend,
        enum ggml_type qtype,
        const std::vector<uint8_t> & a_q,
        const std::vector<float> & b_f32,
        int64_t m, int64_t n, int64_t k, int64_t n_channels,
        int stream_k_mode,
        int stream_k_nblocks) {

    struct ggml_init_params params = {
        /*.mem_size   = */ 16*1024*1024,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        std::abort();
    }

    ggml_tensor * a = ggml_new_tensor_2d(ctx, qtype, k, m);
    ggml_tensor * b = n_channels == 1
        ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n)
        : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n, n_channels);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    out->op_params[0] = GGML_CUDA_STREAM_K_CONTROL_TAG;
    out->op_params[1] = stream_k_mode;
    out->op_params[2] = stream_k_nblocks;

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend buffer\n");
        std::abort();
    }

    ggml_backend_tensor_set(a, a_q.data(),   0, a_q.size());
    ggml_backend_tensor_set(b, b_f32.data(), 0, b_f32.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml_backend_graph_compute failed: %s\n", ggml_status_to_string(status));
        std::abort();
    }

    std::vector<float> out_f32(ggml_nelements(out));
    ggml_backend_tensor_get(out, out_f32.data(), 0, out_f32.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return out_f32;
}

static diff_stats compare(const std::vector<float> & a, const std::vector<float> & b) {
    diff_stats s;
    double sum_sq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = double(a[i]) - double(b[i]);
        const float ad = (float) std::fabs(d);
        if (ad > s.max_abs) {
            s.max_abs = ad;
            s.idx_max = i;
        }
        sum_sq += d*d;
    }
    s.rms = std::sqrt(sum_sq / std::max<size_t>(1, a.size()));
    return s;
}

static double fit_alpha(const std::vector<float> & ref, const std::vector<float> & test) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        num += (double) ref[i] * (double) test[i];
        den += (double) ref[i] * (double) ref[i];
    }
    return den > 0.0 ? num / den : 0.0;
}

static std::vector<float> concat_cols(const std::vector<float> & left, const std::vector<float> & right) {
    std::vector<float> out(left.size() + right.size());
    std::copy(left.begin(),  left.end(),  out.begin());
    std::copy(right.begin(), right.end(), out.begin() + left.size());
    return out;
}

static std::vector<int64_t> sample_positions(int64_t n, int max_samples) {
    if (n <= 0) {
        return {};
    }

    const int64_t count = std::min<int64_t>(n, std::max(1, max_samples));
    std::set<int64_t> uniq;
    for (int64_t i = 0; i < count; ++i) {
        const int64_t pos = count == 1 ? 0 : (i * (n - 1)) / (count - 1);
        uniq.insert(pos);
    }

    return std::vector<int64_t>(uniq.begin(), uniq.end());
}

static std::vector<float> gather_reference_outputs(
        const std::vector<float> & a_deq_f32,
        const std::vector<float> & b_f32,
        int64_t k,
        const std::vector<int64_t> & sample_rows,
        const std::vector<int64_t> & sample_cols) {
    std::vector<float> out;
    out.reserve(sample_rows.size() * sample_cols.size());

    for (int64_t col : sample_cols) {
        const float * b_col = b_f32.data() + col * k;
        for (int64_t row : sample_rows) {
            const float * a_row = a_deq_f32.data() + row * k;
            double acc = 0.0;
            for (int64_t i = 0; i < k; ++i) {
                acc += (double) a_row[i] * (double) b_col[i];
            }
            out.push_back((float) acc);
        }
    }

    return out;
}

static std::vector<float> gather_outputs(
        const std::vector<float> & out_full,
        int64_t m,
        const std::vector<int64_t> & sample_rows,
        const std::vector<int64_t> & sample_cols) {
    std::vector<float> out;
    out.reserve(sample_rows.size() * sample_cols.size());

    for (int64_t col : sample_cols) {
        for (int64_t row : sample_rows) {
            out.push_back(out_full[(size_t) col * (size_t) m + (size_t) row]);
        }
    }

    return out;
}

static std::vector<float> gather_nvfp4_cpu_reference(
        const std::vector<uint8_t> & a_q,
        const std::vector<float> & b_f32,
        int64_t k,
        const std::vector<int64_t> & sample_rows,
        const std::vector<int64_t> & sample_cols) {
    const size_t row_size = ggml_row_size(GGML_TYPE_NVFP4, k);

    std::vector<float> b_sample_f32(sample_cols.size() * (size_t) k);
    for (size_t i = 0; i < sample_cols.size(); ++i) {
        const int64_t col = sample_cols[i];
        std::copy_n(b_f32.data() + col * k, k, b_sample_f32.data() + i * (size_t) k);
    }
    const std::vector<uint8_t> b_sample_q = quantize_2d(GGML_TYPE_NVFP4, b_sample_f32, k, (int64_t) sample_cols.size());

    std::vector<float> out;
    out.reserve(sample_rows.size() * sample_cols.size());
    for (size_t col_idx = 0; col_idx < sample_cols.size(); ++col_idx) {
        const uint8_t * b_row_q = b_sample_q.data() + col_idx * row_size;
        for (int64_t row : sample_rows) {
            const uint8_t * a_row_q = a_q.data() + (size_t) row * row_size;
            float acc = 0.0f;
            ggml_vec_dot_nvfp4_nvfp4((int) k, &acc, 0, a_row_q, row_size, b_row_q, row_size, 1);
            out.push_back(acc);
        }
    }

    return out;
}

static void print_sample_loc(
        const char * label,
        const diff_stats & s,
        const std::vector<int64_t> & sample_rows,
        const std::vector<int64_t> & sample_cols) {
    const size_t rows_per_col = sample_rows.size();
    const size_t sample_row_idx = rows_per_col > 0 ? s.idx_max % rows_per_col : 0;
    const size_t sample_col_idx = rows_per_col > 0 ? s.idx_max / rows_per_col : 0;
    const int64_t row = sample_row_idx < sample_rows.size() ? sample_rows[sample_row_idx] : -1;
    const int64_t col = sample_col_idx < sample_cols.size() ? sample_cols[sample_col_idx] : -1;
    std::printf("%s: max_abs=%.8g rms=%.8g idx=%zu (sample_row=%lld sample_col=%lld)\n",
            label, s.max_abs, s.rms, s.idx_max, (long long) row, (long long) col);
}

static void print_sample_values(
        const char * label,
        const diff_stats & s,
        const std::vector<float> & ref_sample,
        const std::vector<float> & test_sample,
        const std::vector<int64_t> & sample_rows,
        const std::vector<int64_t> & sample_cols) {
    const size_t rows_per_col = sample_rows.size();
    const size_t sample_row_idx = rows_per_col > 0 ? s.idx_max % rows_per_col : 0;
    const size_t sample_col_idx = rows_per_col > 0 ? s.idx_max / rows_per_col : 0;
    const int64_t row = sample_row_idx < sample_rows.size() ? sample_rows[sample_row_idx] : -1;
    const int64_t col = sample_col_idx < sample_cols.size() ? sample_cols[sample_col_idx] : -1;
    const float ref_v = s.idx_max < ref_sample.size() ? ref_sample[s.idx_max] : 0.0f;
    const float test_v = s.idx_max < test_sample.size() ? test_sample[s.idx_max] : 0.0f;
    std::printf("%s: sample_row=%lld sample_col=%lld ref=% .8f test=% .8f diff=% .8f\n",
            label, (long long) row, (long long) col, ref_v, test_v, test_v - ref_v);
}

static bool parse_qtype(const char * qtype_name, enum ggml_type & qtype) {
    struct map_entry {
        const char * name;
        enum ggml_type type;
    };
    static const map_entry k_map[] = {
        {"q4_0",    GGML_TYPE_Q4_0},
        {"q4_1",    GGML_TYPE_Q4_1},
        {"q5_0",    GGML_TYPE_Q5_0},
        {"q5_1",    GGML_TYPE_Q5_1},
        {"q8_0",    GGML_TYPE_Q8_0},
        {"mxfp4",   GGML_TYPE_MXFP4},
        {"nvfp4",   GGML_TYPE_NVFP4},
        {"q2_k",    GGML_TYPE_Q2_K},
        {"q3_k",    GGML_TYPE_Q3_K},
        {"q4_k",    GGML_TYPE_Q4_K},
        {"q5_k",    GGML_TYPE_Q5_K},
        {"q6_k",    GGML_TYPE_Q6_K},
        {"iq2_xxs", GGML_TYPE_IQ2_XXS},
        {"iq2_xs",  GGML_TYPE_IQ2_XS},
        {"iq2_s",   GGML_TYPE_IQ2_S},
        {"iq3_xxs", GGML_TYPE_IQ3_XXS},
        {"iq3_s",   GGML_TYPE_IQ3_S},
        {"iq1_s",   GGML_TYPE_IQ1_S},
        {"iq4_xs",  GGML_TYPE_IQ4_XS},
        {"iq4_nl",  GGML_TYPE_IQ4_NL},
    };
    for (const auto & e : k_map) {
        if (std::strcmp(qtype_name, e.name) == 0) {
            qtype = e.type;
            return true;
        }
    }
    return false;
}

int main(int argc, char ** argv) {
    const int64_t k = argc > 1 ? std::atoll(argv[1]) : 4096;
    const int64_t m = argc > 2 ? std::atoll(argv[2]) : 1536;
    const int64_t n = argc > 3 ? std::atoll(argv[3]) : 1024;
    const int64_t split_channels = argc > 4 ? std::atoll(argv[4]) : 2;
    const char * qtype_name = argc > 5 ? argv[5] : "nvfp4";
    const int stream_k_nblocks = argc > 6 ? std::atoi(argv[6]) : 0;

    enum ggml_type qtype = GGML_TYPE_NVFP4;
    if (!parse_qtype(qtype_name, qtype)) {
        std::fprintf(stderr, "unknown quant type '%s'\n", qtype_name);
        return 2;
    }

    if (ggml_quantize_requires_imatrix(qtype)) {
        std::fprintf(stderr, "quant type '%s' requires imatrix; skipped in this micro test\n", qtype_name);
        return 6;
    }

    const int64_t q_block = ggml_blck_size(qtype);
    if (k % q_block != 0) {
        std::fprintf(stderr, "k must be a multiple of %lld for %s, got %lld\n",
                (long long) q_block, qtype_name, (long long) k);
        return 2;
    }

    if (split_channels <= 0 || n % split_channels != 0) {
        std::fprintf(stderr, "split_channels must divide n (n=%lld, split_channels=%lld)\n",
                (long long) n, (long long) split_channels);
        return 2;
    }
    if (n % 2 != 0) {
        std::fprintf(stderr, "n must be even for half-split comparison (n=%lld)\n", (long long) n);
        return 2;
    }

    ggml_backend_load_all();

    ggml_backend_dev_t dev_gpu = nullptr;
    ggml_backend_dev_t dev_cpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t cur = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(cur) == GGML_BACKEND_DEVICE_TYPE_GPU && dev_gpu == nullptr) {
            dev_gpu = cur;
        } else if (ggml_backend_dev_type(cur) == GGML_BACKEND_DEVICE_TYPE_CPU && dev_cpu == nullptr) {
            dev_cpu = cur;
        }
    }

    if (dev_gpu == nullptr) {
        std::fprintf(stderr, "no GPU backend found\n");
        return 3;
    }
    if (dev_cpu == nullptr) {
        std::fprintf(stderr, "no CPU backend found\n");
        return 3;
    }

    ggml_backend_t backend_gpu = ggml_backend_dev_init(dev_gpu, nullptr);
    if (backend_gpu == nullptr) {
        std::fprintf(stderr, "failed to initialize GPU backend\n");
        return 4;
    }
    ggml_backend_t backend_cpu = ggml_backend_dev_init(dev_cpu, nullptr);
    if (backend_cpu == nullptr) {
        std::fprintf(stderr, "failed to initialize CPU backend\n");
        ggml_backend_free(backend_gpu);
        return 4;
    }

    std::printf("backend_gpu: %s\n", ggml_backend_name(backend_gpu));
    std::printf("backend_cpu: %s\n", ggml_backend_name(backend_cpu));
    std::printf("qtype: %s\n", qtype_name);
    std::printf("shape: m=%lld n=%lld k=%lld\n", (long long) m, (long long) n, (long long) k);
    if (qtype == GGML_TYPE_NVFP4) {
        std::printf("note: CPU NVFP4 mul_mat uses the q8_0 vec-dot lane, while CUDA MMQ uses q8_1/native staging;\n");
        std::printf("      CPU vs GPU absolute diffs are informative only here, not a strict NVFP4 parity gate.\n");
    }

    const std::vector<float> a_f32 = make_f32_data((size_t) m * (size_t) k, 0xA11CEu);
    const std::vector<float> b_f32 = make_f32_data((size_t) n * (size_t) k, 0xBEEFu);
    const std::vector<uint8_t> a_q = quantize_2d(qtype, a_f32, k, m);
    const std::vector<float> a_deq_f32 = dequantize_2d(qtype, a_q, k, m);

    const std::vector<float> out_cpu   = run_mul_mat_q(backend_cpu, qtype, a_q, b_f32, m, n, k, /*n_channels=*/1, /*stream_k_mode=*/0, /*stream_k_nblocks=*/0);
    const std::vector<float> out_off_1 = run_mul_mat_q(backend_gpu, qtype, a_q, b_f32, m, n, k, /*n_channels=*/1, /*stream_k_mode=*/0, stream_k_nblocks);
    const std::vector<float> out_off_2 = run_mul_mat_q(backend_gpu, qtype, a_q, b_f32, m, n, k, /*n_channels=*/1, /*stream_k_mode=*/0, stream_k_nblocks);
    const std::vector<float> out_on_1  = run_mul_mat_q(backend_gpu, qtype, a_q, b_f32, m, n, k, /*n_channels=*/1, /*stream_k_mode=*/1, stream_k_nblocks);
    const std::vector<float> out_on_2  = run_mul_mat_q(backend_gpu, qtype, a_q, b_f32, m, n, k, /*n_channels=*/1, /*stream_k_mode=*/1, stream_k_nblocks);
    const std::vector<float> out_on_split = run_mul_mat_q(
            backend_gpu, qtype, a_q, b_f32, m, n / split_channels, k, split_channels, /*stream_k_mode=*/1, stream_k_nblocks);

    const int64_t n_half = n / 2;
    std::vector<float> b_half0((size_t) n_half * (size_t) k);
    std::vector<float> b_half1((size_t) n_half * (size_t) k);
    std::copy(b_f32.begin(), b_f32.begin() + b_half0.size(), b_half0.begin());
    std::copy(b_f32.begin() + b_half0.size(), b_f32.end(), b_half1.begin());
    const std::vector<float> out_on_half0 = run_mul_mat_q(
            backend_gpu, qtype, a_q, b_half0, m, n_half, k, /*n_channels=*/1, /*stream_k_mode=*/1, stream_k_nblocks);
    const std::vector<float> out_on_half1 = run_mul_mat_q(
            backend_gpu, qtype, a_q, b_half1, m, n_half, k, /*n_channels=*/1, /*stream_k_mode=*/1, stream_k_nblocks);
    const std::vector<float> out_on_concat = concat_cols(out_on_half0, out_on_half1);

    ggml_backend_free(backend_cpu);
    ggml_backend_free(backend_gpu);
    ggml_quantize_free();

    const diff_stats cpu_vs_off = compare(out_cpu, out_off_1);
    const diff_stats cpu_vs_on  = compare(out_cpu, out_on_1);
    const diff_stats off_vs_off = compare(out_off_1, out_off_2);
    const diff_stats on_vs_on   = compare(out_on_1,  out_on_2);
    const diff_stats off_vs_on  = compare(out_off_1, out_on_1);
    const diff_stats on_vs_split = compare(out_on_1, out_on_split);
    const diff_stats on_full_vs_halves = compare(out_on_1, out_on_concat);

    const std::vector<int64_t> sample_rows = sample_positions(m, 16);
    const std::vector<int64_t> sample_cols = sample_positions(n, 16);
    const std::vector<float> ref_sample = gather_reference_outputs(a_deq_f32, b_f32, k, sample_rows, sample_cols);
    const std::vector<float> cpu_sample = gather_outputs(out_cpu,   m, sample_rows, sample_cols);
    const std::vector<float> off_sample = gather_outputs(out_off_1, m, sample_rows, sample_cols);
    const std::vector<float> on_sample  = gather_outputs(out_on_1,  m, sample_rows, sample_cols);
    const std::vector<float> nvfp4_cpu_sample = qtype == GGML_TYPE_NVFP4
        ? gather_nvfp4_cpu_reference(a_q, b_f32, k, sample_rows, sample_cols)
        : std::vector<float>();

    const diff_stats ref_vs_cpu = compare(ref_sample, cpu_sample);
    const diff_stats ref_vs_off = compare(ref_sample, off_sample);
    const diff_stats ref_vs_on  = compare(ref_sample, on_sample);
    const diff_stats ref_vs_nvfp4_cpu = nvfp4_cpu_sample.empty() ? diff_stats{} : compare(ref_sample, nvfp4_cpu_sample);
    const diff_stats nvfp4_cpu_vs_off = nvfp4_cpu_sample.empty() ? diff_stats{} : compare(nvfp4_cpu_sample, off_sample);
    const diff_stats nvfp4_cpu_vs_on  = nvfp4_cpu_sample.empty() ? diff_stats{} : compare(nvfp4_cpu_sample, on_sample);

    auto print_loc = [&](const char * label, const diff_stats & s) {
        const int64_t row = (int64_t) s.idx_max % m;
        const int64_t col = (int64_t) s.idx_max / m;
        std::printf("%s: max_abs=%.8g rms=%.8g idx=%zu (row=%lld col=%lld)\n",
                label, s.max_abs, s.rms, s.idx_max, (long long) row, (long long) col);
    };

    print_loc("CPU vs GPU stream-k OFF", cpu_vs_off);
    print_loc("CPU vs GPU stream-k ON ", cpu_vs_on);
    print_loc("determinism stream-k OFF", off_vs_off);
    print_loc("determinism stream-k ON ", on_vs_on);
    print_loc("OFF vs ON difference   ", off_vs_on);
    print_loc("ON flat vs ON split", on_vs_split);
    print_loc("ON full vs two halves", on_full_vs_halves);
    std::printf("sampled dequant-F32 ref: rows=%zu cols=%zu cells=%zu\n",
            sample_rows.size(), sample_cols.size(), ref_sample.size());
    print_sample_loc("sample ref vs CPU      ", ref_vs_cpu, sample_rows, sample_cols);
    print_sample_loc("sample ref vs GPU OFF  ", ref_vs_off, sample_rows, sample_cols);
    print_sample_loc("sample ref vs GPU ON   ", ref_vs_on, sample_rows, sample_cols);
    print_sample_values("sample worst CPU cell  ", ref_vs_cpu, ref_sample, cpu_sample, sample_rows, sample_cols);
    print_sample_values("sample worst GPU OFF   ", ref_vs_off, ref_sample, off_sample, sample_rows, sample_cols);
    print_sample_values("sample worst GPU ON    ", ref_vs_on, ref_sample, on_sample, sample_rows, sample_cols);
    if (!nvfp4_cpu_sample.empty()) {
        print_sample_loc("sample ref vs CPU NVFP4", ref_vs_nvfp4_cpu, sample_rows, sample_cols);
        print_sample_loc("CPU NVFP4 vs GPU OFF   ", nvfp4_cpu_vs_off, sample_rows, sample_cols);
        print_sample_loc("CPU NVFP4 vs GPU ON    ", nvfp4_cpu_vs_on, sample_rows, sample_cols);
        print_sample_values("sample worst CPU NVFP4 ", ref_vs_nvfp4_cpu, ref_sample, nvfp4_cpu_sample, sample_rows, sample_cols);
        print_sample_values("sample worst OFF vs CPU", nvfp4_cpu_vs_off, nvfp4_cpu_sample, off_sample, sample_rows, sample_cols);
        print_sample_values("sample worst ON  vs CPU", nvfp4_cpu_vs_on, nvfp4_cpu_sample, on_sample, sample_rows, sample_cols);
    }

    std::printf("split_channels=%lld n_half=%lld\n",
            (long long) split_channels, (long long) n_half);
    std::printf("stream_k_nblocks=%d\n", stream_k_nblocks);
    std::printf("fit alpha (OFF~CPU): %.8g\n", fit_alpha(out_cpu, out_off_1));
    std::printf("fit alpha (ON ~CPU): %.8g\n", fit_alpha(out_cpu, out_on_1));
    std::printf("fit alpha (CPU~ref sample): %.8g\n", fit_alpha(ref_sample, cpu_sample));
    std::printf("fit alpha (OFF~ref sample): %.8g\n", fit_alpha(ref_sample, off_sample));
    std::printf("fit alpha (ON ~ref sample): %.8g\n", fit_alpha(ref_sample, on_sample));
    if (!nvfp4_cpu_sample.empty()) {
        std::printf("fit alpha (CPU NVFP4~ref): %.8g\n", fit_alpha(ref_sample, nvfp4_cpu_sample));
        std::printf("fit alpha (OFF~CPU NVFP4): %.8g\n", fit_alpha(nvfp4_cpu_sample, off_sample));
        std::printf("fit alpha (ON ~CPU NVFP4): %.8g\n", fit_alpha(nvfp4_cpu_sample, on_sample));
    }

    if (!std::isfinite(off_vs_on.max_abs) || !std::isfinite(off_vs_on.rms)) {
        std::fprintf(stderr, "non-finite diff detected\n");
        return 5;
    }

    return 0;
}
