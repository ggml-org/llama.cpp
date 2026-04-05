#include "ggml.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {

struct stats {
    double mse      = 0.0;
    double rmse     = 0.0;
    double mae      = 0.0;
    double max_abs  = 0.0;
    double mean_abs = 0.0;
};

struct options {
    int64_t rows      = 128;
    int64_t cols      = 4096;
    int64_t blocksize = 64;
    uint32_t seed     = 7;
    std::string input_path;
};

constexpr float BNB_NF4_TABLE[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f,
};

constexpr float BNB_FP4_TABLE[16] = {
    0.0000f,
    0.0052f,
    0.6667f,
    1.0000f,
    0.3333f,
    0.5000f,
    0.1667f,
    0.2500f,
    0.0000f,
    -0.0052f,
    -0.6667f,
    -1.0000f,
    -0.3333f,
    -0.5000f,
    -0.1667f,
    -0.2500f,
};

void usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " [--rows N] [--cols N] [--blocksize N] [--seed N] [--input file.bin]\n"
        << "  --input expects raw little-endian float32 values. Rows are inferred when omitted.\n";
}

bool parse_int64(const char * value, int64_t & out) {
    char * end = nullptr;
    const auto parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0') {
        return false;
    }
    out = parsed;
    return true;
}

bool parse_u32(const char * value, uint32_t & out) {
    char * end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    out = static_cast<uint32_t>(parsed);
    return true;
}

bool parse_args(int argc, char ** argv, options & opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--rows") {
            const char * value = require_value("--rows");
            if (!value || !parse_int64(value, opts.rows)) {
                return false;
            }
        } else if (arg == "--cols") {
            const char * value = require_value("--cols");
            if (!value || !parse_int64(value, opts.cols)) {
                return false;
            }
        } else if (arg == "--blocksize") {
            const char * value = require_value("--blocksize");
            if (!value || !parse_int64(value, opts.blocksize)) {
                return false;
            }
        } else if (arg == "--seed") {
            const char * value = require_value("--seed");
            if (!value || !parse_u32(value, opts.seed)) {
                return false;
            }
        } else if (arg == "--input") {
            const char * value = require_value("--input");
            if (!value) {
                return false;
            }
            opts.input_path = value;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return false;
        }
    }

    if (opts.cols <= 0 || opts.rows <= 0 || opts.blocksize <= 0) {
        std::cerr << "rows, cols, and blocksize must be positive\n";
        return false;
    }

    return true;
}

std::vector<float> read_input(const options & opts, int64_t & rows, int64_t & cols) {
    std::ifstream in(opts.input_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open input file: " + opts.input_path);
    }

    in.seekg(0, std::ios::end);
    const std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size <= 0 || size % static_cast<std::streamsize>(sizeof(float)) != 0) {
        throw std::runtime_error("input size is not a positive multiple of float32");
    }

    std::vector<float> data(size / static_cast<std::streamsize>(sizeof(float)));
    if (!in.read(reinterpret_cast<char *>(data.data()), size)) {
        throw std::runtime_error("failed to read input file");
    }

    cols = opts.cols;
    if (data.size() % static_cast<size_t>(cols) != 0) {
        throw std::runtime_error("input element count is not divisible by --cols");
    }
    rows = static_cast<int64_t>(data.size() / static_cast<size_t>(cols));
    return data;
}

std::vector<float> make_random_data(int64_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.75f);
    std::vector<float> data(count);
    for (float & v : data) {
        v = dist(rng);
    }
    return data;
}

stats compute_stats(const std::vector<float> & ref, const std::vector<float> & cand) {
    if (ref.size() != cand.size()) {
        throw std::runtime_error("stat input size mismatch");
    }

    stats s;
    double sum_sq = 0.0;
    double sum_abs = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double err = static_cast<double>(cand[i]) - static_cast<double>(ref[i]);
        const double abs_err = std::abs(err);
        sum_sq += err * err;
        sum_abs += abs_err;
        s.max_abs = std::max(s.max_abs, abs_err);
    }

    const double n = static_cast<double>(ref.size());
    s.mse = sum_sq / n;
    s.rmse = std::sqrt(s.mse);
    s.mae = sum_abs / n;
    s.mean_abs = s.mae;
    return s;
}

uint8_t nearest_code(float value, const float * table, int n) {
    uint8_t best = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < n; ++i) {
        const float dist = std::abs(value - table[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best = static_cast<uint8_t>(i);
        }
    }
    return best;
}

std::vector<float> quantize_dequantize_bnb(const std::vector<float> & src, int64_t blocksize, const float * table) {
    std::vector<float> dst(src.size(), 0.0f);
    const int64_t n = static_cast<int64_t>(src.size());
    for (int64_t start = 0; start < n; start += blocksize) {
        const int64_t end = std::min(start + blocksize, n);
        float absmax = 0.0f;
        for (int64_t i = start; i < end; ++i) {
            absmax = std::max(absmax, std::abs(src[i]));
        }
        if (absmax == 0.0f) {
            continue;
        }
        for (int64_t i = start; i < end; ++i) {
            const float normalized = std::clamp(src[i] / absmax, -1.0f, 1.0f);
            const auto q = nearest_code(normalized, table, 16);
            dst[i] = table[q] * absmax;
        }
    }
    return dst;
}

template<typename Block>
std::vector<float> quantize_dequantize_ggml(
        const std::vector<float> & src,
        int64_t cols,
        void (*quantize)(const float *, Block *, int64_t),
        void (*dequantize)(const Block *, float *, int64_t)) {
    const int64_t rows = static_cast<int64_t>(src.size()) / cols;
    const size_t row_bytes = ggml_row_size(std::is_same<Block, block_iq4_nl>::value ? GGML_TYPE_IQ4_NL : GGML_TYPE_MXFP4, cols);
    std::vector<uint8_t> qbuf(row_bytes * rows);
    std::vector<float> out(src.size());

    for (int64_t row = 0; row < rows; ++row) {
        const float * in = src.data() + row * cols;
        auto * qrow = reinterpret_cast<Block *>(qbuf.data() + row * row_bytes);
        float * out_row = out.data() + row * cols;
        quantize(in, qrow, cols);
        dequantize(qrow, out_row, cols);
    }

    return out;
}

void print_stats(const char * label, const stats & s) {
    std::cout << std::left << std::setw(16) << label
              << " mse=" << std::setw(12) << s.mse
              << " rmse=" << std::setw(12) << s.rmse
              << " mae=" << std::setw(12) << s.mae
              << " max_abs=" << s.max_abs << "\n";
}

} // namespace

int main(int argc, char ** argv) {
    options opts;
    if (!parse_args(argc, argv, opts)) {
        usage(argv[0]);
        return 1;
    }

    int64_t rows = opts.rows;
    int64_t cols = opts.cols;

    if (cols % QK4_NL != 0 || cols % QK_MXFP4 != 0) {
        std::cerr << "cols must be divisible by both " << QK4_NL << " (IQ4_NL) and " << QK_MXFP4 << " (MXFP4)\n";
        return 1;
    }

    std::vector<float> input;
    try {
        input = opts.input_path.empty()
            ? make_random_data(rows * cols, opts.seed)
            : read_input(opts, rows, cols);
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    const auto bnb_nf4 = quantize_dequantize_bnb(input, opts.blocksize, BNB_NF4_TABLE);
    const auto bnb_fp4 = quantize_dequantize_bnb(input, opts.blocksize, BNB_FP4_TABLE);
    const auto ggml_iq4_nl = quantize_dequantize_ggml<block_iq4_nl>(input, cols, quantize_row_iq4_nl_ref, dequantize_row_iq4_nl);
    const auto ggml_mxfp4  = quantize_dequantize_ggml<block_mxfp4>(input, cols, quantize_row_mxfp4_ref, dequantize_row_mxfp4);

    std::cout << "llama-bnb-compat\n";
    std::cout << "rows=" << rows << " cols=" << cols << " blocksize=" << opts.blocksize;
    if (!opts.input_path.empty()) {
        std::cout << " input=" << opts.input_path;
    } else {
        std::cout << " seed=" << opts.seed;
    }
    std::cout << "\n";
    std::cout << "Comparison targets: bitsandbytes NF4/FP4 vs ggml IQ4_NL/MXFP4\n\n";

    print_stats("bnb_nf4", compute_stats(input, bnb_nf4));
    print_stats("ggml_iq4_nl", compute_stats(input, ggml_iq4_nl));
    print_stats("bnb_fp4", compute_stats(input, bnb_fp4));
    print_stats("ggml_mxfp4", compute_stats(input, ggml_mxfp4));

    return 0;
}
