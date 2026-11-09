// quantize a raw f32 tensor to a ggml quant type
// usage: quantize-raw <dst_type> <nrows> <n_per_row> [<tensor_name> <imatrix_file>]
// reads nrows * n_per_row f32 values from stdin, writes the quantized bytes to stdout
//
// used by convert_hf_to_gguf.py to quantize K/I-quants directly from safetensors,
// avoiding an FP16 GGUF intermediate. gguf-py can only dequantize these types, not produce them.
//
// if <imatrix_file> is given (non-empty), the importance matrix for <tensor_name> is loaded
// and used. ggml types that require an imatrix (low-bit I-quants) fail if none is available.
#include "ggml.h"
#include "imatrix-loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

// load and normalize the imatrix for one tensor, mirroring tools/quantize/quantize.cpp
static bool load_imatrix_for_tensor(const std::string & fname, const std::string & tensor_name, std::vector<float> & out) {
    common_imatrix loaded;
    if (!common_imatrix_load(fname, loaded)) {
        return false;
    }
    auto it = loaded.entries.find(tensor_name);
    if (it == loaded.entries.end()) {
        return false;
    }
    const auto & entry = it->second;
    out.resize(entry.sums.size());
    if (!loaded.is_legacy) {
        // GGUF format: normalize by per-expert counts
        const int64_t ncounts = entry.counts.size();
        const int64_t ne0     = (int64_t) entry.sums.size() / ncounts;
        for (int64_t j = 0; j < ncounts; ++j) {
            const float count = (float) entry.counts[j];
            for (int64_t i = 0; i < ne0; ++i) {
                out[j*ne0 + i] = count > 0.0f ? entry.sums[j*ne0 + i] / count : 1.0f;
            }
        }
    } else {
        // legacy format: sums contain (raw/count)*ncall, divide by ncall
        const int64_t ncall = entry.counts.empty() ? 0 : entry.counts[0];
        for (size_t i = 0; i < entry.sums.size(); ++i) {
            out[i] = ncall > 0 ? entry.sums[i] / (float) ncall : entry.sums[i];
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 6) {
        fprintf(stderr, "usage: %s <dst_type> <nrows> <n_per_row> [<tensor_name> <imatrix_file>]\n", argv[0]);
        return 1;
    }

    const ggml_type type      = (ggml_type) atoi(argv[1]);
    const int64_t   nrows     = atoll(argv[2]);
    const int64_t   n_per_row = atoll(argv[3]);
    const std::string tensor_name  = (argc == 6) ? argv[4] : "";
    const std::string imatrix_file = (argc == 6) ? argv[5] : "";

    if (nrows <= 0 || n_per_row <= 0) {
        fprintf(stderr, "%s: invalid dimensions\n", argv[0]);
        return 1;
    }

    const size_t n_elem  = (size_t) nrows * (size_t) n_per_row;
    const size_t n_bytes = n_elem * sizeof(float);

    std::vector<float> src(n_elem);
    size_t total = 0;
    while (total < n_bytes) {
        const size_t got = fread(src.data() + total / sizeof(float), 1, n_bytes - total, stdin);
        if (got == 0) {
            fprintf(stderr, "%s: unexpected end of input (got %zu of %zu bytes)\n", argv[0], total, n_bytes);
            return 1;
        }
        total += got;
    }

    // load the imatrix for this tensor: required for low-bit I-quants, optional otherwise.
    // the imatrix has one value per column (n_per_row), reused across all rows - matching
    // ggml's quantize kernels, which apply the same n_per_row weights to every row.
    std::vector<float> imatrix;
    const float * imatrix_ptr = nullptr;
    const bool needs_imatrix = ggml_quantize_requires_imatrix(type);
    if (!imatrix_file.empty() && load_imatrix_for_tensor(imatrix_file, tensor_name, imatrix)) {
        if (imatrix.size() != (size_t) n_per_row) {
            fprintf(stderr, "%s: imatrix size %zu != n_per_row %lld for '%s'\n",
                    argv[0], imatrix.size(), (long long) n_per_row, tensor_name.c_str());
            return 1;
        }
        imatrix_ptr = imatrix.data();
    }

    // low-bit I-quants cannot be quantized without an importance matrix
    if (needs_imatrix && imatrix_ptr == nullptr) {
        fprintf(stderr, "%s: type %s requires an importance matrix; none available for '%s' "
                        "(pass --imatrix to convert_hf_to_gguf.py)\n",
                argv[0], ggml_type_name(type), tensor_name.c_str());
        return 1;
    }

    ggml_quantize_init(type);

    const size_t row_size = ggml_row_size(type, n_per_row);
    std::vector<uint8_t> dst((size_t) nrows * row_size);
    ggml_quantize_chunk(type, src.data(), dst.data(), 0, nrows, n_per_row, imatrix_ptr);

    if (fwrite(dst.data(), 1, dst.size(), stdout) != dst.size()) {
        fprintf(stderr, "%s: failed to write output\n", argv[0]);
        ggml_quantize_free();
        return 1;
    }

    ggml_quantize_free();
    return 0;
}
