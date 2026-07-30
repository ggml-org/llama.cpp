//
// l3_sidecar_v3_helper.cc
//
// Small C++ test binary that writes synthetic L1 and L1.5 dequant
// sidecars via the `tessera_debug` C++ API. Used by the e2e smoke
// test (tools/tessera/l3_sidecar_v3_smoke.py --from-cpp) to verify
// the full pipeline: writer -> on-disk file -> Python reader.
//
// The data is a deterministic 4x8 F32 tensor (rows=ne0, cols=ne1 in
// the L1 fixes convention). The values are chosen to produce a
// non-trivial per-row outlier count against the default 6.0
// threshold. The per-row timing is captured around each row's
// dequant call (the C++ helper just dequantizes from a "fake"
// quantized source: the F32 values themselves, so the "dequant" is
// a memcpy). The L1.5 reference sidecar is auto-written in W4A4
// mode with the same F32 data; this is the current contract until
// a future refactor passes the original FP16 weight to the hook.
//
// Usage:
//
//   LLAMA_TILE640_DEBUG_DEQUANT_DIR=<dir> \
//   LLAMA_TILE640_DEBUG_DEQUANT_MODE=w4a4 \
//   TESSERA_TELEMETRY_MODEL=gemma-3-12b \
//   TESSERA_TELEMETRY_CALIBRATION_CORPUS="prompts/paris.txt + prompts/wikitext-30" \
//   TESSERA_TELEMETRY_CALIBRATION_CORPUS_HASH=sha256:0123abcd \
//   l3_sidecar_v3_helper <tensor_name> <rows> <cols> [w4a4]
//
// Outputs:
//   <dir>/<tensor_name>.dequant.f32         (always, when DEQUANT_DIR is set)
//   <dir>/<tensor_name>.dequant.f32.provenance.json
//   <dir>/<tensor_name>.act.dequant.f32     (only in w4a4 mode)
//   <dir>/<tensor_name>.act.dequant.f32.provenance.json
//

#include "tessera-debug.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void make_synthetic_data(std::vector<float> & data, int64_t rows, int64_t cols, int seed) {
    // Deterministic values. Plant a few values > 6.0 in each row so
    // the per-row outlier count is non-trivial.
    data.assign((size_t) (rows * cols), 0.0f);
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t c = 0; c < cols; c++) {
            float v = (float) ((r * 7 + c * 3 + seed) % 11);
            if ((r + c) % 5 == 0) {
                v = 8.0f + (float) ((r + c) % 3);
            }
            data[(size_t) (r * cols + c)] = v;
        }
    }
}

// A "fake dequant" that copies the F32 row. The timing is captured
// around this call. In a real backend hook this is the dequant
// kernel; here it's a memcpy so the sidecar can be exercised
// without a real quantized tensor.
static void fake_dequant(const void * src, float * dst, int64_t n) {
    std::memcpy(dst, src, (size_t) n * sizeof(float));
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <tensor_name> <rows> <cols> [w4a4]\n", argv[0]);
        return 2;
    }
    const char * tensor_name = argv[1];
    int64_t rows = std::strtoll(argv[2], nullptr, 10);
    int64_t cols = std::strtoll(argv[3], nullptr, 10);
    bool w4a4 = (argc >= 5) && (std::strcmp(argv[4], "w4a4") == 0);

    if (rows <= 0 || cols <= 0) {
        std::fprintf(stderr, "rows and cols must be positive\n");
        return 2;
    }

    // Optional telemetry env vars. They are also read by the
    // foundation (tessera_debug.cpp) on first call, so we don't
    // need to set them again here -- but the env var path is
    // exercised by the e2e harness via subprocess.

    if (w4a4) {
        tessera_debug::set_dequant_mode("w4a4");
    } else {
        tessera_debug::set_dequant_mode("");
    }

    if (!tessera_debug::dequant_debug_enabled()) {
        std::fprintf(stderr,
            "tessera_debug not enabled: set LLAMA_TILE640_DEBUG_DEQUANT_DIR "
            "or pass --tessera-dequant-dir\n");
        return 3;
    }

    std::vector<float> data;
    make_synthetic_data(data, rows, cols, /*seed=*/0);

    // Stage 1: dequant "fakes" (memcpy each row). We capture the
    // wall-clock around each call. In a real hook the dequant is
    // the per-row dequant kernel.
    std::vector<uint64_t> row_timing_ns((size_t) rows, 0);
    for (int64_t r = 0; r < rows; r++) {
        const void * src_row = data.data() + r * cols;
        std::vector<float> dst_row((size_t) cols, 0.0f);
        const auto t0 = std::chrono::steady_clock::now();
        fake_dequant(src_row, dst_row.data(), cols);
        const auto t1 = std::chrono::steady_clock::now();
        row_timing_ns[(size_t) r] = (uint64_t)
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    }

    // Stage 2: write the L1 dequant sidecar.
    tessera_debug::open_dequant_writer(tensor_name, rows, cols);
    for (int64_t r = 0; r < rows; r++) {
        tessera_debug::write_dequant_row(r, data.data() + r * cols, cols);
        // kernel_id: 0xDEAD (placeholder for "CPU fake dequant").
        tessera_debug::set_dequant_row_meta(r, row_timing_ns[(size_t) r],
                                            /*kernel_id=*/0xDEAD,
                                            /*dispatch_count=*/1);
    }
    tessera_debug::close_dequant_writer();

    // Stage 3: in W4A4 mode, also open and close the L1.5 writer.
    // The writer auto-populates the L1.5 file from the same F32 data
    // when mode is w4a4. We also explicitly call the L1.5 API to
    // demonstrate it (and to exercise the timing path for the
    // reference). In the current implementation the L1.5 data is
    // identical to the L1 data; a future refactor will pass the
    // original FP16 weight to the hook so the L1.5 captures the
    // actual ground-truth.
    if (w4a4) {
        if (tessera_debug::dequant_w4a4_enabled()) {
            tessera_debug::open_fp16_reference_writer(tensor_name, rows, cols);
            for (int64_t r = 0; r < rows; r++) {
                tessera_debug::write_fp16_reference_row(r, data.data() + r * cols, cols);
                tessera_debug::set_fp16_reference_row_meta(r, row_timing_ns[(size_t) r],
                                                           /*kernel_id=*/0xDEAD,
                                                           /*dispatch_count=*/1);
            }
            tessera_debug::close_fp16_reference_writer();
        } else {
            std::fprintf(stderr, "expected w4a4 mode but dequant_w4a4_enabled is false\n");
            return 4;
        }
    }

    std::fprintf(stderr, "wrote sidecar(s) for tensor '%s' rows=%lld cols=%lld w4a4=%d\n",
                 tensor_name, (long long) rows, (long long) cols, w4a4 ? 1 : 0);
    return 0;
}
