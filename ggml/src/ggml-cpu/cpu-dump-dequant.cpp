//
// cpu-dump-dequant.cpp
//
// CPU-side helper for the Tessera runtime-aware calibration pipeline
// (Layer 1, see docs/runtime-aware-pipeline.md). See cpu-dump-dequant.h
// for the public API and a description of the call pattern.
//

#include "cpu-dump-dequant.h"

#include "ggml.h"

#include "tessera-debug.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

// process-wide dedup: each tensor name is dumped at most once.
// the sidecar writer is not thread-safe, so a mutex guards the set.
static std::unordered_set<std::string> g_dumped;
static std::mutex                     g_dumped_mutex;

void cpu_dump_dequant(
        const struct ggml_tensor * src0,
        int64_t                   ne0,
        int64_t                   ne1,
        const char              * tensor_name) {
    if (!tessera_debug::dequant_debug_enabled()) {
        return;
    }
    if (src0 == nullptr || tensor_name == nullptr || tensor_name[0] == '\0') {
        return;
    }
    if (!ggml_is_quantized(src0->type)) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(g_dumped_mutex);
        if (!g_dumped.insert(std::string(tensor_name)).second) {
            return;
        }
    }

    const struct ggml_type_traits * traits = ggml_get_type_traits(src0->type);
    if (traits == nullptr || traits->to_float == nullptr) {
        return;
    }
    ggml_to_float_t const dequant = traits->to_float;

    if (ne0 <= 0 || ne1 <= 0) {
        return;
    }

    // ne0/ne1 are the leading-2D shape of the dequantized F32 weight.
    // For weights with broadcast dims (ne02, ne03 > 1) the underlying
    // quantization blocks are shared across the broadcast, so dumping
    // the first 2D slice is sufficient for round-trip validation.
    const int64_t rows = ne0;
    const int64_t cols = ne1;

    if (src0->ne[0] != cols || src0->ne[1] != rows) {
        return;
    }

    // scratch buffer for the dequantized F32 weight. malloc (not static)
    // because the hook is gated on the env var and rarely hit; the
    // memory cost on a calibration pass is small.
    const size_t scratch_bytes = (size_t) rows * (size_t) cols * sizeof(float);
    float * scratch = (float *) std::malloc(scratch_bytes);
    if (scratch == nullptr) {
        return;
    }

    // per-row wall-clock timing captured around the dequant call. Sized
    // to rows so we can populate the v3 per-row meta strip at sidecar
    // write time.
    std::vector<uint64_t> row_timing_ns((size_t) rows, 0);

    const int64_t ne00 = src0->ne[0];
    const int64_t nb01 = src0->nb[1];
    const char * src0_data = (const char *) src0->data;
    if (src0_data == nullptr) {
        std::free(scratch);
        return;
    }

    for (int64_t r = 0; r < rows; r++) {
        const void * src_row = (const void *) (src0_data + r * nb01);
        float * dst_row = scratch + r * cols;
        // Wall-clock around the dequant call. This is the per-row
        // timing consumed by the L6 kernel-direct fitness (latency
        // LUT) and the GA. The matmul itself is not measured here
        // (the hook fires before the matmul); the L6 orchestrator
        // combines the L1 dequant timing with the matmul cost from
        // its own instrumentation.
        const auto t0 = std::chrono::steady_clock::now();
        dequant(src_row, dst_row, ne00);
        const auto t1 = std::chrono::steady_clock::now();
        row_timing_ns[(size_t) r] =
            (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    }

    // kernel_id for the v3 per-row meta: a stable per-quantization-type
    // identifier. dispatch_count is 1 for the CPU reference dequant
    // (one call per row).
    const uint32_t kernel_id = (uint32_t) src0->type;

    tessera_debug::open_dequant_writer(tensor_name, rows, cols);
    for (int64_t r = 0; r < rows; r++) {
        tessera_debug::write_dequant_row(r, scratch + r * cols, cols);
        tessera_debug::set_dequant_row_meta(r, row_timing_ns[(size_t) r],
                                            kernel_id, /*dispatch_count=*/1);
    }
    tessera_debug::close_dequant_writer();
    // Per-row outlier counts (|x| > threshold) are now sealed in the
    // sidecar file's per-row strip. The L3 metric and the L5 IterQuant
    // orchestrator consume this strip via tools/tessera/l3_sidecar_v3_reader.py.
    //
    // In W4A4 mode (LLAMA_TILE640_DEBUG_DEQUANT_MODE=w4a4), the L1.5
    // FP16-reference sidecar (.act.dequant.f32) is also written. The
    // data is the same F32 values; a future refactor will pass the
    // original FP16 weight to the hook so the L1.5 reference captures
    // the actual ground-truth (vs. the dequantized quantized weight
    // captured by L1). For now the L1 and L1.5 files are bit-identical
    // in the data block, which makes the error metric
    // ||FP16(l) - Q_b(l)||_F^2 / ||FP16(l)||_F^2 collapse to zero
    // and the L5 / L6 consumers know to special-case it.

    std::free(scratch);
}
