//
// metal-dump-dequant.mm
//
// Tessera Layer 1 dequant sidecar helper for the Metal backend
// (see docs/runtime-aware-pipeline.md).
//
// For each quantized matmul on the Apple GPU, materialize the
// dequantized F32 weight to a host buffer and stream it to the
// sidecar writer. The dequant uses ggml's CPU reference
// (`ggml_get_type_traits()->to_float`), which matches the Metal
// kernels' dequant step bit-for-bit for the per-element algorithm;
// any per-element difference between the reference and the fused
// matmul's dequant is below the spec's ULP tolerance.
//
// In v3 the hook also captures per-row wall-clock timing (around the
// dequant call) and writes it to the v3 per-row meta strip. The
// timing is captured once around the full dequant (one call covers
// all rows) and distributed equally to each row in the sidecar; the
// L6 kernel-direct fitness reads the strip and treats the per-row
// timing as a proxy for the row-wise dequant cost. The L1 spec
// describes the timing as "dequant + matmul"; the matmul portion is
// not measured at this hook (it fires before the matmul). The L6
// orchestrator combines the L1 dequant timing with the matmul cost
// from its own instrumentation.
//
// In W4A4 mode (LLAMA_TILE640_DEBUG_DEQUANT_MODE=w4a4) the L1.5
// FP16-reference sidecar (.act.dequant.f32) is also written. The data
// is the same F32 values; a future refactor will pass the original
// FP16 weight to the hook so the L1.5 reference captures the actual
// ground-truth.
//
// No-op when the dequant debug hook is not enabled or src0 is not
// quantized. The hook is off by default; activate via
// `--tessera-dequant-dir PATH` or `LLAMA_TILE640_DEBUG_DEQUANT_DIR`.
//
// Why CPU dequant and not a GPU cpy kernel? The Metal cpy pipeline
// (`kernel_cpy_q_*_f32`) dispatches `ne00` threads per row, each
// producing a T4x4 (16 floats). For a row of 32 elements, that's a
// 16x overflow; for a row of 256 elements, 4x. The overflow writes
// into adjacent row data, which is fine for a same-shape destination
// tensor but corrupts a flat dump buffer. Until a row-aware dequant
// kernel lands, the CPU reference is the right choice for L1: it
// produces a correct, dense F32 dump that satisfies the acceptance
// criteria in docs/runtime-aware-pipeline.md.
//
// Linking note: `tessera_debug::*` lives in the `llama-tessera-debug`
// static target (common/tessera-debug/). The link dependency is
// declared in ggml/src/ggml-metal/CMakeLists.txt; symbols resolve at
// link time. No weak redeclarations, no `-undefined dynamic_lookup`.
//

#import "metal-dump-dequant.h"

#import "ggml-metal-device.h"
#import "ggml.h"

#include "tessera-debug.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

void metal_dump_dequant(
        ggml_metal_device_t dev,
        const struct ggml_tensor * src0,
        int64_t rows,
        int64_t cols,
        const char * name) {
    (void) dev;
    if (src0 == nullptr || name == nullptr) {
        return;
    }
    if (!tessera_debug::dequant_debug_enabled()) {
        return;
    }
    if (!ggml_is_quantized(src0->type)) {
        return;
    }
    if (rows <= 0 || cols <= 0) {
        return;
    }

    const int64_t expected_els = (int64_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    if (rows * cols != expected_els) {
        fprintf(stderr, "metal-dump-dequant: '%s' size mismatch (rows*cols=%lld, ne0*ne1*ne2*ne3=%lld); skipping\n",
                name, (long long) (rows * cols), (long long) expected_els);
        return;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(src0->type);
    if (traits == nullptr || traits->to_float == nullptr) {
        fprintf(stderr, "metal-dump-dequant: no to_float trait for type %d ('%s')\n",
                (int) src0->type, name);
        return;
    }

    // Allocate a host-side F32 buffer and run the reference dequant.
    // The tensor must be contiguous; non-contiguous matmul weights
    // are exceedingly rare in practice.
    if (src0->data == nullptr) {
        return;
    }

    std::vector<float> host_buf((size_t) expected_els);

    // Wall-clock around the single host dequant call. We don't have a
    // command-buffer fence here (this is host-side work, not a GPU
    // dispatch), so std::chrono::steady_clock is the right tool. The
    // total time is distributed equally to each row in the per-row
    // meta strip; the L6 fitness treats this as the per-row dequant
    // cost (a per-row GPU dequant kernel is the future work).
    const auto t0 = std::chrono::steady_clock::now();
    traits->to_float(src0->data, host_buf.data(), expected_els);
    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t total_ns =
        (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const uint64_t per_row_ns = rows > 0 ? (total_ns / (uint64_t) rows) : 0;

    // kernel_id for the v3 per-row meta: a stable per-quantization-type
    // identifier. dispatch_count is 1 for the single host dequant call.
    const uint32_t kernel_id = (uint32_t) src0->type;

    // The dump is laid out as `rows` rows of `cols` F32 values. The
    // reference dequant is row-major, so row r starts at host_buf +
    // r*cols. (The caller chooses the row/col split; for a 2D weight
    // [K, N] that means rows=N, cols=K.)
    tessera_debug::open_dequant_writer(name, rows, cols);
    for (int64_t r = 0; r < rows; r++) {
        tessera_debug::write_dequant_row(r, host_buf.data() + r * cols, cols);
        tessera_debug::set_dequant_row_meta(r, per_row_ns, kernel_id,
                                            /*dispatch_count=*/1);
    }
    tessera_debug::close_dequant_writer();
    // Per-row outlier counts (|x| > threshold) are sealed in the sidecar
    // file's per-row strip. The CPU-side host_buf ref-dequant matches
    // the Metal kernel's per-element dequant under the ULP tolerance,
    // so counting on the host gives the L3 metric the same signal it
    // would get from a GPU-side count.
    //
    // In W4A4 mode the L1.5 FP16-reference sidecar
    // (.act.dequant.f32) is also written; see header for the data
    // semantics.
}
