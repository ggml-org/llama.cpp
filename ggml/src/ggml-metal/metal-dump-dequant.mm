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
    traits->to_float(src0->data, host_buf.data(), expected_els);

    // The dump is laid out as `rows` rows of `cols` F32 values. The
    // reference dequant is row-major, so row r starts at host_buf +
    // r*cols. (The caller chooses the row/col split; for a 2D weight
    // [K, N] that means rows=N, cols=K.)
    tessera_debug::open_dequant_writer(name, rows, cols);
    for (int64_t r = 0; r < rows; r++) {
        tessera_debug::write_dequant_row(r, host_buf.data() + r * cols, cols);
    }
    tessera_debug::close_dequant_writer();
}
