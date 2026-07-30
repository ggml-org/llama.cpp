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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>

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
        dequant(src_row, dst_row, ne00);
    }

    tessera_debug::open_dequant_writer(tensor_name, rows, cols);
    for (int64_t r = 0; r < rows; r++) {
        tessera_debug::write_dequant_row(r, scratch + r * cols, cols);
    }
    tessera_debug::close_dequant_writer();

    std::free(scratch);
}
