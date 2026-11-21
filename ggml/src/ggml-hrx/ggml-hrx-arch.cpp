#include "ggml-hrx-arch.h"

#include <cstring>

// The limits follow from the kernel layouts.
static constexpr bool ggml_hrx_tune_valid(const ggml_hrx_tune & t) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const ggml_hrx_mul_mat_vec_tune & mv = t.mul_mat_vec[i];
        if (mv.max_cols < 1 || mv.max_cols > 8 || mv.xs_max_k < 256 || mv.xs_max_k > 4096 || mv.xs_wg_per_cu < 1 ||
            mv.xs_depth < 1 || mv.xs_depth > 4) {
            return false;
        }
    }
    const bool wg_size = t.elementwise_wg_size >= 64 && t.elementwise_wg_size <= 1024 && t.elementwise_wg_size % 64 == 0;
    const bool stream  = t.stream_wg_per_cu >= 0 && t.stream_wg_per_cu <= 16;
    return wg_size && stream;
}

// Tuned on gfx1151.
static constexpr ggml_hrx_tune ggml_hrx_tune_rdna3_5 = [] {
    ggml_hrx_tune t = {};
    t.nodes_per_submit   = 64;
    t.nodes_first_submit = 8;
    t.mapped_readback    = false;
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        t.mul_mat_vec[i] = {
            /* .max_cols     = */ 8,
            /* .xs_max_k     = */ 4096,
            /* .xs_wg_per_cu = */ 4,
            /* .xs_depth     = */ 2,
        };
    }
    t.elementwise_wg_size = 256;
    t.stream_wg_per_cu    = 0;
    return t;
}();

// Untuned. These families use the RDNA3.5 values.
static constexpr ggml_hrx_tune ggml_hrx_tune_rdna3 = ggml_hrx_tune_rdna3_5;
static constexpr ggml_hrx_tune ggml_hrx_tune_rdna4 = ggml_hrx_tune_rdna3_5;

// Tuned on gfx1201.
static constexpr ggml_hrx_tune ggml_hrx_tune_gfx1201 = [] {
    ggml_hrx_tune t = ggml_hrx_tune_rdna4;
    t.nodes_per_submit = 256;
    // hrx_stream_copy_d2h over the host link is slow. A device copy to mapped host memory is faster.
    t.mapped_readback  = true;
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        t.mul_mat_vec[i].xs_wg_per_cu = 2;
    }
    // A dispatch that fills all wave slots of the device costs approximately 30 us more on gfx1201.
    // 4 workgroups of 256 lanes for each CU use half of the slots.
    t.stream_wg_per_cu = 4;
    return t;
}();

static constexpr uint32_t GGML_HRX_WG_32BIT = UINT32_MAX;
// Architected workgroup ids pack y and z into one 32-bit register.
static constexpr uint32_t GGML_HRX_WG_16BIT = 65536;

static constexpr ggml_hrx_arch ggml_hrx_archs[] = {
    // RDNA3. Untuned.
    { "gfx1100", "gfx1100", true, GGML_HRX_MATRIX_OP_WMMA_GFX11, GGML_HRX_WG_32BIT, &ggml_hrx_tune_rdna3 },
    { "gfx1101", "gfx1101", true, GGML_HRX_MATRIX_OP_WMMA_GFX11, GGML_HRX_WG_32BIT, &ggml_hrx_tune_rdna3 },
    { "gfx1102", "gfx1102", true, GGML_HRX_MATRIX_OP_WMMA_GFX11, GGML_HRX_WG_32BIT, &ggml_hrx_tune_rdna3 },
    { "gfx1103", "gfx1103", true, GGML_HRX_MATRIX_OP_WMMA_GFX11, GGML_HRX_WG_32BIT, &ggml_hrx_tune_rdna3 },
    // RDNA3.5. Only gfx1151 is tuned.
    { "gfx1150", "gfx1150", true, GGML_HRX_MATRIX_OP_WMMA_GFX11, GGML_HRX_WG_32BIT, &ggml_hrx_tune_rdna3_5 },
    { "gfx1151", "gfx1151", true, GGML_HRX_MATRIX_OP_WMMA_GFX11, GGML_HRX_WG_32BIT, &ggml_hrx_tune_rdna3_5 },
    { "gfx1152", "gfx1152", true, GGML_HRX_MATRIX_OP_WMMA_GFX11, GGML_HRX_WG_32BIT, &ggml_hrx_tune_rdna3_5 },
    // RDNA4. Only gfx1201 is tuned.
    { "gfx1200", "gfx1200", true, GGML_HRX_MATRIX_OP_WMMA_GFX12, GGML_HRX_WG_16BIT, &ggml_hrx_tune_rdna4 },
    { "gfx1201", "gfx1201", true, GGML_HRX_MATRIX_OP_WMMA_GFX12, GGML_HRX_WG_16BIT, &ggml_hrx_tune_gfx1201 },
};

static constexpr bool ggml_hrx_archs_valid() {
    for (const ggml_hrx_arch & arch : ggml_hrx_archs) {
        if (!ggml_hrx_tune_valid(*arch.tune)) {
            return false;
        }
    }
    return true;
}
static_assert(ggml_hrx_archs_valid(), "ggml_hrx_tune value out of range");

const ggml_hrx_arch * ggml_hrx_arch_find(const char * name) {
    for (const ggml_hrx_arch & arch : ggml_hrx_archs) {
        if (std::strcmp(arch.name, name) == 0) {
            return &arch;
        }
    }
    return nullptr;
}
