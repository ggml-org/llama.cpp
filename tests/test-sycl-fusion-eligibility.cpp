#include "../ggml/src/ggml-sycl/fusion.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace {

enum class multiplier_layout {
    broadcast,
    full_shape,
    noncontiguous,
    padded,
};

const char * layout_name(multiplier_layout layout) {
    switch (layout) {
        case multiplier_layout::broadcast:     return "broadcast";
        case multiplier_layout::full_shape:    return "full-shape";
        case multiplier_layout::noncontiguous: return "noncontiguous";
        case multiplier_layout::padded:        return "padded";
    }
    return "unknown";
}

bool can_fuse_rms_norm_mul(multiplier_layout layout, bool rms_is_src1) {
    constexpr int64_t n_columns = 16;
    constexpr int64_t n_rows    = 4;

    ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::cerr << "failed to initialize ggml context\n";
        std::exit(1);
    }

    ggml_tensor * input    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_columns, n_rows);
    ggml_tensor * rms_norm = ggml_rms_norm(ctx, input, 1e-5f);
    ggml_tensor * mul_w    = layout == multiplier_layout::full_shape
        ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_columns, n_rows)
        : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_columns);

    if (layout == multiplier_layout::noncontiguous) {
        mul_w->nb[0] = 2 * sizeof(float);
        mul_w->nb[1] = mul_w->nb[0] * mul_w->ne[0];
        mul_w->nb[2] = mul_w->nb[1];
        mul_w->nb[3] = mul_w->nb[2];
    } else if (layout == multiplier_layout::padded) {
        mul_w->nb[1] += sizeof(float);
        mul_w->nb[2] = mul_w->nb[1];
        mul_w->nb[3] = mul_w->nb[2];
    }

    ggml_tensor * mul = ggml_mul(ctx, rms_norm, mul_w);
    if (rms_is_src1) {
        std::swap(mul->src[0], mul->src[1]);
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, mul);

    const bool result = ggml_sycl_can_fuse(
        graph,
        0,
        { GGML_OP_RMS_NORM, GGML_OP_MUL },
        {});
    ggml_free(ctx);
    return result;
}

bool expect_eligibility(multiplier_layout layout, bool rms_is_src1, bool expected) {
    const bool actual = can_fuse_rms_norm_mul(layout, rms_is_src1);
    if (actual == expected) {
        return true;
    }

    std::cerr << "unexpected RMS_NORM+MUL eligibility for " << layout_name(layout)
              << " multiplier with RMS_NORM as src[" << (rms_is_src1 ? 1 : 0)
              << "]: expected " << expected << ", got " << actual << '\n';
    return false;
}

} // namespace

int main() {
    g_ggml_sycl_enable_fusion = 1;

    bool ok = true;
    for (const bool rms_is_src1 : { false, true }) {
        ok &= expect_eligibility(multiplier_layout::broadcast, rms_is_src1, true);
        ok &= expect_eligibility(multiplier_layout::full_shape, rms_is_src1, false);
        ok &= expect_eligibility(multiplier_layout::noncontiguous, rms_is_src1, false);
        ok &= expect_eligibility(multiplier_layout::padded, rms_is_src1, false);
    }

    return ok ? 0 : 1;
}
