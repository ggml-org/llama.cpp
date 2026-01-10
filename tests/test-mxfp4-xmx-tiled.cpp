// Unit test for MXFP4 XMX tiled layout conversion (GPU vs CPU reference)
//
// This test verifies that the GPU-side tile conversion kernels produce
// identical output to the CPU reference implementation for both SoA and AoS
// inputs, using a single SYCL device.
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-mxfp4-xmx-tiled

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <sycl/sycl.hpp>

#include "ggml-common.h"
#include "ggml-sycl/moe-tile-convert.hpp"
#include "ggml-sycl/moe-xmx-fused.hpp"

#if !SYCL_XMX_MOE_AVAILABLE
int main() {
    fprintf(stderr, "SYCL XMX matrix extension not available; skipping test.\n");
    return 0;
}
#else

static void fill_random_blocks(std::vector<block_mxfp4> & blocks, std::mt19937 & rng) {
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (auto & block : blocks) {
        block.e = static_cast<uint8_t>(byte_dist(rng));
        for (size_t i = 0; i < sizeof(block.qs); ++i) {
            block.qs[i] = static_cast<uint8_t>(byte_dist(rng));
        }
    }
}

static void aos_to_soa(const std::vector<block_mxfp4> & aos,
                       std::vector<uint8_t> & qs,
                       std::vector<uint8_t> & e) {
    const size_t nblocks = aos.size();
    const size_t qs_stride = sizeof(aos[0].qs);
    for (size_t i = 0; i < nblocks; ++i) {
        std::memcpy(qs.data() + i * qs_stride, aos[i].qs, qs_stride);
        e[i] = aos[i].e;
    }
}

static bool compare_buffers(const std::vector<uint8_t> & ref,
                            const std::vector<uint8_t> & got,
                            size_t & mismatch_idx,
                            uint8_t & ref_val,
                            uint8_t & got_val) {
    if (ref.size() != got.size()) {
        mismatch_idx = 0;
        ref_val = 0;
        got_val = 0;
        return false;
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != got[i]) {
            mismatch_idx = i;
            ref_val = ref[i];
            got_val = got[i];
            return false;
        }
    }
    return true;
}

int main() {
    if (!getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
    }

    const auto & info = ggml_sycl_info();
    if (info.device_count <= 0) {
        fprintf(stderr, "No SYCL devices available; skipping test.\n");
        return 0;
    }

    const int device_id = 0;
    if (!info.devices[device_id].xmx_caps.supported) {
        fprintf(stderr, "Device does not support XMX; skipping test.\n");
        return 0;
    }

    moe_xmx_fused::MXFPXMXConfig cfg = moe_xmx_fused::MXFPXMXConfig::from_device(device_id);

    const int64_t out_dim = cfg.tile_n_total + 7;
    const int64_t in_dim  = 96;  // 3 XMX_K blocks
    const int64_t n_experts = 2;
    const int64_t n_k_blocks = in_dim / moe_xmx_fused::MXFPXMXConfig::XMX_K;
    const size_t blocks_per_expert = static_cast<size_t>(out_dim * n_k_blocks);
    const size_t total_blocks = blocks_per_expert * static_cast<size_t>(n_experts);

    moe_xmx_fused::MXFPXMXLayoutInfo info_layout =
        moe_xmx_fused::MXFPXMXLayoutInfo::compute(out_dim, in_dim, cfg);
    const size_t tiled_bytes = static_cast<size_t>(info_layout.total_bytes) * static_cast<size_t>(n_experts);

    std::mt19937 rng(123);
    std::vector<block_mxfp4> aos(total_blocks);
    fill_random_blocks(aos, rng);

    std::vector<uint8_t> soa_qs(total_blocks * sizeof(aos[0].qs));
    std::vector<uint8_t> soa_e(total_blocks);
    aos_to_soa(aos, soa_qs, soa_e);

    std::vector<uint8_t> expected(tiled_bytes);
    for (int64_t expert = 0; expert < n_experts; ++expert) {
        const uint8_t * expert_qs = soa_qs.data() + expert * blocks_per_expert * sizeof(aos[0].qs);
        const uint8_t * expert_e  = soa_e.data() + expert * blocks_per_expert;
        uint8_t * expert_dst      = expected.data() + expert * info_layout.total_bytes;
        moe_xmx_fused::reorder_mxfp4_to_xmx_layout(expert_qs, expert_e, expert_dst, info_layout);
    }

    sycl::queue q(sycl::gpu_selector_v);

    uint8_t * d_aos    = sycl::malloc_device<uint8_t>(aos.size() * sizeof(aos[0]), q);
    uint8_t * d_soa_qs = sycl::malloc_device<uint8_t>(soa_qs.size(), q);
    uint8_t * d_soa_e  = sycl::malloc_device<uint8_t>(soa_e.size(), q);
    uint8_t * d_tiled  = sycl::malloc_device<uint8_t>(tiled_bytes, q);

    if (!d_aos || !d_soa_qs || !d_soa_e || !d_tiled) {
        fprintf(stderr, "Device allocation failed; skipping test.\n");
        if (d_aos) sycl::free(d_aos, q);
        if (d_soa_qs) sycl::free(d_soa_qs, q);
        if (d_soa_e) sycl::free(d_soa_e, q);
        if (d_tiled) sycl::free(d_tiled, q);
        return 0;
    }

    q.memcpy(d_aos, aos.data(), aos.size() * sizeof(aos[0])).wait();
    q.memcpy(d_soa_qs, soa_qs.data(), soa_qs.size()).wait();
    q.memcpy(d_soa_e, soa_e.data(), soa_e.size()).wait();

    const sycl::range<3> global_range(static_cast<size_t>(n_experts),
                                      static_cast<size_t>(info_layout.n_tile_groups_n),
                                      static_cast<size_t>(info_layout.n_tile_groups_k));
    const sycl::range<3> local_range(1, 1, 1);

    // SoA -> XMX tiled
    ggml_sycl::moe_tile_convert::reorder_mxfp4_to_xmx_tiled(
        q, sycl::event(), d_soa_qs, d_soa_e, d_tiled, 0, info_layout, global_range, local_range).wait();

    std::vector<uint8_t> got_soa(tiled_bytes);
    q.memcpy(got_soa.data(), d_tiled, tiled_bytes).wait();

    size_t mismatch_idx = 0;
    uint8_t ref_val = 0;
    uint8_t got_val = 0;
    if (!compare_buffers(expected, got_soa, mismatch_idx, ref_val, got_val)) {
        fprintf(stderr, "SoA conversion mismatch at byte %zu: expected=%u got=%u\n",
                mismatch_idx, ref_val, got_val);
        sycl::free(d_aos, q);
        sycl::free(d_soa_qs, q);
        sycl::free(d_soa_e, q);
        sycl::free(d_tiled, q);
        return 1;
    }

    // AoS -> XMX tiled
    ggml_sycl::moe_tile_convert::reorder_mxfp4_aos_to_xmx_tiled(
        q, sycl::event(), d_aos, d_tiled, 0, info_layout, global_range, local_range).wait();

    std::vector<uint8_t> got_aos(tiled_bytes);
    q.memcpy(got_aos.data(), d_tiled, tiled_bytes).wait();

    if (!compare_buffers(expected, got_aos, mismatch_idx, ref_val, got_val)) {
        fprintf(stderr, "AoS conversion mismatch at byte %zu: expected=%u got=%u\n",
                mismatch_idx, ref_val, got_val);
        sycl::free(d_aos, q);
        sycl::free(d_soa_qs, q);
        sycl::free(d_soa_e, q);
        sycl::free(d_tiled, q);
        return 1;
    }

    sycl::free(d_aos, q);
    sycl::free(d_soa_qs, q);
    sycl::free(d_soa_e, q);
    sycl::free(d_tiled, q);

    fprintf(stderr, "MXFP4 XMX tiled conversion test passed.\n");
    return 0;
}
#endif
