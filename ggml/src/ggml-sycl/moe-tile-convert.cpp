#include "moe-tile-convert.hpp"
#include "moe-xmx-fused.hpp"
#include <sycl/sycl.hpp>

namespace ggml_sycl {
namespace moe_tile_convert {

// GPU-side tile conversion kernel (adapted from host-side reorder_mxfp4_to_xmx_layout)
//
// Each work-item processes one (tile_group_k, tile_group_n, expert_id) triplet
// and writes the complete tile group data (scales + qs) to the output buffer.
//
// Thread safety: Each work-group writes to disjoint output regions (no races)
// Graph compatibility: Returns event for dependency chaining, no .wait() calls
namespace {
    constexpr int XMX_K        = 32;
    constexpr int PACKED_BYTES = 16;  // 32 nibbles packed into 16 bytes

    // SYCL kernel for tile conversion
    class reorder_mxfp4_kernel;
    class reorder_mxfp4_aos_kernel;
}

sycl::event reorder_mxfp4_to_xmx_tiled(
    sycl::queue& q,
    sycl::event dep_event,
    const uint8_t* d_soa_qs,
    const uint8_t* d_soa_e,
    uint8_t* d_tiled,
    size_t expert_id,
    const moe_xmx_fused::MXFPXMXLayoutInfo& info,
    const sycl::range<3>& global_range,
    const sycl::range<3>& local_range) {

    const int64_t n_k_blocks = info.n_cols / XMX_K;
    const size_t  experts_dim = global_range[0];

    return q.submit([&](sycl::handler& cgh) {
        // Chain dependency for graph compatibility
        cgh.depends_on(dep_event);

        // Tile conversion kernel
        cgh.parallel_for<class reorder_mxfp4_kernel>(
            sycl::nd_range<3>(global_range, local_range),
            [=](sycl::nd_item<3> item) {
                const size_t expert = (experts_dim > 1) ? item.get_global_id(0) : expert_id;
                const size_t tg_n   = item.get_global_id(1);
                const size_t tg_k   = item.get_global_id(2);

                // Calculate output offset for this tile group
                // Each tile group: scales[tile_n_total] + qs[tile_n_total][16]
                const size_t group_bytes = info.tile_n_total * (1 + PACKED_BYTES);
                const size_t group_offset = (tg_k * info.n_tile_groups_n + tg_n) * group_bytes;

                const size_t nblocks_per_expert = static_cast<size_t>(info.n_rows) *
                                                  static_cast<size_t>(n_k_blocks);
                const size_t qs_bytes_per_expert = nblocks_per_expert * PACKED_BYTES;
                const size_t e_bytes_per_expert  = nblocks_per_expert;

                const uint8_t * src_qs = d_soa_qs;
                const uint8_t * src_e  = d_soa_e;
                uint8_t *       dst_ptr = d_tiled;

                if (experts_dim > 1) {
                    src_qs += expert * qs_bytes_per_expert;
                    src_e  += expert * e_bytes_per_expert;
                    dst_ptr += expert * info.total_bytes;
                }

                dst_ptr += group_offset;

                // Write scales for this tile group [tile_n_total]
                for (int64_t tn = 0; tn < info.tile_n_total; tn++) {
                    const int64_t out_col = tg_n * info.tile_n_total + tn;
                    if (out_col < info.n_rows) {
                        // SoA block index: out_col * n_k_blocks + k_block
                        const int64_t src_block_idx = out_col * n_k_blocks + tg_k;
                        *dst_ptr++ = src_e[src_block_idx];
                    } else {
                        *dst_ptr++ = 0;  // Padding for out-of-bounds
                    }
                }

                // Write packed qs for this tile group [tile_n_total][16]
                for (int64_t tn = 0; tn < info.tile_n_total; tn++) {
                    const int64_t out_col = tg_n * info.tile_n_total + tn;
                    if (out_col < info.n_rows) {
                        const int64_t src_block_idx = out_col * n_k_blocks + tg_k;
                        const uint8_t* src_qs_block = src_qs + src_block_idx * PACKED_BYTES;
                        for (int b = 0; b < PACKED_BYTES; b++) {
                            *dst_ptr++ = src_qs_block[b];
                        }
                    } else {
                        // Zero padding
                        for (int b = 0; b < PACKED_BYTES; b++) {
                            *dst_ptr++ = 0;
                        }
                    }
                }
            }
        );
    });
}

sycl::event reorder_mxfp4_aos_to_xmx_tiled(
    sycl::queue& q,
    sycl::event dep_event,
    const uint8_t* d_aos,
    uint8_t* d_tiled,
    size_t expert_id,
    const moe_xmx_fused::MXFPXMXLayoutInfo& info,
    const sycl::range<3>& global_range,
    const sycl::range<3>& local_range) {

    const int64_t n_k_blocks  = info.n_cols / XMX_K;
    const size_t  experts_dim = global_range[0];
    const size_t  nblocks_per_expert =
        static_cast<size_t>(info.n_rows) * static_cast<size_t>(n_k_blocks);
    const size_t aos_bytes_per_expert = nblocks_per_expert * sizeof(block_mxfp4);

    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dep_event);

        cgh.parallel_for<class reorder_mxfp4_aos_kernel>(
            sycl::nd_range<3>(global_range, local_range),
            [=](sycl::nd_item<3> item) {
                const size_t expert = (experts_dim > 1) ? item.get_global_id(0) : expert_id;
                const size_t tg_n   = item.get_global_id(1);
                const size_t tg_k   = item.get_global_id(2);

                const size_t group_bytes  = info.tile_n_total * (1 + PACKED_BYTES);
                const size_t group_offset = (tg_k * info.n_tile_groups_n + tg_n) * group_bytes;

                const uint8_t * src_aos = d_aos;
                uint8_t *       dst_ptr = d_tiled;

                if (experts_dim > 1) {
                    src_aos += expert * aos_bytes_per_expert;
                    dst_ptr += expert * info.total_bytes;
                }

                dst_ptr += group_offset;

                const block_mxfp4 * src_blocks = reinterpret_cast<const block_mxfp4 *>(src_aos);

                for (int64_t tn = 0; tn < info.tile_n_total; tn++) {
                    const int64_t out_col = tg_n * info.tile_n_total + tn;
                    if (out_col < info.n_rows) {
                        const int64_t src_block_idx = out_col * n_k_blocks + tg_k;
                        const block_mxfp4 & src_block = src_blocks[src_block_idx];
                        *dst_ptr++ = src_block.e;
                    } else {
                        *dst_ptr++ = 0;
                    }
                }

                for (int64_t tn = 0; tn < info.tile_n_total; tn++) {
                    const int64_t out_col = tg_n * info.tile_n_total + tn;
                    if (out_col < info.n_rows) {
                        const int64_t src_block_idx = out_col * n_k_blocks + tg_k;
                        const block_mxfp4 & src_block = src_blocks[src_block_idx];
                        for (int b = 0; b < PACKED_BYTES; b++) {
                            *dst_ptr++ = src_block.qs[b];
                        }
                    } else {
                        for (int b = 0; b < PACKED_BYTES; b++) {
                            *dst_ptr++ = 0;
                        }
                    }
                }
            });
    });
}

}  // namespace moe_tile_convert
}  // namespace ggml_sycl
