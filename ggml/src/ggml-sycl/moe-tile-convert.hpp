#pragma once

#include <sycl/sycl.hpp>
#include "common.hpp"
#include "moe-xmx-fused.hpp"  // For MXFPXMXLayoutInfo

namespace ggml_sycl {
namespace moe_tile_convert {

/**
 * Convert MXFP4 SoA/Coalesced layout to XMX tiled layout on GPU.
 *
 * This kernel performs GPU-side tile conversion to replace host-side conversion
 * that requires blocking .wait() calls, enabling SYCL command graph compatibility.
 *
 * @param q SYCL queue for kernel execution
 * @param dep_event Dependency event to chain (ensures ordering)
 * @param d_soa_qs Device pointer to SoA quantized weights (qs component)
 * @param d_soa_e Device pointer to SoA exponents (e component)
 * @param d_tiled Device pointer to output tiled buffer
 * @param expert_id Expert index when global_range[0] == 1 (per-expert pointers)
 * @param info MXFP4 XMX layout information (tile dimensions, sizes)
 * @param global_range Global ND-range for parallel execution (n_experts, tile_groups_n, tile_groups_k)
 * @param local_range Local work-group size (typically matches XMX DPAS shape)
 * @return sycl::event representing kernel completion for dependency chaining
 * @throws sycl::exception on kernel submission or execution failure
 *
 * @threadsafe Multiple conversions can run concurrently (each expert independent)
 * @graphcompatible Returns event for chaining, no .wait() calls
 * @performance GPU conversion is ~10-100x faster than host (parallel vs serial)
 */
sycl::event reorder_mxfp4_to_xmx_tiled(
    sycl::queue& q,
    sycl::event dep_event,
    const uint8_t* d_soa_qs,
    const uint8_t* d_soa_e,
    uint8_t* d_tiled,
    size_t expert_id,
    const moe_xmx_fused::MXFPXMXLayoutInfo& info,
    const sycl::range<3>& global_range,
    const sycl::range<3>& local_range);

/**
 * Convert MXFP4 AoS layout to XMX tiled layout on GPU.
 *
 * @param d_aos Device pointer to AoS blocks (block_mxfp4)
 * @param d_tiled Device pointer to output tiled buffer
 * @param expert_id Expert index when global_range[0] == 1 (per-expert pointers)
 * @param info MXFP4 XMX layout information (tile dimensions, sizes)
 * @param global_range Global ND-range for parallel execution (n_experts, tile_groups_n, tile_groups_k)
 * @param local_range Local work-group size (typically matches XMX DPAS shape)
 */
sycl::event reorder_mxfp4_aos_to_xmx_tiled(
    sycl::queue& q,
    sycl::event dep_event,
    const uint8_t* d_aos,
    uint8_t* d_tiled,
    size_t expert_id,
    const moe_xmx_fused::MXFPXMXLayoutInfo& info,
    const sycl::range<3>& global_range,
    const sycl::range<3>& local_range);

}  // namespace moe_tile_convert
}  // namespace ggml_sycl
