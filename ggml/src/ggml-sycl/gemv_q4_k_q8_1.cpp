/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  MIT License
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  gemvq_q4_k_q8_1.cpp
 *
 *  Description:
 *    Exp gemv for llama.cpp's q4_K X q8_1 quantized types
 *    Expected GEMV Layout: nrows x ncols : ncols x 1;
 **************************************************************************/

#include <sycl/sycl.hpp>

#include "dpct/helper.hpp"
#include "ggml-sycl/builtins.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml.h"
#include "quants.hpp"

template <int ElementSize, int Cols, int Rows, int Values> struct BlockLayout {
    static constexpr int element_size = ElementSize;
    static constexpr int cols         = Cols;
    static constexpr int rows         = Rows;
    static constexpr int values       = Values;
};

template <typename block_q_t, typename block_layout, typename T>
static __dpct_inline__ void get_quant_tile(const void * weights, size_t ncols, size_t nrows, coord_t coord, T * tile) {
#ifdef __SYCL_DEVICE_ONLY__
    using namespace sycl::detail;
    // Width is expected in bytes. Quants are packed in bytes, 1 col == 1 nibble (q4_K)
    size_t width = ncols / (block_q_t::traits::qr);
    XeSubgroup2DBlockLoad<block_layout::element_size, block_layout::cols, block_layout::rows, block_layout::values>()(
        weights, width, nrows, width, coord, tile);
#else
    (void) weights;
    (void) nrows;
    (void) ncols;
    (void) coord;
    (void) tile;
    GGML_ABORT("Host code should not get here");
#endif
}

template <int tile_height> struct BlockLoadType {};

template <> struct BlockLoadType<1> {
    using T = uint16_t[1];
};

template <> struct BlockLoadType<2> {
    using T = sycl::vector_types::uint16_t2;
};

template <> struct BlockLoadType<4> {
    using T = sycl::vector_types::uint16_t4;
};

template <> struct BlockLoadType<8> {
    using T = sycl::vector_types::uint16_t8;
};

template <> struct BlockLoadType<16> {
    using T = sycl::vector_types::uint16_t16;
};

template <int tile_height> using block_load_t = typename BlockLoadType<tile_height>::T;

template <int tile_height> struct LayoutTraits {
    // Block Load
    static constexpr int bytes   = 2;
    static constexpr int rows    = tile_height;
    static constexpr int columns = 16;
    static constexpr int values  = 1;

    using QK_Layout = BlockLayout<bytes, columns, rows, values>;
    using Q8_Layout = BlockLayout<2 * bytes, columns, 1, values>;
    using QK_tile_t = block_load_t<tile_height>;

    // Tiled GemV traits
    static constexpr size_t coord_stride = columns;

    template <size_t prefetch_pipeline> static constexpr size_t prefetch_offset = coord_stride * prefetch_pipeline;

    template <typename block_q_t> __dpct_inline__ static size_t coord_range(size_t ncols) {
        return ncols / (QK_Layout::element_size * block_q_t::traits::qr);
    }
};

__dpct_inline__ static int32_t unpack_q4_tile(uint16_t q4_tile) {
    return ((q4_tile >> 12) & 0x0F) << 16 | ((q4_tile >> 8) & 0x0F) << 24 | ((q4_tile >> 4) & 0x0F) |
           (q4_tile & 0x0F) << 8;
}

// NOTE: Branchless approach gives better performance
static __dpct_inline__ void decode_chunk_scales(int chunk_idx, const uint8_t * scales, uint8_t * sc_m) {
    // chunk_idx < 4
    uint16_t val_a = *reinterpret_cast<const uint16_t *>(&scales[2 * chunk_idx]) & 0x3f3f;

    // chunk_idx >= 4
    uint16_t hbits = *reinterpret_cast<const uint16_t *>(&scales[(chunk_idx - 4) * 2]);
    uint8_t  b0    = ((hbits & 0x00c0) >> 2) | (scales[chunk_idx + 4] & 0x0f);
    uint8_t  b1    = ((hbits & 0xc000) >> 10) | ((scales[chunk_idx + 4] >> 4) & 0x0f);
    uint16_t val_b = static_cast<uint16_t>(b0) | (static_cast<uint16_t>(b1) << 8);

    uint32_t mask                       = -(chunk_idx < 4);
    *reinterpret_cast<uint16_t *>(sc_m) = (val_a & mask) | (val_b & ~mask);
}

static __dpct_inline__ void decode_superblock_scale(const uint8_t * weights_ptr, size_t offset, sycl::float2 * dm4f) {
    const ggml_half2 * dm4 = reinterpret_cast<const ggml_half2 *>(weights_ptr + offset);
    *dm4f                  = dm4->convert<float, sycl::rounding_mode::automatic>();
}

template <typename Traits, size_t prefetch_pipeline>
__dpct_inline__ static void q4_K_q8_1_tiled_gemvq(
    const void * weights, const void * input, float * dst, const size_t ncols, const size_t nrows,
    const sycl::nd_item<1> & it) {
    using block_q_t  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;
    using block_q8_t = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q8_1>;

    using bl_layout                     = typename Traits::QK_Layout;
    using q8_layout                     = typename Traits::Q8_Layout;
    using q4_tile_t                     = typename Traits::QK_tile_t;
    constexpr size_t coord_stride       = Traits::coord_stride;
    constexpr int    tile_height        = Traits::rows;
    constexpr size_t sblock_coord_width = (block_q_t::traits::qk / block_q_t::traits::qr) / Traits::bytes;

    // Since local_range = WARP_RANGE
    size_t       coord_range    = Traits::template coord_range<block_q_t>(ncols);
    const int    workgroup_id   = it.get_group_linear_id();
    auto         local_id       = it.get_local_linear_id();    // subgroup local id = workgroup local id
    const size_t tile_row_begin = tile_height * workgroup_id;  // NOTE: only supports a single sg per wg
    const int    blocks_per_row = ncols / block_q_t::traits::qk;

    const uint8_t *               weights_ptr = static_cast<const uint8_t *>(weights);
    sycl::vec<float, tile_height> partial_sums{ 0 };

    // INFO: Current blockloads grabs entire superblock every 4 iterations
    for (size_t tile_coord_begin = 0; tile_coord_begin < coord_range; tile_coord_begin += sblock_coord_width) {
        const int          chunk_idx      = local_id / 2;
        const int          superblock_idx = tile_coord_begin / 64;
        const int          q8_block_idx   = superblock_idx * 8 + chunk_idx;
        const uint         q8_dm_offset   = block_q8_t::get_d_offset(1, ncols, q8_block_idx).first;
        const ggml_half2 * q8_dm =
            reinterpret_cast<const ggml_half2 *>(reinterpret_cast<const uint8_t *>(input) + q8_dm_offset);
        const float d8 = static_cast<float>(q8_dm->x());

        int32_t      dot1[tile_height] = { 0 };
        int32_t      dot2[tile_height] = { 0 };
        uint8_t      sc_m[2 * tile_height];
        sycl::float2 dm4f[tile_height];

#pragma unroll(sblock_coord_width / Traits::columns)
        for (size_t w = 0; w < sblock_coord_width / Traits::columns; w++) {
            q4_tile_t q4_tile;
            uint      q8_tile;

            const int  w_coord  = tile_coord_begin + coord_stride * w;
            const auto q4_coord = coord_t{ w_coord, tile_row_begin };
            get_quant_tile<block_q_t, bl_layout>(weights, ncols, nrows, q4_coord, &q4_tile);

            const auto q8_coord = coord_t{ w_coord, 0 };
            get_quant_tile<block_q8_t, q8_layout>(input, ncols, 1, q8_coord, &q8_tile);

#pragma unroll(tile_height)
            for (size_t i = 0; i < tile_height; i++) {
                const size_t    q4_k_block_idx = (tile_row_begin + i) * blocks_per_row + superblock_idx;
                const auto      scs_offsets    = block_q_t::get_d_offset(nrows, ncols, q4_k_block_idx);
                const uint8_t * scales         = weights_ptr + scs_offsets.first;

                if (!w) {
                    decode_superblock_scale(weights_ptr, scs_offsets.second, &dm4f[i]);
                    decode_chunk_scales(chunk_idx, scales, &sc_m[2 * i]);
                }

                const int32_t q4_val = unpack_q4_tile(q4_tile[i]);
                const int32_t q8_val = static_cast<int32_t>(q8_tile);
                dot1[i]              = sycl::detail::__builtin_IB_dp4a_ss(dot1[i], q4_val, q8_val);
                dot2[i]              = sycl::detail::__builtin_IB_dp4a_ss(dot2[i], 0x01010101, q8_val);
            }
        }

#pragma unroll(tile_height)
        for (size_t i = 0; i < tile_height; i++) {
            partial_sums[i] +=
                dm4f[i].x() * d8 * (dot1[i] * sc_m[2 * i]) - dm4f[i].y() * d8 * (dot2[i] * sc_m[2 * i + 1]);
        }
    }

#pragma unroll(tile_height)
    for (size_t i = 0; i < tile_height; i++) {
        partial_sums[i] = sycl::reduce_over_group(it.get_sub_group(), partial_sums[i], std::plus<>());
    }

    // INFO: Seems equivalent to the for loop using only the leader
    if (local_id < tile_height) {
        dst[tile_row_begin + local_id] = partial_sums[local_id];
    }
}

template <typename Traits, size_t prefetch_pipeline>
void launch_q4_K_q8_1_tiled_gemvq(sycl::queue * stream, const void * vx, const void * vy, float * dst, size_t ncols,
                                 size_t nrows, sycl::nd_range<1> launch_range) {
    stream->submit([=](sycl::handler & cgh) {
        cgh.parallel_for(launch_range, [=](sycl::nd_item<1> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            q4_K_q8_1_tiled_gemvq<Traits, prefetch_pipeline>(vx, vy, dst, ncols, nrows, nd_item);
        });
    });
}

static void mul_mat_q4_K_q8_1_exp_gemvq(const void * vx, const void * vy, float * dst, const size_t ncols,
                                        const size_t nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    int tile_height = g_ggml_sycl_exp_gemvq_tile_height;
    GGML_ASSERT(nrows % tile_height == 0);

    constexpr size_t prefetch_pipeline = 0;

    const sycl::range<1> global_size(nrows * WARP_SIZE / tile_height);
    const sycl::range<1> wg_size(WARP_SIZE);

    if (tile_height == 16) {
        launch_q4_K_q8_1_tiled_gemvq<LayoutTraits<16>, prefetch_pipeline>(stream, vx, vy, dst, ncols, nrows,
                                                                         sycl::nd_range<1>(global_size, wg_size));
    } else if (tile_height == 8) {
        launch_q4_K_q8_1_tiled_gemvq<LayoutTraits<8>, prefetch_pipeline>(stream, vx, vy, dst, ncols, nrows,
                                                                        sycl::nd_range<1>(global_size, wg_size));
    } else if (tile_height == 4) {
        launch_q4_K_q8_1_tiled_gemvq<LayoutTraits<4>, prefetch_pipeline>(stream, vx, vy, dst, ncols, nrows,
                                                                        sycl::nd_range<1>(global_size, wg_size));
    } else if (tile_height == 2) {
        launch_q4_K_q8_1_tiled_gemvq<LayoutTraits<2>, prefetch_pipeline>(stream, vx, vy, dst, ncols, nrows,
                                                                        sycl::nd_range<1>(global_size, wg_size));
    } else if (tile_height == 1) {
        launch_q4_K_q8_1_tiled_gemvq<LayoutTraits<1>, prefetch_pipeline>(stream, vx, vy, dst, ncols, nrows,
                                                                        sycl::nd_range<1>(global_size, wg_size));
    } else {
        GGML_ABORT("unsupported tile height");
    }
}

void ggml_sycl_op_mul_mat_exp_gemvq(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                   ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                   const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low,
                                   const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_col_size,
                                   const dpct::queue_ptr & stream) {
    constexpr size_t q8_1_ts = sizeof(block_q8_1);
    constexpr size_t q8_1_bs = QK8_1;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % q8_1_bs == 0);

    const int64_t ne00     = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    for (int i = 0; i < src1_ncols; i++) {
        const size_t src1_ddq_i_offset = i * src1_padded_col_size * q8_1_ts / q8_1_bs;
        const char * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_offset;
        float *      dst_dd_i_bs       = dst_dd_i + i * dst->ne[0];
        switch (src0->type) {
            case GGML_TYPE_Q4_K:
                mul_mat_q4_K_q8_1_exp_gemvq(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            default:
                GGML_ABORT("Unsupported quantization reached in exp_gemvq");
        }
    }
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}
