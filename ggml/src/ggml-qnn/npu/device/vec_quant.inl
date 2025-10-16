#pragma once

#include "hexagon_npu.h"

#include <hexagon_types.h>

#include <cstdint>

namespace hexagon::vec::quant {

template <typename _TStruct, size_t _Count, auto _MemberPtr> inline HVX_Vector load_into_vector(const _TStruct * src) {
    static_assert(hexagon::kBytesPerVector >= sizeof(_TStruct) * _Count, "_TStruct too large for vector load");

    return *reinterpret_cast<const HVX_UVector *>(&(src->*_MemberPtr));
}

template <typename _TStruct, size_t _Count> inline HVX_Vector load_struct_into_vector(const _TStruct * src) {
    static_assert(hexagon::kBytesPerVector >= sizeof(_TStruct) * _Count, "_TStruct too large for vector load");

    return *reinterpret_cast<const HVX_UVector *>(src);
}

template <typename _TBlock> inline HVX_Vector load_block_generic(const _TBlock & src) {
    static_assert(hexagon::kBytesPerVector >= sizeof(_TBlock), "wrong block size/padding");
    return load_into_vector<_TBlock, 1, &_TBlock::qs>(&src);
}

template <typename _TBlock> inline HVX_Vector make_scale_load_mask() {
    static_assert(sizeof(_TBlock) < hexagon::kBytesPerVector, "wrong block size/padding");
    static_assert(std::is_same_v<decltype(_TBlock::d), npu_device_fp16_t>,
                  "scale field d must be of type npu_device_fp16_t");

    constexpr const size_t kBytesPerScale  = QUANT_BLOCK_SIZE * sizeof(_TBlock::d);
    const size_t           qs_start_offset = offsetof(_TBlock, d);

    hexagon::HVX_VectorAlias ret;
    size_t                   base_i = qs_start_offset;
    for (size_t ret_idx = 0; ret_idx < hexagon::kBytesPerVector; ++ret_idx) {
        const auto offset = ret_idx % kBytesPerScale;
        const auto i      = base_i + (offset % sizeof(_TBlock::d));
        ret.u8[ret_idx]   = (i & 1) ? (i / 2 + 64) : (i / 2);
        if (offset == kBytesPerScale - 1) {
            base_i += sizeof(_TBlock);
        }
    }

    return ret.v;
}

template <typename _TBlock> inline HVX_Vector make_qs_load_mask() {
    static_assert(sizeof(_TBlock) < hexagon::kBytesPerVector, "wrong block size/padding");

    const size_t qs_start_offset = offsetof(_TBlock, qs);
    const size_t qs_end_offset   = qs_start_offset + sizeof(_TBlock::qs);

    hexagon::HVX_VectorAlias ret;
    size_t                   ret_idx = 0;
    for (size_t i = 0; i < hexagon::kBytesPerVector; ++i) {
        auto offset = i % sizeof(_TBlock);
        if (offset >= qs_start_offset && offset < qs_end_offset) {
            ret.u8[ret_idx++] = (i & 1) ? (i / 2 + 64) : (i / 2);
        }
    }

    return ret.v;
}

template <typename _TBlock> inline HVX_Vector load_dual_block_generic(const _TBlock * srcs, HVX_VectorPred mask) {
    static_assert(hexagon::kBytesPerVector >= sizeof(_TBlock) * 2, "wrong block size/padding");
    constexpr const uint32_t kSizeOfQs    = sizeof(_TBlock::qs);
    constexpr const uint32_t kSizeOfScale = sizeof(_TBlock) - kSizeOfQs;

    HVX_Vector blocks = load_into_vector<_TBlock, 2, &_TBlock::qs>(srcs);
    HVX_Vector block1 = Q6_V_vror_VR(blocks, kSizeOfScale);
    return Q6_V_vmux_QVV(mask, blocks, block1);
}

template <typename _TBlock>
inline hexagon::HVX_Vector_x2 load_dual_block_generic(const _TBlock *  srcs,
                                                      const HVX_Vector qs_indices,
                                                      const HVX_Vector scale_indices) {
    static_assert(hexagon::kBytesPerVector >= sizeof(_TBlock) * 2, "wrong block size/padding");

    const HVX_Vector blocks = load_struct_into_vector<_TBlock, 2>(srcs);

    HVX_Vector block01 = Q6_Vb_vlut32_VbVbI(qs_indices, blocks, 0);
    block01            = Q6_Vb_vlut32or_VbVbVbI(block01, qs_indices, blocks, 2);

    HVX_Vector scale01 = Q6_Vb_vlut32_VbVbI(scale_indices, blocks, 0);
    scale01            = Q6_Vb_vlut32or_VbVbVbI(scale01, scale_indices, blocks, 2);

    if constexpr (sizeof(_TBlock) * 4 > hexagon::kBytesPerVector) {
        block01 = Q6_Vb_vlut32or_VbVbVbI(block01, qs_indices, blocks, 1);
        block01 = Q6_Vb_vlut32or_VbVbVbI(block01, qs_indices, blocks, 3);

        scale01 = Q6_Vb_vlut32or_VbVbVbI(scale01, scale_indices, blocks, 1);
        scale01 = Q6_Vb_vlut32or_VbVbVbI(scale01, scale_indices, blocks, 3);
    }

    hexagon::HVX_Vector_x2 result;
    result.val[0] = block01;
    result.val[1] = scale01;
    return result;
}

template <typename _TBlock>
inline hexagon::HVX_Vector_x3 load_qual_block_generic(const _TBlock *  srcs,
                                                      const HVX_Vector qs_indices,
                                                      const HVX_Vector scale_indices) {
    static_assert(hexagon::kBytesPerVector >= sizeof(_TBlock) * 4, "wrong block size/padding");

    hexagon::HVX_Vector_x3 result;

    const HVX_Vector blocks = load_struct_into_vector<_TBlock, 4>(srcs);

    {
        HVX_Vector block0123 = Q6_Vb_vlut32_VbVbI(qs_indices, blocks, 0);
        block0123            = Q6_Vb_vlut32or_VbVbVbI(block0123, qs_indices, blocks, 1);
        block0123            = Q6_Vb_vlut32or_VbVbVbI(block0123, qs_indices, blocks, 2);
        block0123            = Q6_Vb_vlut32or_VbVbVbI(block0123, qs_indices, blocks, 3);

        result.val[0] = block0123;
    }

    {
        HVX_Vector blocks23 = Q6_V_vror_VR(blocks, sizeof(_TBlock) * 2);

        HVX_Vector scale01 = Q6_Vb_vlut32_VbVbI(scale_indices, blocks, 0);
        scale01            = Q6_Vb_vlut32or_VbVbVbI(scale01, scale_indices, blocks, 2);

        HVX_Vector scale23 = Q6_Vb_vlut32_VbVbI(scale_indices, blocks23, 0);
        scale23            = Q6_Vb_vlut32or_VbVbVbI(scale23, scale_indices, blocks23, 2);

        result.val[1] = scale01;
        result.val[2] = scale23;
    }

    return result;
}

template <typename _TBlock>
inline hexagon::HVX_Vector_x5 load_hexa_block_generic(const _TBlock *  srcs,
                                                      const HVX_Vector qs_indices,
                                                      const HVX_Vector scale_indices) {
    static_assert(hexagon::kBytesPerVector >= sizeof(_TBlock) * 6, "wrong block size/padding");
    constexpr const uint32_t kSizeOfQs = sizeof(_TBlock::qs);

    const HVX_Vector blocks = load_struct_into_vector<_TBlock, 6>(srcs);

    hexagon::HVX_Vector_x5 result;
    {
        HVX_Vector block012345 = Q6_Vb_vlut32_VbVbI(qs_indices, blocks, 0);
        block012345            = Q6_Vb_vlut32or_VbVbVbI(block012345, qs_indices, blocks, 1);
        block012345            = Q6_Vb_vlut32or_VbVbVbI(block012345, qs_indices, blocks, 2);
        block012345            = Q6_Vb_vlut32or_VbVbVbI(block012345, qs_indices, blocks, 3);

        result.val[0] = block012345;
        result.val[3] = Q6_V_vror_VR(block012345, kSizeOfQs * 4);  // block45
    }

    {
        HVX_Vector blocks23 = Q6_V_vror_VR(blocks, sizeof(_TBlock) * 2);
        HVX_Vector blocks45 = Q6_V_vror_VR(blocks, sizeof(_TBlock) * 4);

        HVX_Vector scale01 = Q6_Vb_vlut32_VbVbI(scale_indices, blocks, 0);
        scale01            = Q6_Vb_vlut32or_VbVbVbI(scale01, scale_indices, blocks, 2);

        HVX_Vector scale23 = Q6_Vb_vlut32_VbVbI(scale_indices, blocks23, 0);
        scale23            = Q6_Vb_vlut32or_VbVbVbI(scale23, scale_indices, blocks23, 2);

        HVX_Vector scale45 = Q6_Vb_vlut32_VbVbI(scale_indices, blocks45, 0);
        scale45            = Q6_Vb_vlut32or_VbVbVbI(scale45, scale_indices, blocks45, 2);

        result.val[1] = scale01;
        result.val[2] = scale23;
        result.val[4] = scale45;
    }

    return result;
}

inline HVX_Vector dequantize_vec_q40_qf16_2blocks(HVX_Vector qs, HVX_Vector scale01, HVX_Vector table) {
    constexpr const uint32_t kSizeOfQs = sizeof(npu_device_block_q4_0::qs);

    HVX_Vector     q_lo = qs;
    HVX_Vector     q_hi = Q6_Vub_vlsr_VubR(qs, 4);
    HVX_VectorPair qp0  = Q6_W_vshuff_VVR(q_hi, q_lo, kSizeOfQs * (1 + 2));

    q_lo = Q6_V_lo_W(qp0);
    q_lo = Q6_Vb_vshuff_Vb(q_lo);
    qp0  = Q6_Wh_vlut16_VbVhR_nomatch(q_lo, table, 0);

    return Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(qp0), scale01);
}

inline HVX_VectorPair dequantize_vec_q40_qf32_2blocks(HVX_Vector qs, HVX_Vector scale01, HVX_Vector table) {
    constexpr const uint32_t kSizeOfQs = sizeof(npu_device_block_q4_0::qs);

    HVX_Vector     q_lo = qs;
    HVX_Vector     q_hi = Q6_Vub_vlsr_VubR(qs, 4);
    HVX_VectorPair qp0  = Q6_W_vshuff_VVR(q_hi, q_lo, kSizeOfQs * (1 + 2));

    q_lo = Q6_V_lo_W(qp0);
    q_lo = Q6_Vb_vshuff_Vb(q_lo);
    qp0  = Q6_Wh_vlut16_VbVhR_nomatch(q_lo, table, 0);

    q_lo    = Q6_V_lo_W(qp0);
    scale01 = Q6_Vh_vshuff_Vh(scale01);
    q_lo    = Q6_Vh_vshuff_Vh(q_lo);  // TODO: avoid vshuff here

    return Q6_Wqf32_vmpy_VhfVhf(q_lo, scale01);
}

inline HVX_Vector_x2 dequantize_vec_q40_qf16_4blocks(HVX_Vector qs,
                                                     HVX_Vector scale01,
                                                     HVX_Vector scale23,
                                                     HVX_Vector table) {
    constexpr const uint32_t kSizeOfQs = sizeof(npu_device_block_q4_0::qs);

    HVX_Vector q_lo = qs;
    HVX_Vector q_hi = Q6_Vub_vlsr_VubR(qs, 4);

    HVX_VectorPair qp0 = Q6_W_vshuff_VVR(q_hi, q_lo, kSizeOfQs * (1 + 2 + 4));

    q_lo = Q6_V_lo_W(qp0);
    q_lo = Q6_Vb_vshuff_Vb(q_lo);
    qp0  = Q6_Wh_vlut16_VbVhR_nomatch(q_lo, table, 0);

    q_lo = Q6_V_lo_W(qp0);
    q_hi = Q6_V_hi_W(qp0);

    q_lo = Q6_Vqf16_vmpy_VhfVhf(q_lo, scale01);
    q_hi = Q6_Vqf16_vmpy_VhfVhf(q_hi, scale23);

    hexagon::HVX_Vector_x2 result;
    result.val[0] = q_lo;
    result.val[1] = q_hi;
    return result;
}

inline HVX_VectorPair_x2 dequantize_vec_q40_qf32_4blocks(HVX_Vector qs,
                                                         HVX_Vector scale01,
                                                         HVX_Vector scale23,
                                                         HVX_Vector table) {
    constexpr const uint32_t kSizeOfQs = sizeof(npu_device_block_q4_0::qs);

    HVX_Vector q_lo = qs;
    HVX_Vector q_hi = Q6_Vub_vlsr_VubR(qs, 4);

    HVX_VectorPair qp0 = Q6_W_vshuff_VVR(q_hi, q_lo, kSizeOfQs * (1 + 2 + 4));

    q_lo = Q6_V_lo_W(qp0);
    q_lo = Q6_Vb_vshuff_Vb(q_lo);
    qp0  = Q6_Wh_vlut16_VbVhR_nomatch(q_lo, table, 0);

    q_lo = Q6_V_lo_W(qp0);
    q_hi = Q6_V_hi_W(qp0);

    q_lo    = Q6_Vh_vshuff_Vh(q_lo);
    scale01 = Q6_Vh_vshuff_Vh(scale01);

    q_hi    = Q6_Vh_vshuff_Vh(q_hi);
    scale23 = Q6_Vh_vshuff_Vh(scale23);  // TODO: avoid vshuff here

    hexagon::HVX_VectorPair_x2 result;
    result.val[0] = Q6_Wqf32_vmpy_VhfVhf(q_lo, scale01);
    result.val[1] = Q6_Wqf32_vmpy_VhfVhf(q_hi, scale23);
    return result;
}

inline HVX_Vector load_dequant_vec_q40_qf32_1block(const npu_device_block_q4_0 * src,
                                                   const HVX_Vector              qs_indices,
                                                   const HVX_Vector              scale_indices,
                                                   const HVX_Vector              table) {
    // TODO: can we have a single-block version of load and dequantize?
    auto qs = load_dual_block_generic(src, qs_indices, scale_indices);
    return Q6_V_lo_W(dequantize_vec_q40_qf32_2blocks(qs.val[0], qs.val[1], table));
}

inline HVX_Vector load_dequant_vec_q40_qf16_2blocks(const npu_device_block_q4_0 * src,
                                                    const HVX_Vector              qs_indices,
                                                    const HVX_Vector              scale_indices,
                                                    const HVX_Vector              table) {
    auto qs = load_dual_block_generic(src, qs_indices, scale_indices);
    return dequantize_vec_q40_qf16_2blocks(qs.val[0], qs.val[1], table);
}

inline HVX_VectorPair load_dequant_vec_q40_qf32_2blocks(const npu_device_block_q4_0 * src,
                                                        const HVX_Vector              qs_indices,
                                                        const HVX_Vector              scale_indices,
                                                        const HVX_Vector              table) {
    auto qs = load_dual_block_generic(src, qs_indices, scale_indices);
    return dequantize_vec_q40_qf32_2blocks(qs.val[0], qs.val[1], table);
}

inline HVX_VectorPair_x2 load_dequant_vec_q40_qf32_4blocks(const npu_device_block_q4_0 * src,
                                                           const HVX_Vector              qs_indices,
                                                           const HVX_Vector              scale_indices,
                                                           const HVX_Vector              table) {
    auto qs = load_qual_block_generic(src, qs_indices, scale_indices);
    return dequantize_vec_q40_qf32_4blocks(qs.val[0], qs.val[1], qs.val[2], table);
}

}  // namespace hexagon::vec::quant
