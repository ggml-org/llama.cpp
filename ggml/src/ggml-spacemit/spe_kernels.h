#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <typeinfo>

constexpr uint64_t SpineInvalidCoreID = 0xFFFFFFFF;

enum class SpineCoreArchID : uint16_t {
    CoreArchNone = 0,
    CoreArchX60  = 0x503C,
    CoreArchX100 = 0x5064,
    CoreArchX200 = 0x50C8,
    CoreArchA60  = 0xA03C,
    CoreArchA100 = 0xA064,
    CoreArchA200 = 0xA0C8,
};

union SpineArchExtensionInfoDef {
    uint32_t value;

    struct {
        uint8_t rvv;
        uint8_t ime;
        uint8_t ame;
        uint8_t vme;
    } fields;
};

union SpineArchInfoDef {
    uint64_t value;

    struct {
        SpineCoreArchID           core_arch_id;
        uint8_t                   matrix_info;
        uint8_t                   vector_info;
        SpineArchExtensionInfoDef core_ext_info;
    } fields;
};

namespace mlir::speir::spekernel {

using spe_pack_cast_function_def = std::function<void(void *  in,
                                                      int64_t in_offset,
                                                      int64_t in_stride,
                                                      void *  out,
                                                      int64_t out_offset,
                                                      int64_t out_stride,
                                                      int64_t mc,
                                                      int64_t kc,
                                                      int64_t dst_mc,
                                                      int64_t dst_kc)>;

using spe_reducesum_mxn_function_def = std::function<void(void *  in,
                                                          int64_t lhs_offset,
                                                          int64_t lhs_stride0,
                                                          int64_t lhs_stride1,
                                                          void *  out,
                                                          int64_t out_offset,
                                                          int64_t out_stride,
                                                          int64_t src_mc,
                                                          int64_t src_nc,
                                                          int64_t nth,
                                                          int64_t dst_mc,
                                                          int64_t dst_nc)>;

template <typename in_type, typename out_type, size_t src_mb, size_t src_kb, size_t dst_mb, size_t dst_kb>
void spe_pack_cast(void * in, int64_t in_offset, int64_t in_stride, void * out, int64_t out_offset,
                   int64_t out_stride, int64_t mc, int64_t kc, int64_t dst_mc, int64_t dst_kc);

template <typename in_type, typename out_type, size_t src_mb, size_t src_nb, size_t dst_mb, size_t dst_nb>
void spe_reducesum_mxn(void * in, int64_t lhs_offset, int64_t lhs_stride0, int64_t lhs_stride1, void * out,
                       int64_t out_offset, int64_t out_stride, int64_t src_mc, int64_t src_nc, int64_t nth,
                       int64_t dst_mc, int64_t dst_nc);

}  // namespace mlir::speir::spekernel

namespace mlir::speir::spekernel::ime2 {
// fixed_8 requires every Q41Block zero-point slot to equal 8.
enum class q41_zero_point_mode { per_row, fixed_8 };

spe_pack_cast_function_def get_spe_pack_cast_function(const uint64_t         march,
                                                      const std::type_info & in_type,
                                                      const std::type_info & out_type,
                                                      const size_t           src_mb,
                                                      const size_t           src_kb,
                                                      const size_t           dst_mb,
                                                      const size_t           dst_kb,
                                                      q41_zero_point_mode mode = q41_zero_point_mode::per_row);
}  // namespace mlir::speir::spekernel::ime2

namespace mlir::speir::spekernel::rvv {
spe_reducesum_mxn_function_def get_spe_reducesum_mxn_function(const uint64_t         march,
                                                              const std::type_info & in_type,
                                                              const std::type_info & out_type,
                                                              const size_t           src_mb,
                                                              const size_t           src_nb,
                                                              const size_t           dst_mb,
                                                              const size_t           dst_nb);
}  // namespace mlir::speir::spekernel::rvv
