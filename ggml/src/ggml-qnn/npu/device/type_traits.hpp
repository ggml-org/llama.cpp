
#include "hexagon_npu.h"
#include "tensor.hpp"
#include "util.hpp"

#include <hexagon_types.h>

namespace hexagon {

using dequant_output_type = npu_device_fp16_t;

bool init_f16_f32_table(float * table, size_t count);

typedef void (*quantize_row_type)(const float * src, void * dst, size_t count);
typedef void (*dequantize_row_type)(const void * src, dequant_output_type * dst, size_t count, HVX_Vector table);
typedef float (*vec_dot_type)(const void * src0, const void * src1, size_t count);
typedef bool (*can_use_aligned_vec_dot_type)(const void * src0, const void * src1, size_t count);
typedef HVX_Vector (*load_dequant_table_type)();

struct device_type_traits {
    npu_device_tensor_data_type type;
    const char *                type_name;
    int64_t                     blck_size;
    size_t                      type_size;
    bool                        is_quantized;

    dequantize_row_type          to_float                = nullptr;
    quantize_row_type            from_float              = nullptr;
    vec_dot_type                 vec_dot                 = nullptr;
    vec_dot_type                 vec_dot_aligned         = nullptr;
    can_use_aligned_vec_dot_type can_use_aligned_vec_dot = nullptr;
    load_dequant_table_type      load_dequant_table      = nullptr;
};

const device_type_traits & get_type_traits(npu_device_tensor_data_type type);

inline bool is_quantized_type(npu_device_tensor_data_type type) {
    return get_type_traits(type).is_quantized;
}

size_t get_dequantized_row_size(const tensor * tensor);

inline const char * get_type_name(npu_device_tensor_data_type type) {
    return get_type_traits(type).type_name;
}

}  // namespace hexagon

// TODO: move this to a common header
#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING
namespace hexagon {

inline auto make_scoped_op_perf_timer(tensor * op, size_t tidx) {
    auto * src0 = op->get_src(0);
    auto * src1 = op->get_src(1);
    char   buffer[512];
    if (src1 == nullptr) {
        snprintf(buffer, sizeof(buffer), "[%s][%lldx%lldx%lldx%lld%s], tidx: %zu", op_get_name(op->get_op()),
                 src0->get_ne(0), src0->get_ne(1), src0->get_ne(2), src0->get_ne(3), get_type_name(src0->get_type()),
                 tidx);
    } else {
        snprintf(buffer, sizeof(buffer), "[%s][%lldx%lldx%lldx%lld%s],[%lldx%lldx%lldx%lld%s], tidx: %zu",
                 op_get_name(op->get_op()), src0->get_ne(0), src0->get_ne(1), src0->get_ne(2), src0->get_ne(3),
                 get_type_name(src0->get_type()), src1->get_ne(0), src1->get_ne(1), src1->get_ne(2), src1->get_ne(3),
                 get_type_name(src1->get_type()), tidx);
    }
    return npu_scoped_timer<1024>(buffer);
}

}  // namespace hexagon

#    define DEVICE_SCOPED_OP_PERFORMANCE_TRACKER(op, tidx) \
        auto __npu_op_timer_##__LINE__ = hexagon::make_scoped_op_perf_timer(op, tidx)

#    define DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_WITH_SUB_PROC(op, tidx, sub_prefix) \
        auto __npu_op_timer_##sub_prefix = hexagon::make_scoped_op_perf_timer(op, tidx)

#    define DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_SUB_PROC(sub_prefix)                             \
        hexagon::npu_sub_process_scoped_timer<decltype(__npu_op_timer_##sub_prefix)::kBufferCount, 0> \
        __npu_op_sub_timer##sub_prefix(__npu_op_timer_##sub_prefix, #sub_prefix)

#    define DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_WITH_MULTI_SUB_PROC(op, tidx, tracker_name) \
        auto __npu_op_timer_##tracker_name = hexagon::make_scoped_op_perf_timer(op, tidx)

#    define DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(tracker_name, idx, sub_prefix) \
        hexagon::npu_sub_process_scoped_timer<                                                   \
            std::remove_reference_t<decltype(__npu_op_timer_##tracker_name)>::kBufferCount, idx> \
        __npu_op_sub_timer##sub_prefix(__npu_op_timer_##tracker_name, #sub_prefix)

#else
#    define DEVICE_SCOPED_OP_PERFORMANCE_TRACKER(op, tidx)                                       ((void) 0)
#    define DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_WITH_SUB_PROC(op, tidx, sub_prefix)             ((void) 0)
#    define DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_SUB_PROC(sub_prefix)                        ((void) 0)
#    define DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_WITH_MULTI_SUB_PROC(op, tidx, tracker_name)     ((void) 0)
#    define DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(tracker_name, idx, sub_prefix) ((void) 0)
#endif
