#pragma once

#include "op_types.hpp"

namespace hexagon {

bool get_rows_f32(tensor * out, compute_params * params);
bool set_rows_generic(tensor * out, compute_params * params);

bool is_rows_supported(const npu_device_tensor_op_spec * op_spec,
                       const npu_device_tensor_spec *    dst,
                       const npu_device_tensor_spec *    srcs,
                       size_t                            src_len);
bool is_rows_required_sync(npu_device_tensor_op       prev_op,
                           const npu_device_ne_type & prev_ne,
                           npu_device_tensor_op       op,
                           const npu_device_ne_type & ne);

}  // namespace hexagon
