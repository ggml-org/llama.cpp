#pragma once

#include "op_types.hpp"

namespace hexagon {

compute_func_type get_compute_func(tensor * dst);

bool requires_thread_barrier(npu_device_tensor_op       prev_op,
                             const npu_device_ne_type & prev_ne,
                             npu_device_tensor_op       op,
                             const npu_device_ne_type & ne);

bool support_op(const npu_device_tensor_op_spec * op_spec,
                const npu_device_tensor_spec *    dst,
                const npu_device_tensor_spec *    srcs,
                size_t                            src_len);

}  // namespace hexagon
