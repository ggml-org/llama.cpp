#pragma once

#include "op_types.hpp"

namespace hexagon {

compute_func_type get_compute_func(tensor * dst);

bool requires_thread_barrier(npu_device_tensor_op op);

bool support_op(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1,
                const npu_device_tensor_spec & dst, npu_device_tensor_op op);

}  // namespace hexagon
