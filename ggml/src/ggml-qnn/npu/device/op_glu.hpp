#pragma once

#include "op_types.hpp"

namespace hexagon {

bool glu_f32(hexagon::tensor * out, hexagon::compute_params * params);
bool glu_f16(hexagon::tensor * out, hexagon::compute_params * params);

bool is_glu_op_supported(const npu_device_tensor_op_spec * op_spec,
                         const npu_device_tensor_spec *    dst,
                         const npu_device_tensor_spec *    srcs,
                         size_t                            src_len);

}  // namespace hexagon
