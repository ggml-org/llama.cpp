#include "pipeline.h"

#include "ggml-openvino/openvino/pass/move_scale_before_matmul.h"

namespace ov {
namespace frontend {
namespace ggml {
namespace pipeline {

// NPU graph-level lowerings go here (e.g. GatherMatmul -> Gather + MatMul for MoE, which the NPU plugin
// lacks). NPU compile properties are not passes: they
// are built in device/npu_config.cpp.
void register_npu_passes(ov::pass::Manager & manager, const GraphTraits & traits) {
    (void) traits;

    // f16 range: apply a scalar scale before a weight MatMul instead of after it (granite residual_scale).
    manager.register_pass<pass::MoveScaleBeforeMatMul>();
}

}  // namespace pipeline
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
