#include "pipeline.h"

namespace ov {
namespace frontend {
namespace ggml {
namespace pipeline {

// NPU graph-level lowerings go here (e.g. GatherMatmul -> Gather + MatMul for MoE, which the NPU plugin
// lacks). NPU compile properties are not passes: they
// are built in device/npu_config.cpp.
void register_npu_passes(ov::pass::Manager & manager, const GraphTraits & traits) {
    (void) manager;
    (void) traits;
}

}  // namespace pipeline
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
