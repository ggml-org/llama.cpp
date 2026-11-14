#pragma once

// Post-translation pass pipelines, split by target device.
//
// The translators in op/*.cpp produce a device-neutral OpenVINO model. Everything that is specific to how
// one device runs it -- fusions for one plugin, workarounds for one plugin's defects, lowerings for ops one
// plugin lacks -- is registered here, in that device's pipeline:
//
//   register_common_passes()     every device, before the device passes (dequantization marking, fusions)
//   register_device_passes()     pass_registry/{cpu,gpu,npu}.cpp
//   register_execution_passes()  every device, after the device passes (stateful KV, static shapes)
//
// Every device workaround names the plugin defect it covers and keeps its env opt-out next to the
// registration.

#include <openvino/pass/make_stateful.hpp>
#include <openvino/pass/manager.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder;

namespace pipeline {

enum class Device { CPU, GPU, NPU };

// What the pipelines need to know about one graph, computed once per conversion.
struct GraphTraits {
    Device device = Device::CPU;
    bool is_static = false;    // static shapes (NPU, or GGML_OPENVINO_FORCE_STATIC)
    bool is_stateful = false;  // KV cache kept inside the model (GGML_OPENVINO_STATEFUL_EXECUTION)
};

GraphTraits analyze_graph(const GgmlDecoder & decoder);

void register_common_passes(ov::pass::Manager & manager, const GraphTraits & traits);
void register_device_passes(ov::pass::Manager & manager, const GraphTraits & traits);
void register_execution_passes(ov::pass::Manager & manager,
                               const GraphTraits & traits,
                               const ov::pass::MakeStateful::ParamResPairs & kv_param_res_pairs);

// Per-device halves of register_device_passes().
void register_cpu_passes(ov::pass::Manager & manager, const GraphTraits & traits);
void register_gpu_passes(ov::pass::Manager & manager, const GraphTraits & traits);
void register_npu_passes(ov::pass::Manager & manager, const GraphTraits & traits);

}  // namespace pipeline
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
