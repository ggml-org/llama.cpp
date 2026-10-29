#include "pipeline.h"

#include "ggml-openvino/ggml-openvino-extra.h"
#include "ggml-openvino/openvino/decoder.h"
#include "ggml-openvino/openvino/pass/fuse_argsort_topk.h"
#include "ggml-openvino/openvino/pass/fuse_to_conv.h"
#include "ggml-openvino/openvino/pass/kv_state_seq_axis.h"
#include "ggml-openvino/openvino/pass/mark_decompression_convert_constant_folding.h"
#include "ggml-openvino/openvino/pass/mark_dequantization_subgraph.h"
#include "ggml-openvino/openvino/pass/squeeze_matmul.h"

#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace pipeline {

GraphTraits analyze_graph(const GgmlDecoder & decoder) {
    GraphTraits traits;
    traits.device = ggml_openvino_is_npu() ? Device::NPU : ggml_openvino_is_gpu() ? Device::GPU : Device::CPU;
    traits.is_static = decoder.is_static();
    traits.is_stateful = decoder.is_stateful();
    return traits;
}

void register_common_passes(ov::pass::Manager & manager, const GraphTraits & traits) {
    (void) traits;
    manager.register_pass<ov::pass::MarkCompressedFloatConstants>();
    // Marks the Convert/Subtract/Multiply nodes of our GatherMatmul dequantization chain
    // (make_int4_weights/make_int8_weights, for_gather_matmul=true) with disable_constant_folding,
    // so it survives ConstantFolding regardless of whether the target plugin's own
    // is_decompression_multiply() recognizes GatherMatmul as a valid consumer.
    manager.register_pass<ov::pass::MarkDequantization>(
        std::vector<ov::element::Type>{ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4});
    manager.register_pass<pass::FuseToConv>();
    manager.register_pass<pass::FuseArgsortTopK>();
}

void register_device_passes(ov::pass::Manager & manager, const GraphTraits & traits) {
    switch (traits.device) {
    case Device::CPU:
        register_cpu_passes(manager, traits);
        break;
    case Device::GPU:
        register_gpu_passes(manager, traits);
        break;
    case Device::NPU:
        register_npu_passes(manager, traits);
        break;
    }
}

void register_execution_passes(ov::pass::Manager & manager,
                               const GraphTraits & traits,
                               const ov::pass::MakeStateful::ParamResPairs & kv_param_res_pairs) {
    if (traits.is_stateful) {
        manager.register_pass<ov::pass::MakeStateful>(kv_param_res_pairs);
        // Must run after MakeStateful, which is what creates the ReadValue/Assign pairs.
        if (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT")) {
            manager.register_pass<pass::KVStateSeqAxis>();
        }
    }

    if (traits.is_static) {
        manager.register_pass<pass::SqueezeMatmul>();
    }
}

}  // namespace pipeline
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
