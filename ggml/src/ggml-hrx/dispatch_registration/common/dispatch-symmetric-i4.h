#pragma once

#include "../dispatch-registry.h"
#include "ggml.h"

namespace ggml::hrx {

inline bool common_symmetric_i4_lowrow_route_enabled() {
    // Disabled because the route causes a substantial numerical performance regression.
    return false;
}

inline bool common_symmetric_i4_lowrow_mul_mat_eligible(const Graph &     graph,
                                                        const GraphNode & consumer,
                                                        const Value &     input) {
    if (!common_symmetric_i4_lowrow_route_enabled() || !graph.has_index() || consumer.op != GGML_OP_MUL_MAT ||
        consumer.inputs.size() != 2 || consumer.inputs[1] != input.id || input.type != GGML_TYPE_F32 ||
        !input.contiguous || input.ne[0] < 256 || input.ne[0] > 32768 || input.ne[0] % 64 != 0 ||
        input.element_count <= 0 || input.element_count % input.ne[0] != 0) {
        return false;
    }

    const int64_t token_count = input.element_count / input.ne[0];
    if (token_count < 1 || token_count > 16) {
        return false;
    }

    const Value * weight = graph.values().find(consumer.inputs[0]);
    const Value * output = graph.values().find(consumer.output);
    return weight != nullptr && output != nullptr &&
           (weight->type == GGML_TYPE_Q5_K || weight->type == GGML_TYPE_IQ4_XS) &&
           weight->alias_source.value < 0 && weight->contiguous && output->contiguous &&
           output->type == GGML_TYPE_F32 && weight->ne[0] == input.ne[0] && weight->ne[1] == output->ne[0] &&
           output->ne[0] % 64 == 0 && output->ne[1] == token_count && output->ne[2] == 1 && output->ne[3] == 1;
}

}  // namespace ggml::hrx
