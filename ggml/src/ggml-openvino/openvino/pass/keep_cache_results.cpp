#include "keep_cache_results.h"

#include <openvino/core/model.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>

#include <unordered_set>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

bool KeepCacheResults::run_on_model(const std::shared_ptr<ov::Model> & model) {
    bool changed = false;
    // Copies: remove_result()/remove_parameter() modify the vectors these accessors return.
    const ov::ResultVector results = model->get_results();
    for (const auto & result : results) {
        if (m_cache_outputs.count(result->get_friendly_name()) == 0) {
            model->remove_result(result);
            changed = true;
        }
    }
    if (!changed) {
        return false;
    }

    // Removed Results leave their producers unreachable, but those nodes still hold their inputs, so a Parameter
    // is unused when none of its consumers is still part of the model.
    std::unordered_set<const ov::Node *> live;
    for (const auto & op : model->get_ordered_ops()) {
        live.insert(op.get());
    }
    const ov::ParameterVector params = model->get_parameters();
    for (const auto & param : params) {
        bool used = false;
        for (const auto & input : param->output(0).get_target_inputs()) {
            if (live.count(input.get_node()) != 0) {
                used = true;
                break;
            }
        }
        if (!used) {
            model->remove_parameter(param);
        }
    }
    return true;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
