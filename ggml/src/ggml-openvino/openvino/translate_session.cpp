#include "translate_session.hpp"

#include <cstdlib>
#include <map>
#include <memory>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/make_stateful.hpp>

#include "input_model.hpp"

namespace ov {
namespace frontend {
namespace ggml {

using namespace ov::op;

namespace {
ov::pass::MakeStateful::ParamResPairs get_kv_param_res_pairs(
    const std::shared_ptr<ov::Model>& model, const std::map<std::string, std::string>& kv_param_res_names) {
    ov::pass::MakeStateful::ParamResPairs pairs;
    const auto& params = model->get_parameters();
    const auto& results = model->get_results();

    for (const auto& param_res : kv_param_res_names) {
        const auto& param_name = param_res.first;
        const auto& res_name = param_res.second;

        auto param_it = std::find_if(params.begin(), params.end(), [&](const std::shared_ptr<v0::Parameter>& node) {
            return node->get_friendly_name() == param_name;
        });

        OPENVINO_ASSERT(param_it != params.end(), "The tensor name ", param_name,
                        " is not associated with any of "
                        "Parameters in the network.");

        auto res_it = std::find_if(results.begin(), results.end(), [&](const std::shared_ptr<v0::Result>& node) {
            return node->get_friendly_name() == res_name;
        });

        OPENVINO_ASSERT(res_it != results.end(), "The tensor name ", res_name,
                        " is not associated with any of "
                        "Results in the network.");

        std::shared_ptr<ov::op::v0::Parameter> param = *param_it;
        std::shared_ptr<ov::op::v0::Result> res = *res_it;
        pairs.emplace_back(param, res);
    }
    return pairs;
}
}  // namespace

TranslateSession::TranslateSession(const frontend::InputModel::Ptr& input_model,
                                   const std::unordered_map<std::string, CreatorFunction>& translator_map)
    : m_input_model(input_model),
      m_translator_map(translator_map),
      m_ov_model(nullptr) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    if (m_ov_model) {
        return m_ov_model;
    }
    m_ov_model = translate_graph(m_input_model);
    return m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_graph(const frontend::InputModel::Ptr& input_model) {
    ov::ParameterVector params;
    ov::ResultVector results;
    auto tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;

    const auto& ggml_model = std::dynamic_pointer_cast<InputModel>(input_model);
    std::shared_ptr<GgmlDecoder> ggml_model_decoder = ggml_model->get_model_decoder();

    for (const auto& it : ggml_model_decoder->get_model_inputs()) {
        params.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(it.second));
        (*tensor_map)[it.first] = it.second;
    }

    for (const auto& it : ggml_model_decoder->get_model_extra_inputs()) {
        params.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(it.second));
        (*tensor_map)[it.first] = it.second;
    }

    for (const auto& it : ggml_model_decoder->get_model_weights()) {
        (*tensor_map)[it.first] = it.second;
    }

    auto node_visitor = [&](std::shared_ptr<GgmlDecoder> node) {
        auto operation_type = node->get_op_type();
        ov::OutputVector converted_outputs;
        auto it = m_translator_map.find(operation_type);
        FRONT_END_OP_CONVERSION_CHECK(it != m_translator_map.end(),
                                      "Translation for operation type ",
                                      operation_type,
                                      " is not implemented.");
        NodeContext node_context(node, tensor_map, this);
        converted_outputs = it->second(node_context);

        const auto& node_output_names = node->get_output_names();
        FRONT_END_OP_CONVERSION_CHECK(node_output_names.size() == converted_outputs.size(),
                                      "Number of ",
                                      operation_type,
                                      " outputs greater than number of converted outputs, which are ",
                                      node_output_names.size(),
                                      " and ",
                                      converted_outputs.size(),
                                      " respectively.");

        for (size_t i = 0; i < node_output_names.size(); ++i) {
            auto output_name = node_output_names[i];
            if (i < converted_outputs.size() && converted_outputs[i].get_node_shared_ptr() != nullptr) {
                (*tensor_map)[output_name] = converted_outputs[i];
            }
        }
    };

    ggml_model_decoder->visit_subgraph(node_visitor);

    for (const auto& name : ggml_model_decoder->get_model_output_names()) {
        FRONT_END_GENERAL_CHECK(tensor_map->find(name) != tensor_map->end(),
                                "Output name not found in tensor map: ",
                                name);
        auto result = std::make_shared<v0::Result>(tensor_map->at(name));
        result->set_friendly_name(name);
        results.push_back(result);
    }

    resulting_model = std::make_shared<Model>(results, params);

    apply_transformations(resulting_model);
    return resulting_model;
}

void TranslateSession::apply_transformations(const std::shared_ptr<Model>& model) {
    auto ggml_model_decoder = std::dynamic_pointer_cast<InputModel>(m_input_model)->get_model_decoder();

    ov::pass::Manager manager;
    manager.set_per_pass_validation(true);
    manager.register_pass<ov::pass::ConstantFolding>();

    if (!ggml_model_decoder->is_static()) {
        const auto kv_param_res_names = ggml_model_decoder->get_kv_param_res_names();
        const auto kv_param_res_pairs = get_kv_param_res_pairs(model, kv_param_res_names);
        manager.register_pass<ov::pass::MakeStateful>(kv_param_res_pairs);
    }

    manager.run_passes(model);
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
