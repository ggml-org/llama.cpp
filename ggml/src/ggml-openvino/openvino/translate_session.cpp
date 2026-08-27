#include "translate_session.h"

#include "ggml-impl.h"
#include "ggml-openvino/ggml-openvino-extra.h"
#include "ggml-openvino/openvino/node_context.h"
#include "ggml-openvino/openvino/utils.h"
#include "input_model.h"
#include "pass/mark_decompression_convert_constant_folding.h"
#include "pass/mark_dequantization_subgraph.h"
#include "pass/squeeze_matmul.h"
#include "rt_info/weightless_caching_attributes.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/convert_like.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/range.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/strided_slice.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/make_stateful.hpp>
#include <sstream>

namespace ov {
namespace frontend {
namespace ggml {

using namespace ov::op;

namespace {

std::shared_ptr<ov::op::v0::Parameter> create_parameter(const std::string & name,
                                                        const ModelInputInfo & input_info) {
    auto param_node = std::make_shared<ov::op::v0::Parameter>(input_info.type, input_info.shape);
    param_node->set_friendly_name(name);
    param_node->output(0).get_tensor().set_names({name});
    return param_node;
}

std::shared_ptr<ov::Node> create_extra_input(const std::string & name, const ModelExtraInputInfo & input_info) {
    if (input_info.is_parameter) {
        auto param_node = std::make_shared<ov::op::v0::Parameter>(input_info.type, input_info.shape);
        param_node->set_friendly_name(name);
        param_node->output(0).get_tensor().set_names({name});
        return param_node;
    }

    auto constant = std::make_shared<ov::op::v0::Constant>(input_info.type, input_info.shape,
                                                          std::vector<int64_t>{input_info.value});
    constant->set_friendly_name(name);
    return constant;
}

ov::pass::MakeStateful::ParamResPairs get_kv_param_res_pairs(
    const std::shared_ptr<ov::Model> & model,
    const std::map<std::string, std::string> & kv_param_res_names) {
    ov::pass::MakeStateful::ParamResPairs pairs;
    const auto & params = model->get_parameters();
    const auto & results = model->get_results();

    for (const auto & param_res : kv_param_res_names) {
        const auto & param_name = param_res.first;
        const auto & res_name = param_res.second;

        auto param_it = std::find_if(params.begin(), params.end(), [&](const std::shared_ptr<v0::Parameter> & node) {
            return node->get_friendly_name() == param_name;
        });

        OPENVINO_ASSERT(param_it != params.end(), "The tensor name ", param_name,
                        " is not associated with any of "
                        "Parameters in the network.");

        auto res_it = std::find_if(results.begin(), results.end(), [&](const std::shared_ptr<v0::Result> & node) {
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

void add_sliced_mask_stateful(TensorMap & tensor_map) {
    auto create_sliced_mask = [&](const std::string & mask_name, const std::string & sliced_name) {
        if ((tensor_map.find(mask_name) != tensor_map.end()) &&
            (tensor_map.find("token_len_per_seq") != tensor_map.end())) {
            auto token_len_per_seq = tensor_map.at("token_len_per_seq").get_node_shared_ptr();
            auto mask = tensor_map.at(mask_name).get_node_shared_ptr();
            std::shared_ptr<ov::Node> mask_sliced = mask;
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            auto inp_pos = tensor_map.at("inp_pos").get_node_shared_ptr();
            auto last_inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, neg_one, three);
            auto last_inp_pos_1d = std::make_shared<ov::op::v1::Reshape>(
                last_inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {1}), false);
            auto last_inp_pos_cvt = std::make_shared<ov::op::v0::Convert>(last_inp_pos_1d, ov::element::i64);
            auto last_inp_pos_inc = std::make_shared<ov::op::v1::Add>(last_inp_pos_cvt, one);

            mask_sliced = std::make_shared<ov::op::v8::Slice>(mask, zero, last_inp_pos_inc, step, axes);
            mask_sliced = std::make_shared<ov::op::v0::Convert>(mask_sliced, ov::element::f16);
            mask_sliced->set_friendly_name(sliced_name);

            tensor_map.insert({sliced_name, mask_sliced->output(0)});
        }
    };

    create_sliced_mask("self_kq_mask", "KQ_mask_sliced");
    create_sliced_mask("self_kq_mask_swa", "KQ_mask_swa_sliced");
}

void add_rope_sin_cos(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    // When ROPE ops in the graph have divergent op_params (e.g. gemma4's mixed
    // SWA/non-SWA layers with different n_dims or freq_base), a shared sin/cos
    // precompute cannot broadcast across every ROPE use. Skip it here and let
    // translate_rope() build sin/cos per-op from its own op_params.
    if (ggml_model_decoder.has_mixed_rope_params()) {
        return;
    }
    int32_t * rope_params = ggml_model_decoder.get_rope_params();
    if (tensor_map.find("inp_pos") == tensor_map.end() || rope_params == nullptr) {
        return;
    }
    auto inp_pos = tensor_map.at("inp_pos").get_node_shared_ptr();
    std::shared_ptr<ov::Node> rope_freqs_weight;
    if (tensor_map.find("rope_freqs.weight") != tensor_map.end()) {
        rope_freqs_weight = tensor_map.at("rope_freqs.weight").get_node_shared_ptr();
    }

    auto sin_cos = make_sin_cos(rope_params, inp_pos, rope_freqs_weight);
    auto sin_theta = sin_cos.first;
    auto cos_theta = sin_cos.second;

    cos_theta.get_node_shared_ptr()->set_friendly_name("rope_cos");
    sin_theta.get_node_shared_ptr()->set_friendly_name("rope_sin");
    tensor_map.insert({"rope_cos", cos_theta});
    tensor_map.insert({"rope_sin", sin_theta});
}

// Create common patterns
void preprocess(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    if (ggml_model_decoder.is_stateful()) {
        add_sliced_mask_stateful(tensor_map);
    }
    // This optimization is error-prone
    // add_rope_sin_cos(tensor_map, ggml_model_decoder);
}

}  // namespace

TranslateSession::TranslateSession(const frontend::InputModel::Ptr & input_model,
                                   const std::unordered_map<std::string, CreatorFunction> & translator_map,
                                   bool naive) :
    m_input_model(input_model),
    m_translator_map(translator_map),
    m_ov_model(nullptr),
    m_naive(naive) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    if (m_ov_model) {
        return m_ov_model;
    }
    m_ov_model = translate_graph(m_input_model);
    return m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_graph(const frontend::InputModel::Ptr & input_model) {
    ov::ParameterVector params;
    ov::ResultVector results;
    auto tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;

    const auto & ggml_model = std::dynamic_pointer_cast<InputModel>(input_model);
    std::shared_ptr<GgmlDecoder> ggml_model_decoder = ggml_model->get_model_decoder();

    for (const auto & it : ggml_model_decoder->get_model_inputs()) {
        auto param_node = create_parameter(it.first, it.second);
        params.push_back(param_node);
        (*tensor_map)[it.first] = param_node;
    }

    for (const auto & it : ggml_model_decoder->get_model_extra_inputs()) {
        auto input_node = create_extra_input(it.first, it.second);
        if (it.second.is_parameter) {
            params.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(input_node));
        }
        (*tensor_map)[it.first] = input_node;
    }

    for (const auto & it : ggml_model_decoder->get_model_weights()) {
        (*tensor_map)[it.first] = it.second;
    }

    auto translate_node = [&](const std::shared_ptr<GgmlDecoder> & decoder, int node_idx) {
        auto operation_type = decoder->get_op_type(node_idx);
        if (operation_type == "GGML_OP_NONE") {
            return ov::OutputVector{};
        }

        auto it = m_translator_map.find(operation_type);
        FRONT_END_OP_CONVERSION_CHECK(it != m_translator_map.end(), "Translation for operation type ", operation_type,
                                      " is not implemented.");
        NodeContext node_context(decoder, tensor_map, node_idx, this);
        ov::OutputVector converted_outputs = it->second(node_context);

        const auto & node_output_names = decoder->get_output_names(node_idx);
        FRONT_END_OP_CONVERSION_CHECK(node_output_names.size() == converted_outputs.size(), "Number of ",
                                      operation_type, " outputs greater than number of converted outputs, which are ",
                                      node_output_names.size(), " and ", converted_outputs.size(), " respectively.");

        for (size_t i = 0; i < node_output_names.size(); ++i) {
            auto output_name = node_output_names[i];
            if (i < converted_outputs.size() && converted_outputs[i].get_node_shared_ptr() != nullptr) {
                (*tensor_map)[output_name] = converted_outputs[i];
            }
        }
        return converted_outputs;
    };

    // To handle cases like this
    // 3: [ 18432,     1,     1,     1] RESHAPE              cache_r_l0 (reshaped)#3
    //     [ 18432,     1,     1,     1]            0: NONE        cache_r_l0
    // 4: [     0,     1,     1,     1] VIEW                 cache_r_l0 (reshaped) (view)#4
    //     [ 18432,     1,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)#3
    // 5: [     0,     1,     1,     1] SCALE                cache_r_l0 (reshaped) (view) (view)#5
    //     [     0,     1,     1,     1]            0: VIEW        cache_r_l0 (reshaped) (view)#4
    // 6: [     1,     1,     1,     1] VIEW                  (view)#6
    //     [     1,     1,     1,     1]            0: NONE        leaf_5
    // 7: [ 18432,     1,     1,     1] GET_ROWS             conv_states-0#7
    //     [ 18432,     1,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)#3
    //     [     1,     1,     1,     1]            1: VIEW         (view)#6
    // The scale is in-place which modifies cache_r_l0 (reshaped)#3
    // The translation of scale overwrites cache_r in the tensor_map,
    // but we also need to overwrite the old cache_r_l0 (reshaped)#3
    auto refresh_inplace_aliases = [&](const std::shared_ptr<GgmlDecoder> & decoder, int inplace_node_idx,
                                       const std::string & view_src_name) {
        for (int node_idx = 0; node_idx < inplace_node_idx; node_idx++) {
            if (decoder->is_view_like_alias_of(node_idx, view_src_name)) {
                translate_node(decoder, node_idx);
            }
        }
    };

    auto node_visitor = [&](std::shared_ptr<GgmlDecoder> decoder, int node_idx) {
        auto converted_outputs = translate_node(decoder, node_idx);
        if (converted_outputs.empty()) {
            return;
        }
        const auto inplace_src = decoder->get_inplace_op_src(node_idx);
        if (inplace_src.empty()) {
            return;
        }
        if (converted_outputs[0].get_node_shared_ptr() != nullptr) {
            (*tensor_map)[inplace_src] = converted_outputs[0];
        }
        refresh_inplace_aliases(decoder, node_idx, inplace_src);
    };

    if (!m_naive) {
        preprocess(*tensor_map, *ggml_model_decoder);
    }
    ggml_model_decoder->visit_subgraph(node_visitor);

    for (const auto & name : ggml_model_decoder->get_model_output_names()) {
        FRONT_END_GENERAL_CHECK(tensor_map->find(name) != tensor_map->end(),
                                "Output name not found in tensor map: ", name);
        auto result = std::make_shared<v0::Result>(tensor_map->at(name));
        result->set_friendly_name(name);
        results.push_back(result);
    }

    // Debug-only hook: GGML_OPENVINO_DEBUG_NODE=<name1>,<name2>,... adds extra
    // Result nodes for arbitrary intermediate tensors (looked up by name in
    // tensor_map), on top of the real model outputs above. These debug
    // Results are deliberately NOT added to ggml_decoder's model outputs, so
    // the caller (ov_graph_compute_dynamic in utils.cpp) will not bind them
    // to any ggml tensor buffer -- OpenVINO allocates its own tensor for
    // them. This avoids the risk of reading a ggml buffer that has since
    // been overwritten by a later in-place op (ggml aggressively reuses
    // buffers), which can happen if trying to inspect an intermediate value
    // via GGML_OPENVINO_DEBUG_OUTPUT by hacking it into a real output.
    //
    // tensor_map keys are usually the plain ggml tensor name (e.g. "embd"),
    // but tensors that are recomputed multiple times in the same cgraph
    // (GGML_TENSOR_FLAG_COMPUTE) are disambiguated with a "#<hash>" suffix
    // (e.g. "cache_k_l0#4853", see get_tensor_ov_name()) which is not
    // predictable ahead of time. To keep the env var usable, a requested
    // name is matched either exactly, or as the "name" part before "#" of a
    // suffixed key (first match wins; ambiguous requests should include the
    // full "name#hash" form seen in a previous run's log/dump).
    if (const char * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        std::stringstream ss(debug_nodes);
        std::string name;
        while (std::getline(ss, name, ',')) {
            auto it = tensor_map->find(name);
            if (it == tensor_map->end()) {
                it = std::find_if(tensor_map->begin(), tensor_map->end(), [&](const auto & entry) {
                    return entry.first.compare(0, name.size(), name) == 0 && entry.first.size() > name.size() &&
                           entry.first[name.size()] == '#';
                });
            }
            if (it == tensor_map->end()) {
                GGML_LOG_WARN("GGML_OPENVINO_DEBUG_NODE: node '%s' not found in tensor map, skipping\n", name.c_str());
                continue;
            }
            auto result = std::make_shared<v0::Result>(it->second);
            result->set_friendly_name("__debug_" + it->first);
            results.push_back(result);
        }
    }

    ov::ParameterVector used_params;
    for (const auto & param : params) {
        if (!param->output(0).get_target_inputs().empty()) {
            used_params.push_back(param);
        }
    }
    // if (auto diff = params.size() - used_params.size()) {
    //     GGML_LOG_INFO("%zu parameters are not used in the model.", diff);
    // }
    resulting_model = std::make_shared<Model>(results, used_params);

    apply_transformations(resulting_model);

    // Set WeightlessCacheAttribute on large constants to avoid unnecessary memory copies
    // in the NPUW plugin. Without this attribute, NPUW's LazyTensor constructor
    // (lazy_tensor.cpp, op::Const::Const) will memcpy every constant "in case export
    // occurs", doubling memory usage per compile_model call.
    //
    // The bin_offset field serves as a unique key (not a real file offset) — this is
    // the same convention the GPU plugin uses for non-IR models (see
    // Plugin::set_weightless_cache_attributes in intel_gpu/src/plugin/plugin.cpp).
    // Each constant must have a distinct bin_offset, otherwise GPU's weightless cache
    // import will map multiple constants to the same data.
    //
    // Small constants (< 16 elements) are excluded since they may be introduced by
    // optimization patterns and the overhead is negligible.
    //
    // Note: use shape_size() rather than byte_size()/element_type().size() - GatherMatmul's default
    // bias is a Constant(element::dynamic, Shape{0}), whose element_type().size() is 0 and would
    // divide by zero.
    size_t offset = 0;
    for (auto & node : resulting_model->get_ordered_ops()) {
        if (auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(node); cnst && ov::shape_size(cnst->get_shape()) >= 16) {
            auto & rt_info = cnst->get_rt_info();
            if (rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()) == rt_info.end()) {
                rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] =
                    ov::WeightlessCacheAttribute(cnst->get_byte_size(), offset++, cnst->get_element_type());
            }
        }
    }
    return resulting_model;
}

std::shared_ptr<Model> TranslateSession::apply_transformations(std::shared_ptr<Model> model) {
    auto ggml_model_decoder = std::dynamic_pointer_cast<InputModel>(m_input_model)->get_model_decoder();
    {
        ov::pass::Manager manager;
        manager.set_per_pass_validation(true);
        manager.register_pass<ov::pass::MarkCompressedFloatConstants>();
        // Marks the Convert/Subtract/Multiply nodes of our GatherMatmul dequantization chain
        // (make_int4_weights/make_int8_weights, for_gather_matmul=true) with disable_constant_folding,
        // so it survives ConstantFolding regardless of whether the target plugin's own
        // is_decompression_multiply() recognizes GatherMatmul as a valid consumer.
        manager.register_pass<ov::pass::MarkDequantization>(
            std::vector<ov::element::Type>{ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4});

        if (ggml_model_decoder->is_stateful()) {
            const auto kv_param_res_names = ggml_model_decoder->get_kv_param_res_names();
            const auto kv_param_res_pairs = get_kv_param_res_pairs(model, kv_param_res_names);
            manager.register_pass<ov::pass::MakeStateful>(kv_param_res_pairs);
        }

        if (ggml_model_decoder->is_static()) {
            manager.register_pass<pass::SqueezeMatmul>();
        }
        manager.run_passes(model);
        if (ggml_model_decoder->is_stateful()) {
            ov::preprocess::PrePostProcessor ppp(model);
            for (size_t i = 0; i < model->get_output_size(); i++) {
                auto model_output_shape = model->output(i).get_partial_shape();
                if (model_output_shape.rank().is_static() && model_output_shape.rank().get_length() == 3) {
                    ppp.output(i).postprocess().custom([](const ov::Output<ov::Node>& node) {
                        auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
                        return std::make_shared<ov::op::v0::Unsqueeze>(node, axes);
                    });
                }
            }
            model = ppp.build();
        }
    }
    return model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
