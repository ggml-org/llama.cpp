#include "utils.h"

#include "device/npu_config.h"
#include "device/npu_quant_policy.h"
#include "ggml-impl.h"
#include "ggml-openvino-extra.h"
#include "ggml-openvino.h"
#include "ggml-openvino/ggml-decoder.h"
#include "ggml.h"
#include "model-cache.h"
#include "openvino/frontend.h"
#include "openvino/input_model.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <openvino/core/any.hpp>
#include <openvino/core/graph_util.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/runtime/tensor.hpp>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
// For a KV cache input, return an ov::Tensor sized to n_kv (== attention_size
// for that layer) instead of the fully-allocated ctx_per_seq. Pre-conditions:
//   * dynamic CPU/GPU or opt-in static NPU prefill, single sequence, seq_active_start == 0
//   * ggml KV layout is a contiguous [1, 1, ctx_per_seq, n_heads_kv*head_size]
//     with one token per row (not the transposed V cache of `-fa off`), so the
//     first n_kv rows are the live prefix and shrinking the ctx axis gives a
//     valid tensor over the same host storage
//   * not an SWA layer (ring cache): once the window has wrapped the first
//     n_kv rows no longer contain the live prefix
// On any unmet pre-condition returns std::nullopt; the caller falls back to
// the full-size tensor.
// The KV address the NPU binds: on-demand L0 storage (GGML_OPENVINO_NPU_KV_ON_DEMAND) or the ggml memory itself.
static void * kv_bind_ptr(void * data, const ov::element::Type & type, const ov::Shape & shape) {
    void * storage = ggml_openvino_kv_storage(data, ov::shape_size(shape) * type.size());
    return storage ? storage : data;
}

// Largest on-demand KV storage capacity (rows) among the graph's KV cache tensors.
static int kv_storage_rows(const ggml_cgraph * cgraph) {
    size_t rows = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        for (const auto * src : cgraph->nodes[i]->src) {
            const auto * base = src && src->view_src ? src->view_src : src;
            if (base && (strncmp(base->name, "cache_k_l", 9) == 0 || strncmp(base->name, "cache_v_l", 9) == 0)) {
                rows = std::max(rows, ggml_openvino_kv_storage_capacity(base->data));
            }
        }
    }
    return static_cast<int>(rows);
}

std::optional<ov::Tensor> try_make_kv_sliced_tensor(const std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                                    const std::string & name,
                                                    const ggml_tensor * ggml_tensor) {
    // Static graphs bake the sliced length into the prefill graph; decode is sliced to its KV window
    // with GGML_OPENVINO_NPU_BLOB_PREBUILD and otherwise keeps its full-context graph.
    const bool slice = !ggml_decoder->is_static() ? ggml_openvino_kv_slice_enabled() :
                       ggml_decoder->m_is_prefill ? ggml_openvino_kv_slice_enabled() :
                                                    ggml_openvino_npu_decode_bucket() > 0;
    if (!slice || ggml_decoder->is_stateful()) {
        return std::nullopt;
    }
    if (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != nullptr || !ggml_is_contiguous(ggml_tensor)) {
        return std::nullopt;
    }
    // USAGE_ANY alone is not proof of an attention cache. Only the ordinary
    // llama.cpp K/V roots have this prefix layout; recurrent caches (including
    // rollback snapshots) must always bind their complete stored row bank.
    if (GgmlOvDecoder::is_recurrent_cache(ggml_tensor) ||
        (strncmp(ggml_tensor->name, "cache_k_l", strlen("cache_k_l")) != 0 &&
         strncmp(ggml_tensor->name, "cache_v_l", strlen("cache_v_l")) != 0)) {
        return std::nullopt;
    }

    const auto & compute_params = ggml_decoder->get_compute_params();
    if (compute_params.n_seq_active != 1 || compute_params.seq_active_start != 0) {
        return std::nullopt;
    }

    const int ctx_per_seq = ggml_decoder->get_ctx_per_seq();
    const int n_kv = compute_params.attention_size;
    if (ggml_decoder->is_static() && ggml_openvino_npu_kv_head_major_enabled() && ctx_per_seq > 0) {
        // The static graphs take the whole head-major storage, whose capacity is the attention window.
        const auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);
        if (!GgmlOvDecoder::is_kvcache(ggml_tensor, op) || !ggml_decoder->is_kvcache_token_rows(ggml_tensor)) {
            return std::nullopt;
        }
        const auto layer = extract_layer_from_name(name);
        if (!layer.has_value() || ggml_decoder->is_swa_layer(*layer)) {
            return std::nullopt;
        }
        const auto type = ggml_decoder->get_ov_type(ggml_tensor);
        const size_t heads = ggml_decoder->get_n_heads_kv_for_tensor(ggml_tensor);
        const size_t rows = n_kv > 0 ? std::min(n_kv, ctx_per_seq) : ctx_per_seq;
        const size_t head_size = static_cast<size_t>(ggml_tensor->ne[0]) / heads;
        const ov::Shape shape = name.rfind("cache_v_l", 0) == 0 ?
                                    ov::Shape{1, heads, head_size, rows} :
                                    ov::Shape{1, heads, rows, head_size};
        return ov::Tensor(type, shape, kv_bind_ptr(ggml_tensor->data, type, shape));
    }
    if (ctx_per_seq <= 0 || n_kv <= 0 || n_kv >= ctx_per_seq) {
        return std::nullopt;
    }

    ov::Shape full_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    if (full_shape.size() != 4 || full_shape[0] != 1 || full_shape[1] != 1 ||
        static_cast<int>(full_shape[2]) != ctx_per_seq) {
        return std::nullopt;
    }
    // The consumer lookup scans the whole graph, so it runs after the cheap checks.
    const auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);
    if (!GgmlOvDecoder::is_kvcache(ggml_tensor, op) || !ggml_decoder->is_kvcache_token_rows(ggml_tensor)) {
        return std::nullopt;
    }

    int layer;
    if (auto layer_opt = extract_layer_from_name(name); layer_opt.has_value()) {
        layer = layer_opt.value();
    } else {
        return std::nullopt;
    }

    const bool is_swa = ggml_decoder->is_swa_layer(layer);
    if (is_swa) {
        return std::nullopt;
    }

    if (ggml_tensor->data == nullptr) {
        // unallocated input: let the caller build an owning zero tensor rather than slice a null
        // pointer
        return std::nullopt;
    }

    ov::Shape sliced_shape = full_shape;
    sliced_shape[2] = static_cast<size_t>(n_kv);

    const auto type = GgmlOvDecoder::get_ov_type(ggml_tensor);
    // On-demand KV storage moves when it grows, so it bypasses the per-pointer tensor cache.
    if (void * storage = kv_bind_ptr(ggml_tensor->data, type, sliced_shape); storage != ggml_tensor->data) {
        return ov::Tensor(type, sliced_shape, storage);
    }
    return ggml_decoder->cached_kv_tensor(ggml_tensor, type, sliced_shape, [&]() -> ov::Tensor {
        if (ggml_openvino_buffer_is_remote(ggml_tensor)) {
            auto remote_context = ggml_openvino_get_remote_context(ggml_decoder->device);
            auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
            return gpu_context.create_tensor(type, sliced_shape, ggml_tensor->data);
        }
        return ov::Tensor(type, sliced_shape, ggml_tensor->data);
    });
}

// For a KV cache in the 4-D layout (GgmlOvDecoder::is_kvcache_4d), a [1, ctx, n_heads_kv, head_size] tensor over the
// whole cache. Its shape stays the same from one inference to the next, so the GPU plugin keeps writing the cache in
// place and reading it without a copy.
std::optional<ov::Tensor> try_make_kv_4d_tensor(const std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                                const ggml_tensor * ggml_tensor) {
    if (!ggml_decoder->is_kvcache_4d_input(ggml_tensor)) {
        return std::nullopt;
    }
    const size_t n_heads_kv = static_cast<size_t>(ggml_decoder->get_n_heads_kv_for_tensor(ggml_tensor));
    const ov::Shape shape{1, static_cast<size_t>(ggml_tensor->ne[1]), n_heads_kv,
                          static_cast<size_t>(ggml_tensor->ne[0]) / n_heads_kv};
    const auto type = GgmlOvDecoder::get_ov_type(ggml_tensor);
    return ggml_decoder->cached_kv_tensor(ggml_tensor, type, shape, [&]() -> ov::Tensor {
        if (ggml_openvino_buffer_is_remote(ggml_tensor)) {
            auto remote_context = ggml_openvino_get_remote_context(ggml_decoder->device);
            auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
            return gpu_context.create_tensor(type, shape, ggml_tensor->data);
        }
        return ov::Tensor(type, shape, ggml_tensor->data);
    });
}

// Column count of a static KQ mask input; must match GgmlOvDecoder::get_graph_input_shape().
static size_t get_static_attention_size(const std::shared_ptr<GgmlOvDecoder> & ggml_decoder, const std::string & name) {
    const bool is_swa = ggml_decoder->is_swa_mask(ggml_decoder->get_input_ggml_tensor(name));
    if (ggml_decoder->m_is_prefill ? ggml_openvino_kv_slice_enabled() :
                                     ggml_openvino_npu_decode_bucket() > 0) {
        const auto & params = ggml_decoder->get_compute_params();
        const int attention_size = is_swa ? params.attention_size_swa : params.attention_size;
        if (attention_size > 0) {
            return static_cast<size_t>(attention_size);
        }
    }
    if (is_swa) {
        const auto & params = ggml_decoder->get_model_params();
        return static_cast<size_t>(params.ctx_per_seq_swa * params.n_seq);
    }
    return static_cast<size_t>(ggml_decoder->get_ctx_size());
}

static uint64_t ggml_openvino_model_cache_extra_cfg(bool stateful, bool recurrent) {
    const bool manual_gqa_enabled = ggml_openvino_manual_gqa_enabled();

    uint64_t extra_cfg = 15;  // Recurrent caches are writable inputs without cache Results.
#ifdef GGML_OPENVINO_HAS_CACHE_UPDATE
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_FUSED_RECURRENCE") ? 0u : 1u);
#else
    extra_cfg = extra_cfg * 131;
#endif
    extra_cfg = extra_cfg * 131 + (stateful ? (recurrent ? 2u : 1u) : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_reduce_compile_mem_enabled() ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_KV_4D") > 0 ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (manual_gqa_enabled ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_shape_from_mask_enabled() ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_npu_hfa_prefill_enabled() ? 4u : 0u);  // HFA input ABI v4.
    extra_cfg = extra_cfg * 131 + (ggml_openvino_npu_host_gather_enabled() ? 1u : 0u);
    if (ggml_openvino_is_npu()) {
        extra_cfg = extra_cfg * 131 + static_cast<uint64_t>(ggml_openvino_get_npu_requant_type());
    }
    return extra_cfg;
}

std::map<std::string, std::shared_ptr<ov::Node>> get_weight_names(ggml_cgraph * cgraph) {
    std::map<std::string, std::shared_ptr<ov::Node>> names;
    for (const auto & name : GgmlOvDecoder::collect_weight_names(cgraph)) {
        names[name] = nullptr;
    }
    return names;
}

// A conservative, exact in-process key, evaluated only on a context-local cache
// miss. Include topology, layouts, op parameters, constant extra inputs and weight
// allocation identities. Never use a sampled weight hash or a graph name alone:
// different models can have identical topology. OV buffer IDs survive address reuse.
static std::string compiled_graph_key(const ggml_cgraph * graph,
                                      const GgmlOvDecoder & decoder,
                                      const std::string & device,
                                      int prefill_chunk_size = 0,
                                      bool disk_cache = false) {
    std::string key;
    auto append = [&key](const auto & value) {
        key.append(reinterpret_cast<const char *>(&value), sizeof(value));
    };
    auto append_string = [&](const std::string & value) {
        append(value.size());
        key.append(value);
    };
    append_string(device);
    append(decoder.is_static());
    append(decoder.is_stateful());
    append(prefill_chunk_size);
    append(ggml_openvino_npu_hfa_prefill_enabled());
    bool has_weight_buffer_id = false;
    std::unordered_map<const ggml_tensor *, size_t> ids;
    std::function<void(const ggml_tensor *)> visit = [&](const ggml_tensor * tensor) {
        if (!tensor) {
            append(size_t(0));
            return;
        }
        auto inserted = ids.emplace(tensor, ids.size() + 1);
        append(inserted.first->second);
        if (!inserted.second) {
            return;
        }
        append_string(tensor->name);
        append(tensor->type);
        append(tensor->op);
        append(tensor->flags);
        append(tensor->ne);
        append(tensor->nb);
        append(tensor->op_params);
        append(tensor->view_offs);
        const auto * base = tensor->view_src ? tensor->view_src : tensor;
        const bool weight = base->buffer && base->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        append(weight);
        if (weight && !disk_cache) {
            const size_t buffer_id = ggml_backend_openvino_buffer_get_ctx_id(base->buffer);
            has_weight_buffer_id |= buffer_id != 0;
            append(buffer_id);
            append(tensor->data);
        }
        visit(tensor->view_src);
        for (const auto * src : tensor->src) {
            visit(src);
        }
    };
    append(graph->n_nodes);
    for (int i = 0; i < graph->n_nodes; ++i) {
        visit(graph->nodes[i]);
    }
    append(graph->n_leafs);
    for (int i = 0; i < graph->n_leafs; ++i) {
        visit(graph->leafs[i]);
    }
    for (const auto & input : decoder.get_model_extra_inputs()) {
        append_string(input.first);
        append_string(input.second.type.get_type_name());
        append(input.second.shape.size());
        for (auto dim : input.second.shape) {
            append(dim);
        }
        append(input.second.is_parameter);
        if (!input.second.is_parameter) {
            append(input.second.value);
        }
        append_string(input.second.shape_source);
        append(input.second.shape_axis);
    }
    // Without an allocation generation, pointer reuse could select stale weights.
    // Such graphs still get private requests; they simply do not share compilation.
    return disk_cache || has_weight_buffer_id ? key : std::string{};
}

static std::string dynamic_graph_signature(const ggml_cgraph * graph,
                                           const GgmlOvDecoder & decoder,
                                           const ModelParams & params) {
    std::string key = "dynamic-1";
    auto append = [&key](const auto & value) {
        key.append(reinterpret_cast<const char *>(&value), sizeof(value));
    };
    auto append_string = [&](const std::string & value) {
        append(value.size());
        key.append(value);
    };
    graph_key graph_id(graph, true);
    append(graph_id.n_nodes);
    append(graph_id.n_leaves);
    append_string(graph_id.first_node_name);
    append_string(graph_id.last_node_name);
    for (const auto & name : graph_id.input_srcs) {
        append_string(name);
    }
    append(params.n_rs_slots);
    append(params.has_rs_rollback);
    append(params.mixed_rope_params);
    append(params.is_cacheless_attn);
    for (int layer : params.swa_layers) {
        append(layer);
    }
    for (const auto & [layer, heads] : params.n_heads_kv_per_layer) {
        append(layer);
        append(heads);
    }
    for (const auto & [name, input] : decoder.get_model_inputs()) {
        append_string(name);
        append_string(input.type.get_type_name());
    }
    for (const auto & [name, tensor] : decoder.get_model_outputs()) {
        append_string(name);
        append(tensor->type);
    }
    for (const auto & [name, input] : decoder.get_model_extra_inputs()) {
        append_string(name);
        append_string(input.type.get_type_name());
        append(input.shape.size());
        for (auto dim : input.shape) {
            append(dim);
        }
        append(input.is_parameter);
        if (!input.is_parameter) {
            append(input.value);
        }
        append_string(input.shape_source);
        append(input.shape_axis);
    }
    return key;
}

// The CPU plugin exports its MoE op (GatherMatmulCompressed) but cannot import it again ("unsupported opset",
// OpenVINO 2026.4 and 2026.5). Writing such a blob makes every later run fail the import and compile again.
static bool ggml_openvino_cpu_blob_importable(const std::string & device, const std::shared_ptr<ov::Model> & model) {
    if (device != "CPU") {
        return true;
    }
    for (const auto & op : model->get_ops()) {
        if (std::string(op->get_type_name()).rfind("GatherMatmul", 0) == 0) {
            return false;
        }
    }
    return true;
}

static bool compiled_model_matches_graph(const ov::CompiledModel & model,
                                         const std::vector<std::string> & inputs,
                                         const std::vector<std::string> & outputs,
                                         const GgmlOvDecoder & decoder) {
    if (model.inputs().size() != inputs.size() || model.outputs().size() != outputs.size()) {
        return false;
    }
    const auto & graph_inputs = decoder.get_model_inputs();
    const auto & extra_inputs = decoder.get_model_extra_inputs();
    const auto & graph_outputs = decoder.get_model_outputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto & port = model.input(i);
        auto graph_it = graph_inputs.find(inputs[i]);
        if (graph_it != graph_inputs.end()) {
            if (port.get_element_type() != graph_it->second.type ||
                !port.get_partial_shape().compatible(graph_it->second.shape)) {
                return false;
            }
            continue;
        }
        auto extra_it = extra_inputs.find(inputs[i]);
        if (extra_it == extra_inputs.end() || port.get_element_type() != extra_it->second.type ||
            !port.get_partial_shape().compatible(ov::PartialShape(extra_it->second.shape))) {
            return false;
        }
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto graph_it = graph_outputs.find(outputs[i]);
        if (graph_it == graph_outputs.end()) {
            if (outputs[i].compare(0, 8, "__debug_") != 0) {
                return false;
            }
            continue;
        }
        const auto & port = model.output(i);
        const auto * tensor = graph_it->second;
        auto expected_shape = GgmlOvDecoder::get_shape(tensor);
        if (decoder.is_kvcache_4d_input(tensor)) {
            // Cache Results use the same head-split layout as their Parameters.
            const auto heads = static_cast<size_t>(decoder.get_n_heads_kv_for_tensor(tensor));
            expected_shape = {1, static_cast<size_t>(tensor->ne[1]), heads,
                              static_cast<size_t>(tensor->ne[0]) / heads};
        }
        if (port.get_element_type() != GgmlOvDecoder::get_ov_type(graph_it->second) ||
            !port.get_partial_shape().compatible(ov::PartialShape(expected_shape))) {
            return false;
        }
    }
    return true;
}

static std::string ov_profiling_csv_field(const std::string & value) {
    std::string escaped = "\"";
    for (char c : value) {
        escaped += c;
        if (c == '"') {
            escaped += c;
        }
    }
    escaped += '"';
    return escaped;
}

static void dump_ov_profiling_info(const ov::InferRequest & infer_request) {
    if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") < 2) {
        return;
    }

    static std::atomic<uint64_t> inference_index{0};
    const uint64_t index = inference_index.fetch_add(1);
    const std::string path = "openvino_profile_" + std::to_string(index) + ".csv";
    std::ofstream output(path, std::ios::trunc);
    if (!output.is_open()) {
        GGML_LOG_WARN("ggml-openvino: failed to write profiling data to %s\n", path.c_str());
        return;
    }

    output << "status,real_time_us,cpu_time_us,start_time_us,node_type,node_name,exec_type\n";
    int64_t total_real_time = 0;
    int64_t total_cpu_time = 0;
    size_t executed_count = 0;
    for (const auto & info : infer_request.get_profiling_info()) {
        const char * status = "NOT_RUN";
        if (info.status == ov::ProfilingInfo::Status::EXECUTED) {
            status = "EXECUTED";
            total_real_time += info.real_time.count();
            total_cpu_time += info.cpu_time.count();
            executed_count++;
        } else if (info.status == ov::ProfilingInfo::Status::OPTIMIZED_OUT) {
            status = "OPTIMIZED_OUT";
        }

        output << status << ',' << info.real_time.count() << ',' << info.cpu_time.count() << ','
               << info.start_time.count() << ',' << ov_profiling_csv_field(info.node_type) << ','
               << ov_profiling_csv_field(info.node_name) << ',' << ov_profiling_csv_field(info.exec_type) << '\n';
    }

    GGML_LOG_INFO("ggml-openvino: profile %s: %zu executed nodes, %.3f ms device, %.3f ms CPU\n", path.c_str(),
                  executed_count, total_real_time / 1000.0, total_cpu_time / 1000.0);
}

static uint64_t ggml_openvino_model_cache_mix_string(uint64_t hash, const char * value) {
    if (!value) {
        return hash * 131;
    }
    while (*value) {
        hash = hash * 131 + static_cast<unsigned char>(*value++);
    }
    return hash * 131 + 1;
}

ov::Tensor create_ov_output_tensor(const std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                   const std::shared_ptr<ov::InferRequest> & infer_request,
                                   int output_index,
                                   const ggml_tensor * ggml_tensor) {
    if (auto kv_4d = try_make_kv_4d_tensor(ggml_decoder, ggml_tensor)) {
        return *kv_4d;
    }
    if (auto sliced = try_make_kv_sliced_tensor(ggml_decoder, std::string(ggml_tensor->name), ggml_tensor)) {
        return *sliced;
    }

    auto output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    void * output_data = ggml_tensor->data;
    if (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_index).get_shape();
        output_data = kv_bind_ptr(output_data, output_type, output_shape);
    } else {
        // For a CPY into a padded view_src (e.g. a padded KV cache buffer), the
        // OV ScatterUpdate node outputs the full view_src shape, not the CPY node's
        // own (smaller) shape.  Using the CPY shape here causes set_output_tensor to
        // fail with a shape-incompatibility error.  Use view_src's shape and data
        // pointer instead so the OV tensor matches the model output exactly.
        if (ggml_tensor->op == GGML_OP_CPY && ggml_tensor->view_src != nullptr &&
            ggml_nbytes(ggml_tensor) != ggml_nbytes(ggml_tensor->view_src)) {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
            output_data = ggml_tensor->view_src->data;
        } else {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor);
            if (ggml_tensor->op == GGML_OP_GATED_DELTA_NET && ggml_tensor->op_params[0] > 1) {
                const auto * value = ggml_tensor->src[2];
                const int64_t valid_count = std::min<int64_t>(value->ne[2], ggml_tensor->op_params[0]);
                output_shape[2] = (value->ne[2] + valid_count * value->ne[0]) * value->ne[3];
            }
        }
    }

    // Reuse the caller-owned full-cache binding when its shape and storage match.
    if (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS") && !ggml_decoder->is_static() &&
        !ggml_decoder->is_splited_model() && ggml_openvino_buffer_is_remote(ggml_tensor)) {
        auto resolved = output_data;
        if (ggml_tensor->extra != nullptr) {
            auto * base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
            if (base->type == ggml_openvino_extra_base::Type::TENSOR) {
                auto & cached = static_cast<ggml_openvino_tensor_extra *>(base)->tensor;
                if (cached && cached->get_element_type() == output_type && cached->get_shape() == output_shape) {
                    auto remote = cached->as<ov::intel_gpu::ocl::USMTensor>();
                    if (remote.get() == resolved) {
                        return *cached;
                    }
                }
            }
        }
        auto context = ggml_openvino_get_remote_context(ggml_decoder->device)->as<ov::intel_gpu::ocl::ClContext>();
        return context.create_tensor(output_type, output_shape, resolved);
    }

    // A graph output can also have no backing storage. Give OV an owning tensor so the infer request
    // has somewhere to write; nothing downstream reads it.
    if (output_data == nullptr) {
        return ov::Tensor(output_type, output_shape);
    }
    ov::Tensor output_tensor(output_type, output_shape, output_data);
    return output_tensor;
}

// Rewrite ggml's KV rows into a relayout state that keeps the sequence on dim 2.
// ggml stores [seq][n_heads_kv * head_size]; the state wants [1, n_heads_kv, seq, head_size],
// a different element order, so the rows are copied instead of reinterpreted.
ov::Tensor kv_rows_to_seq_axis_2(const ov::Tensor & kv_tensor, size_t n_heads_kv) {
    const size_t rows = kv_tensor.get_shape()[2];
    const size_t head_size = kv_tensor.get_shape()[3] / n_heads_kv;
    const size_t elem = kv_tensor.get_element_type().size();
    const size_t head_bytes = head_size * elem;

    ov::Tensor out(kv_tensor.get_element_type(), ov::Shape{1, n_heads_kv, rows, head_size});
    const auto * src = static_cast<const uint8_t *>(kv_tensor.data());
    auto * dst = static_cast<uint8_t *>(out.data());
    for (size_t s = 0; s < rows; s++) {
        for (size_t h = 0; h < n_heads_kv; h++) {
            memcpy(dst + (h * rows + s) * head_bytes, src + (s * n_heads_kv + h) * head_bytes, head_bytes);
        }
    }
    return out;
}

template <typename T> void set_zero_diagonal(std::vector<T> & matrix, size_t rows, size_t cols, T zero_value = T{}) {
    for (size_t i = 0; i < rows; ++i) {
        size_t diag_col = std::min(i, cols - 1);
        matrix[i * cols + diag_col] = zero_value;
    }
}

// build the padded causal mask directly in the destination buffer
template <typename T>
void fill_prefill_mask(T * dst, const T * src, size_t valid_rows, size_t src_cols, size_t padded_rows,
                       size_t padded_cols, T pad_value, T zero_value) {
    for (size_t i = 0; i < padded_rows; ++i) {
        T * row = dst + i * padded_cols;
        if (i < valid_rows) {
            const size_t ncopy = std::min(src_cols, padded_cols);
            std::memcpy(row, src + i * src_cols, ncopy * sizeof(T));
            std::fill(row + ncopy, row + padded_cols, pad_value);
        } else {
            std::fill(row, row + padded_cols, pad_value);
        }
        row[std::min(i, padded_cols - 1)] = zero_value;
    }
}

ov::Tensor make_contiguous_split_input_tensor(const struct ggml_tensor * ggml_tensor, const ov::Shape & input_shape) {
    const size_t element_size = ggml_type_size(ggml_tensor->type);
    const size_t block_size = ggml_blck_size(ggml_tensor->type);

    GGML_ASSERT(block_size == 1 && "non-contiguous split inputs must be plain element types");

    const struct ggml_tensor * source_tensor = ggml_tensor->view_src != nullptr ? ggml_tensor->view_src : ggml_tensor;
    const size_t source_offset = ggml_tensor->view_src != nullptr ? ggml_tensor->view_offs : 0;

    std::vector<uint8_t> source_data(ggml_nbytes(source_tensor));
    ggml_backend_tensor_get(source_tensor, source_data.data(), 0, source_data.size());

    ov::Tensor input_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), input_shape);
    auto * dst = static_cast<uint8_t *>(input_tensor.data());
    size_t dst_offset = 0;

    for (size_t i3 = 0; i3 < static_cast<size_t>(ggml_tensor->ne[3]); ++i3) {
        for (size_t i2 = 0; i2 < static_cast<size_t>(ggml_tensor->ne[2]); ++i2) {
            for (size_t i1 = 0; i1 < static_cast<size_t>(ggml_tensor->ne[1]); ++i1) {
                for (size_t i0 = 0; i0 < static_cast<size_t>(ggml_tensor->ne[0]); ++i0) {
                    const size_t src_offset = source_offset + i3 * ggml_tensor->nb[3] + i2 * ggml_tensor->nb[2] +
                                              i1 * ggml_tensor->nb[1] + i0 * ggml_tensor->nb[0];
                    std::memcpy(dst + dst_offset, source_data.data() + src_offset, element_size);
                    dst_offset += element_size;
                }
            }
        }
    }

    return input_tensor;
}

ov::Tensor convert_ggml_input_to_ov(const std::shared_ptr<GgmlOvDecoder> & ggml_decoder, const std::string & name) {
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(name);

    if (name == "rs_active_rows") {
        const auto count = ggml_decoder->get_compute_params().s_copy_active_slot_len;
        GGML_ASSERT(count > 0 && count <= ggml_nelements(ggml_tensor));
        return ov::Tensor(ov::element::i32, ov::Shape{1, 1, 1, size_t(count)}, ggml_tensor->data);
    }

    if (auto kv_4d = try_make_kv_4d_tensor(ggml_decoder, ggml_tensor)) {
        return *kv_4d;
    }
    if (auto sliced = try_make_kv_sliced_tensor(ggml_decoder, name, ggml_tensor)) {
        return *sliced;
    }

    // KQ mask with fewer columns than the rounded KV length (see compute_llm_params): pad it with -inf
    const auto & compute_params = ggml_decoder->get_compute_params();
    if (!ggml_decoder->is_static() && name == compute_params.mask_name &&
        compute_params.attention_size > ggml_tensor->ne[0] && ggml_is_contiguous(ggml_tensor)) {
        const size_t rows = ggml_nrows(ggml_tensor);
        const size_t cols = ggml_tensor->ne[0];
        const size_t padded_cols = compute_params.attention_size;
        ov::Shape shape = GgmlOvDecoder::get_shape(ggml_tensor);
        shape.back() = padded_cols;
        ov::Tensor padded(GgmlOvDecoder::get_ov_type(ggml_tensor), shape);
        auto pad = [&](auto fill) {
            using T = decltype(fill);
            auto data = pad_input<T>((const T *) ggml_tensor->data, rows, cols, rows, padded_cols, fill);
            std::memcpy(padded.data(), data.data(), padded.get_byte_size());
        };
        if (ggml_tensor->type == GGML_TYPE_F16) {
            pad(GGML_FP32_TO_FP16(-INFINITY));
        } else {
            GGML_ASSERT(ggml_tensor->type == GGML_TYPE_F32);
            pad(-INFINITY);
        }
        return padded;
    }

    if (ggml_decoder->is_static() &&
        GgmlOvDecoder::is_kvcache(ggml_tensor, ggml_decoder->get_tensor_used_op(ggml_tensor))) {
        const auto type = ggml_decoder->get_ov_type(ggml_tensor);
        const auto shape = ggml_decoder->get_shape(ggml_tensor);
        if (void * storage = ggml_openvino_kv_storage(ggml_tensor->data, ov::shape_size(shape) * type.size())) {
            return ov::Tensor(type, shape, storage);
        }
    }

    if (ggml_tensor->extra != nullptr && !ggml_decoder->is_splited_model()) {
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        if (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
            // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
            auto * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
            return *tensor_extra->tensor;
        }
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    auto * input_data = ggml_tensor->data;
    ov::Shape input_shape;
    if (ggml_tensor->op == GGML_OP_VIEW && !ggml_decoder->is_splited_model()) {
        // This case is added to make test-backend-ops work
        input_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
    } else {
        input_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    }

    if (ggml_decoder->is_splited_model() && !ggml_is_contiguous(ggml_tensor)) {
        return make_contiguous_split_input_tensor(ggml_tensor, input_shape);
    }

    // A graph input may legitimately be left unallocated (data == nullptr, no buffer): llama.cpp
    // skips filling the attention mask when a graph only stores K/V without attending, which is what
    // a speculative KV-injection pass does. Such an input is never read, but OpenVINO still requires
    // a backing tensor for the Parameter and constructing an ov::Tensor over a null pointer asserts.
    // Hand OV an owning zero-filled tensor; its contents cannot affect the result because no op
    // consumes them.
    if (input_data == nullptr) {
        GGML_LOG_WARN("ggml-openvino: unallocated INPUT '%s' (ggml name '%s', type %s, ne=[%ld,%ld,%ld,%ld]) "
                      "-> zero tensor\n",
                      name.c_str(), ggml_tensor->name, ggml_type_name(ggml_tensor->type),
                      (long) ggml_tensor->ne[0], (long) ggml_tensor->ne[1], (long) ggml_tensor->ne[2],
                      (long) ggml_tensor->ne[3]);
        return ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), input_shape);
    }
    auto input_tensor = ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), input_shape, input_data);
    return input_tensor;
}

ov::Tensor get_ov_input_tensor(const std::shared_ptr<GgmlOvDecoder> & ggml_decoder, const std::string & param_name) {
    ov::Tensor input_tensor;
    auto extra_input = ggml_decoder->get_model_extra_inputs().find(param_name);
    if (extra_input != ggml_decoder->get_model_extra_inputs().end()) {
        auto & cached = ggml_decoder->runtime_input_tensors[param_name];
        if (!cached || cached.get_element_type() != extra_input->second.type ||
            cached.get_shape() != extra_input->second.shape) {
            if (ggml_openvino_is_gpu() && !ggml_decoder->is_static() && !ggml_decoder->is_stateful()) {
                auto context =
                    ggml_openvino_get_remote_context(ggml_decoder->device)->as<ov::intel_gpu::ocl::ClContext>();
                auto storage = context.create_usm_host_tensor(extra_input->second.type, extra_input->second.shape);
                cached = ov::Tensor(extra_input->second.type, extra_input->second.shape, storage.get());
                ggml_decoder->runtime_input_storage[param_name] = storage;
            } else {
                cached = ov::Tensor(extra_input->second.type, extra_input->second.shape);
            }
        }
        input_tensor = cached;
        if (param_name == "sampling_rows") {
            auto * data = input_tensor.data<int64_t>();
            for (const auto & row : ggml_decoder->get_compute_params().sampling_rows) {
                *data++ = row.second;
            }
        } else if (param_name.compare(0, 11, "rs_offsets_") == 0 || param_name.compare(0, 11, "rs_windows_") == 0) {
            const auto & controls = ggml_decoder->get_compute_params();
            const auto & values = controls.recurrent_controls.at(param_name);
            std::copy(values.begin(), values.end(), input_tensor.data<int64_t>());
        } else if (param_name.compare(0, 10, "rs_update_") == 0) {
            const auto & controls = ggml_decoder->get_compute_params();
            const auto & write = controls.rs_writebacks.at(param_name.substr(10));
            auto data = input_tensor.data<int64_t>();
            data[0] = write.slot_begin;
            data[1] = write.seq_count >= 0 ? write.seq_count : controls.s_copy_active_slot_len;
            data[2] = write.snapshot_stride >= 0 ? write.snapshot_stride : data[1];
            // Direct GDN states contain no attention prefix.
            data[3] = 0;
        } else {
            if (extra_input->second.type == ov::element::i64) {
                *input_tensor.data<int64_t>() = extra_input->second.value;
            } else {
                std::memset(input_tensor.data(), 0, input_tensor.get_byte_size());
            }
        }
    } else {
        input_tensor = convert_ggml_input_to_ov(ggml_decoder, param_name);
    }
    return input_tensor;
}

// Rows |first|..|first + count| of a host-gathered table (GGML_OPENVINO_NPU_HOST_GATHER), dequantized from the raw
// ggml table into a [1, 1, padded, row] f32 tensor. Padded rows repeat the last valid row; their outputs are unused.
static ov::Tensor make_host_rows_tensor(const ggml_tensor * node, size_t first, size_t count, size_t padded) {
    const ggml_tensor * table = node->src[0];
    const auto * ids = static_cast<const int32_t *>(node->src[1]->data);
    const size_t row = table->ne[0];
    const auto to_float = ggml_get_type_traits(table->type)->to_float;
    ov::Tensor rows(ov::element::f32, ov::Shape{1, 1, padded, row});
    auto * dst = rows.data<float>();
    for (size_t i = 0; i < padded; i++) {
        const int64_t id = ids[first + std::min(i, count - 1)];
        to_float(static_cast<const char *>(table->data) + id * table->nb[1], dst + i * row, static_cast<int64_t>(row));
    }
    return rows;
}

// Input of the static decode model for token |row| of the graph: row 0 of a decode graph, or a row of a prefill graph
// whose outputs a KV-only prefill computes with the decode model (the decoder is then in decode mode, so that the KV
// cache and the masks bind at full context).
ov::Tensor get_ov_input_tensor_static_decode(const std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                             const std::string & param_name,
                                             int64_t row = 0) {
    if (const auto * node = ggml_decoder->get_host_gather_rows(param_name)) {
        return make_host_rows_tensor(node, row, 1, 1);
    }
    // NPU decoding stage
    if (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        return get_ov_input_tensor(ggml_decoder, param_name);
    }
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(param_name);
    const auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    if (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        // IMROPE's inp_pos holds one plane per t/h/w/e, each as long as the graph has tokens; take the row of each plane.
        const int n_planes = GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ? GgmlOvDecoder::get_inp_pos_n_planes(op) : 1;
        const int64_t n_rows = ggml_tensor->ne[0] / n_planes;
        const size_t element_size = ggml_type_size(ggml_tensor->type);
        ov::Tensor input_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ov::Shape{1, 1, 1, (size_t) n_planes});
        for (int p = 0; p < n_planes; p++) {
            std::memcpy(static_cast<char *>(input_tensor.data()) + p * element_size,
                        static_cast<const char *>(ggml_tensor->data) + (p * n_rows + row) * element_size, element_size);
        }
        return input_tensor;
    }

    if (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        // The decode model computes the outputs of its one token. out_ids is empty when the graph has no output, so it
        // is not read.
        ov::Tensor input_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ov::Shape{1, 1, 1, 1});
        *input_tensor.data<int32_t>() = 0;
        return input_tensor;
    }

    if (GgmlOvDecoder::is_inp_scale_rows(ggml_tensor, op) && ggml_nelements(ggml_tensor) > 1) {
        // One scale per token of the graph.
        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, 1, 1});
        *input_tensor.data<float>() = static_cast<const float *>(ggml_tensor->data)[row];
        return input_tensor;
    }

    if (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        const size_t context_size = get_static_attention_size(ggml_decoder, param_name);
        const size_t cols = ggml_tensor->ne[0];
        if (ggml_tensor->type == GGML_TYPE_F16) {
            const auto * src = static_cast<const ggml_fp16_t *>(ggml_tensor->data) + row * cols;
            std::vector<ggml_fp16_t> padded_data =
                pad_input<ggml_fp16_t>(src, 1, cols, 1, context_size, GGML_FP32_TO_FP16(-INFINITY));
            ov::Tensor input_tensor(ov::element::f16, ov::Shape{1, 1, 1, context_size});
            std::memcpy(input_tensor.data(), padded_data.data(), padded_data.size() * sizeof(ggml_fp16_t));
            return input_tensor;
        }

        const auto * src = static_cast<const float *>(ggml_tensor->data) + row * cols;
        std::vector<float> padded_data = pad_input<float>(src, 1, cols, 1, context_size, -INFINITY);
        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_size});
        std::memcpy(input_tensor.data(), padded_data.data(), padded_data.size() * sizeof(float));
        return input_tensor;
    }

    return get_ov_input_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_input_tensor_static_prefill(const std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                              const std::string & param_name,
                                              int chunk_index) {
    // NPU prompt processing stage
    const size_t input_len = ggml_decoder->get_input_len();
    const size_t chunk_size = ggml_decoder->m_prefill_chunk_size;
    const size_t chunk_valid_size = std::min(chunk_size, input_len - chunk_index * chunk_size);
    const size_t chunk_pad_size = chunk_size - chunk_valid_size;

    if (param_name == "chunk_valid_len") {
        ov::Tensor input_tensor(ov::element::i64, ov::Shape{1});
        *input_tensor.data<int64_t>() = (int64_t) chunk_valid_size;
        return input_tensor;
    }
    if (param_name.rfind("ggml_hfa_past_", 0) == 0) {
        const std::string cache_name = param_name.substr(strlen("ggml_hfa_past_"));
        const ggml_tensor * kv_tensor = nullptr;
        for (const auto & [name, _] : ggml_decoder->get_model_inputs()) {
            if (name.substr(0, name.find('#')) == cache_name) {
                kv_tensor = ggml_decoder->get_input_ggml_tensor(name);
                break;
            }
        }
        if (kv_tensor == nullptr) {
            throw std::runtime_error("HFA past KV source not found for " + param_name);
        }
        const auto & info = ggml_decoder->get_model_extra_inputs().at(param_name);
        // Shape is [1, past, heads, head_size], or [1, heads, past, head_size] for head-major KV
        // ([1, heads, head_size, past] for transposed V).
        const bool head_major = ggml_openvino_npu_kv_head_major_enabled();
        const bool v_transposed = head_major && cache_name.rfind("cache_v_l", 0) == 0;
        const size_t past_capacity = v_transposed ? info.shape[3] : head_major ? info.shape[2] : info.shape[1];
        const size_t heads = head_major ? info.shape[1] : info.shape[2];
        const size_t head_size = v_transposed ? info.shape[2] : info.shape[3];
        const size_t row_bytes = heads * head_size * ggml_type_size(kv_tensor->type);
        if (static_cast<size_t>(kv_tensor->ne[0]) != heads * head_size || kv_tensor->nb[1] != row_bytes) {
            throw std::runtime_error("HFA past KV source shape is incompatible with " + param_name);
        }
        if (past_capacity <= static_cast<size_t>(kv_tensor->ne[1])) {
            // Zero-copy: the first past_capacity cache rows; HFA only reads the rows of the explicit history.
            return ov::Tensor(info.type, info.shape, kv_bind_ptr(kv_tensor->data, info.type, info.shape));
        }
        // Single-chunk prompt in a cache shorter than one chunk: history is empty, share one zero tensor.
        static std::mutex zeros_mutex;
        static std::map<std::string, ov::Tensor> zeros;
        std::lock_guard<std::mutex> lock(zeros_mutex);
        auto & zero = zeros[info.type.get_type_name() + info.shape.to_string()];
        if (!zero) {
            zero = ov::Tensor(info.type, info.shape);
            std::memset(zero.data(), 0, zero.get_byte_size());
        }
        return zero;
    }
    if (param_name == "ggml_hfa_mask") {
        const auto & info = ggml_decoder->get_model_extra_inputs().at(param_name);
        ov::Tensor input_tensor(info.type, info.shape);
        auto * data = static_cast<ggml_fp16_t *>(input_tensor.data());
        const size_t context_size = info.shape[3];
        const size_t past_capacity = context_size - chunk_size;
        const size_t history_size = std::min(static_cast<size_t>(chunk_index) * chunk_size, past_capacity);
        const auto neg_inf = GGML_FP32_TO_FP16(-INFINITY);
        const auto zero = GGML_FP32_TO_FP16(0.0f);
        std::fill(data, data + chunk_size * context_size, neg_inf);
        for (size_t row = 0; row < chunk_size; ++row) {
            std::fill(data + row * context_size, data + row * context_size + history_size, zero);
            for (size_t col = 0; col <= row; ++col) {
                data[row * context_size + past_capacity + col] = zero;
            }
        }
        return input_tensor;
    }
    if (const auto * node = ggml_decoder->get_host_gather_rows(param_name)) {
        return make_host_rows_tensor(node, chunk_index * chunk_size, chunk_valid_size, chunk_size);
    }
    if (chunk_index > 0 && param_name == "cache_rs_reset_len") {
        // The recurrent-state clear belongs to the start of the sequence. Re-applying it on every
        // chunk would wipe the state accumulated by the preceding chunks, so disable it (a zero
        // length makes translate_recurrent_cache_reset keep every slot) after the first chunk.
        ov::Tensor input_tensor(ov::element::i64, ov::Shape{1});
        *input_tensor.data<int64_t>() = 0;
        return input_tensor;
    }
    if (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        return get_ov_input_tensor(ggml_decoder, param_name);
    }
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(param_name);
    const auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    if (GgmlOvDecoder::is_inp_pos(ggml_tensor, op) && GgmlOvDecoder::get_inp_pos_n_planes(op) > 1) {
        // IMROPE: inp_pos stacks n_planes (t/h/w/e) position planes, each of length
        // input_len; pad every plane independently so they stay aligned to chunk_size.
        const int n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
        const size_t element_size = ggml_type_size(ggml_tensor->type);
        ov::Shape input_shape = {1, 1, 1, (size_t) n_planes * chunk_size};
        ov::Tensor input_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), input_shape);
        for (int p = 0; p < n_planes; p++) {
            const char * src =
                (const char *) ggml_tensor->data + (p * input_len + chunk_index * chunk_size) * element_size;
            char * dst = (char *) input_tensor.data() + p * chunk_size * element_size;
            std::memcpy(dst, src, chunk_valid_size * element_size);
            if (chunk_pad_size > 0) {
                if (ggml_tensor->type == GGML_TYPE_I32) {
                    int32_t last_value = *((const int32_t *) src + chunk_valid_size - 1);
                    int32_t * out = (int32_t *) dst;
                    std::fill(out + chunk_valid_size, out + chunk_size, last_value + 1);
                } else if (ggml_tensor->type == GGML_TYPE_I64) {
                    int64_t last_value = *((const int64_t *) src + chunk_valid_size - 1);
                    int64_t * out = (int64_t *) dst;
                    std::fill(out + chunk_valid_size, out + chunk_size, last_value + 1);
                } else {
                    throw std::runtime_error("Unexpected tensor type for " + param_name);
                }
            }
        }
        return input_tensor;
    }

    if (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape input_shape = {1, 1, 1, chunk_size};
        ov::Tensor input_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), input_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_size = ggml_type_size(ggml_tensor->type);
        void * input_data = (char *) ggml_tensor->data + chunk_index * chunk_size * element_size;
        std::memcpy(input_tensor.data(), input_data, chunk_valid_size * element_size);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        if (chunk_pad_size > 0) {
            if (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t last_value =
                    *((int32_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int32_t * output_data = input_tensor.data<int32_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_value + 1);
            } else if (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t last_value =
                    *((int64_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int64_t * output_data = input_tensor.data<int64_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_value + 1);
            } else {
                throw std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        return input_tensor;
    }

    if (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        size_t output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape input_shape = {1, 1, 1, output_len};
        ov::Tensor input_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), input_shape);
        if (ggml_tensor->ne[0] == 0) {
            *input_tensor.data<int32_t>() = 0;
        } else {
            auto * data_addr = input_tensor.data<int32_t>();
            for (size_t i = 0; i < output_len; i++) {
                data_addr[i] = ((int32_t *) ggml_tensor->data)[i] % chunk_size;
            }
        }
        return input_tensor;
    }

    if (GgmlOvDecoder::is_inp_scale_rows(ggml_tensor, op)) {
        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, 1});
        auto * dst = input_tensor.data<float>();
        const auto * src = static_cast<const float *>(ggml_tensor->data) + chunk_index * chunk_size;
        std::copy(src, src + chunk_valid_size, dst);
        std::fill(dst + chunk_valid_size, dst + chunk_size, 1.0f);
        return input_tensor;
    }

    if (GgmlOvDecoder::is_inp_mean(ggml_tensor, op)) {
        const size_t n_seqs = ggml_tensor->ne[1];
        const size_t src_stride = ggml_tensor->ne[0];
        const size_t copy_len = std::min<size_t>(chunk_valid_size, src_stride - chunk_index * chunk_size);
        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, n_seqs, chunk_size});
        auto * dst = input_tensor.data<float>();
        std::fill(dst, dst + n_seqs * chunk_size, 0.0f);
        const auto * src = static_cast<const float *>(ggml_tensor->data) + chunk_index * chunk_size;
        for (size_t s = 0; s < n_seqs; s++) {
            std::memcpy(dst + s * chunk_size, src + s * src_stride, copy_len * sizeof(float));
        }
        return input_tensor;
    }

    if (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t cols = ggml_tensor->ne[0];
        size_t rows = ggml_tensor->ne[1];
        size_t chunk_valid_rows = std::min(chunk_size, rows - chunk_index * chunk_size);
        size_t context_size = get_static_attention_size(ggml_decoder, param_name);
        static const bool fast_mask = ggml_openvino_npu_fast_mask_enabled();
        if (ggml_tensor->type == GGML_TYPE_F16) {
            const auto * ggml_data =
                static_cast<const ggml_fp16_t *>(ggml_tensor->data) + chunk_index * chunk_size * cols;
            ov::Tensor input_tensor(ov::element::f16, ov::Shape{1, 1, chunk_size, context_size});
            if (fast_mask) {
                fill_prefill_mask<ggml_fp16_t>(static_cast<ggml_fp16_t *>(input_tensor.data()), ggml_data,
                                               chunk_valid_rows, cols, chunk_size, context_size,
                                               GGML_FP32_TO_FP16(-INFINITY), GGML_FP32_TO_FP16(0.0f));
                return input_tensor;
            }
            std::vector<ggml_fp16_t> padded_data = pad_input<ggml_fp16_t>(ggml_data, chunk_valid_rows, cols, chunk_size,
                                                                          context_size, GGML_FP32_TO_FP16(-INFINITY));
            set_zero_diagonal(padded_data, chunk_size, context_size, GGML_FP32_TO_FP16(0.0f));
            std::memcpy(input_tensor.data(), padded_data.data(), padded_data.size() * sizeof(ggml_fp16_t));
            return input_tensor;
        }

        const auto * ggml_data = static_cast<const float *>(ggml_tensor->data) + chunk_index * chunk_size * cols;
        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, context_size});
        if (fast_mask) {
            fill_prefill_mask<float>(input_tensor.data<float>(), ggml_data, chunk_valid_rows, cols, chunk_size,
                                     context_size, -INFINITY, 0.0f);
            return input_tensor;
        }
        std::vector<float> padded_data =
            pad_input<float>(ggml_data, chunk_valid_rows, cols, chunk_size, context_size, -INFINITY);
        set_zero_diagonal(padded_data, chunk_size, context_size);
        auto * data_ptr = input_tensor.data<float>();
        std::copy(padded_data.begin(), padded_data.begin() + chunk_size * context_size, data_ptr);
        return input_tensor;
    }

    return get_ov_input_tensor(ggml_decoder, param_name);
}

enum ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config,
                               ov_compiled_model_cache & cache) {
    if (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        return GGML_STATUS_SUCCESS;
    }

    std::unique_lock<std::mutex> compile_lock(cache.mutex);
    bool naive = true;
    auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, naive);
    auto decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    decoder->device = device;
    auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(decoder);
    auto model = ov::frontend::ggml::FrontEnd::convert(input_model, naive);
    if (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    auto remote_context = ggml_openvino_get_remote_context(device);
    ov::AnyMap compile_config = config;
    if (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
    } else {
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
    }
    if (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
    } else {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, device, compile_config).create_infer_request());
    }
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    for (const auto & param : model->get_parameters()) {
        input_names.push_back(param->get_friendly_name());
    }
    for (const auto & result : model->get_results()) {
        output_names.push_back(result->get_friendly_name());
    }
    // Destroy the frontend graph under the compilation lock as well: it can
    // still own edges into the shared weight nodes.
    model.reset();
    input_model.reset();
    decoder->clear_model_weights();
    model_weights.clear();
    compile_lock.unlock();

    for (size_t i = 0; i < input_names.size(); i++) {
        const auto & param_name = input_names[i];
        auto input_tensor = get_ov_input_tensor(decoder, param_name);
        infer_request->set_input_tensor(i, input_tensor);
    }

    // Use get_output_tensor + memcpy instead of set_output_tensor to avoid memory overwritten
    // when i/o buffer overlaps, e.g. the cgraph is a single PERMUTE

    infer_request->infer();

    for (size_t i = 0; i < output_names.size(); i++) {
        auto output_tensor = infer_request->get_output_tensor(i);
        const auto & model_outputs = decoder->get_model_outputs();
        auto model_output_it = model_outputs.find(output_names[i]);
        if (model_output_it == model_outputs.end()) {
            // Debug-only output added via GGML_OPENVINO_DEBUG_NODE; nothing to copy into.
            if (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                print_output_tensor_info(output_names[i], output_tensor, nullptr);
            }
            continue;
        }
        auto * ggml_tensor = model_output_it->second;
        std::memcpy(ggml_tensor->data, output_tensor.data(), output_tensor.get_byte_size());
    }
    return GGML_STATUS_SUCCESS;
}

namespace {

// Small host inputs and outputs go through backend-owned USM host tensors, so the GPU plugin has no per-call copies.
constexpr size_t k_io_staging_max_bytes = 4u << 20;

bool io_staging_enabled() {
    static const bool enabled =
        ggml_openvino_is_gpu() && ggml_openvino_get_remote_context(ggml_openvino_get_device_name()).has_value();
    return enabled;
}

// Told from the ggml side: asking an ov::Tensor if it is remote is slow (it throws for a host tensor).
bool is_host_input(const std::shared_ptr<GgmlOvDecoder> & ggml_decoder, const std::string & name) {
    if (ggml_decoder->get_model_extra_inputs().count(name) != 0) {
        return true;
    }
    const auto * t = ggml_decoder->get_input_ggml_tensor(name);
    return t != nullptr && !ggml_openvino_buffer_is_remote(t);
}

// Off until the OpenVINO GPU plugin is fixed: it writes out of bounds when a dynamic output it bound to USM host memory
// is later bound to plain host memory, which output staging does when the output count changes between calls.
constexpr bool k_stage_outputs = false;

bool stageable_size(const ov::Tensor & t) {
    return t.get_byte_size() > 0 && t.get_byte_size() <= k_io_staging_max_bytes;
}

ov::Tensor & staging_for(const GgmlOvDecoder & decoder, GgmlOvDecoder::ov_port_binding & b, const ov::Tensor & like) {
    const size_t bytes = like.get_byte_size();
    if (!b.staging || b.staging_bytes < bytes || b.staging.get_element_type() != like.get_element_type()) {
        auto gpu_context = ggml_openvino_get_remote_context(decoder.device)->as<ov::intel_gpu::ocl::ClContext>();
        auto usm = gpu_context.create_usm_host_tensor(like.get_element_type(), like.get_shape());
        b.staging_ptr = usm.get();
        b.staging = usm;
        b.staging_bytes = bytes;
        b.staging_id = decoder.next_tensor_id();
    } else if (b.staging.get_shape() != like.get_shape()) {
        b.staging.set_shape(like.get_shape());
    }
    return b.staging;
}

// False when the port already holds this tensor (same id) with the same shape and type.
bool needs_bind(GgmlOvDecoder::ov_port_binding & b, const ov::Tensor & t, uint64_t id) {
    if (io_staging_enabled() && id != 0 && id == b.bound_id && t.get_element_type() == b.bound_type &&
        t.get_shape() == b.bound_shape) {
        return false;
    }
    b.bound_id = id;
    b.bound_shape = t.get_shape();
    b.bound_type = t.get_element_type();
    return true;
}

}  // namespace

enum ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, const std::shared_ptr<ov_runtime_context> & r_ctx) {
    auto & core = ov_singleton_core();
    const auto & config = ggml_openvino_get_compile_config();
    const auto & device = r_ctx->device;
    static auto is_static = false;

    static const bool cache_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    auto start_time = ggml_time_us();

    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    const int64_t cache_key_start_time = ggml_time_us();
    graph_key key(cgraph);
    const int64_t cache_key_compute_time = ggml_time_us() - cache_key_start_time;
    int64_t cache_lookup_time = 0;
    bool key_seen = false;
    if (!cache_disabled) {
        const int64_t cache_lookup_start_time = ggml_time_us();
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        key_seen = r_ctx->decoder_cache.find(key) != r_ctx->decoder_cache.end();
        cache_lookup_time += ggml_time_us() - cache_lookup_start_time;
    }

    const int64_t model_split_check_start_time = ggml_time_us();
    bool model_is_splitted = key_seen ? false : is_model_splitted(cgraph);
    const int64_t model_split_check_time = ggml_time_us() - model_split_check_start_time;

    if (ggml_openvino_model_cache_only() && (model_is_splitted || is_naive(cgraph))) {
        GGML_LOG_ERROR("ggml-openvino: cache-only mode requires a complete model graph\n");
        return GGML_STATUS_FAILED;
    }
    if (is_naive(cgraph)) {
        if (!model_is_splitted) {
            return naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
        }
    }

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    const int64_t graph_param_compute_start_time = ggml_time_us();
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);
    const int64_t graph_param_compute_time = ggml_time_us() - graph_param_compute_start_time;

    const bool cache_enabled = !model_is_splitted && !cache_disabled;
    const bool stateful_recurrent = r_ctx->stateful && m_params.state_size >= 0;
    const bool stateful_kv_only = r_ctx->stateful && !stateful_recurrent;
    if (r_ctx->stateful &&
        (m_params.n_seq > 1 || c_params.n_seq_active > 1 || m_params.n_rs_slots > 1 || m_params.has_rs_rollback)) {
        GGML_LOG_ERROR("OpenVINO stateful execution requires a single slot without recurrent rollback. Use -np 1.\n");
        return GGML_STATUS_FAILED;
    }
    if (stateful_recurrent && !cache_enabled) {
        GGML_LOG_ERROR("OpenVINO recurrent stateful execution requires an unsplit graph with model caching enabled.\n");
        return GGML_STATUS_FAILED;
    }
    bool cache_hit = false;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;
    int64_t ov_raw_infer_start;

    {
        std::shared_ptr<decoder_runtime_ctx> entry;
        ModelParams old_m_params;

        const int64_t cache_lookup_start_time = ggml_time_us();
        if (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            auto it = r_ctx->decoder_cache.find(key);
            cache_hit = it != r_ctx->decoder_cache.end();
            if (cache_hit) {
                entry = it->second;
                r_ctx->touch_locked(key);
            } else {
                if (r_ctx->stateful) {
                    r_ctx->clear_caches_locked();
                }
                auto mutex = std::make_shared<std::mutex>();
                entry = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[key] = entry;
                r_ctx->admit_locked(key);
            }
        } else {
            auto mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            cache_hit = false;
        }
        cache_lookup_time += ggml_time_us() - cache_lookup_start_time;

        std::lock_guard<std::mutex> lock(*(entry->mutex));
        cache_hit = cache_hit && entry->ptr && r_ctx->infer_request_cache.count(key) != 0;

        if (cache_hit) {
            ggml_decoder = entry->ptr;
            // The cache key is only (n_nodes, first/last node name), so it cannot see a change in
            // WHICH tensors are graph outputs -- and the compiled model's Results are fixed when it is
            // built. A consumer that starts requesting extra outputs later (speculative decoding marks
            // the target's per-layer residuals only once the draft is initialised) would otherwise keep
            // reusing a model that never produces them, and the host would read unwritten buffers as
            // zeros. Recompile when the output set differs.
            old_m_params = ggml_decoder->get_model_params();
            if (!ggml_decoder->is_splited_model()) {
                cache_hit = old_m_params.can_reuse_dynamically(m_params);
            }
            // Must come AFTER can_reuse_dynamically(), which ASSIGNS to cache_hit rather than
            // and-ing into it and would otherwise silently undo this invalidation.
            if (cache_hit && !ggml_decoder->has_same_graph_io(cgraph)) {
                cache_hit = false;
            }
        }

        std::vector<std::string> ov_input_names;
        std::vector<std::string> ov_output_names;

        if (cache_hit) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->set_compute_params(c_params);
            ggml_decoder->set_model_params(m_params);
            if (old_m_params.kv_buffer_changed(m_params) || !ggml_decoder->is_bound_to(cgraph)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->add_extra_inputs();
            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(key);
                ov_input_names = r_ctx->ov_input_names_cache.at(key);
                ov_output_names = r_ctx->ov_output_names_cache.at(key);
            }

            if (stateful_kv_only) {
                const auto * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_data = (int32_t *) inp_pos->data;
                auto pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                if (pos_data[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_size = pos_shape[3];
                } else if (r_ctx->stateful_kv_size == static_cast<size_t>(pos_data[0])) {
                    r_ctx->stateful_kv_size += pos_shape[3];
                } else {
                    const size_t pos_begin = static_cast<size_t>(pos_data[0]);
                    const bool refill = pos_begin > r_ctx->stateful_kv_size;

                    // A refill seeds the state from ggml's KV cache, so it needs that cache to be a
                    // plain prefix: cell i must hold position i. An SWA layer keeps only the last
                    // n_swa positions, so once a position leaves the window ggml drops it and the
                    // remaining cells shift - cell i stops holding position i. While every position
                    // is still inside the window nothing has been dropped and the refill is sound.
                    if (refill && !ggml_decoder->get_model_params().swa_layers.empty()) {
                        const int n_swa = ggml_decoder->get_compute_params().swa_window;
                        if (n_swa < 0 || static_cast<size_t>(n_swa) < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: cannot resume at position %zu from a "
                                "state that holds %zu tokens, because the sliding-window layers keep only the last %d "
                                "positions. Run without GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                                pos_begin, r_ctx->stateful_kv_size, n_swa);
                            return GGML_STATUS_FAILED;
                        }
                    }

                    const bool relayout_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT");

                    auto states = infer_request->query_state();
                    for (auto state : states) {
                        auto state_tensor = state.get_state();
                        auto state_tensor_shape = state_tensor.get_shape();

                        std::string state_name;
                        if (auto it = r_ctx->kv_state_input_name_map.find(state.get_name());
                            it != r_ctx->kv_state_input_name_map.end()) {
                            state_name = it->second;
                        }

                        // Which axis holds the sequence: pass::KVStateSeqAxis moves it from dim 1
                        // to dim 2. The head count is still needed below, because only a 1-head
                        // state stays byte-compatible with ggml's cache buffer. gemma-4 12B mixes
                        // 1-head full layers with 8-head sliding layers, so it is per state.
                        int n_heads_kv = ggml_decoder->get_model_params().n_heads_kv;
                        if (auto layer = extract_layer_from_name(state_name); layer.has_value()) {
                            n_heads_kv = ggml_decoder->get_n_heads_kv_for_layer(layer.value());
                        }
                        const bool relayout_this_state = relayout_enabled;
                        const size_t seq_axis = relayout_this_state ? 2 : 1;
                        const size_t head_axis = seq_axis == 2 ? 1 : 2;

                        if (refill) {
                            if (state_name.empty()) {
                                GGML_LOG_ERROR(
                                    "GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                return GGML_STATUS_FAILED;
                            }
                            auto kv_tensor = get_ov_input_tensor(ggml_decoder, state_name);
                            if (relayout_this_state && n_heads_kv != 1) {
                                // several heads with seq on dim 2: not the same bytes as ggml's
                                // buffer, so the rows have to be copied into the new order
                                state_tensor = kv_rows_to_seq_axis_2(kv_tensor, (size_t) n_heads_kv);
                            } else {
                                ov::Shape refill_shape(4);
                                refill_shape[0] = state_tensor_shape[0];
                                refill_shape[seq_axis] = kv_tensor.get_shape()[2];
                                refill_shape[head_axis] = state_tensor_shape[head_axis];
                                refill_shape[3] = state_tensor_shape[3];
                                kv_tensor.set_shape(refill_shape);
                                state_tensor = kv_tensor;
                            }
                            state_tensor_shape = state_tensor.get_shape();
                        }
                        // Only ever shrink to a prefix the source really has. Slicing past it used to
                        // surface as a bare ov::Exception from the ROI constructor.
                        if (state_tensor_shape[seq_axis] < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: state '%s' holds %zu tokens on axis "
                                "%zu, cannot resume at position %zu\n",
                                state.get_name().c_str(), state_tensor_shape[seq_axis], seq_axis, pos_begin);
                            return GGML_STATUS_FAILED;
                        }
                        ov::Coordinate begin = {0, 0, 0, 0};
                        ov::Coordinate end(state_tensor_shape.begin(), state_tensor_shape.end());
                        end[seq_axis] = pos_begin;
                        ov::Tensor new_state_tensor(state_tensor, begin, end);
                        state.set_state(new_state_tensor);
                    }
                    r_ctx->stateful_kv_size = pos_begin + pos_shape[3];
                }
            }

            decoder_end_time = ggml_time_us();
            conversion_end_time = decoder_end_time;
            compile_end_time = decoder_end_time;
        } else {
            // Compilation can mutate shared weight nodes, so serialize cold paths.
            // The lock is released before binding tensors or running inference.
            auto shared_cache = r_ctx->compiled_cache;
            std::unique_lock<std::mutex> compile_lock(shared_cache->mutex);
            auto weight_names = get_weight_names(cgraph);
            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, weight_names,
                                                          is_static, r_ctx->stateful, model_is_splitted);
            ggml_decoder->device = device;
            const std::string model_cache_dir = ggml_openvino_model_cache_dir();
            const std::string exact_key = cache_enabled ? compiled_graph_key(cgraph, *ggml_decoder, device) : "";
            uint64_t model_fp = 0;
            uint64_t exact_fp = 0;
            std::string blob_path, manifest_path;
            if (!model_cache_dir.empty() && !model_is_splitted) {
                const uint64_t extra_cfg =
                    ggml_openvino_model_cache_extra_cfg(r_ctx->stateful, stateful_recurrent);
                model_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/true, m_params.rope_params, 16, extra_cfg,
                                                    dynamic_graph_signature(cgraph, *ggml_decoder, m_params));
                exact_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/true, m_params.rope_params, 16, extra_cfg,
                                                    compiled_graph_key(cgraph, *ggml_decoder, device, 0, true));
                blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
            }
            std::string shared_key =
                cache_enabled ? (model_fp ? "dynamic:" + std::to_string(model_fp) : exact_key) : "";
            ov::CompiledModel shared_model;
            bool imported = false;
            auto shared_it = shared_cache->graphs.find(shared_key);
            if (!shared_key.empty() && shared_it != shared_cache->graphs.end() &&
                (model_fp == 0 || compiled_model_matches_graph(shared_it->second.decode, shared_it->second.input_names,
                                                               shared_it->second.output_names, *ggml_decoder))) {
                shared_model = shared_it->second.decode;
                infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                ov_input_names = shared_it->second.input_names;
                ov_output_names = shared_it->second.output_names;
                imported = true;
                GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (dynamic)\n");
            } else if (model_fp && shared_it != shared_cache->graphs.end()) {
                shared_key = exact_key;
                shared_it = shared_cache->graphs.find(shared_key);
                if (shared_it != shared_cache->graphs.end()) {
                    shared_model = shared_it->second.decode;
                    infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                    ov_input_names = shared_it->second.input_names;
                    ov_output_names = shared_it->second.output_names;
                    imported = true;
                }
                model_fp = exact_fp;
                blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
            }
            if (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(key);
            }

            // Import standalone blobs with weights before graph conversion and compilation.
            // Split models use the plugin cache instead. The decoder still maps graph I/O.
            ov::AnyMap mc_config = config;
            if (!model_cache_dir.empty()) {
                mc_config.erase("CACHE_DIR");
                mc_config[ov::cache_mode.name()] = ov::CacheMode::OPTIMIZE_SPEED;
            }
            if (!imported && !model_cache_dir.empty() && !model_is_splitted) {
                const uint64_t preferred_fp = model_fp;
                bool dynamic_incompatible = preferred_fp == exact_fp;
                for (int candidate = 0; candidate < 2; ++candidate) {
                    if (imported || (candidate == 1 && preferred_fp == exact_fp)) {
                        break;
                    }
                    const uint64_t candidate_fp = candidate == 0 ? preferred_fp : exact_fp;
                    const std::string candidate_blob =
                        ggml_openvino_model_cache_blob_path(model_cache_dir, candidate_fp);
                    const std::string candidate_manifest =
                        ggml_openvino_model_cache_manifest_path(model_cache_dir, candidate_fp);
                    std::vector<std::string> saved_inputs, saved_outputs;
                    if (!std::ifstream(candidate_blob, std::ios::binary).is_open() ||
                        !ggml_openvino_model_cache_verify_manifest(candidate_manifest, cgraph, candidate_fp,
                                                                   saved_inputs, saved_outputs)) {
                        continue;
                    }
                    const int64_t import_start = ggml_time_us();
                    try {
                        // Import from a memory-mapped tensor instead of a stream: a stream import reads the whole
                        // blob into a heap buffer while the plugin builds its own copy of the weights.
                        const ov::Tensor blob = ov::read_tensor_data(candidate_blob);
                        ov::CompiledModel cm;
                        auto remote_context = ggml_openvino_get_remote_context(device);
                        if (remote_context.has_value()) {
                            cm = core.import_model(blob, remote_context.value(), mc_config);
                        } else {
                            cm = core.import_model(blob, device, mc_config);
                        }
                        // CPU serialization can rename Results that share a name with a Parameter.
                        if (!saved_inputs.empty() || !saved_outputs.empty()) {
                            ov_input_names = std::move(saved_inputs);
                            ov_output_names = std::move(saved_outputs);
                        } else {
                            for (const auto & p : cm.inputs()) {
                                ov_input_names.push_back(p.get_node()->get_friendly_name());
                            }
                            for (const auto & o : cm.outputs()) {
                                ov_output_names.push_back(o.get_node()->get_friendly_name());
                            }
                        }
                        if (!compiled_model_matches_graph(cm, ov_input_names, ov_output_names, *ggml_decoder)) {
                            dynamic_incompatible |= candidate == 0;
                            ov_input_names.clear();
                            ov_output_names.clear();
                            continue;
                        }
                        infer_request = std::make_shared<ov::InferRequest>(cm.create_infer_request());
                        shared_model = cm;
                        entry->ptr = ggml_decoder;
                        imported = true;
                        model_fp = candidate_fp;
                        blob_path = candidate_blob;
                        manifest_path = candidate_manifest;
                        if (candidate_fp == exact_fp) {
                            shared_key = exact_key;
                            shared_it = shared_cache->graphs.find(shared_key);
                        }
                        if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            GGML_LOG_INFO("  - Model cache import time: %.3f ms \n",
                                          (ggml_time_us() - import_start) / 1000.0);
                        }
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", candidate_blob.c_str());
                    } catch (const std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed: %s\n", e.what());
                        dynamic_incompatible |= candidate == 0;
                        imported = false;
                        ov_input_names.clear();
                        ov_output_names.clear();
                        infer_request.reset();
                        shared_model = {};
                    }
                }
                if (!imported && dynamic_incompatible) {
                    model_fp = exact_fp;
                    shared_key = exact_key;
                    shared_it = shared_cache->graphs.find(shared_key);
                    blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                    manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
                }
            }

            std::shared_ptr<ov::Model> model;
            if (!imported && ggml_openvino_model_cache_only()) {
                // No blob is ever written for these (see ggml_openvino_cpu_blob_importable), so the
                // generic advice to run once without cache-only cannot help. MXFP4 does not use GatherMatmul.
                const auto is_gather_matmul_moe = [](const ggml_tensor * n) {
                    return n->op == GGML_OP_MUL_MAT_ID && n->src[0]->type != GGML_TYPE_MXFP4;
                };
                if (device == "CPU" &&
                    std::any_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_gather_matmul_moe)) {
                    GGML_LOG_ERROR(
                        "ggml-openvino: GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY does not work for MoE models on CPU: "
                        "the CPU plugin cannot import MoE models with this OpenVINO version, so they are not cached\n");
                    return GGML_STATUS_FAILED;
                }
                GGML_LOG_ERROR(
                    "ggml-openvino: missing or incompatible compiled model: %s; run once without "
                    "GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY\n",
                    blob_path.c_str());
                return GGML_STATUS_FAILED;
            }
            if (!imported && ggml_openvino_weight_buffers_released()) {
                GGML_LOG_ERROR("ggml-openvino: cannot compile a new graph after releasing host weights\n");
                return GGML_STATUS_FAILED;
            }
            if (imported) {
                decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            } else {
                auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                               r_ctx->stateful, model_is_splitted);
                ggml_decoder->device = device;
                decoder_end_time = ggml_time_us();

                auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder);
                model = ov::frontend::ggml::FrontEnd::convert(input_model);
                ggml_decoder->clear_model_weights();
                conversion_end_time = ggml_time_us();

                if (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
                    char timestamped_filename[64];
                    auto timestamp = (long long) ggml_time_us();
                    snprintf(timestamped_filename, sizeof(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                // Use the cache-stripped config when the frontend model cache is active, so
                // the resulting CompiledModel can be exported and later re-imported.
                const ov::AnyMap & compile_config = model_cache_dir.empty() ? config : mc_config;
                ov::CompiledModel compiled_model;
                auto remote_context = ggml_openvino_get_remote_context(device);
                if (remote_context.has_value()) {
                    compiled_model = core.compile_model(model, remote_context.value(), compile_config);
                } else {
                    compiled_model = core.compile_model(model, device, compile_config);
                }
                compile_end_time = ggml_time_us();

                for (const auto & ov_param : model->get_parameters()) {
                    ov_input_names.push_back(ov_param->get_friendly_name());
                }
                for (const auto & ov_output : model->get_results()) {
                    ov_output_names.push_back(ov_output->get_friendly_name());
                }

                // Export to the frontend model cache for next time. Publish the blob first,
                // then the manifest, so a cache hit only sees fully written artifacts.
                const bool blob_importable = ggml_openvino_cpu_blob_importable(device, model);
                if (!blob_importable && !model_cache_dir.empty()) {
                    GGML_LOG_WARN(
                        "ggml-openvino: model cache not written: the CPU plugin cannot import MoE models with this "
                        "OpenVINO version\n");
                }
                if (!model_cache_dir.empty() && !model_is_splitted && model_fp != 0 && blob_importable) {
                    try {
                        const std::string blob_tmp = ggml_openvino_model_cache_temp_path(blob_path);
                        const std::string manifest_tmp = ggml_openvino_model_cache_temp_path(manifest_path);
                        if (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp, ov_input_names,
                                                                     ov_output_names)) {
                            std::ofstream blob_out(blob_tmp, std::ios::binary | std::ios::trunc);
                            if (blob_out.is_open()) {
                                compiled_model.export_model(blob_out);
                                blob_out.close();
                                if (blob_out.good()) {
                                    if (std::rename(blob_tmp.c_str(), blob_path.c_str()) == 0 &&
                                        std::rename(manifest_tmp.c_str(), manifest_path.c_str()) == 0) {
                                        GGML_LOG_INFO("ggml-openvino: model cache WROTE %s\n", blob_path.c_str());
                                    } else {
                                        std::remove(blob_tmp.c_str());
                                        std::remove(manifest_tmp.c_str());
                                    }
                                } else {
                                    std::remove(blob_tmp.c_str());
                                    std::remove(manifest_tmp.c_str());
                                }
                            } else {
                                std::remove(manifest_tmp.c_str());
                            }
                        }
                    } catch (const std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache export failed: %s\n", e.what());
                    }
                }

                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                shared_model = compiled_model;
            }  // end non-imported (compile) path

            entry->ptr = ggml_decoder;
            if (!shared_key.empty() && shared_it == shared_cache->graphs.end()) {
                ov_compiled_graph graph;
                graph.decode = shared_model;
                graph.input_names = ov_input_names;
                graph.output_names = ov_output_names;
                shared_cache->graphs.emplace(shared_key, std::move(graph));
            }
            if (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[key] = infer_request;
                r_ctx->ov_input_names_cache[key] = ov_input_names;
                r_ctx->ov_output_names_cache[key] = ov_output_names;
            }

            if (stateful_kv_only && cache_enabled) {
                const auto * inp_pos = get_inp_pos_tensor(cgraph);
                auto pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                // A freshly compiled model starts with an empty state, so it can only serve a
                // sequence from its beginning. A non-zero start position means the KV history was
                // built elsewhere (a restored ggml cache), which the state cannot adopt.
                const int32_t pos_begin = ((int32_t *) inp_pos->data)[0];
                if (pos_begin != 0) {
                    GGML_LOG_ERROR(
                        "GGML OpenVINO backend stateful inference failed: a new model was compiled for a sequence that "
                        "starts at position %d, but its state is empty. Run without "
                        "GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                        pos_begin);
                    return GGML_STATUS_FAILED;
                }
                r_ctx->stateful_kv_size = pos_shape[3];
                const auto kv_param_res_names = ggml_decoder->get_kv_param_res_names();
                for (const auto & pair : kv_param_res_names) {
                    r_ctx->kv_state_input_name_map[pair.first + pair.second] = pair.first;
                }
            }
        }

        if (stateful_recurrent) {
            const auto * inp_pos = get_inp_pos_tensor(cgraph);
            const int32_t pos_begin = static_cast<const int32_t *>(inp_pos->data)[0];
            if (pos_begin == 0) {
                infer_request->reset_state();
            } else if (!cache_hit || old_m_params.kv_buffer_changed(m_params) || c_params.cache_rs_reset_len > 0 ||
                       pos_begin < 0 || static_cast<size_t>(pos_begin) != r_ctx->stateful_kv_size) {
                GGML_LOG_ERROR(
                    "OpenVINO recurrent stateful execution cannot restore or rewind a sequence. Restart at position 0 "
                    "or disable GGML_OPENVINO_STATEFUL_EXECUTION.\n");
                return GGML_STATUS_FAILED;
            }
        }

        auto & in_bindings = ggml_decoder->port_bindings(true, ov_input_names.size(), infer_request.get());
        for (size_t i = 0; i < ov_input_names.size(); i++) {
            const auto & param_name = ov_input_names[i];
            ggml_decoder->take_last_kv_tensor_id();  // clear
            auto input_tensor = get_ov_input_tensor(ggml_decoder, param_name);
            uint64_t id = ggml_decoder->take_last_kv_tensor_id();
            auto & binding = in_bindings[i];
            if (id == 0 && io_staging_enabled() && is_host_input(ggml_decoder, param_name) &&
                stageable_size(input_tensor)) {
                auto & staging = staging_for(*ggml_decoder, binding, input_tensor);
                std::memcpy(binding.staging_ptr, input_tensor.data(), input_tensor.get_byte_size());
                input_tensor = staging;
                id = binding.staging_id;
            }
            if (needs_bind(binding, input_tensor, id)) {
                try {
                    infer_request->set_input_tensor(i, input_tensor);
                } catch (const std::exception & e) {
                    if (ggml_openvino_getenv_int("GGML_OPENVINO_DIAG_GRAPH_IO")) {
                        auto port = infer_request->get_compiled_model().input(i);
                        GGML_LOG_WARN(
                            "ggml-openvino: DIAG set_input_tensor[%zu] '%s' threw for n_nodes=%d first=%s last=%s: %s\n"
                            "  model_shape=%s bound_shape=%s\n",
                            i, param_name.c_str(), key.n_nodes, key.first_node_name.c_str(),
                            key.last_node_name.c_str(), e.what(), port.get_partial_shape().to_string().c_str(),
                            input_tensor.get_shape().to_string().c_str());
                    }
                    throw;
                }
            }

            if (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                print_input_tensor_info(param_name, input_tensor);
            }
        }

        auto & out_bindings = ggml_decoder->port_bindings(false, ov_output_names.size(), infer_request.get());

        struct staged_output {
            void * dst;
            const void * src;
            size_t bytes;
        };

        std::vector<staged_output> staged_outputs;  // copied back after infer()
        for (size_t i = 0; i < ov_output_names.size(); i++) {
            // Debug-only outputs added via GGML_OPENVINO_DEBUG_NODE (see
            // translate_session.cpp) have no corresponding ggml tensor; leave
            // them unbound so OpenVINO allocates its own tensor for them,
            // rather than aliasing a ggml buffer that may be overwritten by a
            // later in-place op before we get to read it.
            const auto & model_outputs = ggml_decoder->get_model_outputs();
            auto model_output_it = model_outputs.find(ov_output_names[i]);
            if (model_output_it == model_outputs.end()) {
                continue;
            }
            auto * ggml_tensor = model_output_it->second;
            ggml_decoder->take_last_kv_tensor_id();  // clear
            ov::Tensor output_tensor;
            const bool inactive = (ggml_tensor->op != GGML_OP_NONE && !(ggml_tensor->flags & GGML_TENSOR_FLAG_COMPUTE)) ||
                                  ggml_nbytes(ggml_tensor) == 0;
            if (inactive) {
                // Inactive pure sampling branches have no scheduler allocation. Never bind a stale active output.
                auto & scratch = ggml_decoder->runtime_output_scratch[ov_output_names[i]];
                const auto type = GgmlOvDecoder::get_ov_type(ggml_tensor);
                const auto shape = GgmlOvDecoder::get_shape(ggml_tensor);
                if (!scratch || scratch.get_element_type() != type || scratch.get_shape() != shape) {
                    if (ggml_openvino_is_gpu() && ggml_nbytes(ggml_tensor) > 0) {
                        auto context = ggml_openvino_get_remote_context(device)->as<ov::intel_gpu::ocl::ClContext>();
                        scratch = context.create_usm_device_tensor(type, shape);
                    } else {
                        scratch = ov::Tensor(type, shape);
                    }
                }
                output_tensor = scratch;
            } else {
                output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            }
            uint64_t id = ggml_decoder->take_last_kv_tensor_id();
            auto & binding = out_bindings[i];
            if (k_stage_outputs && !inactive && id == 0 && io_staging_enabled() &&
                !ggml_openvino_buffer_is_remote(ggml_tensor) && stageable_size(output_tensor)) {
                void * dst = output_tensor.data();
                const size_t bytes = output_tensor.get_byte_size();
                output_tensor = staging_for(*ggml_decoder, binding, output_tensor);
                staged_outputs.push_back({dst, binding.staging_ptr, bytes});
                id = binding.staging_id;
            }
            if (needs_bind(binding, output_tensor, id)) {
                try {
                    infer_request->set_output_tensor(i, output_tensor);
                } catch (const std::exception & e) {
                    if (ggml_openvino_getenv_int("GGML_OPENVINO_DIAG_GRAPH_IO")) {
                        auto port = infer_request->get_compiled_model().output(i);
                        GGML_LOG_WARN(
                            "ggml-openvino: DIAG set_output_tensor[%zu] '%s' threw for n_nodes=%d first=%s last=%s: %s\n"
                            "  model_shape=%s bound_shape=%s\n",
                            i, ov_output_names[i].c_str(), key.n_nodes, key.first_node_name.c_str(),
                            key.last_node_name.c_str(), e.what(), port.get_partial_shape().to_string().c_str(),
                            output_tensor.get_shape().to_string().c_str());
                    }
                    throw;
                }
            }
        }

        ov_raw_infer_start = ggml_time_us();
        try {
            infer_request->infer();
        } catch (const std::exception & e) {
            if (ggml_openvino_getenv_int("GGML_OPENVINO_DIAG_GRAPH_IO")) {
                auto cm = infer_request->get_compiled_model();
                GGML_LOG_WARN("ggml-openvino: DIAG infer() threw for n_nodes=%d first=%s last=%s: %s\n", key.n_nodes,
                              key.first_node_name.c_str(), key.last_node_name.c_str(), e.what());
                for (size_t i = 0; i < ov_input_names.size(); i++) {
                    auto port = cm.input(i);
                    GGML_LOG_WARN("  param[%zu] name=%s model_shape=%s bound_shape=%s\n", i,
                                  ov_input_names[i].c_str(), port.get_partial_shape().to_string().c_str(),
                                  infer_request->get_input_tensor(i).get_shape().to_string().c_str());
                }
            }
            throw;
        }
        infer_end_time = ggml_time_us();
        for (const auto & out : staged_outputs) {
            std::memcpy(out.dst, out.src, out.bytes);
        }
        if (stateful_recurrent) {
            const auto * inp_pos = get_inp_pos_tensor(cgraph);
            r_ctx->stateful_kv_size =
                static_cast<const int32_t *>(inp_pos->data)[0] + get_inp_pos_n_tokens(cgraph, inp_pos);
        }
        dump_ov_profiling_info(*infer_request);

        if (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            for (size_t i = 0; i < ov_output_names.size(); i++) {
                const auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names[i], output_tensor, nullptr);
            }
        }

        if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_end_time - start_time) / 1000.0);
            GGML_LOG_INFO("    - cache key compute time: %.3f ms \n", cache_key_compute_time / 1000.0);
            GGML_LOG_INFO("    - model split check time: %.3f ms \n", model_split_check_time / 1000.0);
            GGML_LOG_INFO("    - graph param compute time: %.3f ms \n", graph_param_compute_time / 1000.0);
            GGML_LOG_INFO("    - cache lookup time: %.3f ms \n", cache_lookup_time / 1000.0);
            if (!cache_hit) {
                GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n",
                              (conversion_end_time - decoder_end_time) / 1000.0);
                GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_end_time - conversion_end_time) / 1000.0);
            }
            GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_end_time - compile_end_time) / 1000.0);
            GGML_LOG_INFO("    - OV raw infer time: %.3f ms \n", (infer_end_time - ov_raw_infer_start) / 1000.0);
        }
    }

    // GGML_OPENVINO_RELEASE_WEIGHTS (or GGML_OPENVINO_MEMORY_OPTIMIZE on GPU): the plugin holds its own device copy of
    // every weight after compile, so the host weight buffers can be dropped to reclaim
    // RSS. Release only while holding the compilation mutex so another context cannot
    // be reading host weights during conversion/compilation. Pin the shared compiled
    // models across backend teardown; a later context can create its own request without
    // reading the dropped pages. A new, uncached graph still fails fast above.
    if (cache_hit && ggml_openvino_release_weights_enabled()) {
        std::lock_guard<std::mutex> compile_lock(r_ctx->compiled_cache->mutex);
        if (!ggml_openvino_weight_buffers_released()) {
            ggml_openvino_release_weight_buffers();
        }
    }

    return GGML_STATUS_SUCCESS;
}

static uint64_t static_model_salt(uint64_t base_extra, bool prefill, int chunk, int attn, int attn_swa, bool unfold) {
    uint64_t h = base_extra;
    h = h * 131 + (prefill ? 1u : 0u);
    h = h * 131 + static_cast<uint64_t>(static_cast<uint32_t>(chunk));
    h = h * 131 + static_cast<uint64_t>(static_cast<uint32_t>(attn));
    h = h * 131 + static_cast<uint64_t>(static_cast<uint32_t>(attn_swa));
    h = h * 131 + (unfold ? 1u : 0u);
    return h;
}

static bool static_blob_valid(const std::string & model_cache_dir, uint64_t fp, const ggml_cgraph * cgraph) {
    if (model_cache_dir.empty() || fp == 0) {
        return false;
    }
    std::ifstream blob_in(ggml_openvino_model_cache_blob_path(model_cache_dir, fp), std::ios::binary);
    std::vector<std::string> inputs, outputs;  // ports are bound in frontend order
    return blob_in.is_open() &&
           ggml_openvino_model_cache_verify_manifest(ggml_openvino_model_cache_manifest_path(model_cache_dir, fp),
                                                     cgraph, fp, inputs, outputs);
}

struct static_build_args {
    ov::Core *    core;
    std::string   device;
    bool          dump_ir;
    long long     dump_ir_timestamp;
    std::string   model_cache_dir;
    ggml_cgraph * cgraph;
    bool          self_contained;
    bool          skip_weights;
    // GGML_OPENVINO_NPU_WEIGHTLESS_BANK: weightless blob whose weights live in weights_path.
    bool          weightless_bank = false;
    std::string   weights_path;
};

// Cached blobs imported in this process, keyed by fingerprint, so a pair or window switch reuses them. Self-contained
// models compiled in this process are kept here too. Cleared with the shared compiled-model cache when the last
// backend is freed.
static std::mutex g_imported_blobs_mutex;
static std::unordered_map<uint64_t, ov::CompiledModel> g_imported_blobs;

static ov::CompiledModel import_cached_blob(ov::Core & core,
                                            const std::string & device,
                                            const std::string & blob_path,
                                            uint64_t fp,
                                            const ov::AnyMap & import_config,
                                            const char * tag) {
    {
        std::lock_guard<std::mutex> lock(g_imported_blobs_mutex);
        auto it = g_imported_blobs.find(fp);
        if (it != g_imported_blobs.end()) {
            return it->second;
        }
    }
    std::ifstream blob_in(blob_path, std::ios::binary);
    const int64_t import_start_time = ggml_time_us();
    ov::CompiledModel compiled = core.import_model(blob_in, device, import_config);
    if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        fprintf(stderr, "ggml-openvino: model cache IMPORT %s: %.3f ms\n", tag,
                (ggml_time_us() - import_start_time) / 1000.0);
    }
    std::lock_guard<std::mutex> lock(g_imported_blobs_mutex);
    return g_imported_blobs.emplace(fp, std::move(compiled)).first->second;
}

static std::shared_ptr<ov::Model> convert_static_model(const static_build_args & a,
                                                       const std::shared_ptr<GgmlOvDecoder> & decoder,
                                                       const char * tag,
                                                       int64_t & local_conversion_end_time) {
    if (a.skip_weights) {
        return nullptr;
    }
    auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(decoder);
    auto model = ov::frontend::ggml::FrontEnd::convert(input_model);
    decoder->clear_model_weights();
    local_conversion_end_time = ggml_time_us();

    if (a.dump_ir) {
        char timestamped_filename[64];
        snprintf(timestamped_filename, sizeof(timestamped_filename), "model_%s_%lld.xml", tag, a.dump_ir_timestamp);
        ov::serialize(model, timestamped_filename);
    }
    return model;
}

static void build_static_model(const static_build_args & a,
                               std::shared_ptr<GgmlOvDecoder> decoder,
                               const char * tag,
                               std::shared_ptr<ov::Model> model,
                               const ov::AnyMap & model_config,
                               uint64_t model_fp,
                               ov::CompiledModel & compiled_model,
                               std::shared_ptr<ov::InferRequest> & infer_request,
                               std::vector<std::string> & names_in,
                               std::vector<std::string> & names_out,
                               int64_t & local_conversion_end_time,
                               int64_t & local_compile_end_time) {
    auto & core = *a.core;
    const auto & device = a.device;
    const auto & model_cache_dir = a.model_cache_dir;
    auto * cgraph = a.cgraph;
    std::string blob_path;
    std::string manifest_path;
    if (!model_cache_dir.empty() && model_fp != 0) {
        blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
        manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
    }

    if (a.skip_weights) {
        try {
            ov::AnyMap import_config = model_config;
            if (a.weightless_bank) {
                import_config[ov::weights_path.name()] = a.weights_path;
                import_config[ov::enable_weightless.name()] = true;
            }
            compiled_model = import_cached_blob(core, device, blob_path, model_fp, import_config, tag);
        } catch (const std::exception & e) {
            std::remove(blob_path.c_str());
            std::remove(manifest_path.c_str());
            throw std::runtime_error(std::string(a.weightless_bank ? "weightless-bank" : "self-contained") +
                                     " cache import failed for " + std::string(tag) +
                                     "; removed the invalid entry, rerun to rebuild it: " + e.what());
        }
        infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
        std::set<std::string> compiled_input_names;
        for (const auto & input : compiled_model.inputs()) {
            compiled_input_names.insert(input.get_node()->get_friendly_name());
        }
        // The frontend drops inputs nothing reads: HFA prefill, KV-only prefill (out_ids) and host gather (token
        // ids) can leave decoder inputs out of the compiled model. Keep the decoder order.
        for (const auto & [name, _] : decoder->get_model_inputs()) {
            if (compiled_input_names.count(name) != 0) {
                names_in.push_back(name);
            }
        }
        for (const auto & [name, info] : decoder->get_model_extra_inputs()) {
            if (info.is_parameter && compiled_input_names.count(name) != 0) {
                names_in.push_back(name);
            }
        }
        // A KV-only prefill model keeps only the cache outputs (pass::KeepCacheResults), in decoder order.
        for (const auto & name : decoder->is_kv_only_outputs() ? decoder->get_cache_output_names() :
                                                                 decoder->get_model_output_names()) {
            names_out.push_back(name);
        }
        if (names_in.size() != compiled_model.inputs().size() || names_out.size() != compiled_model.outputs().size()) {
            throw std::runtime_error(
                "self-contained cache port count mismatch for " + std::string(tag) +
                ": decoder inputs=" + std::to_string(names_in.size()) +
                ", compiled inputs=" + std::to_string(compiled_model.inputs().size()) +
                ", decoder outputs=" + std::to_string(names_out.size()) +
                ", compiled outputs=" + std::to_string(compiled_model.outputs().size()));
        }
        decoder->clear_model_weights();
        local_conversion_end_time = local_compile_end_time = ggml_time_us();
        GGML_LOG_INFO("ggml-openvino: model cache HIT (%s) %s (%s)\n",
                      a.weightless_bank ? "weightless bank" : "self-contained", blob_path.c_str(), tag);
        return;
    }

    bool imported = false;
    // A self-contained blob embeds its weights bank, and NPUW cannot load a blob into a bank that already holds
    // weights. Pairs that differ only in the prefill model share the decode model and its bank, so build each model
    // once per process.
    const bool keep_in_process = a.self_contained && !a.weightless_bank && model_fp != 0;
    if (keep_in_process) {
        std::lock_guard<std::mutex> lock(g_imported_blobs_mutex);
        auto it = g_imported_blobs.find(model_fp);
        if (it != g_imported_blobs.end()) {
            compiled_model = it->second;
            imported = true;
        }
    }
    // A weightless-bank blob is only valid with the weights file it was exported with, which is rewritten whenever
    // the blob is rebuilt, so never import a leftover blob here.
    if (!imported && !model_cache_dir.empty() && model_fp != 0 && !a.weightless_bank) {
        std::ifstream blob_in(blob_path, std::ios::binary);
        std::vector<std::string> cached_inputs, cached_outputs;
        if (blob_in.is_open() &&
            ggml_openvino_model_cache_verify_manifest(manifest_path, cgraph, model_fp, cached_inputs, cached_outputs)) {
            try {
                ov::AnyMap import_config = model_config;
                if (!a.self_contained) {
                    import_config[ov::hint::model.name()] = std::shared_ptr<const ov::Model>(model);
                }
                const int64_t import_start_time = ggml_time_us();
                compiled_model = core.import_model(blob_in, device, import_config);
                const int64_t import_end_time = ggml_time_us();
                if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                    fprintf(stderr, "ggml-openvino: model cache IMPORT %s: %.3f ms\n", tag,
                            (import_end_time - import_start_time) / 1000.0);
                }
                imported = true;
                GGML_LOG_INFO("ggml-openvino: model cache HIT %s (%s)\n", blob_path.c_str(), tag);
            } catch (const std::exception & e) {
                GGML_LOG_WARN("ggml-openvino: model cache import failed (%s), recompiling %s\n", e.what(), tag);
                imported = false;
            }
        }
    }

    if (!imported) {
        ov::AnyMap compile_config = model_config;
        if (a.self_contained) {
            compile_config[ov::enable_weightless.name()] = a.weightless_bank;
        }
        compiled_model = core.compile_model(model, device, compile_config);
    }
    if (keep_in_process) {
        std::lock_guard<std::mutex> lock(g_imported_blobs_mutex);
        g_imported_blobs.emplace(model_fp, compiled_model);
    }
    infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
    local_compile_end_time = ggml_time_us();

    for (const auto & ov_param : model->get_parameters()) {
        names_in.push_back(ov_param->get_friendly_name());
    }
    for (const auto & ov_output : model->get_results()) {
        names_out.push_back(ov_output->get_friendly_name());
    }

    if (!imported && !model_cache_dir.empty() && model_fp != 0) {
        try {
            const std::string blob_tmp = blob_path + ".tmp";
            const std::string manifest_tmp = manifest_path + ".tmp";
            const std::string source_manifest_tmp = manifest_path + ".source.tmp";
            const bool manifest_ready =
                a.self_contained ?
                    std::ifstream(source_manifest_tmp).good() :
                    ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp, names_in, names_out);
            const std::string & manifest_publish = a.self_contained ? source_manifest_tmp : manifest_tmp;
            if (manifest_ready) {
                std::ofstream blob_out(blob_tmp, std::ios::binary | std::ios::trunc);
                if (blob_out.is_open()) {
                    compiled_model.export_model(blob_out);
                    blob_out.close();
                    // std::rename does not replace an existing file on Windows, which would keep a stale blob.
                    if (blob_out.good()) {
                        std::remove(manifest_path.c_str());
                        std::remove(blob_path.c_str());
                    }
                    if (blob_out.good() && std::rename(blob_tmp.c_str(), blob_path.c_str()) == 0 &&
                        std::rename(manifest_publish.c_str(), manifest_path.c_str()) == 0) {
                        GGML_LOG_INFO("ggml-openvino: model cache WROTE %s (%s)\n", blob_path.c_str(), tag);
                    } else {
                        std::remove(blob_tmp.c_str());
                        std::remove(manifest_publish.c_str());
                    }
                } else {
                    std::remove(manifest_publish.c_str());
                }
            }
        } catch (const std::exception & e) {
            GGML_LOG_WARN("ggml-openvino: model cache export failed (%s): %s\n", tag, e.what());
        }
    }
}

// A layer that reuses the KV cache of an earlier layer (Gemma 3n/4 E-series) reads a KV cache tensor that another
// attention op also reads. Every other layer reads its own cache, whichever caches it writes (MLA has no V cache).
static bool has_kv_shared_attention(const ggml_cgraph * cgraph) {
    std::unordered_map<const ggml_tensor *, int> readers;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_FLASH_ATTN_EXT || node->src[1] == nullptr) {
            continue;
        }
        const ggml_tensor * k = node->src[1];
        while (k->view_src != nullptr) {
            k = k->view_src;
        }
        if (GgmlOvDecoder::is_kvcache(k, nullptr) && ++readers[k] > 1) {
            return true;
        }
    }
    return false;
}

// The output step runs a token of the batch through the decode model again. That rewrites the token's KV rows with the
// same values, but would apply a recurrent state update twice.
static bool has_recurrent_state(const ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (GgmlOvDecoder::is_recurrent_cache(node->view_src)) {
            return true;
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (GgmlOvDecoder::is_recurrent_cache(node->src[j])) {
                return true;
            }
        }
    }
    return false;
}

// Set when a KV-only prefill could not compute the outputs of a batch; later graphs use the full prefill.
static std::atomic<bool> g_kv_only_prefill_failed{false};

enum ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, const std::shared_ptr<ov_runtime_context> & r_ctx) {
    auto & core = ov_singleton_core();

    auto get_prefill_chunk_size = [] {
        static const int chunk_size = []() {
            int env_prefill_chunk_size = ggml_openvino_getenv_int("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
            return env_prefill_chunk_size > 0 ? env_prefill_chunk_size : 256;
        }();
        return chunk_size;
    };

    // Normally NPU, but honors GGML_OPENVINO_DEVICE so GGML_OPENVINO_FORCE_STATIC can run the
    // static-shape path on CPU/GPU to isolate translation bugs from NPUW/NPU-driver issues.
    const auto & device = r_ctx->device;
    static auto is_static = true;
    static auto stateful = false;

    auto prefill_chunk_size = get_prefill_chunk_size();
    const auto & config = ggml_openvino_get_compile_config();

    if (is_naive(cgraph)) {
        return naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
    }

    auto start_time = ggml_time_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    const auto * inp_pos = get_inp_pos_tensor(cgraph);
    const bool no_kv_cache = m_params.is_cacheless_attn;
    const auto is_prefill = no_kv_cache ? true : get_is_prefill(cgraph, inp_pos);
    const int64_t input_tokens = get_inp_pos_n_tokens(cgraph, inp_pos);
    if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        fprintf(stderr, "GGML OpenVINO profiling: static graph kind=%s, input_tokens=%lld, chunk_size=%d\n",
                is_prefill ? "prefill" : "decode", static_cast<long long>(input_tokens), prefill_chunk_size);
    }
    const ov::AnyMap compile_config = no_kv_cache              ? ggml_openvino_npu_without_npuw(config) :
                                      m_params.state_size >= 0 ? ggml_openvino_npu_gdn_config(config) :
                                                                 config;
    // Decided per compiled pair from the graph that triggers the compile; later calls follow entry->kv_only_prefill.
    // The output step runs the decode model of the KV window covering the output token (the full context without
    // decode KV windows).
    const bool kv_only_prefill = !no_kv_cache && ggml_openvino_kv_only_prefill_enabled() && !g_kv_only_prefill_failed &&
                                 !ggml_openvino_npu_hfa_prefill_enabled() &&
                                 c_params.output_len <= 1 && has_kv_shared_attention(cgraph) &&
                                 !has_recurrent_state(cgraph);
    std::shared_ptr<ov::InferRequest> decode_request_for_outputs;
    std::vector<std::string> decode_input_names_for_outputs;
    std::vector<std::string> decode_output_names_for_outputs;
    if (m_params.n_heads_kv == -1) {
        prefill_chunk_size = inp_pos->ne[0];
    }
    graph_key key(cgraph);
    static const bool cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    bool cache_hit = false;

    // Decode KV window: smallest base*2^k covering the next decode position, clamped to ctx. Prefill
    // calls pick the window the decode model of the compiled pair is built with.
    const int decode_bucket = (!no_kv_cache && cache_enabled && inp_pos != nullptr && inp_pos->type == GGML_TYPE_I32) ?
                                  ggml_openvino_npu_decode_bucket() :
                                  0;
    int decode_window = m_params.ctx_per_seq;
    // With decode KV windows and sliced (not head-major) KV, prefill sizes follow the decode KV windows.
    const bool prefill_window_aligned = decode_bucket > 0 && !ggml_openvino_npu_kv_head_major_enabled();
    if (decode_bucket > 0 && m_params.ctx_per_seq > 0) {
        const int last_pos = static_cast<const int32_t *>(inp_pos->data)[std::max<int64_t>(input_tokens, 1) - 1];
        const int needed = last_pos + (is_prefill ? 2 : 1);
        decode_window = decode_bucket;
        while (decode_window < needed && decode_window < m_params.ctx_per_seq) {
            decode_window *= 2;
        }
        decode_window = std::min(decode_window, m_params.ctx_per_seq);
        if (ggml_openvino_npu_kv_head_major_enabled()) {
            // Head-major KV is bound whole, so prefill and decode both use the window of the storage capacity.
            decode_window = std::min(std::max(decode_window, kv_storage_rows(cgraph)), m_params.ctx_per_seq);
            c_params.attention_size = decode_window;
        } else if (is_prefill && prefill_window_aligned && c_params.attention_size > 0) {
            // A prefill attends at least the decode KV window, so the prefill sizes are the window sizes and the
            // weightless-bank export can build them all up front (see the prefill window export below).
            c_params.attention_size = std::min(std::max(c_params.attention_size, decode_window), m_params.ctx_per_seq);
        }
        if (!is_prefill) {
            c_params.attention_size = decode_window;
        }
    }

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;
    int64_t ov_raw_infer_start;
    int64_t ov_raw_infer_total = 0;

    std::shared_ptr<decoder_runtime_ctx> entry;
    ModelParams old_m_params;

    if (cache_enabled) {
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        auto it = r_ctx->decoder_cache.find(key);
        cache_hit = it != r_ctx->decoder_cache.end();
        if (cache_hit) {
            entry = it->second;
            r_ctx->touch_locked(key);
        } else {
            auto mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[key] = entry;
            r_ctx->admit_locked(key);
        }
    } else {
        auto mutex = std::make_shared<std::mutex>();
        entry = std::make_shared<decoder_runtime_ctx>(mutex);
        cache_hit = false;
    }

    std::lock_guard<std::mutex> lock(*(entry->mutex));
    cache_hit = cache_hit && entry->ptr;
    if (cache_hit) {
        cache_hit = r_ctx->infer_request_cache.count(key) != 0 &&
                    r_ctx->infer_request_cache_prefill.count(key) != 0;
    }

    // Same model, only the prefill attention size differs from the entry's pair.
    bool prefill_size_changed = false;
    if (cache_hit) {
        ggml_decoder = entry->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_hit = old_m_params.can_reuse_statically(m_params);
        if (cache_hit && is_prefill && ggml_openvino_kv_slice_enabled()) {
            // A larger compiled prefill also serves a smaller attended length: the extra KV rows are masked, as with
            // the power-of-two rounding. Without this, a new prompt walks up the sizes again, and since only one
            // prefill model is kept, every prompt longer than one chunk recompiles. Head-major KV is bound at the
            // storage size, so it keeps the exact match.
            // The output count is part of the static prefill shape (more than 1 only for llama-perplexity).
            const bool reuse_larger = !ggml_openvino_npu_kv_head_major_enabled();
            cache_hit = entry->prefill_output_len == c_params.output_len &&
                        (reuse_larger ? entry->prefill_attention_size >= c_params.attention_size &&
                                            entry->prefill_attention_size_swa >= c_params.attention_size_swa :
                                        entry->prefill_attention_size == c_params.attention_size &&
                                            entry->prefill_attention_size_swa == c_params.attention_size_swa);
            if (cache_hit) {
                c_params.attention_size = entry->prefill_attention_size;
                c_params.attention_size_swa = entry->prefill_attention_size_swa;
            }
            prefill_size_changed = !cache_hit;
        }
    }

    std::vector<std::string> ov_input_names_local;
    bool pair_built = false;
    std::vector<std::string> ov_output_names_local;

    if (cache_hit) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->set_model_params(m_params);
        ggml_decoder->set_compute_params(c_params);
        if (old_m_params.kv_buffer_changed(m_params) || !ggml_decoder->is_bound_to(cgraph)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->add_extra_inputs();
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(key) : r_ctx->infer_request_cache.at(key);
            ov_input_names_local = is_prefill ? r_ctx->ov_input_names_cache_prefill.at(key) :
                                                r_ctx->ov_input_names_cache.at(key);
            ov_output_names_local = is_prefill ? r_ctx->ov_output_names_cache_prefill.at(key) :
                                                 r_ctx->ov_output_names_cache.at(key);
            decode_request_for_outputs = r_ctx->infer_request_cache.at(key);
            decode_input_names_for_outputs = r_ctx->ov_input_names_cache.at(key);
            decode_output_names_for_outputs = r_ctx->ov_output_names_cache.at(key);
        }

        decoder_end_time = ggml_time_us();
        conversion_end_time = decoder_end_time;
        compile_end_time = decoder_end_time;
    } else {
        // Same fail-fast as the dynamic path: once the host weight pages have been dropped
        // they read as zeros, so a recompile here would bake garbage into the new model.
        if (ggml_openvino_weight_buffers_released()) {
            GGML_LOG_ERROR("ggml-openvino: cannot compile a new graph after releasing host weights\n");
            return GGML_STATUS_FAILED;
        }
        if (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(key);
            r_ctx->infer_request_cache_prefill.erase(key);
        }

        // Static execution shares a compiled prefill/decode pair. Each backend
        // creates and retains its own requests for both phases.
        auto shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_lock(shared_cache->mutex);
        auto weight_names = get_weight_names(cgraph);
        auto local_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, weight_names, is_static,
                                                             stateful, false, is_prefill, prefill_chunk_size);
        local_decoder->device = device;
        const std::string shared_key =
            cache_enabled ? compiled_graph_key(cgraph, *local_decoder, device, prefill_chunk_size) +
                                (kv_only_prefill ? "|kv-only-prefill" : "") :
                            "";
        auto shared_it = shared_cache->graphs.find(shared_key);
        if (!shared_key.empty() && shared_it != shared_cache->graphs.end()) {
            auto & compiled = shared_it->second;
            auto prefill_request = std::make_shared<ov::InferRequest>(compiled.prefill.create_infer_request());
            auto decode_request = no_kv_cache ?
                                      prefill_request :
                                      std::make_shared<ov::InferRequest>(compiled.decode.create_infer_request());
            ggml_decoder = local_decoder;
            entry->ptr = ggml_decoder;
            entry->prefill_attention_size = c_params.attention_size;
            entry->prefill_attention_size_swa = c_params.attention_size_swa;
            entry->prefill_output_len = c_params.output_len;
            entry->decode_attention_size = compiled.decode_attention_size;
            entry->shared_key = shared_key;
            entry->decode_window_requests.clear();
            infer_request = is_prefill ? prefill_request : decode_request;
            ov_input_names_local = is_prefill ? compiled.prefill_input_names : compiled.input_names;
            ov_output_names_local = is_prefill ? compiled.prefill_output_names : compiled.output_names;
            r_ctx->infer_request_cache_prefill[key] = prefill_request;
            r_ctx->infer_request_cache[key] = decode_request;
            r_ctx->ov_input_names_cache[key] = compiled.input_names;
            r_ctx->ov_output_names_cache[key] = compiled.output_names;
            r_ctx->ov_input_names_cache_prefill[key] = compiled.prefill_input_names;
            r_ctx->ov_output_names_cache_prefill[key] = compiled.prefill_output_names;
            entry->kv_only_prefill = compiled.kv_only_prefill;
            decode_request_for_outputs = decode_request;
            decode_input_names_for_outputs = compiled.input_names;
            decode_output_names_for_outputs = compiled.output_names;
            decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (static)\n");
            if (prefill_size_changed) {
                GGML_LOG_INFO("ggml-openvino: reuse the compiled prefill model of attention size %d\n",
                              c_params.attention_size);
            }
        } else {
            auto decode_c_params = c_params;
            // A decode step outputs its one token. llama-perplexity asks for many outputs in the prefill graph; the
            // decode model of the pair keeps one (decode_signature below), or its last layer would gather more rows
            // than the graph has tokens.
            decode_c_params.output_len = 1;
            if (ggml_openvino_kv_slice_enabled()) {
                decode_c_params.attention_size = m_params.ctx_per_seq;
                decode_c_params.attention_size_swa = m_params.ctx_per_seq_swa;
            }
            if (decode_bucket > 0) {
                decode_c_params.attention_size = decode_window;
            }
            const ov::AnyMap prefill_config = ggml_openvino_npu_prefill_config(compile_config, *local_decoder);
            const ov::AnyMap decode_config = ggml_openvino_npu_decode_config(compile_config, device);

            const std::string model_cache_dir = ggml_openvino_model_cache_dir();
            const bool self_contained = device == "NPU" && !model_cache_dir.empty() &&
                                        ggml_openvino_npu_raw_weight_cache_enabled();
            // Weightless prefill/decode blobs plus one shared weights file. Needs NPUW, which cacheless graphs
            // compile without, so those keep the embedded-weight blob.
            const bool weightless_bank =
                self_contained && !no_kv_cache && ggml_openvino_npu_weightless_bank_enabled();
            // Another prefill attention size of the entry's model. The decode models (base and KV windows) do not
            // depend on it, so the new pair takes them over from the entry's current pair, and that pair is dropped
            // before the new prefill model compiles: each loaded prefill model keeps NPU memory for its inferences, and
            // next to a second one the NPU can run out (pfnAppendGraphExecute: ZE_RESULT_ERROR_UNINITIALIZED). A pair
            // another backend may use stays cached. Self-contained pairs are exported together, so they keep their own
            // decode.
            std::optional<ov_compiled_graph> decode_source;
            if (prefill_size_changed && !self_contained && !no_kv_cache) {
                auto source_it = shared_cache->graphs.find(entry->shared_key);
                if (source_it != shared_cache->graphs.end()) {
                    decode_source = source_it->second;
                    decode_source->prefill = {};
                    decode_c_params = decode_source->decode_c_params;
                    // Make the KV window of the next decode step the base decode model, as a newly built pair has
                    // it, so that step does not switch to a window request it has to create first.
                    auto window_it = decode_source->decode_windows.find(decode_window);
                    if (decode_bucket > 0 && window_it != decode_source->decode_windows.end()) {
                        ov::CompiledModel window_model = window_it->second;
                        decode_source->decode_windows.erase(window_it);
                        decode_source->decode_windows.emplace(decode_c_params.attention_size, decode_source->decode);
                        decode_source->decode = window_model;
                        decode_c_params.attention_size = decode_window;
                    }
                    if (shared_cache->backend_count == 1) {
                        shared_cache->graphs.erase(source_it);
                    }
                }
            }
            ov::AnyMap mc_config = prefill_config;
            ov::AnyMap mc_decode_config = decode_config;
            if (!model_cache_dir.empty()) {
                mc_config.erase("CACHE_DIR");
                mc_config.erase("CACHE_MODE");
                mc_decode_config.erase("CACHE_DIR");
                mc_decode_config.erase("CACHE_MODE");
            }

            uint64_t prefill_fp = 0;
            uint64_t decode_fp = 0;
            uint64_t base_extra = 0;
            const bool decode_unfold = mc_decode_config.find("NPUW_UNFOLD_IREQS") != mc_decode_config.end();
            if (!model_cache_dir.empty()) {
                base_extra = ggml_openvino_model_cache_extra_cfg(stateful, /*recurrent=*/false);
                base_extra = base_extra * 131 + (weightless_bank ? 3u : (self_contained ? 1u : 0u));
                base_extra = ggml_openvino_model_cache_mix_string(
                    base_extra, ggml_openvino_getenv_str("GGML_OPENVINO_NPU_TOKEN_EMBD_QUANT"));
                // KV slice window base; the default base (1024) keeps the fingerprint of the former on/off flag.
                const int kv_slice_base = ggml_openvino_kv_slice_base();
                base_extra = base_extra * 131 +
                             static_cast<uint64_t>(kv_slice_base == 1024 ? 1 : kv_slice_base);
                base_extra = base_extra * 131 + (ggml_openvino_npu_kv_head_major_enabled() ? 1u : 0u);
                base_extra = base_extra * 131 + (kv_only_prefill ? 1u : 0u);
                base_extra = ggml_openvino_model_cache_mix_string(
                    base_extra, ggml_openvino_getenv_str("GGML_OPENVINO_NPU_CONFIG"));
                base_extra = ggml_openvino_model_cache_mix_string(
                    base_extra, ggml_openvino_getenv_str("GGML_OPENVINO_NPU_COMPILE_CONFIG"));
                base_extra = ggml_openvino_model_cache_mix_string(
                    base_extra, ggml_openvino_getenv_str("GGML_OPENVINO_NPU_COMPILER_TYPE"));
                base_extra = ggml_openvino_model_cache_mix_string(
                    base_extra, ggml_openvino_getenv_str("GGML_OPENVINO_NPUW_FUNCALL_FOR_ALL"));
                base_extra = ggml_openvino_model_cache_mix_string(
                    base_extra, ggml_openvino_getenv_str("GGML_OPENVINO_NPUW_UNFOLD_IREQS"));
                base_extra = ggml_openvino_model_cache_mix_string(
                    base_extra, ggml_openvino_getenv_str("GGML_OPENVINO_COMPILATION_NUM_THREADS"));
                // out_ids is bound at a fixed length, so the output count is part of the static shape
                const std::string prefill_signature = "static-1 out=" + std::to_string(c_params.output_len);
                const std::string decode_signature = "static-1 out=1";
                prefill_fp = ggml_openvino_model_fingerprint(
                    cgraph, device, true, m_params.rope_params, 16,
                    static_model_salt(base_extra, true, prefill_chunk_size, c_params.attention_size,
                                      c_params.attention_size_swa, false),
                    prefill_signature);
                decode_fp = ggml_openvino_model_fingerprint(
                    cgraph, device, true, m_params.rope_params, 16,
                    static_model_salt(base_extra, false, prefill_chunk_size, decode_c_params.attention_size,
                                      decode_c_params.attention_size_swa, decode_unfold),
                    decode_signature);
            }

            std::string weights_path;
            std::string pair_weights_path;
            std::string model_weights_path;
            std::string pair_bank;
            std::string model_bank;
            if (weightless_bank) {
                // One weights file and NPUW bank for every pair of this model (pairs differ only in KV window):
                // NPUW matches weights by the LazyTensor hash recorded at compile time and keeps one resident copy.
                // A pair whose weights differ from the model file falls back to its own file and bank.
                const uint64_t weights_fp = ggml_openvino_model_fingerprint(cgraph, device, true, m_params.rope_params,
                                                                            16, base_extra, "weights");
                char model_bank_name[64];
                char shared_bank[96];
                snprintf(model_bank_name, sizeof(model_bank_name), "ggml-wl-model-%016llx",
                         static_cast<unsigned long long>(weights_fp));
                snprintf(shared_bank, sizeof(shared_bank), "ggml-wl-shared-%016llx-%016llx",
                         static_cast<unsigned long long>(prefill_fp), static_cast<unsigned long long>(decode_fp));
                pair_bank = shared_bank;
                model_bank = model_bank_name;
                pair_weights_path = ggml_openvino_model_cache_weights_path(model_cache_dir, prefill_fp, decode_fp);
                model_weights_path = ggml_openvino_model_cache_weights_path(model_cache_dir, weights_fp, weights_fp);
                // Only a pair exported with differing weights has its own file.
                const bool pair_own = ggml_openvino_model_cache_weights_valid(pair_weights_path);
                weights_path = pair_own ? pair_weights_path : model_weights_path;
                mc_config["NPUW_WEIGHTS_BANK"] = pair_own ? pair_bank : model_bank;
                mc_decode_config["NPUW_WEIGHTS_BANK"] = mc_config["NPUW_WEIGHTS_BANK"];
            } else if (self_contained) {
                char prefill_bank[64];
                char decode_bank[64];
                snprintf(prefill_bank, sizeof(prefill_bank), "ggml-sc-prefill-%016llx",
                         static_cast<unsigned long long>(prefill_fp));
                snprintf(decode_bank, sizeof(decode_bank), "ggml-sc-decode-%016llx",
                         static_cast<unsigned long long>(decode_fp));
                if (ggml_openvino_getenv_int("GGML_OPENVINO_NPU_SHARED_BANK")) {
                    // Share ONE weights bank between prefill and decode. Both decoders are built
                    // from the same model_weights map, so NPUW dedups them into a single in-process
                    // bank; on warm import the shared blob's identical (uid,tensor) pairs are
                    // content-verified and reused (patched NPUW weights_bank). Halves the resident
                    // compiled-weight footprint (~2 GiB) at the cost of larger per-graph blobs.
                    char shared_bank[96];
                    snprintf(shared_bank, sizeof(shared_bank), "ggml-sc-shared-%016llx-%016llx",
                             static_cast<unsigned long long>(prefill_fp),
                             static_cast<unsigned long long>(decode_fp));
                    mc_config["NPUW_WEIGHTS_BANK"] = std::string(shared_bank);
                    mc_decode_config["NPUW_WEIGHTS_BANK"] = std::string(shared_bank);
                } else {
                    mc_config["NPUW_WEIGHTS_BANK"] = std::string(prefill_bank);
                    mc_decode_config["NPUW_WEIGHTS_BANK"] = std::string(decode_bank);
                }
            }

            auto blob_valid = [&](uint64_t fp) {
                if (model_cache_dir.empty() || fp == 0) {
                    return false;
                }
                std::ifstream blob_in(ggml_openvino_model_cache_blob_path(model_cache_dir, fp), std::ios::binary);
                std::vector<std::string> inputs, outputs;  // ports are bound in frontend order, see below
                return blob_in.is_open() &&
                       ggml_openvino_model_cache_verify_manifest(
                           ggml_openvino_model_cache_manifest_path(model_cache_dir, fp), cgraph, fp, inputs, outputs);
            };
            const bool self_contained_mmap =
                self_contained && ggml_openvino_getenv_int("GGML_OPENVINO_SELF_CONTAINED_MMAP") != 0;
            const bool skip_weights = self_contained && blob_valid(prefill_fp) && blob_valid(decode_fp) &&
                                      (!weightless_bank || ggml_openvino_model_cache_weights_valid(weights_path));
            if (self_contained_mmap && !skip_weights) {
                throw std::runtime_error(
                    "GGML_OPENVINO_SELF_CONTAINED_MMAP requires valid self-contained blob(s); "
                    "run once with GGML_OPENVINO_SELF_CONTAINED_MMAP=0 to build the cache");
            }

            if (self_contained && !skip_weights && !model_cache_dir.empty()) {
                if (prefill_fp != 0) {
                    ggml_openvino_model_cache_write_manifest(
                        ggml_openvino_model_cache_manifest_path(model_cache_dir, prefill_fp) + ".source.tmp",
                        cgraph, prefill_fp, {}, {});
                }
                if (decode_fp != 0) {
                    ggml_openvino_model_cache_write_manifest(
                        ggml_openvino_model_cache_manifest_path(model_cache_dir, decode_fp) + ".source.tmp",
                        cgraph, decode_fp, {}, {});
                }
            }

            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            if (skip_weights) {
                for (const auto & name : GgmlOvDecoder::collect_weight_names(cgraph)) {
                    model_weights[name] = nullptr;
                }
            } else {
                model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);
            }

            auto ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(
                cgraph, m_params, c_params, model_weights, is_static, stateful, false, true, prefill_chunk_size);
            ggml_decoder_prefill->set_kv_only_outputs(kv_only_prefill);
            auto ggml_decoder_decode =
                no_kv_cache ? ggml_decoder_prefill :
                              std::make_shared<GgmlOvDecoder>(cgraph, m_params, decode_c_params, model_weights,
                                                              is_static, stateful, false, false, prefill_chunk_size);
            ggml_decoder_prefill->device = device;
            ggml_decoder_decode->device = device;
            decoder_end_time = ggml_time_us();

            const bool dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
            const auto dump_ir_timestamp = static_cast<long long>(ggml_time_us());

            // Both decoders share the weight nodes, and converting attaches each new graph to them. ov::Node is not
            // thread-safe, so the models are always converted one at a time; only compilation can run in parallel.
            const static_build_args build_args{&core,           device,         dump_ir,        dump_ir_timestamp,
                                               model_cache_dir, cgraph,         self_contained, skip_weights,
                                               weightless_bank, weights_path};
            auto convert_static_model = [&build_args](std::shared_ptr<GgmlOvDecoder> decoder, const char * tag,
                                                      int64_t & local_conversion_end_time) {
                return ::convert_static_model(build_args, decoder, tag, local_conversion_end_time);
            };
            auto build_static_model = [&build_args](std::shared_ptr<GgmlOvDecoder> decoder,
                                                    const char * tag,
                                                    std::shared_ptr<ov::Model> model,
                                                    const ov::AnyMap & model_config,
                                                    uint64_t model_fp,
                                                    ov::CompiledModel & compiled_model,
                                                    std::shared_ptr<ov::InferRequest> & infer_request,
                                                    std::vector<std::string> & names_in,
                                                    std::vector<std::string> & names_out,
                                                    int64_t & local_conversion_end_time,
                                                    int64_t & local_compile_end_time) {
                ::build_static_model(build_args, std::move(decoder), tag, std::move(model), model_config, model_fp,
                                     compiled_model, infer_request, names_in, names_out, local_conversion_end_time,
                                     local_compile_end_time);
            };
            ov::CompiledModel compiled_model_prefill;
            ov::CompiledModel compiled_model_decode;
            std::shared_ptr<ov::InferRequest> infer_request_prefill;
            std::shared_ptr<ov::InferRequest> infer_request_decode;
            std::vector<std::string> names_in_prefill, names_out_prefill;
            std::vector<std::string> names_in_decode, names_out_decode;
            int64_t prefill_conversion_end_time = 0;
            int64_t decode_conversion_end_time = 0;
            int64_t prefill_compile_end_time = 0;
            int64_t decode_compile_end_time = 0;

            // Convert decode before compiling prefill: a decode model converted after the prefill compile gives
            // wrong output on some NPU models (Bonsai-8B). Compiling both in parallel raises the host memory peak,
            // so it is opt-in.
            const bool parallel_compile =
                !self_contained && !no_kv_cache && !decode_source && ggml_openvino_npu_parallel_compile_enabled();
            auto model_prefill = convert_static_model(ggml_decoder_prefill, "prefill", prefill_conversion_end_time);
            std::shared_ptr<ov::Model> model_decode;
            if (!no_kv_cache && !decode_source) {
                model_decode = convert_static_model(ggml_decoder_decode, "decode", decode_conversion_end_time);
            }
            std::shared_ptr<ggml_openvino_weights_layout> weights_layout;
            if (weightless_bank && !skip_weights) {
                // Lay both models' weights out in one file before compiling, so both blobs reference the same file.
                // Old blobs reference the old layout, so drop them first in case the export does not complete.
                for (const uint64_t fp : { prefill_fp, decode_fp }) {
                    std::remove(ggml_openvino_model_cache_blob_path(model_cache_dir, fp).c_str());
                    std::remove(ggml_openvino_model_cache_manifest_path(model_cache_dir, fp).c_str());
                    // A decode window built earlier on its own may have left a weights file for this fingerprint.
                    ggml_openvino_model_cache_remove_weights(
                        ggml_openvino_model_cache_weights_path(model_cache_dir, fp, fp));
                }
                ggml_openvino_model_cache_remove_weights(pair_weights_path);
                ggml_openvino_weights_status status = ggml_openvino_weights_status::written;
                weights_path = model_weights_path;
                mc_config["NPUW_WEIGHTS_BANK"] = model_bank;
                mc_decode_config["NPUW_WEIGHTS_BANK"] = model_bank;
                weights_layout = ggml_openvino_model_cache_write_weights(weights_path, {model_prefill, model_decode},
                                                                         /*keep_existing=*/true, &status);
                if (status == ggml_openvino_weights_status::mismatch) {
                    GGML_LOG_WARN("ggml-openvino: weights differ from %s, using a per-pair weights file\n",
                                  model_weights_path.c_str());
                    weights_path = pair_weights_path;
                    mc_config["NPUW_WEIGHTS_BANK"] = pair_bank;
                    mc_decode_config["NPUW_WEIGHTS_BANK"] = pair_bank;
                    weights_layout =
                        ggml_openvino_model_cache_write_weights(weights_path, {model_prefill, model_decode});
                }
                if (!weights_layout) {
                    // E.g. a full disk. The models still compile and run; only this pair is left uncached, since its
                    // blobs would reference a missing weights file.
                    GGML_LOG_WARN("ggml-openvino: could not write the NPU weights file %s; not caching this graph\n",
                                  weights_path.c_str());
                    for (const uint64_t fp : { prefill_fp, decode_fp }) {
                        std::remove((ggml_openvino_model_cache_manifest_path(model_cache_dir, fp) + ".source.tmp").c_str());
                    }
                    prefill_fp = decode_fp = 0;
                }
            }
            if (parallel_compile) {
                auto prefill_future = std::async(std::launch::async, build_static_model, ggml_decoder_prefill, "prefill",
                                                 std::move(model_prefill), std::cref(mc_config), prefill_fp,
                                                 std::ref(compiled_model_prefill), std::ref(infer_request_prefill),
                                                 std::ref(names_in_prefill), std::ref(names_out_prefill),
                                                 std::ref(prefill_conversion_end_time),
                                                 std::ref(prefill_compile_end_time));
                auto decode_future = std::async(std::launch::async, build_static_model, ggml_decoder_decode, "decode",
                                                std::move(model_decode), std::cref(mc_decode_config), decode_fp,
                                                std::ref(compiled_model_decode), std::ref(infer_request_decode),
                                                std::ref(names_in_decode), std::ref(names_out_decode),
                                                std::ref(decode_conversion_end_time),
                                                std::ref(decode_compile_end_time));
                prefill_future.get();
                decode_future.get();
            } else {
                build_static_model(ggml_decoder_prefill, "prefill", std::move(model_prefill), mc_config, prefill_fp,
                                   compiled_model_prefill, infer_request_prefill, names_in_prefill, names_out_prefill,
                                   prefill_conversion_end_time, prefill_compile_end_time);
                if (no_kv_cache) {
                    compiled_model_decode = compiled_model_prefill;
                    infer_request_decode = infer_request_prefill;
                    names_in_decode = names_in_prefill;
                    names_out_decode = names_out_prefill;
                    decode_conversion_end_time = prefill_conversion_end_time;
                    decode_compile_end_time = prefill_compile_end_time;
                } else if (decode_source) {
                    compiled_model_decode = decode_source->decode;
                    infer_request_decode =
                        std::make_shared<ov::InferRequest>(compiled_model_decode.create_infer_request());
                    names_in_decode = decode_source->input_names;
                    names_out_decode = decode_source->output_names;
                    decode_conversion_end_time = prefill_conversion_end_time;
                    decode_compile_end_time = prefill_compile_end_time;
                } else {
                    build_static_model(ggml_decoder_decode, "decode", std::move(model_decode), mc_decode_config,
                                       decode_fp, compiled_model_decode, infer_request_decode, names_in_decode,
                                       names_out_decode, decode_conversion_end_time, decode_compile_end_time);
                }
            }
            conversion_end_time = std::max(prefill_conversion_end_time, decode_conversion_end_time);
            compile_end_time = std::max(prefill_compile_end_time, decode_compile_end_time);

            ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
            infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
            ov_input_names_local = is_prefill ? names_in_prefill : names_in_decode;
            ov_output_names_local = is_prefill ? names_out_prefill : names_out_decode;
            entry->ptr = ggml_decoder;
            entry->prefill_attention_size = c_params.attention_size;
            entry->prefill_attention_size_swa = c_params.attention_size_swa;
            entry->prefill_output_len = c_params.output_len;
            entry->decode_attention_size = decode_c_params.attention_size;
            entry->shared_key = shared_key;
            entry->decode_window_requests.clear();
            entry->kv_only_prefill = kv_only_prefill;
            decode_request_for_outputs = infer_request_decode;
            decode_input_names_for_outputs = names_in_decode;
            decode_output_names_for_outputs = names_out_decode;
            if (kv_only_prefill) {
                GGML_LOG_INFO("ggml-openvino: KV-only prefill model (outputs via one decode step)\n");
            }

            if (!shared_key.empty()) {
                ov_compiled_graph graph;
                graph.decode = compiled_model_decode;
                graph.prefill = compiled_model_prefill;
                graph.input_names = names_in_decode;
                graph.output_names = names_out_decode;
                graph.prefill_input_names = names_in_prefill;
                graph.prefill_output_names = names_out_prefill;
                graph.decode_attention_size = decode_c_params.attention_size;
                if (decode_source) {
                    graph.decode_windows = decode_source->decode_windows;
                }
                graph.decode_config = mc_decode_config;
                graph.cache_base_extra = base_extra;
                graph.prefill_fp = prefill_fp;
                graph.decode_fp = decode_fp;
                graph.self_contained = self_contained;
                graph.decode_unfold = decode_unfold;
                graph.weightless_bank = weightless_bank;
                graph.weights_path = weights_path;
                graph.weights_layout = decode_bucket > 0 ? std::move(weights_layout) : nullptr;
                graph.decode_c_params = decode_c_params;
                graph.kv_only_prefill = kv_only_prefill;
                shared_cache->graphs.emplace(shared_key, std::move(graph));
                pair_built = !decode_source;
            }

            if (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache_prefill[key] = infer_request_prefill;
                r_ctx->infer_request_cache[key] = infer_request_decode;
                r_ctx->ov_input_names_cache[key] = names_in_decode;
                r_ctx->ov_output_names_cache[key] = names_out_decode;
                r_ctx->ov_input_names_cache_prefill[key] = names_in_prefill;
                r_ctx->ov_output_names_cache_prefill[key] = names_out_prefill;
            }
        }
    }

    // Decode model for another KV window than the pair's; built/imported once per window with the pair's
    // settings and weights bank, then cached in the shared compiled cache. Caller holds the shared cache mutex.
    auto build_decode_window = [&](ov_compiled_graph & graph, const ComputeParams & base_c_params,
                                   int window) -> ov::CompiledModel & {
        auto window_it = graph.decode_windows.find(window);
        if (window_it != graph.decode_windows.end()) {
            return window_it->second;
        }
        auto window_c_params = base_c_params;
        window_c_params.attention_size = window;
        const std::string model_cache_dir = ggml_openvino_model_cache_dir();
        const uint64_t fp = model_cache_dir.empty() ?
                                0 :
                                ggml_openvino_model_fingerprint(
                                    cgraph, device, true, m_params.rope_params, 16,
                                    static_model_salt(graph.cache_base_extra, false, prefill_chunk_size, window,
                                                      window_c_params.attention_size_swa, graph.decode_unfold),
                                    "static-1 out=1");
        if (fp != 0) {
            std::lock_guard<std::mutex> blobs_lock(g_imported_blobs_mutex);
            auto cached = g_imported_blobs.find(fp);
            if (cached != g_imported_blobs.end()) {
                return graph.decode_windows.emplace(window, cached->second).first->second;
            }
        }
        // Weightless bank: a window compiled with the pair resolves the shared weights through the pair's file
        // (the rest is embedded in its blob); otherwise it gets its own weights file.
        const std::string own_weights_path =
            graph.weightless_bank ? ggml_openvino_model_cache_weights_path(model_cache_dir, fp, fp) : "";
        const bool own_weights_valid =
            graph.weightless_bank && ggml_openvino_model_cache_weights_valid(own_weights_path);
        // A window blob resolved through the pair's file must not predate it (the pair may have been re-exported).
        auto pair_weights_usable = [&]() {
            std::error_code ec1, ec2;
            const auto blob_time = std::filesystem::last_write_time(
                std::filesystem::u8path(ggml_openvino_model_cache_blob_path(model_cache_dir, fp)), ec1);
            const auto weights_time =
                std::filesystem::last_write_time(std::filesystem::u8path(graph.weights_path), ec2);
            return !ec1 && !ec2 && blob_time >= weights_time &&
                   ggml_openvino_model_cache_weights_valid(graph.weights_path);
        };
        const bool skip_weights = graph.self_contained && static_blob_valid(model_cache_dir, fp, cgraph) &&
                                  (!graph.weightless_bank || own_weights_valid || pair_weights_usable());
        if (graph.self_contained && !skip_weights &&
            ggml_openvino_getenv_int("GGML_OPENVINO_SELF_CONTAINED_MMAP") != 0) {
            throw std::runtime_error("GGML_OPENVINO_SELF_CONTAINED_MMAP: no cached decode model for KV window " +
                                     std::to_string(window) + "; rerun with GGML_OPENVINO_SELF_CONTAINED_MMAP=0");
        }
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        if (skip_weights) {
            for (const auto & name : GgmlOvDecoder::collect_weight_names(cgraph)) {
                model_weights[name] = nullptr;
            }
        } else {
            if (graph.self_contained && fp != 0) {
                ggml_openvino_model_cache_write_manifest(
                    ggml_openvino_model_cache_manifest_path(model_cache_dir, fp) + ".source.tmp", cgraph, fp, {}, {});
            }
            model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);
        }
        auto window_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, window_c_params, model_weights,
                                                              is_static, stateful, false, false, prefill_chunk_size);
        window_decoder->device = device;
        static_build_args args{&core,  device,       false, 0, model_cache_dir, cgraph, graph.self_contained,
                               skip_weights, graph.weightless_bank, {}};
        ov::CompiledModel compiled;
        std::shared_ptr<ov::InferRequest> built_request;
        std::vector<std::string> names_in, names_out;
        int64_t t_conv = 0, t_compile = 0;
        const int64_t t0 = ggml_time_us();
        auto model = convert_static_model(args, window_decoder, "decode", t_conv);
        if (graph.weightless_bank) {
            args.weights_path = own_weights_valid ? own_weights_path : graph.weights_path;
            if (model && graph.weights_layout) {
                const size_t resolved = ggml_openvino_model_cache_retag_weights(model, *graph.weights_layout);
                GGML_LOG_INFO("ggml-openvino: decode KV window %d resolves %zu weights through %s\n", window,
                              resolved, graph.weights_path.c_str());
                ggml_openvino_model_cache_remove_weights(own_weights_path);
                args.weights_path = graph.weights_path;
            } else if (model) {
                if (!ggml_openvino_model_cache_write_weights(own_weights_path, { model })) {
                    throw std::runtime_error("ggml-openvino: failed to write NPU weights file " + own_weights_path);
                }
                args.weights_path = own_weights_path;
            }
        }
        build_static_model(args, window_decoder, "decode", std::move(model), graph.decode_config, fp, compiled,
                           built_request, names_in, names_out, t_conv, t_compile);
        if (names_in != graph.input_names || names_out != graph.output_names) {
            throw std::runtime_error("ggml-openvino: decode window model ports differ from the base decode model");
        }
        GGML_LOG_INFO("ggml-openvino: decode KV window %d model ready in %.1f ms\n", window,
                      (ggml_time_us() - t0) / 1000.0);
        return graph.decode_windows.emplace(window, std::move(compiled)).first->second;
    };

    auto get_decode_window_request = [&](int window) -> std::shared_ptr<ov::InferRequest> {
        auto & request = entry->decode_window_requests[window];
        if (request) {
            return request;
        }
        auto shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_lock(shared_cache->mutex);
        auto graph_it = shared_cache->graphs.find(entry->shared_key);
        if (graph_it == shared_cache->graphs.end()) {
            throw std::runtime_error("ggml-openvino: decode window requested without a shared compiled graph");
        }
        auto & compiled = build_decode_window(graph_it->second, graph_it->second.decode_c_params, window);
        request = std::make_shared<ov::InferRequest>(compiled.create_infer_request());
        return request;
    };

    // Build or import every other decode window right after the pair, from the pair's own graph and params,
    // so a growing sequence only switches requests.
    if (pair_built && decode_bucket > 0) {
        auto shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_lock(shared_cache->mutex);
        auto graph_it = shared_cache->graphs.find(entry->shared_key);
        if (graph_it != shared_cache->graphs.end()) {
            auto & graph = graph_it->second;
            const int64_t t0 = ggml_time_us();
            for (int window = decode_bucket;; window *= 2) {
                window = std::min(window, m_params.ctx_per_seq);
                if (window != graph.decode_attention_size) {
                    build_decode_window(graph, graph.decode_c_params, window);
                }
                if (window >= m_params.ctx_per_seq) {
                    break;
                }
            }
            GGML_LOG_INFO("ggml-openvino: prebuilt %zu decode KV window model(s) in %.1f ms\n",
                          graph.decode_windows.size(), (ggml_time_us() - t0) / 1000.0);

            // With head-major KV (or window-aligned prefill) a prefill also runs at a KV window, picked by prompt
            // length. Export the prefill model of every other window too, so an import-only run has a pair for any
            // prompt length.
            const std::string model_cache_dir = ggml_openvino_model_cache_dir();
            if (graph.weightless_bank && graph.weights_layout &&
                (ggml_openvino_npu_kv_head_major_enabled() || prefill_window_aligned) &&
                ggml_openvino_getenv_int("GGML_OPENVINO_SELF_CONTAINED_MMAP") == 0) {
                const int64_t t1 = ggml_time_us();
                size_t exported = 0;
                for (int window = decode_bucket;; window *= 2) {
                    window = std::min(window, m_params.ctx_per_seq);
                    if (window != entry->prefill_attention_size) {
                        auto window_c_params = c_params;
                        window_c_params.attention_size = window;
                        const uint64_t fp = ggml_openvino_model_fingerprint(
                            cgraph, device, true, m_params.rope_params, 16,
                            static_model_salt(graph.cache_base_extra, true, prefill_chunk_size, window,
                                              window_c_params.attention_size_swa, false),
                            "static-1 out=" + std::to_string(window_c_params.output_len));
                        std::error_code ec1, ec2;
                        const auto blob_time = std::filesystem::last_write_time(
                            std::filesystem::u8path(ggml_openvino_model_cache_blob_path(model_cache_dir, fp)), ec1);
                        const auto weights_time =
                            std::filesystem::last_write_time(std::filesystem::u8path(graph.weights_path), ec2);
                        const bool cached =
                            !ec1 && !ec2 && blob_time >= weights_time && static_blob_valid(model_cache_dir, fp, cgraph);
                        if (!cached) {
                            ggml_openvino_model_cache_write_manifest(
                                ggml_openvino_model_cache_manifest_path(model_cache_dir, fp) + ".source.tmp", cgraph,
                                fp, {}, {});
                            auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);
                            auto window_decoder =
                                std::make_shared<GgmlOvDecoder>(cgraph, m_params, window_c_params, model_weights,
                                                                is_static, stateful, false, true, prefill_chunk_size);
                            window_decoder->set_kv_only_outputs(graph.kv_only_prefill);
                            ov::AnyMap window_config =
                                ggml_openvino_npu_prefill_config(compile_config, *window_decoder);
                            window_config.erase("CACHE_DIR");
                            window_config.erase("CACHE_MODE");
                            window_config["NPUW_WEIGHTS_BANK"] = graph.decode_config.at("NPUW_WEIGHTS_BANK");
                            static_build_args args{&core,  device, false, 0,    model_cache_dir,
                                                   cgraph, true,   false, true, graph.weights_path};
                            int64_t t_conv = 0, t_compile = 0;
                            auto model = convert_static_model(args, window_decoder, "prefill", t_conv);
                            ggml_openvino_model_cache_retag_weights(model, *graph.weights_layout);
                            ov::CompiledModel compiled;
                            std::shared_ptr<ov::InferRequest> built_request;
                            std::vector<std::string> names_in, names_out;
                            build_static_model(args, window_decoder, "prefill", std::move(model), window_config, fp,
                                               compiled, built_request, names_in, names_out, t_conv, t_compile);
                            ++exported;
                        }
                    }
                    if (window >= m_params.ctx_per_seq) {
                        break;
                    }
                }
                GGML_LOG_INFO("ggml-openvino: exported %zu prefill KV window model(s) in %.1f ms\n", exported,
                              (ggml_time_us() - t1) / 1000.0);
            }

            // Import the prefill model of every other window now, so a prompt that needs another window builds its
            // pair from already imported models instead of importing them mid-request.
            if (graph.weightless_bank && ggml_openvino_npu_kv_head_major_enabled() &&
                ggml_openvino_model_cache_weights_valid(graph.weights_path)) {
                const int64_t t2 = ggml_time_us();
                size_t imported = 0;
                for (int window = decode_bucket;; window *= 2) {
                    window = std::min(window, m_params.ctx_per_seq);
                    if (window != entry->prefill_attention_size) {
                        auto window_c_params = c_params;
                        window_c_params.attention_size = window;
                        const uint64_t fp = ggml_openvino_model_fingerprint(
                            cgraph, device, true, m_params.rope_params, 16,
                            static_model_salt(graph.cache_base_extra, true, prefill_chunk_size, window,
                                              window_c_params.attention_size_swa, false),
                            "static-1 out=" + std::to_string(window_c_params.output_len));
                        bool already_imported;
                        {
                            std::lock_guard<std::mutex> blobs_lock(g_imported_blobs_mutex);
                            already_imported = g_imported_blobs.count(fp) != 0;
                        }
                        if (!already_imported && static_blob_valid(model_cache_dir, fp, cgraph)) {
                            std::map<std::string, std::shared_ptr<ov::Node>> weight_names;
                            for (const auto & name : GgmlOvDecoder::collect_weight_names(cgraph)) {
                                weight_names[name] = nullptr;
                            }
                            GgmlOvDecoder window_decoder(cgraph, m_params, window_c_params, weight_names, is_static,
                                                         stateful, false, true, prefill_chunk_size);
                            window_decoder.set_kv_only_outputs(graph.kv_only_prefill);
                            ov::AnyMap import_config = ggml_openvino_npu_prefill_config(compile_config, window_decoder);
                            import_config.erase("CACHE_DIR");
                            import_config.erase("CACHE_MODE");
                            import_config["NPUW_WEIGHTS_BANK"] = graph.decode_config.at("NPUW_WEIGHTS_BANK");
                            import_config[ov::weights_path.name()] = graph.weights_path;
                            import_config[ov::enable_weightless.name()] = true;
                            const std::string blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, fp);
                            try {
                                import_cached_blob(core, device, blob_path, fp, import_config, "prefill");
                                ++imported;
                            } catch (const std::exception & e) {
                                // Drop the invalid entry, as the pair import does; the window is then built
                                // when a prompt first needs it.
                                std::remove(blob_path.c_str());
                                std::remove(ggml_openvino_model_cache_manifest_path(model_cache_dir, fp).c_str());
                                GGML_LOG_WARN(
                                    "ggml-openvino: removed the invalid cached prefill model of KV window %d: %s\n",
                                    window, e.what());
                            }
                        }
                    }
                    if (window >= m_params.ctx_per_seq) {
                        break;
                    }
                }
                GGML_LOG_INFO("ggml-openvino: preloaded %zu prefill KV window model(s) in %.1f ms\n", imported,
                              (ggml_time_us() - t2) / 1000.0);
            }
        }
    }

    // On-demand KV: grow the storage to this call's KV window before binding, so every view of the call
    // (KV inputs, HFA past, KV outputs) addresses the same storage.
    bool kv_grew = false;
    size_t kv_rows = 0;
    if (ggml_openvino_npu_kv_on_demand_enabled()) {
        for (const auto & input_name : ov_input_names_local) {
            // A transposed-V HFA prefill only takes V through its past input; resolve it to the cache input.
            std::string name = input_name;
            if (input_name.rfind("ggml_hfa_past_", 0) == 0) {
                const std::string cache_name = input_name.substr(strlen("ggml_hfa_past_"));
                for (const auto & [model_input, _] : ggml_decoder->get_model_inputs()) {
                    if (model_input.substr(0, model_input.find('#')) == cache_name) {
                        name = model_input;
                        break;
                    }
                }
            }
            if (name.rfind("cache_k_l", 0) == 0 || name.rfind("cache_v_l", 0) == 0) {
                const auto * kv_tensor = ggml_decoder->get_input_ggml_tensor(name);
                size_t heads =
                    ggml_openvino_npu_kv_head_major_enabled() ? ggml_decoder->get_n_heads_kv_for_tensor(kv_tensor) : 0;
                if (heads > 0 && name.rfind("cache_v_l", 0) == 0) {
                    // [heads][head_size][rows] is head-major storage with one "head" per channel.
                    heads = static_cast<size_t>(kv_tensor->ne[0]);
                }
                kv_rows = get_static_attention_size(ggml_decoder, name);
                kv_grew |= ggml_openvino_kv_storage_reserve(kv_tensor->data, kv_rows, heads);
            }
        }
    }

    if (is_prefill) {
        auto inp_len = get_inp_pos_n_tokens(cgraph, inp_pos);
        // Transposed-V HFA prefill outputs each chunk's V^T rows (see set_rows); they are copied into the cache here.
        const bool vt_rows = ggml_openvino_npu_kv_head_major_enabled() && ggml_openvino_npu_hfa_prefill_enabled();
        auto is_vt_rows_output = [&](const std::string & name) {
            if (!vt_rows || name.rfind("cache_v_l", 0) != 0) {
                return false;
            }
            const auto layer = extract_layer_from_name(name);
            return layer.has_value() && !ggml_decoder->is_swa_layer(*layer);
        };
        std::map<const ggml_tensor *, const ggml_tensor *> set_rows_indices;
        if (vt_rows) {
            for (int n = 0; n < cgraph->n_nodes; n++) {
                const auto * node = cgraph->nodes[n];
                if (node->op == GGML_OP_SET_ROWS && node->view_src) {
                    set_rows_indices[node->view_src] = node->src[1];
                }
            }
        }
        for (int chunk_index = 0; chunk_index * prefill_chunk_size < inp_len; chunk_index++) {
            const int64_t chunk_start = static_cast<int64_t>(chunk_index) * prefill_chunk_size;
            const int64_t chunk_valid_len = std::min<int64_t>(prefill_chunk_size, inp_len - chunk_start);
            for (size_t i = 0; i < ov_input_names_local.size(); i++) {
                const auto & param_name = ov_input_names_local[i];
                auto input_tensor = get_ov_input_tensor_static_prefill(ggml_decoder, param_name, chunk_index);
                infer_request->set_input_tensor(i, input_tensor);

                if (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                    const auto input_tensor = infer_request->get_input_tensor(i);
                    print_input_tensor_info(param_name, input_tensor);
                }
            }

            for (size_t i = 0; i < ov_output_names_local.size(); i++) {
                const auto & model_outputs = ggml_decoder->get_model_outputs();
                auto model_output_it = model_outputs.find(ov_output_names_local[i]);
                if (model_output_it == model_outputs.end()) {
                    continue;
                }
                auto * ggml_tensor = model_output_it->second;
                if (ggml_nbytes(ggml_tensor) == 0) {
                    // Zero-row in-place writeback (e.g. the empty s_copy defrag remainder). The OV
                    // Result is the full cache, so binding it over this 0-byte buffer overflows it.
                    continue;
                }
                if (is_vt_rows_output(ov_output_names_local[i])) {
                    continue;
                }
                auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->set_output_tensor(i, output_tensor);
            }

            ov_raw_infer_start = ggml_time_us();
            infer_request->infer();
            const int64_t chunk_infer_us = ggml_time_us() - ov_raw_infer_start;
            ov_raw_infer_total += chunk_infer_us;

            if (vt_rows) {
                const size_t base_row = static_cast<size_t>(chunk_start);
                const size_t valid = static_cast<size_t>(chunk_valid_len);
                const auto & model_outputs = ggml_decoder->get_model_outputs();
                for (size_t i = 0; i < ov_output_names_local.size(); i++) {
                    if (!is_vt_rows_output(ov_output_names_local[i])) {
                        continue;
                    }
                    const auto * kv_tensor = model_outputs.at(ov_output_names_local[i]);
                    const auto * idx = set_rows_indices.at(kv_tensor);
                    auto row_index = [&](size_t r) -> int64_t {
                        return idx->type == GGML_TYPE_I64 ? static_cast<const int64_t *>(idx->data)[base_row + r] :
                                                            static_cast<const int32_t *>(idx->data)[base_row + r];
                    };
                    const auto rows = infer_request->get_output_tensor(i);
                    for (size_t r0 = 0; r0 < valid;) {
                        const int64_t dst0 = row_index(r0);
                        size_t len = 1;
                        while (r0 + len < valid && row_index(r0 + len) == dst0 + static_cast<int64_t>(len)) {
                            ++len;
                        }
                        if (!ggml_openvino_kv_storage_write_rows(kv_tensor->data, rows.data(), rows.get_shape()[3], r0,
                                                                 static_cast<size_t>(dst0), len)) {
                            throw std::runtime_error("ggml-openvino: transposed V rows need head-major KV storage");
                        }
                        r0 += len;
                    }
                }
            }
            if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                fprintf(stderr,
                        "GGML OpenVINO profiling: prefill chunk %d: valid_len=%lld, padded_len=%d, logical_history=%lld, raw infer=%.3f ms\n",
                        chunk_index, static_cast<long long>(chunk_valid_len), prefill_chunk_size,
                        static_cast<long long>(chunk_start), chunk_infer_us / 1000.0);
            }

            if (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                for (size_t i = 0; i < ov_output_names_local.size(); i++) {
                    const auto output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_names_local[i], output_tensor, nullptr);
                }
            }
        }

        if (entry->kv_only_prefill) {
            // The prefill model only filled the KV cache. Compute the outputs of the requested token by running the
            // decode model on it: it rewrites that token's KV rows with the same values and runs the layers the
            // prefill model dropped, reading the now complete cache.
            const ggml_tensor * out_ids = nullptr;
            for (const auto & [name, info] : ggml_decoder->get_model_inputs()) {
                const ggml_tensor * tensor = ggml_decoder->get_input_ggml_tensor(name);
                // gemma-4's inp_prune_keep indexes the gathered rows, not the rows of the graph
                if (tensor != nullptr && tensor != ggml_decoder->get_model_params().prune_keep &&
                    GgmlOvDecoder::is_output_idx(tensor, ggml_decoder->get_tensor_used_op(tensor))) {
                    out_ids = tensor;
                    break;
                }
            }
            const int64_t n_outputs = out_ids != nullptr ? out_ids->ne[0] : 0;
            std::string kv_only_error;
            if (n_outputs > 1) {
                kv_only_error = "the batch needs " + std::to_string(n_outputs) + " output tokens";
            } else if (n_outputs == 1) {
                const int64_t row = static_cast<const int32_t *>(out_ids->data)[0];
                const ComputeParams prefill_c_params = ggml_decoder->get_compute_params();
                ggml_decoder->m_is_prefill = false;
                try {
                    // With decode KV windows, run the decode model of the window the next decode step uses (it
                    // covers the output token); its inputs are sized by the decoder's attention size.
                    auto step_request = decode_request_for_outputs;
                    if (decode_bucket > 0) {
                        auto step_c_params = prefill_c_params;
                        step_c_params.attention_size = decode_window;
                        ggml_decoder->set_compute_params(step_c_params);
                        if (decode_window != entry->decode_attention_size) {
                            step_request = get_decode_window_request(decode_window);
                        }
                    }
                    const auto step_model = step_request->get_compiled_model();
                    const auto & compiled_inputs = step_model.inputs();
                    for (size_t i = 0; i < decode_input_names_for_outputs.size(); i++) {
                        const auto & name = decode_input_names_for_outputs[i];
                        auto input_tensor = get_ov_input_tensor_static_decode(ggml_decoder, name, row);
                        if (input_tensor.get_shape() != compiled_inputs[i].get_shape()) {
                            throw std::runtime_error("decode input " + name + " has shape " +
                                                     input_tensor.get_shape().to_string() + ", expected " +
                                                     compiled_inputs[i].get_shape().to_string());
                        }
                        step_request->set_input_tensor(i, input_tensor);
                    }
                    const auto & model_outputs = ggml_decoder->get_model_outputs();
                    for (size_t i = 0; i < decode_output_names_for_outputs.size(); i++) {
                        auto it = model_outputs.find(decode_output_names_for_outputs[i]);
                        if (it == model_outputs.end() || ggml_nbytes(it->second) == 0) {
                            continue;
                        }
                        step_request->set_output_tensor(
                            i, create_ov_output_tensor(ggml_decoder, step_request, i, it->second));
                    }
                    const int64_t decode_start = ggml_time_us();
                    step_request->infer();
                    const int64_t decode_us = ggml_time_us() - decode_start;
                    ov_raw_infer_total += decode_us;
                    if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                        fprintf(stderr,
                                "GGML OpenVINO profiling: KV-only prefill output step: row=%lld, kv_window=%d, raw "
                                "infer=%.3f ms\n",
                                static_cast<long long>(row), ggml_decoder->get_compute_params().attention_size,
                                decode_us / 1000.0);
                    }
                } catch (const std::exception & e) {
                    kv_only_error = e.what();
                }
                ggml_decoder->set_compute_params(prefill_c_params);
                ggml_decoder->m_is_prefill = true;
            }
            if (!kv_only_error.empty()) {
                // Fall back to the full prefill: drop this backend's requests and run the graph again. The KV-only prefill
                // has already written the batch's KV rows, and the full prefill writes the same values.
                GGML_LOG_WARN("ggml-openvino: KV-only prefill cannot compute the outputs (%s); using the full prefill\n",
                              kv_only_error.c_str());
                g_kv_only_prefill_failed = true;
                {
                    std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                    r_ctx->clear_caches_locked();
                }
                {
                    // Free the KV-only pair, including its models in the process-wide blob cache; a backend that still
                    // uses it keeps its own requests alive.
                    std::lock_guard<std::mutex> cache_lock(r_ctx->compiled_cache->mutex);
                    auto it = r_ctx->compiled_cache->graphs.find(entry->shared_key);
                    if (it != r_ctx->compiled_cache->graphs.end()) {
                        std::lock_guard<std::mutex> blobs_lock(g_imported_blobs_mutex);
                        g_imported_blobs.erase(it->second.prefill_fp);
                        g_imported_blobs.erase(it->second.decode_fp);
                        r_ctx->compiled_cache->graphs.erase(it);
                    }
                }
                infer_request.reset();
                decode_request_for_outputs.reset();
                return ov_graph_compute_static(cgraph, r_ctx);
            }
        }
        infer_end_time = ggml_time_us();
    } else {
        if (decode_bucket > 0 && c_params.attention_size != entry->decode_attention_size) {
            infer_request = get_decode_window_request(c_params.attention_size);
        }

        for (size_t i = 0; i < ov_input_names_local.size(); i++) {
            const auto & param_name = ov_input_names_local[i];
            auto input_tensor = get_ov_input_tensor_static_decode(ggml_decoder, param_name);
            infer_request->set_input_tensor(i, input_tensor);

            if (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                const auto input_tensor = infer_request->get_input_tensor(i);
                print_input_tensor_info(param_name, input_tensor);
            }
        }

        for (size_t i = 0; i < ov_output_names_local.size(); i++) {
            const auto & model_outputs = ggml_decoder->get_model_outputs();
            auto model_output_it = model_outputs.find(ov_output_names_local[i]);
            if (model_output_it == model_outputs.end()) {
                continue;
            }
            auto * ggml_tensor = model_output_it->second;
            if (ggml_nbytes(ggml_tensor) == 0) {
                continue;
            }
            auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        ov_raw_infer_start = ggml_time_us();
        infer_request->infer();
        infer_end_time = ggml_time_us();
        ov_raw_infer_total = infer_end_time - ov_raw_infer_start;

        if (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            for (size_t i = 0; i < ov_output_names_local.size(); i++) {
                const auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names_local[i], output_tensor, nullptr);
            }
        }
    }

    if (kv_grew) {
        GGML_LOG_INFO("ggml-openvino: on-demand KV storage moved to cover a %zu-row window\n", kv_rows);
    }
    if (kv_grew && cache_enabled && !entry->shared_key.empty()) {
        // Idle requests still bound to the previous KV storage keep it (and their own buffers) alive: drop the
        // other window requests and recreate the pair's idle requests.
        for (auto it = entry->decode_window_requests.begin(); it != entry->decode_window_requests.end();) {
            it = it->second == infer_request ? std::next(it) : entry->decode_window_requests.erase(it);
        }
        auto shared_cache = r_ctx->compiled_cache;
        std::lock_guard<std::mutex> compile_lock(shared_cache->mutex);
        auto graph_it = shared_cache->graphs.find(entry->shared_key);
        if (graph_it != shared_cache->graphs.end()) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            auto renew = [&](auto & cache, ov::CompiledModel & compiled) {
                auto it = cache.find(key);
                if (it != cache.end() && it->second != infer_request) {
                    it->second.reset();
                    it->second = std::make_shared<ov::InferRequest>(compiled.create_infer_request());
                }
            };
            renew(r_ctx->infer_request_cache_prefill, graph_it->second.prefill);
            renew(r_ctx->infer_request_cache, graph_it->second.decode);
        }
    }

    if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_end_time - start_time) / 1000.0);
        if (!cache_hit) {
            GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n", (conversion_end_time - decoder_end_time) / 1000.0);
            GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_end_time - conversion_end_time) / 1000.0);
        }
        GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_end_time - compile_end_time) / 1000.0);
        GGML_LOG_INFO("  - OV raw infer time: %.3f ms \n", ov_raw_infer_total / 1000.0);
    }

    return GGML_STATUS_SUCCESS;
}
}  // namespace

void ov_clear_imported_blobs() {
    std::lock_guard<std::mutex> lock(g_imported_blobs_mutex);
    g_imported_blobs.clear();
}

// Drop unselected branches except pure top-k sampling. Keeping those branches gives MTP a stable
// model interface as active slots change; inactive results use private output buffers.
static ggml_cgraph * get_compute_graph(ggml_cgraph * cgraph, ov_runtime_context & r_ctx, bool stable_sampling) {
    auto is_skipped = [](const ggml_tensor * node) {
        return node->op != GGML_OP_NONE && !(node->flags & GGML_TENSOR_FLAG_COMPUTE);
    };
    if (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
        return cgraph;
    }
    std::unordered_set<const ggml_tensor *> retained;
    if (stable_sampling) {
        auto use_count = [&](const ggml_tensor * node) {
            return cgraph->use_counts[ggml_hash_find(&cgraph->visited_hash_set, node)];
        };
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const auto * scores = cgraph->nodes[i];
            if (scores->op != GGML_OP_GET_ROWS || !GgmlOvDecoder::is_sampling_top_k(scores->src[1])) {
                continue;
            }
            const auto * top_k = scores->src[1];
            const auto * flat = top_k->src[0];
            const auto * rows = scores->src[0];
            if (!rows || rows->op != GGML_OP_RESHAPE || rows->src[0] != flat ||
                use_count(scores) != 0 || use_count(top_k) != 1 || use_count(rows) != 1 ||
                use_count(flat) != 2 || use_count(flat->src[0]) != 1) {
                continue;
            }
            // Retain only read-only top-k chains. Stateful samplers must keep their branch selection semantics.
            std::vector<const ggml_tensor *> pending = {scores};
            while (!pending.empty()) {
                const auto * node = pending.back();
                pending.pop_back();
                if (!is_skipped(node) || !retained.insert(node).second) {
                    continue;
                }
                for (const auto * src : node->src) {
                    if (src) {
                        pending.push_back(src);
                    }
                }
            }
        }
    }
    auto & compute = r_ctx.compute_graphs[cgraph];
    compute.nodes.clear();
    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
                 [&](const ggml_tensor * node) { return !is_skipped(node) || retained.count(node); });
    compute.graph = *cgraph;
    compute.graph.nodes = compute.nodes.data();
    compute.graph.n_nodes = (int) compute.nodes.size();
    return &compute.graph;
}

// Both execution paths use two cache levels:
// 1. Reuse this backend's decoder/request via graph_key and compatibility checks.
// 2. On a local miss, look up compiled_graph_key in the shared compilation cache,
//    compile if needed, then create a private request from the compiled model.
// The shared lock covers compilation and frontend cleanup, never inference.
enum ggml_status ov_graph_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    try {
        if (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_CGRAPH")) {
            std::string filename = "cgraph_ov.txt";
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        const auto is_static = ggml_openvino_is_static();

        GGML_ASSERT(ctx->runtime_context != nullptr);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        std::lock_guard<std::mutex> execution_lock(r_ctx->execution_mutex);
        cgraph = get_compute_graph(cgraph, *r_ctx, !is_static && !r_ctx->stateful);

        return is_static ? ov_graph_compute_static(cgraph, r_ctx) : ov_graph_compute_dynamic(cgraph, r_ctx);
    } catch (const ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        // A plugin reports a property it rejects as a failed internal check that does not name the property
        const std::string overrides = ggml_openvino_config_overrides_in_use();
        if (!overrides.empty()) {
            GGML_LOG_ERROR("GGML OpenVINO backend: plugin properties from the environment: %s\n", overrides.c_str());
        }
        if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            fprintf(stderr, "GGML OpenVINO profiling: ov::Exception: %s\n", e.what());
        }
        return GGML_STATUS_FAILED;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            fprintf(stderr, "GGML OpenVINO profiling: std::exception: %s\n", e.what());
        }
        return GGML_STATUS_FAILED;
    } catch (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            fprintf(stderr, "GGML OpenVINO profiling: unknown exception\n");
        }
        return GGML_STATUS_FAILED;
    }
}

// Detect whether a cgraph is a split subgraph or not.
// Step 1 compares each node's recorded use_count with actual fan-out references in node->src.
// Step 2 verifies that node inputs come from model nodes/weights/leafs; external sources imply split.
bool is_model_splitted(ggml_cgraph * cgraph) {
    static const bool fallback_enabled = ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK") != 0;
    if (!fallback_enabled) {
        return false;
    }

    // Backend op tests execute each node through ggml_graph_view(), which preserves the original
    // graph use_counts while exposing only one node. Treat those single-node views as regular
    // naive graphs so intermediate ops do not look like split-model fragments.
    if (cgraph->n_nodes <= 1 && cgraph->n_leafs == 0) {
        return false;
    }

    // check the nodes of the model are used by the following nodes, through compare the node's use count and the count of nodes that use it as input. If does not match, return true, else return false.
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        int use_count = cgraph->use_counts[ggml_hash_find(&cgraph->visited_hash_set, node)];
        // TODO: this is a workround for the tests case from llama.cpp, fix should from the root cause in the future.
        if ((cgraph->n_nodes <= 1 && use_count == 0) ||
            (cgraph->n_nodes <= 1 && node->op == GGML_OP_VIEW && use_count == 1 && node->src[0] != nullptr &&
             node->src[0]->op == GGML_OP_NONE)) {
            return false;
        }
        if (cgraph->n_nodes == 1 &&
            (cgraph->nodes[0]->op == GGML_OP_TRANSPOSE || cgraph->nodes[0]->op == GGML_OP_PERMUTE)) {
            return false;
        }
        int input_use_count = 0;
        for (int j = 0; j < cgraph->n_nodes; j++) {
            ggml_tensor * other_node = cgraph->nodes[j];
            for (int k = 0; k < GGML_MAX_SRC; k++) {
                if (other_node->src[k] == node) {
                    input_use_count++;
                }
            }
        }
        if (use_count != input_use_count && node->op != GGML_OP_NONE) {
            return true;
        }
    }
    // if all nodes's src node's src is not come from the nodes in the model, we think the model is splitted. This is a complementary check for the above check, because for some special case like the output node is not used by any node, the use count and input use count are both 0, we can not determine whether the model is splitted or not just based on the first check.
    // Only weight-name membership is needed below. With GGML_OPENVINO_REDUCE_COMPILE_MEM
    // use the name-only collector (no weight extraction); otherwise keep the original
    // behavior of building (naive) weight nodes and take their names.
    std::set<std::string> model_weights;
    if (ggml_openvino_reduce_compile_mem_enabled()) {
        model_weights = GgmlOvDecoder::collect_weight_names(cgraph);
    } else {
        for (const auto & kv : GgmlOvDecoder::create_weight_nodes(cgraph, true)) {
            model_weights.insert(kv.first);
        }
    }
    std::set<ggml_tensor *> model_nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    // leaf nodes
    std::set<ggml_tensor *> model_leafs(cgraph->leafs, cgraph->leafs + cgraph->n_leafs);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = node->src[j];
            // the src is also not the model weights, we think the model is splitted.
            // the src is also not in model leafs, we think the model is splitted.
            if (src != nullptr && model_nodes.find(src) == model_nodes.end() &&
                model_weights.find(std::string(src->name)) == model_weights.end() &&
                model_leafs.find(src) == model_leafs.end()) {
                if (GgmlOvDecoder::is_inp_tok(src, node)) {
                    return false;
                }
                return true;
            }
        }
    }
    return false;
}

bool is_naive(ggml_cgraph * cgraph) {
    constexpr int naive_graph_size_threshold = 20;
    int count = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    return count < naive_graph_size_threshold;
}

void print_input_tensor_info(const std::string & name, const ov::Tensor & input) {
    // GPU inputs can be device (USM) tensors, whose data() throws; print a host copy of those.
    const bool is_remote = input.is<ov::RemoteTensor>();
    ov::Tensor tensor = input;
    if (is_remote) {
        tensor = ov::Tensor(input.get_element_type(), input.get_shape());
        input.copy_to(tensor);
    }
    std::cout << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: ";
    if (is_remote) {
        std::cout << "device";
    } else {
        std::cout << tensor.data();
    }
    std::cout << '\n';
    switch (tensor.get_element_type()) {
    case ov::element::f32: {
        if (name.find("self_kq_mask") == std::string::npos && name.find("KQ_mask") == std::string::npos) {
            std::cout << *(tensor.data<float>()) << '\n';
        } else {
            size_t rows = tensor.get_shape()[2];
            size_t cols = tensor.get_shape()[3];
            const float * data = tensor.data<float>();
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    float val = data[i * cols + j];
                    if (std::isinf(val) && val < 0) {
                        std::cout << std::setw(5) << "-inf";
                    } else {
                        std::cout << std::setw(5) << val;
                    }
                }
                std::cout << '\n';
            }
        }

        break;
    }
    case ov::element::f16:
        std::cout << *(tensor.data<ov::float16>()) << '\n';
        break;
    case ov::element::i32:
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int32_t>()[i] << ' ';
        }
        std::cout << '\n';
        break;
    case ov::element::i64:
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int64_t>()[i] << ' ';
        }
        std::cout << '\n';
        break;
    default:
        break;
    }
}

void print_output_tensor_info(const std::string & name, const ov::Tensor & tensor, const void * output_dst) {
    ov::Tensor host(tensor.get_element_type(), tensor.get_shape());
    tensor.copy_to(host);
    std::cout << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << '\n';

    auto print_float_stats = [](const std::string & type_name, size_t size, auto get_value) {
        if (size == 0) {
            return;
        }

        float first = get_value(0);
        float min = first;
        float max = first;
        size_t max_idx = 0;
        double sum = first;

        for (size_t i = 1; i < size; ++i) {
            float v = get_value(i);
            min = std::min(v, min);
            if (v > max) {
                max = v;
                max_idx = i;
            }
            sum += v;
        }
        double mean = sum / size;

        std::cout << std::right << std::setw(6) << type_name << std::right << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << std::setw(12) << "MaxIdx" << '\n';
        std::cout << std::right << std::setw(6) << "" << std::right << std::setw(12) << first << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << mean << std::setw(12) << max_idx << '\n';
    };

    switch (tensor.get_element_type()) {
    case ov::element::f32: {
        const float * data = host.data<float>();
        size_t size = tensor.get_size();
        print_float_stats("[f32]", size, [data](size_t i) { return data[i]; });
        break;
    }
    case ov::element::f16: {
        const ov::float16 * data = host.data<ov::float16>();
        size_t size = tensor.get_size();
        print_float_stats("[f16]", size, [data](size_t i) { return static_cast<float>(data[i]); });
        break;
    }
    default:
        break;
    }
}

const ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            auto * src = op->src[j];
            if (src == nullptr) {
                break;
            }
            if (GgmlOvDecoder::is_inp_pos(src, op)) {
                return src;
            }
        }
    }
    GGML_LOG_ERROR("get_inp_pos_tensor: inp_pos not found in cgraph");
    throw std::runtime_error("get_inp_pos_tensor: inp_pos not found in cgraph");
}

int64_t get_inp_pos_n_tokens(ggml_cgraph * cgraph, const ggml_tensor * inp_pos) {
    // IMROPE stacks n_planes (t/h/w/e) position planes into inp_pos, so ne[0] is
    // n_planes * n_tokens. Callers that need a token count must divide the planes out.
    int n_planes = 1;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (op->src[j] == inp_pos) {
                n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
                break;
            }
        }
    }
    return inp_pos->ne[0] / n_planes;
}

bool get_is_prefill(ggml_cgraph * cgraph, const ggml_tensor * inp_pos) {
    return get_inp_pos_n_tokens(cgraph, inp_pos) > 1;
}
