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
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
std::optional<ov::Tensor> try_make_kv_sliced_tensor(const std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                                    const std::string & name,
                                                    const ggml_tensor * ggml_tensor) {
    // Static graphs bake the sliced length into the prefill graph; decode keeps its full-context graph.
    const bool slice =
        ggml_openvino_kv_slice_enabled() && (!ggml_decoder->is_static() || ggml_decoder->m_is_prefill);
    if (!slice || ggml_decoder->is_stateful()) {
        return std::nullopt;
    }
    if (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != nullptr) {
        return std::nullopt;
    }

    const auto & compute_params = ggml_decoder->get_compute_params();
    if (compute_params.n_seq_active != 1 || compute_params.seq_active_start != 0) {
        return std::nullopt;
    }

    const int ctx_per_seq = ggml_decoder->get_ctx_per_seq();
    const int n_kv = compute_params.attention_size;
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

    ov::Shape sliced_shape = full_shape;
    sliced_shape[2] = static_cast<size_t>(n_kv);

    const auto type = GgmlOvDecoder::get_ov_type(ggml_tensor);
    return ggml_decoder->cached_kv_tensor(ggml_tensor, type, sliced_shape, [&]() -> ov::Tensor {
        if (ggml_openvino_buffer_is_remote(ggml_tensor)) {
            auto remote_context = ggml_openvino_get_remote_context();
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
            auto remote_context = ggml_openvino_get_remote_context();
            auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
            return gpu_context.create_tensor(type, shape, ggml_tensor->data);
        }
        return ov::Tensor(type, shape, ggml_tensor->data);
    });
}

// Column count of a static KQ mask input; must match GgmlOvDecoder::get_graph_input_shape().
static size_t get_static_attention_size(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & name) {
    const bool is_swa = ggml_decoder->is_swa_mask(ggml_decoder->get_input_ggml_tensor(name));
    if (ggml_decoder->m_is_prefill && ggml_openvino_kv_slice_enabled()) {
        const auto params = ggml_decoder->get_compute_params();
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
    const char * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    const bool manual_gqa_enabled = manual_gqa_env != nullptr ?
                                        ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0 :
                                        ggml_openvino_is_gpu();

    uint64_t extra_cfg = 1;  // Graph-ordinal port names (invalidate older disk-cache blobs).
    extra_cfg = extra_cfg * 131 + (stateful ? (recurrent ? 2u : 1u) : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_reduce_compile_mem_enabled() ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_kv_slice_enabled() ? 0u : 1u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_KV_4D") > 0 ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (manual_gqa_enabled ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_shape_from_mask_enabled() ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_npu_hfa_prefill_enabled() ? 3u : 0u);  // HFA input ABI v3.
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
        append(input.is_parameter);
        if (!input.is_parameter) {
            append(input.value);
        }
        append_string(input.shape_source);
        append(input.shape_axis);
    }
    return key;
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
        if (port.get_element_type() != GgmlOvDecoder::get_ov_type(graph_it->second) ||
            !port.get_partial_shape().compatible(ov::PartialShape(GgmlOvDecoder::get_shape(graph_it->second)))) {
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

static void reset_single_slot_recurrent_cache(ggml_cgraph * cgraph,
                                              const ModelParams & model_params,
                                              const ComputeParams & compute_params) {
    if (model_params.n_rs_slots != 1 || compute_params.cache_rs_reset_len == 0) {
        return;
    }
    GGML_ASSERT(compute_params.cache_rs_reset_idx == 0 && compute_params.cache_rs_reset_len == 1);

    std::set<ggml_backend_buffer_t> buffers;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        auto * node = cgraph->nodes[i];
        if (node->op == GGML_OP_SCALE && GgmlOvDecoder::is_recurrent_cache(node->view_src) &&
            node->view_src->ne[1] == 1 && node->view_src->buffer != nullptr) {
            buffers.insert(node->view_src->buffer);
        }
    }
    for (auto * buffer : buffers) {
        ggml_backend_buffer_clear(buffer, 0);
    }
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
        }
    }

    // The sliced path above covers eligible KV outputs. This also covers recurrent state and full-size KV fallbacks.
    if (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS") && !ggml_decoder->is_static() &&
        !ggml_decoder->is_splited_model() && ggml_openvino_buffer_is_remote(ggml_tensor) &&
        ggml_tensor->extra != nullptr) {
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        if (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
            auto * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
            if (tensor_extra->tensor != nullptr && tensor_extra->tensor->get_element_type() == output_type &&
                tensor_extra->tensor->get_shape() == output_shape) {
                return *tensor_extra->tensor;
            }
        }
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

    if (auto kv_4d = try_make_kv_4d_tensor(ggml_decoder, ggml_tensor)) {
        return *kv_4d;
    }
    if (auto sliced = try_make_kv_sliced_tensor(ggml_decoder, name, ggml_tensor)) {
        return *sliced;
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

    auto input_tensor = ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), input_shape, input_data);
    return input_tensor;
}

ov::Tensor get_ov_input_tensor(const std::shared_ptr<GgmlOvDecoder> & ggml_decoder, const std::string & param_name) {
    ov::Tensor input_tensor;
    auto extra_input = ggml_decoder->get_model_extra_inputs().find(param_name);
    if (extra_input != ggml_decoder->get_model_extra_inputs().end()) {
        input_tensor = ov::Tensor(extra_input->second.type, extra_input->second.shape);
        if (extra_input->second.type == ov::element::i64) {
            *input_tensor.data<int64_t>() = extra_input->second.value;
        } else {
            std::memset(input_tensor.data(), 0, input_tensor.get_byte_size());
        }
    } else {
        input_tensor = convert_ggml_input_to_ov(ggml_decoder, param_name);
    }
    return input_tensor;
}

ov::Tensor get_ov_input_tensor_static_decode(const std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                             const std::string & param_name) {
    // NPU decoding stage
    if (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        return get_ov_input_tensor(ggml_decoder, param_name);
    }
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(param_name);
    const auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    if (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        // IMROPE's inp_pos holds one value per t/h/w/e plane instead of a single position;
        // with a single decode token the planes are still contiguous, so a flat copy works.
        const int n_planes = GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ? GgmlOvDecoder::get_inp_pos_n_planes(op) : 1;
        assert(ggml_tensor->ne[0] == n_planes);
        ov::Shape input_shape = {1, 1, 1, (size_t) n_planes};
        ov::Tensor input_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), input_shape);
        std::memcpy(input_tensor.data(), ggml_tensor->data, n_planes * ggml_type_size(ggml_tensor->type));
        return input_tensor;
    }

    if (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape input_shape = {1, 1, 1, 1};
        ov::Tensor input_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), input_shape);
        int32_t inp_out_id = *((int32_t *) ggml_tensor->data);
        assert(ggml_tensor->ne[0] == 1);
        assert(inp_out_id == 0);
        *input_tensor.data<int32_t>() = inp_out_id;
        return input_tensor;
    }

    if (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t context_size = get_static_attention_size(ggml_decoder, param_name);
        if (ggml_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> padded_data =
                pad_input<ggml_fp16_t>(ggml_tensor, 1, context_size, GGML_FP32_TO_FP16(-INFINITY));
            ov::Tensor input_tensor(ov::element::f16, ov::Shape{1, 1, 1, context_size});
            std::memcpy(input_tensor.data(), padded_data.data(), padded_data.size() * sizeof(ggml_fp16_t));
            return input_tensor;
        }

        std::vector<float> padded_data = pad_input<float>(ggml_tensor, 1, context_size, -INFINITY);
        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_size});
        auto * data_ptr = input_tensor.data<float>();
        std::copy(padded_data.begin(), padded_data.begin() + context_size, data_ptr);
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
        const std::string kv_name = param_name.substr(strlen("ggml_hfa_past_"));
        const auto * kv_tensor = ggml_decoder->get_input_ggml_tensor(kv_name);
        const auto & info = ggml_decoder->get_model_extra_inputs().at(param_name);
        ov::Tensor input_tensor = ggml_decoder->get_or_create_hfa_past_tensor(param_name, info.type, info.shape);
        if (chunk_index == 0) {
            std::memset(input_tensor.data(), 0, input_tensor.get_byte_size());
        }

        const size_t n_heads_kv = info.shape[1];
        const size_t past_capacity = info.shape[2];
        const size_t head_size = info.shape[3];
        const size_t row_begin = chunk_index == 0 ? 0 : std::min(static_cast<size_t>(chunk_index - 1) * chunk_size,
                                                                 past_capacity);
        const size_t row_end = std::min(static_cast<size_t>(chunk_index) * chunk_size, past_capacity);
        if (static_cast<size_t>(kv_tensor->ne[0]) != n_heads_kv * head_size ||
            row_end > static_cast<size_t>(kv_tensor->ne[1])) {
            throw std::runtime_error("HFA past KV source shape is incompatible with " + param_name);
        }
        const size_t head_bytes = head_size * ggml_type_size(kv_tensor->type);
        const auto * src = static_cast<const uint8_t *>(kv_tensor->data);
        auto * dst = static_cast<uint8_t *>(input_tensor.data());
        for (size_t row = row_begin; row < row_end; ++row) {
            for (size_t head = 0; head < n_heads_kv; ++head) {
                std::memcpy(dst + (head * past_capacity + row) * head_bytes,
                            src + (row * n_heads_kv + head) * head_bytes, head_bytes);
            }
        }
        return input_tensor;
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
    if (chunk_index > 0 && param_name == "cache_rs_reset_len") {
        // The recurrent-state clear belongs to the start of the sequence. Re-applying it on every
        // chunk would wipe the state accumulated by the preceding chunks, so disable it (a zero
        // length makes scale.cpp's keep-mask select every slot) after the first chunk.
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
    auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(decoder);
    auto model = ov::frontend::ggml::FrontEnd::convert(input_model, naive);
    if (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    auto remote_context = ggml_openvino_get_remote_context();
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
                print_output_tensor_info(output_names[i], output_tensor, output_tensor.data());
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
    static const bool enabled = ggml_openvino_is_gpu() && ggml_openvino_get_remote_context().has_value();
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

bool stageable_size(const ov::Tensor & t) {
    return t.get_byte_size() > 0 && t.get_byte_size() <= k_io_staging_max_bytes;
}

ov::Tensor & staging_for(const GgmlOvDecoder & decoder, GgmlOvDecoder::ov_port_binding & b, const ov::Tensor & like) {
    const size_t bytes = like.get_byte_size();
    if (!b.staging || b.staging_bytes < bytes || b.staging.get_element_type() != like.get_element_type()) {
        auto gpu_context = ggml_openvino_get_remote_context()->as<ov::intel_gpu::ocl::ClContext>();
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
            } else {
                r_ctx->clear_caches_locked();
                auto mutex = std::make_shared<std::mutex>();
                entry = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[key] = entry;
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
            old_m_params = ggml_decoder->get_model_params();
            if (!ggml_decoder->is_splited_model()) {
                cache_hit = old_m_params.can_reuse_dynamically(m_params);
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
                    std::ifstream blob_in(candidate_blob, std::ios::binary);
                    std::vector<std::string> saved_inputs, saved_outputs;
                    if (!blob_in.is_open() ||
                        !ggml_openvino_model_cache_verify_manifest(candidate_manifest, cgraph, candidate_fp,
                                                                   saved_inputs, saved_outputs)) {
                        continue;
                    }
                    const int64_t import_start = ggml_time_us();
                    try {
                        ov::CompiledModel cm;
                        auto remote_context = ggml_openvino_get_remote_context();
                        if (remote_context.has_value()) {
                            cm = core.import_model(blob_in, remote_context.value(), mc_config);
                        } else {
                            cm = core.import_model(blob_in, device, mc_config);
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
                auto remote_context = ggml_openvino_get_remote_context();
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
                if (!model_cache_dir.empty() && !model_is_splitted && model_fp != 0) {
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
                entry->ptr = ggml_decoder;

            }  // end non-imported (compile) path

            entry->ptr = ggml_decoder;
            if (!shared_key.empty() && shared_it == shared_cache->graphs.end()) {
                shared_cache->graphs.emplace(shared_key,
                                             ov_compiled_graph{shared_model, {}, ov_input_names, ov_output_names, {}, {}});
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
        } else {
            reset_single_slot_recurrent_cache(cgraph, m_params, c_params);
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
                infer_request->set_input_tensor(i, input_tensor);
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
            if (ggml_nbytes(ggml_tensor) == 0) {
                continue;
            }
            ggml_decoder->take_last_kv_tensor_id();  // clear
            auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            uint64_t id = ggml_decoder->take_last_kv_tensor_id();
            auto & binding = out_bindings[i];
            if (id == 0 && io_staging_enabled() && !ggml_openvino_buffer_is_remote(ggml_tensor) &&
                stageable_size(output_tensor)) {
                void * dst = output_tensor.data();
                const size_t bytes = output_tensor.get_byte_size();
                output_tensor = staging_for(*ggml_decoder, binding, output_tensor);
                staged_outputs.push_back({dst, binding.staging_ptr, bytes});
                id = binding.staging_id;
            }
            if (needs_bind(binding, output_tensor, id)) {
                infer_request->set_output_tensor(i, output_tensor);
            }
        }

        ov_raw_infer_start = ggml_time_us();
        infer_request->infer();
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
                print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.data());
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
    static std::string device = ggml_openvino_get_device_name();
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
    const ov::AnyMap compile_config = no_kv_cache ? ggml_openvino_npu_without_npuw(config) : config;
    if (m_params.n_heads_kv == -1) {
        prefill_chunk_size = inp_pos->ne[0];
    }
    graph_key key(cgraph);
    static const bool cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    bool cache_hit = false;

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
        } else {
            r_ctx->clear_caches_locked();
            auto mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[key] = entry;
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

    if (cache_hit) {
        ggml_decoder = entry->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_hit = old_m_params.can_reuse_statically(m_params);
        if (cache_hit && is_prefill && ggml_openvino_kv_slice_enabled()) {
            const auto old_c_params = ggml_decoder->get_compute_params();
            cache_hit = old_c_params.attention_size == c_params.attention_size &&
                        old_c_params.attention_size_swa == c_params.attention_size_swa;
        }
    }

    std::vector<std::string> ov_input_names_local;
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
        }

        decoder_end_time = ggml_time_us();
        conversion_end_time = decoder_end_time;
        compile_end_time = decoder_end_time;
    } else {
        // Same fail-fast as the dynamic path: once the host weight pages have been dropped
        // they read as zeros, so a recompile here would bake garbage into the new model.
        if (ggml_openvino_weight_buffers_released()) {
            GGML_ABORT(
                "ggml-openvino: a new graph needs to be compiled but host weight buffers were already "
                "released via GGML_OPENVINO_RELEASE_WEIGHTS. This mode requires "
                "stable graph shapes; disable host weight release for dynamic workloads.");
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
        const std::string shared_key =
            cache_enabled ? compiled_graph_key(cgraph, *local_decoder, device, prefill_chunk_size) : "";
        auto shared_it = shared_cache->graphs.find(shared_key);
        if (!shared_key.empty() && shared_it != shared_cache->graphs.end()) {
            auto & compiled = shared_it->second;
            auto prefill_request = std::make_shared<ov::InferRequest>(compiled.prefill.create_infer_request());
            auto decode_request = no_kv_cache ?
                                      prefill_request :
                                      std::make_shared<ov::InferRequest>(compiled.decode.create_infer_request());
            ggml_decoder = local_decoder;
            entry->ptr = ggml_decoder;
            infer_request = is_prefill ? prefill_request : decode_request;
            ov_input_names_local = is_prefill ? compiled.prefill_input_names : compiled.input_names;
            ov_output_names_local = is_prefill ? compiled.prefill_output_names : compiled.output_names;
            r_ctx->infer_request_cache_prefill[key] = prefill_request;
            r_ctx->infer_request_cache[key] = decode_request;
            r_ctx->ov_input_names_cache[key] = compiled.input_names;
            r_ctx->ov_output_names_cache[key] = compiled.output_names;
            r_ctx->ov_input_names_cache_prefill[key] = compiled.prefill_input_names;
            r_ctx->ov_output_names_cache_prefill[key] = compiled.prefill_output_names;
            decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (static)\n");
        } else {
            auto decode_c_params = c_params;
            if (ggml_openvino_kv_slice_enabled()) {
                decode_c_params.attention_size = m_params.ctx_per_seq;
                decode_c_params.attention_size_swa = m_params.ctx_per_seq_swa;
            }
            const ov::AnyMap prefill_config = ggml_openvino_npu_prefill_config(compile_config, *local_decoder);
            const ov::AnyMap decode_config = ggml_openvino_npu_decode_config(compile_config, device);

            const std::string model_cache_dir = ggml_openvino_model_cache_dir();
            const bool self_contained = device == "NPU" && !model_cache_dir.empty() &&
                                        ggml_openvino_getenv_int("GGML_OPENVINO_SELF_CONTAINED_BLOB") != 0;
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
            if (!model_cache_dir.empty()) {
                uint64_t base_extra = ggml_openvino_model_cache_extra_cfg(stateful, /*recurrent=*/false);
                base_extra = base_extra * 131 + (self_contained ? 1u : 0u);
                base_extra = ggml_openvino_model_cache_mix_string(
                    base_extra, ggml_openvino_getenv_str("GGML_OPENVINO_NPU_TOKEN_EMBD_QUANT"));
                base_extra = base_extra * 131 + static_cast<uint64_t>(ggml_openvino_getenv_int("GGML_OPENVINO_NPU_KEEP_Q4_0"));
                base_extra = base_extra * 131 + (ggml_openvino_kv_slice_enabled() ? 1u : 0u);
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
                auto static_salt = [base_extra](bool prefill, int chunk, int attn, int attn_swa, bool unfold) {
                    uint64_t h = base_extra;
                    h = h * 131 + (prefill ? 1u : 0u);
                    h = h * 131 + static_cast<uint64_t>(static_cast<uint32_t>(chunk));
                    h = h * 131 + static_cast<uint64_t>(static_cast<uint32_t>(attn));
                    h = h * 131 + static_cast<uint64_t>(static_cast<uint32_t>(attn_swa));
                    h = h * 131 + (unfold ? 1u : 0u);
                    return h;
                };
                const bool decode_unfold = mc_decode_config.find("NPUW_UNFOLD_IREQS") != mc_decode_config.end();
                // out_ids is bound at a fixed length, so the output count is part of the static shape
                const std::string prefill_signature = "static-1 out=" + std::to_string(c_params.output_len);
                const std::string decode_signature = "static-1 out=1";
                prefill_fp = ggml_openvino_model_fingerprint(
                    cgraph, device, true, m_params.rope_params, 16,
                    static_salt(true, prefill_chunk_size, c_params.attention_size, c_params.attention_size_swa, false),
                    prefill_signature);
                decode_fp = ggml_openvino_model_fingerprint(
                    cgraph, device, true, m_params.rope_params, 16,
                    static_salt(false, prefill_chunk_size, decode_c_params.attention_size,
                                decode_c_params.attention_size_swa, decode_unfold),
                    decode_signature);
            }

            if (self_contained) {
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
            const bool skip_weights =
                self_contained && blob_valid(prefill_fp) && blob_valid(decode_fp);
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
            auto ggml_decoder_decode =
                no_kv_cache ? ggml_decoder_prefill :
                              std::make_shared<GgmlOvDecoder>(cgraph, m_params, decode_c_params, model_weights,
                                                              is_static, stateful, false, false, prefill_chunk_size);
            decoder_end_time = ggml_time_us();

            const bool dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
            const auto dump_ir_timestamp = static_cast<long long>(ggml_time_us());

            // Both decoders share the weight nodes, and converting attaches each new graph to them. ov::Node is not
            // thread-safe, so the models are always converted one at a time; only compilation can run in parallel.
            auto convert_static_model = [dump_ir, dump_ir_timestamp, skip_weights](
                                            std::shared_ptr<GgmlOvDecoder> decoder, const char * tag,
                                            int64_t & local_conversion_end_time) -> std::shared_ptr<ov::Model> {
                if (skip_weights) {
                    return nullptr;
                }
                auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(decoder);
                auto model = ov::frontend::ggml::FrontEnd::convert(input_model);
                decoder->clear_model_weights();
                local_conversion_end_time = ggml_time_us();

                if (dump_ir) {
                    char timestamped_filename[64];
                    snprintf(timestamped_filename, sizeof(timestamped_filename), "model_%s_%lld.xml", tag,
                             dump_ir_timestamp);
                    ov::serialize(model, timestamped_filename);
                }
                return model;
            };

            auto build_static_model = [&core, &model_cache_dir, cgraph, self_contained, skip_weights](
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
                std::string blob_path;
                std::string manifest_path;
                if (!model_cache_dir.empty() && model_fp != 0) {
                    blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                    manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
                }

                if (skip_weights) {
                    try {
                        std::ifstream blob_in(blob_path, std::ios::binary);
                        const int64_t import_start_time = ggml_time_us();
                        compiled_model = core.import_model(blob_in, device, model_config);
                        const int64_t import_end_time = ggml_time_us();
                        if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            fprintf(stderr, "ggml-openvino: model cache IMPORT %s: %.3f ms\n", tag,
                                    (import_end_time - import_start_time) / 1000.0);
                        }
                    } catch (const std::exception & e) {
                        std::remove(blob_path.c_str());
                        std::remove(manifest_path.c_str());
                        throw std::runtime_error("self-contained cache import failed for " + std::string(tag) +
                                                 "; removed the invalid entry, rerun to rebuild it: " + e.what());
                    }
                    infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                    std::set<std::string> compiled_input_names;
                    for (const auto & input : compiled_model.inputs()) {
                        compiled_input_names.insert(input.get_node()->get_friendly_name());
                    }
                    for (const auto & [name, _] : decoder->get_model_inputs()) {
                        if (!ggml_openvino_npu_hfa_prefill_enabled() || std::string(tag) != "prefill" ||
                            compiled_input_names.count(name) != 0) {
                            names_in.push_back(name);
                        }
                    }
                    for (const auto & [name, info] : decoder->get_model_extra_inputs()) {
                        if (info.is_parameter) {
                            names_in.push_back(name);
                        }
                    }
                    for (const auto & name : decoder->get_model_output_names()) {
                        names_out.push_back(name);
                    }
                    if (names_in.size() != compiled_model.inputs().size() ||
                        names_out.size() != compiled_model.outputs().size()) {
                        throw std::runtime_error(
                            "self-contained cache port count mismatch for " + std::string(tag) +
                            ": decoder inputs=" + std::to_string(names_in.size()) +
                            ", compiled inputs=" + std::to_string(compiled_model.inputs().size()) +
                            ", decoder outputs=" + std::to_string(names_out.size()) +
                            ", compiled outputs=" + std::to_string(compiled_model.outputs().size()));
                    }
                    decoder->clear_model_weights();
                    local_conversion_end_time = local_compile_end_time = ggml_time_us();
                    GGML_LOG_INFO("ggml-openvino: model cache HIT (self-contained) %s (%s)\n", blob_path.c_str(), tag);
                    return;
                }

                bool imported = false;
                if (!model_cache_dir.empty() && model_fp != 0) {
                    std::ifstream blob_in(blob_path, std::ios::binary);
                    std::vector<std::string> cached_inputs, cached_outputs;
                    if (blob_in.is_open() && ggml_openvino_model_cache_verify_manifest(manifest_path, cgraph, model_fp,
                                                                                       cached_inputs, cached_outputs)) {
                        try {
                            ov::AnyMap import_config = model_config;
                            if (!self_contained) {
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
                            GGML_LOG_WARN("ggml-openvino: model cache import failed (%s), recompiling %s\n", e.what(),
                                          tag);
                            imported = false;
                        }
                    }
                }

                if (!imported) {
                    ov::AnyMap compile_config = model_config;
                    if (self_contained) {
                        compile_config[ov::enable_weightless.name()] = false;
                    }
                    compiled_model = core.compile_model(model, device, compile_config);
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
                            self_contained ? std::ifstream(source_manifest_tmp).good() :
                                             ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp,
                                                                                      names_in, names_out);
                        const std::string & manifest_publish = self_contained ? source_manifest_tmp : manifest_tmp;
                        if (manifest_ready) {
                            std::ofstream blob_out(blob_tmp, std::ios::binary | std::ios::trunc);
                            if (blob_out.is_open()) {
                                compiled_model.export_model(blob_out);
                                blob_out.close();
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
            const bool parallel_compile = !self_contained && !no_kv_cache && ggml_openvino_npu_parallel_compile_enabled();
            auto model_prefill = convert_static_model(ggml_decoder_prefill, "prefill", prefill_conversion_end_time);
            std::shared_ptr<ov::Model> model_decode;
            if (!no_kv_cache) {
                model_decode = convert_static_model(ggml_decoder_decode, "decode", decode_conversion_end_time);
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

            if (!shared_key.empty()) {
                shared_cache->graphs.emplace(
                    shared_key, ov_compiled_graph{compiled_model_decode, compiled_model_prefill, names_in_decode,
                                                  names_out_decode, names_in_prefill, names_out_prefill});
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

    reset_single_slot_recurrent_cache(cgraph, m_params, c_params);

    if (is_prefill) {
        auto inp_len = get_inp_pos_n_tokens(cgraph, inp_pos);
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
                auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->set_output_tensor(i, output_tensor);
            }

            ov_raw_infer_start = ggml_time_us();
            infer_request->infer();
            const int64_t chunk_infer_us = ggml_time_us() - ov_raw_infer_start;
            ov_raw_infer_total += chunk_infer_us;
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
                    print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
                }
            }
        }
        infer_end_time = ggml_time_us();
    } else {
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
                print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
            }
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

// Nodes on the unselected branches of ggml_build_forward_select() stay in the graph but must not be
// computed. Keep them out of the OV model, or their inputs become parameters with fixed shapes.
static ggml_cgraph * get_compute_graph(ggml_cgraph * cgraph, ov_runtime_context & r_ctx) {
    auto is_skipped = [](const ggml_tensor * node) {
        return node->op != GGML_OP_NONE && !(node->flags & GGML_TENSOR_FLAG_COMPUTE);
    };
    if (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
        return cgraph;
    }
    auto & compute = r_ctx.compute_graphs[cgraph];
    compute.nodes.clear();
    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
                 [&](const ggml_tensor * node) { return !is_skipped(node); });
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

        const auto is_static = ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC");

        GGML_ASSERT(ctx->runtime_context != nullptr);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        std::lock_guard<std::mutex> execution_lock(r_ctx->execution_mutex);
        cgraph = get_compute_graph(cgraph, *r_ctx);

        return is_static ? ov_graph_compute_static(cgraph, r_ctx) : ov_graph_compute_dynamic(cgraph, r_ctx);
    } catch (const ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
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
                model_weights.find(std::string(src->name)) == model_weights.end() && !model_leafs.empty() == false &&
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

size_t checksum(const void * data, size_t size) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    size_t sum = 0;
    for (size_t i = 0; i < size; ++i) {
        sum += (uint8_t) i;
        sum += bytes[i];
    }
    return sum;
}

bool save_ggml_tensor_data_to_txt(const ggml_tensor * tensor, const std::string & file_path) {
    if (tensor == nullptr || tensor->data == nullptr) {
        return false;
    }

    std::ofstream out(file_path);
    if (!out.is_open()) {
        return false;
    }

    const size_t n = ggml_nelements(tensor);
    out << "name: " << tensor->name << ", type: " << ggml_type_name(tensor->type) << ", shape: [" << tensor->ne[0]
        << ", " << tensor->ne[1] << ", " << tensor->ne[2] << ", " << tensor->ne[3] << "]" << ", elements: " << n
        << ", data:" << '\n';

    switch (tensor->type) {
    case GGML_TYPE_F32: {
        const auto * data = static_cast<const float *>(tensor->data);
        for (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        break;
    }
    case GGML_TYPE_F16: {
        const auto * data = static_cast<const ggml_fp16_t *>(tensor->data);
        for (size_t i = 0; i < n; ++i) {
            out << ggml_fp16_to_fp32(data[i]) << '\n';
        }
        break;
    }
    case GGML_TYPE_BF16: {
        const auto * data = static_cast<const ggml_bf16_t *>(tensor->data);
        for (size_t i = 0; i < n; ++i) {
            out << ggml_bf16_to_fp32(data[i]) << '\n';
        }
        break;
    }
    case GGML_TYPE_I32: {
        const auto * data = static_cast<const int32_t *>(tensor->data);
        for (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        break;
    }
    case GGML_TYPE_I64: {
        const auto * data = static_cast<const int64_t *>(tensor->data);
        for (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        break;
    }
    default:
        out << "unsupported tensor type for text dump" << '\n';
        return false;
    }

    return true;
}

void print_input_tensor_info(const std::string & name, const ov::Tensor & tensor) {
    std::cout << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.data()
              << '\n';
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
        const float * data = tensor.data<float>();
        size_t size = tensor.get_size();
        print_float_stats("[f32]", size, [data](size_t i) { return data[i]; });
        break;
    }
    case ov::element::f16: {
        const ov::float16 * data = tensor.data<ov::float16>();
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
