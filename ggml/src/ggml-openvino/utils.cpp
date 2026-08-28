#include "utils.h"

#include "ggml-impl.h"
#include "ggml-openvino-extra.h"
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
#include <string>
#include <unordered_map>
#include <vector>

// Suppress  deprecation warning for ov::Tensor::data()
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

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

        return is_static ? ov_graph_compute_static(cgraph, r_ctx) : ov_graph_compute_dynamic(cgraph, r_ctx);
    } catch (const ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        return GGML_STATUS_FAILED;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        return GGML_STATUS_FAILED;
    } catch (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        return GGML_STATUS_FAILED;
    }
}

// For a KV cache input, return an ov::Tensor sized to n_kv (== attention_size
// for that layer) instead of the fully-allocated ctx_per_seq. Pre-conditions:
//   * non-static (CPU/GPU) backend, single sequence, seq_active_start == 0
//   * ggml KV layout is a contiguous [1, 1, ctx_per_seq, n_heads_kv*head_size]
//     so the first n_kv rows are the live prefix and shrinking the ctx axis
//     gives a valid tensor over the same host storage
//   * not an SWA layer (ring cache): once the window has wrapped the first
//     n_kv rows no longer contain the live prefix
// On any unmet pre-condition returns std::nullopt; the caller falls back to
// the full-size tensor.
static std::optional<ov::Tensor> try_make_kv_sliced_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                                           const std::string & name,
                                                           const ggml_tensor * ggml_tensor) {
    static const bool kv_slice_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE");
    if (kv_slice_disabled) {
        return std::nullopt;
    }
    if (ggml_decoder->is_static() || ggml_decoder->is_stateful()) {
        return std::nullopt;
    }
    if (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != nullptr) {
        return std::nullopt;
    }
    const auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);
    if (!GgmlOvDecoder::is_kvcache(ggml_tensor, op)) {
        return std::nullopt;
    }

    const auto & compute_params = ggml_decoder->get_compute_params();
    if (compute_params.n_seq_active != 1 || compute_params.seq_active_start != 0) {
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
    const int ctx_per_seq = ggml_decoder->get_ctx_per_seq();
    const int n_kv = compute_params.attention_size;
    if (ctx_per_seq <= 0 || n_kv <= 0 || n_kv >= ctx_per_seq) {
        return std::nullopt;
    }

    ov::Shape full_shape = ggml_decoder->get_shape(ggml_tensor);
    if (full_shape.size() != 4 || full_shape[0] != 1 || full_shape[1] != 1 ||
        static_cast<int>(full_shape[2]) != ctx_per_seq) {
        return std::nullopt;
    }

    ov::Shape sliced_shape = full_shape;
    sliced_shape[2] = static_cast<size_t>(n_kv);

    // Disabling for now as gpu has bug with in-place ScatterUpdate with remote tensors, can re-enable once CVS-186519 is fixed
    // if (ggml_openvino_buffer_is_remote(ggml_tensor)) {
    //     auto remote_context = ggml_openvino_get_remote_context();
    //     auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
    //     return gpu_context.create_tensor(ggml_decoder->get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->data);
    // }

    return ov::Tensor(ggml_decoder->get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->data);
}

static uint64_t ggml_openvino_model_cache_extra_cfg(const std::string & device, bool stateful) {
    const char * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    const bool manual_gqa_enabled = manual_gqa_env != nullptr ?
                                        ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0 :
                                        device == "GPU";

    uint64_t extra_cfg = 0;
    extra_cfg = extra_cfg * 131 + (stateful ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_reduce_compile_mem_enabled() ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE") ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (manual_gqa_enabled ? 1u : 0u);
    return extra_cfg;
}

ov::Tensor create_ov_output_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                   std::shared_ptr<ov::InferRequest> infer_request,
                                   int output_index,
                                   const ggml_tensor * ggml_tensor) {
    if (auto sliced = try_make_kv_sliced_tensor(ggml_decoder, std::string(ggml_tensor->name), ggml_tensor)) {
        return *sliced;
    }

    // Disabling for now as gpu has bug with in-place ScatterUpdate with remote tensors, can re-enable once CVS-186519 is fixed
    // if (ggml_tensor->extra != nullptr && !ggml_decoder->is_splited_model()) {
    //     auto * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
    //     if (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
    //         auto * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
    //         return *tensor_extra->tensor;
    //     }
    // }

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
            output_shape = ggml_decoder->get_shape(ggml_tensor->view_src);
            output_data = ggml_tensor->view_src->data;
        } else {
            output_shape = ggml_decoder->get_shape(ggml_tensor);
        }
    }
    ov::Tensor output_tensor(output_type, output_shape, output_data);
    return output_tensor;
}

enum ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx) {
    auto & core = ov_singleton_core();
    const auto & config = ggml_openvino_get_compile_config();
    const auto & device = r_ctx->device;
    const auto & stateful = r_ctx->stateful;
    static auto is_static = false;

    static const bool cache_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    graph_key key(cgraph);
    bool key_seen = false;
    if (!cache_disabled) {
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        key_seen = r_ctx->decoder_cache.find(key) != r_ctx->decoder_cache.end();
    }

    bool model_is_splitted = key_seen ? false : is_model_splitted(cgraph);

    if (is_naive(cgraph)) {
        if (!model_is_splitted) {
            return naive_compute(cgraph, core, device, config);
        }
    }

    auto start_time = ggml_time_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    const bool cache_enabled = !model_is_splitted && !cache_disabled;
    bool cache_hit = false;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;
    int64_t ov_raw_infer_start;

    {
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
            if (old_m_params.kv_buffer_changed(m_params)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->add_extra_inputs();
            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(key);
                ov_input_names = r_ctx->ov_input_names_cache.at(key);
                ov_output_names = r_ctx->ov_output_names_cache.at(key);
            }

            if (stateful) {
                const auto * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_data = (int32_t *) inp_pos->data;
                auto pos_shape = ggml_decoder->get_shape(inp_pos);
                if (pos_data[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_size = pos_shape[3];
                } else if (r_ctx->stateful_kv_size == static_cast<size_t>(pos_data[0])) {
                    r_ctx->stateful_kv_size += pos_shape[3];
                } else {
                    auto states = infer_request->query_state();
                    for (auto state : states) {
                        auto state_tensor = state.get_state();
                        auto state_tensor_shape = state_tensor.get_shape();
                        if (static_cast<uint32_t>(pos_data[0]) > r_ctx->stateful_kv_size) {
                            std::string state_name;
                            try {
                                state_name = r_ctx->kv_state_input_name_map.at(state.get_name());
                            } catch (...) {
                                GGML_LOG_ERROR(
                                    "GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                return GGML_STATUS_FAILED;
                            }
                            auto kv_tensor = get_ov_input_tensor(ggml_decoder, state_name);
                            kv_tensor.set_shape({state_tensor_shape[0], kv_tensor.get_shape()[2], state_tensor_shape[2],
                                                 state_tensor_shape[3]});
                            state_tensor = kv_tensor;
                            state_tensor_shape = state_tensor.get_shape();
                        }
                        ov::Coordinate begin = {0, 0, 0, 0};
                        ov::Coordinate end = {state_tensor_shape[0], static_cast<uint32_t>(pos_data[0]),
                                              state_tensor_shape[2], state_tensor_shape[3]};
                        ov::Tensor new_state_tensor(state_tensor, begin, end);
                        state.set_state(new_state_tensor);
                    }
                    r_ctx->stateful_kv_size = pos_data[0] + pos_shape[3];
                }
            }

            decoder_end_time = ggml_time_us();
            conversion_end_time = decoder_end_time;
            compile_end_time = decoder_end_time;
        } else {
            // Fail fast: a cache-miss recompile feeds weight data to compile_model, but
            // GGML_OPENVINO_RELEASE_WEIGHTS (or GGML_OPENVINO_MEMORY_OPTIMIZE on GPU)
            // may have already dropped the host weight pages
            // (they would read as zeros). That mode requires stable graph shapes.
            if (ggml_openvino_weight_buffers_released()) {
                GGML_ABORT(
                    "ggml-openvino: a new graph needs to be compiled but host weight buffers were already "
                    "released via GGML_OPENVINO_RELEASE_WEIGHTS/GGML_OPENVINO_MEMORY_OPTIMIZE. This mode requires "
                    "stable graph shapes; disable host weight release for dynamic workloads.");
            }
            if (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(key);
            }

            // Frontend-level compiled-model cache (GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR): if this model
            // was compiled before, import the saved blob and skip requant + convert +
            // compile. Only the dynamic single-model path is cached (split models compile
            // two graphs and are left to the plugin-level ov::cache_dir). The decoder is
            // still needed for I/O mapping, but can be built without weight nodes since
            // the weights are baked into the imported CompiledModel.
            const std::string model_cache_dir = ggml_openvino_model_cache_dir();
            uint64_t model_fp = 0;
            std::string blob_path, manifest_path;
            bool imported = false;
            // When the frontend model cache is active it supersedes the plugin-level
            // ov::cache_dir: a blob exported from a model compiled WITH cache_dir cannot
            // be re-imported (import returns an uninitialized model). Strip cache_dir /
            // cache_mode from the config used for the cached compile and the import.
            ov::AnyMap mc_config = config;
            if (!model_cache_dir.empty()) {
                mc_config.erase("CACHE_DIR");
                mc_config.erase("CACHE_MODE");
            }
            if (!model_cache_dir.empty() && !model_is_splitted) {
                const uint64_t extra_cfg = ggml_openvino_model_cache_extra_cfg(device, stateful);
                model_fp = ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/true, m_params.rope_params,
                                                           15, extra_cfg);
                blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);

                std::ifstream blob_in(blob_path, std::ios::binary);
                bool blob_ok = blob_in.is_open();
                bool manifest_ok = blob_ok && ggml_openvino_model_cache_verify_manifest(manifest_path, cgraph, model_fp);
                if (blob_ok && manifest_ok) {
                    int64_t import_start = ggml_time_us();
                    try {
                        ov::CompiledModel cm;
                        auto remote_context = ggml_openvino_get_remote_context();
                        if (remote_context.has_value()) {
                            cm = core.import_model(blob_in, remote_context.value(), mc_config);
                        } else {
                            cm = core.import_model(blob_in, device, mc_config);
                        }
                        // Lightweight decoder: names-only weight map (membership is all the
                        // decoder needs; weights live in the imported model).
                        std::map<std::string, std::shared_ptr<ov::Node>> weight_names;
                        for (const auto & n : GgmlOvDecoder::collect_weight_names(cgraph)) {
                            weight_names[n] = nullptr;
                        }
                        ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, weight_names,
                                                                       is_static, stateful, model_is_splitted);
                        infer_request = std::make_shared<ov::InferRequest>(cm.create_infer_request());
                        entry->ptr = ggml_decoder;
                        // Names must match the decoder's ggml-tensor keys. The non-cached
                        // path keys off Parameter/Result *friendly names* (set by the
                        // frontend); export_model preserves these, and each compiled-model
                        // port's node is exactly that Parameter/Result. Use the port nodes
                        // directly (NOT get_runtime_model(), whose graph differs and is
                        // unsafe to deref this way).
                        for (const auto & p : cm.inputs()) {
                            ov_input_names.push_back(p.get_node()->get_friendly_name());
                        }
                        for (const auto & o : cm.outputs()) {
                            ov_output_names.push_back(o.get_node()->get_friendly_name());
                        }
                        imported = true;
                        if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            GGML_LOG_INFO("  - Model cache import time: %.3f ms \n",
                                          (ggml_time_us() - import_start) / 1000.0);
                        }
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", blob_path.c_str());
                    } catch (const std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed (%s), recompiling\n", e.what());
                        imported = false;
                    }
                }
            }

            std::shared_ptr<ov::Model> model;
            if (imported) {
                decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            } else {
                auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                               stateful, model_is_splitted);
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

                // Export to the frontend model cache for next time. Publish the blob first,
                // then the manifest, so a cache hit only sees fully written artifacts.
                if (!model_cache_dir.empty() && !model_is_splitted && model_fp != 0) {
                    try {
                        const std::string blob_tmp = blob_path + ".tmp";
                        const std::string manifest_tmp = manifest_path + ".tmp";
                        if (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp)) {
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
                entry->ptr = ggml_decoder;

                for (const auto & ov_param : model->get_parameters()) {
                    ov_input_names.push_back(ov_param->get_friendly_name());
                }
                for (const auto & ov_output : model->get_results()) {
                    ov_output_names.push_back(ov_output->get_friendly_name());
                }
            }  // end non-imported (compile) path

            if (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[key] = infer_request;
                r_ctx->ov_input_names_cache[key] = ov_input_names;
                r_ctx->ov_output_names_cache[key] = ov_output_names;
            }

            if (stateful && cache_enabled) {
                const auto * inp_pos = get_inp_pos_tensor(cgraph);
                auto pos_shape = ggml_decoder->get_shape(inp_pos);
                r_ctx->stateful_kv_size = pos_shape[3];
                const auto kv_param_res_names = ggml_decoder->get_kv_param_res_names();
                for (const auto & pair : kv_param_res_names) {
                    r_ctx->kv_state_input_name_map[pair.first + pair.second] = pair.first;
                }
            }
        }

        for (size_t i = 0; i < ov_input_names.size(); i++) {
            auto param_name = ov_input_names[i];
            auto input_tensor = get_ov_input_tensor(ggml_decoder, param_name);
            infer_request->set_input_tensor(i, input_tensor);

            if (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                print_input_tensor_info(param_name, input_tensor);
            }
        }

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
            auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        ov_raw_infer_start = ggml_time_us();
        infer_request->infer();
        infer_end_time = ggml_time_us();

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
            if (!cache_hit) {
                GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n",
                              (conversion_end_time - decoder_end_time) / 1000.0);
                GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_end_time - conversion_end_time) / 1000.0);
            }
            GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_end_time - compile_end_time) / 1000.0);
            GGML_LOG_INFO("  - OV raw infer time: %.3f ms \n", (infer_end_time - ov_raw_infer_start) / 1000.0);
        }
    }

    // GGML_OPENVINO_RELEASE_WEIGHTS (or GGML_OPENVINO_MEMORY_OPTIMIZE on GPU): the plugin holds its own device copy of
    // every weight after compile, so the host weight buffers can be dropped to reclaim
    // RSS. The GPU backend uses a single dynamic-shape model for both prefill and decode,
    // so once a graph is compiled it is reused for the whole session — the only thing
    // that forces a recompile is clear_caches() on backend teardown. We therefore release
    // on the first cache-hit (model compiled, plugin has its copy) and, crucially, pin the
    // compiled-model cache so it survives backend teardown (see ggml_backend_openvino_free).
    // Without the pin, a later test/context would recompile against the now-dropped pages.
    // A genuinely new graph still fails fast at the cache-miss compile branch.
    if (cache_hit && ggml_openvino_release_weights_enabled(device) &&
        !ggml_openvino_weight_buffers_released()) {
        ggml_openvino_release_weight_buffers();
    }

    return GGML_STATUS_SUCCESS;
}

enum ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx) {
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
        return naive_compute(cgraph, core, device, config);
    }

    auto start_time = ggml_time_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    const auto * inp_pos = get_inp_pos_tensor(cgraph);
    const auto is_prefill = get_is_prefill(cgraph, inp_pos);
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

    if (cache_hit) {
        ggml_decoder = entry->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_hit = old_m_params.can_reuse_statically(m_params);
    }

    std::vector<std::string> ov_input_names_local;
    std::vector<std::string> ov_output_names_local;

    if (cache_hit) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->set_model_params(m_params);
        ggml_decoder->set_compute_params(c_params);
        if (old_m_params.kv_buffer_changed(m_params)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->add_extra_inputs();
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(key) : r_ctx->infer_request_cache.at(key);
            ov_input_names_local = r_ctx->ov_input_names_cache.at(key);
            ov_output_names_local = r_ctx->ov_output_names_cache.at(key);
        }

        decoder_end_time = ggml_time_us();
        conversion_end_time = decoder_end_time;
        compile_end_time = decoder_end_time;
    } else {
        if (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(key);
            r_ctx->infer_request_cache_prefill.erase(key);
        }

        std::shared_ptr<ov::Model> model;
        auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

        if (m_params.n_heads_kv == -1) {
            // graph is not a LLM, e.g. context-shift graph
            prefill_chunk_size = inp_pos->ne[0];
        }
        auto ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(
            cgraph, m_params, c_params, model_weights, is_static, stateful, false, true, prefill_chunk_size);
        auto ggml_decoder_decode = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                                   stateful, false, false, prefill_chunk_size);
        decoder_end_time = ggml_time_us();

        const bool dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
        const auto dump_ir_timestamp = static_cast<long long>(ggml_time_us());

        auto build_static_model = [&core, &config, dump_ir, dump_ir_timestamp](
                          std::shared_ptr<GgmlOvDecoder> decoder,
                          const char * tag,
                          std::shared_ptr<ov::Model> & model,
                          ov::CompiledModel & compiled_model,
                          std::shared_ptr<ov::InferRequest> & infer_request,
                          int64_t & local_conversion_end_time,
                          int64_t & local_compile_end_time) {
            auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(decoder);
            model = ov::frontend::ggml::FrontEnd::convert(input_model);
            decoder->clear_model_weights();
            local_conversion_end_time = ggml_time_us();

            if (dump_ir) {
                char timestamped_filename[64];
                snprintf(timestamped_filename, sizeof(timestamped_filename), "model_%s_%lld.xml", tag,
                         dump_ir_timestamp);
                ov::serialize(model, timestamped_filename);
            }

            compiled_model = core.compile_model(model, device, config);
            infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
            local_compile_end_time = ggml_time_us();
        };
        std::shared_ptr<ov::Model> model_prefill;
        std::shared_ptr<ov::Model> model_decode;
        ov::CompiledModel compiled_model_prefill;
        ov::CompiledModel compiled_model_decode;
        std::shared_ptr<ov::InferRequest> infer_request_prefill;
        std::shared_ptr<ov::InferRequest> infer_request_decode;
        int64_t prefill_conversion_end_time;
        int64_t decode_conversion_end_time;
        int64_t prefill_compile_end_time;
        int64_t decode_compile_end_time;
        auto prefill_future = std::async(std::launch::async, build_static_model, ggml_decoder_prefill, "prefill",
                                         std::ref(model_prefill), std::ref(compiled_model_prefill),
                                         std::ref(infer_request_prefill), std::ref(prefill_conversion_end_time),
                                         std::ref(prefill_compile_end_time));
        auto decode_future = std::async(std::launch::async, build_static_model, ggml_decoder_decode, "decode",
                                        std::ref(model_decode), std::ref(compiled_model_decode),
                                        std::ref(infer_request_decode), std::ref(decode_conversion_end_time),
                                        std::ref(decode_compile_end_time));
        prefill_future.get();
        decode_future.get();
        conversion_end_time = std::max(prefill_conversion_end_time, decode_conversion_end_time);
        compile_end_time = std::max(prefill_compile_end_time, decode_compile_end_time);

        model = is_prefill ? model_prefill : model_decode;
        ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
        infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
        entry->ptr = ggml_decoder;

        for (const auto & ov_param : model->get_parameters()) {
            ov_input_names_local.push_back(ov_param->get_friendly_name());
        }
        for (const auto & ov_output : model->get_results()) {
            ov_output_names_local.push_back(ov_output->get_friendly_name());
        }

        if (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache_prefill[key] = infer_request_prefill;
            r_ctx->infer_request_cache[key] = infer_request_decode;
            r_ctx->ov_input_names_cache[key] = ov_input_names_local;
            r_ctx->ov_output_names_cache[key] = ov_output_names_local;
        }
    }

    if (is_prefill) {
        auto inp_len = get_inp_pos_n_tokens(cgraph, inp_pos);
        for (int chunk_index = 0; chunk_index * prefill_chunk_size < inp_len; chunk_index++) {
            for (size_t i = 0; i < ov_input_names_local.size(); i++) {
                auto param_name = ov_input_names_local[i];
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
            ov_raw_infer_total += ggml_time_us() - ov_raw_infer_start;

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
            auto param_name = ov_input_names_local[i];
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

enum ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config) {
    if (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        return GGML_STATUS_SUCCESS;
    }

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
    if (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        core.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE));
    } else {
        core.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
    }
    if (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), config).create_infer_request());
    } else {
        infer_request =
            std::make_shared<ov::InferRequest>(core.compile_model(model, device, config).create_infer_request());
    }

    auto ov_params = model->get_parameters();
    for (size_t i = 0; i < ov_params.size(); i++) {
        auto param_name = ov_params[i]->get_friendly_name();
        auto input_tensor = get_ov_input_tensor(decoder, param_name);
        infer_request->set_input_tensor(i, input_tensor);
    }

    // Use get_output_tensor + memcpy instead of set_output_tensor to avoid memory overwritten
    // when i/o buffer overlaps, e.g. the cgraph is a single PERMUTE

    infer_request->infer();

    auto ov_results = model->get_results();
    for (size_t i = 0; i < ov_results.size(); i++) {
        auto output_tensor = infer_request->get_output_tensor(i);
        const auto & model_outputs = decoder->get_model_outputs();
        auto model_output_it = model_outputs.find(ov_results[i]->get_friendly_name());
        if (model_output_it == model_outputs.end()) {
            // Debug-only output added via GGML_OPENVINO_DEBUG_NODE; nothing to copy into.
            if (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                print_output_tensor_info(ov_results[i]->get_friendly_name(), output_tensor, output_tensor.data());
            }
            continue;
        }
        auto * ggml_tensor = model_output_it->second;
        std::memcpy(ggml_tensor->data, output_tensor.data(), output_tensor.get_byte_size());
    }
    return GGML_STATUS_SUCCESS;
}

namespace {
template <typename T> void set_zero_diagonal(std::vector<T> & matrix, size_t rows, size_t cols, T zero_value = T{}) {
    for (size_t i = 0; i < rows; ++i) {
        size_t diag_col = std::min(i, cols - 1);
        matrix[i * cols + diag_col] = zero_value;
    }
}

ov::Tensor make_contiguous_split_input_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                              const struct ggml_tensor * ggml_tensor,
                                              const ov::Shape & input_shape) {
    const size_t element_size = ggml_type_size(ggml_tensor->type);
    const size_t block_size = ggml_blck_size(ggml_tensor->type);

    GGML_ASSERT(block_size == 1 && "non-contiguous split inputs must be plain element types");

    const struct ggml_tensor * source_tensor = ggml_tensor->view_src != nullptr ? ggml_tensor->view_src : ggml_tensor;
    const size_t source_offset = ggml_tensor->view_src != nullptr ? ggml_tensor->view_offs : 0;

    std::vector<uint8_t> source_data(ggml_nbytes(source_tensor));
    ggml_backend_tensor_get(source_tensor, source_data.data(), 0, source_data.size());

    ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
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

ov::Tensor convert_ggml_input_to_ov(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & name) {
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(name);

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
        input_shape = ggml_decoder->get_shape(ggml_tensor->view_src);
    } else {
        input_shape = ggml_decoder->get_shape(ggml_tensor);
    }

    if (ggml_decoder->is_splited_model() && !ggml_is_contiguous(ggml_tensor)) {
        return make_contiguous_split_input_tensor(ggml_decoder, ggml_tensor, input_shape);
    }

    auto input_tensor = ov::Tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape, input_data);
    return input_tensor;
}
}  // namespace

ov::Tensor get_ov_input_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & param_name) {
    ov::Tensor input_tensor;
    auto extra_input = ggml_decoder->get_model_extra_inputs().find(param_name);
    if (extra_input != ggml_decoder->get_model_extra_inputs().end()) {
        input_tensor = ov::Tensor(extra_input->second.type, extra_input->second.shape);
        *input_tensor.data<int64_t>() = extra_input->second.value;
    } else {
        input_tensor = convert_ggml_input_to_ov(ggml_decoder, param_name);
    }
    return input_tensor;
}

ov::Tensor get_ov_input_tensor_static_decode(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
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
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        std::memcpy(input_tensor.data(), ggml_tensor->data, n_planes * ggml_type_size(ggml_tensor->type));
        return input_tensor;
    }

    if (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape input_shape = {1, 1, 1, 1};
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        int32_t inp_out_id = *((int32_t *) ggml_tensor->data);
        assert(ggml_tensor->ne[0] == 1);
        assert(inp_out_id == 0);
        *input_tensor.data<int32_t>() = inp_out_id;
        return input_tensor;
    }

    if (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t context_size = ggml_decoder->get_ctx_size();
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

ov::Tensor get_ov_input_tensor_static_prefill(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
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
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
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
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
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
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
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

    if (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t cols = ggml_tensor->ne[0];
        size_t rows = ggml_tensor->ne[1];
        size_t chunk_valid_rows = std::min(chunk_size, rows - chunk_index * chunk_size);
        size_t context_size = ggml_decoder->get_ctx_size();
        if (ggml_tensor->type == GGML_TYPE_F16) {
            const auto * ggml_data =
                static_cast<const ggml_fp16_t *>(ggml_tensor->data) + chunk_index * chunk_size * cols;
            std::vector<ggml_fp16_t> padded_data = pad_input<ggml_fp16_t>(ggml_data, chunk_valid_rows, cols, chunk_size,
                                                                          context_size, GGML_FP32_TO_FP16(-INFINITY));
            set_zero_diagonal(padded_data, chunk_size, context_size, GGML_FP32_TO_FP16(0.0f));
            ov::Tensor input_tensor(ov::element::f16, ov::Shape{1, 1, chunk_size, context_size});
            std::memcpy(input_tensor.data(), padded_data.data(), padded_data.size() * sizeof(ggml_fp16_t));
            return input_tensor;
        }

        const auto * ggml_data = static_cast<const float *>(ggml_tensor->data) + chunk_index * chunk_size * cols;
        std::vector<float> padded_data =
            pad_input<float>(ggml_data, chunk_valid_rows, cols, chunk_size, context_size, -INFINITY);
        set_zero_diagonal(padded_data, chunk_size, context_size);
        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, context_size});
        auto * data_ptr = input_tensor.data<float>();
        std::copy(padded_data.begin(), padded_data.begin() + chunk_size * context_size, data_ptr);
        return input_tensor;
    }

    return get_ov_input_tensor(ggml_decoder, param_name);
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
              << std::endl;
    switch (tensor.get_element_type()) {
    case ov::element::f32: {
        if (name.find("self_kq_mask") == std::string::npos && name.find("KQ_mask") == std::string::npos) {
            std::cout << *(tensor.data<float>()) << std::endl;
        } else {
            size_t rows = tensor.get_shape()[2];
            size_t cols = tensor.get_shape()[3];
            auto * data = tensor.data<float>();
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    float val = data[i * cols + j];
                    if (std::isinf(val) && val < 0) {
                        std::cout << std::setw(5) << "-inf";
                    } else {
                        std::cout << std::setw(5) << val;
                    }
                }
                std::cout << std::endl;
            }
        }

        break;
    }
    case ov::element::f16:
        std::cout << *(tensor.data<ov::float16>()) << std::endl;
        break;
    case ov::element::i32:
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int32_t>()[i] << " ";
        }
        std::cout << std::endl;
        break;
    case ov::element::i64:
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int64_t>()[i] << " ";
        }
        std::cout << std::endl;
        break;
    default:
        break;
    }
}

void print_output_tensor_info(const std::string & name, const ov::Tensor & tensor, const void * output_dst) {
    std::cout << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << std::endl;

    auto print_float_stats = [](const std::string & type_name, size_t size, auto get_value) {
        if (size == 0) {
            return;
        }

        float first = get_value(0);
        float min = first;
        float max = first;
        double sum = first;

        for (size_t i = 1; i < size; ++i) {
            float v = get_value(i);
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
            sum += v;
        }
        double mean = sum / size;

        std::cout << std::right << std::setw(6) << type_name << std::right << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << std::endl;
        std::cout << std::right << std::setw(6) << "" << std::right << std::setw(12) << first << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << mean << std::endl;
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

#pragma GCC diagnostic pop
