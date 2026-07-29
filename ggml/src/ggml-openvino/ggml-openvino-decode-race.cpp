#include "ggml-openvino-decode-race.h"

#include "ggml-decoder.h"
#include "ggml-impl.h"
#include "ggml-openvino-extra.h"
#include "utils.h"
#include "openvino/frontend.h"
#include "openvino/input_model.h"
#include "openvino/translate_session.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

struct race_slot {
    std::shared_ptr<GgmlOvDecoder>    decoder;
    std::shared_ptr<ov::InferRequest> infer;
    std::vector<std::string>          input_names;
    std::vector<std::string>          output_names;
};

struct race_cache {
    std::mutex                                 mutex;
    std::unordered_map<std::string, race_slot> slots;
};

static race_cache           g_race_cache;
static std::mutex           g_cpu_infer_reap_mu;
static std::mutex           g_gpu_infer_reap_mu;
static int                  g_cpu_sleep_remaining = 0;
static int                  g_gpu_sleep_remaining = 0;
static int                  g_decode_step         = 0;
static bool                 g_atexit_registered   = false;

enum class race_step_mode : uint8_t { dual = 0, cpu_only = 1, gpu_only = 2 };

struct race_diag_row {
    int           step   = 0;
    race_step_mode mode  = race_step_mode::dual;
    int           winner = 0;
    float         cpu_ms = -1.f;
    float         gpu_ms = -1.f;
};

static std::mutex              g_diag_mu;
static std::vector<race_diag_row> g_diag_rows;

static graph_key race_key_for(const graph_key & base, const std::string & device) {
    graph_key k = base;
    k.last_node_name += ":race:" + device;
    return k;
}

static std::vector<const ggml_tensor *> collect_cache_tensors(ggml_cgraph * cgraph) {
    std::vector<const ggml_tensor *>       out;
    std::unordered_map<const void *, bool> seen;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        auto try_add = [&](ggml_tensor * t) {
            if (t == nullptr || t->name == nullptr) {
                return;
            }
            if (strncmp(t->name, "cache_", 6) != 0) {
                return;
            }
            if (seen.find(t->data) == seen.end()) {
                seen[t->data] = true;
                out.push_back(t);
            }
        };
        try_add(node);
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            try_add(node->src[j]);
        }
    }
    return out;
}

struct shadow_store {
    std::unordered_map<std::string, std::vector<uint8_t>> bufs;

    void sync_from_primary(const ggml_tensor * t) {
        const size_t n = ggml_nbytes(t);
        auto & b       = bufs[t->name];
        if (b.size() < n) {
            b.resize(n);
        }
        memcpy(b.data(), t->data, n);
    }

    void sync_all_from_primary(const std::vector<const ggml_tensor *> & tensors) {
        for (const ggml_tensor * t : tensors) {
            sync_from_primary(t);
        }
    }

    void sync_to_primary(const ggml_tensor * t) {
        const size_t n = ggml_nbytes(t);
        auto it        = bufs.find(t->name);
        if (it != bufs.end() && it->second.size() >= n) {
            memcpy(t->data, it->second.data(), n);
        }
    }

    void * ptr(const ggml_tensor * t) {
        sync_from_primary(t);
        return bufs[t->name].data();
    }
};

static void drain_infer_request(const std::shared_ptr<ov::InferRequest> & infer, std::mutex & reap_mu) {
    if (!infer) {
        return;
    }
    std::lock_guard<std::mutex> lock(reap_mu);
    try {
        infer->cancel();
    } catch (...) {
    }
    try {
        infer->wait();
    } catch (...) {
    }
}

static void reap_loser_async(const std::shared_ptr<ov::InferRequest> & infer, std::mutex & reap_mu) {
    std::thread([infer, &reap_mu]() { drain_infer_request(infer, reap_mu); }).detach();
}

static enum ggml_status ensure_race_slot(ggml_cgraph *       cgraph,
                                         const graph_key &   base_key,
                                         const std::string & device,
                                         const ov::AnyMap &  config,
                                         ov::Core &          core,
                                         race_slot &         slot) {
    graph_key         key     = race_key_for(base_key, device);
    const std::string map_key = key.first_node_name + "|" + key.last_node_name + "|" + device;

    {
        std::lock_guard<std::mutex> lock(g_race_cache.mutex);
        auto it = g_race_cache.slots.find(map_key);
        if (it != g_race_cache.slots.end()) {
            slot = it->second;
            return GGML_STATUS_SUCCESS;
        }
    }

    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, false);

    auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);
    auto ggml_decoder  = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, false, false,
                                                         is_model_splitted(cgraph), false);
    auto input_model   = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder);
    auto model         = ov::frontend::ggml::FrontEnd::convert(input_model);
    ggml_decoder->clear_model_weights();

    ov::CompiledModel compiled_model;
    auto remote_context = ggml_openvino_get_remote_context();
    if (remote_context.has_value() && ggml_openvino_device_is_gpu(device)) {
        compiled_model = core.compile_model(model, remote_context.value(), config);
    } else {
        compiled_model = core.compile_model(model, device, config);
    }

    race_slot fresh;
    fresh.decoder = ggml_decoder;
    fresh.infer   = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
    for (const auto & ov_param : model->get_parameters()) {
        fresh.input_names.push_back(ov_param->get_friendly_name());
    }
    for (const auto & ov_output : model->get_results()) {
        fresh.output_names.push_back(ov_output->get_friendly_name());
    }

    std::lock_guard<std::mutex> lock(g_race_cache.mutex);
    g_race_cache.slots[map_key] = fresh;
    slot                        = fresh;
    return GGML_STATUS_SUCCESS;
}

static bool is_cache_param(const std::string & name) {
    return name.rfind("cache_", 0) == 0;
}

static enum ggml_status bind_decode_infer(race_slot &    slot,
                                          ggml_cgraph *  cgraph,
                                          bool           use_shadow_kv,
                                          shadow_store * kv_shadows,
                                          shadow_store * out_shadows) {
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, false);
    slot.decoder->set_compute_params(c_params);
    slot.decoder->set_model_params(m_params);
    slot.decoder->update_io(cgraph);
    slot.decoder->add_extra_inputs();

    auto & infer = *slot.infer;

    for (size_t i = 0; i < slot.input_names.size(); ++i) {
        const auto & param_name = slot.input_names[i];
        ov::Tensor   input_tensor;
        if (use_shadow_kv && is_cache_param(param_name) && kv_shadows != nullptr) {
            const ggml_tensor * gt = slot.decoder->get_input_ggml_tensor(param_name);
            input_tensor           = get_ov_input_tensor(slot.decoder, param_name);
            input_tensor           = ov::Tensor(input_tensor.get_element_type(), input_tensor.get_shape(),
                                                kv_shadows->ptr(const_cast<ggml_tensor *>(gt)));
        } else {
            input_tensor = get_ov_input_tensor(slot.decoder, param_name);
        }
        infer.set_input_tensor(i, input_tensor);
    }

    for (size_t i = 0; i < slot.output_names.size(); ++i) {
        auto * ggml_tensor = slot.decoder->get_model_outputs().at(slot.output_names[i]);
        if (ggml_nbytes(ggml_tensor) == 0) {
            continue;
        }
        ov::Tensor output_tensor;
        if (use_shadow_kv && out_shadows != nullptr) {
            output_tensor = create_ov_output_tensor(slot.decoder, slot.infer, i, ggml_tensor);
            out_shadows->sync_from_primary(ggml_tensor);
            output_tensor = ov::Tensor(output_tensor.get_element_type(), output_tensor.get_shape(),
                                       out_shadows->ptr(ggml_tensor));
        } else {
            output_tensor = create_ov_output_tensor(slot.decoder, slot.infer, i, ggml_tensor);
        }
        infer.set_output_tensor(i, output_tensor);
    }

    return GGML_STATUS_SUCCESS;
}

static void copy_outputs_from_shadow(race_slot & slot, shadow_store & out_shadows) {
    for (size_t i = 0; i < slot.output_names.size(); ++i) {
        auto * ggml_tensor = slot.decoder->get_model_outputs().at(slot.output_names[i]);
        if (ggml_nbytes(ggml_tensor) == 0) {
            continue;
        }
        out_shadows.sync_to_primary(ggml_tensor);
    }
}

static void record_diag(const race_diag_row & row) {
    if (!ggml_openvino_decode_race_diag_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_diag_mu);
    g_diag_rows.push_back(row);
}

static void dump_race_diag_summary() {
    if (!ggml_openvino_decode_race_diag_enabled()) {
        return;
    }
    std::vector<race_diag_row> rows;
    {
        std::lock_guard<std::mutex> lock(g_diag_mu);
        rows = g_diag_rows;
    }
    if (rows.empty()) {
        return;
    }

    int dual_n = 0;
    int cpu_only_n = 0;
    int gpu_only_n = 0;
    int dual_cpu_wins = 0;
    int dual_gpu_wins = 0;
    int winner_flips = 0;
    int last_dual_winner = -1;
    std::vector<int> flip_steps;
    std::vector<float> dual_gaps;
    std::vector<float> dual_cpu_ms;
    std::vector<float> dual_gpu_ms;

    for (const auto & r : rows) {
        switch (r.mode) {
        case race_step_mode::dual:
            dual_n++;
            if (r.winner == 0) {
                dual_cpu_wins++;
            } else {
                dual_gpu_wins++;
            }
            if (last_dual_winner >= 0 && last_dual_winner != r.winner) {
                winner_flips++;
                if (flip_steps.size() < 32) {
                    flip_steps.push_back(r.step);
                }
            }
            last_dual_winner = r.winner;
            if (r.cpu_ms >= 0.f && r.gpu_ms >= 0.f) {
                dual_gaps.push_back(r.gpu_ms - r.cpu_ms);
                dual_cpu_ms.push_back(r.cpu_ms);
                dual_gpu_ms.push_back(r.gpu_ms);
            }
            break;
        case race_step_mode::cpu_only:
            cpu_only_n++;
            break;
        case race_step_mode::gpu_only:
            gpu_only_n++;
            break;
        }
    }

    auto avg_first_last = [](const std::vector<float> & v) -> std::pair<float, float> {
        if (v.empty()) {
            return { 0.f, 0.f };
        }
        const size_t n = std::min<size_t>(10, v.size());
        float a = 0.f;
        float b = 0.f;
        for (size_t i = 0; i < n; i++) {
            a += v[i];
            b += v[v.size() - n + i];
        }
        return { a / n, b / n };
    };

    fprintf(stderr, "\n=== OpenVINO decode race diagnostics (process exit) ===\n");
    fprintf(stderr, "decode steps: %zu (dual: %d, cpu_only: %d, gpu_only: %d)\n", rows.size(), dual_n, cpu_only_n,
            gpu_only_n);
    fprintf(stderr, "dual races: CPU wins %d, GPU wins %d, winner changes %d\n", dual_cpu_wins, dual_gpu_wins,
            winner_flips);
    if (!flip_steps.empty()) {
        fprintf(stderr, "winner flip at decode steps (first %zu):", flip_steps.size());
        for (int s : flip_steps) {
            fprintf(stderr, " %d", s);
        }
        fprintf(stderr, "\n");
    }
    if (!dual_gaps.empty()) {
        float gap_sum = 0.f;
        float gap_min = dual_gaps[0];
        float gap_max = dual_gaps[0];
        for (float g : dual_gaps) {
            gap_sum += g;
            gap_min = std::min(gap_min, g);
            gap_max = std::max(gap_max, g);
        }
        auto [cpu_a0, cpu_a1] = avg_first_last(dual_cpu_ms);
        auto [gpu_a0, gpu_a1] = avg_first_last(dual_gpu_ms);
        fprintf(stderr, "dual infer latency ms (GPU-CPU gap, both finished): mean %+.2f min %+.2f max %+.2f (n=%zu)\n",
                gap_sum / dual_gaps.size(), gap_min, gap_max, dual_gaps.size());
        fprintf(stderr, "dual CPU ms avg first10/last10: %.2f / %.2f; GPU ms avg first10/last10: %.2f / %.2f\n", cpu_a0,
                cpu_a1, gpu_a0, gpu_a1);
    }

    std::vector<float> dual_winner_ms;
    for (const auto & r : rows) {
        if (r.mode != race_step_mode::dual) {
            continue;
        }
        const float w = (r.winner == 0) ? r.cpu_ms : r.gpu_ms;
        if (w >= 0.f) {
            dual_winner_ms.push_back(w);
        }
    }
    if (!dual_winner_ms.empty()) {
        float sum = 0.f;
        for (float w : dual_winner_ms) {
            sum += w;
        }
        auto [w0, w1] = avg_first_last(dual_winner_ms);
        fprintf(stderr, "dual race winner latency ms: mean %.2f, avg first10/last10 %.2f / %.2f (n=%zu)\n",
                sum / dual_winner_ms.size(), w0, w1, dual_winner_ms.size());
    }
    fprintf(stderr, "====================================================\n\n");
}

static void register_race_diag_atexit_once() {
    if (g_atexit_registered) {
        return;
    }
    g_atexit_registered = true;
    std::atexit(dump_race_diag_summary);
}

static enum ggml_status run_sync_infer(race_slot &                  slot,
                                       ggml_cgraph *                cgraph,
                                       bool                         use_shadow_kv,
                                       shadow_store *               kv_shadows,
                                       shadow_store *               out_shadows,
                                       float *                      out_ms) {
    drain_infer_request(slot.infer, use_shadow_kv ? g_gpu_infer_reap_mu : g_cpu_infer_reap_mu);
    if (bind_decode_infer(slot, cgraph, use_shadow_kv, kv_shadows, out_shadows) != GGML_STATUS_SUCCESS) {
        return GGML_STATUS_FAILED;
    }
    const int64_t t0 = ggml_time_us();
    slot.infer->infer();
    if (out_ms) {
        *out_ms = (ggml_time_us() - t0) / 1000.f;
    }
    return GGML_STATUS_SUCCESS;
}

}  // namespace

enum ggml_status ov_graph_compute_decode_race(ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx) {
    GGML_UNUSED(r_ctx);

    register_race_diag_atexit_once();

    auto &            core    = ov_singleton_core();
    const auto &      config  = ggml_openvino_get_compile_config();
    const std::string cpu_dev = ggml_openvino_get_race_cpu_device();
    const std::string gpu_dev = ggml_openvino_get_race_gpu_device();
    const int         loser_sleep = ggml_openvino_race_loser_sleep_tokens();

    graph_key base_key(cgraph);
    base_key.last_node_name += ":ov_tg";

    race_slot cpu_slot;
    race_slot gpu_slot;
    if (ensure_race_slot(cgraph, base_key, cpu_dev, config, core, cpu_slot) != GGML_STATUS_SUCCESS) {
        return GGML_STATUS_FAILED;
    }
    if (ensure_race_slot(cgraph, base_key, gpu_dev, config, core, gpu_slot) != GGML_STATUS_SUCCESS) {
        return GGML_STATUS_FAILED;
    }

    g_decode_step++;
    const int step = g_decode_step;

    bool race_cpu = (g_cpu_sleep_remaining == 0);
    bool race_gpu = (g_gpu_sleep_remaining == 0);
    if (!race_cpu) {
        g_cpu_sleep_remaining--;
    }
    if (!race_gpu) {
        g_gpu_sleep_remaining--;
    }

    const auto cache_tensors = collect_cache_tensors(cgraph);
    shadow_store kv_shadow;

    race_diag_row diag{};
    diag.step = step;

    if (race_cpu && race_gpu) {
        diag.mode = race_step_mode::dual;
        drain_infer_request(cpu_slot.infer, g_cpu_infer_reap_mu);
        drain_infer_request(gpu_slot.infer, g_gpu_infer_reap_mu);

        shadow_store out_shadow_gpu;
        kv_shadow.sync_all_from_primary(cache_tensors);

        if (bind_decode_infer(cpu_slot, cgraph, false, nullptr, nullptr) != GGML_STATUS_SUCCESS) {
            return GGML_STATUS_FAILED;
        }
        if (bind_decode_infer(gpu_slot, cgraph, true, &kv_shadow, &out_shadow_gpu) != GGML_STATUS_SUCCESS) {
            return GGML_STATUS_FAILED;
        }

        cpu_slot.infer->start_async();
        gpu_slot.infer->start_async();

        const int64_t                t0 = ggml_time_us();
        const std::chrono::milliseconds poll(0);
        int                          winner     = -1;
        int64_t                      cpu_done_us = -1;
        int64_t                      gpu_done_us = -1;

        for (;;) {
            if (cpu_done_us < 0 && cpu_slot.infer->wait_for(poll)) {
                cpu_done_us = ggml_time_us();
            }
            if (gpu_done_us < 0 && gpu_slot.infer->wait_for(poll)) {
                gpu_done_us = ggml_time_us();
            }
            if (winner < 0 && cpu_done_us >= 0) {
                winner = 0;
                try {
                    gpu_slot.infer->cancel();
                } catch (...) {
                }
                reap_loser_async(gpu_slot.infer, g_gpu_infer_reap_mu);
                g_gpu_sleep_remaining = loser_sleep;
            }
            if (winner < 0 && gpu_done_us >= 0) {
                winner = 1;
                try {
                    cpu_slot.infer->cancel();
                } catch (...) {
                }
                reap_loser_async(cpu_slot.infer, g_cpu_infer_reap_mu);
                g_cpu_sleep_remaining = loser_sleep;
            }
            if (winner >= 0) {
                break;
            }
            std::this_thread::yield();
        }

        diag.winner = winner;
        if (cpu_done_us >= 0) {
            diag.cpu_ms = (cpu_done_us - t0) / 1000.f;
        }
        if (gpu_done_us >= 0) {
            diag.gpu_ms = (gpu_done_us - t0) / 1000.f;
        }

        if (winner == 1) {
            copy_outputs_from_shadow(gpu_slot, out_shadow_gpu);
            for (const ggml_tensor * t : cache_tensors) {
                kv_shadow.sync_to_primary(t);
            }
            if (cl_command_queue queue = ggml_openvino_get_cl_queue()) {
                clFinish(queue);
            }
        }

        record_diag(diag);
        return GGML_STATUS_SUCCESS;
    }

    float ms = 0.f;
    if (race_cpu) {
        diag.mode   = race_step_mode::cpu_only;
        diag.winner = 0;
        if (run_sync_infer(cpu_slot, cgraph, false, nullptr, nullptr, &ms) != GGML_STATUS_SUCCESS) {
            return GGML_STATUS_FAILED;
        }
        diag.cpu_ms = ms;
    } else {
        diag.mode   = race_step_mode::gpu_only;
        diag.winner = 1;
        if (run_sync_infer(gpu_slot, cgraph, false, nullptr, nullptr, &ms) != GGML_STATUS_SUCCESS) {
            return GGML_STATUS_FAILED;
        }
        diag.gpu_ms = ms;
        if (cl_command_queue queue = ggml_openvino_get_cl_queue()) {
            clFinish(queue);
        }
    }

    record_diag(diag);
    return GGML_STATUS_SUCCESS;
}
