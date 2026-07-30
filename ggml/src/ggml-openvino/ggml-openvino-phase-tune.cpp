#include "ggml-openvino-phase-tune.h"

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
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct tune_slot {
    std::shared_ptr<GgmlOvDecoder>    decoder;
    std::shared_ptr<ov::InferRequest> infer;
    std::vector<std::string>          input_names;
    std::vector<std::string>          output_names;
};

struct tune_cache {
    std::mutex                                 mutex;
    std::unordered_map<std::string, tune_slot> slots;
};

static tune_cache g_tune_cache;
static bool       g_tune_production = false;
static bool       g_atexit_registered = false;

static int                  g_pp_token_cursor = 0;
static int                  g_tg_token_cursor = 0;

static int tune_progress_interval() {
    static const int n = []() {
        const int v = ggml_openvino_getenv_int("GGML_OPENVINO_PHASE_TUNE_PROGRESS", 50);
        return v > 0 ? v : 0;
    }();
    return n;
}

enum class tune_phase : uint8_t { prefill = 0, decode = 1 };

struct tune_sample {
    double sum_ms = 0.0;
    int    count  = 0;
};

static std::mutex g_tune_mu;
static std::unordered_map<int, tune_sample> g_pp_dev0;
static std::unordered_map<int, tune_sample> g_pp_dev1;
static std::unordered_map<int, tune_sample> g_tg_dev0;
static std::unordered_map<int, tune_sample> g_tg_dev1;

static graph_key tune_key_for(const graph_key & base, bool is_prefill, const std::string & device) {
    graph_key k = base;
    k.last_node_name += is_prefill ? ":ov_pp:tune:" : ":ov_tg:tune:";
    k.last_node_name += device;
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

struct kv_backup_entry {
    std::vector<uint8_t> bytes;
};

static void backup_kv(const std::vector<const ggml_tensor *> & tensors,
                      std::vector<kv_backup_entry> &            out) {
    out.clear();
    out.reserve(tensors.size());
    for (const ggml_tensor * t : tensors) {
        kv_backup_entry e;
        const size_t n = ggml_nbytes(t);
        e.bytes.resize(n);
        memcpy(e.bytes.data(), t->data, n);
        out.push_back(std::move(e));
    }
}

static void restore_kv(const std::vector<const ggml_tensor *> & tensors,
                       const std::vector<kv_backup_entry> &    backup) {
    for (size_t i = 0; i < tensors.size() && i < backup.size(); ++i) {
        const size_t n = ggml_nbytes(tensors[i]);
        if (backup[i].bytes.size() >= n) {
            memcpy(tensors[i]->data, backup[i].bytes.data(), n);
        }
    }
}

static enum ggml_status ensure_tune_slot(ggml_cgraph *       cgraph,
                                         const graph_key &   base_key,
                                         bool                is_prefill,
                                         const std::string & device,
                                         const ov::AnyMap &  config,
                                         ov::Core &          core,
                                         tune_slot &         slot) {
    graph_key         key     = tune_key_for(base_key, is_prefill, device);
    const std::string map_key = key.first_node_name + "|" + key.last_node_name + "|" + device;

    {
        std::lock_guard<std::mutex> lock(g_tune_cache.mutex);
        auto it = g_tune_cache.slots.find(map_key);
        if (it != g_tune_cache.slots.end()) {
            slot = it->second;
            return GGML_STATUS_SUCCESS;
        }
    }

    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, false);

    auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);
    auto ggml_decoder  = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, false, false,
                                                         is_model_splitted(cgraph), is_prefill);
    auto input_model   = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder);
    auto model         = ov::frontend::ggml::FrontEnd::convert(input_model);
    ggml_decoder->clear_model_weights();

    ov::CompiledModel compiled_model;
    auto remote_context = ggml_openvino_get_remote_context();
    const bool compile_via_remote =
        remote_context.has_value() && ggml_openvino_device_is_gpu(device) && ggml_openvino_phase_split_enabled();
    if (compile_via_remote) {
        compiled_model = core.compile_model(model, remote_context.value(), config);
    } else {
        compiled_model = core.compile_model(model, device, config);
    }

    tune_slot fresh;
    fresh.decoder = ggml_decoder;
    fresh.infer   = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
    for (const auto & ov_param : model->get_parameters()) {
        fresh.input_names.push_back(ov_param->get_friendly_name());
    }
    for (const auto & ov_output : model->get_results()) {
        fresh.output_names.push_back(ov_output->get_friendly_name());
    }

    std::lock_guard<std::mutex> lock(g_tune_cache.mutex);
    g_tune_cache.slots[map_key] = fresh;
    slot                        = fresh;
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status bind_and_infer(tune_slot & slot, ggml_cgraph * cgraph, float * out_ms) {
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
        infer.set_input_tensor(i, get_ov_input_tensor(slot.decoder, param_name));
    }

    for (size_t i = 0; i < slot.output_names.size(); ++i) {
        auto * ggml_tensor = slot.decoder->get_model_outputs().at(slot.output_names[i]);
        if (ggml_nbytes(ggml_tensor) == 0) {
            continue;
        }
        infer.set_output_tensor(i, create_ov_output_tensor(slot.decoder, slot.infer, i, ggml_tensor));
    }

    const int64_t t0 = ggml_time_us();
    infer.infer();
    if (out_ms) {
        *out_ms = (ggml_time_us() - t0) / 1000.f;
    }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status timed_infer_on_device(ggml_cgraph *       cgraph,
                                              const graph_key &   base_key,
                                              bool                is_prefill,
                                              const std::string & device,
                                              float *             out_ms) {
    auto &       core   = ov_singleton_core();
    const auto & config = ggml_openvino_get_compile_config();
    tune_slot    slot;
    if (ensure_tune_slot(cgraph, base_key, is_prefill, device, config, core, slot) != GGML_STATUS_SUCCESS) {
        return GGML_STATUS_FAILED;
    }
    if (bind_and_infer(slot, cgraph, out_ms) != GGML_STATUS_SUCCESS) {
        return GGML_STATUS_FAILED;
    }
    if (ggml_openvino_device_is_gpu(device)) {
        if (cl_command_queue queue = ggml_openvino_get_cl_queue()) {
            clFinish(queue);
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void record_samples(tune_phase phase, int base_token, int n_tokens, int device_idx, float infer_ms) {
    if (n_tokens <= 0) {
        return;
    }
    const double per_token = infer_ms / static_cast<double>(n_tokens);
    std::lock_guard<std::mutex> lock(g_tune_mu);
    for (int i = 0; i < n_tokens; ++i) {
        const int idx = base_token + i;
        tune_sample * dst = nullptr;
        if (phase == tune_phase::prefill) {
            dst = (device_idx == 0) ? &g_pp_dev0[idx] : &g_pp_dev1[idx];
        } else {
            dst = (device_idx == 0) ? &g_tg_dev0[idx] : &g_tg_dev1[idx];
        }
        dst->sum_ms += per_token;
        dst->count += 1;
    }
}

static std::string device_slug(const std::string & device) {
    std::string s = device;
    for (char & c : s) {
        if (c == '.') {
            c = '_';
        }
    }
    return s;
}

static void write_phase_csv(const char *               phase_name,
                            const std::string &        device,
                            const std::unordered_map<int, tune_sample> & data,
                            const std::string &        out_dir) {
    const std::string path = out_dir + "/" + phase_name + "_" + device_slug(device) + ".csv";
    FILE *            f    = fopen(path.c_str(), "w");
    if (!f) {
        GGML_LOG_WARN("OpenVINO phase tune: could not write %s\n", path.c_str());
        return;
    }
    fprintf(f, "token_index,avg_ms,sample_count\n");
    std::vector<int> keys;
    keys.reserve(data.size());
    for (const auto & kv : data) {
        keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());
    for (int k : keys) {
        const tune_sample & s = data.at(k);
        const double avg      = s.count > 0 ? s.sum_ms / s.count : 0.0;
        fprintf(f, "%d,%.6f,%d\n", k, avg, s.count);
    }
    fclose(f);
    fprintf(stderr, "OpenVINO phase tune: wrote %s\n", path.c_str());
}

static void write_all_tune_csv(const std::string &                        out_dir,
                               const std::string &                        d0,
                               const std::string &                        d1,
                               const std::unordered_map<int, tune_sample> & pp0,
                               const std::unordered_map<int, tune_sample> & pp1,
                               const std::unordered_map<int, tune_sample> & tg0,
                               const std::unordered_map<int, tune_sample> & tg1) {
    write_phase_csv("pp", d0, pp0, out_dir);
    write_phase_csv("pp", d1, pp1, out_dir);
    write_phase_csv("tg", d0, tg0, out_dir);
    write_phase_csv("tg", d1, tg1, out_dir);
}

static void dump_tune_csv_files() {
    if (!ggml_openvino_phase_tune_enabled()) {
        return;
    }
    const std::string out_dir = ggml_openvino_get_phase_tune_output_dir();
    const std::string d0      = ggml_openvino_get_phase_tune_device(0);
    const std::string d1      = ggml_openvino_get_phase_tune_device(1);
    const int         pass    = ggml_openvino_phase_tune_pass();

    std::unordered_map<int, tune_sample> pp0, pp1, tg0, tg1;
    {
        std::lock_guard<std::mutex> lock(g_tune_mu);
        pp0 = g_pp_dev0;
        pp1 = g_pp_dev1;
        tg0 = g_tg_dev0;
        tg1 = g_tg_dev1;
    }

    if (pass < 0) {
        write_all_tune_csv(out_dir, d0, d1, pp0, pp1, tg0, tg1);
        return;
    }
    if (pass == 0) {
        write_phase_csv("pp", d0, pp0, out_dir);
        write_phase_csv("tg", d0, tg0, out_dir);
        return;
    }
    write_phase_csv("pp", d1, pp1, out_dir);
    write_phase_csv("tg", d1, tg1, out_dir);
}

static void register_tune_atexit_once() {
    if (g_atexit_registered) {
        return;
    }
    g_atexit_registered = true;
    std::atexit(dump_tune_csv_files);
}

}  // namespace

bool ggml_openvino_phase_tune_in_production() {
    return g_tune_production;
}

enum ggml_status ov_graph_compute_phase_tune(ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx) {
    GGML_UNUSED(r_ctx);

    register_tune_atexit_once();

    const auto * inp_pos = get_inp_pos_tensor(cgraph);
    const bool   is_prefill = get_is_prefill(inp_pos);
    const int    n_tokens_in_graph = inp_pos ? static_cast<int>(inp_pos->ne[0]) : 1;

    if (is_prefill && inp_pos != nullptr) {
        const int32_t * pos_data = static_cast<const int32_t *>(inp_pos->data);
        if (pos_data[0] == 0) {
            g_pp_token_cursor = 0;
            g_tg_token_cursor = 0;
            std::lock_guard<std::mutex> lock(g_tune_mu);
            g_pp_dev0.clear();
            g_pp_dev1.clear();
            g_tg_dev0.clear();
            g_tg_dev1.clear();
        }
    }

    graph_key base_key(cgraph);
    base_key.last_node_name += is_prefill ? ":ov_pp" : ":ov_tg";

    const std::string dev0 = ggml_openvino_get_phase_tune_device(0);
    const std::string dev1 = ggml_openvino_get_phase_tune_device(1);
    const int         pass = ggml_openvino_phase_tune_pass();
    const int         base_token = is_prefill ? g_pp_token_cursor : g_tg_token_cursor;

    if (pass >= 0) {
        const std::string & device = (pass == 0) ? dev0 : dev1;
        float               ms     = 0.f;
        if (timed_infer_on_device(cgraph, base_key, is_prefill, device, &ms) != GGML_STATUS_SUCCESS) {
            return GGML_STATUS_FAILED;
        }
        record_samples(is_prefill ? tune_phase::prefill : tune_phase::decode, base_token, n_tokens_in_graph, pass, ms);
    } else {
        const auto cache_tensors = collect_cache_tensors(cgraph);
        std::vector<kv_backup_entry> kv_saved;
        backup_kv(cache_tensors, kv_saved);

        for (int di = 0; di < 2; ++di) {
            restore_kv(cache_tensors, kv_saved);
            const std::string & device = (di == 0) ? dev0 : dev1;
            float               ms     = 0.f;
            if (timed_infer_on_device(cgraph, base_key, is_prefill, device, &ms) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            record_samples(is_prefill ? tune_phase::prefill : tune_phase::decode, base_token, n_tokens_in_graph, di,
                           ms);
        }

        restore_kv(cache_tensors, kv_saved);
    }

    g_tune_production = true;
    const enum ggml_status st = ov_graph_compute_dynamic(cgraph, r_ctx);
    g_tune_production         = false;

    if (st == GGML_STATUS_SUCCESS) {
        if (is_prefill) {
            g_pp_token_cursor += n_tokens_in_graph;
        } else {
            const int iv = tune_progress_interval();
            if (iv > 0 && (g_tg_token_cursor % iv) == 0) {
                fprintf(stderr, "OpenVINO phase tune: decode token %d\n", g_tg_token_cursor);
            }
            g_tg_token_cursor += n_tokens_in_graph;
        }
    }

    return st;
}
