// Linux-only, test-side interposition. It does not change server or kernel admission.
#include "llama.h"
#include "../src/llama-ext.h"
#include "../src/llama-batch.h"
#include "../src/llama-kv-cache-iswa.h"
#include "../src/llama-memory-hybrid.h"
#include "../common/ngram-map.h"
#include "../common/speculative.h"
#include "ggml-backend.h"
#include "../vendor/hash/hash.h"
#include "../vendor/nlohmann/json.hpp"
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

using json = nlohmann::ordered_json;

namespace {
struct trace_context {
    std::string role;
    ggml_backend_sched_eval_callback original = nullptr;
    size_t call = 0;
    size_t rows = 0;
    std::string kind;
    bool injected = false;
    std::map<std::string, size_t> captured;
};

std::map<llama_context *, std::unique_ptr<trace_context>> contexts;
std::mutex trace_mutex;
std::map<const common_ngram_map *, size_t> ngram_calls;
std::map<const common_ngram_map *, std::vector<size_t>> ngram_cursors;
thread_local bool nested_context_init = false;
size_t payload_bytes = 0;
size_t payload_id = 0;
const size_t payload_limit = 1024*1024*1024;
const size_t payload_file_limit = 128*1024*1024;

uint64_t clock_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string ngram_outcome(const common_ngram_map & map, const llama_tokens & history,
        llama_token sampled, const llama_tokens & draft) {
    if (!draft.empty()) return "supplied";
    const size_t n = map.size_key;
    if (history.size() < 2*n+map.size_value) return "short_history";
    for (const auto & key : map.keys) {
        if (key.key_idx > history.size() || n > history.size()-key.key_idx) {
            throw std::runtime_error("invalid traced ngram key index");
        }
        bool matches = true;
        for (size_t i = 0; i < n; ++i) {
            const auto token = i+1 == n ? sampled : history[history.size()-n+1+i];
            if (history[key.key_idx+i] != token) { matches = false; break; }
        }
        if (!matches) continue;
        if (key.key_num < map.min_hits) return "minimum_hits";
        uint32_t total = 0;
        size_t best = 0;
        for (size_t i = 0; i < COMMON_NGRAM_MAX_VALUES; ++i) {
            total += key.values[i].value_num;
            if (key.values[i].value_num > key.values[best].value_num) best = i;
        }
        const uint32_t maximum = key.values[best].value_num;
        if (total > maximum && maximum < 2*(total-maximum)) return "ambiguous";
        if (key.values[best].n_accepted <= 0) return "acceptance_suppressed";
        return "empty";
    }
    return "no_match";
}

json ngram_memory(const common_ngram_map & map, const llama_tokens & history) {
    return {{"keys_size", map.keys.size()}, {"keys_capacity", map.keys.capacity()},
            {"map_reserved_bytes", map.key_map.capacity()*sizeof(uint32_t)+map.keys.capacity()*sizeof(common_ngram_map_key)},
            {"history_size", history.size()}, {"history_capacity_bytes", history.capacity()*sizeof(llama_token)},
            {"hash_indexed_through", map.key_map_last_idx}};
}

bool ngram_statistics_started(const common_ngram_map & map, std::vector<size_t> & seen) {
    seen.resize(map.keys.size(), 0);
    bool started = false;
    for (size_t i = 0; i < map.keys.size(); ++i) {
        started |= seen[i] == 0 && map.keys[i].stat_idx > 0;
        seen[i] = map.keys[i].stat_idx;
    }
    return started;
}

bool payload_fits(size_t bytes) {
    return bytes && bytes <= payload_file_limit && payload_bytes <= payload_limit && bytes <= payload_limit-payload_bytes;
}

bool capture_enabled() {
    const char * value = std::getenv("DFLASH_CONTROL_CAPTURE");
    return value && std::strcmp(value, "1") == 0;
}

std::filesystem::path trace_directory() {
    const char * value = std::getenv("DFLASH_CONTROL_TRACE");
    if (!value || !*value) throw std::runtime_error("DFLASH_CONTROL_TRACE is required");
    return value;
}

// A trace failure must not masquerade as a successful native control.
void record(json row) {
    std::lock_guard<std::mutex> lock(trace_mutex);
    std::ofstream stream(trace_directory()/"trace.jsonl", std::ios::app);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << row.dump() << '\n';
}

json payload(const void * data, size_t bytes) {
    if (!data || !payload_fits(bytes)) {
        throw std::runtime_error("invalid or exhausted trace payload budget: bytes="+std::to_string(bytes));
    }
    const std::string name = "payload-"+std::to_string(++payload_id)+".bin";
    std::ofstream stream(trace_directory()/name, std::ios::binary);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream.write(static_cast<const char *>(data), bytes);
    payload_bytes += bytes;
    return {{"file", name}, {"bytes", bytes}, {"sha256", hash_sha256_hex(data, bytes)}};
}

json shape(const ggml_tensor * tensor) {
    return {{"name", tensor->name}, {"type", ggml_type_name(tensor->type)},
            {"ne", {tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]}},
            {"nb", {tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]}}};
}

json save_tensor(ggml_tensor * tensor) {
    const size_t bytes = ggml_nbytes(tensor);
    if (!payload_fits(bytes)) throw std::runtime_error("trace tensor exceeds remaining payload bounds");
    std::vector<uint8_t> data(bytes);
    ggml_backend_tensor_get(tensor, data.data(), 0, bytes);
    json row = shape(tensor);
    row.update(payload(data.data(), bytes));
    return row;
}

// Physical mask columns are retained, so window visibility is not inferred from capacity.
json save_attention(ggml_tensor * tensor, size_t rows) {
    auto * mask = tensor->src[3];
    if (!mask || (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) ||
            mask->ne[1] < int64_t(rows) || mask->ne[2] != 1 || mask->ne[3] != 1) {
        throw std::runtime_error("unexpected serial attention mask");
    }
    auto result = save_tensor(mask);
    std::vector<uint8_t> data(ggml_nbytes(mask));
    ggml_backend_tensor_get(mask, data.data(), 0, data.size());
    json visible = json::array();
    json columns = json::array();
    for (size_t q = 0; q < rows; ++q) {
        json indices = json::array();
        for (int64_t k = 0; k < mask->ne[0]; ++k) {
            const size_t offset = q*mask->nb[1]+size_t(k)*mask->nb[0];
            float value;
            if (mask->type == GGML_TYPE_F16) {
                ggml_fp16_t encoded;
                std::memcpy(&encoded, data.data()+offset, sizeof(encoded));
                value = ggml_fp16_to_fp32(encoded);
            } else {
                std::memcpy(&value, data.data()+offset, sizeof(value));
            }
            if (std::isnan(value) || value == INFINITY) throw std::runtime_error("invalid mask value");
            if (std::isfinite(value)) indices.push_back(k);
        }
        visible.push_back(indices.size());
        columns.push_back(std::move(indices));
    }
    return {{"query_rows", rows}, {"visible_rows", visible}, {"visible_columns", columns},
            {"mask_file", result["file"]}, {"mask", result},
            {"q", shape(tensor->src[0])}, {"k", save_tensor(tensor->src[1])},
            {"v", save_tensor(tensor->src[2])}};
}

bool evaluate(ggml_tensor * tensor, bool ask, void * opaque) {
    try {
        auto & state = *static_cast<trace_context *>(opaque);
        if (state.original) throw std::runtime_error("trace cannot replace an existing eval callback");
        if (!state.rows) return false;
        const bool attention = state.role == "draft" && state.kind == "tokens" && tensor->op == GGML_OP_FLASH_ATTN_EXT;
        const bool logits = std::strcmp(tensor->name, "result_output") == 0 && ggml_nelements(tensor) > 0;
        const std::string key = state.kind+":"+tensor->name;
        const bool selected = (attention || logits) && state.captured[key] < 2;
        if (ask) return selected;
        if (!selected) return true;
        ++state.captured[key];
        json row = attention ? save_attention(tensor, state.rows) : save_tensor(tensor);
        row.update({{"event", attention ? "attention" : "tensor"}, {"role", state.role},
                    {"call", state.call}, {"kind", state.kind}, {"rows", state.rows},
                    {"warmup", state.role == "draft" && !state.injected}});
        record(std::move(row));
        return true;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "DFlash control trace failed: %s\n", error.what());
        std::abort();
    }
}

void memory(llama_context * ctx, const trace_context & state) {
    json buffers = json::array();
    for (const auto & [type, bytes] : llama_get_memory_breakdown(ctx)) {
        buffers.push_back({{"buffer", ggml_backend_buft_name(type)}, {"model", bytes.model},
                           {"context", bytes.context}, {"compute", bytes.compute}});
    }
    json devices = json::array();
    for (int i = 0; i < llama_model_n_devices(llama_get_model(ctx)); ++i) {
        auto * device = llama_model_get_device(llama_get_model(ctx), i);
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        devices.push_back({{"name", ggml_backend_dev_name(device)}, {"free", free}, {"total", total}});
    }
    json components = json::array();
    auto component = [&](const char * name, llama_memory_i * memory) {
        for (const auto & [type, bytes] : memory->memory_breakdown()) {
            components.push_back({{"name", name}, {"buffer", ggml_backend_buft_name(type)}, {"bytes", bytes}});
        }
    };
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx))) {
        component("target_kv", hybrid->get_mem_attn());
        component("target_recurrent", hybrid->get_mem_recr());
    }
    record({{"event", "memory"}, {"role", state.role}, {"call", state.call},
            {"buffers", buffers}, {"devices", devices}, {"components", components},
            {"n_rs_seq", llama_n_rs_seq(ctx)}});
    if (state.role == "draft" && capture_enabled()) {
        auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx));
        if (!iswa) throw std::runtime_error("ordinary DFlash control has no iSWA cache");
        auto * swa = iswa->get_swa();
        json positions = json::array();
        for (const auto & [pos, cell] : swa->get_cells(0).seq_pos_get(0)) {
            positions.push_back({pos, cell});
        }
        record({{"event", "cache"}, {"role", state.role}, {"call", state.call},
                {"capacity", swa->get_size()}, {"layers", swa->get_layer_ids()},
                {"base_layers", iswa->get_base()->get_layer_ids()}, {"positions", positions}});
    }
}

template<typename T> T next(const char * name) {
    auto symbol = dlsym(RTLD_NEXT, name);
    if (!symbol) throw std::runtime_error(std::string("missing traced symbol: ")+name);
    return reinterpret_cast<T>(symbol);
}
}

namespace {
llama_context * traced_init(llama_model * model, llama_context_params params, uint32_t prediction_rows, bool extended) {
    try {
        auto state = std::make_unique<trace_context>();
        char arch[64] = {};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        state->role = std::strcmp(arch, "dflash") == 0 ? "draft" : "target";
        state->original = params.cb_eval;
        if (capture_enabled()) {
            if (params.cb_eval) throw std::runtime_error("trace requires no existing eval callback");
            params.cb_eval = evaluate;
            params.cb_eval_user_data = state.get();
        }
        using ordinary_init = llama_context * (*)(llama_model *, llama_context_params);
        using extended_init = llama_context * (*)(llama_model *, llama_context_params, uint32_t);
        nested_context_init = true;
        auto * ctx = extended ? next<extended_init>("_Z21llama_init_from_modelP11llama_model20llama_context_paramsj")(model, params, prediction_rows) :
            next<ordinary_init>("llama_init_from_model")(model, params);
        nested_context_init = false;
        record({{"event", "context"}, {"role", state->role}, {"ok", ctx != nullptr},
                {"n_ctx", params.n_ctx}, {"n_ubatch", params.n_ubatch}, {"n_rs_seq", params.n_rs_seq},
                {"serial_prediction_rows", prediction_rows},
                {"type_k", ggml_type_name(params.type_k)}, {"type_v", ggml_type_name(params.type_v)}});
        if (ctx) {
            memory(ctx, *state);
            contexts[ctx] = std::move(state);
        }
        return ctx;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "DFlash control trace init failed: %s\n", error.what());
        std::abort();
    }
}
}

extern "C" llama_context * llama_init_from_model(llama_model * model, llama_context_params params) {
    return traced_init(model, params, 0, false);
}

llama_context * llama_init_from_model(llama_model * model, llama_context_params params, uint32_t prediction_rows) {
    if (nested_context_init) {
        using fn = llama_context * (*)(llama_model *, llama_context_params, uint32_t);
        return next<fn>("_Z21llama_init_from_modelP11llama_model20llama_context_paramsj")(model, params, prediction_rows);
    }
    return traced_init(model, params, prediction_rows, true);
}

extern "C" int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    try {
        auto found = contexts.find(ctx);
        if (found == contexts.end()) throw std::runtime_error("untracked DFlash control context");
        auto & state = *found->second;
        state.rows = batch->tokens.size();
        state.kind = batch->n_embd ? "features" : "tokens";
        if (state.kind == "features") state.injected = true;
        ++state.call;
        const std::string label = state.role+"/"+state.kind+"/"+std::to_string(state.rows);
        nvtxRangePushA(label.c_str());
        json row = {{"event", "process"}, {"role", state.role}, {"call", state.call},
                    {"kind", state.kind}, {"rows", state.rows}, {"feature_width", batch->n_embd},
                    {"warmup", state.role == "draft" && !state.injected}};
        json positions = json::array(), tokens = json::array();
        for (const auto & token : batch->tokens) {
            positions.push_back(token.pos[0]);
            tokens.push_back(token.id);
        }
        row["positions"] = positions;
        row["tokens"] = tokens;
        if (capture_enabled() && batch->n_embd && state.captured["features"]++ < 2) {
            row["features"] = payload(batch->embd.data(), batch->embd.size()*sizeof(float));
        }
        const auto started = clock_ns();
        const int ret = next<decltype(&llama_process)>("llama_process")(ctx, type, batch);
        row["host_wall_ns"] = clock_ns()-started;
        row["ret"] = ret;
        if (capture_enabled() && state.role == "draft" && !batch->n_embd && ret == 0 && state.captured["lattice"]++ < 2) {
            const size_t width = llama_model_n_embd(llama_get_model(ctx));
            json lattice = payload(llama_get_embeddings_nextn(ctx), state.rows*width*sizeof(float));
            lattice.update({{"event", "tensor"}, {"role", "draft"}, {"name", "lattice"},
                            {"call", state.call}, {"rows", state.rows}, {"width", width},
                            {"warmup", !state.injected}});
            record(std::move(lattice));
        }
        record(std::move(row));
        memory(ctx, state);
        nvtxRangePop();
        state.rows = 0;
        return ret;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "DFlash control trace process failed: %s\n", error.what());
        std::abort();
    }
}

extern "C" void llama_free(llama_context * ctx) {
    next<decltype(&llama_free)>("llama_free")(ctx);
    contexts.erase(ctx);
}

void common_ngram_map_begin(common_ngram_map & map, const llama_tokens & history) {
    const auto fn = next<decltype(&common_ngram_map_begin)>("_Z22common_ngram_map_beginR16common_ngram_mapRKSt6vectorIiSaIiEE");
    const auto started = clock_ns();
    fn(map, history);
    const auto elapsed = clock_ns()-started;
    ngram_calls[&map] = 0;
    ngram_cursors[&map].clear();
    auto row = ngram_memory(map, history);
    row.update({{"event", "ngram_begin"}, {"host_wall_ns", elapsed}});
    record(std::move(row));
}

void common_ngram_map_draft(common_ngram_map & map, const llama_tokens & history,
        llama_token sampled, llama_tokens & draft) {
    const auto fn = next<decltype(&common_ngram_map_draft)>("_Z22common_ngram_map_draftR16common_ngram_mapRKSt6vectorIiSaIiEEiRS3_");
    const size_t indexed_before = map.key_map_last_idx;
    const auto started = clock_ns();
    fn(map, history, sampled, draft);
    const auto elapsed = clock_ns()-started;
    auto row = ngram_memory(map, history);
    row.update({{"event", "ngram_draft"}, {"host_wall_ns", elapsed}, {"first_use", ngram_calls[&map]++ == 0},
            {"first_statistics", ngram_statistics_started(map, ngram_cursors[&map])}, {"indexed_before", indexed_before},
            {"draft_tokens", draft.size()}, {"outcome", ngram_outcome(map, history, sampled, draft)}});
    record(std::move(row));
}

void common_speculative_draft(common_speculative * spec) {
    const auto fn = next<decltype(&common_speculative_draft)>("_Z24common_speculative_draftP18common_speculative");
    record({{"event", "draft_driver_begin"}});
    const auto started = clock_ns();
    fn(spec);
    const auto elapsed = clock_ns()-started;
    json sources = json::array();
    for (const auto & row : common_speculative_get_stats(spec)) {
        sources.push_back({{"type", common_speculative_type_to_str(row.type)}, {"draft_calls", row.n_call_draft},
                {"generated_drafts", row.n_gen_drafts}, {"generated_tokens", row.n_gen_tokens},
                {"accepted_tokens", row.n_acc_tokens}, {"prediction_forwards", row.n_predict}});
    }
    record({{"event", "draft_driver"}, {"host_wall_ns", elapsed}, {"sources", sources}});
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq, const llama_tokens & history) {
    const auto fn = next<decltype(&common_speculative_begin)>("_Z24common_speculative_beginP18common_speculativeiRKSt6vectorIiSaIiEE");
    fn(spec, seq, history);
    record({{"event", "generation_begin"}, {"sequence", seq}, {"prompt_tokens", history.size()}});
}

#ifdef DFLASH_CONTROL_TRACE_TEST
#include "../tests/testing.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

int main() {
    testing t;
    t.test("ngram_trace_distinguishes_empty_reasons_without_mutating_history", [&](testing & t) {
        common_ngram_map map(2, 2, false, 2);
        const llama_tokens history{91, 1, 2, 3, 4, 1, 2, 5, 6, 1};
        const auto saved = history;
        t.assert_true(ngram_outcome(map, history, 2, {}) == "no_match");
        common_ngram_map_key key{};
        key.key_idx = 1;
        key.key_num = 1;
        key.values[0] = {3, 2, 2};
        key.values[1] = {7, 2, 2};
        map.keys.push_back(key);
        t.assert_true(ngram_outcome(map, history, 2, {}) == "minimum_hits");
        map.keys[0].key_num = 2;
        t.assert_true(ngram_outcome(map, history, 2, {}) == "ambiguous");
        map.keys[0].values[0] = {3, 4, 0};
        t.assert_true(ngram_outcome(map, history, 2, {}) == "acceptance_suppressed");
        std::vector<size_t> seen;
        t.assert_true(!ngram_statistics_started(map, seen));
        map.keys[0].stat_idx = 5;
        t.assert_true(ngram_statistics_started(map, seen));
        t.assert_true(!ngram_statistics_started(map, seen));
        map.keys.push_back(key);
        t.assert_true(!ngram_statistics_started(map, seen));
        map.keys[1].stat_idx = 5;
        t.assert_true(ngram_statistics_started(map, seen));
        t.assert_true(ngram_outcome(map, history, 2, {3, 4}) == "supplied");
        t.assert_true(ngram_outcome(map, {1}, 2, {}) == "short_history");
        t.assert_true(history == saved);
        map.keys[0].key_idx = history.size();
        bool rejected = false;
        try { ngram_outcome(map, history, 2, {}); } catch (const std::runtime_error &) { rejected = true; }
        t.assert_true(rejected);
    });
    auto * ctx = ggml_init({65536, nullptr, true});
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 2);
    auto * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, 8, 8);
    auto * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, 8, 8);
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 2, 32);
    auto * empty = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 248320, 0);
    ggml_set_name(empty, "result_output");
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_clear(buffer, 0);
    ggml_tensor attention{};
    attention.src[0] = q;
    attention.src[1] = k;
    attention.src[2] = v;
    attention.src[3] = mask;
    std::array<ggml_fp16_t, 16> values;
    std::fill(values.begin(), values.end(), ggml_fp32_to_fp16(-INFINITY));
    values[0] = values[7] = values[8] = values[10] = values[15] = ggml_fp32_to_fp16(0);
    ggml_backend_tensor_set(mask, values.data(), 0, sizeof(values));
    t.test("trace_preserves_mask_visibility_and_bounded_payloads", [&](testing & t) {
        const auto row = save_attention(&attention, 2);
        t.assert_equal(size_t(2), row["visible_rows"][0].get<size_t>());
        t.assert_equal(size_t(3), row["visible_rows"][1].get<size_t>());
        t.assert_true(row["visible_columns"][0] == json::array({0, 7}));
        t.assert_true(row["visible_columns"][1] == json::array({0, 2, 7}));
        trace_context state;
        state.role = "target";
        state.kind = "tokens";
        state.rows = 2;
        t.assert_true(!evaluate(empty, true, &state));
        t.assert_true(payload_fits(100*1024*1024));
        t.assert_true(payload_fits(payload_file_limit));
        for (size_t bytes : {size_t(0), payload_file_limit+1}) {
            bool rejected = false;
            try { payload(values.data(), bytes); } catch (const std::runtime_error &) { rejected = true; }
            t.assert_true(rejected);
        }
        const auto saved = payload_bytes;
        payload_bytes = payload_limit;
        bool rejected = false;
        try { payload(values.data(), sizeof(values)); } catch (const std::runtime_error &) { rejected = true; }
        t.assert_true(rejected);
        payload_bytes = saved;
        values[0] = ggml_fp32_to_fp16(NAN);
        ggml_backend_tensor_set(mask, values.data(), 0, sizeof(values));
        rejected = false;
        try { save_attention(&attention, 2); } catch (const std::runtime_error &) { rejected = true; }
        t.assert_true(rejected);
    });
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return t.summary();
}
#endif
