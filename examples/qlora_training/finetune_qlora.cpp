// QLoRA fine-tuning for quantized GGUF models.
//
// The base model weights stay frozen (quantized tensors are skipped by
// llama_set_param because they are not GGML_TYPE_F32).  Only the freshly
// allocated F32 LoRA A/B tensors are trained.  After training the adapter
// is saved as a GGUF file that is directly compatible with the existing
// llama_adapter_lora_init() loader and llama-export-lora merge tool.
//
// Usage example:
/*   llama-finetune-qlora \
         --model model-q4_k_m.gguf \
         --train-file train.jsonl \
         --lora-rank 16 --lora-alpha 16 \
         --lr-scheduler cosine --warmup-steps 10 --warmup-init-ratio 0.1 \
         --lora-out adapter.gguf \
         --epochs 3 -c 4096 -b 4096 -ub 512
*/
// Default targets: attn_q, attn_output, ffn_gate, ffn_up, ffn_down
// Override with --lora-targets "comma,separated,substrings"
//
// NOTE: attn_k and attn_v are excluded from defaults.  The KV write path uses
// ggml_set_rows (scatter op) — backward cannot propagate gradients through it.
// LoRA K/V would receive zero gradient.
//
// NOTE: ssm_in and ssm_out (Mamba/NemotronH) are excluded from defaults.
// SSM_SCAN/SSM_CONV have no backward implementation — LoRA on these layers
// would receive zero gradient.  Adding them wastes memory with no benefit.
//
// NOTE: MoE expert tensors (*_exps) are excluded regardless of --lora-targets.
// The quantized expert weights are frozen (stop-gradient), but LoRA on dense
// FFN layers (ffn_gate, ffn_up, ffn_down) works via MUL_MAT_ID backward.
//
// Target substrings use llama.cpp internal GGUF names (NOT HuggingFace names):
//   attn_q      = q_proj       attn_k     = k_proj
//   attn_v      = v_proj       attn_output= o_proj
//   ffn_gate    = gate_proj    ffn_up     = up_proj    ffn_down = down_proj
//   ssm_in      = in_proj (Mamba/NemotronH)  — zero gradient, not in defaults
//   ssm_out     = out_proj (Mamba/NemotronH)  — zero gradient, not in defaults

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "gguf.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "critical-token-sft.h"

// Internal adapter struct — included directly to avoid the temp-GGUF roundtrip
// for wiring trainable LoRA tensors into the compute graph.
#include "../../src/llama-adapter.h"

#include <cerrno>
#include <csignal>
#include <iostream>

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <clocale>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Expand a leading ~/ to the HOME directory (the shell doesn't do this for us
// when a path is passed as a string argument to std::ofstream).
static std::string expand_tilde(const std::string & path) {
    if (path.size() >= 2 && path[0] == '~' && path[1] == '/') {
        const char * home = getenv("HOME");
        if (!home) home = getenv("USERPROFILE"); // Windows fallback
        if (home) return std::string(home) + path.substr(1);
    }
    return path;
}

static std::vector<std::string> split_csv(const std::string & s) {
    std::vector<std::string> out;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(tok);
    }
    return out;
}

static int64_t train_split_from_val_fraction(int64_t ndata, float val_split) {
    if (ndata <= 0) {
        return 0;
    }
    if (val_split <= 0.0f) {
        return ndata;
    }
    if (val_split >= 1.0f) {
        return 0;
    }

    const int64_t split = (int64_t) (ndata * (1.0f - val_split));
    return std::max<int64_t>(1, split);
}

static std::string preview_text(const std::string & s, size_t max_len = 240) {
    std::string out;
    out.reserve(std::min(s.size(), max_len));
    for (char c : s) {
        if (out.size() >= max_len) {
            out += "...";
            break;
        }
        if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else {
            out += c;
        }
    }
    return out;
}

struct qlora_lr_schedule {
    const lr_opt * lr;
    std::string    type;
    int64_t        warmup_steps;
    float          warmup_init_ratio;
    int64_t        total_steps;
    int64_t        step;
    float          current_lr = 0.0f;

    float get_lr() const {
        const float lr_base = lr->lr0;
        if (step < warmup_steps) {
            const float lr_start = lr_base * warmup_init_ratio;
            const float progress = (float) step / (float) warmup_steps;
            return lr_start + (lr_base - lr_start) * progress;
        }
        if (type != "cosine" || total_steps <= warmup_steps) {
            return lr->get_lr();
        }

        const float lr_min = std::max(0.0f, lr->lr_min);
        const double progress = (double) (step - warmup_steps) / (double) (total_steps - warmup_steps);
        const double cosine = 0.5 * (1.0 + std::cos(std::acos(-1.0) * std::min(1.0, progress)));
        return lr_min + (lr_base - lr_min) * (float) cosine;
    }
};

static ggml_opt_optimizer_params qlora_opt_lr_pars(void * userdata) {
    qlora_lr_schedule & schedule = *(qlora_lr_schedule *) userdata;
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(nullptr);
    schedule.current_lr = schedule.get_lr();
    result.adamw.alpha = result.sgd.alpha = schedule.current_lr;
    result.sgd.wd = result.adamw.wd = schedule.lr->wd;
    return result;
}

// Tensors whose names contain these substrings use MUL_MAT_ID (sparse MoE expert dispatch)
// which has no backward implementation — exclude them from LoRA targets unconditionally.
static const std::vector<std::string> EXCLUDED_SUBSTRINGS = {
    "_exps",      // MoE expert weight stacks (ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_gate_up_exps)
};

static bool tensor_is_excluded(const char * name) {
    const std::string n(name);
    for (const auto & ex : EXCLUDED_SUBSTRINGS) {
        if (n.find(ex) != std::string::npos) return true;
    }
    return false;
}

// Extract the transformer block index from a tensor name of the form "blk.NN.<rest>".
// Returns -1 if the name does not follow this pattern.
static int tensor_layer_index(const char * name) {
    // All per-layer tensors in llama.cpp GGUF are named "blk.<N>.<suffix>"
    const char * p = strstr(name, "blk.");
    if (!p) return -1;
    p += 4; // skip "blk."
    char * end = nullptr;
    long idx = strtol(p, &end, 10);
    if (end == p || (*end != '.' && *end != '\0')) return -1;
    return (int) idx;
}

static bool tensor_matches_targets(const char * name, const std::vector<std::string> & targets,
                                   int freeze_layers = 0) {
    if (tensor_is_excluded(name)) return false;
    if (freeze_layers > 0) {
        const int layer = tensor_layer_index(name);
        if (layer >= 0 && layer < freeze_layers) return false;
    }
    for (const auto & t : targets) {
        if (std::string(name).find(t) != std::string::npos) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// JSONL dataset loading
// ---------------------------------------------------------------------------

struct training_sample {
    std::vector<llama_token> tokens;   // full token sequence
    std::vector<bool>        is_label; // true for tokens that contribute to loss
    std::vector<float>       critical_weights;
    float                    reward;   // reward/score weight (1.0 = neutral, 0.0 = ignore)
};

struct tokenization_request {
    std::string prompt;
    std::string response;
    std::string annotated_response;
    std::vector<critical_span> critical_spans;
    float reward = 1.0f;
};

struct jsonl_input_line {
    std::string text;
    int lineno;
};

struct jsonl_load_result {
    tokenization_request request;
    training_sample      sample;
    bool                 parsed = false;
    bool                 valid = false;
};

struct jsonl_line_state {
    nlohmann::json data;
    std::vector<const nlohmann::json *> message_data;
    std::vector<common_chat_msg> messages;
    std::vector<common_chat_msg> prompt_messages;
    std::vector<std::string> chatml_messages;
    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> response_tokens;
    std::string last_assistant_content;
    std::string prompt_text;
    std::string full_text;
    std::string response_text;
    size_t last_assistant_index = std::numeric_limits<size_t>::max();
    float reward = 1.0f;
    int lineno = 0;
    bool json_valid = false;
    bool is_chat = false;
    bool parallel_messages = false;
    bool prepared = false;
    std::atomic<bool> failed { false };
};

class dataset_worker_pool {
public:
    explicit dataset_worker_pool(size_t n_workers) {
        if (n_workers <= 1) {
            return;
        }

        workers.reserve(n_workers);
        for (size_t i = 0; i < n_workers; ++i) {
            workers.emplace_back([this, i] { worker_loop(i); });
        }
    }

    ~dataset_worker_pool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        work_ready.notify_all();
        for (std::thread & worker : workers) {
            worker.join();
        }
    }

    void parallel_for(size_t n_tasks, const std::function<void(size_t, size_t)> & fn) {
        if (n_tasks == 0) {
            return;
        }
        if (workers.empty()) {
            for (size_t i = 0; i < n_tasks; ++i) {
                fn(i, 0);
            }
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            task = fn;
            task_count = n_tasks;
            next_task.store(0);
            workers_pending = workers.size();
            exception = nullptr;
            ++generation;
        }
        work_ready.notify_all();

        std::unique_lock<std::mutex> lock(mutex);
        work_done.wait(lock, [this] { return workers_pending == 0; });
        if (exception) {
            std::rethrow_exception(exception);
        }
    }

private:
    void worker_loop(size_t worker_index) {
        size_t worker_generation = 0;
        for (;;) {
            std::function<void(size_t, size_t)> current_task;
            size_t current_task_count = 0;
            {
                std::unique_lock<std::mutex> lock(mutex);
                work_ready.wait(lock, [&] { return stop || generation != worker_generation; });
                if (stop) {
                    return;
                }
                worker_generation = generation;
                current_task = task;
                current_task_count = task_count;
            }

            for (;;) {
                const size_t i = next_task.fetch_add(1);
                if (i >= current_task_count) {
                    break;
                }
                try {
                    current_task(i, worker_index);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (!exception) {
                        exception = std::current_exception();
                    }
                }
            }

            {
                std::lock_guard<std::mutex> lock(mutex);
                if (--workers_pending == 0) {
                    task = {};
                    work_done.notify_one();
                }
            }
        }
    }

    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable work_ready;
    std::condition_variable work_done;
    std::function<void(size_t, size_t)> task;
    std::atomic<size_t> next_task { 0 };
    size_t task_count = 0;
    size_t workers_pending = 0;
    size_t generation = 0;
    std::exception_ptr exception;
    bool stop = false;
};

static size_t physical_core_count() {
#ifdef __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::set<std::pair<int, int>> cores;
        int package_id = -1;
        int core_id = -1;
        std::string line;

        auto add_core = [&] {
            if (package_id >= 0 && core_id >= 0) {
                cores.insert({ package_id, core_id });
            }
            package_id = -1;
            core_id = -1;
        };

        while (std::getline(cpuinfo, line)) {
            if (line.empty()) {
                add_core();
                continue;
            }

            const size_t colon = line.find(':');
            if (colon == std::string::npos) continue;
            const std::string key = line.substr(0, colon);
            if (key.find("physical id") != std::string::npos) {
                package_id = std::stoi(line.substr(colon + 1));
            } else if (key.find("core id") != std::string::npos) {
                core_id = std::stoi(line.substr(colon + 1));
            }
        }
        add_core();

        if (!cores.empty()) {
            return cores.size();
        }
    }
#endif

    return std::max(1u, std::thread::hardware_concurrency());
}

static size_t dataset_worker_limit(int32_t requested) {
    return requested > 0 ? (size_t) requested : physical_core_count();
}

static size_t dataset_worker_count(int32_t requested, size_t n_tasks) {
    return std::min(n_tasks, dataset_worker_limit(requested));
}

static bool critical_mode_uses_spans(const std::string & mode) {
    return mode == "spans" || mode == "hybrid";
}

static bool response_token_ranges(
        const llama_vocab                   * vocab,
        const std::vector<llama_token>      & tokens,
        const std::string                   & formatted_response,
        const std::string                   & annotated_response,
        std::vector<critical_token_range>   & ranges,
        std::string                         & error) {
    std::vector<critical_token_range> decoded_ranges;
    decoded_ranges.reserve(tokens.size());

    std::string decoded;
    for (llama_token token : tokens) {
        const size_t start = decoded.size();
        decoded += common_token_to_piece(vocab, token, true);
        decoded_ranges.push_back({ start, decoded.size() });
    }

    const std::string detokenized = common_detokenize(vocab, tokens, true);
    if (decoded != detokenized) {
        decoded.clear();
        decoded_ranges.clear();
        std::vector<llama_token> prefix;
        prefix.reserve(tokens.size());
        for (llama_token token : tokens) {
            const size_t start = decoded.size();
            prefix.push_back(token);
            const std::string current = common_detokenize(vocab, prefix, true);
            if (current.rfind(decoded, 0) != 0) {
                error = "token pieces cannot reconstruct response byte ranges";
                return false;
            }
            decoded = current;
            decoded_ranges.push_back({ start, decoded.size() });
        }
    }

    const size_t decoded_offset = decoded.rfind(annotated_response);
    if (formatted_response.rfind(annotated_response) == std::string::npos || decoded_offset == std::string::npos) {
        error = "annotated response text was not preserved by formatting/tokenization";
        return false;
    }

    const size_t annotated_end = decoded_offset + annotated_response.size();
    ranges.resize(decoded_ranges.size());
    for (size_t i = 0; i < decoded_ranges.size(); ++i) {
        const size_t start = std::max(decoded_ranges[i].start, decoded_offset);
        const size_t end = std::min(decoded_ranges[i].end, annotated_end);
        if (start < end) {
            ranges[i] = { start - decoded_offset, end - decoded_offset };
        } else {
            ranges[i] = { 0, 0 };
        }
    }
    return true;
}

static std::string apply_chatml_message(const common_chat_msg & msg) {
    size_t content_size = 0;
    for (const common_chat_msg_content_part & part : msg.content_parts) {
        content_size += part.text.size();
    }

    std::string out;
    out.reserve(13 + msg.role.size() + 1 + content_size + 11);
    out += "<|im_start|>";
    out += msg.role;
    out += '\n';
    for (const common_chat_msg_content_part & part : msg.content_parts) {
        out += part.text;
    }
    out += "<|im_end|>\n";
    return out;
}

static std::string apply_qlora_chat_template(
        common_chat_templates              * tmpls,
        const std::vector<common_chat_msg> & messages,
        bool                                  add_generation_prompt) {
    common_chat_templates_inputs inputs;
    inputs.messages = messages;
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    inputs.enable_thinking = false;
    return common_chat_templates_apply(tmpls, inputs).prompt;
}

struct jsonl_message_task {
    size_t line_index;
    size_t first_message;
    size_t message_count;
};

struct jsonl_render_task {
    size_t line_index;
    bool render_prompt;
};

struct jsonl_token_task {
    size_t line_index;
    int part;
};

static std::vector<training_sample> load_jsonl(
        const std::string & path,
        const llama_model  * model,
        const llama_vocab  * vocab,
        common_chat_templates * tmpls,
        int32_t              n_threads,
        const std::string  & critical_mode,
        float                critical_default_weight) {

    std::ifstream f(path);
    if (!f.is_open()) {
        LOG_ERR("%s: cannot open %s\n", __func__, path.c_str());
        return {};
    }

    std::vector<training_sample> samples;
    std::vector<jsonl_input_line> lines;
    std::string line;
    int lineno = 0;
    bool logged_preview = false;

    while (std::getline(f, line)) {
        ++lineno;
        if (line.empty()) continue;

        lines.push_back({ std::move(line), lineno });
    }

    std::vector<jsonl_load_result> results(lines.size());
    std::vector<std::unique_ptr<jsonl_line_state>> states;
    states.reserve(lines.size());
    for (const jsonl_input_line & input : lines) {
        auto state = std::make_unique<jsonl_line_state>();
        state->lineno = input.lineno;
        states.push_back(std::move(state));
    }

    const size_t worker_limit = dataset_worker_limit(n_threads);
    const size_t n_json_workers = dataset_worker_count(n_threads, lines.size());
    auto parse_json_line = [&](size_t i) {
        jsonl_line_state & state = *states[i];
        try {
            state.data = nlohmann::json::parse(lines[i].text);
            state.json_valid = true;
        } catch (...) {
            LOG_WRN("%s: skipping invalid JSON on line %d\n", __func__, state.lineno);
        }
    };

    if (n_json_workers <= 1) {
        for (size_t i = 0; i < lines.size(); ++i) {
            parse_json_line(i);
        }
    } else {
        std::atomic<size_t> next_line { 0 };
        std::vector<std::thread> workers;
        workers.reserve(n_json_workers);
        for (size_t worker = 0; worker < n_json_workers; ++worker) {
            workers.emplace_back([&] {
                for (;;) {
                    const size_t i = next_line.fetch_add(1);
                    if (i >= lines.size()) {
                        break;
                    }
                    parse_json_line(i);
                }
            });
        }
        for (std::thread & worker : workers) {
            worker.join();
        }
    }

    const size_t message_parallel_min = 8;
    size_t n_valid_json = 0;
    size_t message_task_hint = 0;
    for (std::unique_ptr<jsonl_line_state> & state_ptr : states) {
        jsonl_line_state & state = *state_ptr;
        if (!state.json_valid) {
            continue;
        }
        ++n_valid_json;

        const nlohmann::json & data = state.data;
        if (!data.contains("messages")) {
            continue;
        }

        state.is_chat = true;
        const nlohmann::json & messages = data.at("messages");
        state.message_data.reserve(messages.size());
        for (const nlohmann::json & message : messages) {
            state.message_data.push_back(&message);
        }
        state.messages.resize(state.message_data.size());
        if (!tmpls) {
            state.chatml_messages.resize(state.message_data.size());
        }
        state.parallel_messages = worker_limit > 1 && state.message_data.size() >= message_parallel_min;
        message_task_hint += state.parallel_messages ? state.message_data.size() : 1;
    }

    const size_t n_workers = std::min(worker_limit, std::max(n_valid_json, message_task_hint));
    std::vector<common_chat_templates_ptr> template_copies;
    std::vector<common_chat_templates *> worker_templates(n_workers, tmpls);
    if (tmpls && n_workers > 1) {
        GGML_ASSERT(model != nullptr);
        template_copies.reserve(n_workers - 1);
        for (size_t i = 1; i < n_workers; ++i) {
            template_copies.push_back(common_chat_templates_init(model, ""));
            worker_templates[i] = template_copies.back().get();
        }
    }

    {
        dataset_worker_pool worker_pool(n_workers);
        std::vector<jsonl_message_task> message_tasks;
        message_tasks.reserve(message_task_hint);
        for (size_t i = 0; i < states.size(); ++i) {
            const jsonl_line_state & state = *states[i];
            if (!state.json_valid || !state.is_chat || state.message_data.empty()) {
                continue;
            }
            if (state.parallel_messages) {
                for (size_t mi = 0; mi < state.message_data.size(); ++mi) {
                    message_tasks.push_back({ i, mi, 1 });
                }
            } else {
                message_tasks.push_back({ i, 0, state.message_data.size() });
            }
        }

        worker_pool.parallel_for(message_tasks.size(), [&](size_t task_index, size_t) {
            const jsonl_message_task & task = message_tasks[task_index];
            jsonl_line_state & state = *states[task.line_index];
            try {
                for (size_t mi = task.first_message; mi < task.first_message + task.message_count; ++mi) {
                    const nlohmann::json & data = *state.message_data[mi];
                    common_chat_msg msg;
                    msg.role = data.value("role", "user");
                    common_chat_msg_content_part part;
                    part.type = "text";
                    part.text = data.value("content", "");
                    msg.content_parts.push_back(std::move(part));
                    state.messages[mi] = std::move(msg);
                    if (!tmpls) {
                        state.chatml_messages[mi] = apply_chatml_message(state.messages[mi]);
                    }
                }
            } catch (const std::exception & e) {
                state.failed.store(true);
                LOG_WRN("%s: skipping invalid message on line %d: %s\n", __func__, state.lineno, e.what());
            }
        });

        worker_pool.parallel_for(states.size(), [&](size_t i, size_t worker_index) {
            jsonl_line_state & state = *states[i];
            if (!state.json_valid || state.failed.load()) {
                return;
            }

            try {
                const nlohmann::json & data = state.data;
                common_chat_templates * task_templates = tmpls ? worker_templates[worker_index] : nullptr;
                if (data.contains("reward")) {
                    state.reward = data.at("reward").get<float>();
                } else if (data.contains("score")) {
                    state.reward = data.at("score").get<float>();
                }

                if (state.is_chat) {
                    if (state.messages.empty()) {
                        return;
                    }
                    for (size_t mi = state.messages.size(); mi-- > 0;) {
                        if (state.messages[mi].role == "assistant") {
                            state.last_assistant_index = mi;
                            break;
                        }
                    }
                    if (state.last_assistant_index == std::numeric_limits<size_t>::max()) {
                        LOG_DBG("%s: skipping line %d: no assistant turn\n", __func__, state.lineno);
                        return;
                    }

                    const common_chat_msg & assistant = state.messages[state.last_assistant_index];
                    state.last_assistant_content = assistant.content_parts.empty() ? "" : assistant.content_parts[0].text;
                    state.prompt_messages.assign(state.messages.begin(), state.messages.begin() + state.last_assistant_index);

                    if (!task_templates) {
                        size_t total_size = 0;
                        for (size_t mi = 0; mi <= state.last_assistant_index; ++mi) {
                            total_size += state.chatml_messages[mi].size();
                        }
                        state.response_text.reserve(total_size);
                        for (size_t mi = 0; mi <= state.last_assistant_index; ++mi) {
                            state.response_text += state.chatml_messages[mi];
                        }
                    } else if (!state.parallel_messages) {
                        state.prompt_text = apply_qlora_chat_template(task_templates, state.prompt_messages, true);
                        std::vector<common_chat_msg> all_messages = state.prompt_messages;
                        all_messages.push_back(assistant);
                        state.full_text = apply_qlora_chat_template(task_templates, all_messages, false);
                    }
                } else if (data.contains("prompt") && data.contains("response")) {
                    state.response_text = data.at("response").get<std::string>();
                    state.prompt_text = data.at("prompt").get<std::string>();
                    state.last_assistant_content = state.response_text;
                } else if (data.contains("text")) {
                    state.response_text = data.at("text").get<std::string>();
                    state.last_assistant_content = state.response_text;
                } else {
                    LOG_WRN("%s: unknown format on line %d, skipping\n", __func__, state.lineno);
                    return;
                }
                state.prepared = true;
            } catch (const std::exception & e) {
                state.failed.store(true);
                LOG_WRN("%s: skipping invalid data on line %d: %s\n", __func__, state.lineno, e.what());
            }
        });

        std::vector<jsonl_render_task> render_tasks;
        for (size_t i = 0; i < states.size(); ++i) {
            const jsonl_line_state & state = *states[i];
            if (state.prepared && state.is_chat && state.parallel_messages && tmpls) {
                render_tasks.push_back({ i, true });
                render_tasks.push_back({ i, false });
            }
        }

        worker_pool.parallel_for(render_tasks.size(), [&](size_t task_index, size_t worker_index) {
            const jsonl_render_task & task = render_tasks[task_index];
            jsonl_line_state & state = *states[task.line_index];
            common_chat_templates * task_templates = worker_templates[worker_index];
            try {
                if (task.render_prompt) {
                    state.prompt_text = apply_qlora_chat_template(task_templates, state.prompt_messages, true);
                } else {
                    std::vector<common_chat_msg> all_messages = state.prompt_messages;
                    all_messages.push_back(state.messages[state.last_assistant_index]);
                    state.full_text = apply_qlora_chat_template(task_templates, all_messages, false);
                }
            } catch (const std::exception & e) {
                state.failed.store(true);
                LOG_WRN("%s: chat template failed on line %d: %s\n", __func__, state.lineno, e.what());
            }
        });

        worker_pool.parallel_for(states.size(), [&](size_t i, size_t worker_index) {
            jsonl_line_state & state = *states[i];
            jsonl_load_result & result = results[i];
            if (!state.prepared || state.failed.load()) {
                return;
            }

            try {
                if (state.is_chat && tmpls) {
                    if (state.full_text.rfind(state.prompt_text, 0) == 0) {
                        state.response_text = state.full_text.substr(state.prompt_text.size());
                    } else {
                        std::string marker = "__LLAMA_QLORA_RESPONSE_MARKER__";
                        while (state.last_assistant_content.find(marker) != std::string::npos) {
                            marker += "_";
                        }

                        common_chat_msg marked_message = state.messages[state.last_assistant_index];
                        marked_message.content_parts.clear();
                        common_chat_msg_content_part marker_part;
                        marker_part.type = "text";
                        marker_part.text = marker;
                        marked_message.content_parts.push_back(std::move(marker_part));

                        std::vector<common_chat_msg> marked_messages = state.prompt_messages;
                        marked_messages.push_back(std::move(marked_message));
                        const std::string marked_text = apply_qlora_chat_template(worker_templates[worker_index], marked_messages, false);
                        const size_t marker_pos = marked_text.find(marker);
                        if (marker_pos != std::string::npos) {
                            const std::string suffix = marked_text.substr(marker_pos + marker.size());
                            state.response_text.reserve(state.last_assistant_content.size() + suffix.size());
                            state.response_text = state.last_assistant_content;
                            state.response_text += suffix;
                            LOG_WRN("%s: line %d full chat render is not prefixed by prompt render; using marker-derived response suffix\n",
                                    __func__, state.lineno);
                        } else {
                            LOG_WRN("%s: line %d could not derive template response suffix; using raw assistant content\n",
                                    __func__, state.lineno);
                            state.response_text = state.last_assistant_content;
                        }
                    }
                }

                std::vector<critical_span> spans;
                if (critical_mode_uses_spans(critical_mode)) {
                    std::string span_error;
                    if (!critical_token_parse_spans(state.data, state.last_assistant_content, critical_default_weight, spans, span_error)) {
                        LOG_WRN("%s: invalid critical_spans on line %d: %s; expected [{\"start\":BYTE,\"end\":BYTE,\"weight\":FLOAT?}]\n",
                                __func__, state.lineno, span_error.c_str());
                        return;
                    }
                }

                result.parsed = true;
                result.request = { std::move(state.prompt_text), std::move(state.response_text),
                                   std::move(state.last_assistant_content), std::move(spans), state.reward };
            } catch (const std::exception & e) {
                state.failed.store(true);
                LOG_WRN("%s: formatting failed on line %d: %s\n", __func__, state.lineno, e.what());
            }
        });

        std::vector<jsonl_token_task> token_tasks;
        token_tasks.reserve(results.size() * 2);
        for (size_t i = 0; i < results.size(); ++i) {
            if (!results[i].parsed || states[i]->failed.load()) {
                continue;
            }
            if (states[i]->parallel_messages) {
                token_tasks.push_back({ i, 0 });
                token_tasks.push_back({ i, 1 });
            } else {
                token_tasks.push_back({ i, 2 });
            }
        }

        worker_pool.parallel_for(token_tasks.size(), [&](size_t task_index, size_t) {
            const jsonl_token_task & task = token_tasks[task_index];
            jsonl_line_state & state = *states[task.line_index];
            const tokenization_request & request = results[task.line_index].request;
            try {
                if (task.part == 0 || task.part == 2) {
                    state.prompt_tokens = common_tokenize(vocab, request.prompt, /*add_special=*/true, /*parse_special=*/true);
                }
                if (task.part == 1 || task.part == 2) {
                    state.response_tokens = common_tokenize(vocab, request.response, /*add_special=*/false, /*parse_special=*/true);
                }
            } catch (const std::exception & e) {
                state.failed.store(true);
                LOG_WRN("%s: tokenization failed on line %d: %s\n", __func__, state.lineno, e.what());
            }
        });

        worker_pool.parallel_for(results.size(), [&](size_t i, size_t) {
            jsonl_line_state & state = *states[i];
            jsonl_load_result & result = results[i];
            if (!result.parsed || state.failed.load()) {
                return;
            }
            if (state.prompt_tokens.empty() && state.response_tokens.empty()) {
                return;
            }

            training_sample & sample = result.sample;
            sample.reward = result.request.reward;
            sample.tokens.reserve(state.prompt_tokens.size() + state.response_tokens.size());
            sample.tokens.insert(sample.tokens.end(), state.prompt_tokens.begin(), state.prompt_tokens.end());
            sample.tokens.insert(sample.tokens.end(), state.response_tokens.begin(), state.response_tokens.end());
            sample.is_label.resize(sample.tokens.size(), false);
            if (critical_mode != "none") {
                sample.critical_weights.resize(sample.tokens.size(), 1.0f);
            }
            for (size_t j = state.prompt_tokens.size(); j < sample.tokens.size(); ++j) {
                sample.is_label[j] = true;
            }
            if (critical_mode_uses_spans(critical_mode) && !result.request.critical_spans.empty()) {
                std::vector<critical_token_range> ranges;
                std::vector<float> response_weights;
                std::string alignment_error;
                if (!response_token_ranges(vocab, state.response_tokens, result.request.response, result.request.annotated_response, ranges, alignment_error) ||
                    !critical_token_apply_spans(result.request.annotated_response.size(), result.request.critical_spans, ranges, response_weights, alignment_error)) {
                    LOG_WRN("%s: invalid critical_spans alignment on line %d: %s\n",
                            __func__, state.lineno, alignment_error.c_str());
                    return;
                }
                std::copy(response_weights.begin(), response_weights.end(), sample.critical_weights.begin() + state.prompt_tokens.size());
            }
            result.valid = true;
        });
    }

    samples.reserve(results.size());
    size_t n_requests = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        const jsonl_load_result & result = results[i];
        if (result.parsed) ++n_requests;
        if (!result.valid) continue;
        if (!logged_preview) {
            logged_preview = true;
            LOG_INF("%s: first prompt preview: \"%s\"\n", __func__, preview_text(result.request.prompt).c_str());
            LOG_INF("%s: first response preview: \"%s\"\n", __func__, preview_text(result.request.response).c_str());
            const size_t prompt_tokens = result.sample.tokens.size() - std::count(result.sample.is_label.begin(), result.sample.is_label.end(), true);
            LOG_INF("%s: first sample tokens: prompt=%zu response=%zu\n", __func__, prompt_tokens, result.sample.tokens.size() - prompt_tokens);
        }
        samples.push_back(std::move(results[i].sample));
    }

    LOG_INF("%s: tokenized %zu input lines with %zu workers\n", __func__, n_requests, n_workers);

    LOG_INF("%s: loaded %zu samples from %s\n", __func__, samples.size(), path.c_str());
    return samples;
}

// Pack variable-length samples into fixed-context-length windows and create
// an ggml_opt_dataset. Labels for prompt tokens are set to -1 (ignored by
// the loss in the epoch loop).
// window_rewards is filled with one reward weight per window (averaged over
// the sample tokens that fall in that window). If all samples have reward=1.0
// the vector is all-ones and has no effect.
static ggml_opt_dataset_t build_dataset(
        const std::vector<training_sample> & samples,
        int32_t                              n_ctx,
        std::vector<float>                 & window_rewards,
        bool                                 train_on_prompt = false,
        llama_token                          bos_token = -1,
        bool                                 critical_enabled = false,
        std::vector<llama_opt_critical_token_metadata> * critical_metadata_out = nullptr) {

    // Flatten samples into token/label/reward streams
    std::vector<llama_token> flat_tokens;
    std::vector<int32_t>     flat_labels;  // -1 = no loss, token_id = loss target
    std::vector<float>       flat_rewards; // per-token reward from the source sample
    std::vector<float>       flat_span_weights;

    for (size_t si = 0; si < samples.size(); ++si) {
        const auto & s = samples[si];

        // Insert BOS separator between samples to prevent cross-sample predictions.
        // The first sample already has BOS from tokenization (add_special=true).
        if (si > 0 && bos_token >= 0 && !s.tokens.empty()) {
            flat_tokens .push_back(bos_token);
            flat_labels .push_back(-1);  // no loss on separator
            flat_rewards.push_back(s.reward);
            if (critical_enabled) flat_span_weights.push_back(0.0f);
        }

        for (size_t i = 0; i + 1 < s.tokens.size(); ++i) {
            flat_tokens .push_back(s.tokens[i]);
            bool active = false;
            if (train_on_prompt) {
                // All positions get correct next-token label (prompt + response)
                flat_labels.push_back((int32_t)s.tokens[i + 1]);
                active = true;
            } else {
                // Only response positions get loss; prompt positions get -1 (sentinel).
                // The sentinel is passed through to labels_sparse; opt_epoch_iter skips
                // writing to the label tensor for those positions, leaving them zeroed →
                // zero cross-entropy contribution.  No gradient flows from prompt tokens.
                flat_labels.push_back(s.is_label[i + 1] ? (int32_t)s.tokens[i + 1] : -1);
                active = s.is_label[i + 1];
            }
            flat_rewards.push_back(s.reward);
            if (critical_enabled) {
                GGML_ASSERT(s.critical_weights.size() == s.tokens.size());
                const float span_weight = s.is_label[i + 1] ? s.critical_weights[i + 1] : 1.0f;
                flat_span_weights.push_back(active ? span_weight : 0.0f);
            }
        }
    }

    if (flat_tokens.empty()) {
        LOG_ERR("%s: dataset has no tokens\n", __func__);
        return nullptr;
    }

    const int64_t stride = n_ctx / 2;
    const int64_t n_tokens = (int64_t) flat_tokens.size();
    int64_t ndata = 1;
    if (n_tokens > n_ctx) {
        ndata += (n_tokens - n_ctx + stride - 1) / stride;
    }

    window_rewards.resize(ndata);

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
            GGML_TYPE_I32, GGML_TYPE_I32, n_ctx, n_ctx, ndata, 1);

    int32_t * data   = (int32_t *) ggml_opt_dataset_data  (dataset)->data;
    int32_t * labels = (int32_t *) ggml_opt_dataset_labels(dataset)->data;
    int64_t n_labels = 0;
    int64_t n_padded = 0;
    std::vector<llama_opt_critical_token_metadata> critical_metadata;
    if (critical_enabled) {
        critical_metadata.resize(ndata * n_ctx);
    }

    for (int64_t i = 0; i < ndata; ++i) {
        const int64_t off = i * stride;
        float reward_sum = 0.0f;
        int64_t reward_count = 0;
        for (int32_t j = 0; j < n_ctx; ++j) {
            const int64_t idx = off + j;
            if (idx >= n_tokens) {
                data  [i * n_ctx + j] = bos_token >= 0 ? bos_token : flat_tokens.back();
                labels[i * n_ctx + j] = -1;
                reward_sum += 1.0f;
                if (critical_enabled) critical_metadata[i * n_ctx + j] = { 0.0f, 0.0f };
                ++n_padded;
                continue;
            }

            data  [i * n_ctx + j] = flat_tokens[idx];
            // Pass -1 sentinel through unchanged for masked (prompt) positions.
            // opt_epoch_iter skips these positions (no label tensor write → zero
            // cross-entropy contribution).  Do NOT substitute the current token
            // here — that trains the model to predict itself (off-by-one) and
            // causes repetition degeneration.
            labels[i * n_ctx + j] = flat_labels[idx];
            if (flat_labels[idx] >= 0) {
                ++n_labels;
                reward_sum += flat_rewards[idx];
                ++reward_count;
                if (critical_enabled) {
                    critical_metadata[i * n_ctx + j] = { flat_span_weights[idx], flat_rewards[idx] };
                }
            } else if (critical_enabled) {
                critical_metadata[i * n_ctx + j] = { 0.0f, 0.0f };
            }
            reward_sum += flat_rewards[idx];
        }
        window_rewards[i] = reward_sum / n_ctx;
    }
    LOG_INF("%s: packed %ld tokens into %ld windows (ctx=%d stride=%ld, supervised labels=%ld, padded=%ld)\n",
            __func__, (long)n_tokens, (long)ndata, n_ctx, (long)stride, (long)n_labels, (long)n_padded);

    // Normalize window rewards to [0, 1].
    // Step 1: clip to [-1, 1] — outliers like 1.3/1.4 would otherwise compress the
    //         useful signal range after min-max scaling (a reward=1.0 would map to
    //         only 0.83 instead of 1.0 if the max is 1.4).
    // Step 2: min-max scale clipped values → [0, 1].
    //         min → 0.0 (window ignored), max → 1.0 (full weight).
    // If all rewards are identical (pure SFT dataset) keep at 1.0.
    for (float & r : window_rewards) {
        r = std::max(-1.0f, std::min(1.0f, r));
    }
    float rmin = *std::min_element(window_rewards.begin(), window_rewards.end());
    float rmax = *std::max_element(window_rewards.begin(), window_rewards.end());
    const float rrange = rmax - rmin;
    if (rrange > 1e-6f) {
        for (float & r : window_rewards) {
            r = (r - rmin) / rrange;
        }
        LOG_INF("%s: reward range [%.4f, %.4f] (after clip to [-1,1]) → normalized to [0, 1]\n", __func__, rmin, rmax);
    } else {
        std::fill(window_rewards.begin(), window_rewards.end(), 1.0f);
    }

    if (critical_enabled) {
        float token_reward_min = 1.0f;
        float token_reward_max = -1.0f;
        for (llama_opt_critical_token_metadata & metadata : critical_metadata) {
            if (metadata.span_weight == 0.0f) continue;
            metadata.reward_weight = std::max(-1.0f, std::min(1.0f, metadata.reward_weight));
            token_reward_min = std::min(token_reward_min, metadata.reward_weight);
            token_reward_max = std::max(token_reward_max, metadata.reward_weight);
        }
        const float token_reward_range = token_reward_max - token_reward_min;
        for (llama_opt_critical_token_metadata & metadata : critical_metadata) {
            if (metadata.span_weight == 0.0f) continue;
            metadata.reward_weight = token_reward_range > 1e-6f
                ? (metadata.reward_weight - token_reward_min) / token_reward_range
                : 1.0f;
        }
        ggml_opt_dataset_set_aux(dataset, critical_metadata.data(), n_ctx * sizeof(critical_metadata[0]));
        GGML_ASSERT(ggml_opt_dataset_aux_size(dataset) == n_ctx * sizeof(critical_metadata[0]));
        if (critical_metadata_out) {
            *critical_metadata_out = std::move(critical_metadata);
        }
    }

    return dataset;
}

// ---------------------------------------------------------------------------
// LoRA tensor allocation
// ---------------------------------------------------------------------------

struct lora_tensors {
    struct ggml_context      * ctx  = nullptr;
    struct ggml_backend_buffer * buf = nullptr;
    // map: base tensor name → {lora_a, lora_b}
    std::unordered_map<std::string, std::pair<ggml_tensor*, ggml_tensor*>> ab;
};

static lora_tensors alloc_lora_tensors(
        const std::string        & model_path,
        const std::vector<std::string> & targets,
        int32_t                   rank,
        std::mt19937            & rng,
        int32_t                   freeze_layers = 0) {

    lora_tensors lt;

    // Open the model GGUF to discover tensor names and shapes
    // without needing access to private llama_model internals.
    struct ggml_context * ctx_meta = nullptr;
    struct gguf_init_params gguf_params = { /*.no_alloc=*/true, /*.ctx=*/&ctx_meta };
    struct gguf_context * ctx_gguf = gguf_init_from_file(model_path.c_str(), gguf_params);
    if (!ctx_gguf) {
        LOG_ERR("%s: failed to open model GGUF for tensor discovery: %s\n",
                __func__, model_path.c_str());
        return lt;
    }

    // Collect matching 2-D tensors
    struct tensor_info { std::string name; int64_t ne0, ne1; };
    std::vector<tensor_info> matched;

    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta);
         t; t = ggml_get_next_tensor(ctx_meta, t)) {
        if (ggml_n_dims(t) < 2) continue;
        if (!tensor_matches_targets(t->name, targets, freeze_layers)) continue;
        matched.push_back({t->name, t->ne[0], t->ne[1]});
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);

    if (matched.empty()) {
        LOG_ERR("%s: no model tensors matched --lora-targets; check spelling\n", __func__);
        return lt;
    }

    if (freeze_layers > 0) {
        LOG_INF("%s: freezing layers blk.0 .. blk.%d (no LoRA allocated; backward already pruned by grads_needed)\n",
                __func__, freeze_layers - 1);
    }
    LOG_INF("%s: allocating LoRA A/B tensors for %zu weight matrices, rank=%d\n",
            __func__, matched.size(), rank);

    // Allocate ggml context for A+B tensors (2 tensors per matched weight)
    const size_t mem = (2 * matched.size() + 16) * ggml_tensor_overhead();
    struct ggml_init_params ip = { mem, nullptr, /*no_alloc=*/true };
    lt.ctx = ggml_init(ip);

    for (const auto & ti : matched) {
        const int64_t in_dim  = ti.ne0; // columns (input features)
        const int64_t out_dim = ti.ne1; // rows    (output features)

        // lora_a: [in_dim, rank]   applied first: a @ x
        // lora_b: [rank,   out_dim] applied second: b @ (a @ x)
        // Convention matches llama-adapter.cpp:48-60:
        //   a->ne[0] == in_dim,  a->ne[1] == rank
        //   b->ne[0] == rank,    b->ne[1] == out_dim
        ggml_tensor * la = ggml_new_tensor_2d(lt.ctx, GGML_TYPE_F32, in_dim, rank);
        ggml_tensor * lb = ggml_new_tensor_2d(lt.ctx, GGML_TYPE_F32, rank,   out_dim);

        ggml_set_name(la, (ti.name + ".lora_a").c_str());
        ggml_set_name(lb, (ti.name + ".lora_b").c_str());

        lt.ab[ti.name] = {la, lb};
    }

    // Allocate backend buffer for all LoRA tensors at once
    lt.buf = ggml_backend_alloc_ctx_tensors_from_buft(lt.ctx, ggml_backend_cpu_buffer_type());

    // Initialize: A ~ N(0, 1/sqrt(rank)), B = 0
    const float std_a = 1.0f / std::sqrt((float)rank);
    std::normal_distribution<float> dist(0.0f, std_a);

    for (auto & kv : lt.ab) {
        ggml_tensor * la = kv.second.first;
        ggml_tensor * lb = kv.second.second;

        // Fill A
        float * data_a = (float *) la->data;
        for (int64_t i = 0; i < ggml_nelements(la); ++i) data_a[i] = dist(rng);
        // Zero B
        memset(lb->data, 0, ggml_nbytes(lb));
    }

    return lt;
}

// ---------------------------------------------------------------------------
// Param filter: only train lora_a / lora_b tensors
// ---------------------------------------------------------------------------

static bool lora_param_filter(const struct ggml_tensor * t, void * /*ud*/) {
    const char * n = t->name;
    const size_t len = strlen(n);
    if (len > 7 && strcmp(n + len - 7, ".lora_a") == 0) return true;
    if (len > 7 && strcmp(n + len - 7, ".lora_b") == 0) return true;
    return false;
}

static enum llama_lora_qat_type lora_qat_type_from_string(const std::string & type) {
    if (type == "q3_k") return  LLAMA_LORA_QAT_TYPE_Q3_K;
    if (type == "q4_k") return  LLAMA_LORA_QAT_TYPE_Q4_K;
    if (type == "q4_0") return  LLAMA_LORA_QAT_TYPE_Q4_0;
    if (type == "mxfp4") return LLAMA_LORA_QAT_TYPE_MXFP4;
    if (type == "q6_k") return  LLAMA_LORA_QAT_TYPE_Q6_K;
    if (type == "q8_0") return  LLAMA_LORA_QAT_TYPE_Q8_0;
    return LLAMA_LORA_QAT_TYPE_NONE;
}

static enum llama_opt_critical_token_mode critical_token_mode_from_string(const std::string & mode) {
    if (mode == "spans") return LLAMA_OPT_CRITICAL_TOKEN_MODE_SPANS;
    if (mode == "confidence") return LLAMA_OPT_CRITICAL_TOKEN_MODE_CONFIDENCE;
    if (mode == "hybrid") return LLAMA_OPT_CRITICAL_TOKEN_MODE_HYBRID;
    return LLAMA_OPT_CRITICAL_TOKEN_MODE_NONE;
}

static enum llama_opt_critical_weight_shape critical_weight_shape_from_string(const std::string & shape) {
    return shape == "linear" ? LLAMA_OPT_CRITICAL_WEIGHT_SHAPE_LINEAR : LLAMA_OPT_CRITICAL_WEIGHT_SHAPE_CONSTANT;
}

// ---------------------------------------------------------------------------
// Save adapter GGUF
// ---------------------------------------------------------------------------

static std::string basename_from_path(const std::string & p) {
    const size_t pos = p.find_last_of("/\\");
    if (pos == std::string::npos) return p;
    return p.substr(pos + 1);
}

struct checkpoint_state {
    std::string mode;
    int64_t     epoch           = 0;
    int64_t     window          = 0;
    int64_t     step            = 0;
    int64_t     dataset_windows = -1;
    int64_t     context_length  = -1;
    bool        shuffle         = false;
    bool        has_metadata    = false;
    float       alpha           = 0.0f;
    bool        has_alpha       = false;
    int64_t     schedule_step   = -1;
};

static bool parse_checkpoint_number(
        const std::string & path,
        size_t              begin,
        size_t              end,
        int64_t           & value) {
    if (begin >= end) return false;

    const std::string number = path.substr(begin, end - begin);
    char * number_end = nullptr;
    errno = 0;
    const long long parsed = strtoll(number.c_str(), &number_end, 10);
    if (errno != 0 || number_end != number.c_str() + number.size() || parsed < 0) return false;

    value = (int64_t) parsed;
    return true;
}

static bool checkpoint_state_from_filename(
        const std::string & path,
        const std::string & mode,
        checkpoint_state  & state) {
    const size_t suffix = path.rfind(".gguf");
    if (suffix == std::string::npos || suffix + 5 != path.size()) return false;

    if (mode == "sft") {
        const size_t epoch_tag = path.rfind(".epoch", suffix);
        const size_t ckpt_tag  = path.rfind(".ckpt",  suffix);
        if (epoch_tag == std::string::npos || ckpt_tag == std::string::npos || epoch_tag >= ckpt_tag) return false;
        if (!parse_checkpoint_number(path, epoch_tag + 6, ckpt_tag, state.epoch)) return false;
        if (!parse_checkpoint_number(path, ckpt_tag + 5, suffix, state.window)) return false;
    } else {
        const size_t ckpt_tag = path.rfind(".ckpt", suffix);
        if (ckpt_tag == std::string::npos) return false;
        if (!parse_checkpoint_number(path, ckpt_tag + 5, suffix, state.step)) return false;
    }

    state.mode = mode;
    return true;
}

static bool gguf_get_i64(const gguf_context * gctx, const char * key, int64_t & value) {
    const int64_t key_id = gguf_find_key(gctx, key);
    if (key_id < 0) return false;

    switch (gguf_get_kv_type(gctx, key_id)) {
        case GGUF_TYPE_UINT32: value = gguf_get_val_u32(gctx, key_id); return true;
        case GGUF_TYPE_INT32:  value = gguf_get_val_i32(gctx, key_id); return true;
        case GGUF_TYPE_UINT64: {
            const uint64_t val = gguf_get_val_u64(gctx, key_id);
            if (val > INT64_MAX) return false;
            value = (int64_t) val;
            return true;
        }
        case GGUF_TYPE_INT64: value = gguf_get_val_i64(gctx, key_id); return true;
        default: return false;
    }
}

static bool load_checkpoint_state(
        const std::string & path,
        const std::string & expected_mode,
        checkpoint_state  & state) {
    ggml_context * ctx_meta = nullptr;
    const gguf_init_params params = { true, &ctx_meta };
    gguf_context * gctx = gguf_init_from_file(expand_tilde(path).c_str(), params);
    if (!gctx) {
        LOG_ERR("%s: cannot open checkpoint %s\n", __func__, path.c_str());
        return false;
    }

    const int64_t mode_id = gguf_find_key(gctx, "training.checkpoint.mode");
    if (mode_id >= 0 && gguf_get_kv_type(gctx, mode_id) == GGUF_TYPE_STRING) {
        state.mode = gguf_get_val_str(gctx, mode_id);
        state.has_metadata = true;
        gguf_get_i64(gctx, "training.checkpoint.epoch", state.epoch);
        gguf_get_i64(gctx, "training.checkpoint.window", state.window);
        gguf_get_i64(gctx, "training.checkpoint.step", state.step);
        gguf_get_i64(gctx, "training.dataset.windows", state.dataset_windows);
        gguf_get_i64(gctx, "training.context_length", state.context_length);
        gguf_get_i64(gctx, "training.scheduler.step", state.schedule_step);

        const int64_t shuffle_id = gguf_find_key(gctx, "training.dataset.shuffle");
        if (shuffle_id >= 0 && gguf_get_kv_type(gctx, shuffle_id) == GGUF_TYPE_BOOL) {
            state.shuffle = gguf_get_val_bool(gctx, shuffle_id);
        }
    }

    const int64_t alpha_id = gguf_find_key(gctx, "adapter.lora.alpha");
    if (alpha_id >= 0 && gguf_get_kv_type(gctx, alpha_id) == GGUF_TYPE_FLOAT32) {
        state.alpha = gguf_get_val_f32(gctx, alpha_id);
        state.has_alpha = true;
    }

    gguf_free(gctx);
    ggml_free(ctx_meta);

    if (!state.has_metadata && !checkpoint_state_from_filename(path, expected_mode, state)) {
        LOG_ERR("%s: %s has no checkpoint metadata and its filename does not contain a checkpoint position\n",
                __func__, path.c_str());
        return false;
    }
    if (state.mode != expected_mode) {
        LOG_ERR("%s: checkpoint mode is %s, expected %s\n",
                __func__, state.mode.c_str(), expected_mode.c_str());
        return false;
    }
    if (state.mode == "sft" && state.epoch <= 0) {
        LOG_ERR("%s: invalid checkpoint epoch %ld\n", __func__, (long) state.epoch);
        return false;
    }
    if (state.window < 0 || state.step < 0 || state.schedule_step < -1) {
        LOG_ERR("%s: checkpoint contains a negative training position\n", __func__);
        return false;
    }
    return true;
}

static void save_adapter(
        const lora_tensors & lt,
        const std::string  & out_path,
        const std::string  & arch,
        float                alpha,
        const std::string  & base_model_path,
        const checkpoint_state * checkpoint = nullptr) {

    // Build output GGUF context
    struct gguf_context * gctx = gguf_init_empty();

    // Metadata required by llama_adapter_lora_init
    gguf_set_val_str(gctx, "general.type",         "adapter");
    gguf_set_val_str(gctx, "general.architecture",  arch.c_str());
    gguf_set_val_str(gctx, "adapter.type",          "lora");
    gguf_set_val_f32(gctx, "adapter.lora.alpha",    alpha);
    gguf_set_val_str(gctx, "adapter.base_model",    basename_from_path(base_model_path).c_str());

    if (checkpoint) {
        gguf_set_val_u32(gctx, "training.checkpoint.version", 1);
        gguf_set_val_str(gctx, "training.checkpoint.mode", checkpoint->mode.c_str());
        gguf_set_val_i64(gctx, "training.checkpoint.epoch", checkpoint->epoch);
        gguf_set_val_i64(gctx, "training.checkpoint.window", checkpoint->window);
        gguf_set_val_i64(gctx, "training.checkpoint.step", checkpoint->step);
        gguf_set_val_i64(gctx, "training.dataset.windows", checkpoint->dataset_windows);
        gguf_set_val_i64(gctx, "training.context_length", checkpoint->context_length);
        gguf_set_val_bool(gctx, "training.dataset.shuffle", checkpoint->shuffle);
        gguf_set_val_i64(gctx, "training.scheduler.step", checkpoint->schedule_step);
    }

    // Register tensors
    for (const auto & kv : lt.ab) {
        gguf_add_tensor(gctx, kv.second.first);   // lora_a
        gguf_add_tensor(gctx, kv.second.second);  // lora_b
    }

    // Write: meta placeholder → tensor data → rewrite meta
    const std::string real_path = expand_tilde(out_path);
    std::ofstream fout(real_path, std::ios::binary);
    if (!fout.is_open()) {
        LOG_ERR("%s: cannot open %s for writing\n", __func__, real_path.c_str());
        gguf_free(gctx);
        return;
    }

    // Write meta placeholder
    const size_t meta_size = gguf_get_meta_size(gctx);
    std::vector<char> zeros_buf(meta_size, 0);
    fout.write(zeros_buf.data(), meta_size);

    // Write tensor data — copy to CPU first in case tensors live on GPU
    for (const auto & kv : lt.ab) {
        for (ggml_tensor * t : {kv.second.first, kv.second.second}) {
            const size_t nb = ggml_nbytes(t);
            std::vector<char> cpu_buf(nb);
            ggml_backend_tensor_get(t, cpu_buf.data(), 0, nb);
            fout.write(cpu_buf.data(), nb);
            // GGUF tensors are 32-byte aligned
            const size_t pad = GGML_PAD(nb, 32) - nb;
            if (pad > 0) {
                std::vector<char> pad_buf(pad, 0);
                fout.write(pad_buf.data(), pad);
            }
        }
    }

    // Re-write metadata at offset 0
    std::vector<uint8_t> meta(meta_size);
    gguf_get_meta_data(gctx, meta.data());
    fout.seekp(0);
    fout.write((const char *) meta.data(), meta_size);

    fout.close();
    gguf_free(gctx);

    LOG_INF("%s: adapter saved to %s\n", __func__, real_path.c_str());
}

struct adapter_stats {
    double a_l2      = 0.0;
    double b_l2      = 0.0;
    double total_l2  = 0.0;
    double a_max_abs = 0.0;
    double b_max_abs = 0.0;
};

static void adapter_stats_accum(ggml_tensor * t, double & sum_sq, double & max_abs) {
    const size_t nb = ggml_nbytes(t);
    std::vector<float> buf(nb / sizeof(float));
    ggml_backend_tensor_get(t, buf.data(), 0, nb);

    for (float v : buf) {
        const double vd = (double) v;
        const double av = std::abs(vd);
        sum_sq += vd * vd;
        max_abs = std::max(max_abs, av);
    }
}

static adapter_stats adapter_get_stats(const lora_tensors & lt) {
    adapter_stats stats;
    double a_sum_sq = 0.0;
    double b_sum_sq = 0.0;

    for (const auto & kv : lt.ab) {
        adapter_stats_accum(kv.second.first,  a_sum_sq, stats.a_max_abs);
        adapter_stats_accum(kv.second.second, b_sum_sq, stats.b_max_abs);
    }

    stats.a_l2     = std::sqrt(a_sum_sq);
    stats.b_l2     = std::sqrt(b_sum_sq);
    stats.total_l2 = std::sqrt(a_sum_sq + b_sum_sq);
    return stats;
}

// ---------------------------------------------------------------------------
// Periodic checkpoint callback
// ---------------------------------------------------------------------------

struct save_ctx {
    const lora_tensors * lt;
    const std::string  * lora_out;
    const std::string  * arch;
    const std::string  * base_model_path;
    float                lora_alpha;
    int32_t              save_every;     // 0 = disabled
    int32_t              ubatch_per_ctx;
    int64_t              window_offset;  // completed windows before this epoch call
    int64_t              last_saved;     // last window index at which we saved
    int64_t              epoch;
    int64_t              dataset_windows;
    int64_t              context_length;
    bool                 shuffle;
    bool                 verbose_loss;
    qlora_lr_schedule * schedule;
    bool                 loss_ema_initialized = false;
    double               loss_ema16 = 0.0;
    double               loss_ema64 = 0.0;
    double               loss_cumulative_sum = 0.0;
    int64_t              loss_cumulative_count = 0;
    double               loss_epoch_sum = 0.0;
    int64_t              loss_epoch_count = 0;
};

// TLS pointer set before each epoch so the static callback can access it.
static thread_local save_ctx * g_save_ctx = nullptr;

static void save_every_callback(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    ggml_opt_epoch_callback_progress_bar(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us);

    // Log loss at every window boundary so we can see if/when it diverges.
    if (train && g_save_ctx) {
        const int64_t window = g_save_ctx->window_offset + ibatch / g_save_ctx->ubatch_per_ctx;
        const int64_t ubatch_in_window = ibatch % g_save_ctx->ubatch_per_ctx;
        if (ibatch > 0 && ubatch_in_window == 0) {
            if (g_save_ctx->verbose_loss) {
                double cumulative_loss_mean = 0.0;
                ggml_opt_result_loss(result, &cumulative_loss_mean, nullptr);

                const int64_t cumulative_loss_count = ibatch;
                const double cumulative_loss_sum = cumulative_loss_mean * cumulative_loss_count;
                const double window_loss_sum = cumulative_loss_sum - g_save_ctx->loss_cumulative_sum;
                const int64_t window_loss_count = cumulative_loss_count - g_save_ctx->loss_cumulative_count;
                const double window_loss = window_loss_sum / std::max<int64_t>(1, window_loss_count);

                g_save_ctx->loss_ema16 = g_save_ctx->loss_ema_initialized
                    ? 0.8825 * g_save_ctx->loss_ema16 + 0.1175 * window_loss
                    : window_loss;
                g_save_ctx->loss_ema64 = g_save_ctx->loss_ema_initialized
                    ? 0.9692 * g_save_ctx->loss_ema64 + 0.0308 * window_loss
                    : window_loss;
                g_save_ctx->loss_ema_initialized = true;
                g_save_ctx->loss_cumulative_sum = cumulative_loss_sum;
                g_save_ctx->loss_cumulative_count = cumulative_loss_count;
                g_save_ctx->loss_epoch_sum += window_loss_sum;
                g_save_ctx->loss_epoch_count += window_loss_count;

                const double epoch_mean = g_save_ctx->loss_epoch_sum /
                    std::max<int64_t>(1, g_save_ctx->loss_epoch_count);
                fprintf(stderr,
                        "\nepoch=%ld window=%ld window_loss=%.6f ema16=%.6f ema64=%.6f epoch_mean=%.6f lr=%.6g\n",
                        (long) g_save_ctx->epoch, (long) window, window_loss,
                        g_save_ctx->loss_ema16, g_save_ctx->loss_ema64, epoch_mean,
                        (double) g_save_ctx->schedule->current_lr);
            }
            ++g_save_ctx->schedule->step;
        }
    }

    if (!train || !g_save_ctx || g_save_ctx->save_every <= 0) return;
    const int64_t window = g_save_ctx->window_offset + ibatch / g_save_ctx->ubatch_per_ctx;
    if (window > 0 && window != g_save_ctx->last_saved && window % g_save_ctx->save_every == 0) {
        g_save_ctx->last_saved = window;
        const std::string ckpt = *g_save_ctx->lora_out
            + ".epoch" + std::to_string(g_save_ctx->epoch)
            + ".ckpt" + std::to_string(window) + ".gguf";
        checkpoint_state state;
        state.mode            = "sft";
        state.epoch           = g_save_ctx->epoch;
        state.window          = window;
        state.dataset_windows = g_save_ctx->dataset_windows;
        state.context_length  = g_save_ctx->context_length;
        state.shuffle         = g_save_ctx->shuffle;
        state.schedule_step   = g_save_ctx->schedule->step;
        save_adapter(*g_save_ctx->lt, ckpt, *g_save_ctx->arch, g_save_ctx->lora_alpha,
                     *g_save_ctx->base_model_path, &state);
        fprintf(stderr, "\n");
        LOG_INF("save_every_callback: checkpoint saved -> %s (window %ld)\n", ckpt.c_str(), (long)window);
    }
}

// ---------------------------------------------------------------------------
// IPC helpers  (stdout protocol, stdin commands)
// ---------------------------------------------------------------------------

// Escape newlines and backslashes for single-line IPC transmission.
// Mirrors _escape() in gguf_trainer.py.
static std::string ipc_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if      (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else                out += c;
    }
    return out;
}

static void ipc_emit(const char * msg) {
    fputs(msg, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

// Read one line from stdin, trimming the trailing newline.
// Returns false on EOF or error.
static bool ipc_read_line(std::string & out) {
    out.clear();
    if (!std::getline(std::cin, out)) return false;
    // Strip trailing \r if present (Windows line endings)
    if (!out.empty() && out.back() == '\r') out.pop_back();
    return true;
}

// Parse "REWARD r1 r2 ... rN" into a float vector.
static std::vector<float> ipc_parse_rewards(const std::string & line) {
    std::vector<float> rewards;
    if (line.size() < 8 || line.substr(0, 7) != "REWARD ") return rewards;
    std::istringstream ss(line.substr(7));
    float r;
    while (ss >> r) rewards.push_back(r);
    return rewards;
}

// ---------------------------------------------------------------------------
// Greedy / temperature sampling for GRPO rollout generation
// ---------------------------------------------------------------------------

static std::string generate_response(
        llama_context      * ctx,
        llama_model        * model,
        const std::string  & prompt,
        int32_t              max_tokens,
        float                temperature,
        std::mt19937       & rng) {

    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto tokens = common_tokenize(ctx, prompt, /*add_special=*/true, /*parse_special=*/true);
    if (tokens.empty()) return "";

    // Clear KV cache before each generation (don't carry over previous prompt state)
    llama_memory_clear(llama_get_memory(ctx), true);
    {
        llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode failed on prompt\n", __func__);
            return "";
        }
    }

    std::string output;
    const llama_token eos = llama_vocab_eos(vocab);
    //const llama_token nl  = llama_vocab_nl(vocab);

    // For ChatML models <|im_end|> is the turn-end marker but may not be the
    // vocab EOS token.  Look it up by tokenizing the string and taking the
    // first token if it tokenizes to exactly one piece.
    llama_token im_end = -1;
    {
        std::vector<llama_token> im_end_tokens(8);
        static const char im_end_str[] = "<|im_end|>";
        int n = llama_tokenize(vocab, im_end_str, (int32_t)strlen(im_end_str), im_end_tokens.data(), (int32_t)im_end_tokens.size(), /*add_special=*/false, /*parse_special=*/true);
        if (n == 1) im_end = im_end_tokens[0];
    }
    const llama_token eot = llama_vocab_eot(vocab);  // may equal eos on some models

    for (int32_t i = 0; i < max_tokens; ++i) {
        // Sample next token — use ith=-1 to always get the LAST output position's
        // logits.  llama_get_logits(ctx) returns position 0 which is wrong when the
        // prompt batch has multiple output tokens (training context).
        float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            LOG_ERR("%s: llama_get_logits_ith(-1) returned NULL\n", __func__);
            break;
        }
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);

        llama_token next_token;
        if (temperature <= 0.0f) {
            // Greedy
            next_token = (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
        } else {
            // Temperature sampling via softmax + categorical draw
            std::vector<float> probs(n_vocab);
            float max_logit = *std::max_element(logits, logits + n_vocab);
            float sum = 0.0f;
            for (int32_t k = 0; k < n_vocab; ++k) {
                probs[k] = std::exp((logits[k] - max_logit) / temperature);
                sum += probs[k];
            }
            for (float & p : probs) p /= sum;
            std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
            next_token = dist(rng);
        }

        if (next_token == eos)     break;
        if (next_token == eot)     break;
        if (im_end >= 0 && next_token == im_end && !output.empty()) break;

        // Decode token to text
        char buf[256] = {};
        llama_token_to_piece(vocab, next_token, buf, sizeof(buf) - 1, 0, true);
        output += buf;

        // Feed token back for next step
        llama_batch batch = llama_batch_get_one(&next_token, 1);
        if (llama_decode(ctx, batch) != 0) break;
    }

    return output;
}

// ---------------------------------------------------------------------------
// GRPO IPC training loop
// ---------------------------------------------------------------------------

// Volatile flag set by SIGINT so the loop can exit cleanly.
static volatile sig_atomic_t g_grpo_stop = 0;
static void grpo_sigint_handler(int) { g_grpo_stop = 1; }

static int run_grpo_mode(
        common_params    & params,
        llama_model      * model,
        llama_context    * ctx,
        lora_tensors     & lt,
        const std::string & arch,
        float               lora_alpha,
        const std::string & base_model_path,
        const checkpoint_state * resume) {

    const int32_t n_ctx    = llama_n_ctx(ctx);
    const int32_t n_gen    = params.grpo_n_gen;
    const int32_t n_steps  = params.grpo_n_steps;
    const float   temp     = params.grpo_temperature;
    const int32_t max_tok  = params.grpo_max_tokens;
    int           step     = resume ? (int) resume->step : 0;

    if (resume && resume->context_length >= 0 && resume->context_length != n_ctx) {
        LOG_ERR("%s: checkpoint context length is %ld, current context length is %d\n",
                __func__, (long) resume->context_length, n_ctx);
        return 1;
    }
    if (n_steps <= 0 || step > n_steps) {
        LOG_ERR("%s: invalid GRPO step range %d/%d\n", __func__, step, n_steps);
        return 1;
    }
    if (params.warmup_steps > n_steps) {
        LOG_ERR("%s: --warmup-steps %d exceeds total GRPO steps %d\n",
                __func__, params.warmup_steps, n_steps);
        return 1;
    }
    if (params.lr.lr0 <= 0.0f || (params.lr_scheduler == "cosine" && params.lr.lr_min > params.lr.lr0)) {
        LOG_ERR("%s: learning-rate must be positive and cosine requires learning-rate-min <= learning-rate\n", __func__);
        return 1;
    }

    const int64_t schedule_step = resume && resume->schedule_step >= 0
        ? resume->schedule_step : step;
    if (schedule_step != step) {
        LOG_ERR("%s: checkpoint scheduler step %ld does not match GRPO step %d\n",
                __func__, (long) schedule_step, step);
        return 1;
    }
    qlora_lr_schedule schedule {
        &params.lr, params.lr_scheduler, params.warmup_steps, params.warmup_init_ratio, n_steps, schedule_step
    };

    std::mt19937 rng(params.sampling.seed != LLAMA_DEFAULT_SEED
                     ? params.sampling.seed : 42);

    // Initialize optimizer
    struct llama_opt_params lopt_params {
        /*.n_ctx_train              =*/0,
        /*.param_filter             =*/lora_param_filter,
        /*.param_filter_ud          =*/nullptr,
        /*.get_opt_pars             =*/qlora_opt_lr_pars,
        /*.get_opt_pars_ud          =*/&schedule,
        /*.optimizer_type           =*/params.optimizer,
        /*.lora_qat_type            =*/lora_qat_type_from_string(params.lora_qat),
        /*.grad_checkpoint_interval =*/params.grad_checkpoint_interval,
        /*.critical_token_mode      =*/LLAMA_OPT_CRITICAL_TOKEN_MODE_NONE,
        /*.critical_token_weight    =*/1.0f,
        /*.critical_confidence_threshold =*/0.25f,
        /*.critical_weight_shape    =*/LLAMA_OPT_CRITICAL_WEIGHT_SHAPE_CONSTANT,
        /*.critical_warmup_steps    =*/0,
        /*.critical_max_fraction    =*/1.0f,
        /*.critical_step            =*/nullptr,
        /*.critical_stats_every     =*/0,
    };
    llama_opt_init(ctx, model, lopt_params);

    const llama_token bos = llama_vocab_bos(llama_model_get_vocab(model));

    signal(SIGINT, grpo_sigint_handler);

    // Signal Python that we are ready
    ipc_emit("[QLORA:READY]");

    float last_loss = 0.0f;

    if (resume) {
        LOG_INF("%s: resuming GRPO after step %d/%d; optimizer state starts fresh\n",
                __func__, step, n_steps);
    }
    LOG_INF("%s: lr scheduler=%s warmup_steps=%d warmup_init_ratio=%.3g total_steps=%d start_step=%ld lr=%.3g lr_min=%.3g\n",
            __func__, params.lr_scheduler.c_str(), params.warmup_steps, (double) params.warmup_init_ratio, n_steps,
            (long) schedule.step, (double) params.lr.lr0, (double) std::max(0.0f, params.lr.lr_min));

    while (step < n_steps && !g_grpo_stop) {
        params.lr.epoch = step;

        // ── Request prompt ────────────────────────────────────────────────
        {
            char buf[64];
            snprintf(buf, sizeof(buf), "[QLORA:PROMPT_REQ:%d]", step + 1);
            ipc_emit(buf);
        }

        std::string prompt_line;
        if (!ipc_read_line(prompt_line)) break;
        if (prompt_line == "STOP") {
            LOG_INF("grpo: received STOP from Python\n");
            break;
        }
        if (prompt_line.size() < 8 || prompt_line.substr(0, 7) != "PROMPT ") {
            char buf[128];
            snprintf(buf, sizeof(buf), "[QLORA:ERROR] expected PROMPT, got: %.80s", prompt_line.c_str());
            ipc_emit(buf);
            return 1;
        }
        // Unescape the prompt (\\n → \n etc.)
        std::string prompt;
        {
            const std::string esc = prompt_line.substr(7);
            prompt.reserve(esc.size());
            for (size_t i = 0; i < esc.size(); ++i) {
                if (esc[i] == '\\' && i + 1 < esc.size()) {
                    char next = esc[i+1];
                    if      (next == 'n')  { prompt += '\n'; ++i; }
                    else if (next == 'r')  { prompt += '\r'; ++i; }
                    else if (next == '\\') { prompt += '\\'; ++i; }
                    else                   { prompt += esc[i]; }
                } else {
                    prompt += esc[i];
                }
            }
        }

        // ── Generate N responses ──────────────────────────────────────────
        std::vector<std::string> generations(n_gen);
        for (int k = 0; k < n_gen; ++k) {
            generations[k] = generate_response(ctx, model, prompt, max_tok, temp, rng);

            char hdr[64];
            snprintf(hdr, sizeof(hdr), "[QLORA:GEN:%d/%d] ", k + 1, n_gen);
            std::string msg = std::string(hdr) + ipc_escape(generations[k]);
            ipc_emit(msg.c_str());
        }

        // ── Request rewards ───────────────────────────────────────────────
        {
            char buf[64];
            snprintf(buf, sizeof(buf), "[QLORA:REWARD_REQ:%d]", n_gen);
            ipc_emit(buf);
        }

        std::string reward_line;
        if (!ipc_read_line(reward_line)) break;
        if (reward_line == "STOP") {
            LOG_INF("grpo: received STOP from Python\n");
            break;
        }
        std::vector<float> rewards = ipc_parse_rewards(reward_line);
        if ((int32_t)rewards.size() != n_gen) {
            char buf[128];
            snprintf(buf, sizeof(buf), "[QLORA:ERROR] expected %d rewards, got %zu", n_gen, rewards.size());
            ipc_emit(buf);
            return 1;
        }

        // ── Build single-step mini-dataset: prompt+generations with rewards ─
        // Each generation is a separate sample; prompt = no-loss, generation = loss.
        std::vector<training_sample> step_samples;
        step_samples.reserve(n_gen);
        for (int k = 0; k < n_gen; ++k) {
            training_sample s;
            s.reward = rewards[k];

            auto tok_prompt = common_tokenize(ctx, prompt,         /*add_special=*/true,  /*parse_special=*/true);
            auto tok_gen    = common_tokenize(ctx, generations[k], /*add_special=*/false, /*parse_special=*/true);

            s.tokens.insert(s.tokens.end(), tok_prompt.begin(), tok_prompt.end());
            s.tokens.insert(s.tokens.end(), tok_gen.begin(),    tok_gen.end());
            s.is_label.resize(s.tokens.size(), false);
            for (size_t i = tok_prompt.size(); i < s.tokens.size(); ++i) {
                s.is_label[i] = true;
            }
            step_samples.push_back(std::move(s));
        }

        // Ensure minimum token count for one context window.
        // build_dataset drops the last token per sample during flattening,
        // so we need total raw tokens > n_ctx to guarantee ndata >= 1.
        while (true) {
            size_t total = 0;
            for (const auto & s : step_samples) total += s.tokens.size();
            if ((int64_t)total > n_ctx + (int64_t)step_samples.size()) break;
            step_samples.push_back(step_samples.back());
        }

        std::vector<float> window_rewards;
        ggml_opt_dataset_t step_dataset = build_dataset(
                step_samples, n_ctx, window_rewards, /*train_on_prompt=*/false, bos);
        if (!step_dataset) {
            ipc_emit("[QLORA:ERROR] build_dataset failed for step");
            return 1;
        }

        // Apply reward weights for this step
        const bool has_rewards = std::any_of(window_rewards.begin(), window_rewards.end(),
                                             [](float r){ return std::abs(r - 1.0f) > 1e-4f; });
        if (has_rewards) {
            llama_opt_set_reward_weights(window_rewards.data(), (int64_t)window_rewards.size());
        }

        // ── One optimizer step (full dataset = one mini-epoch) ────────────
        const int64_t idata_all = ggml_opt_dataset_ndata(step_dataset);
        ggml_opt_result_t step_result = ggml_opt_result_init();

        llama_opt_epoch(ctx, step_dataset, step_result, nullptr, idata_all,
                        nullptr,   // no progress bar callback — clean stdout
                        nullptr,
                        false);    // no shuffle for single-step

        double loss = 0.0, loss_unc = 0.0;
        ggml_opt_result_loss(step_result, &loss, &loss_unc);
        last_loss = (float)loss;

        ggml_opt_result_free(step_result);
        ggml_opt_dataset_free(step_dataset);
        llama_opt_set_reward_weights(nullptr, 0);

        ++step;
        ++schedule.step;

        // ── Emit progress ─────────────────────────────────────────────────
        {
            char buf[128];
            snprintf(buf, sizeof(buf),
                     "[QLORA:PROGRESS] step=%d/%d loss=%.4f epoch=1/1",
                     step, n_steps, last_loss);
            ipc_emit(buf);
        }

        // ── Optional checkpoint ───────────────────────────────────────────
        if (params.save_every > 0 && step % params.save_every == 0) {
            std::string ckpt = params.lora_out + ".ckpt" + std::to_string(step) + ".gguf";
            checkpoint_state state;
            state.mode           = "grpo";
            state.step           = step;
            state.context_length = n_ctx;
            state.schedule_step  = schedule.step;
            save_adapter(lt, ckpt, arch, lora_alpha, base_model_path, &state);
            char buf[512];
            snprintf(buf, sizeof(buf), "[QLORA:CHECKPOINT] %s", ckpt.c_str());
            ipc_emit(buf);
        }
    }

    // Save final adapter
    save_adapter(lt, params.lora_out, arch, lora_alpha, base_model_path);

    {
        char buf[64];
        snprintf(buf, sizeof(buf), "[QLORA:DONE] final_loss=%.4f", last_loss);
        ipc_emit(buf);
    }

    return 0;
}

// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE_QLORA)) {
        return 1;
    }

    if (!params.grpo_mode && params.train_file.empty()) {
        LOG_ERR("%s: --train-file is required (or use --grpo-mode for IPC training)\n", __func__);
        return 1;
    }

    checkpoint_state resume_state;
    const bool resume_requested = !params.lora_resume.empty();
    if (resume_requested) {
        if (!params.lora_adapters.empty()) {
            LOG_ERR("%s: --resume cannot be combined with --lora\n", __func__);
            return 1;
        }
        const std::string mode = params.grpo_mode ? "grpo" : "sft";
        if (!load_checkpoint_state(params.lora_resume, mode, resume_state)) return 1;
        if (mode == "grpo" && resume_state.step > INT32_MAX) {
            LOG_ERR("%s: checkpoint step is too large\n", __func__);
            return 1;
        }
        common_adapter_lora_info adapter_info;
        adapter_info.path  = params.lora_resume;
        adapter_info.scale = 1.0f;
        params.lora_adapters.push_back(adapter_info);
    }

    // Force settings required for training
    params.load_mode    = LLAMA_LOAD_MODE_NONE;
    params.cache_type_k = GGML_TYPE_F32;
    params.cache_type_v = GGML_TYPE_F32;
    // Warmup runs inference with PARAM-flagged tensors which causes a segfault;
    // training never benefits from warmup, so disable it unconditionally.
    params.warmup       = false;
    // Flash attention has no backward implementation; force standard attention for training.
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    float lora_alpha = (params.lora_alpha > 0.0f)
        ? params.lora_alpha : (float) params.lora_rank;
    if (resume_requested && resume_state.has_alpha) {
        if (params.lora_alpha > 0.0f && std::abs(params.lora_alpha - resume_state.alpha) > 1e-6f) {
            LOG_WRN("%s: ignoring --lora-alpha %.3f; checkpoint alpha is %.3f\n",
                    __func__, (double) params.lora_alpha, (double) resume_state.alpha);
        }
        lora_alpha = resume_state.alpha;
    }
    const auto targets = split_csv(params.lora_targets);

    // --- Step 1: Discover tensor shapes from model GGUF (no model load yet) ---
    std::string arch;
    {
        struct ggml_context * ctx_meta = nullptr;
        struct gguf_init_params gp = { true, &ctx_meta };
        struct gguf_context * ctx_gguf = gguf_init_from_file(params.model.path.c_str(), gp);
        if (!ctx_gguf) { LOG_ERR("failed to open model GGUF\n"); return 1; }
        int kid = gguf_find_key(ctx_gguf, "general.architecture");
        if (kid >= 0) arch = gguf_get_val_str(ctx_gguf, kid);
        gguf_free(ctx_gguf);
        ggml_free(ctx_meta);
    }

    // --- Step 2: Allocate LoRA tensors and save initial adapter GGUF ---
    // If the user already supplied a --lora adapter we reuse it (resume training).
    // Otherwise we allocate fresh tensors (B=0, A=random), write them to a temp
    // .init.gguf so common_init_from_params can load them before context creation
    // (this makes sched_reserve size the graph to include LoRA nodes).
    const bool resume_from_lora = !params.lora_adapters.empty();

    std::mt19937 rng(42);
    lora_tensors lt; // will be populated after context load (Step 4)
    std::string init_adapter_path;

    if (!resume_from_lora) {
        lt = alloc_lora_tensors(params.model.path, targets, params.lora_rank, rng, params.lora_freeze_layers);
        if (lt.ab.empty()) return 1;

        init_adapter_path = params.lora_out + ".init.gguf";
        save_adapter(lt, init_adapter_path, arch, lora_alpha, params.model.path);

        // Register adapter so common_init_from_params loads it before context creation
        common_adapter_lora_info adapter_info;
        adapter_info.path  = init_adapter_path;
        adapter_info.scale = 1.0f;
        params.lora_adapters.push_back(adapter_info);
    } else {
        LOG_INF("%s: loading existing LoRA adapter: %s\n",
                __func__, params.lora_adapters.back().path.c_str());
    }

    // --- Step 3: Load model + context (graph sized with LoRA nodes) ---
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model) { LOG_ERR("failed to load model\n"); return 1; }

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    // Arch fallback if not in GGUF metadata
    if (arch.empty()) {
        char buf[256] = {};
        llama_model_desc(model, buf, sizeof(buf));
        arch = std::string(buf);
        arch = arch.substr(0, arch.find_first_of(" /"));
    }

    // --- Step 4: Mark the loaded adapter tensors as trainable ---
    // common_init_from_params loaded the adapter; params.lora_adapters[back].ptr
    // points to the live llama_adapter_lora with its own tensor copies in device
    // memory. Mark those tensors trainable so the optimizer graph includes them.
    {
        llama_adapter_lora * loaded = params.lora_adapters.back().ptr;
        if (!loaded) {
            LOG_ERR("%s: adapter was not loaded by common_init_from_params\n", __func__);
            return 1;
        }
        for (auto & kv : loaded->ab_map) {
            ggml_set_param(kv.second.a);  // lora_a → trainable
            ggml_set_param(kv.second.b);  // lora_b → trainable
        }
        // Point lt.ab at the live device tensors so save_adapter writes
        // the trained weights (not the original init tensors).
        lt.ab.clear();
        for (auto & kv : loaded->ab_map) {
            lt.ab[kv.first] = {kv.second.a, kv.second.b};
        }
    }

    // Remove temp init file when we created it (resume path has no init file)
    if (!resume_from_lora && !init_adapter_path.empty()) {
        std::remove(expand_tilde(init_adapter_path).c_str());
    }

    // --- Step 5: Load dataset ---
    // In GRPO mode the dataset comes from Python via stdin/stdout — skip file loading.
    auto tmpls = common_chat_templates_init(model, "");
    if (params.grpo_mode) {
        int rc = run_grpo_mode(params, model, ctx, lt, arch, lora_alpha, params.model.path,
                               resume_requested ? &resume_state : nullptr);
        if (lt.buf) ggml_backend_buffer_free(lt.buf);
        if (lt.ctx) ggml_free(lt.ctx);
        llama_backend_free();
        return rc;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto samples = load_jsonl(params.train_file, model, vocab, tmpls.get(), params.dataset_threads,
                              params.critical_token_mode, params.critical_token_weight);
    if (samples.empty()) {
        LOG_ERR("%s: no training samples loaded\n", __func__);
        return 1;
    }

    const int32_t n_ctx = llama_n_ctx(ctx);
    std::vector<float> window_rewards;
    const llama_token bos = llama_vocab_bos(llama_model_get_vocab(model));
    const bool critical_enabled = params.critical_token_mode != "none";
    std::vector<llama_opt_critical_token_metadata> critical_metadata;
    auto dataset = build_dataset(samples, n_ctx, window_rewards, params.train_on_prompt, bos, critical_enabled, &critical_metadata);
    if (!dataset) return 1;

    if (critical_enabled) {
        LOG_INF("critical_sft: mode=%s token_weight=%.6f confidence_threshold=%.6f weight_shape=%s warmup_steps=%d max_fraction=%.6f stats_every=%d\n",
                params.critical_token_mode.c_str(), (double) params.critical_token_weight,
                (double) params.critical_confidence_threshold, params.critical_weight_shape.c_str(),
                params.critical_warmup_steps, (double) params.critical_max_fraction, params.critical_stats_every);
    }

    qlora_lr_schedule schedule {
        &params.lr, params.lr_scheduler, params.warmup_steps, params.warmup_init_ratio, 0, 0
    };

    // Initialize optimizer - our custom param filter restricts training to lora_a/b.
    struct llama_opt_params lopt_params {
        /*.n_ctx_train              =*/0,
        /*.param_filter             =*/lora_param_filter,
        /*.param_filter_ud          =*/nullptr,
        /*.get_opt_pars             =*/qlora_opt_lr_pars,
        /*.get_opt_pars_ud          =*/&schedule,
        /*.optimizer_type           =*/params.optimizer,
        /*.lora_qat_type            =*/lora_qat_type_from_string(params.lora_qat),
        /*.grad_checkpoint_interval =*/params.grad_checkpoint_interval,
        /*.critical_token_mode      =*/critical_token_mode_from_string(params.critical_token_mode),
        /*.critical_token_weight    =*/params.critical_token_weight,
        /*.critical_confidence_threshold =*/params.critical_confidence_threshold,
        /*.critical_weight_shape    =*/critical_weight_shape_from_string(params.critical_weight_shape),
        /*.critical_warmup_steps    =*/params.critical_warmup_steps,
        /*.critical_max_fraction    =*/params.critical_max_fraction,
        /*.critical_step            =*/&schedule.step,
        /*.critical_stats_every     =*/params.critical_stats_every,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t ndata = ggml_opt_dataset_ndata(dataset);
    const int64_t idata_split = train_split_from_val_fraction(ndata, params.val_split);
    if (idata_split <= 0) {
        LOG_ERR("%s: no training windows after val split (ndata=%ld val_split=%.3f)\n",
                __func__, (long) ndata, (double) params.val_split);
        return 1;
    }
    if (params.val_split > 0.0f && idata_split == ndata) {
        LOG_WRN("%s: validation split skipped because dataset has only %ld training window(s)\n",
                __func__, (long) ndata);
    }
    if (params.lr.epochs > 0 && idata_split > INT64_MAX / params.lr.epochs) {
        LOG_ERR("%s: total training step count overflows int64\n", __func__);
        return 1;
    }
    schedule.total_steps = idata_split * params.lr.epochs;
    if (schedule.total_steps <= 0) {
        LOG_ERR("%s: total training step count must be positive\n", __func__);
        return 1;
    }
    if (params.warmup_steps > schedule.total_steps) {
        LOG_ERR("%s: --warmup-steps %d exceeds total training steps %ld\n",
                __func__, params.warmup_steps, (long) schedule.total_steps);
        return 1;
    }
    if (params.lr.lr0 <= 0.0f || (params.lr_scheduler == "cosine" && params.lr.lr_min > params.lr.lr0)) {
        LOG_ERR("%s: learning-rate must be positive and cosine requires learning-rate-min <= learning-rate\n", __func__);
        return 1;
    }
    const auto train_reward_end = window_rewards.begin() + idata_split;
    const bool all_train_rewards_zero = std::all_of(
            window_rewards.begin(), train_reward_end, [](float r) { return std::abs(r) <= 1e-6f; });
    if (all_train_rewards_zero) {
        std::fill(window_rewards.begin(), train_reward_end, 1.0f);
        if (critical_enabled) {
            for (int64_t i = 0; i < idata_split; ++i) {
                for (int32_t j = 0; j < n_ctx; ++j) {
                    llama_opt_critical_token_metadata & metadata = critical_metadata[i * n_ctx + j];
                    if (metadata.span_weight > 0.0f) {
                        metadata.reward_weight = 1.0f;
                    }
                }
            }
            ggml_opt_dataset_set_aux(dataset, critical_metadata.data(), n_ctx * sizeof(critical_metadata[0]));
        }
        LOG_WRN("%s: all training window rewards normalized to zero; using weight 1.0 for the small training split\n",
                __func__);
    }
    critical_metadata.clear();
    critical_metadata.shrink_to_fit();

    const bool has_rewards = std::any_of(window_rewards.begin(), window_rewards.end(),
                                         [](float r){ return std::abs(r - 1.0f) > 1e-4f; });
    if (has_rewards && !critical_enabled) {
        LOG_INF("%s: reward-weighted SFT enabled (found non-uniform rewards in dataset)\n", __func__);
        llama_opt_set_reward_weights(window_rewards.data(), (int64_t)window_rewards.size());
    }

    unsigned epoch_start = 0;
    int64_t window_start = 0;
    if (resume_requested) {
        if ((uint64_t) resume_state.epoch > UINT32_MAX) {
            LOG_ERR("%s: checkpoint epoch is too large\n", __func__);
            return 1;
        }
        epoch_start  = (unsigned) resume_state.epoch - 1;
        window_start = resume_state.window;

        if (resume_state.context_length >= 0 && resume_state.context_length != n_ctx) {
            LOG_ERR("%s: checkpoint context length is %ld, current context length is %d\n",
                    __func__, (long) resume_state.context_length, n_ctx);
            return 1;
        }
        if (resume_state.dataset_windows >= 0 && resume_state.dataset_windows != idata_split) {
            LOG_ERR("%s: checkpoint has %ld training windows, current dataset has %ld\n",
                    __func__, (long) resume_state.dataset_windows, (long) idata_split);
            return 1;
        }
        if (window_start > idata_split) {
            LOG_ERR("%s: checkpoint window %ld exceeds the %ld training windows in this dataset\n",
                    __func__, (long) window_start, (long) idata_split);
            return 1;
        }
        if (resume_state.has_metadata && resume_state.shuffle != params.shuffle_dataset) {
            LOG_ERR("%s: checkpoint shuffle setting does not match --shuffle-dataset\n", __func__);
            return 1;
        }
        if (window_start == idata_split) {
            ++epoch_start;
            window_start = 0;
        }
        LOG_INF("%s: resuming SFT at epoch %u/%u after window %ld; next window is %ld/%ld; optimizer state starts fresh\n",
                __func__, epoch_start + 1, params.lr.epochs, (long) window_start,
                (long) window_start + 1, (long) idata_split);
    }
    if (epoch_start > params.lr.epochs) {
        LOG_ERR("%s: checkpoint epoch %u exceeds target epochs %u\n",
                __func__, epoch_start, params.lr.epochs);
        return 1;
    }
    const int64_t schedule_step = (int64_t) epoch_start * idata_split + window_start;
    if (schedule_step > schedule.total_steps) {
        LOG_ERR("%s: checkpoint step %ld exceeds total training steps %ld\n",
                __func__, (long) schedule_step, (long) schedule.total_steps);
        return 1;
    }
    if (resume_requested && resume_state.schedule_step >= 0 && resume_state.schedule_step != schedule_step) {
        LOG_ERR("%s: checkpoint scheduler step %ld does not match derived step %ld\n",
                __func__, (long) resume_state.schedule_step, (long) schedule_step);
        return 1;
    }
    schedule.step = schedule_step;

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    const int32_t n_ubatch       = llama_n_ubatch(ctx);
    const int32_t ubatch_per_ctx = (n_ubatch > 0) ? (n_ctx / n_ubatch) : 1;

    save_ctx sctx {
        &lt, &params.lora_out, &arch, &params.model.path, lora_alpha,
        params.save_every, ubatch_per_ctx, 0, 0, 0, idata_split, n_ctx,
        params.shuffle_dataset, params.verbose_loss, &schedule
    };
    g_save_ctx = &sctx;

    if (resume_requested && params.shuffle_dataset && epoch_start > 0) {
        for (unsigned epoch = 0; epoch < epoch_start; ++epoch) {
            llama_opt_dataset_shuffle(ctx, dataset, idata_split);
            if (params.optimizer_restart_every > 0 &&
                epoch + 1 < params.lr.epochs &&
                (epoch + 1) % params.optimizer_restart_every == 0) {
                llama_opt_reset(ctx, true);
            }
        }
        LOG_INF("%s: replayed %u completed epoch shuffle(s)\n", __func__, epoch_start);
    }

    const int64_t total_windows = ggml_opt_dataset_ndata(dataset);
    LOG_INF("%s: starting QLoRA training — rank=%d alpha=%.1f epochs=%d loss=%s\n",
            __func__, params.lora_rank, lora_alpha, params.lr.epochs,
            params.train_on_prompt ? "prompt+response" : "response-only");
    llama_synchronize(ctx);
    const adapter_stats adapter_start = adapter_get_stats(lt);
    LOG_INF("%s: adapter_start: total_l2=%.9f a_l2=%.9f b_l2=%.9f a_max=%.9f b_max=%.9f\n",
            __func__, adapter_start.total_l2, adapter_start.a_l2, adapter_start.b_l2,
            adapter_start.a_max_abs, adapter_start.b_max_abs);
    LOG_INF("%s: dataset: %ld windows × %d ubatches = %ld steps per epoch  (n_ctx=%d n_ubatch=%d stride=%d)\n",
            __func__, (long)total_windows, ubatch_per_ctx, (long)(idata_split * ubatch_per_ctx),
            n_ctx, n_ubatch, n_ctx / 2);
    LOG_INF("%s: lr scheduler=%s warmup_steps=%d warmup_init_ratio=%.3g total_steps=%ld start_step=%ld lr=%.3g lr_min=%.3g\n",
            __func__, params.lr_scheduler.c_str(), params.warmup_steps, (double) params.warmup_init_ratio,
            (long) schedule.total_steps,
            (long) schedule.step, (double) params.lr.lr0, (double) std::max(0.0f, params.lr.lr_min));
    if (params.save_every > 0) {
        LOG_INF("%s: will save checkpoint every %d windows → %s.ckptN.gguf\n",
                __func__, params.save_every, params.lora_out.c_str());
    }

    ggml_opt_epoch_callback cb_train = save_every_callback;

    for (params.lr.epoch = epoch_start; params.lr.epoch < params.lr.epochs; ++params.lr.epoch) {
        const int64_t idata_start = params.lr.epoch == epoch_start ? window_start : 0;
        sctx.window_offset = idata_start;
        sctx.last_saved = idata_start;
        sctx.epoch = params.lr.epoch + 1;
        sctx.loss_cumulative_sum = 0.0;
        sctx.loss_cumulative_count = 0;
        sctx.loss_epoch_sum = 0.0;
        sctx.loss_epoch_count = 0;
        llama_opt_epoch_range(ctx, dataset, result_train, result_eval, idata_start, idata_split,
                              cb_train,
                              ggml_opt_epoch_callback_progress_bar,
                              params.shuffle_dataset);
        fprintf(stderr, "\n");

        // Per-epoch loss summary
        {
            double train_loss = 0.0, train_unc = 0.0;
            ggml_opt_result_loss(result_train, &train_loss, &train_unc);
            if (idata_split < ggml_opt_dataset_ndata(dataset)) {
                double val_loss = 0.0, val_unc = 0.0;
                ggml_opt_result_loss(result_eval, &val_loss, &val_unc);
                LOG_INF("epoch %d/%d: train_loss=%.4f ± %.4f  val_loss=%.4f ± %.4f\n",
                        params.lr.epoch + 1, params.lr.epochs, train_loss, train_unc, val_loss, val_unc);
            } else {
                LOG_INF("epoch %d/%d: train_loss=%.4f ± %.4f\n",
                        params.lr.epoch + 1, params.lr.epochs, train_loss, train_unc);
            }
            llama_synchronize(ctx);
            const adapter_stats adapter_cur = adapter_get_stats(lt);
            LOG_INF("epoch %d/%d: adapter: total_l2=%.9f delta=%.9f a_l2=%.9f b_l2=%.9f a_max=%.9f b_max=%.9f\n",
                    params.lr.epoch + 1, params.lr.epochs,
                    adapter_cur.total_l2, adapter_cur.total_l2 - adapter_start.total_l2,
                    adapter_cur.a_l2, adapter_cur.b_l2,
                    adapter_cur.a_max_abs, adapter_cur.b_max_abs);
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);

        if (params.optimizer_restart_every > 0 &&
            params.lr.epoch + 1 < params.lr.epochs &&
            (params.lr.epoch + 1) % params.optimizer_restart_every == 0) {
            LOG_INF("%s: resetting optimizer state after epoch %d\n", __func__, params.lr.epoch + 1);
            llama_opt_reset(ctx, true);
        }
    }

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);
    llama_opt_set_reward_weights(nullptr, 0);

    // Save final trained adapter
    save_adapter(lt, params.lora_out, arch, lora_alpha, params.model.path);

    // Free scratch buffers only when we allocated them (not in resume path)
    if (lt.buf) ggml_backend_buffer_free(lt.buf);
    if (lt.ctx) ggml_free(lt.ctx);
    ggml_opt_dataset_free(dataset);
    llama_backend_free();

    return 0;
}
