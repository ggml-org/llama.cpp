// Minimal OpenAI-compatible /v1/chat/completions server backed by the tiered
// CUDA backend, for MoE models that need llama_tiered_model_load_from_file
// instead of a plain llama_model_load_from_file. tools/server/ does not know
// about llama-tiered.h, so this is a separate, much smaller server rather
// than a mode of that one: one model, one context, requests processed
// sequentially (no parallel slots).

#include "llama-tiered.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

constexpr uint64_t MiB = 1024ull * 1024ull;

void print_usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s -m MODEL.gguf --dram-mib N [--vram-mib N] [--cache-mib N] "
            "[--reserve-mib N] [-c CTX] [--host HOST] [--port PORT] [--alias NAME]\n",
            argv0);
}

std::string apply_chat_template(
        const llama_model * model,
        const std::vector<llama_chat_message> & chat,
        bool add_assistant) {
    const char * tmpl = llama_model_chat_template(model, nullptr);
    std::vector<char> buf(2048);
    int32_t needed = llama_chat_apply_template(
            tmpl, chat.data(), chat.size(), add_assistant, buf.data(), static_cast<int32_t>(buf.size()));
    if (needed < 0) {
        throw std::runtime_error("this model's chat template is not supported");
    }
    if (static_cast<size_t>(needed) > buf.size()) {
        buf.resize(static_cast<size_t>(needed));
        needed = llama_chat_apply_template(
                tmpl, chat.data(), chat.size(), add_assistant, buf.data(), static_cast<int32_t>(buf.size()));
    }
    return std::string(buf.data(), static_cast<size_t>(needed));
}

// Owns the one live conversation this process serves. A request that does not
// continue the previous one (different history prefix) resets the KV cache
// and restarts from scratch, same as any other single-slot server.
struct server_state {
    llama_context * context = nullptr;
    const llama_model * model = nullptr;
    llama_sampler * sampler = nullptr;
    int n_predict_default = 512;

    std::mutex mutex;
    std::vector<llama_chat_message> history;
    std::deque<std::string> owned_content;
    std::string formatted_prev;
    int position = 0;

    void reset_conversation() {
        llama_memory_clear(llama_get_memory(context), true);
        history.clear();
        owned_content.clear();
        formatted_prev.clear();
        position = 0;
    }
};

json error_json(const std::string & message, int code) {
    return json{
        {"error", {
            {"message", message},
            {"type", "invalid_request_error"},
            {"code", code},
        }},
    };
}

void handle_chat_completions(server_state & st, const httplib::Request & req, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(st.mutex);

    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception & error) {
        res.status = 400;
        res.set_content(error_json(std::string("invalid JSON: ") + error.what(), 400).dump(), "application/json");
        return;
    }

    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
        res.status = 400;
        res.set_content(error_json("messages is required", 400).dump(), "application/json");
        return;
    }

    // Earlier messages must match what is already in this process's history
    // so the existing KV cache can be reused; a mismatch (new session, edited
    // history) restarts the conversation. deque never invalidates existing
    // element references on push_back, so pointers into owned_content handed
    // to earlier llama_chat_message entries stay valid as more are appended.
    const auto & messages = body["messages"];
    bool prefix_matches = messages.size() > st.history.size();
    for (size_t i = 0; prefix_matches && i < st.history.size(); ++i) {
        const std::string role = messages[i].value("role", "");
        const std::string content = messages[i].value("content", "");
        if (role != st.history[i].role || content != st.history[i].content) {
            prefix_matches = false;
        }
    }
    if (!prefix_matches) {
        st.reset_conversation();
    }

    auto push_message = [&](const json & m) {
        st.owned_content.push_back(m.value("content", ""));
        const std::string & content_ref = st.owned_content.back();
        st.owned_content.push_back(m.value("role", "user"));
        const std::string & role_ref = st.owned_content.back();
        st.history.push_back({ role_ref.c_str(), content_ref.c_str() });
    };
    for (size_t i = st.history.size(); i < messages.size(); ++i) {
        push_message(messages[i]);
    }

    const int n_predict = body.value("max_tokens", st.n_predict_default);

    std::string formatted;
    try {
        formatted = apply_chat_template(st.model, st.history, true);
    } catch (const std::exception & error) {
        res.status = 400;
        res.set_content(error_json(error.what(), 400).dump(), "application/json");
        st.history.pop_back();
        return;
    }

    const std::string turn = formatted.substr(st.formatted_prev.size());
    const llama_vocab * vocab = llama_model_get_vocab(st.model);

    const int n_turn = -llama_tokenize(vocab, turn.c_str(), turn.size(), nullptr, 0, false, true);
    if (n_turn <= 0) {
        res.status = 500;
        res.set_content(error_json("failed to tokenize turn", 500).dump(), "application/json");
        return;
    }
    std::vector<llama_token> turn_tokens(static_cast<size_t>(n_turn));
    llama_tokenize(vocab, turn.c_str(), turn.size(), turn_tokens.data(), turn_tokens.size(), false, true);

    if (st.position + n_turn + n_predict > static_cast<int>(llama_n_ctx(st.context))) {
        res.status = 400;
        res.set_content(error_json("context window full for this conversation", 400).dump(), "application/json");
        return;
    }

    // Streaming (SSE) is not implemented; stream:true still gets a complete,
    // non-chunked response.
    llama_batch batch = llama_batch_get_one(turn_tokens.data(), static_cast<int32_t>(turn_tokens.size()));
    std::string reply;
    std::string finish_reason = "stop";
    int n_generated = 0;

    const int64_t t_gen_start = ggml_time_us();
    for (; n_generated < n_predict; ++n_generated) {
        if (llama_decode(st.context, batch) != 0) {
            res.status = 500;
            res.set_content(error_json("decode failed", 500).dump(), "application/json");
            return;
        }
        st.position += batch.n_tokens;

        llama_token token = llama_sampler_sample(st.sampler, st.context, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        char piece[256];
        const int length = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
        if (length > 0) {
            reply.append(piece, static_cast<size_t>(length));
        }
        if (n_generated + 1 == n_predict) {
            finish_reason = "length";
        }
        batch = llama_batch_get_one(&token, 1);
    }
    const double gen_seconds = (ggml_time_us() - t_gen_start) / 1e6;
    const double tokens_per_second = n_generated > 0 ? n_generated / gen_seconds : 0.0;

    st.owned_content.push_back(reply);
    st.history.push_back({ "assistant", st.owned_content.back().c_str() });
    st.formatted_prev = apply_chat_template(st.model, st.history, false);

    json response = {
        {"id", "chatcmpl-tiered"},
        {"object", "chat.completion"},
        {"model", body.value("model", "tiered")},
        {"choices", json::array({{
            {"index", 0},
            {"message", {{"role", "assistant"}, {"content", reply}}},
            {"finish_reason", finish_reason},
        }})},
        {"usage", {
            {"prompt_tokens", n_turn},
            {"completion_tokens", n_generated},
            {"tokens_per_second", tokens_per_second},
        }},
    };
    res.set_content(response.dump(), "application/json");
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string host = "127.0.0.1";
    std::string alias = "tiered";
    int port = 8081;
    int n_ctx_arg = 8192;
    int n_predict_default = 512;

    llama_tiered_memory_params tiered_params = llama_tiered_memory_default_params();

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char * option) -> const char * {
            if (++i >= argc) {
                std::fprintf(stderr, "missing value for %s\n", option);
                print_usage(argv[0]);
                std::exit(1);
            }
            return argv[i];
        };
        auto parse_mib = [&](const char * option) -> uint64_t {
            const uint64_t value = std::stoull(require_value(option));
            if (value > std::numeric_limits<uint64_t>::max() / MiB) {
                throw std::out_of_range("MiB value is too large");
            }
            return value * MiB;
        };

        try {
            if (arg == "-m" || arg == "--model") {
                model_path = require_value(arg.c_str());
            } else if (arg == "--vram-mib") {
                tiered_params.vram_budget_bytes = parse_mib(arg.c_str());
            } else if (arg == "--dram-mib") {
                tiered_params.dram_budget_bytes = parse_mib(arg.c_str());
            } else if (arg == "--cache-mib") {
                tiered_params.ssd_cache_bytes = parse_mib(arg.c_str());
            } else if (arg == "--reserve-mib") {
                tiered_params.vram_reserve_bytes = parse_mib(arg.c_str());
            } else if (arg == "--main-gpu") {
                tiered_params.main_gpu = std::stoi(require_value(arg.c_str()));
            } else if (arg == "-c" || arg == "--ctx-size") {
                n_ctx_arg = std::stoi(require_value(arg.c_str()));
            } else if (arg == "-n" || arg == "--max-tokens") {
                n_predict_default = std::stoi(require_value(arg.c_str()));
            } else if (arg == "--host") {
                host = require_value(arg.c_str());
            } else if (arg == "--port") {
                port = std::stoi(require_value(arg.c_str()));
            } else if (arg == "--alias") {
                alias = require_value(arg.c_str());
            } else if (arg == "-h" || arg == "--help") {
                print_usage(argv[0]);
                return 0;
            } else {
                std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
                print_usage(argv[0]);
                return 1;
            }
        } catch (const std::exception & error) {
            std::fprintf(stderr, "invalid %s value: %s\n", arg.c_str(), error.what());
            return 1;
        }
    }

    if (model_path.empty() || tiered_params.dram_budget_bytes == 0) {
        print_usage(argv[0]);
        return 1;
    }

    llama_model_params model_params = llama_model_default_params();
    llama_tiered_model * tiered = llama_tiered_model_load_from_file(
            model_path.c_str(), model_params, tiered_params);
    if (!tiered) {
        std::fprintf(stderr, "tiered model load failed: %s\n", llama_tiered_last_error());
        return 1;
    }

    llama_model * model = llama_tiered_model_get_model(tiered);
    const llama_tiered_memory_stats * stats = llama_tiered_model_get_stats(tiered);
    std::fprintf(stderr,
            "tiered weights: VRAM %.2f MiB, DRAM %.2f MiB, SSD %.2f MiB (%u streamed tensors)\n",
            stats->vram_bytes / 1024.0 / 1024.0,
            stats->dram_bytes / 1024.0 / 1024.0,
            stats->ssd_bytes / 1024.0 / 1024.0,
            stats->ssd_tensor_count);

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<uint32_t>(n_ctx_arg);
    context_params.n_batch = static_cast<uint32_t>(n_ctx_arg);
    context_params.no_perf = false;

    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) {
        std::fprintf(stderr, "failed to create context\n");
        llama_tiered_model_free(tiered);
        return 1;
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    server_state state;
    state.context = context;
    state.model = model;
    state.sampler = sampler;
    state.n_predict_default = n_predict_default;

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        json response = {
            {"object", "list"},
            {"data", json::array({{
                {"id", alias},
                {"object", "model"},
                {"owned_by", "summer.cpp"},
            }})},
        };
        res.set_content(response.dump(), "application/json");
    });

    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            handle_chat_completions(state, req, res);
        } catch (const std::exception & error) {
            res.status = 500;
            res.set_content(error_json(error.what(), 500).dump(), "application/json");
        }
    });

    std::fprintf(stderr, "summer-server: listening on http://%s:%d (alias: %s)\n",
            host.c_str(), port, alias.c_str());
    if (!svr.listen(host.c_str(), port)) {
        std::fprintf(stderr, "summer-server: failed to bind %s:%d\n", host.c_str(), port);
    }

    llama_sampler_free(sampler);
    llama_free(context);
    llama_tiered_model_free(tiered);
    return 0;
}
