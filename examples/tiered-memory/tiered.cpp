#include "llama-tiered.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint64_t MiB = 1024ull * 1024ull;

// Chat mode reprompts after every generated token; DEBUG-level lines like
// "CUDA Graph id N reused" would otherwise fire once per token and bury the
// conversation.
void suppress_debug_log(enum ggml_log_level level, const char * text, void *) {
    if (level != GGML_LOG_LEVEL_DEBUG) {
        std::fputs(text, stderr);
    }
}

void print_usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s -m MODEL.gguf --dram-mib N [--vram-mib N] "
            "[--cache-mib N] [--reserve-mib N] [-n N] [-i] [-c CTX] [PROMPT]\n",
            argv0);
}

// Applies the model's own chat template (falls back to the built-in ChatML
// family if the GGUF does not carry one) and grows the buffer if needed.
std::string apply_chat_template(
        const llama_model * model,
        const std::vector<llama_chat_message> & chat,
        bool add_assistant) {
    const char * tmpl = llama_model_chat_template(model, nullptr);
    std::vector<char> buf(1024);
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

int run_chat(llama_context * context, const llama_model * model, llama_sampler * sampler, int n_predict) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_chat_message> history;
    std::deque<std::string> owned_content; // llama_chat_message.content points into this; deque never invalidates existing element references

    int position = 0;
    std::string line;
    for (;;) {
        std::fprintf(stderr, "\n> ");
        std::fflush(stderr);
        if (!std::getline(std::cin, line)) {
            break;
        }
        if (line.empty()) {
            continue;
        }

        owned_content.push_back(line);
        history.push_back({ "user", owned_content.back().c_str() });

        std::string formatted;
        try {
            formatted = apply_chat_template(model, history, true);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "%s\n", error.what());
            history.pop_back();
            owned_content.pop_back();
            continue;
        }

        // Only the newly formatted suffix (this turn's template wrapper plus
        // the assistant preamble) needs tokenizing and decoding; everything
        // before it is already in the KV cache from prior turns.
        static size_t formatted_prev_size = 0;
        const std::string turn = formatted.substr(formatted_prev_size);

        const int n_turn = -llama_tokenize(vocab, turn.c_str(), turn.size(), nullptr, 0, false, true);
        if (n_turn <= 0) {
            std::fprintf(stderr, "failed to tokenize turn\n");
            break;
        }
        std::vector<llama_token> turn_tokens(static_cast<size_t>(n_turn));
        if (llama_tokenize(vocab, turn.c_str(), turn.size(),
                    turn_tokens.data(), turn_tokens.size(), false, true) < 0) {
            std::fprintf(stderr, "failed to tokenize turn\n");
            break;
        }

        if (position + n_turn + n_predict > static_cast<int>(llama_n_ctx(context))) {
            std::fprintf(stderr, "context window full; start a new run for a fresh conversation\n");
            break;
        }

        llama_batch batch = llama_batch_get_one(turn_tokens.data(), static_cast<int32_t>(turn_tokens.size()));
        std::string reply;
        for (int generated = 0; generated < n_predict; ++generated) {
            if (llama_decode(context, batch) != 0) {
                std::fprintf(stderr, "llama_decode failed\n");
                return 1;
            }
            position += batch.n_tokens;

            llama_token token = llama_sampler_sample(sampler, context, -1);
            if (llama_vocab_is_eog(vocab, token)) {
                break;
            }

            char piece[256];
            const int length = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
            if (length > 0) {
                std::fwrite(piece, 1, static_cast<size_t>(length), stdout);
                std::fflush(stdout);
                reply.append(piece, static_cast<size_t>(length));
            }
            batch = llama_batch_get_one(&token, 1);
        }
        std::fputc('\n', stdout);

        owned_content.push_back(reply);
        history.push_back({ "assistant", owned_content.back().c_str() });
        formatted_prev_size = apply_chat_template(model, history, false).size();
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(suppress_debug_log, nullptr);

    std::string model_path;
    std::string prompt = "Hello, my name is";
    int n_predict = 32;
    int n_ctx_arg = 0;
    bool chat_mode = false;

    llama_tiered_memory_params tiered_params = llama_tiered_memory_default_params();

    int i = 1;
    for (; i < argc; ++i) {
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
            } else if (arg == "-n") {
                n_predict = std::stoi(require_value(arg.c_str()));
            } else if (arg == "-c" || arg == "--ctx-size") {
                n_ctx_arg = std::stoi(require_value(arg.c_str()));
            } else if (arg == "-i" || arg == "--chat") {
                chat_mode = true;
            } else if (arg == "-h" || arg == "--help") {
                print_usage(argv[0]);
                return 0;
            } else {
                break;
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

    if (i < argc) {
        prompt = argv[i++];
        for (; i < argc; ++i) {
            prompt += " ";
            prompt += argv[i];
        }
    }

    if (chat_mode && n_ctx_arg <= 0) {
        n_ctx_arg = 4096;
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
            "tiered weights: VRAM %.2f MiB, DRAM %.2f MiB, SSD %.2f MiB "
            "(%u streamed tensors)\n",
            stats->vram_bytes / 1024.0 / 1024.0,
            stats->dram_bytes / 1024.0 / 1024.0,
            stats->ssd_bytes / 1024.0 / 1024.0,
            stats->ssd_tensor_count);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    int n_prompt = 0;
    std::vector<llama_token> prompt_tokens;
    if (!chat_mode) {
        n_prompt = -llama_tokenize(
                vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
        if (n_prompt <= 0) {
            std::fprintf(stderr, "failed to size prompt tokenization\n");
            llama_tiered_model_free(tiered);
            return 1;
        }

        prompt_tokens.resize(static_cast<size_t>(n_prompt));
        if (llama_tokenize(
                    vocab, prompt.c_str(), prompt.size(),
                    prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
            std::fprintf(stderr, "failed to tokenize prompt\n");
            llama_tiered_model_free(tiered);
            return 1;
        }
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = chat_mode
            ? static_cast<uint32_t>(n_ctx_arg)
            : static_cast<uint32_t>(n_prompt + n_predict);
    context_params.n_batch = chat_mode
            ? static_cast<uint32_t>(n_ctx_arg)
            : static_cast<uint32_t>(n_prompt);
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

    if (chat_mode) {
        std::fprintf(stderr, "chat mode: type a message and press enter, Ctrl-D to quit\n");
        const int status = run_chat(context, model, sampler, n_predict);
        llama_perf_sampler_print(sampler);
        llama_perf_context_print(context);
        llama_sampler_free(sampler);
        llama_free(context);
        llama_tiered_model_free(tiered);
        return status;
    }

    // Do not echo the tokenized prompt. Besides producing duplicate CLI text,
    // token-to-piece conversion can expose model-specific BOS/EOS markers.
    // The caller already owns the prompt and only generated tokens belong on
    // standard output.
    llama_batch batch = llama_batch_get_one(
            prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
    int position = 0;

    while (position + batch.n_tokens < n_prompt + n_predict) {
        if (llama_decode(context, batch) != 0) {
            std::fprintf(stderr, "llama_decode failed\n");
            break;
        }
        position += batch.n_tokens;

        llama_token token = llama_sampler_sample(sampler, context, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        char piece[256];
        const int length = llama_token_to_piece(
                vocab, token, piece, sizeof(piece), 0, true);
        if (length > 0) {
            std::fwrite(piece, 1, static_cast<size_t>(length), stdout);
            std::fflush(stdout);
        }
        batch = llama_batch_get_one(&token, 1);
    }

    std::fputc('\n', stdout);
    llama_perf_sampler_print(sampler);
    llama_perf_context_print(context);

    llama_sampler_free(sampler);
    llama_free(context);
    llama_tiered_model_free(tiered);
    return 0;
}
