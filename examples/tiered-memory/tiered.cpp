#include "llama-tiered.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint64_t MiB = 1024ull * 1024ull;

constexpr const char * LLAMA_ASCII_LOGO = R"(
▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀
)";

void print_banner() {
    std::fputs(LLAMA_ASCII_LOGO, stderr);
    std::fputs("Summer.cpp tiered-memory CLI\n\n", stderr);
}

void print_usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s -m MODEL.gguf --dram-mib N [--vram-mib N] "
            "[--cache-mib N] [--reserve-mib N] [-n N] [PROMPT]\n",
            argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string prompt = "Hello, my name is";
    int n_predict = 32;

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

    print_banner();

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
    const int n_prompt = -llama_tokenize(
            vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        std::fprintf(stderr, "failed to size prompt tokenization\n");
        llama_tiered_model_free(tiered);
        return 1;
    }

    std::vector<llama_token> prompt_tokens(static_cast<size_t>(n_prompt));
    if (llama_tokenize(
                vocab, prompt.c_str(), prompt.size(),
                prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        std::fprintf(stderr, "failed to tokenize prompt\n");
        llama_tiered_model_free(tiered);
        return 1;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<uint32_t>(n_prompt + n_predict);
    context_params.n_batch = static_cast<uint32_t>(n_prompt);
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
