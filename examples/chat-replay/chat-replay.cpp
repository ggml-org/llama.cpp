// Replays prompts through a chat session and reports GenAI chat.py-style perf metrics.
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

using clk = std::chrono::steady_clock;

#define PM "\xC2\xB1"  // UTF-8 plus-minus sign, independent of the compiler's source charset

static double ms_between(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-p prompt | -pf prompts.jsonl|prompt.txt] [-i] [-pc] [-o max_new_tokens]\n", argv[0]);
    printf("        [-c context_size] [-b n_batch] [-ub n_ubatch] [-fa 0|1] [-ngl n_gpu_layers] [-v]\n");
    printf("\n");
}

static std::string trim(const std::string & s) {
    const char * ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) {
        return "";
    }
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

static std::vector<std::string> read_prompt_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open prompt file: " + path);
    }
    std::vector<std::string> prompts;
    const bool is_jsonl = path.size() >= 6 && path.compare(path.size() - 6, 6, ".jsonl") == 0;
    if (!is_jsonl) {
        std::stringstream ss;
        ss << in.rdbuf();
        prompts.push_back(ss.str());
        return prompts;
    }
    std::string line;
    int line_number = 0;
    while (std::getline(in, line)) {
        line_number++;
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        const auto data = nlohmann::json::parse(line);
        if (!data.is_object() || !data.contains("prompt") || !data["prompt"].is_string()) {
            throw std::runtime_error("expected string field 'prompt' in " + path + ":" + std::to_string(line_number));
        }
        const std::string prompt = trim(data["prompt"].get<std::string>());
        if (prompt.empty()) {
            throw std::runtime_error("prompt cannot be empty in " + path + ":" + std::to_string(line_number));
        }
        prompts.push_back(prompt);
    }
    if (prompts.empty()) {
        throw std::runtime_error("no prompts found in JSONL file: " + path);
    }
    return prompts;
}

static std::pair<double, double> mean_std(const std::vector<double> & v) {
    if (v.empty()) {
        return {0.0, 0.0};
    }
    double sum = 0.0;
    for (double x : v) {
        sum += x;
    }
    const double mean = sum / v.size();
    double var = 0.0;
    for (double x : v) {
        var += (x - mean) * (x - mean);
    }
    return {mean, std::sqrt(var / v.size())};
}

struct chat_state {
    const char *                                     tmpl;
    std::vector<std::pair<std::string, std::string>> messages;  // role, content
    int                                              prev_len = 0;

    std::string format(bool add_ass) const {
        std::vector<llama_chat_message> msgs;
        for (const auto & m : messages) {
            msgs.push_back({m.first.c_str(), m.second.c_str()});
        }
        std::vector<char> buf(4096);
        int len = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_ass, buf.data(), buf.size());
        if (len > (int) buf.size()) {
            buf.resize(len);
            len = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_ass, buf.data(), buf.size());
        }
        if (len < 0) {
            throw std::runtime_error("failed to apply the chat template");
        }
        return std::string(buf.data(), len);
    }
};

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    const int n = -llama_tokenize(vocab, text.c_str(), text.size(), NULL, 0, add_special, true);
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_special, true) < 0) {
        throw std::runtime_error("failed to tokenize the prompt");
    }
    return tokens;
}

// llama_decode rejects batches larger than n_batch, so long prompts are submitted in n_batch pieces.
static bool decode_prompt(llama_context * ctx, std::vector<llama_token> & tokens) {
    const int n_batch = (int) llama_n_batch(ctx);
    for (size_t i = 0; i < tokens.size(); i += n_batch) {
        const int n = std::min<int>(n_batch, (int) (tokens.size() - i));
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, n)) != 0) {
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    const auto t_start = clk::now();
    std::setlocale(LC_NUMERIC, "C");
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::string model_path;
    std::string prompt;
    std::string prompt_file;
    bool        interactive    = false;
    bool        report_perf    = false;
    bool        verbose        = false;
    int         max_new_tokens = 128;
    int         n_ctx          = 8320;
    int         n_batch        = 3072;
    int         n_ubatch       = 3072;
    int         flash_attn     = 1;
    int         ngl            = 99;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for " + arg);
            }
            return argv[++i];
        };
        try {
            if (arg == "-m") {
                model_path = next();
            } else if (arg == "-p" || arg == "--prompt") {
                prompt = next();
            } else if (arg == "-pf" || arg == "--prompt_file") {
                prompt_file = next();
            } else if (arg == "-i" || arg == "--interactive") {
                interactive = true;
            } else if (arg == "-pc" || arg == "--perf") {
                report_perf = true;
            } else if (arg == "-o" || arg == "--max_new_tokens") {
                max_new_tokens = std::stoi(next());
            } else if (arg == "-c") {
                n_ctx = std::stoi(next());
            } else if (arg == "-b") {
                n_batch = std::stoi(next());
            } else if (arg == "-ub") {
                n_ubatch = std::stoi(next());
            } else if (arg == "-fa") {
                flash_attn = std::stoi(next());
            } else if (arg == "-ngl") {
                ngl = std::stoi(next());
            } else if (arg == "-v" || arg == "--verbose") {
                verbose = true;
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } catch (std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            print_usage(argc, argv);
            return 1;
        }
    }
    if (model_path.empty() || (!prompt.empty() && !prompt_file.empty())) {
        print_usage(argc, argv);
        return 1;
    }

    std::vector<std::string> replay;
    try {
        if (!prompt_file.empty()) {
            replay = read_prompt_file(prompt_file);
        } else if (!prompt.empty()) {
            replay.push_back(prompt);
        }
    } catch (std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    if (!verbose) {
        llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
            if (level >= GGML_LOG_LEVEL_ERROR) {
                fprintf(stderr, "%s", text);
            }
        }, nullptr);
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = n_ctx;
    ctx_params.n_batch         = n_batch;
    ctx_params.n_ubatch        = n_ubatch;
    ctx_params.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return 1;
    }
    llama_memory_t mem = llama_get_memory(ctx);

    // GenAI's default generation config is greedy.
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    chat_state chat{llama_model_chat_template(model, /* name */ nullptr)};

    // Warm up with the first prompt so the prefill and decode models are compiled/imported inside Load time,
    // like GenAI does in the pipeline constructor; prefill graphs are keyed by the prompt's KV-slice size.
    double warmup_ms = 0.0;
    if (!replay.empty()) {
        const auto t0 = clk::now();
        try {
            chat_state warm{chat.tmpl, {{"user", replay.front()}}};
            auto tokens = tokenize(vocab, warm.format(true), true);
            if ((int) tokens.size() + 1 > (int) llama_n_ctx(ctx) || !decode_prompt(ctx, tokens)) {
                throw std::runtime_error("warmup prefill failed");
            }
            llama_token tok = llama_sampler_sample(smpl, ctx, -1);
            if (llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0) {
                throw std::runtime_error("warmup decode failed");
            }
        } catch (std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            return 1;
        }
        // Metadata-only clear like llama-bench: zero-filling the ~1 GiB KV buffer slows the next NPU prefill.
        llama_memory_clear(mem, false);
        llama_sampler_reset(smpl);
        warmup_ms = ms_between(t0, clk::now());
    }
    const double load_ms = ms_between(t_start, clk::now());

    if (interactive) {
        printf("Chat history is enabled!\n");
    }

    auto infer = [&](const std::string & user) {
        if (!interactive) {
            chat.messages.clear();
            chat.prev_len = 0;
            llama_memory_clear(mem, false);
        }
        llama_sampler_reset(smpl);

        const auto t0 = clk::now();
        chat.messages.push_back({"user", user});
        const std::string formatted = chat.format(true);
        const std::string turn      = formatted.substr(chat.prev_len);

        const bool is_first = llama_memory_seq_pos_max(mem, 0) == -1;
        const auto t_tok0   = clk::now();
        auto       tokens   = tokenize(vocab, turn, is_first);
        const double tok_ms = ms_between(t_tok0, clk::now());

        const int n_past = llama_memory_seq_pos_max(mem, 0) + 1;
        if (n_past + (int) tokens.size() + 1 > (int) llama_n_ctx(ctx)) {
            fprintf(stderr, "context size exceeded\n");
            exit(1);
        }
        if (!decode_prompt(ctx, tokens)) {
            fprintf(stderr, "failed to decode the prompt\n");
            exit(1);
        }

        llama_token tok    = llama_sampler_sample(smpl, ctx, -1);
        auto        t_prev = clk::now();
        const double ttft_ms = ms_between(t0, t_prev);

        std::string         response;
        std::vector<double> tpot;
        double              detok_ms = 0.0;
        int                 n_gen    = 0;
        while (!llama_vocab_is_eog(vocab, tok)) {
            const auto t_d0 = clk::now();
            char       buf[256];
            const int  n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "failed to convert token to piece\n");
                exit(1);
            }
            detok_ms += ms_between(t_d0, clk::now());
            response.append(buf, n);
            fwrite(buf, 1, n, stdout);
            fflush(stdout);
            n_gen++;

            if (max_new_tokens > 0 && n_gen >= max_new_tokens) {
                break;
            }
            if (llama_memory_seq_pos_max(mem, 0) + 2 > (int) llama_n_ctx(ctx)) {
                fprintf(stderr, "\ncontext size exceeded\n");
                break;
            }
            if (llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0) {
                fprintf(stderr, "failed to decode\n");
                exit(1);
            }
            tok = llama_sampler_sample(smpl, ctx, -1);
            const auto t = clk::now();
            tpot.push_back(ms_between(t_prev, t));
            t_prev = t;
        }
        const double gen_ms = ms_between(t0, clk::now());

        chat.messages.push_back({"assistant", response});
        chat.prev_len = (int) chat.format(false).size();

        if (report_perf) {
            const auto [tpot_mean, tpot_std] = mean_std(tpot);
            const double tput     = tpot_mean > 0 ? 1000.0 / tpot_mean : 0.0;
            const double tput_std = tpot_mean > 0 ? tput * tpot_std / tpot_mean : 0.0;
            const int    n_input  = interactive ? n_past + (int) tokens.size() : (int) tokens.size();
            printf("\n\nLoad time: %.2f ms\n", load_ms);
            printf("Warmup time (in load): %.2f ms\n", warmup_ms);
            printf("%s: %d tokens\n", interactive ? "Input token count (accumulated)" : "Input token count", n_input);
            printf("Output token count: %d tokens\n", n_gen);
            printf("Generate time: %.2f " PM " 0.00 ms\n", gen_ms);
            printf("Tokenization time: %.2f " PM " 0.00 ms\n", tok_ms);
            printf("Detokenization time: %.2f " PM " 0.00 ms\n", detok_ms);
            printf("TTFT: %.2f " PM " 0.00 ms\n", ttft_ms);
            printf("TPOT: %.2f " PM " %.2f ms\n", tpot_mean, tpot_std);
            printf("Throughput : %.2f " PM " %.2f tokens/s\n", tput, tput_std);
        }
        printf("\n----------\n");
        fflush(stdout);
    };

    try {
        if (!replay.empty()) {
            for (const auto & p : replay) {
                printf("> %s\n", p.c_str());
                infer(p);
            }
        } else {
            std::string user;
            while (true) {
                printf("> ");
                fflush(stdout);
                if (!std::getline(std::cin, user) || user == "quit") {
                    break;
                }
                infer(user);
            }
        }
    } catch (std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
