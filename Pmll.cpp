// SPDX-License-Identifier: MIT
// Persistent Memory Logic Loop adapter for llama.cpp
//
// © 2025 Dr. Josef Kurk Edwards & John Trompeter
// Simplified BSD-style—see LICENSE-PMLL.

#include "llama.h"
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace pmll {

using fs = std::filesystem;

struct LoopHook {
    // Override this to inject your own logic each step.
    // Return false to abort generation.
    virtual bool operator()(const std::string& prompt,
                            const std::vector<llama_token>& last_out) = 0;
    virtual ~LoopHook() = default;
};

class Loop {
  public:
    Loop(const std::string& model_path,
         const std::string& state_dir,
         uint32_t           n_ctx      = 4096,
         LoopHook*          user_hook  = nullptr)
        : model_path_(model_path),
          state_dir_(state_dir),
          user_hook_(user_hook) {

        fs::create_directories(state_dir_);

        llama_backend_init();                                     // init ggml backend
        llama_model_params mp = llama_model_default_params();
        model_ = llama_model_load_from_file(model_path_.c_str(), mp);
        if (!model_) throw std::runtime_error("model load failed");

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx;
        ctx_     = llama_init_from_model(model_, cp);
        if (!ctx_) throw std::runtime_error("context init failed");

        mem_ = llama_get_memory(ctx_);                            // unified KV handle
    }

    ~Loop() {
        llama_free(ctx_);
        llama_model_free(model_);
        llama_backend_free();
    }

    /// Generate up to n_predict tokens, persisting state after each decode
    std::string generate(const std::string& prompt,
                         int n_predict = 128,
                         llama_seq_id seq = 0) {
        std::lock_guard<std::mutex> lock(mu_);
        restore(seq);                                             // 1⃣ try resume

        // --- tokenize prompt --------------------------------------------------
        std::vector<llama_token> tokens(prompt.size() + 8);
        int n = llama_tokenize(model_, prompt.c_str(),
                               tokens.data(), tokens.size(), true, true);
        tokens.resize(n);

        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            batch.token[i]  = tokens[i];
            batch.pos[i]    = i;
            batch.seq_id[i] = &seq;
            batch.n_seq_id[i] = 1;
        }
        llama_decode(ctx_, batch);                                // prompt
        llama_batch_free(batch);

        std::vector<llama_token> out;
        out.reserve(n_predict);

        for (int step = 0; step < n_predict; ++step) {
            llama_batch b1 = llama_batch_init(1, 0, 1);
            b1.token[0]      = sample_next();                     // greedy / top-k
            b1.pos[0]        = tokens.size() + step;
            b1.seq_id[0]     = &seq;
            b1.n_seq_id[0]   = 1;
            llama_decode(ctx_, b1);
            out.push_back(b1.token[0]);
            llama_batch_free(b1);

            // 2⃣ optional user logic-loop
            if (user_hook_ && !(*user_hook_)(prompt, out)) break;

            // 3⃣ persist every step
            persist(seq);
        }
        std::string txt = tokens_to_str(out);
        return txt;
    }

  private:
    llama_token sample_next() {
        const float* logits = llama_get_logits(ctx_);
        int n_vocab         = llama_n_vocab(llama_model_get_vocab(model_));
        int best = 0;
        for (int i = 1; i < n_vocab; ++i)
            if (logits[i] > logits[best]) best = i;
        return best;                                              // naive greedy
    }

    void persist(llama_seq_id seq) {
        std::string file = state_dir_ + "/seq-" + std::to_string(seq) + ".pmll";
        llama_state_seq_save_file(ctx_, file.c_str(), seq, nullptr, 0);
    }

    void restore(llama_seq_id seq) {
        std::string file = state_dir_ + "/seq-" + std::to_string(seq) + ".pmll";
        if (fs::exists(file)) {
            llama_state_seq_load_file(ctx_, file.c_str(), seq,
                                      nullptr, 0, nullptr);
        }
    }

    std::string tokens_to_str(const std::vector<llama_token>& t) {
        std::string s;
        for (auto tok : t) {
            char buf[8];
            int n = llama_token_to_str(model_, tok, buf, sizeof(buf));
            if (n > 0) s.append(buf, n);
        }
        return s;
    }

    std::mutex            mu_;
    std::string           model_path_;
    std::string           state_dir_;
    LoopHook*             user_hook_;
    llama_model*          model_ = nullptr;
    llama_context*        ctx_   = nullptr;
    llama_memory_t        mem_   = nullptr;
};

} // namespace pmll
