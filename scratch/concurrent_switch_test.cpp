// Concurrent multi-sequence isolation test for granite-switch.
//
// Two sequences are decoded together in one batch / one context:
//   seq 0: answerability  ON (<|answerability|>) -> expect "unanswerable"
//   seq 1: NO control token                       -> expect a normal answer
//
// The OLD global `poc_sticky_index` could not isolate these (last writer wins):
// seq 1 would inherit seq 0's adapter. The in-graph router stores each sequence's
// selection in its own KV-cache cells, so they must not cross-contaminate.
//
// PASS: seq0 contains "unanswerable" AND seq1 does NOT.

#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::vector<llama_token> tokz(const llama_vocab * v, const std::string & s, bool bos) {
    int n = -llama_tokenize(v, s.c_str(), s.size(), NULL, 0, bos, true);
    std::vector<llama_token> out(n);
    llama_tokenize(v, s.c_str(), s.size(), out.data(), out.size(), bos, true);
    return out;
}

int main(int argc, char ** argv) {
    std::string model_path;
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
    if (model_path.empty()) { fprintf(stderr, "usage: %s -m model.gguf\n", argv[0]); return 1; }

    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const std::string p0 =
        "<|start_of_role|>user<|end_of_role|>Document: The Eiffel Tower is in Paris. "
        "Question: What is the capital of Australia?<|end_of_role|><|end_of_text|>"
        "<|start_of_role|>assistant<|end_of_role|><|answerability|>";
    const std::string p1 =
        "<|start_of_role|>user<|end_of_role|>What is the capital of Australia?"
        "<|end_of_role|><|end_of_text|><|start_of_role|>assistant<|end_of_role|>";

    std::vector<llama_token> t0 = tokz(vocab, p0, true);
    std::vector<llama_token> t1 = tokz(vocab, p1, true);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = 2048;
    cp.n_batch   = 2048;
    cp.n_seq_max = 2;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    auto sp = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // build a batch holding BOTH prompts, seq 0 and seq 1 interleaved by sequence id.
    const int n_gen = 14;
    llama_batch batch = llama_batch_init((int) (t0.size() + t1.size()), 0, 2);

    auto add = [&](llama_token id, llama_pos pos, llama_seq_id seq, bool logits) {
        int k = batch.n_tokens;
        batch.token[k]     = id;
        batch.pos[k]       = pos;
        batch.n_seq_id[k]  = 1;
        batch.seq_id[k][0] = seq;
        batch.logits[k]    = logits;
        batch.n_tokens++;
    };

    for (size_t i = 0; i < t0.size(); ++i) add(t0[i], (llama_pos) i, 0, i == t0.size() - 1);
    for (size_t i = 0; i < t1.size(); ++i) add(t1[i], (llama_pos) i, 1, i == t1.size() - 1);

    if (llama_decode(ctx, batch)) { fprintf(stderr, "prompt decode failed\n"); return 1; }

    // remember the logits index for each sequence's last prompt token.
    int idx0 = (int) t0.size() - 1;
    int idx1 = (int) (t0.size() + t1.size()) - 1;
    llama_pos pos0 = (llama_pos) t0.size();
    llama_pos pos1 = (llama_pos) t1.size();

    std::string out0, out1;
    for (int step = 0; step < n_gen; ++step) {
        llama_token id0 = llama_sampler_sample(smpl, ctx, idx0);
        llama_token id1 = llama_sampler_sample(smpl, ctx, idx1);

        auto piece = [&](llama_token id, std::string & dst) {
            if (llama_vocab_is_eog(vocab, id)) return;
            char buf[256];
            int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
            if (n > 0) dst.append(buf, n);
        };
        piece(id0, out0);
        piece(id1, out1);

        batch.n_tokens = 0;
        bool e0 = llama_vocab_is_eog(vocab, id0);
        bool e1 = llama_vocab_is_eog(vocab, id1);
        if (e0 && e1) break;
        if (!e0) { add(id0, pos0++, 0, true); idx0 = batch.n_tokens - 1; }
        if (!e1) { add(id1, pos1++, 1, true); idx1 = batch.n_tokens - 1; }
        if (llama_decode(ctx, batch)) { fprintf(stderr, "gen decode failed\n"); break; }
    }

    printf("\n============ CONCURRENT ISOLATION TEST ============\n");
    printf("SEQ 0 (answerability ON): %s\n", out0.c_str());
    printf("SEQ 1 (no control token): %s\n", out1.c_str());
    auto has = [](const std::string & h, const char * n){ return h.find(n) != std::string::npos; };
    bool s0_ok      = has(out0, "unanswerable");
    bool s1_clean   = !has(out1, "unanswerable");
    printf("---------------------------------------------------\n");
    printf("seq0 fired its adapter:        %s\n", s0_ok ? "YES" : "NO");
    printf("seq1 isolated from seq0:       %s\n", s1_clean ? "clean" : "CONTAMINATED");
    bool pass = s0_ok && s1_clean;
    printf("RESULT: %s\n", pass ? "PASS (isolated)" : "FAIL");
    printf("===================================================\n");

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return pass ? 0 : 1;
}
