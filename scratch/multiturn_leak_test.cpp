// Multi-turn behavior probe for granite-switch (DOCUMENTS A KNOWN LIMITATION).
//
// Both turns run on the SAME llama_context — one persistent KV cache, no reset
// between turns, exactly like `ollama run` continuing a chat:
//   turn 1: answerability ON (<|answerability|>) -> "unanswerable"
//   turn 2: NO control token                     -> ??? (same cache)
//
// EXPECTED (documented) behavior: turn 2 STILL reads the adapter. This is the
// single-switch contract, identical to the vLLM/HF backends, whose own docstring
// states: "SingleSwitch has no mechanism to transition back to base
// mid-sequence." The in-graph router selection lives in the per-sequence KV
// cache; turn 1's control-token K is still causally visible to turn 2's tokens,
// and with flat (recency-free) gain it keeps winning the softmax.
//
// vLLM/HF never observe this because each served request is a FRESH sequence
// (turn 2's context does not contain turn 1's control token). A chat client that
// continues one cache across turns (ollama) must therefore start a fresh
// sequence per turn, OR opt into a recency-biased router (a deliberate
// divergence from vLLM — not done here).
//
// This probe asserts the DOCUMENTED behavior so it stays a faithful vLLM copy:
//   turn 1 -> "unanswerable"
//   turn 2 -> ALSO "unanswerable" (carry-over expected on a persisted cache)

#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::string g_model;

static std::vector<llama_token> tok(const llama_vocab * v, const std::string & s, bool bos) {
    int n = -llama_tokenize(v, s.c_str(), s.size(), NULL, 0, bos, true);
    std::vector<llama_token> out(n);
    llama_tokenize(v, s.c_str(), s.size(), out.data(), out.size(), bos, true);
    return out;
}

// Decode `prompt_tokens` (appended at current position `pos`) then greedily
// generate up to n_gen tokens, advancing `pos`. Returns the generated text.
static std::string run_turn(llama_context * ctx, const llama_vocab * vocab,
                            llama_sampler * smpl,
                            std::vector<llama_token> prompt_tokens,
                            int & pos, int n_gen) {
    std::string text;

    // decode the prompt chunk (positions assigned automatically by the batch helper
    // continue from the cache, since we never clear it between turns).
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_decode(ctx, batch)) { fprintf(stderr, "decode (prompt) failed\n"); return text; }
    pos += (int) prompt_tokens.size();

    for (int i = 0; i < n_gen; ++i) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;
        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) text.append(buf, n);
        llama_batch nb = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx, nb)) { fprintf(stderr, "decode (gen) failed\n"); break; }
        pos += 1;
    }
    return text;
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) g_model = argv[++i];
    }
    if (g_model.empty()) { fprintf(stderr, "usage: %s -m model.gguf\n", argv[0]); return 1; }

    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(g_model.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = 2048;
    cp.n_batch = 2048;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    auto sp = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    int pos = 0;

    // --- TURN 1: answerability ON (control token fires the adapter) ---
    // Doc is about the Eiffel Tower; question asks Australia's capital -> adapter
    // should judge it "unanswerable".
    std::string t1 =
        "<|start_of_role|>user<|end_of_role|>Document: The Eiffel Tower is in Paris. "
        "Question: What is the capital of Australia?<|end_of_role|><|end_of_text|>"
        "<|start_of_role|>assistant<|end_of_role|><|answerability|>";
    std::string out1 = run_turn(ctx, vocab, smpl, tok(vocab, t1, /*bos=*/true), pos, 12);

    // --- TURN 2: NO control token, same context (cache persists) ---
    // A normal question. If the adapter leaked from turn 1, this would also be
    // judged "unanswerable". With the router fix it answers normally.
    std::string t2 =
        "<|end_of_text|><|start_of_role|>user<|end_of_role|>"
        "What is the capital of Australia?<|end_of_role|><|end_of_text|>"
        "<|start_of_role|>assistant<|end_of_role|>";
    std::string out2 = run_turn(ctx, vocab, smpl, tok(vocab, t2, /*bos=*/false), pos, 16);

    printf("\n========== MULTI-TURN CARRY-OVER PROBE (KNOWN LIMIT) ==========\n");
    printf("TURN 1 (answerability ON): %s\n", out1.c_str());
    printf("TURN 2 (no control token): %s\n", out2.c_str());

    auto contains = [](const std::string & h, const char * n) {
        return h.find(n) != std::string::npos;
    };
    bool t1_ok      = contains(out1, "unanswerable");
    bool t2_carried = contains(out2, "unanswerable");

    printf("--------------------------------------------------------------\n");
    printf("turn1 fired adapter (expect unanswerable):        %s\n", t1_ok ? "YES" : "NO");
    printf("turn2 carried adapter (DOCUMENTED: expect carry): %s\n", t2_carried ? "carried" : "reverted");
    // Faithful vLLM/HF contract: on a persisted cache, selection carries over.
    bool pass = t1_ok && t2_carried;
    printf("RESULT: %s\n", pass ? "PASS (matches vLLM/HF single-switch contract)"
                                : "UNEXPECTED (behavior differs from vLLM/HF)");
    printf("==============================================================\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return pass ? 0 : 1;
}
