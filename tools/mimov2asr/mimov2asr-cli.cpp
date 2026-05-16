#include "llama.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

// Helper: Read discrete audio tokens from a local file
std::vector<llama_token> read_tokens_from_file(const std::string& filename) {
    std::vector<llama_token> tokens;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open token file: " << filename << std::endl;
        return tokens;
    }
    llama_token t;
    while (file >> t) {
        tokens.push_back(t);
    }
    return tokens;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <tokens_file>\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string tokens_file = argv[2];

    llama_backend_init();

    // 1. Initialize model
    auto mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    // 2. Initialize context
    auto cparams = llama_context_default_params();
    // Enlarge context and batch sizes to handle long audio inputs safely
    cparams.n_ctx = 8192;
    cparams.n_batch = 8192;
    cparams.n_ubatch = 8192;
    // CRITICAL: Enable embeddings so the underlying graph retains the computed features
    cparams.embeddings = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::cerr << "Failed to initialize context\n";
        return 1;
    }

    // 3. Load audio tokens from file
    std::vector<llama_token> audio_tokens = read_tokens_from_file(tokens_file);
    int n_input_tokens = audio_tokens.size();

    // Clean tokens: -100 padding becomes 0, out-of-vocab becomes empty token (151667)
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (int i = 0; i < n_input_tokens; i++) {
        if (audio_tokens[i] < 0) {
            audio_tokens[i] = 0;
        } else if (audio_tokens[i] >= n_vocab) {
            audio_tokens[i] = 151667;
        }
    }
    std::cout << "[MiMo] Loaded " << n_input_tokens << " audio tokens.\n";

    // =================================================================
    // Phase 1: Encoder (Prefill Front-end)
    // =================================================================
    std::cout << "[MiMo] Phase 1: Running Encoder graph...\n";

    llama_batch batch_enc = llama_batch_init(n_input_tokens, 0, 1);
    batch_enc.n_tokens = n_input_tokens;
    int n_outputs = n_input_tokens / 36; // 36 = (8 audio channels + 1 text channel) * group_size(4)
    
    for (int i = 0; i < n_input_tokens; i++) {
        batch_enc.token[i]    = audio_tokens[i];
        batch_enc.pos[i]      = i;
        batch_enc.n_seq_id[i] = 1;
        batch_enc.seq_id[i][0]= 0;
        batch_enc.logits[i]   = (i < n_outputs) ? true : false;
    }

    if (llama_encode(ctx, batch_enc) != 0) {
        std::cerr << "llama_encode failed!\n";
        return 1;
    }

    // =================================================================
    // The Magic Bridge: Zero-copy feature transfer
    // =================================================================
    // Retrieve the fused features computed by the encoder
    float* fused_features = llama_get_embeddings(ctx);
    if (!fused_features) {
        std::cerr << "Failed to get fused embeddings from encoder!\n";
        return 1;
    }

    std::cout << "[MiMo] Extracted " << n_outputs << " fused features.\n";

    // Allocate a batch for decoding without an internal token array
    llama_batch batch_dec = llama_batch_init(n_outputs, 0, 1);
    batch_dec.n_tokens = n_outputs;

    // Backup original token pointer and mount the fused features directly
    llama_token* original_token_ptr = batch_dec.token; 
    batch_dec.token = nullptr;
    batch_dec.embd = fused_features; 

    for (int i = 0; i < n_outputs; i++) {
        batch_dec.pos[i]      = i;
        batch_dec.n_seq_id[i] = 1;
        batch_dec.seq_id[i][0]= 0;
        // Only require logits for the last feature step
        batch_dec.logits[i]   = (i == n_outputs - 1) ? true : false; 
    }

    // =================================================================
    // Phase 2: Decoder (Auto-regressive Back-end)
    // =================================================================
    std::cout << "[MiMo] Phase 2: Running Decoder backbone...\n";
    if (llama_decode(ctx, batch_dec) != 0) {
        std::cerr << "llama_decode failed!\n";
        return 1;
    }

    // Initialize greedy sampler
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    int n_gen = 0;
    const int max_gen = 128;
    int n_past = n_outputs;

    std::cout << "\n[MiMo] Generation Result: \n";
    while (n_gen < max_gen) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, id);

        if (llama_vocab_is_eog(llama_model_get_vocab(model), id)) {
            break;
        }

        // Print generated token
        char buf[128] = {0};
        int n_chars = llama_token_to_piece(llama_model_get_vocab(model), id, buf, sizeof(buf), 0, true);
        if (n_chars > 0) {
            std::string piece(buf, n_chars);
            std::cout << piece << std::flush;
        }

        // Prepare next step batch
        llama_batch batch_next = llama_batch_init(1, 0, 1);
        batch_next.n_tokens = 1;
        batch_next.token[0] = id;
        batch_next.pos[0]   = n_past++;
        batch_next.n_seq_id[0] = 1;
        batch_next.seq_id[0][0] = 0;
        batch_next.logits[0]   = true; 

        if (llama_decode(ctx, batch_next) != 0) {
            std::cerr << "\nDecode failed during generation!\n";
            break;
        }

        llama_batch_free(batch_next);
        n_gen++;
    }
    std::cout << "\n\n[MiMo] Inference finished.\n";

    // =================================================================
    // Cleanup
    // =================================================================
    batch_dec.embd = nullptr;             // Unmount feature pointer safely
    batch_dec.token = original_token_ptr; // Restore original token pointer for proper memory deallocation

    llama_batch_free(batch_dec);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}