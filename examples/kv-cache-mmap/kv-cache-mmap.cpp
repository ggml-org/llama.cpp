// example: mmap-backed persistent KV cache
//
// demonstrates saving and restoring the KV cache across process restarts
// using a MAP_SHARED file. run twice with the same --kv-cache-mmap path:
//
//   session 1:  generates tokens and saves KV state to disk
//   session 2:  detects prior state, skips prompt eval, continues generating
//
// usage:
//
//   ./llama-kv-cache-mmap -m model.gguf --kv-cache-mmap /tmp/session.kv [-p prompt] [-n tokens]
//

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <sys/stat.h>
#endif

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf --kv-cache-mmap /tmp/session.kv [-p prompt] [-n n_predict]\n", argv[0]);
    printf("\n");
}

// read the .meta sidecar to determine how many cells were used in a prior session
static int read_prior_cells(const std::string & mmap_path) {
    const std::string meta_path = mmap_path + ".meta";

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct stat sb;
    if (stat(meta_path.c_str(), &sb) != 0 || sb.st_size == 0) {
        return 0;
    }
#endif

    FILE * f = fopen(meta_path.c_str(), "rb");
    if (!f) {
        return 0;
    }

    uint32_t magic = 0;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0x4B564D54) {
        fclose(f);
        return 0;
    }

    uint32_t n_streams = 0;
    fread(&n_streams, sizeof(n_streams), 1, f);
    if (n_streams < 1) {
        fclose(f);
        return 0;
    }

    uint32_t n_cells = 0, head = 0;
    fread(&n_cells, sizeof(n_cells), 1, f);
    fread(&head,    sizeof(head),    1, f);

    int n_used = 0;
    for (uint32_t i = 0; i < n_cells; i++) {
        int32_t p = -1;
        fread(&p, sizeof(p), 1, f);
        if (p >= 0) {
            n_used++;
        }
    }

    fclose(f);
    return n_used;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string mmap_path;
    std::string prompt = "Once upon a time";
    int n_predict = 32;

    // parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--kv-cache-mmap") == 0 && i + 1 < argc) {
            mmap_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argc, argv);
            return 0;
        } else {
            // treat remaining args as prompt
            prompt = argv[i];
        }
    }

    if (model_path.empty() || mmap_path.empty()) {
        print_usage(argc, argv);
        return 1;
    }

    // check for a prior session
    const int n_prior = read_prior_cells(mmap_path);
    const bool resuming = n_prior > 0;

    if (resuming) {
        printf("resuming from prior session (%d cells in %s)\n", n_prior, mmap_path.c_str());
    } else {
        printf("starting new session (mmap: %s)\n", mmap_path.c_str());
    }

    // load model
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "error: could not load model\n");
        return 1;
    }

    // create context with mmap KV cache
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = 512;
    cparams.n_batch      = 128;
    cparams.offload_kqv  = false;
    cparams.kv_mmap_path = mmap_path.c_str();

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "error: could not create context\n");
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_past = 0;

    // prompt eval (skip if resuming)
    if (!resuming) {
        std::vector<llama_token> tokens(512);
        int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(),
                                       tokens.data(), (int) tokens.size(), true, false);
        if (n_tokens < 0) {
            fprintf(stderr, "error: tokenization failed\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        tokens.resize(n_tokens);

        printf("prompt: \"%s\" (%d tokens)\n", prompt.c_str(), n_tokens);

        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (int i = 0; i < n_tokens; i++) {
            batch.token[i]    = tokens[i];
            batch.pos[i]      = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]   = (i == n_tokens - 1) ? 1 : 0;
        }
        batch.n_tokens = n_tokens;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "error: decode failed\n");
            llama_batch_free(batch);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        llama_batch_free(batch);
        n_past = n_tokens;
    } else {
        // resuming: KV data is in the mmap file, cell metadata was restored
        // re-decode the last token to produce logits for sampling
        n_past = n_prior;

        // decode one token at the next position to produce logits for sampling
        llama_token bos = llama_vocab_bos(vocab);
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.token[0]    = bos;
        batch.pos[0]      = n_past; // next position after the restored cells
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]   = 1;
        batch.n_tokens    = 1;

        llama_decode(ctx, batch);
        llama_batch_free(batch);

        printf("prompt eval skipped (KV cache restored from file)\n");
    }

    // generate
    printf("generating %d tokens...\n\n", n_predict);

    // sample the first token from the logits of the last decoded token
    // (index -1 tells the sampler to use the last token's logits)
    auto greedy = [&](int logit_idx) -> llama_token {
        const float * logits = llama_get_logits_ith(ctx, logit_idx);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        llama_token best = 0;
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > logits[best]) {
                best = v;
            }
        }
        return best;
    };

    // first sample uses the last logit from the prompt/warmup batch
    llama_token next = greedy(resuming ? 0 : n_past - 1);

    for (int i = 0; i < n_predict; i++) {
        if (llama_vocab_is_eog(vocab, next)) {
            printf("\n[end of text]\n");
            break;
        }

        char buf[128];
        int len = llama_token_to_piece(vocab, next, buf, sizeof(buf) - 1, 0, false);
        if (len > 0) {
            buf[len] = '\0';
            printf("%s", buf);
            fflush(stdout);
        }

        // decode and sample next
        llama_batch batch = llama_batch_get_one(&next, 1);
        llama_decode(ctx, batch);
        n_past++;

        next = greedy(0);
    }

    printf("\n\nsession saved to %s (%d cells)\n", mmap_path.c_str(), n_past);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
