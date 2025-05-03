#include "llama.h"
#include "common.h"
#include "console.h"

#include "../src/unicode.h"

#include <cassert>
#include <codecvt>
#include <cstdio>
#include <cstring>
#include <locale>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <vocab-file> [--ignore-merges]\n", argv[0]);
        return 1;
    }

    const std::string fname = argv[1];
    bool ignore_merges = false;
    if (argc == 3) {
        if (std::strcmp(argv[2], "--ignore-merges") != 0) {
            fprintf(stderr, "Usage: %s <vocab-file> [--ignore-merges]\n", argv[0]);
            return 1;
        }
        ignore_merges = true;
    }

    fprintf(stderr, "%s : reading vocab from: '%s'\n", __func__, fname.c_str());

    if (ignore_merges) {
        fprintf(stderr, "%s : ignoring merges for tokens inside vocab\n", __func__);
    }

    llama_model * model;
    llama_context * ctx;

    llama_backend_init();

    // load the vocab
    {
        auto mparams = llama_model_default_params();

        mparams.vocab_only = true;

        model = llama_model_load_from_file(fname.c_str(), mparams);

        if (model == NULL) {
            fprintf(stderr, "%s: error: failed to load vocab '%s'\n", __func__, fname.c_str());
            return 1;
        }

        auto cparams = llama_context_default_params();

        ctx = llama_init_from_model(model, cparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load vocab '%s'\n", __func__, fname.c_str());
            llama_model_free(model);
            return 1;
        }
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    //GGML_ASSERT(llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_BPE);
    if (llama_vocab_type(vocab) != LLAMA_VOCAB_TYPE_BPE) {
        return 99;
    }

#ifdef _WIN32
    // We need this for unicode console support
    console::init(false, false);
    atexit([]() { console::cleanup(); });
#endif

    const int n_vocab = llama_vocab_n_tokens(vocab);

    for (int i = 0; i < n_vocab; ++i) {
        std::string str = common_detokenize(ctx, std::vector<int>(1, i));
        try {
            auto cps = unicode_cpts_from_utf8(str);
            std::vector<llama_token> tokens = common_tokenize(ctx, str, false, true);
            if (ignore_merges && tokens.size() > 1) {
                fprintf(stderr,
                        "%s : error: token %d detokenizes to '%s'(%zu) but "
                        "tokenization of this to multiple tokens: [",
                        __func__, i, str.c_str(), str.length());
                fprintf(stderr, "%d", tokens[0]);
                for (size_t i = 1; i < tokens.size(); i++) {
                    fprintf(stderr, ", %d", tokens[i]);
                }
                fprintf(stderr, "]\n");
                return 2;
            }
            std::string check = common_detokenize(ctx, tokens);
            if (check != str) {
                fprintf(stderr, "%s : error: token %d detokenizes to '%s'(%zu) but tokenization of this detokenizes to '%s'(%zu)\n",
                    __func__, i, str.c_str(), str.length(), check.c_str(), check.length());
                return 2;
            }
        }
        catch (const std::invalid_argument &) {
            //fprintf(stderr, "%s : info: utf8 conversion %d '%s'\n", __func__, i, str.c_str());
        }
    }

    // unicode
    {
        const int nthread = std::thread::hardware_concurrency();

        std::vector<std::thread> threads(nthread);

        std::atomic_int errcode = {};

        for (int i = 0; i < nthread; ++i) {
            threads[i] = std::thread([i, nthread, ctx, &errcode]() {
                for (uint32_t cp = i; !errcode && cp < 0x00110000; cp += nthread) {
                    if ((0x0000D800 <= cp && cp <= 0x0000DFFF) ||  // surrogates \p{Cs}
                        (0x00040000 <= cp && cp <= 0x000E0000)) {  // undefined  \p{Cn}
                        continue;
                    }

                    std::string str = unicode_cpt_to_utf8(cp);
                    std::vector<llama_token> tokens = common_tokenize(ctx, str, false);
                    std::string check = common_detokenize(ctx, tokens);
                    if (cp != 9601 && str != check) {
                        fprintf(stderr, "error: codepoint 0x%x detokenizes to '%s'(%zu) instead of '%s'(%zu)\n",
                                cp, check.c_str(), check.length(), str.c_str(), str.length());
                        errcode = 3;
                    }
                }
            });
        }

        for (auto & t : threads) {
            t.join();
        }

        if (errcode) {
            return errcode;
        }
    }

    llama_model_free(model);
    llama_free(ctx);

    llama_backend_free();

    return 0;
}
