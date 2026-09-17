#include "llama.h"
#include "common.h"
#include "console.h"

#include "arg.h"
#include "log.h"

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

int main(int argc, char ** argv) {
    common_params params;
    params.model.path = "."; // this test takes no model
    common_init();

    std::string fname;
    std::vector<char *> common_argv;
    common_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
            common_argv.push_back(argv[i]); // an option: let common_params_parse handle it
        } else if (fname.empty()) {
            fname = argv[i];
        }
    }
    common_argv.push_back(nullptr);
    if (!common_params_parse((int) common_argv.size() - 1, common_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (fname.empty()) {
        LOG_ERR("Usage: %s <vocab-file>\n", argv[0]);
        common_log_flush(common_log_main());
        return 1;
    }

    LOG_INF("%s : reading vocab from: '%s'\n", __func__, fname.c_str());

    llama_model * model;
    llama_context * ctx;

    llama_backend_init();

    // load the vocab
    {
        auto mparams = llama_model_default_params();

        mparams.vocab_only = true;

        model = llama_model_load_from_file(fname.c_str(), mparams);

        if (model == NULL) {
            LOG_ERR("%s: error: failed to load vocab '%s'\n", __func__, fname.c_str());
            common_log_flush(common_log_main());
            return 1;
        }

        auto cparams = llama_context_default_params();

        ctx = llama_init_from_model(model, cparams);

        if (ctx == NULL) {
            LOG_ERR("%s: error: failed to load vocab '%s'\n", __func__, fname.c_str());
            llama_model_free(model);
            common_log_flush(common_log_main());
            return 1;
        }
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    //GGML_ASSERT(llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM);
    if (llama_vocab_type(vocab) != LLAMA_VOCAB_TYPE_SPM) {
        // a vocab of another type is a skip (99), so it does not get a running line
        common_log_flush(common_log_main());
        return 99;
    }

    LOG("%s: running\n", "test-tokenizer-1-spm");

#ifdef _WIN32
    // We need this for unicode console support
    console::init(false, false);
    atexit([]() { console::cleanup(); });
#endif

    const int n_vocab = llama_vocab_n_tokens(vocab);

    LOG_INF("  running vocab detokenize round-trip (%d tokens)\n", n_vocab);

    for (int i = 0; i < n_vocab; ++i) {
        std::string str = common_detokenize(ctx, std::vector<int>(1, i), true);
        std::vector<llama_token> tokens = common_tokenize(ctx, str, false, true);
        std::string check = common_detokenize(ctx, tokens);
        if (check != str) {
            LOG_ERR("%s : error: token %d detokenizes to '%s'(%zu) but tokenization of this detokenizes to '%s'(%zu)\n",
                __func__, i, str.c_str(), str.length(), check.c_str(), check.length());
            LOG("%s: %s\n", "test-tokenizer-1-spm", "FAILED");
            common_log_flush(common_log_main());
            return 2;
        }
    }

    // unicode
    {
        LOG_INF("  running unicode codepoint round-trip\n");

        const int nthread = std::thread::hardware_concurrency();

        std::vector<std::thread> threads(nthread);

        std::atomic_int errcode = {};

        for (int i = 0; i < nthread; ++i) {
            threads[i] = std::thread([i, nthread, ctx, &errcode]() {
                for (uint32_t cp = i; !errcode && cp < 0x00110000; cp += nthread) {
                    if ((0x0000D800 <= cp && cp <= 0x0000DFFF) ||  // surrogates \p{Cs}
                        (0x00040000 <= cp && cp <= 0x000E0000)) {  // undefined \p{Cn}
                        continue;
                    }

                    std::string str = unicode_cpt_to_utf8(cp);
                    std::vector<llama_token> tokens = common_tokenize(ctx, str, false, true);
                    std::string check = common_detokenize(ctx, tokens);
                    if (cp != 9601 && str != check) {
                        LOG_ERR("error: codepoint 0x%x detokenizes to '%s'(%zu) instead of '%s'(%zu)\n",
                                cp, check.c_str(), check.length(), str.c_str(), str.length());
                        errcode = 3;
                    }
                }
            });
        }

        for (auto & t : threads) {
            t.join();
        }

        if(errcode) {
            LOG("%s: %s\n", "test-tokenizer-1-spm", "FAILED");
            common_log_flush(common_log_main());
            return errcode;
        }
    }

    llama_free(ctx);
    llama_model_free(model);

    llama_backend_free();

    LOG("%s: %s\n", "test-tokenizer-1-spm", "PASSED");
    common_log_flush(common_log_main());
    return 0;
}
