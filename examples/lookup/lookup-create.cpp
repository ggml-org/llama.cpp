#include "ggml.h"
#include "llama.h"
#include "common.h"
#include "ngram-cache.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

int main(int argc, char ** argv){
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model
    llama_init_result llama_init = llama_init_from_gpt_params(params);

    llama_model * model = llama_init.model;
    llama_context * ctx = llama_init.context;
    GGML_ASSERT(model != nullptr);

    // tokenize the prompt
    std::vector<llama_token> inp;
    inp = ::llama_tokenize(ctx, params.prompt, true, true);
    fprintf(stderr, "%s: tokenization done\n", __func__);


    llama_ngram_cache ngram_cache;
    llama_ngram_cache_update(ngram_cache, LLAMA_NGRAM_STATIC, LLAMA_NGRAM_STATIC, inp, inp.size(), true);
    fprintf(stderr, "%s: hashing done, writing file to %s\n", __func__, params.lookup_cache_static.c_str());

    llama_ngram_cache_save(ngram_cache, params.lookup_cache_static);
}
