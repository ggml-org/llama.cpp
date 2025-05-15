#include "arg.h"
#include "common.h"
#include "ngram-cache.h"
#include "ggml.h"
#include "llama.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

int main(int argc, char ** argv){
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LOOKUP)) {
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model;
    llama_context * ctx = llama_init.context;
    GGML_ASSERT(model != nullptr);

    // tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx, params.prompt, true, true);
    fprintf(stderr, "%s: tokenization done\n", __func__);


    common_ngram_cache ngram_cache;
    common_ngram_cache_update(ngram_cache, LLAMA_NGRAM_STATIC, LLAMA_NGRAM_STATIC, inp, inp.size(), true);
    fprintf(stderr, "%s: hashing done, writing file to %s\n", __func__, params.lookup_cache_static.c_str());

    common_ngram_cache_save(ngram_cache, params.lookup_cache_static);

    return 0;
}
