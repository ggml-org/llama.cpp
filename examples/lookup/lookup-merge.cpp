#include "ggml.h"
#include "llama.h"
#include "common.h"
#include "ngram-cache.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

static void print_usage(char* argv0) {
    fprintf(stderr, "Merges multiple lookup cache files into a single one.\n");
    fprintf(stderr, "Usage: %s [--help] lookup_part_1.bin lookup_part_2.bin ... lookup_merged.bin\n", argv0);
}

int main(int argc, char ** argv){
    if (argc < 3) {
        print_usage(argv[0]);
        exit(1);
    }

    std::vector<std::string> args;
    args.resize(argc-1);
    for (int i = 0; i < argc-1; ++i) {
        args[i] = argv[i+1];
        if (args[i] == "-h" || args[i] == "--help") {
            print_usage(argv[0]);
            exit(0);
        }
    }

    fprintf(stderr, "lookup-merge: loading file %s\n", args[0].c_str());
    common_ngram_cache ngram_cache_merged = common_ngram_cache_load(args[0]);

    for (size_t i = 1; i < args.size()-1; ++i) {
        fprintf(stderr, "lookup-merge: loading file %s\n", args[i].c_str());
        common_ngram_cache ngram_cache = common_ngram_cache_load(args[i]);

        common_ngram_cache_merge(ngram_cache_merged, ngram_cache);
    }

    fprintf(stderr, "lookup-merge: saving file %s\n", args.back().c_str());
    common_ngram_cache_save(ngram_cache_merged, args.back());
}
