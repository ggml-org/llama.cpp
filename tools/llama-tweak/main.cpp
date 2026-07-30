#include "llama-tweak.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstring>

static void usage() {
    fprintf(stderr,
            "usage: llama-tweak record -m model.gguf [--pp 128,512] [--tg 64,128] [--runs 3]\n"
            "       llama-tweak explain -m model.gguf [--pp N] [--tg N]\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    if (strcmp(argv[1], "record") == 0) {
        ggml_backend_load_all();
    }
    return llama_tweak_record_main(argc, argv);
}
