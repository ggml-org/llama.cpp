#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "get-model.h"

char * get_model_or_exit(int argc, char *argv[]) {
    char * model_path;
    if (argc > 1) {
        model_path = argv[1];

    } else {
        model_path = getenv("LLAMACPP_TEST_MODELFILE");
        if (!model_path || strlen(model_path) == 0) {
            fprintf(stderr, "\033[33mWARNING: No model file provided. Skipping this test. Set LLAMACPP_TEST_MODELFILE=<gguf_model_path> to silence this warning and run this test.\n\033[0m");
            exit(EXIT_SUCCESS);
        }
    }

    return model_path;
}
