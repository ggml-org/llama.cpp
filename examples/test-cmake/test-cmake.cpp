#include "llama.h"
#include <cstdio>

int main(void) {
#ifdef LLAMA_BUILD_NUMBER
    printf("[test-cmake] llama.cpp version: %s, build: %d (%s)\n",
           llama_version(), LLAMA_BUILD_NUMBER, LLAMA_BUILD_COMMIT);
#else
    printf("[test-cmake] llama.cpp version: %s\n", llama_version());
#endif
    printf("[test-cmake] ggml version: %s, commit: %s\n", ggml_version(), ggml_commit());
    printf("[test-cmake] Initializing backend...\n");
    llama_backend_init();
    printf("[test-cmake] Backend initialized.\n");
    llama_backend_free();
    return 0;
}
