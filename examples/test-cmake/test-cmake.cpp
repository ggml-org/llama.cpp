#include "llama.h"
#include <cstdio>

int main(void) {
    printf("[test-cmake] version: %s, build: %d (%s)\n",
           llama_version(), LLAMA_BUILD_NUMBER, LLAMA_BUILD_COMMIT);
    printf("[test-cmake] Initializing backend...\n");
    llama_backend_init();
    printf("[test-cmake] Backend initialized.\n");
    llama_backend_free();
    return 0;
}
