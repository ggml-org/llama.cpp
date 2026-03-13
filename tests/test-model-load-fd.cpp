#include "llama.h"
#include "get-model.h"

#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
int main(int /*argc*/, char ** /*argv*/) {
    fprintf(stderr, "skipping on Windows\n");
    return EXIT_SUCCESS;
}
#else
#    include <fcntl.h>
#    include <unistd.h>

int main(int argc, char ** argv) {
    auto * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    const int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "failed to open %s\n", model_path);
        return EXIT_FAILURE;
    }

    auto params = llama_model_default_params();
    params.use_mmap = true;
    params.vocab_only = true;

    struct llama_model * model = llama_model_load_from_fd(fd, params);
    close(fd);

    if (model == nullptr) {
        fprintf(stderr, "load from fd failed\n");
        return EXIT_FAILURE;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    fprintf(stderr, "loaded %d tokens from fd\n", n_vocab);

    llama_model_free(model);
    llama_backend_free();

    return n_vocab > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif // _WIN32
