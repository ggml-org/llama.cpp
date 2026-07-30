#include "llama.h"

#include <cstdio>
#include <memory>

struct model_deleter {
    void operator()(llama_model * model) const {
        llama_model_free(model);
    }
};

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf [--component-only]\n", argv[0]);
        return 2;
    }

    llama_backend_init();

    std::unique_ptr<llama_model, model_deleter> primary;
    if (argc == 2) {
        llama_model_params primary_params = llama_model_default_params();
        primary_params.n_gpu_layers = 0;
        primary.reset(llama_model_load_from_file(argv[1], primary_params));
        if (!primary) {
            std::fprintf(stderr, "failed to load primary model view\n");
            return 1;
        }
    }

    llama_model_params mtp_params = llama_model_default_params();
    mtp_params.n_gpu_layers = 0;
    mtp_params.component_prefix = "mtp.";
    std::unique_ptr<llama_model, model_deleter> mtp(
        llama_model_load_from_file(argv[1], mtp_params));
    if (!mtp) {
        std::fprintf(stderr, "failed to load mtp component view\n");
        return 1;
    }

    char architecture[64] = {};
    if (llama_model_meta_val_str(mtp.get(), "mtp.general.architecture",
                                 architecture, sizeof(architecture)) <= 0) {
        std::fprintf(stderr, "missing namespaced MTP architecture metadata\n");
        return 1;
    }

    if (llama_model_n_embd(mtp.get()) != 1024 ||
        llama_model_n_embd_out(mtp.get()) != 3840) {
        std::fprintf(stderr, "unexpected MTP component dimensions\n");
        return 1;
    }

    llama_backend_free();
    return 0;
}
