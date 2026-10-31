#include "llama.h"
#include "common.h"

#include <cstdlib>
#include <thread>
#include <vector>

struct progress_state {
    std::thread::id thread;
    float cancel_at;
    float previous = -1.0f;
    bool cancelled = false;
};

int main(int argc, char *argv[] ) {
    auto * model_path = common_get_model_or_exit(argc, argv);
    auto * file = fopen(model_path, "r");
    if (file == nullptr) {
        fprintf(stderr, "no model at '%s' found\n", model_path);
        return EXIT_FAILURE;
    }

    fprintf(stderr, "using '%s'\n", model_path);
    fclose(file);

    ggml_backend_load_all();
    llama_backend_init();
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.progress_callback = [](float progress, void * ctx) {
        auto & state = *(progress_state *) ctx;
        GGML_ASSERT(std::this_thread::get_id() == state.thread);
        GGML_ASSERT(!state.cancelled);
        GGML_ASSERT(progress >= state.previous && progress <= 1.0f);
        state.previous = progress;
        state.cancelled = progress >= state.cancel_at;
        return !state.cancelled;
    };

    // Optional arguments after the model path are RPC endpoints.
    std::vector<ggml_backend_dev_t> devices;
    if (argc > 2) {
        auto * rpc = ggml_backend_reg_by_name("RPC");
        GGML_ASSERT(rpc != nullptr);
        auto add_server = (ggml_backend_reg_t (*)(const char *)) ggml_backend_reg_get_proc_address(rpc, "ggml_backend_rpc_add_server");
        GGML_ASSERT(add_server != nullptr);
        for (int i = 2; i < argc; i++) {
            auto * reg = add_server(argv[i]);
            GGML_ASSERT(reg != nullptr);
            for (size_t j = 0; j < ggml_backend_reg_dev_count(reg); j++) {
                auto * dev = ggml_backend_reg_dev_get(reg, j);
                ggml_backend_device_register(dev);
                devices.push_back(dev);
            }
        }
        devices.push_back(nullptr);
        params.devices = devices.data();
        params.n_gpu_layers = 99;
    }

    // The default test still cancels immediately. RPC tests also cover partial loads, completion, and reuse after cancellation.
    const float cutoffs[] = {0.0f, 0.5f, 1.0f, 2.0f};
    const llama_load_mode modes[] = {LLAMA_LOAD_MODE_NONE, LLAMA_LOAD_MODE_MMAP, LLAMA_LOAD_MODE_DIRECT_IO};
    const int n_cutoffs = devices.empty() ? 1 : 4;
    const int n_modes   = devices.empty() ? 1 : 3;
    for (int i = 0; i < n_modes; i++) {
        params.load_mode = modes[i];
        for (int j = 0; j < n_cutoffs; j++) {
            const float cutoff = cutoffs[j];
            progress_state state{std::this_thread::get_id(), cutoff};
            params.progress_callback_user_data = &state;
            auto * model = llama_model_load_from_file(model_path, params);
            GGML_ASSERT(state.previous >= 0.0f);
            if (cutoff <= 1.0f) {
                GGML_ASSERT(model == nullptr && state.cancelled);
            } else {
                GGML_ASSERT(model != nullptr && state.previous == 1.0f);
            }
            llama_model_free(model);
        }
    }
    llama_backend_free();
    return EXIT_SUCCESS;
}
