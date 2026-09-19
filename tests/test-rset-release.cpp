// ref: https://github.com/ggml-org/llama.cpp/issues/25937
// only works reliably when run with a large model that occupies 3GB+ of wired memory
// thus, this test is not run by default
// example model to run with: google/gemma-4-E4B-it-qat-q4_0-gguf

#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"

#include <cstdint>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <unistd.h>
#include <vector>

static uint64_t wired_memory() {
    vm_statistics64_data_t vmstat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vmstat, &count) != KERN_SUCCESS) {
        return UINT64_MAX;
    }
    return static_cast<uint64_t>(vmstat.wire_count) * vm_kernel_page_size;
}

int main(int argc, char ** argv) {
    // the model path is this test's only positional argument
    char * model_argv[2] = { argv[0], nullptr };

    {
        common_params params;
        params.model.path = "."; // this test takes no model
        common_init();

        std::vector<char *> common_argv;
        common_argv.push_back(argv[0]);
        for (int i = 1; i < argc; i++) {
            if (argv[i][0] == '-') {
                common_argv.push_back(argv[i]); // an option: let common_params_parse handle it
            } else if (model_argv[1] == nullptr) {
                model_argv[1] = argv[i];
            }
        }
        common_argv.push_back(nullptr);
        if (!common_params_parse((int) common_argv.size() - 1, common_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
            return 1;
        }
    }

    // falls back to LLAMACPP_TEST_MODELFILE, or warns and exits if no model is given
    auto * model_path = common_get_model_or_exit(model_argv[1] == nullptr ? 1 : 2, model_argv);

    // that call exits when no model is given, so the verdict below is only reached with a model
    LOG("%s: running\n", "test-rset-release");

    llama_backend_init();

    const uint64_t wired_initial = wired_memory();

    llama_model_params params = llama_model_default_params();
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    struct llama_model* model = llama_model_load_from_file(model_path, params);

    const uint64_t wired_loaded = wired_memory();
    const uint64_t wired_delta = wired_loaded - wired_initial;
    // system memory fluctuates, so we need to allocate enough to reliably detect the release
    GGML_ASSERT(wired_delta > 2'000'000'000); // 2GB

    llama_model_free(model);

    const uint64_t t_start_ms = ggml_time_ms();

    // expect most of the allocated memory to be released within 10 seconds
    // we allow for some tolerance due to system-wide memory fluctuations
    while (wired_memory() > wired_loaded - 0.75 * wired_delta) {
        GGML_ASSERT(ggml_time_ms() - t_start_ms < 10'000);
        usleep(100'000); // 100ms
    }

    llama_backend_free();

    LOG("%s: %s\n", "test-rset-release", "PASSED");
    common_log_flush(common_log_main());
    return 0;
}
