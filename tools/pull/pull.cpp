#include "arg.h"
#include "common.h"
#include "log.h"

#include <cstdio>
#include <string>

static void print_usage(int, char ** argv) {
    LOG("Usage: %s [options]\n", argv[0]);
    LOG("\n");
    LOG("Download models from HuggingFace or Docker Hub\n");
    LOG("\n");
    LOG("Options:\n");
    LOG("  -h, --help                show this help message and exit\n");
    LOG("  -hf, -hfr, --hf-repo REPO download model from HuggingFace repo\n");
    LOG("                            format: <user>/<model>[:<quant>]\n");
    LOG("                            example: microsoft/DialoGPT-medium\n");
    LOG("  -dr, --docker-repo REPO   download model from Docker Hub\n");
    LOG("                            format: [<repo>/]<model>[:<quant>]\n");
    LOG("                            example: gemma3\n");
    LOG("  --hf-token TOKEN          HuggingFace token for private repos\n");
    LOG("\n");
    LOG("Examples:\n");
    LOG("  %s -hf microsoft/DialoGPT-medium\n", argv[0]);
    LOG("  %s -dr gemma3\n", argv[0]);
    LOG("  %s -hf microsoft/DialoGPT-medium\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    common_params params;

    // Parse command line arguments
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        print_usage(argc, argv);
        return 1;
    }

    // Check if help was requested or no download option provided
    if (params.model.hf_repo.empty() && params.model.docker_repo.empty()) {
        LOG_ERR("error: must specify either -hf <repo> or -dr <repo>\n");
        print_usage(argc, argv);
        return 1;
    }

    LOG_INF("llama-pull: downloading model...\n");
    try {
        // Use the existing model handling logic which downloads the model
        common_init_result llama_init = common_init_from_params(params);
        if (llama_init.model != nullptr) {
            LOG_INF("Model downloaded and loaded successfully to: %s\n", params.model.path.c_str());

            // We only want to download, not keep the model loaded
            // The download happens during common_init_from_params
        } else {
            LOG_ERR("Failed to download or load model\n");
            return 1;
        }
    } catch (const std::exception & e) {
        LOG_ERR("Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
