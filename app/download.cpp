#include "arg.h"
#include "common.h"
#include "download.h"
#include "log.h"

#include <cstdio>
#include <filesystem>

static void print_usage(int /*argc*/, char ** argv) {
    printf(
        "\nexamples:\n"
        "  %s -hf ggml-org/gemma-3-4b-it-qat-GGUF\n"
        "  %s -hf ggml-org/gemma-3-4b-it-qat-GGUF:Q4_K_M\n"
        "  %s -hf ggml-org/models -hff model.gguf\n"
        "  %s -mu https://example.com/model.gguf -m model.gguf\n"
        "\n",
        argv[0], argv[0], argv[0], argv[0]
    );
}

int llama_download(int argc, char ** argv);

int llama_download(int argc, char ** argv) {
    common_init();

    common_params params;
    params.verbosity = LOG_LEVEL_ERROR;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DOWNLOAD, print_usage)) {
        return 1;
    }

    const bool has_source = !params.model.hf_repo.empty() || !params.model.url.empty() ||
                            !params.model.path.empty()    || !params.model.docker_repo.empty();
    if (!has_source) {
        fprintf(stderr, "error: no model source specified (use --hf-repo, --model-url, --model or --docker-repo)\n");
        return 1;
    }

    try {
        common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_DOWNLOAD);
        common_models_handler_apply(handler, params);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    if (!params.models_preset.empty()) {
        // -hf pointed at a preset repo: print the preset path and stop
        printf("%s\n", params.models_preset.c_str());
        return 0;
    }
    if (params.model.path.empty()) {
        fprintf(stderr, "error: model download failed\n");
        return 1;
    }
    if (!std::filesystem::exists(params.model.path)) {
        fprintf(stderr, "error: model file does not exist: %s\n", params.model.path.c_str());
        return 1;
    }

    printf("%s\n", params.model.path.c_str());
    if (!params.mmproj.path.empty()) {
        printf("%s\n", params.mmproj.path.c_str());
    }
    if (!params.speculative.draft.mparams.path.empty()) {
        printf("%s\n", params.speculative.draft.mparams.path.c_str());
    }

    return 0;
}
