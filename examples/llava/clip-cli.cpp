//
// Example usage of just the vision encoder (CLIP) part of the LLAVA codebase.
// It loads a CLIP model (gguf file) and an image file,
// computes the image embedding, and prints out (a few elements of) the embedding.
//
// Build and run (for example):
//     ./bin/llama-clip-cli -c model.gguf -i input.png --threads 1 --verbosity 1
//     ./bin/llama-clip-cli -c clip.gguf -i input.png --threads 1 --verbosity 1 

#include "arg.h"
#include "base64.hpp"
#include "log.h"
#include "common.h"
#include "clip.h"
#include "llava.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

// Structure to hold our command line parameters.
struct vision_params {
    std::string clip_model;   // Path to the CLIP model file (gguf)
    std::string image_file;   // Path to the image file to process
    int         n_threads = 1;    // Number of CPU threads to use
    int         verbosity = 1;    // Verbosity level for model loading
};

static void print_usage(const char* progname) {
    LOG("\nUsage: %s -c <clip_model_path> -i <image_file> [--threads <n_threads>] [--verbosity <level>]\n\n", progname);
}

int main(int argc, char ** argv) {
    ggml_time_init();

    vision_params params;

    // Simple command line parsing
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-c" || arg == "--clip") {
            if (i + 1 < argc) {
                params.clip_model = argv[++i];
            } else {
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "-i" || arg == "--image") {
            if (i + 1 < argc) {
                params.image_file = argv[++i];
            } else {
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "--threads") {
            if (i + 1 < argc) {
                params.n_threads = std::atoi(argv[++i]);
            } else {
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "--verbosity") {
            if (i + 1 < argc) {
                params.verbosity = std::atoi(argv[++i]);
            } else {
                print_usage(argv[0]);
                return 1;
            }
        } else {
            // Unknown argument.
            print_usage(argv[0]);
            return 1;
        }
    }

    if (params.clip_model.empty() || params.image_file.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Load the CLIP model.
    struct clip_ctx * ctx_clip = clip_model_load(params.clip_model.c_str(), params.verbosity);
    if (!ctx_clip) {
        LOG_ERR("Failed to load clip model from %s\n", params.clip_model.c_str());
        return 1;
    }
    LOG_INF("Clip model loaded from %s\n", params.clip_model.c_str());

    // Load and process the image.
    llava_image_embed * embed = llava_image_embed_make_with_filename(ctx_clip, params.n_threads, params.image_file.c_str());
    if (!embed) {
        LOG_ERR("Failed to load or process image from %s\n", params.image_file.c_str());
        clip_free(ctx_clip);
        return 1;
    }
    LOG_INF("Image loaded and processed from %s\n", params.image_file.c_str());
    LOG_INF("Image embedding computed with %d positions.\n", embed->n_image_pos);
    int print_count = (embed->n_image_pos < 10 ? embed->n_image_pos : 10);
    LOG_INF("First %d elements: ", print_count);

    for (int i = 0; i < print_count; i++) {
        LOG_INF("%f ", embed->embed[i]);
    }
    LOG_INF("\n");

    llava_image_embed_free(embed);
    clip_free(ctx_clip);

    return 0;
}
