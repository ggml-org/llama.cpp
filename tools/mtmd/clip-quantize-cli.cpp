#include "arg.h"
#include "base64.hpp"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "clip.h"
#include "llava.h"
#include "llama.h"
#include "ggml.h"

static void print_usage(int argc, char ** argv) {
    (void) argc;

    fprintf(stderr, "usage: %s /path/to/ggml-model-f32.gguf /path/to/ggml-model-quantized.gguf type\n", argv[0]);
    fprintf(stderr, "  type = 2 - q4_0\n");
    fprintf(stderr, "  type = 3 - q4_1\n");
    fprintf(stderr, "  type = 6 - q5_0\n");
    fprintf(stderr, "  type = 7 - q5_1\n");
    fprintf(stderr, "  type = 8 - q8_0\n");
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        print_usage(argc, argv);
        return 1;
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    const int itype = atoi(argv[3]);

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!clip_model_quantize(fname_inp.c_str(), fname_out.c_str(), itype)) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = ggml_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us / 1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    return 0;
}
