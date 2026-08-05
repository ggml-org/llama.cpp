// tessera: llama-clip-capture CLI.
//
// Standalone binary that the Python multimodal_calibrate.py
// driver invokes via subprocess. Mirrors the imatrix CLI
// pattern: the binary is the surface the orchestrator uses, the
// library is the reusable side.
//
// Usage:
//   llama-clip-capture --model PATH --inputs PATH [repeat] \
//                      --output PATH --mode vision|audio \
//                      [--batch-size N] [--peak-rss-budget-gb N] \
//                      [--threads N]
//   llama-clip-capture --model PATH --mm-projector PATH \
//                      --inputs PATH [repeat] \
//                      --output PATH --mode mm_projector_via_vision|mm_projector_via_audio \
//                      [--batch-size N] [--peak-rss-budget-gb N] \
//                      [--threads N]
//
// Exits 0 on success, non-zero on failure. Progress is logged
// to stderr; the JSON output is written to --output.

#include "clip-capture.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static void usage(const char * prog) {
    std::fprintf(stderr,
        "Usage: %s --model PATH --inputs PATH [repeat]\n"
        "         --output PATH --mode vision|audio\n"
        "         [--batch-size N] [--peak-rss-budget-gb N]\n"
        "         [--threads N]\n"
        "\n"
        "   or, for the mm_projector capture:\n"
        "\n"
        "%s --model PATH --mm-projector PATH --inputs PATH [repeat]\n"
        "         --output PATH --mode mm_projector_via_vision|mm_projector_via_audio\n"
        "         [--batch-size N] [--peak-rss-budget-gb N]\n"
        "         [--threads N]\n"
        "\n"
        "tessera: real forward-pass activation capture for the\n"
        "clip graph. Mirrors the imatrix CLI pattern; the\n"
        "Python multimodal_calibrate.py driver invokes this\n"
        "binary via subprocess.\n"
        "\n"
        "Options:\n"
        "  --model PATH              Path to the clip GGUF (vision\n"
        "                            tower / audio tower).\n"
        "  --mm-projector PATH       Path to the mm_projector GGUF\n"
        "                            (required for mm_projector_*\n"
        "                            modes; ignored otherwise).\n"
        "  --inputs PATH [repeat]    One or more image (vision) or\n"
        "                            audio (audio) files.\n"
        "  --output PATH             Where to write the JSON.\n"
        "  --mode {vision,audio,mm_projector_via_vision,mm_projector_via_audio}\n"
        "                            Modality (default: vision).\n"
        "  --batch-size N            Inputs per forward pass (default 1).\n"
        "                            When the input list is larger than\n"
        "                            N, the capture chunks the inputs\n"
        "                            into multiple forward calls and\n"
        "                            accumulates per-tensor stats.\n"
        "  --peak-rss-budget-gb N    Refuse to run if the estimated\n"
        "                            peak-RSS exceeds N GB (default 0 =\n"
        "                            no limit).\n"
        "  --threads N               Forward-pass thread count (default 4).\n"
        "  --help                    Show this help and exit.\n",
        prog, prog);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 64;  // EX_USAGE
    }
    const char * model = nullptr;
    const char * mm_projector = nullptr;
    const char * output = nullptr;
    std::vector<std::string> inputs;
    ts_clip_capture_mode mode = TS_CLIP_CAPTURE_MODE_VISION;
    int64_t peak_rss_budget_bytes = 0;
    int n_threads = 4;
    int batch_size = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (arg == "--mm-projector" && i + 1 < argc) {
            mm_projector = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output = argv[++i];
        } else if (arg == "--inputs" && i + 1 < argc) {
            // Collect all remaining positional args until the next
            // -- flag. This matches the convention the imatrix
            // CLI uses (one or more inputs after the flag).
            i += 1;
            while (i < argc && argv[i][0] != '-') {
                inputs.emplace_back(argv[i]);
                i += 1;
            }
            i -= 1;
        } else if (arg == "--mode" && i + 1 < argc) {
            const std::string m = argv[++i];
            if (m == "vision") {
                mode = TS_CLIP_CAPTURE_MODE_VISION;
            } else if (m == "audio") {
                mode = TS_CLIP_CAPTURE_MODE_AUDIO;
            } else if (m == "mm_projector_via_vision") {
                mode = TS_CLIP_CAPTURE_MODE_MM_PROJECTOR_VIA_VISION;
            } else if (m == "mm_projector_via_audio") {
                mode = TS_CLIP_CAPTURE_MODE_MM_PROJECTOR_VIA_AUDIO;
            } else {
                std::fprintf(stderr,
                    "llama-clip-capture: unknown --mode %s\n",
                    m.c_str());
                return 64;
            }
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = std::atoi(argv[++i]);
            if (batch_size < 1) batch_size = 1;
        } else if (arg == "--peak-rss-budget-gb" && i + 1 < argc) {
            const int gb = std::atoi(argv[++i]);
            if (gb > 0) {
                peak_rss_budget_bytes = (int64_t) gb * 1024LL
                                     * 1024LL * 1024LL;
            }
        } else if (arg == "--threads" && i + 1 < argc) {
            n_threads = std::atoi(argv[++i]);
            if (n_threads < 1) n_threads = 1;
        } else {
            std::fprintf(stderr,
                "llama-clip-capture: unknown arg %s\n",
                arg.c_str());
            usage(argv[0]);
            return 64;
        }
    }
    if (model == nullptr) {
        std::fprintf(stderr,
            "llama-clip-capture: --model is required\n");
        return 64;
    }
    if (output == nullptr) {
        std::fprintf(stderr,
            "llama-clip-capture: --output is required\n");
        return 64;
    }
    if (inputs.empty()) {
        std::fprintf(stderr,
            "llama-clip-capture: --inputs requires at least "
            "one path\n");
        return 64;
    }
    if ((mode == TS_CLIP_CAPTURE_MODE_MM_PROJECTOR_VIA_VISION ||
         mode == TS_CLIP_CAPTURE_MODE_MM_PROJECTOR_VIA_AUDIO) &&
        mm_projector == nullptr) {
        std::fprintf(stderr,
            "llama-clip-capture: --mm-projector is required "
            "for mm_projector_* modes\n");
        return 64;
    }
    std::string err;
    int rc = ts_clip_capture_activations(
            model, mm_projector, inputs, mode, output,
            batch_size, peak_rss_budget_bytes, n_threads, &err);
    if (rc != 0) {
        std::fprintf(stderr, "llama-clip-capture: failed: %s\n",
                err.c_str());
        return 1;
    }
    return 0;
}
