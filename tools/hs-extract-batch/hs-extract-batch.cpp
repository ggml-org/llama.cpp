/*
 * Batch hidden-state extraction CLI.
 *
 * Three modes:
 *   Batch:     hs-extract-batch <model> <prompts.txt> [layers] <output.bin> --batch --assignments <file> [flags]
 *   Raw:       hs-extract-batch <model> <prompts.txt> [layers] <output.bin> --raw [flags]
 *   Self-test: hs-extract-batch --self-test
 *
 * Production mode is --batch: reads prompts.txt + assignments.bin, streams
 * one prompt at a time, computes masked means per assignment, accumulates
 * per group/mask/layer, writes output.bin with final means.
 *
 * --raw is a debug/parity mode that dumps per-prompt binary data.
 *
 * Flags:
 *   --mean           In --raw: output token means instead of full per-token output
 *   --token-skip N   Skip first N tokens when computing mean (default: 0)
 *
 * Batch-only flags:
 *   --assignments F  Path to assignments.bin (required)
 *   --ctx-size N     Override auto context sizing (default: auto)
 *   --checkpoint-every N  Write checkpoint every N prompts (default: 10000)
 *   --resume         Resume from last checkpoint
 */

#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cmath>
#include <utility>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#ifndef _WIN32
#include <unistd.h>  // fsync — durability before rename (POSIX)
#endif
#include <queue>
#include <atomic>

#if defined(_OPENMP)
#define CR_SIMD _Pragma("omp simd")
#else
#define CR_SIMD
#endif

// -- Checked Write Macro ------------------------------------------------

// Wraps fwrite to check return value. On failure, prints an error and
// returns false from the calling function. Used for all output writes
// to ensure corrupt files are never silently produced.
#define CHECKED_WRITE(ptr, size, count, f)                                      \
    do {                                                                        \
        if (fwrite((ptr), (size), (count), (f)) != (size_t)(count)) {          \
            fprintf(stderr, "Error: write failed at %s:%d (expected %zu, wrote less)\n", \
                    __FILE__, __LINE__, (size_t)(count));                      \
            return false;                                                       \
        }                                                                       \
    } while (0)

// -- RAII Wrappers for Automatic Resource Cleanup -----------------------

struct FilePtr {
    FILE* fp;
    FilePtr() : fp(nullptr) {}
    FilePtr(FILE* f) : fp(f) {}
    ~FilePtr() { if (fp) fclose(fp); }
    FilePtr(const FilePtr&) = delete;
    FilePtr& operator=(const FilePtr&) = delete;
    // Move semantics (needed for stories_file pattern)
    FilePtr(FilePtr&& other) noexcept : fp(other.fp) { other.fp = nullptr; }
    FilePtr& operator=(FilePtr&& other) noexcept {
        if (this != &other) {
            if (fp) fclose(fp);
            fp = other.fp;
            other.fp = nullptr;
        }
        return *this;
    }
    operator FILE*() const { return fp; }
    explicit operator bool() const { return fp != nullptr; }
    void reset() { if (fp) { fclose(fp); fp = nullptr; } }
#ifndef _WIN32
    // Flush + fsync to guarantee data reaches durable storage before a rename.
    // fflush alone only pushes libc buffers to the kernel; fsync forces the
    // kernel to write them to disk. Without this, a power loss between rename()
    // and kernel writeback can leave the renamed file containing zeros.
    // Returns false if either step fails (disk full, I/O error) — the caller
    // MUST check this; ignoring it defeats the entire atomic-rename guarantee.
    bool sync() {
        if (!fp) return false;
        if (fflush(fp) != 0) return false;
        if (fsync(fileno(fp)) != 0) return false;
        return true;
    }
#else
    bool sync() {
        if (!fp) return false;
        if (fflush(fp) != 0) return false;
        if (_commit(_fileno(fp)) != 0) return false;
        return true;
    }
#endif
};

// RAII wrappers (LlamaModel, LlamaContext, LlamaBackend, LlamaBatch) are in
// common/llama-raii.h (shared with hs-extract, tests, examples).
#include "llama-raii.h"

// -- Named Constants (replaces magic numbers throughout) -------------------

static constexpr int32_t MIN_CTX_SIZE       = 512;     // minimum context size
static constexpr double   CHARS_PER_TOKEN   = 3.5;     // tokenizer estimate for auto-ctx
static constexpr int32_t MAX_PROMPTS        = 10000000; // max prompts in file
static constexpr int32_t MAX_N_EMBD         = 65536;   // max embedding dimension
static constexpr int32_t MAX_GROUPS         = 10000;   // max label groups
static constexpr int32_t MAX_LAYERS         = 10000;   // max layers in checkpoint
static constexpr int32_t MAX_MASKS          = 65536;   // max masks in checkpoint
static constexpr int32_t MAX_ASSIGNMENTS    = 100000;  // max assignments per prompt
static constexpr int32_t MAX_RANGES         = 100000;  // max ranges per assignment
static constexpr int32_t MAX_GROUP_NAME_LEN = 1000;    // max group name length
static constexpr int32_t ALL_GPU_LAYERS     = 99;      // load all layers on GPU
static constexpr int      PROGRESS_INTERVAL  = 100;     // prompts between progress reports
static constexpr int32_t MAX_BATCH_SIZE     = 256;     // max batch size for multi mode

// -- Argument Parsing ---------------------------------------------------

struct Args {
    const char* model_path = nullptr;
    const char* prompts_file = nullptr;
    std::string layers_str = "all";
    const char* output_path = nullptr;
    bool mean_mode = false;      // kept for --raw mean output
    int token_skip = 0;
    bool self_test = false;
    bool batch_mode = false;     // --batch: accumulate mode (production)
    bool raw_mode = false;       // --raw: per-prompt dump (debug/parity)
    const char* assignments_file = nullptr;   // --assignments: assignments.bin
    int ctx_size = 0;            // --ctx-size: override auto (0=auto)
    int checkpoint_every = 10000; // --checkpoint-every N
    bool resume = false;          // --resume: continue from checkpoint
    int n_gpu_layers = ALL_GPU_LAYERS; // --n-gpu-layers / -ngl: GPU offload (ALL_GPU_LAYERS = all)
    bool save_per_story = false;  // --save-per-story: write per-story vectors sidecar
    int batch_size = 1;           // --batch-size N: pack N prompts per decode (default: 1)
    bool profile = false;         // --profile: print per-phase timing breakdown
    bool no_bos = false;          // --no-bos: do not add BOS token (for models like Qwen3.5)
    int generate_tokens = 0;      // --generate N: autoregressive generation mode (extract from generated tokens)
    float temperature = 0.0f;     // --temperature: sampling temperature (0 = greedy argmax, the default)
    float repeat_penalty = 1.0f;  // --repeat-penalty: penalty for repeated tokens (1.0 = disabled)
    int top_k = 0;                // --top-k: top-k sampling (0 = disabled)
    float top_p = 1.0f;           // --top-p: nucleus sampling threshold (1.0 = disabled)
};

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  Batch (production): %s <model> <prompts.txt> [layers] <output.bin> --batch --assignments <file> [flags]\n", prog);
    fprintf(stderr, "  Raw (debug):        %s <model> <prompts.txt> [layers] <output.bin> --raw [flags]\n", prog);
    fprintf(stderr, "  Self-test:          %s --self-test\n", prog);
    fprintf(stderr, "\nModes:\n");
    fprintf(stderr, "  --batch          Accumulate masked means per group/mask/layer -> output.bin (production)\n");
    fprintf(stderr, "  --raw            Per-prompt dump for debugging and parity testing\n");
    fprintf(stderr, "  --self-test      Run synthetic tests on compute_masked_mean(), no model needed\n");
    fprintf(stderr, "\nFlags:\n");
    fprintf(stderr, "  --mean           In --raw mode: output token means instead of full per-token data\n");
    fprintf(stderr, "  --token-skip N   Skip first N tokens for mean computation (default: 0)\n");
    fprintf(stderr, "  --assignments F  Path to assignments.bin (required with --batch)\n");
    fprintf(stderr, "  --ctx-size N     Override auto context sizing (default: auto from prompts)\n");
    fprintf(stderr, "  --checkpoint-every N  Write checkpoint every N prompts (default: 10000)\n");
    fprintf(stderr, "  --resume         Resume from last checkpoint\n");
    fprintf(stderr, "  --n-gpu-layers N / -ngl N  Layers to offload to GPU (default: 99 = all)\n");
    fprintf(stderr, "  --save-per-story  Also write per-story vectors to <output>.stories.bin\n");
    fprintf(stderr, "  --batch-size N   Pack N prompts per decode call (default: 1, no batching)\n");
    fprintf(stderr, "  --profile       Print per-phase timing breakdown (KV clear, decode, sync, extract, mean, accumulate)\n");
    fprintf(stderr, "  --no-bos        Do not add BOS token (for models like Qwen3.5 with bos_offset=0)\n");
    fprintf(stderr, "  --generate N    Generation-based extraction: generate N tokens after each prompt,\n");
    fprintf(stderr, "                  extract hidden states from GENERATED tokens only (Anthropic/SLM methodology).\n");
    fprintf(stderr, "                  Without this flag, extraction is comprehension-based (forward pass only).\n");
    fprintf(stderr, "  --token-skip N  In generation mode: skip the first N GENERATED tokens when computing the mean.\n");
    fprintf(stderr, "                  In comprehension mode: skip the first N input tokens. Default: 0.\n");
    fprintf(stderr, "  --temperature F Sampling temperature for generation mode (0 = greedy argmax, default).\n");
    fprintf(stderr, "                  temperature > 0 reduces cross-chip divergence from greedy decoding.\n");
    fprintf(stderr, "  --top-k K       Top-k sampling (generation mode only, default: 0 = disabled).\n");
    fprintf(stderr, "  --top-p F       Nucleus sampling threshold (generation mode only, default: 1.0 = disabled).\n");
    fprintf(stderr, "  --repeat-penalty F  Repeat penalty (generation mode only, default: 1.0 = disabled).\n");
    fprintf(stderr, "\nLayers: comma-separated list (e.g. '0,5,10') or 'all' (default)\n");
    fprintf(stderr, "Output: file path (required)\n");
}

static Args parse_args(int argc, char** argv) {
    Args args;
    std::vector<const char*> positional;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--mean") {
            args.mean_mode = true;
        } else if (arg == "--token-skip") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --token-skip requires a value\n");
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0' || endptr == argv[i] || val < 0) {
                fprintf(stderr, "Error: --token-skip requires a non-negative integer, got '%s'\n", argv[i]);
                exit(1);
            }
            args.token_skip = (int)val;
        } else if (arg == "--self-test") {
            args.self_test = true;
        } else if (arg == "--batch") {
            args.batch_mode = true;
        } else if (arg == "--raw") {
            args.raw_mode = true;
        } else if (arg == "--no-bos") {
            args.no_bos = true;
        } else if (arg == "--generate") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --generate requires a value (number of tokens to generate)\n");
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0' || endptr == argv[i] || val <= 0) {
                fprintf(stderr, "Error: --generate requires a positive integer, got '%s'\n", argv[i]);
                exit(1);
            }
            args.generate_tokens = (int)val;
        } else if (arg == "--temperature") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --temperature requires a value\n");
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            args.temperature = strtof(argv[i], &endptr);
            if (*endptr != '\0' || endptr == argv[i] || args.temperature < 0.0f) {
                fprintf(stderr, "Error: --temperature requires a non-negative float, got '%s'\n", argv[i]);
                exit(1);
            }
        } else if (arg == "--repeat-penalty") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --repeat-penalty requires a value\n");
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            args.repeat_penalty = strtof(argv[i], &endptr);
            if (*endptr != '\0' || endptr == argv[i] || args.repeat_penalty < 1.0f) {
                fprintf(stderr, "Error: --repeat-penalty requires a float >= 1.0, got '%s'\n", argv[i]);
                exit(1);
            }
        } else if (arg == "--top-k") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --top-k requires a value\n");
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0' || endptr == argv[i] || val < 0) {
                fprintf(stderr, "Error: --top-k requires a non-negative integer, got '%s'\n", argv[i]);
                exit(1);
            }
            args.top_k = (int)val;
        } else if (arg == "--top-p") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --top-p requires a value\n");
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            args.top_p = strtof(argv[i], &endptr);
            if (*endptr != '\0' || endptr == argv[i] || args.top_p <= 0.0f || args.top_p > 1.0f) {
                fprintf(stderr, "Error: --top-p requires a float in (0, 1], got '%s'\n", argv[i]);
                exit(1);
            }
        } else if (arg == "--assignments") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --assignments requires a value\n");
                exit(1);
            }
            args.assignments_file = argv[++i];
        } else if (arg == "--ctx-size") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --ctx-size requires a value\n");
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0' || endptr == argv[i] || val < 0) {
                fprintf(stderr, "Error: --ctx-size requires a non-negative integer, got '%s'\n", argv[i]);
                exit(1);
            }
            args.ctx_size = (int)val;
        } else if (arg == "--checkpoint-every") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --checkpoint-every requires a value\n");
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0' || endptr == argv[i] || val <= 0) {
                fprintf(stderr, "Error: --checkpoint-every requires a positive integer, got '%s'\n", argv[i]);
                exit(1);
            }
            args.checkpoint_every = (int)val;
        } else if (arg == "--resume") {
            args.resume = true;
        } else if (arg == "--save-per-story") {
            args.save_per_story = true;
        } else if (arg == "--profile") {
            args.profile = true;
        } else if (arg == "--batch-size") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --batch-size requires a value\n");
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0' || endptr == argv[i] || val < 1 || val > MAX_BATCH_SIZE) {
                fprintf(stderr, "Error: --batch-size requires an integer in [1, %d], got '%s'\n", MAX_BATCH_SIZE, argv[i]);
                exit(1);
            }
            args.batch_size = (int)val;
        } else if (arg == "--n-gpu-layers" || arg == "-ngl") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires a value\n", arg.c_str());
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0' || endptr == argv[i] || val < 0) {
                fprintf(stderr, "Error: %s requires a non-negative integer, got '%s'\n", arg.c_str(), argv[i]);
                exit(1);
            }
            args.n_gpu_layers = (int)val;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        } else {
            positional.push_back(argv[i]);
        }
    }

    // --self-test needs no model or prompts -- skip positional validation
    if (args.self_test) {
        return args;
    }

    // --mean is only consumed in --raw mode; run_batch ignores it. Reject the
    // combination so the user gets a clear error instead of silently receiving
    // batch output without mean pooling.
    if (args.batch_mode && args.mean_mode) {
        fprintf(stderr, "Error: --mean is not supported with --batch (it only applies to --raw mode)\n");
        exit(1);
    }

    // --token-skip >= --generate silently produces empty output: the gate
    // `gen_step >= token_skip` never passes, every layer's gen_count stays 0,
    // every layer is skipped, and output.bin gets a header with no data while
    // the run exits 0. Reject it loudly rather than waste a 239K-prompt run.
    if (args.generate_tokens > 0 && args.token_skip >= args.generate_tokens) {
        fprintf(stderr,
                "Error: --token-skip (%d) must be < --generate (%d); otherwise every "
                "generated token is skipped and the output is empty.\n",
                args.token_skip, args.generate_tokens);
        exit(1);
    }

    // --batch mode has its own positional layout: [model] [prompts] [layers] [output]
    if (args.batch_mode) {
        if (positional.empty()) {
            fprintf(stderr, "Error: --batch requires a model path\n");
            print_usage(argv[0]);
            exit(1);
        }
        args.model_path = positional[0];
        if (positional.size() >= 2) args.prompts_file = positional[1];
        if (positional.size() >= 3) args.layers_str = positional[2];
        if (positional.size() >= 4) args.output_path = positional[3];

        if (!args.prompts_file) {
            fprintf(stderr, "Error: --batch requires a prompts file\n");
            print_usage(argv[0]);
            exit(1);
        }
        if (!args.assignments_file) {
            fprintf(stderr, "Error: --batch requires --assignments <file>\n");
            print_usage(argv[0]);
            exit(1);
        }
        if (!args.output_path) {
            fprintf(stderr, "Error: --batch requires an output file path\n");
            print_usage(argv[0]);
            exit(1);
        }
        return args;
    }

    // --raw mode: [model] [prompts] [layers] [output]
    if (args.raw_mode) {
        if (positional.empty()) {
            fprintf(stderr, "Error: --raw requires a model path\n");
            print_usage(argv[0]);
            exit(1);
        }
        args.model_path = positional[0];
        if (positional.size() >= 2) args.prompts_file = positional[1];
        if (positional.size() >= 3) args.layers_str = positional[2];
        if (positional.size() >= 4) args.output_path = positional[3];

        if (!args.prompts_file) {
            fprintf(stderr, "Error: --raw requires a prompts file\n");
            print_usage(argv[0]);
            exit(1);
        }
        if (!args.output_path) {
            fprintf(stderr, "Error: --raw requires an output file path\n");
            print_usage(argv[0]);
            exit(1);
        }
        return args;
    }

    // No mode specified -- show usage and exit
    fprintf(stderr, "Error: must specify --batch, --raw, or --self-test\n\n");
    print_usage(argv[0]);
    exit(1);
}

// Wraps fwrite for per-story sidecar writes. On failure, prints error,
// signals producer to stop, joins producer thread, and returns 1 from the calling function.
// All captured state is passed explicitly — no implicit scope capture.
#define STORIES_WRITE(ptr, size, count, fp, pfq_ref, thread_ref)               \
    do {                                                                        \
        if (fwrite((ptr), (size), (count), (fp)) != (size_t)(count)) {         \
            fprintf(stderr, "Error: per-story write failed at %s:%d\n",         \
                    __FILE__, __LINE__);                                        \
            { std::lock_guard<std::mutex> _lk((pfq_ref).mtx); (pfq_ref).producer_done = true; } \
            (pfq_ref).cv_space.notify_all();                                    \
            (thread_ref).join();                                                \
            return 1;                                                           \
        }                                                                       \
    } while (0)

// Stop the producer thread cleanly from a consumer error path.
// Sets producer_done (unblocks a producer waiting on cv_space backpressure)
// and joins the thread. Used at every consumer error exit point.
//
// B11: also remove the per-story temp file if one is open. Every consumer-loop
// error exit goes through this macro, and a leftover .stories.bin.tmp would be
// picked up by a subsequent run's parser or accumulate across failed retries.
// `stories_temp_path` is an empty std::string when --save-per-story is not set,
// so std::remove on "" is a harmless no-op (returns ENOENT).
#define STOP_PRODUCER_AND_JOIN()                                                \
    do {                                                                        \
        { std::lock_guard<std::mutex> _lk(pfq.mtx); pfq.producer_done = true; } \
        pfq.cv_space.notify_all();                                              \
        producer_thread.join();                                                 \
        llama_synchronize(ctx);                                                 \
        if (!stories_temp_path.empty()) std::remove(stories_temp_path.c_str()); \
    } while (0)

// -- Batch Mode Constants & Data Structures -----------------------------

static constexpr int32_t ASSIGNMENTS_MAGIC = 0x43524431;  // "CRD1"
static constexpr int32_t OUTPUT_MAGIC      = 0x43524432;  // binary accumulator format v2

// Per (group, mask, layer): running sum and count for accumulating means.
// Flat key: (group_id << 24) | (mask_id << 16) | layer_idx for single hash lookup.
struct AccumulatedVector {
    std::vector<float> sum;  // size n_embd, initialized lazily on first access
    int count = 0;
};

// Flat accumulator map: single-level hash for better cache locality than nested maps.
using AccumulatorMap = std::unordered_map<uint64_t, AccumulatedVector>;

// Helper to construct flat key from components.
// Layout: bits 0-15 = layer_idx, bits 16-31 = mask_id, bits 32-63 = group_id
inline uint64_t make_accum_key(int32_t group_id, int32_t mask_id, int32_t layer_idx) {
    return ((uint64_t)(uint32_t)group_id << 32) | ((uint64_t)(uint16_t)mask_id << 16) | (uint64_t)(uint16_t)layer_idx;
}

// Helper to extract components from flat key.
inline void decode_accum_key(uint64_t key, int32_t& group_id, int32_t& mask_id, int32_t& layer_idx) {
    layer_idx = (int32_t)(key & 0xFFFF);
    mask_id   = (int32_t)((key >> 16) & 0xFFFF);
    group_id  = (int32_t)((key >> 32) & 0xFFFFFFFF);
}

// One (group, mask) assignment for a prompt.
struct Assignment {
    int32_t group_id;
    int32_t mask_id;
    int32_t mask_type;    // 0 = simple_skip, 1 = explicit_ranges
    int32_t skip;         // valid when mask_type == 0
    std::vector<std::pair<int,int>> ranges;  // valid when mask_type == 1
};

// Group name table from assignments.bin header.
struct GroupTable {
    int32_t n_groups = 0;
    std::vector<std::string> names;  // indexed by group_id
};

enum class AssignmentReadStatus {
    ok,
    eof,
    error,
};

struct AssignmentReadResult {
    AssignmentReadStatus status = AssignmentReadStatus::error;
    std::vector<Assignment> assignments;
};

// -- Layer Parsing ------------------------------------------------------

static std::vector<int> parse_layers(const std::string& s, int n_layer) {
    std::vector<int> out;
    if (s == "all") {
        for (int i = 0; i < n_layer; i++) out.push_back(i);
        return out;
    }
    const char* p = s.c_str();
    while (*p) {
        char* endptr = nullptr;
        errno = 0;
        long val = strtol(p, &endptr, 10);
        if (endptr == p) {
            fprintf(stderr, "Error: invalid layer '%c' in layers string '%s'\n", *p, s.c_str());
            return {};  // caller checks for empty target_layers
        }
        if (errno == ERANGE) {
            fprintf(stderr, "Error: layer value out of range in '%s'\n", s.c_str());
            return {};
        }
        // B13: validate at parse time, not downstream. Negative indices are
        // resolved later (l < 0 → l += n_layer), but must be in [-n_layer, n_layer-1].
        if (val >= n_layer || val < -n_layer) {
            fprintf(stderr, "Error: layer %ld out of range [0, %d) or [%d, -1]\n", val, n_layer, -n_layer);
            return {};
        }
        out.push_back((int)val);
        while (*endptr && *endptr != ',') endptr++;
        if (*endptr == ',') endptr++;
        p = endptr;
    }
    return out;
}

// -- Masked Mean Computation --------------------------------------------

/**
 * Compute mean of hidden state data over specified token ranges.
 *
 * Generalizes the single-contiguous-range mean (token_skip -> n_tokens) to
 * arbitrary collections of [start, end) token index pairs. Used by both the
 * refactored one-shot/persistent mean mode (single range) and the future
 * batch-accumulate mode (arbitrary ranges for deflection masking).
 *
 * @param data      Pointer to hidden state data, shape (n_tokens, n_embd).
 * @param n_tokens  Number of tokens in the sequence.
 * @param n_embd    Hidden dimension size.
 * @param ranges    Vector of (start, end) token index pairs.
 * @param out       Output buffer, size n_embd. Must be zeroed by caller.
 * @return          Number of tokens included in the mean (0 = empty mask).
 */
static int64_t compute_masked_mean(
    const float* data,
    int n_tokens,
    int n_embd,
    const std::vector<std::pair<int,int>>& ranges,
    float* out
) {
    int64_t count = 0;
    for (auto [start, end] : ranges) {
        // All range violations are hard errors. The Python side must produce
        // correct ranges  -  no silent clamping, no graceful degradation.
        if (start < 0 || end < 0 || start >= end) {
            fprintf(stderr, "Error: invalid masked-mean range [%d, %d)  -  negative or empty range\n", start, end);
            return -1;
        }
        if (end > n_tokens || start > n_tokens) {
            fprintf(stderr, "Error: masked-mean range [%d, %d) exceeds n_tokens=%d  -  "
                            "token span computation is incorrect (check BOS_OFFSET in dialogue_utils.py)\n",
                    start, end, n_tokens);
            return -1;
        }

        for (int t = start; t < end; t++) {
            const float* row = data + (size_t)t * (size_t)n_embd;
            CR_SIMD
            for (int d = 0; d < n_embd; d++) {
                out[d] += row[d];
            }
        }
        count += (end - start);
    }

    if (count > 0) {
        float inv = 1.0f / (float)count;
        CR_SIMD
        for (int d = 0; d < n_embd; d++) {
            out[d] *= inv;
        }
    }
    return count;
}

static int compute_single_range_mean(
    const float* data,
    int n_tokens,
    int n_embd,
    int start,
    int end,
    float* out
) {
    // All range violations are hard errors  -  no silent clamping.
    if (start < 0 || end < 0 || start >= end) {
        fprintf(stderr, "Error: invalid single-range [%d, %d)  -  negative or empty range\n", start, end);
        return -1;
    }
    if (end > n_tokens || start > n_tokens) {
        fprintf(stderr, "Error: single-range [%d, %d) exceeds n_tokens=%d  -  "
                        "token span computation is incorrect\n", start, end, n_tokens);
        return -1;
    }

    for (int t = start; t < end; t++) {
        const float* row = data + (size_t)t * (size_t)n_embd;
        CR_SIMD
        for (int d = 0; d < n_embd; d++) {
            out[d] += row[d];
        }
    }

    const int count = end - start;
    float inv = 1.0f / (float)count;
    CR_SIMD
    for (int d = 0; d < n_embd; d++) {
        out[d] *= inv;
    }
    return count;
}

// -- Prompt Processing --------------------------------------------------

/**
 * Process a single prompt: decode, synchronize, extract hidden states, write to file.
 *
 * In full mode: writes n_tokens * hidden_size floats per layer.
 * In mean mode: writes hidden_size floats per layer (mean from token_skip to n_tokens).
 *
 * Returns true on success, false on fatal error (decode failure, I/O error).
 */
static bool process_prompt(
    llama_context* ctx,
    llama_batch& batch,
    float* mean_buf,
    const std::vector<llama_token>& tokens,
    const std::vector<int32_t>& target_layers,
    int32_t n_embd,
    size_t prompt_idx,
    FILE* out,
    bool mean_mode,
    int token_skip
) {
    int n_tokens = (int)tokens.size();

    if (n_tokens <= 0) {
        fprintf(stderr, "Error: prompt %zu tokenized to empty\n", prompt_idx);
        return false;
    }

    // Clear KV cache before processing this prompt
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }

    // Fill pre-allocated batch buffer (reused across prompts to avoid heap fragmentation)
    // seq_id=0 (KV cache is cleared per prompt, so no collision)
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 0;  // No logits needed  -  we only extract hidden states
    }
    batch.n_tokens = n_tokens;

    // Decode
    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        fprintf(stderr, "Error: decode failed for prompt %zu (ret=%d)\n", prompt_idx, ret);
        return false;
    }

    // CRITICAL: synchronize before reading hidden states.
    // llama_decode() submits the compute graph asynchronously on CUDA.
    // Without sync, llama_get_hidden_state() can read a partially-written
    // tensor, triggering GGML_ASSERT (tensor read out of bounds).
    llama_synchronize(ctx);

    // Extract hidden state metadata
    int n_hidden_tokens = llama_get_hidden_state_n_tokens(ctx);
    if (n_hidden_tokens <= 0) {
        fprintf(stderr, "Error: decode produced no hidden state tokens for prompt %zu\n", prompt_idx);
        return false;
    }

    // Write per-prompt header: [prompt_idx][n_tokens][n_layers][layer_indices...]
    std::vector<int32_t> header(3 + target_layers.size());
    header[0] = (int32_t)prompt_idx;
    header[1] = n_hidden_tokens;
    header[2] = (int32_t)target_layers.size();
    for (size_t i = 0; i < target_layers.size(); i++) {
        header[3 + i] = target_layers[i];
    }

    CHECKED_WRITE(header.data(), sizeof(int32_t), 3 + target_layers.size(), out);

    // Write hidden state data for each layer
    for (int32_t layer : target_layers) {
        float* data = llama_get_hidden_state(ctx, layer);
        if (!data) {
            fprintf(stderr, "Error: null hidden state for layer %d\n", layer);
            return false;
        }

        if (mean_mode) {
            std::fill(mean_buf, mean_buf + n_embd, 0.0f);
            if (compute_single_range_mean(data, n_hidden_tokens, n_embd, token_skip, n_hidden_tokens, mean_buf) < 0) {
                fprintf(stderr, "Error: compute_single_range_mean failed for prompt %zu\n", prompt_idx);
                return false;
            }

            CHECKED_WRITE(mean_buf, sizeof(float), n_embd, out);
        } else {
            // Full mode: write all tokens x hidden_size
            size_t data_size = (size_t)n_hidden_tokens * (size_t)n_embd;
            CHECKED_WRITE(data, sizeof(float), data_size, out);
        }
    }

    return true;
}

// -- Tokenization -------------------------------------------------------

/**
 * Tokenize a prompt. Returns token vector (empty on failure).
 */
static std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text, bool add_bos = true) {
    int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_bos, true);
    if (n <= 0) return {};
    std::vector<llama_token> toks(n);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), (int)toks.size(), add_bos, true);
    if (n_tokens > 0) {
        toks.resize(n_tokens);
    } else {
        toks.clear();
        return toks;
    }

    // Validate token bounds (C6: prevent crashes from invalid token IDs)
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token tok : toks) {
        if (tok < 0 || tok >= n_vocab) {
            fprintf(stderr, "Error: tokenizer returned out-of-bounds token %d (vocab size: %d)\n", tok, n_vocab);
            return {};
        }
    }
    return toks;
}

// -- Batch Mode Helpers -------------------------------------------------

/**
 * Pre-scan prompts.txt to estimate max token count and auto-size n_ctx.
 * Uses chars / 3.5 as token estimate. Returns max(estimated_max + 64, 512).
 */
static bool auto_size_ctx(const char* prompts_file, int32_t& n_ctx) {
    std::ifstream fin(prompts_file);
    if (!fin) {
        fprintf(stderr, "Error: cannot open %s for ctx pre-scan\n", prompts_file);
        return false;
    }

    size_t max_len = 0;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.size() > max_len) max_len = line.size();
    }

    int32_t estimated = (int32_t)(max_len / CHARS_PER_TOKEN) + 64;
    n_ctx = estimated > MIN_CTX_SIZE ? estimated : MIN_CTX_SIZE;
    fprintf(stderr, "Auto-ctx: max line %zu chars -> estimated %d tokens -> n_ctx=%d\n",
            max_len, (int)(max_len / CHARS_PER_TOKEN), n_ctx);
    return true;
}

/**
 * Read assignments.bin header: magic, n_prompts, n_embd, group name table.
 * Leaves file position at the start of per-prompt data.
 * Returns true on success.
 */
static bool read_assignments_header(
    FILE* f, int32_t& n_prompts, int32_t& n_embd_expected, GroupTable& groups
) {
    int32_t magic = 0;
    if (fread(&magic, sizeof(int32_t), 1, f) != 1 || magic != ASSIGNMENTS_MAGIC) {
        fprintf(stderr, "Error: invalid assignments.bin magic (got 0x%08X, expected 0x%08X)\n",
                magic, ASSIGNMENTS_MAGIC);
        return false;
    }
    if (fread(&n_prompts, sizeof(int32_t), 1, f) != 1) return false;
    if (fread(&n_embd_expected, sizeof(int32_t), 1, f) != 1) return false;

    // Validate n_prompts and n_embd to prevent UB on corrupt file
    if (n_prompts <= 0 || n_prompts > MAX_PROMPTS) {
        fprintf(stderr, "Error: n_prompts %d out of range [1, %d] - corrupt assignments.bin\n", n_prompts, MAX_PROMPTS);
        return false;
    }
    if (n_embd_expected < 0 || n_embd_expected > MAX_N_EMBD) {
        fprintf(stderr, "Error: n_embd %d out of range [0, %d] - corrupt assignments.bin\n", n_embd_expected, MAX_N_EMBD);
        return false;
    }

    if (fread(&groups.n_groups, sizeof(int32_t), 1, f) != 1) return false;

    // Validate n_groups to prevent OOM on corrupt file
    if (groups.n_groups < 0 || groups.n_groups > MAX_GROUPS) {
        fprintf(stderr, "Error: n_groups %d out of range [0, %d] - corrupt assignments.bin\n", groups.n_groups, MAX_GROUPS);
        return false;
    }
    groups.names.resize(groups.n_groups);
    for (int32_t i = 0; i < groups.n_groups; i++) {
        int32_t name_len = 0;
        if (fread(&name_len, sizeof(int32_t), 1, f) != 1) return false;
        // Validate name_len to prevent OOM on corrupt file
        if (name_len < 0 || name_len > MAX_GROUP_NAME_LEN) {
            fprintf(stderr, "Error: group name length %d out of range [0, %d] - corrupt assignments.bin\n", name_len, MAX_GROUP_NAME_LEN);
            return false;
        }
        std::vector<char> buf(name_len > 0 ? name_len : 1, 0);
        if (name_len > 0 && fread(buf.data(), 1, name_len, f) != (size_t)name_len) return false;
        groups.names[i] = std::string(buf.data(), name_len);
    }

    fprintf(stderr, "Assignments: %d prompts, %d groups, n_embd_expected=%d\n",
            n_prompts, groups.n_groups, n_embd_expected);
    return true;
}

/**
 * Read one prompt's assignments from assignments.bin (sequential read).
 * Must be called in prompt order, matching prompts.txt line order.
 * Returns explicit status so EOF, valid zero-assignment prompts, and truncated
 * records cannot be confused.
 */
static AssignmentReadResult read_prompt_assignments(FILE* f) {
    int32_t n_assignments = 0;
    if (fread(&n_assignments, sizeof(int32_t), 1, f) != 1) {
        if (feof(f)) return {AssignmentReadStatus::eof, {}};
        fprintf(stderr, "Error: failed to read n_assignments at offset %ld\n", ftell(f));
        return {AssignmentReadStatus::error, {}};
    }

    if (n_assignments < 0 || n_assignments > MAX_ASSIGNMENTS) {
        fprintf(stderr, "Error: n_assignments %d out of range [0, %d]\n", n_assignments, MAX_ASSIGNMENTS);
        return {AssignmentReadStatus::error, {}};
    }

    std::vector<Assignment> assignments(n_assignments);
    for (int32_t i = 0; i < n_assignments; i++) {
        Assignment& a = assignments[i];
        if (fread(&a.group_id, sizeof(int32_t), 1, f) != 1) return {AssignmentReadStatus::error, {}};
        if (fread(&a.mask_id, sizeof(int32_t), 1, f) != 1) return {AssignmentReadStatus::error, {}};
        if (fread(&a.mask_type, sizeof(int32_t), 1, f) != 1) return {AssignmentReadStatus::error, {}};

        if (a.mask_type == 0) {
            if (fread(&a.skip, sizeof(int32_t), 1, f) != 1) return {AssignmentReadStatus::error, {}};
        } else if (a.mask_type == 1) {
            int32_t n_ranges = 0;
            if (fread(&n_ranges, sizeof(int32_t), 1, f) != 1) return {AssignmentReadStatus::error, {}};
            if (n_ranges < 0 || n_ranges > MAX_RANGES) {
                fprintf(stderr, "Error: n_ranges %d out of range [0, %d]\n", n_ranges, MAX_RANGES);
                return {AssignmentReadStatus::error, {}};
            }
            a.ranges.resize(n_ranges);
            for (int32_t r = 0; r < n_ranges; r++) {
                int32_t start = 0, end = 0;
                if (fread(&start, sizeof(int32_t), 1, f) != 1) return {AssignmentReadStatus::error, {}};
                if (fread(&end, sizeof(int32_t), 1, f) != 1) return {AssignmentReadStatus::error, {}};
                if (start < 0 || end < 0) {
                    fprintf(stderr, "Error: assignment range [%d, %d) has negative bound  -  corrupt assignments.bin\n", start, end);
                    return {AssignmentReadStatus::error, {}};
                }
                if (start > end) {
                    fprintf(stderr, "Error: assignment range [%d, %d) has start > end  -  corrupt assignments.bin\n", start, end);
                    return {AssignmentReadStatus::error, {}};
                }
                if (start == end) {
                    // M4: compute_masked_mean treats start >= end as a hard error, so
                    // an empty range would abort the entire run deep in the kernel.
                    // Reject it here at read time with a clear diagnostic instead.
                    fprintf(stderr, "Error: assignment range [%d, %d) is empty (start == end)  -  "
                                    "degenerate token span in assignments.bin\n", start, end);
                    return {AssignmentReadStatus::error, {}};
                }
                a.ranges[r] = {start, end};
            }
        } else {
            fprintf(stderr, "Error: unknown mask_type %d\n", a.mask_type);
            return {AssignmentReadStatus::error, {}};
        }
    }
    return {AssignmentReadStatus::ok, std::move(assignments)};
}

// -- Raw Mode (debug/parity) --------------------------------------------

/**
 * Raw mode: per-prompt binary dump for debugging and parity testing.
 *
 * Streams prompts.txt, decodes each, writes per-prompt hidden state data.
 * Same output format as the old one-shot mode:
 *   [n_prompts: int32]
 *   per prompt: [prompt_idx][n_tokens][n_layers][layer_indices][data]
 *
 * With --mean: writes mean vectors (token_skip->n_tokens) per layer.
 * Without --mean: writes full per-token data (n_tokens x hidden_size per layer).
 */
static int run_raw(const Args& args) {
    LlamaBackend backend;
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;
    LlamaModel model(llama_model_load_from_file(args.model_path, mparams));
    if (!model) {
        fprintf(stderr, "Error: failed to load model %s\n", args.model_path);
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int32_t n_ctx_train = llama_model_n_ctx_train(model);
    const int32_t n_embd = llama_model_n_embd_out(model);
    const int32_t n_layers = llama_model_n_layer(model);

    if (n_embd <= 0) {
        fprintf(stderr, "Error: invalid model n_embd=%d (model corrupt or unsupported)\n", n_embd);
        return 1;
    }

    fprintf(stderr, "Model loaded: n_ctx_train=%d, n_embd=%d, n_layers=%d, n_gpu_layers=%d\n",
            n_ctx_train, n_embd, n_layers, args.n_gpu_layers);

    // Resolve layers
    auto raw_layers = parse_layers(args.layers_str, n_layers);
    std::vector<int32_t> target_layers;
    for (int l : raw_layers) {
        if (l < 0) l = n_layers + l;
        if (l >= 0 && l < n_layers) {
            target_layers.push_back(l);
        } else {
            fprintf(stderr, "Error: layer %d out of range [0, %d)\n", l, n_layers);
            return 1;
        }
    }
    if (target_layers.empty()) {
        fprintf(stderr, "Error: no valid target layers\n");
        return 1;
    }

    // Auto-size context or use --ctx-size
    int32_t n_ctx;
    if (args.ctx_size > 0) {
        n_ctx = args.ctx_size;
    } else if (!auto_size_ctx(args.prompts_file, n_ctx)) {
        return 1;
    }
    if (n_ctx > n_ctx_train) n_ctx = n_ctx_train;
    fprintf(stderr, "Using n_ctx=%d\n", n_ctx);

    // Create context FIRST (Qwen3.5 Gated Delta Net OOM fix)
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;
    cparams.n_ubatch = n_ctx;  // Match run_batch: process full context in one ubatch pass
    cparams.extract_hidden_states = true;

    LlamaContext ctx(llama_init_from_model(model, cparams));
    if (!ctx) {
        fprintf(stderr, "Error: failed to create context\n");
        return 1;
    }

    // Open output file. Write to a temp path and atomically rename at the end
    // (P2-2 fix), matching the batch/checkpoint/stories atomicity guarantees.
    // A crash mid-run previously left a truncated raw dump at the final path,
    // which the Python parser would misread. Now a crash leaves only a .tmp
    // file that is unlinked on the error paths below.
    std::string tmp_path = std::string(args.output_path) + ".tmp";
    FilePtr out(fopen(tmp_path.c_str(), "wb"));
    if (!out) {
        fprintf(stderr, "Error: cannot open output file %s\n", tmp_path.c_str());
        return 1;
    }
    fprintf(stderr, "Writing output to %s\n", args.output_path);

    // Stream prompts.txt -- count lines for header first
    // Two-pass: first count lines, then process
    int32_t n_prompts_total = 0;
    {
        std::ifstream count_fin(args.prompts_file);
        std::string line;
        while (std::getline(count_fin, line)) {
            if (!line.empty()) n_prompts_total++;
        }
    }

    // Global header
    if (fwrite(&n_prompts_total, sizeof(int32_t), 1, out) != 1) {
        fprintf(stderr, "Error: failed to write raw output header\n");
        out.reset();
        std::remove(tmp_path.c_str());
        return 1;
    }

    // Process prompts (streaming)
    std::ifstream fin(args.prompts_file);
    auto start_time = std::chrono::steady_clock::now();
    int prompt_idx = 0;

    // Pre-allocate batch and mean buffer once (avoid per-prompt heap fragmentation)
    LlamaBatch batch_wrapper;
    batch_wrapper.init(n_ctx, 0, 1);
    std::vector<float> mean_buf(n_embd, 0.0f);

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) {
            // Skip empty lines -- matches the counting pass (lines 902-906).
            // The header n_prompts_total already excludes these.
            continue;
        }

        auto tokens = tokenize(vocab, line, /*add_bos=*/!args.no_bos);
        if (tokens.empty()) {
            fprintf(stderr, "Error: prompt %d tokenized to empty\n", prompt_idx);
            out.reset();
            std::remove(tmp_path.c_str());
            return 1;
        }
        if ((int)tokens.size() > n_ctx) {
            fprintf(stderr, "Error: prompt %d has %zu tokens, exceeds n_ctx=%d\n",
                    prompt_idx, tokens.size(), n_ctx);
            out.reset();
            std::remove(tmp_path.c_str());
            return 1;
        }

        if (!process_prompt(ctx, batch_wrapper.batch, mean_buf.data(),
                            tokens, target_layers, n_embd, prompt_idx, out,
                            args.mean_mode, args.token_skip)) {
            fprintf(stderr, "Error: process_prompt failed at prompt %d\n", prompt_idx);
            out.reset();
            std::remove(tmp_path.c_str());
            return 1;  // fatal I/O error
        }

        prompt_idx++;

        if (prompt_idx % PROGRESS_INTERVAL == 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            float pps = prompt_idx * 1000.0f / (elapsed > 0 ? elapsed : 1);
            fprintf(stderr, "Processed %d/%d prompts (%.2f prompts/sec)\n",
                    prompt_idx, n_prompts_total, pps);
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    fprintf(stderr, "Done. Processed %d prompts in %.2f seconds (%.2f prompts/sec)\n",
            prompt_idx, total_ms / 1000.0,
            prompt_idx * 1000.0 / (total_ms > 0 ? total_ms : 1));

    // Atomic finalize: sync to durable storage, close, then rename into place.
    // The final path appears only when the write is complete (rename is atomic
    // on POSIX), so a kill mid-write can never leave a truncated raw dump.
    if (!out.sync()) {   // fflush + fsync(fileno) — check for disk-full / I/O errors
        fprintf(stderr, "Error: sync failed for %s (disk full or I/O error)\n", tmp_path.c_str());
        out.reset();
        std::remove(tmp_path.c_str());
        return 1;
    }
    out.reset();  // fclose before rename
    if (rename(tmp_path.c_str(), args.output_path) != 0) {
        fprintf(stderr, "Error: cannot rename %s to %s\n", tmp_path.c_str(), args.output_path);
        std::remove(tmp_path.c_str());
        return 1;
    }

    return 0;
}

// -- Batch-Accumulate Output Writer -------------------------------------

/**
 * Write accumulator state to an open FILE* in binary accumulator format.
 * Shared core used by both write_batch_output() and write_checkpoint().
 * Returns false on write error.
 */
static bool _write_accumulator_to_file(
    const AccumulatorMap& accumulators,
    FILE* out,
    int32_t n_embd,
    bool write_sum = false
) {
    // Collect and sort flat keys for deterministic output.
    // Key layout: group_id (bits 32-63) | mask_id (bits 16-31) | layer_idx (bits 0-15)
    // Sorting the full 64-bit key yields: group_id ASC, mask_id ASC, layer_idx ASC.
    std::vector<uint64_t> flat_keys;
    flat_keys.reserve(accumulators.size());
    for (const auto& [key, _] : accumulators) flat_keys.push_back(key);
    std::sort(flat_keys.begin(), flat_keys.end());

    // Compute n_layers (max layer_idx + 1)
    int32_t max_layer = 0;
    for (uint64_t key : flat_keys) {
        int32_t group_id, mask_id, layer_idx;
        decode_accum_key(key, group_id, mask_id, layer_idx);
        if (layer_idx > max_layer) max_layer = layer_idx;
    }

    // Build unique (group_id, mask_id) pairs, each with its sorted layer indices.
    // Since flat_keys is sorted, sequential iteration naturally groups identical
    // (group, mask) pairs contiguously and layers arrive in ascending order.
    struct GroupMask {
        int32_t group_id;
        int32_t mask_id;
        std::vector<int32_t> layer_indices;
    };
    std::vector<GroupMask> gm_pairs;

    for (uint64_t key : flat_keys) {
        int32_t group_id, mask_id, layer_idx;
        decode_accum_key(key, group_id, mask_id, layer_idx);

        bool new_pair = gm_pairs.empty()
            || gm_pairs.back().group_id != group_id
            || gm_pairs.back().mask_id   != mask_id;
        if (new_pair) {
            gm_pairs.push_back({group_id, mask_id, {}});
        }
        gm_pairs.back().layer_indices.push_back(layer_idx);
    }

    // Count distinct groups (first element of each new group in gm_pairs)
    int32_t n_groups = 0;
    {
        int32_t prev_group = -1;
        for (const auto& gm : gm_pairs) {
            if (gm.group_id != prev_group) { n_groups++; prev_group = gm.group_id; }
        }
    }

    // Write header
    int32_t magic = OUTPUT_MAGIC;
    CHECKED_WRITE(&magic, sizeof(int32_t), 1, out);
    CHECKED_WRITE(&n_groups, sizeof(int32_t), 1, out);
    int32_t n_layers = max_layer + 1;
    CHECKED_WRITE(&n_layers, sizeof(int32_t), 1, out);
    CHECKED_WRITE(&n_embd, sizeof(int32_t), 1, out);

    // Write per-group data: iterate over groups, and within each group
    // over its (group, mask) pairs. Each mask block is emitted exactly once.
    std::vector<float> mean(n_embd);
    size_t gm_idx = 0;

    while (gm_idx < gm_pairs.size()) {
        int32_t group_id = gm_pairs[gm_idx].group_id;

        // Count how many mask pairs belong to this group
        int32_t n_masks = 0;
        size_t mask_start = gm_idx;
        while (gm_idx < gm_pairs.size() && gm_pairs[gm_idx].group_id == group_id) {
            n_masks++;
            gm_idx++;
        }

        CHECKED_WRITE(&group_id, sizeof(int32_t), 1, out);
        CHECKED_WRITE(&n_masks, sizeof(int32_t), 1, out);

        for (size_t mi = mask_start; mi < mask_start + (size_t)n_masks; mi++) {
            const auto& gm = gm_pairs[mi];
            CHECKED_WRITE(&gm.mask_id, sizeof(int32_t), 1, out);

            int32_t n_layers_data = (int32_t)gm.layer_indices.size();
            CHECKED_WRITE(&n_layers_data, sizeof(int32_t), 1, out);

            for (int32_t li : gm.layer_indices) {
                uint64_t layer_key = make_accum_key(group_id, gm.mask_id, li);
                const auto& av = accumulators.at(layer_key);
                CHECKED_WRITE(&li, sizeof(int32_t), 1, out);
                CHECKED_WRITE(&av.count, sizeof(int32_t), 1, out);

                if (write_sum) {
                    // Checkpoint format (v2+): write raw sum directly to avoid
                    // precision loss from mean=sum/count then sum=mean*count roundtrip.
                    if (av.count > 0 && !av.sum.empty()) {
                        CHECKED_WRITE(av.sum.data(), sizeof(float), n_embd, out);
                    } else {
                        std::fill(mean.begin(), mean.end(), 0.0f);
                        CHECKED_WRITE(mean.data(), sizeof(float), n_embd, out);
                    }
                } else {
                    // Output format: write mean = sum / count for downstream consumers.
                    if (av.count > 0 && !av.sum.empty()) {
                        float inv = 1.0f / (float)av.count;
                        for (int d = 0; d < n_embd; d++) mean[d] = av.sum[d] * inv;
                    } else {
                        std::fill(mean.begin(), mean.end(), 0.0f);
                    }
                    CHECKED_WRITE(mean.data(), sizeof(float), n_embd, out);
                }
            }
        }
    }
    return true;
}

/**
 * Write accumulated means to output.bin (binary accumulator format).
 */
static bool write_batch_output(
    const AccumulatorMap& accumulators,
    const char* output_path,
    int32_t n_embd
) {
    // Write to a temp file then atomically rename, so a crash/disk-full during
    // the final write can never leave a truncated output.bin in place (the
    // checkpoint code below uses the same pattern). A truncated final output
    // would be silently misread by the Python parser on the next run.
    std::string temp_path = std::string(output_path) + ".tmp";
    FilePtr out(fopen(temp_path.c_str(), "wb"));
    if (!out) {
        fprintf(stderr, "Error: cannot open output file %s\n", temp_path.c_str());
        return false;
    }
    bool ok = _write_accumulator_to_file(accumulators, out, n_embd);
    if (ok) {
        if (!out.sync()) {  // flush + fsync — check for failures before rename
            fprintf(stderr, "Error: sync failed for %s (disk full or I/O error)\n", temp_path.c_str());
            out.reset();
            std::remove(temp_path.c_str());
            return false;
        }
        out.reset();  // close file before rename
        if (rename(temp_path.c_str(), output_path) != 0) {
            fprintf(stderr, "Error: cannot rename %s to %s\n", temp_path.c_str(), output_path);
            return false;
        }
        fprintf(stderr, "Output: written to %s\n", output_path);
    }
    return ok;
}

// -- Checkpoint / Resume ------------------------------------------------

static constexpr int32_t CHECKPOINT_VERSION = 2;
static constexpr int32_t CHECKPOINT_VERSION_V1 = 1;  // legacy: stored mean, restored via sum=mean*count

/**
 * Write checkpoint: version + n_iterated + accumulator state (binary accumulator format).
 * The checkpoint file is output_path + ".checkpoint".
 */
static bool write_checkpoint(
    const AccumulatorMap& accumulators,
    const char* output_path,
    int32_t n_embd,
    int32_t n_iterated
) {
    std::string ckpt_path = std::string(output_path) + ".checkpoint";
    std::string temp_path = ckpt_path + ".tmp";

    // Write to temporary file first
    FilePtr f(fopen(temp_path.c_str(), "wb"));
    if (!f) {
        fprintf(stderr, "Error: cannot write checkpoint to %s\n", temp_path.c_str());
        return false;
    }
    CHECKED_WRITE(&CHECKPOINT_VERSION, sizeof(int32_t), 1, f);
    CHECKED_WRITE(&n_iterated, sizeof(int32_t), 1, f);
    bool ok = _write_accumulator_to_file(accumulators, f, n_embd, /*write_sum=*/true);
    if (ok) {
        if (!f.sync()) {  // flush + fsync — check for failures before rename
            fprintf(stderr, "Error: sync failed for checkpoint %s\n", temp_path.c_str());
            f.reset();
            std::remove(temp_path.c_str());
            return false;
        }
        f.reset();  // close file before rename
        // Atomic rename: temp -> final
        if (rename(temp_path.c_str(), ckpt_path.c_str()) != 0) {
            fprintf(stderr, "Error: cannot rename %s to %s\n", temp_path.c_str(), ckpt_path.c_str());
            return false;
        }
        fprintf(stderr, "Checkpoint saved: %d prompts -> %s\n", n_iterated, ckpt_path.c_str());
    }
    return ok;
}

/**
 * Read checkpoint: restore accumulator state and return n_iterated (skip count).
 * Returns false if checkpoint doesn't exist or is corrupt.
 */
static bool read_checkpoint(
    const char* output_path,
    AccumulatorMap& accumulators,
    int32_t& n_iterated,
    int32_t expected_n_embd
) {
    std::string ckpt_path = std::string(output_path) + ".checkpoint";
    FilePtr f(fopen(ckpt_path.c_str(), "rb"));
    if (!f) return false;  // no checkpoint -- fresh start

    // Read and validate checkpoint version
    int32_t version = 0;
    if (fread(&version, sizeof(int32_t), 1, f) != 1) return false;
    if (version != CHECKPOINT_VERSION && version != CHECKPOINT_VERSION_V1) {
        fprintf(stderr, "Error: checkpoint version mismatch (got %d, expected %d) - incompatible checkpoint format\n",
                version, CHECKPOINT_VERSION);
        return false;
    }
    // v1 checkpoints stored mean (sum/count); v2 stores raw sum (no precision loss).
    // The read loop below branches on version to restore sums correctly.
    const bool is_v1 = (version == CHECKPOINT_VERSION_V1);

    // Read n_iterated
    if (fread(&n_iterated, sizeof(int32_t), 1, f) != 1) return false;

    // R4-R6: Validate n_iterated to prevent negative loop counts
    if (n_iterated < 0) {
        fprintf(stderr, "Error: checkpoint contains invalid n_iterated=%d (would cause infinite loop)\n", n_iterated);
        return false;
    }

    // Read accumulator state (binary accumulator format)
    int32_t magic = 0;
    if (fread(&magic, sizeof(int32_t), 1, f) != 1 || magic != OUTPUT_MAGIC) {
        return false;
    }
    int32_t n_groups = 0, n_layers = 0, n_embd = 0;
    if (fread(&n_groups, sizeof(int32_t), 1, f) != 1) return false;
    if (fread(&n_layers, sizeof(int32_t), 1, f) != 1) return false;
    if (fread(&n_embd, sizeof(int32_t), 1, f) != 1) return false;

    // Validate checkpoint dimensions to prevent OOM from corrupt data
    if (n_groups < 0 || n_groups > MAX_GROUPS) {
        fprintf(stderr, "Error: checkpoint n_groups=%d out of range [0, %d] - corrupt checkpoint\n", n_groups, MAX_GROUPS);
        return false;
    }
    if (n_layers < 0 || n_layers > MAX_LAYERS) {
        fprintf(stderr, "Error: checkpoint n_layers=%d out of range [0, %d] - corrupt checkpoint\n", n_layers, MAX_LAYERS);
        return false;
    }

    // Validate n_embd matches the model being used
    if (n_embd != expected_n_embd) {
        fprintf(stderr, "Error: checkpoint n_embd=%d does not match model n_embd=%d - checkpoint is from a different model\n",
                n_embd, expected_n_embd);
        return false;
    }

    for (int32_t g = 0; g < n_groups; g++) {
        int32_t group_id = 0, n_masks = 0;
        if (fread(&group_id, sizeof(int32_t), 1, f) != 1) return false;
        if (fread(&n_masks, sizeof(int32_t), 1, f) != 1) return false;

        // Validate group_id fits in 16-bit range (matches runtime validation at line ~1136)
        if (group_id < 0 || group_id > 0xFFFF) {
            fprintf(stderr, "Error: checkpoint contains group_id=%d out of range [0, 65535] - corrupt checkpoint\n", group_id);
            return false;
        }

        // Validate n_masks to prevent unbounded loop on corrupt checkpoint
        if (n_masks < 0 || n_masks > MAX_MASKS) {
            fprintf(stderr, "Error: checkpoint contains n_masks=%d out of range [0, %d] - corrupt checkpoint\n", n_masks, MAX_MASKS);
            return false;
        }

        for (int32_t m = 0; m < n_masks; m++) {
            int32_t mask_id = 0, n_layers_data = 0;
            if (fread(&mask_id, sizeof(int32_t), 1, f) != 1) return false;
            if (fread(&n_layers_data, sizeof(int32_t), 1, f) != 1) return false;

            if (mask_id < 0 || mask_id > 0xFFFF) {
                fprintf(stderr, "Error: checkpoint contains mask_id=%d out of range [0, 65535] - corrupt checkpoint\n", mask_id);
                return false;
            }

            // Validate n_layers_data to prevent unbounded loop on corrupt checkpoint
            if (n_layers_data < 0 || n_layers_data > MAX_LAYERS) {
                fprintf(stderr, "Error: checkpoint contains n_layers_data=%d out of range [0, %d] - corrupt checkpoint\n", n_layers_data, MAX_LAYERS);
                return false;
            }

            for (int32_t l = 0; l < n_layers_data; l++) {
                int32_t layer_idx = 0, count = 0;
                if (fread(&layer_idx, sizeof(int32_t), 1, f) != 1) return false;
                if (fread(&count, sizeof(int32_t), 1, f) != 1) return false;

                // Validate layer_idx and count to prevent silent corruption
                if (layer_idx < 0 || layer_idx > 0xFFFF) {
                    fprintf(stderr, "Error: checkpoint contains layer_idx=%d out of range [0, 65535] - corrupt checkpoint\n", layer_idx);
                    return false;
                }
                if (count < 0) {
                    fprintf(stderr, "Error: checkpoint contains count=%d (negative) - corrupt checkpoint\n", count);
                    return false;
                }
                // Upper-bound count: a single (group,mask,layer) accumulator can
                // receive at most one increment per (prompt, assignment) pair, so
                // its count cannot exceed n_iterated prompts * MAX_ASSIGNMENTS.
                // An unbounded count here is later used as a DIVISOR (write: 1/count)
                // and a v1 MULTIPLIER (sum *= count), so a corrupt huge value would
                // silently produce garbage means. Use int64 to avoid overflow in the
                // bound computation.
                if ((int64_t)count > (int64_t)n_iterated * MAX_ASSIGNMENTS) {
                    fprintf(stderr, "Error: checkpoint contains count=%d exceeding max %lld (n_iterated=%d * MAX_ASSIGNMENTS=%d) - corrupt checkpoint\n",
                            count, (long long)((int64_t)n_iterated * MAX_ASSIGNMENTS), n_iterated, MAX_ASSIGNMENTS);
                    return false;
                }

                // Use flat key for single-level map lookup
                uint64_t key = make_accum_key(group_id, mask_id, layer_idx);
                auto& av = accumulators[key];
                av.count = count;
                av.sum.resize(n_embd);
                if (is_v1) {
                    // v1 format: stored mean = sum/count. Read mean, then multiply
                    // by count to restore sum. Introduces ~1 ULP error per dimension.
                    if (fread(av.sum.data(), sizeof(float), n_embd, f) != (size_t)n_embd) {
                        return false;
                    }
                    if (count > 0) {
                        for (int d = 0; d < n_embd; d++) av.sum[d] *= (float)count;
                    }
                } else {
                    // v2+ format: stores raw sum directly (no precision loss).
                    if (fread(av.sum.data(), sizeof(float), n_embd, f) != (size_t)n_embd) {
                        return false;
                    }
                }
            }
        }
    }

    fprintf(stderr, "Checkpoint restored: %d prompts already processed\n", n_iterated);
    return true;
}

// -- Batch-Accumulate Mode ----------------------------------------------

/**
 * Batch-accumulate mode: the production extraction path.
 *
 * Reads prompts.txt (streamed one line at a time) and assignments.bin (sequential).
 * For each prompt: tokenize -> decode -> compute masked means per assignment -> accumulate.
 * At the end: write output.bin with per-group/mask/layer mean vectors.
 *
 * Memory is O(groups x masks x layers x n_embd), independent of prompt count.
 */
static int run_batch(const Args& args) {
    // -- Setup (RAII: destructors handle cleanup on all return paths) --
    LlamaBackend backend;
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;
    LlamaModel model(llama_model_load_from_file(args.model_path, mparams));
    if (!model) {
        fprintf(stderr, "Error: failed to load model %s\n", args.model_path);
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int32_t n_embd = llama_model_n_embd_out(model);
    const int32_t n_layers = llama_model_n_layer(model);
    const int32_t n_ctx_train = llama_model_n_ctx_train(model);

    fprintf(stderr, "Model loaded: n_ctx_train=%d, n_embd=%d, n_layers=%d, n_gpu_layers=%d\n",
            n_ctx_train, n_embd, n_layers, args.n_gpu_layers);

    // Resolve layers
    auto raw_layers = parse_layers(args.layers_str, n_layers);
    std::vector<int32_t> target_layers;
    for (int l : raw_layers) {
        if (l < 0) l = n_layers + l;
        if (l >= 0 && l < n_layers) {
            target_layers.push_back(l);
        } else {
            fprintf(stderr, "Error: layer %d out of range [0, %d)\n", l, n_layers);
            return 1;
        }
    }
    if (target_layers.empty()) {
        fprintf(stderr, "Error: no valid target layers\n");
        return 1;
    }

    // Auto-size context or use --ctx-size
    int32_t n_ctx;
    if (args.ctx_size > 0) {
        n_ctx = args.ctx_size;
    } else if (!auto_size_ctx(args.prompts_file, n_ctx)) {
        return 1;
    }
    if (n_ctx > n_ctx_train) n_ctx = n_ctx_train;
    fprintf(stderr, "Using n_ctx=%d\n", n_ctx);

    // Create context BEFORE opening output files or reading assignments.
    // Qwen3.5 Gated Delta Net requires the context to exist before model
    // graph optimization, and creating it first prevents an OOM when the
    // graph optimizer allocates CUDA buffers that would conflict with
    // subsequent allocations.
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;
    cparams.n_ubatch = n_ctx;  // Size ubatch to full context for stable CUDA graph capture
    cparams.extract_hidden_states = true;

    LlamaContext ctx(llama_init_from_model(model, cparams));
    if (!ctx) {
        fprintf(stderr, "Error: failed to create context\n");
        return 1;
    }

    // H1: --batch-size > 1 is accepted and range-checked by the parser but has
    // NO effect — processing is always sequential. A caller passing --batch-size 8
    // expecting 8× prompt packing gets 1× with only a stderr warning and exit 0.
    // Per the project's no-silent-errors rule: reject it loudly rather than lie.
    if (args.batch_size > 1) {
        fprintf(stderr, "Error: --batch-size %d is not supported. Multi-prompt batching is "
                        "not implemented (Gemma-4 shared-KV-cache corruption). Use --batch-size 1 "
                        "or omit the flag.\n", args.batch_size);
        return 1;
    }

    // -- Open input files --
    std::ifstream prompts_fin(args.prompts_file);
    if (!prompts_fin) {
        fprintf(stderr, "Error: cannot open prompts file %s\n", args.prompts_file);
        return 1;
    }

    FilePtr assign_fin(fopen(args.assignments_file, "rb"));
    if (!assign_fin) {
        fprintf(stderr, "Error: cannot open assignments file %s\n", args.assignments_file);
        return 1;
    }

    // Read assignments header
    int32_t n_prompts_expected = 0;
    int32_t n_embd_expected = 0;
    GroupTable groups;  // read but not used yet  -  reserved for future group-name output
    if (!read_assignments_header(assign_fin, n_prompts_expected, n_embd_expected, groups)) {
        return 1;
    }

    // Validate prompt count: assignments.bin and prompts.txt must agree.
    // Without this, a mismatch between files from different pipeline runs
    // causes silent desync (EOF or leftover assignments) with a generic
    // error message instead of a clear upfront diagnostic.
    {
        std::ifstream count_fin(args.prompts_file);
        if (!count_fin) {
            fprintf(stderr, "Error: cannot re-open prompts file for line count\n");
            return 1;
        }
        int32_t n_prompts_actual = 0;
        std::string cline;
        while (std::getline(count_fin, cline)) {
            if (!cline.empty()) n_prompts_actual++;
        }
        if (n_prompts_actual != n_prompts_expected) {
            fprintf(stderr, "Error: prompt count mismatch - assignments.bin expects %d prompts "
                            "but prompts.txt has %d non-empty lines\n",
                    n_prompts_expected, n_prompts_actual);
            return 1;
        }
    }

    // Validate n_embd if specified
    if (n_embd_expected > 0 && n_embd_expected != n_embd) {
        fprintf(stderr, "Error: assignments.bin expects n_embd=%d but model has n_embd=%d\n",
                n_embd_expected, n_embd);
        return 1;
    }

    // -- Hot loop --
    // Accumulator: flat key = make_accum_key(group_id, mask_id, layer_idx) -> AccumulatedVector
    AccumulatorMap accumulators;
    std::vector<float> mean_buf(n_embd, 0.0f);  // reused buffer for masked mean
    std::vector<std::pair<int,int>> ranges_buf;   // reused buffer for token ranges
    std::vector<float*> layer_ptrs(target_layers.size());  // Phase 2: reused buffer for layer pointers

    // Pre-allocate a reusable batch buffer sized to n_ctx.
    // Allocating/freeing a llama_batch on every prompt iteration caused
    // heap fragmentation over 90K+ iterations, eventually triggering glibc
    // "corrupted size vs. prev_size" aborts. By allocating once and reusing,
    // we eliminate all per-prompt heap churn from the batch structure.
    LlamaBatch batch_wrapper;
    batch_wrapper.init(n_ctx, 0, 1);
    llama_batch& batch = batch_wrapper.batch;

    // Resume from checkpoint if --resume is set
    int skip_count = 0;
    if (args.resume) {
        if (!read_checkpoint(args.output_path, accumulators, skip_count, n_embd)) {
            fprintf(stderr, "Error: --resume requested but no valid checkpoint found at %s.checkpoint\n",
                    args.output_path);
            return 1;
        }
    }

    int prompt_idx = 0;
    int n_processed = 0;
    auto start_time = std::chrono::steady_clock::now();

    // Profiling accumulators
    double prof_total_kv = 0, prof_total_decode = 0, prof_total_extract = 0, prof_total_mean = 0;
    int prof_count = 0;

    // Per-story sidecar file (optional)
    // Written to a temp path and atomically renamed to the final .stories.bin
    // only on full success, so a mid-run crash can never leave a truncated
    // sidecar that the Python parser would misread on the next run.
    std::string stories_path;
    std::string stories_temp_path;
    FilePtr stories_closer;
    if (args.save_per_story) {
        stories_path = std::string(args.output_path) + ".stories.bin";
        stories_temp_path = stories_path + ".tmp";
        stories_closer.fp = fopen(stories_temp_path.c_str(), "wb");
        if (!stories_closer) {
            fprintf(stderr, "Error: cannot open per-story file %s\n", stories_temp_path.c_str());
            return 1;
        }
        FILE* stories_fp = stories_closer.fp;
        int32_t story_magic = 0x53545231;  // per-story sidecar format
        if (fwrite(&story_magic, sizeof(int32_t), 1, stories_fp) != 1 ||
            fwrite(&n_embd, sizeof(int32_t), 1, stories_fp) != 1) {
            fprintf(stderr, "Error: per-story header write failed\n");
            // B11: remove the partially-written temp file.
            stories_closer.reset();
            std::remove(stories_temp_path.c_str());
            return 1;
        }
        int32_t n_target_layers = (int32_t)target_layers.size();
        if (fwrite(&n_target_layers, sizeof(int32_t), 1, stories_fp) != 1) {
            fprintf(stderr, "Error: per-story header write failed\n");
            // B11: remove the partially-written temp file.
            stories_closer.reset();
            std::remove(stories_temp_path.c_str());
            return 1;
        }
        fprintf(stderr, "Per-story output: %s (n_embd=%d, n_layers=%d)\n",
                stories_path.c_str(), n_embd, n_target_layers);
    }

    // -- Async prefetch pipeline (Tier 3.2) --
    // Producer-consumer: prefetch prompt N+1's tokens + assignments while GPU
    // processes prompt N. The producer thread does file I/O + tokenization (CPU
    // work that was previously serialized with GPU decode). The consumer (main
    // thread) owns the llama_context exclusively -- only it calls decode/extract.
    //
    // This overlaps ~1ms of I/O + ~0.5ms of tokenization with every decode,
    // recovering the 1-2% that was previously lost. Not a huge win on its own,
    // but it establishes the pipeline structure for the future async hidden
    // state extraction.

    struct PrefetchedPrompt {
        int prompt_idx = -1;
        std::vector<llama_token> tokens;
        std::vector<Assignment> assignments;
        bool skip = false;  // tokenization failed or exceeds ctx
        bool error = false; // set by producer on assignment read failure
    };

    struct PrefetchQueue {
        std::queue<PrefetchedPrompt> queue;
        std::mutex mtx;
        std::condition_variable cv;          // consumer waits: queue non-empty or producer done
        std::condition_variable cv_space;    // producer waits: queue has space
        std::atomic<bool> producer_done{false};
        std::atomic<int> n_produced{0};
    };

    // Backpressure: cap queue depth to bound memory usage.
    // Each item holds tokens + assignments (~2KB typical, up to ~50KB for long prompts).
    // 64 items -> max ~3MB queued, prevents OOM on 200K-prompt runs where
    // tokenization outpaces GPU decode.
    const size_t MAX_PREFETCH = 64;

    PrefetchQueue pfq;

    // Producer thread: reads prompts, tokenizes, reads assignments
    auto producer = [&]() {
      try {
        std::string p_line;
        while (std::getline(prompts_fin, p_line)) {
            // B2 fix: skip empty lines to match validation's non-empty count.
            // Without this, an empty line in prompts.txt reads an assignment
            // record meant for the next real prompt, silently desyncing all
            // subsequent prompt↔assignment pairings.
            if (p_line.empty()) continue;
            auto assignment_read = read_prompt_assignments(assign_fin);
            if (assignment_read.status != AssignmentReadStatus::ok) {
                PrefetchedPrompt p;
                p.error = true;
                {
                    std::lock_guard<std::mutex> lk(pfq.mtx);
                    pfq.queue.push(std::move(p));
                }
                pfq.cv.notify_one();
                return;
            }

            int p_idx = pfq.n_produced.fetch_add(1);

            PrefetchedPrompt pp;
            pp.prompt_idx = p_idx;
            pp.tokens = tokenize(vocab, p_line, /*add_bos=*/!args.no_bos);
            pp.assignments = std::move(assignment_read.assignments);
            pp.skip = pp.tokens.empty() || ((int)pp.tokens.size() > n_ctx);

            {
                std::unique_lock<std::mutex> lk(pfq.mtx);
                // Backpressure: wait if queue is full before pushing.
                pfq.cv_space.wait(lk, [&]{ return pfq.queue.size() < MAX_PREFETCH || pfq.producer_done.load(); });
                if (pfq.producer_done.load()) return;  // consumer signaled abort
                pfq.queue.push(std::move(pp));
            }
            pfq.cv.notify_one();
        }
        // B1 fix: detect I/O errors on the prompts stream. std::getline returns
        // false for both EOF and I/O errors — without this check, a disk failure
        // mid-file is indistinguishable from normal end-of-file, silently
        // truncating the run and exiting 0.
        if (prompts_fin.bad()) {
            fprintf(stderr, "Error: I/O error reading prompts file (bad stream)\n");
            PrefetchedPrompt ep;
            ep.error = true;
            {
                std::lock_guard<std::mutex> lk(pfq.mtx);
                pfq.producer_done = true;
                try { pfq.queue.push(std::move(ep)); } catch (...) {}
            }
            pfq.cv.notify_all();
            pfq.cv_space.notify_all();
            return;
        }
        {
            std::lock_guard<std::mutex> lk(pfq.mtx);
            pfq.producer_done = true;
        }
        pfq.cv.notify_all();
      } catch (const std::exception& e) {
        // An uncaught exception in the producer thread (e.g. std::bad_alloc from
        // tokenization, or a stream exception) would call std::terminate and abort
        // the whole process, leaving the consumer on the main thread hanging with
        // no diagnostic. Instead, surface a clean error to the consumer via an
        // error item so it exits gracefully (the consumer already handles pp.error).
        fprintf(stderr, "Error: producer thread exception: %s\n", e.what());
        // B5 fix: set producer_done FIRST (no allocation needed), so the consumer
        // is guaranteed to wake even if the queue.push throws bad_alloc again.
        {
            std::lock_guard<std::mutex> lk(pfq.mtx);
            pfq.producer_done = true;
            try {
                PrefetchedPrompt ep;
                ep.error = true;
                pfq.queue.push(std::move(ep));
            } catch (...) { /* consumer will see producer_done + empty queue */ }
        }
        pfq.cv.notify_all();
        pfq.cv_space.notify_all();
      } catch (...) {
        fprintf(stderr, "Error: producer thread unknown exception\n");
        PrefetchedPrompt ep;
        ep.error = true;
        {
            std::lock_guard<std::mutex> lk(pfq.mtx);
            pfq.queue.push(std::move(ep));
            pfq.producer_done = true;
        }
        pfq.cv.notify_all();
        pfq.cv_space.notify_all();
      }
    };

    std::thread producer_thread(producer);

    // Consumer (main thread): GPU decode + extract + accumulate
    while (true) {
        PrefetchedPrompt pp;
        {
            std::unique_lock<std::mutex> lk(pfq.mtx);
            pfq.cv.wait(lk, [&]{ return !pfq.queue.empty() || pfq.producer_done; });
            if (pfq.queue.empty() && pfq.producer_done) break;
            pp = std::move(pfq.queue.front());
            pfq.queue.pop();
            // Notify producer that queue has space (backpressure release).
            pfq.cv_space.notify_one();
        }

        if (pp.error) {
            fprintf(stderr, "Error: assignment read failed\n");
            STOP_PRODUCER_AND_JOIN();
            return 1;
        }

        // Skip already-processed prompts when resuming
        if (pp.prompt_idx < skip_count) {
            prompt_idx++;
            continue;
        }

        if (pp.skip) {
            if (pp.tokens.empty()) {
                fprintf(stderr, "Error: prompt %d tokenized to empty\n", pp.prompt_idx);
            } else {
                fprintf(stderr, "Error: prompt %d has %zu tokens, exceeds n_ctx=%d\n",
                        pp.prompt_idx, pp.tokens.size(), n_ctx);
            }
            STOP_PRODUCER_AND_JOIN();
            return 1;
        }

        const auto& assignments = pp.assignments;
        const auto& tokens = pp.tokens;

        // Profiling timers
        auto t_kv_start = std::chrono::steady_clock::now();

        // Clear KV cache
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) llama_memory_clear(mem, true);

        auto t_kv_end = std::chrono::steady_clock::now();

        // Fill pre-allocated batch buffer
        int n_tokens = (int)tokens.size();
        for (int i = 0; i < n_tokens; i++) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (args.generate_tokens > 0 && i == n_tokens - 1) ? 1 : 0;
            // In generation mode, enable logits on last prompt token for sampling
        }
        batch.n_tokens = n_tokens;

        // Decode (prefill)
        auto t_decode_start = std::chrono::steady_clock::now();
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "Error: decode failed for prompt %d (ret=%d)\n", prompt_idx, ret);
            STOP_PRODUCER_AND_JOIN();
            return 1;
        }

        llama_synchronize(ctx);
        auto t_decode_end = std::chrono::steady_clock::now();

        // -- Generation-based extraction (Anthropic/SLM methodology) --
        // If --generate N is set, autoregressively generate N tokens and extract
        // hidden states from the GENERATED tokens only. This matches the Anthropic
        // paper ("prompted Sonnet 4.5 to write stories... extracted activations")
        // and the SLM paper's finding that generation > comprehension (p=0.007).
        if (args.generate_tokens > 0) {
            // M1: --generate + --save-per-story is unsupported — the per-story
            // per-story sidecar records are written by the comprehension block below, which
            // this path skips via `continue`. Fail loud rather than emit a
            // header-only sidecar that silently produces zero .pt files.
            if (args.save_per_story) {
                fprintf(stderr, "Error: --generate is incompatible with --save-per-story "
                        "(per-story records are not written in generation mode)\n");
                STOP_PRODUCER_AND_JOIN();
                return 1;
            }
            // M2: generation mode means over ALL generated tokens. Assignments
            // with skip>0 or ranges (mask_type==1) have no defined semantics here
            // since there is no pre-written content span to mask. Fail loud.
            for (const auto& a : assignments) {
                if (a.skip > 0 || a.mask_type == 1) {
                    fprintf(stderr, "Error: --generate does not support assignment skip/ranges "
                            "(group=%d mask=%d has skip=%d mask_type=%d). Use skip=0, mask_type=0 "
                            "for generation-based extraction.\n",
                            a.group_id, a.mask_id, a.skip, a.mask_type);
                    STOP_PRODUCER_AND_JOIN();
                    return 1;
                }
            }
            // Accumulate hidden states from generated tokens into per-layer buffers.
            // We reuse mean_buf for per-token accumulation and a separate gen_accum
            // buffer for the running sum.
            // P2-3 fix: these were `static thread_local`, implying per-thread
            // safety. run_batch is single-consumer, so thread_local adds no
            // protection and only misleads a reader into thinking parallel
            // generation is safe. Plain function-scope static keeps the
            // perf benefit (one allocation, reused via resize/fill) without
            // the false concurrency implication.
            static std::vector<float> gen_accum;
            static std::vector<int64_t> gen_count;
            const size_t n_layers_total = target_layers.size();
            // Sanity-bound the accumulation buffer before allocating. n_layers_total
            // and n_embd come from the loaded model (target_layers is validated
            // against the model's layer count; n_embd is the model's embedding dim),
            // so the product is normally small (e.g. 42 layers x 2560 embd = 107K
            // floats). Guard against a corrupt/absurd configuration that would request
            // a multi-GB allocation: resize() throws std::bad_alloc on failure and
            // this block has no surrounding try, so an unbounded resize would call
            // std::terminate. 256M floats (1 GB) is far above any real model's
            // layer*embd product.
            constexpr size_t MAX_GEN_ACCUM_FLOATS = 256ull * 1024 * 1024;  // 1 GB
            if (n_layers_total > 0 && (size_t)n_embd > MAX_GEN_ACCUM_FLOATS / n_layers_total) {
                fprintf(stderr, "Error: generation accumulator size (%zu layers x %d embd) "
                        "exceeds 1GB cap - configuration is corrupt or absurdly large\n",
                        n_layers_total, n_embd);
                STOP_PRODUCER_AND_JOIN();
                return 1;
            }
            gen_accum.resize(n_layers_total * n_embd, 0.0f);
            gen_count.assign(n_layers_total, 0);
            std::fill(gen_accum.begin(), gen_accum.end(), 0.0f);

            int cur_pos = n_tokens;  // next position after prompt
            llama_token next_token = LLAMA_TOKEN_NULL;

            // Build sampler chain for generation. Default (temperature=0) is greedy,
            // which is deterministic but divergent across CUDA/Vulkan backends because
            // small logit differences flip the argmax. temperature > 0 adds stochasticity
            // that, combined with repeat_penalty, stabilizes cross-chip trajectories.
            const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
            int n_vocab = llama_vocab_n_tokens(vocab);
            llama_sampler * sampler = nullptr;
            if (args.temperature > 0.0f) {
                llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
                sampler = llama_sampler_chain_init(sparams);
                if (args.repeat_penalty > 1.0f) {
                    // upstream API change (2026-08): llama_sampler_init_penalties now
                    // takes n_vocab first to bound the repeat-scan range.
                    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(n_vocab, 64, args.repeat_penalty, 0.0f, 0.0f));
                }
                if (args.top_k > 0) {
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(args.top_k));
                }
                if (args.top_p < 1.0f) {
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(args.top_p, 1));
                }
                llama_sampler_chain_add(sampler, llama_sampler_init_temp(args.temperature));
                llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));  // seed=0 for determinism with temperature
            }

            for (int gen_step = 0; gen_step < args.generate_tokens; gen_step++) {
                // KV-cache bounds check: cur_pos must stay < n_ctx or llama_decode
                // writes/reads OOB KV state — the exact fault class that hard-locks
                // the GPU. Without this, a long prompt + large --generate (or a model
                // that never emits EOS) overruns the context window.
                if (cur_pos >= n_ctx) {
                    fprintf(stderr, "Warning: generation reached n_ctx=%d at step %d for prompt %d — stopping\n",
                            n_ctx, gen_step, prompt_idx);
                    break;
                }
                // Sample first token from prefill's last position
                if (gen_step == 0) {
                    if (sampler) {
                        next_token = llama_sampler_sample(sampler, ctx, n_tokens - 1);
                    } else {
                        // Greedy fallback (temperature=0): argmax
                        float* token_logits = llama_get_logits_ith(ctx, n_tokens - 1);
                        if (!token_logits) {
                            fprintf(stderr, "Error: no logits available for generation sampling\n");
                            if (sampler) { llama_sampler_free(sampler); sampler = nullptr; }
                            STOP_PRODUCER_AND_JOIN();
                            return 1;
                        }
                        next_token = 0;
                        float best_val = token_logits[0];
                        for (int v = 1; v < n_vocab; v++) {
                            if (token_logits[v] > best_val) {
                                best_val = token_logits[v];
                                next_token = v;
                            }
                        }
                        // B8: if all logits are NaN, every comparison is false,
                        // so next_token stays 0 and NaN propagates into the
                        // hidden-state accumulation silently. Detect and abort.
                        if (std::isnan(best_val)) {
                            fprintf(stderr, "Error: NaN logits at generation step %d for prompt %d "
                                            "(numerical instability or corrupt model)\n", gen_step, prompt_idx);
                            if (sampler) { llama_sampler_free(sampler); sampler = nullptr; }
                            STOP_PRODUCER_AND_JOIN();
                            return 1;
                        }
                    }
                }

                // Check for EOS
                if (llama_vocab_is_eog(llama_model_get_vocab(llama_get_model(ctx)), next_token)) {
                    break;
                }

                // Decode single generated token with logits enabled for next step
                batch.n_tokens = 1;
                batch.token[0] = next_token;
                batch.pos[0] = cur_pos;
                batch.n_seq_id[0] = 1;
                batch.seq_id[0][0] = 0;
                batch.logits[0] = (gen_step < args.generate_tokens - 1) ? 1 : 0;

                ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    fprintf(stderr, "Error: generation decode failed at step %d for prompt %d (ret=%d)\n",
                            gen_step, prompt_idx, ret);
                    if (sampler) { llama_sampler_free(sampler); sampler = nullptr; }
                    STOP_PRODUCER_AND_JOIN();
                    return 1;
                }
                llama_synchronize(ctx);
                cur_pos++;

                // Extract hidden states from this generated token
                if (llama_get_hidden_states_batch(ctx, target_layers.data(), target_layers.size(), layer_ptrs.data()) != 0) {
                    fprintf(stderr, "Error: hidden state extraction failed during generation step %d\n", gen_step);
                    if (sampler) { llama_sampler_free(sampler); sampler = nullptr; }
                    STOP_PRODUCER_AND_JOIN();
                    return 1;
                }

                // Accumulate: each layer has 1 token × n_embd floats.
                // Skip the first args.token_skip generated tokens (matching
                // comprehension-mode behavior): the initial generated tokens
                // are "warm-up" content that dilutes the concept signal and
                // amplifies cross-chip divergence (the first ~50 tokens of
                // greedy decoding diverge most across CUDA/Vulkan backends).
                if (gen_step >= args.token_skip) {
                    for (size_t li = 0; li < n_layers_total; li++) {
                        float* data = layer_ptrs[li];
                        if (data) {
                            float* accum_ptr = &gen_accum[li * n_embd];
                            CR_SIMD
                            for (int d = 0; d < n_embd; d++) {
                                accum_ptr[d] += data[d];
                            }
                            gen_count[li]++;
                        }
                    }
                }

                // Sample next token from this step's logits (for next iteration)
                if (gen_step < args.generate_tokens - 1) {
                    if (sampler) {
                        next_token = llama_sampler_sample(sampler, ctx, 0);
                    } else {
                        // Greedy fallback
                        float* gen_logits = llama_get_logits_ith(ctx, 0);
                        if (gen_logits) {
                            next_token = 0;
                            float best_val = gen_logits[0];
                            for (int v = 1; v < n_vocab; v++) {
                                if (gen_logits[v] > best_val) {
                                    best_val = gen_logits[v];
                                    next_token = v;
                                }
                            }
                            // B8: NaN guard (same as the token_logits argmax above).
                            // If every logit is NaN, no comparison is true and
                            // next_token stays 0, propagating NaN into accumulation.
                            if (std::isnan(best_val)) {
                                fprintf(stderr, "Error: NaN logits at generation step %d for prompt %d "
                                                "(numerical instability or corrupt model)\n", gen_step, prompt_idx);
                                STOP_PRODUCER_AND_JOIN();
                                return 1;
                            }
                        }
                    }
                }
            }

            // Cleanup sampler
            if (sampler) {
                llama_sampler_free(sampler);
                sampler = nullptr;
            }

            // Compute means from generated tokens and accumulate into the assignment buffers
            // (same path as comprehension mode, but using gen_accum instead of masked mean)
            // Use per-layer gen_count[li] — a null layer_ptrs[li] skips that layer's
            // accumulation, so counts can differ across layers.
            for (const auto& assign : assignments) {
                for (size_t li = 0; li < target_layers.size(); li++) {
                    int64_t n_gen = gen_count[li];
                    if (n_gen == 0) continue;  // layer had no valid hidden states
                    float* accum_ptr = &gen_accum[li * n_embd];
                    // Divide by count to get mean
                    float inv = 1.0f / (float)n_gen;
                    CR_SIMD
                    for (int d = 0; d < n_embd; d++) {
                        mean_buf[d] = accum_ptr[d] * inv;
                    }
                    // Accumulate into the group/mask/layer accumulator.
                    // Use target_layers[li] (the REAL layer number), matching the
                    // comprehension path at line 1812. Using the loop index `li`
                    // would silently mislabel layers in the binary accumulator output.
                    uint64_t acc_key = make_accum_key(assign.group_id, assign.mask_id, target_layers[li]);
                    auto& acc = accumulators[acc_key];
                    if (acc.sum.empty()) {
                        acc.sum.assign(mean_buf.data(), mean_buf.data() + n_embd);
                        acc.count = 1;
                    } else {
                        CR_SIMD
                        for (int d = 0; d < n_embd; d++) {
                            acc.sum[d] += mean_buf[d];
                        }
                        acc.count++;
                    }
                }
            }

            // Skip the normal comprehension-mode extraction below.
            // Replicate the comprehension path's bookkeeping so progress,
            // checkpointing, and the final summary count work identically.
            prompt_idx++;
            n_processed++;

            if (args.checkpoint_every > 0 && n_processed % args.checkpoint_every == 0) {
                if (!write_checkpoint(accumulators, args.output_path, n_embd, prompt_idx)) {
                    fprintf(stderr, "Error: checkpoint write failed  -  aborting to prevent data loss\n");
                    STOP_PRODUCER_AND_JOIN();
                    return 1;
                }
            }

            if (n_processed % PROGRESS_INTERVAL == 0) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                float pps = n_processed * 1000.0f / (elapsed > 0 ? elapsed : 1);
                if (skip_count > 0) {
                    fprintf(stderr, "Position %d/%d (%d new since resume, %.2f prompts/sec)\n",
                            prompt_idx, n_prompts_expected, n_processed, pps);
                } else {
                    fprintf(stderr, "Processed %d/%d prompts (%.2f prompts/sec)\n",
                            prompt_idx, n_prompts_expected, pps);
                }
            }
            continue;
        }

        // Phase 2: batch-fetch all layer pointers
        auto t_extract_start = std::chrono::steady_clock::now();
        if (llama_get_hidden_states_batch(ctx, target_layers.data(), target_layers.size(), layer_ptrs.data()) != 0) {
            fprintf(stderr, "Error: llama_get_hidden_states_batch failed\n");
            STOP_PRODUCER_AND_JOIN();
            return 1;
        }

        int n_hidden = llama_get_hidden_state_n_tokens(ctx);
        if (n_hidden <= 0) {
            fprintf(stderr, "Error: decode produced no hidden state tokens for prompt %d\n", pp.prompt_idx);
            STOP_PRODUCER_AND_JOIN();
            return 1;
        }
        auto t_extract_end = std::chrono::steady_clock::now();

        // For each assignment: compute masked mean per layer and accumulate
        auto t_mean_start = std::chrono::steady_clock::now();
        for (const auto& assign : assignments) {
            if (assign.group_id < 0 || assign.group_id > 0xFFFF) {
                fprintf(stderr, "Error: group_id %d exceeds 16-bit range\n", assign.group_id);
                STOP_PRODUCER_AND_JOIN();
                return 1;
            }
            if (assign.mask_id < 0 || assign.mask_id > 0xFFFF) {
                fprintf(stderr, "Error: mask_id %d exceeds 16-bit range\n", assign.mask_id);
                STOP_PRODUCER_AND_JOIN();
                return 1;
            }

            if (assign.mask_type == 0) {
                ranges_buf.clear();
                ranges_buf.push_back({assign.skip, n_hidden});
            } else {
                ranges_buf = assign.ranges;
            }

            for (size_t li = 0; li < target_layers.size(); li++) {
                float* data = layer_ptrs[li];
                if (!data) {
                    fprintf(stderr, "Error: null hidden state for layer %d\n", target_layers[li]);
                    STOP_PRODUCER_AND_JOIN();
                    return 1;
                }

                std::fill(mean_buf.begin(), mean_buf.end(), 0.0f);
                int64_t tokens_in_mask = compute_masked_mean(data, n_hidden, n_embd, ranges_buf, mean_buf.data());
                if (tokens_in_mask < 0) {
                    fprintf(stderr, "Error: compute_masked_mean failed for prompt %d\n", pp.prompt_idx);
                    STOP_PRODUCER_AND_JOIN();
                    return 1;
                }
                if (tokens_in_mask == 0) continue;

                uint64_t flat_key = make_accum_key(assign.group_id, assign.mask_id, target_layers[li]);
                auto& av = accumulators[flat_key];
                if (av.sum.empty()) av.sum.resize(n_embd, 0.0f);
                CR_SIMD
                for (int d = 0; d < n_embd; d++) av.sum[d] += mean_buf[d];
                av.count++;

                if (args.save_per_story && stories_closer) {
                    FILE* stories_fp = stories_closer.fp;
                    STORIES_WRITE(&prompt_idx, sizeof(int32_t), 1, stories_fp, pfq, producer_thread);
                    int32_t gid = assign.group_id;
                    int32_t mid = assign.mask_id;
                    int32_t layer_idx = target_layers[li];
                    STORIES_WRITE(&gid, sizeof(int32_t), 1, stories_fp, pfq, producer_thread);
                    STORIES_WRITE(&mid, sizeof(int32_t), 1, stories_fp, pfq, producer_thread);
                    STORIES_WRITE(&layer_idx, sizeof(int32_t), 1, stories_fp, pfq, producer_thread);
                    STORIES_WRITE(mean_buf.data(), sizeof(float), n_embd, stories_fp, pfq, producer_thread);
                }
            }
        }
        auto t_mean_end = std::chrono::steady_clock::now();

        // Accumulate profiling data
        if (args.profile) {
            prof_total_kv     += std::chrono::duration<double, std::milli>(t_kv_end - t_kv_start).count();
            prof_total_decode += std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
            prof_total_extract+= std::chrono::duration<double, std::milli>(t_extract_end - t_extract_start).count();
            prof_total_mean   += std::chrono::duration<double, std::milli>(t_mean_end - t_mean_start).count();
            prof_count++;
        }

        prompt_idx++;
        n_processed++;

        if (args.checkpoint_every > 0 && n_processed % args.checkpoint_every == 0) {
            if (!write_checkpoint(accumulators, args.output_path, n_embd, prompt_idx)) {
                fprintf(stderr, "Error: checkpoint write failed  -  aborting to prevent data loss\n");
                STOP_PRODUCER_AND_JOIN();
                return 1;
            }
        }

        if (n_processed % PROGRESS_INTERVAL == 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            float pps = n_processed * 1000.0f / (elapsed > 0 ? elapsed : 1);
            if (skip_count > 0) {
                fprintf(stderr, "Position %d/%d (%d new since resume, %.2f prompts/sec)\n",
                        prompt_idx, n_prompts_expected, n_processed, pps);
            } else {
                fprintf(stderr, "Processed %d/%d prompts (%.2f prompts/sec)\n",
                        prompt_idx, n_prompts_expected, pps);
            }
        }
    }

    producer_thread.join();

    // -- Finalize --
    auto end_time = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Print profiling breakdown
    if (args.profile && prof_count > 0) {
        double wall_total = total_ms;
        double avg_kv      = prof_total_kv / prof_count;
        double avg_decode  = prof_total_decode / prof_count;
        double avg_extract = prof_total_extract / prof_count;
        double avg_mean    = prof_total_mean / prof_count;
        double avg_wall    = wall_total / prof_count;
        double measured    = avg_kv + avg_decode + avg_extract + avg_mean;
        double other       = avg_wall - measured;

        fprintf(stderr, "\n=== PROFILING BREAKDOWN (%d prompts) ===\n", prof_count);
        fprintf(stderr, "Phase                   Avg ms/prompt    %% of wall\n");
        fprintf(stderr, "-----------------------------------------------------\n");
        fprintf(stderr, "KV cache clear:         %8.2f ms    %5.1f%%\n", avg_kv,      avg_kv / avg_wall * 100);
        fprintf(stderr, "Decode + sync:          %8.2f ms    %5.1f%%\n", avg_decode,  avg_decode / avg_wall * 100);
        fprintf(stderr, "HS batch fetch:         %8.2f ms    %5.1f%%\n", avg_extract, avg_extract / avg_wall * 100);
        fprintf(stderr, "Mean + accumulate:      %8.2f ms    %5.1f%%\n", avg_mean,    avg_mean / avg_wall * 100);
        fprintf(stderr, "-----------------------------------------------------\n");
        fprintf(stderr, "Measured total:         %8.2f ms    %5.1f%%\n", measured, measured / avg_wall * 100);
        fprintf(stderr, "Other (overhead):       %8.2f ms    %5.1f%%\n", other,    other / avg_wall * 100);
        fprintf(stderr, "Wall per prompt:        %8.2f ms\n", avg_wall);
        fprintf(stderr, "===============================================\n\n");
    }

    // B10: TOCTOU guard — verify the actual processed count matches the expected
    // count from the assignments header. A mismatch means the prompts file was
    // modified between the count pass and the producer read.
    if (skip_count == 0 && n_processed != n_prompts_expected) {
        fprintf(stderr, "Error: processed %d prompts but assignments expected %d "
                        "(prompts file may have been modified during extraction)\n",
                n_processed, n_prompts_expected);
        return 1;
    }

    if (skip_count > 0) {
        fprintf(stderr, "Done. Position %d/%d (%d new since resume) in %.2f seconds (%.2f prompts/sec)\n",
                prompt_idx, n_prompts_expected, n_processed,
                total_ms / 1000.0,
                n_processed * 1000.0 / (total_ms > 0 ? total_ms : 1));
    } else {
        fprintf(stderr, "Done. Processed %d prompts in %.2f seconds (%.2f prompts/sec)\n",
                n_processed, total_ms / 1000.0,
                n_processed * 1000.0 / (total_ms > 0 ? total_ms : 1));
    }

    // Guard: if we processed prompts but produced no accumulators (e.g. every
    // prompt hit EOS before gen_step reached token_skip, or all masked means
    // were empty), refuse to write a header-only output.bin and exit non-zero.
    // A silent empty file would be misread downstream as a valid-but-empty
    // result rather than flagged as a failed extraction.
    if (n_processed > 0 && accumulators.empty()) {
        fprintf(stderr, "Error: processed %d prompts but accumulated no hidden-state "
                        "means (check token_skip / EOS / masks); refusing to write an "
                        "empty output.bin\n", n_processed);
        // B11: clean up the per-story temp file on this error path.
        if (!stories_temp_path.empty()) std::remove(stories_temp_path.c_str());
        return 1;
    }

    // Write output
    if (!write_batch_output(accumulators, args.output_path, n_embd)) {
        // B11: write_batch_output failed (its own temp was already cleaned inside
        // the function); remove the per-story temp file here.
        if (!stories_temp_path.empty()) std::remove(stories_temp_path.c_str());
        return 1;
    }

    // Per-story sidecar: close the temp file and atomically rename to the final
    // path now that the run fully succeeded.
    if (stories_closer) {
        if (!stories_closer.sync()) {  // flush + fsync — check for failures
            fprintf(stderr, "Error: sync failed for %s\n", stories_temp_path.c_str());
            stories_closer.reset();
            std::remove(stories_temp_path.c_str());
            return 1;
        }
        stories_closer.reset();  // close before rename
        if (rename(stories_temp_path.c_str(), stories_path.c_str()) != 0) {
            fprintf(stderr, "Error: cannot rename %s to %s\n",
                    stories_temp_path.c_str(), stories_path.c_str());
            // B11: rename failed — remove the orphaned temp file.
            std::remove(stories_temp_path.c_str());
            return 1;
        }
        fprintf(stderr, "Per-story output complete: %s\n", stories_path.c_str());
    }

    // Delete checkpoint on successful completion
    std::string ckpt_path = std::string(args.output_path) + ".checkpoint";
    // B7: check remove() result — a stale checkpoint surviving a successful run
    // would cause a redundant --resume. Warn (not abort) since data is fine.
    if (remove(ckpt_path.c_str()) != 0 && errno != ENOENT) {
        fprintf(stderr, "Warning: could not remove checkpoint %s (run completed successfully; "
                        "delete it manually to avoid a redundant --resume)\n", ckpt_path.c_str());
    }

    return 0;
}

// -- Self-Test Mode -----------------------------------------------------

/**
 * Run synthetic known-value tests on compute_masked_mean() with no model loaded.
 * Exits 0 on pass, 1 on fail.
 */
static int run_self_test() {
    fprintf(stderr, "Running compute_masked_mean self-tests...\n\n");

    int passed = 0;
    const int total = 17;
    bool all_ok = true;

    // All tests use data layout: 3 tokens x 2 dims, row-major
    // data[0..5] = {1, 2, 3, 4, 5, 6}
    //   token0 = [1, 2], token1 = [3, 4], token2 = [5, 6]
    float data1[6] = {1, 2, 3, 4, 5, 6};

    // Test 1: single contiguous range = mean over all tokens
    {
        std::vector<std::pair<int,int>> ranges = {{0, 3}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        // dim0: (1+3+5)/3 = 3.0   dim1: (2+4+6)/3 = 4.0
        bool ok = (count == 3)
               && (std::abs(out[0] - 3.0f) < 1e-6f)
               && (std::abs(out[1] - 4.0f) < 1e-6f);
        fprintf(stderr, "  Test 1 (single contiguous range): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 2: skip first token
    {
        std::vector<std::pair<int,int>> ranges = {{1, 3}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        // dim0: (3+5)/2 = 4.0   dim1: (4+6)/2 = 5.0
        bool ok = (count == 2)
               && (std::abs(out[0] - 4.0f) < 1e-6f)
               && (std::abs(out[1] - 5.0f) < 1e-6f);
        fprintf(stderr, "  Test 2 (skip first token): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 3: non-contiguous ranges (deflection pattern: tokens 0 and 2, skip 1)
    {
        std::vector<std::pair<int,int>> ranges = {{0, 1}, {2, 3}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        // dim0: (1+5)/2 = 3.0   dim1: (2+6)/2 = 4.0
        bool ok = (count == 2)
               && (std::abs(out[0] - 3.0f) < 1e-6f)
               && (std::abs(out[1] - 4.0f) < 1e-6f);
        fprintf(stderr, "  Test 3 (non-contiguous ranges): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 4: empty mask (zero ranges)
    {
        std::vector<std::pair<int,int>> ranges = {};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        // count should be 0, out stays zeroed
        bool ok = (count == 0)
               && (std::abs(out[0]) < 1e-6f)
               && (std::abs(out[1]) < 1e-6f);
        fprintf(stderr, "  Test 4 (empty mask): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 5: hard error  -  range end exceeds n_tokens (no soft clamp)
    {
        std::vector<std::pair<int,int>> ranges = {{0, 100}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        bool ok = (count == -1);  // hard error  -  no clamping
        fprintf(stderr, "  Test 5 (hard error end > n_tokens): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 6: hard error  -  both start and end overshoot
    {
        std::vector<std::pair<int,int>> ranges = {{3, 5}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        bool ok = (count == -1);  // hard error  -  no clamping
        fprintf(stderr, "  Test 6 (hard error overshoot range): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 7: hard error  -  fully out-of-bounds
    {
        std::vector<std::pair<int,int>> ranges = {{50, 100}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        bool ok = (count == -1);  // hard error  -  no clamping
        fprintf(stderr, "  Test 7 (hard error fully out-of-bounds): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 8: hard error  -  negative range
    {
        std::vector<std::pair<int,int>> ranges = {{-1, 3}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        bool ok = (count == -1);  // hard error
        fprintf(stderr, "  Test 8 (hard error negative range): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 9: hard error  -  inverted range (start > end)
    {
        std::vector<std::pair<int,int>> ranges = {{2, 1}};
        float out[2] = {0, 0};
        int count = compute_masked_mean(data1, 3, 2, ranges, out);
        bool ok = (count == -1);  // hard error
        fprintf(stderr, "  Test 9 (hard error inverted range): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 10: compute_single_range_mean  -  basic mean
    {
        float out[2] = {0, 0};
        int count = compute_single_range_mean(data1, 3, 2, 0, 3, out);
        // dim0: (1+3+5)/3 = 3.0   dim1: (2+4+6)/3 = 4.0
        bool ok = (count == 3)
               && (std::abs(out[0] - 3.0f) < 1e-6f)
               && (std::abs(out[1] - 4.0f) < 1e-6f);
        fprintf(stderr, "  Test 10 (single-range basic mean): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 11: compute_single_range_mean  -  skip first token
    {
        float out[2] = {0, 0};
        int count = compute_single_range_mean(data1, 3, 2, 1, 3, out);
        // dim0: (3+5)/2 = 4.0   dim1: (4+6)/2 = 5.0
        bool ok = (count == 2)
               && (std::abs(out[0] - 4.0f) < 1e-6f)
               && (std::abs(out[1] - 5.0f) < 1e-6f);
        fprintf(stderr, "  Test 11 (single-range skip first): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 12: compute_single_range_mean  -  single token
    {
        float out[2] = {0, 0};
        int count = compute_single_range_mean(data1, 3, 2, 1, 2, out);
        // token1 = [3, 4]
        bool ok = (count == 1)
               && (std::abs(out[0] - 3.0f) < 1e-6f)
               && (std::abs(out[1] - 4.0f) < 1e-6f);
        fprintf(stderr, "  Test 12 (single-range single token): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 13: compute_single_range_mean  -  hard error (end > n_tokens)
    {
        float out[2] = {0, 0};
        int count = compute_single_range_mean(data1, 3, 2, 0, 100, out);
        bool ok = (count == -1);
        fprintf(stderr, "  Test 13 (single-range hard error end > n_tokens): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 14: compute_single_range_mean  -  hard error (negative)
    {
        float out[2] = {0, 0};
        int count = compute_single_range_mean(data1, 3, 2, -1, 3, out);
        bool ok = (count == -1);
        fprintf(stderr, "  Test 14 (single-range hard error negative): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 15: overlapping ranges  -  verify correct token deduplication
    // ranges = {{0, 2}, {1, 3}} means tokens 0,1 and 1,2 -> token 1 counted twice
    // dim0: (1+3) + (3+5) = 4 + 8 = 12, count = 4, mean = 3.0
    // dim1: (2+4) + (4+6) = 6 + 10 = 16, count = 4, mean = 4.0
    {
        std::vector<std::pair<int,int>> ranges = {{0, 2}, {1, 3}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        bool ok = (count == 4)  // 2 + 2 tokens (token 1 counted twice)
               && (std::abs(out[0] - 3.0f) < 1e-6f)  // (1+3+3+5)/4 = 12/4 = 3.0
               && (std::abs(out[1] - 4.0f) < 1e-6f); // (2+4+4+6)/4 = 16/4 = 4.0
        fprintf(stderr, "  Test 15 (overlapping ranges): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 16: key encode/decode roundtrip
    // Verifies that make_accum_key and decode_accum_key are exact inverses
    // for all valid (group_id, mask_id, layer_idx) combinations.
    {
        bool ok = true;
        const int32_t test_cases[][3] = {
            {0, 0, 0},       // minimum
            {0xFFFF, 0xFFFF, 0xFFFF},  // maximum (16-bit each)
            {42, 17, 5},     // typical
            {1, 2, 3},       // small
            {1000, 500, 99}, // medium
        };
        for (const auto& tc : test_cases) {
            uint64_t key = make_accum_key(tc[0], tc[1], tc[2]);
            int32_t g, m, l;
            decode_accum_key(key, g, m, l);
            if (g != tc[0] || m != tc[1] || l != tc[2]) {
                fprintf(stderr, "  key roundtrip failed: (%d,%d,%d) -> key=%llu -> (%d,%d,%d)\n",
                        tc[0], tc[1], tc[2], (unsigned long long)key, g, m, l);
                ok = false;
            }
        }
        // Verify key uniqueness: different inputs must produce different keys
        uint64_t k1 = make_accum_key(1, 0, 0);
        uint64_t k2 = make_accum_key(0, 1, 0);
        uint64_t k3 = make_accum_key(0, 0, 1);
        if (k1 == k2 || k1 == k3 || k2 == k3) ok = false;

        fprintf(stderr, "  Test 16 (key encode/decode roundtrip): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 17: checkpoint write/read roundtrip
    // Verifies that checkpoint format v2 (sum-based) survives a write+read
    // cycle with zero precision loss.
    {
        AccumulatorMap test_acc;
        // Create a few test entries with known sum values
        uint64_t key1 = make_accum_key(0, 0, 0);
        test_acc[key1].sum = {1.0f, 2.0f, 3.0f, 4.0f};
        test_acc[key1].count = 10;
        uint64_t key2 = make_accum_key(1, 2, 5);
        test_acc[key2].sum = {0.1f, -0.2f, 0.3f, -0.4f};
        test_acc[key2].count = 7;

        // Write checkpoint to temp file — include PID to avoid collision
        // between concurrent self-test runs (CI) and symlink attacks.
        char test_ckpt_buf[256];
        snprintf(test_ckpt_buf, sizeof(test_ckpt_buf), "/tmp/cr_self_test_ckpt_%d.bin", (int)getpid());
        const char* test_ckpt = test_ckpt_buf;
        bool write_ok = write_checkpoint(test_acc, test_ckpt, 4, 42);
        if (!write_ok) {
            fprintf(stderr, "  Test 17 (checkpoint roundtrip): FAIL (write failed)\n");
            all_ok = false;
        } else {
            // Read it back
            AccumulatorMap restored_acc;
            int32_t n_iterated = 0;
            bool read_ok = read_checkpoint(test_ckpt, restored_acc, n_iterated, 4);

            bool ok = read_ok && (n_iterated == 42);
            if (ok) {
                // Verify sums match exactly (v2 format stores sum directly)
                auto& av1 = restored_acc[key1];
                auto& av2 = restored_acc[key2];
                ok = (av1.count == 10) && (av2.count == 7);
                for (int d = 0; d < 4 && ok; d++) {
                    if (std::abs(av1.sum[d] - test_acc[key1].sum[d]) > 1e-6f) ok = false;
                    if (std::abs(av2.sum[d] - test_acc[key2].sum[d]) > 1e-6f) ok = false;
                }
            }
            fprintf(stderr, "  Test 17 (checkpoint roundtrip): %s\n", ok ? "PASS" : "FAIL");
            if (ok) passed++; else all_ok = false;
        }
        // Cleanup
        remove(test_ckpt);
        remove((std::string(test_ckpt) + ".tmp").c_str());
    }

    fprintf(stderr, "\n%d/%d tests passed\n", passed, total);

    if (all_ok) {
        fprintf(stderr, "All self-tests passed\n");
        return 0;
    } else {
        fprintf(stderr, "SELF-TEST FAILED\n");
        return 1;
    }
}

// -- Main ----------------------------------------------------------------

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.self_test) {
        return run_self_test();
    }

    if (args.batch_mode) {
        return run_batch(args);
    }

    if (args.raw_mode) {
        return run_raw(args);
    }

    // Should not reach here -- parse_args exits with usage if no mode set
    fprintf(stderr, "Error: must specify --batch, --raw, or --self-test\n");
    return 1;
}
