#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "common.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#endif

enum split_operation : uint8_t {
    OP_NONE,
    OP_SPLIT,
    OP_MERGE,
};

enum split_mode : uint8_t {
    MODE_NONE,
    MODE_TENSOR,
    MODE_SIZE,
};

struct split_params {
    split_operation operation = OP_NONE;
    split_mode mode = MODE_NONE;
    size_t n_bytes_split = 0;
    int n_split_tensors = 128;
    std::string input;
    std::string output;
    bool no_tensor_first_split = false;
    bool dry_run = false;
};

static void split_print_usage(const char * executable) {
    const split_params default_params;
    printf("\n");
    printf("usage: %s [options] GGUF_IN GGUF_OUT\n", executable);
    printf("\n");
    printf("Apply a GGUF operation on IN to OUT.");
    printf("\n");
    printf("options:\n");
    printf("  -h, --help              show this help message and exit\n");
    printf("  --version               show version and build info\n");
    printf("  --split                 split GGUF to multiple GGUF (enabled by default)\n");
    printf("  --merge                 merge multiple GGUF to a single GGUF\n");
    printf("  --split-max-tensors     max tensors in each split (default: %d)\n", default_params.n_split_tensors);
    printf("  --split-max-size N(M|G) max size per split\n");
    printf("  --no-tensor-first-split do not add tensors to the first split (disabled by default)\n");
    printf("  --dry-run               only print out a split plan and exit, without writing any new files\n");
    printf("\n");
}

// return convert string, for example "128M" or "4G" to number of bytes
static size_t split_str_to_n_bytes(std::string str) {
    size_t n_bytes = 0;
    int n;
    if (str.back() == 'M') {
        sscanf(str.c_str(), "%d", &n);
        n_bytes = (size_t)n * 1000 * 1000; // megabytes
    } else if (str.back() == 'G') {
        sscanf(str.c_str(), "%d", &n);
        n_bytes = (size_t)n * 1000 * 1000 * 1000; // gigabytes
    } else {
        throw std::invalid_argument("error: supported units are M (megabytes) or G (gigabytes), but got: " + std::string(1, str.back()));
    }
    if (n <= 0) {
        throw std::invalid_argument("error: size must be a positive value");
    }
    return n_bytes;
}

static void split_params_parse_ex(int argc, const char ** argv, split_params & params) {
    std::string arg;
    const std::string arg_prefix = "--";
    bool invalid_param = false;

    int arg_idx = 1;
    for (; arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0; arg_idx++) {
        arg = argv[arg_idx];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        bool arg_found = false;
        if (arg == "-h" || arg == "--help") {
            split_print_usage(argv[0]);
            exit(0);
        } else if (arg == "--version") {
            fprintf(stderr, "version: %d (%s)\n", LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
            fprintf(stderr, "built with %s for %s\n", LLAMA_COMPILER, LLAMA_BUILD_TARGET);
            exit(0);
        } else if (arg == "--dry-run") {
            arg_found = true;
            params.dry_run = true;
        } else if (arg == "--no-tensor-first-split") {
            arg_found = true;
            params.no_tensor_first_split = true;
        } else if (arg == "--merge") {
            arg_found = true;
            if (params.operation != OP_NONE && params.operation != OP_MERGE) {
                throw std::invalid_argument("error: either --split or --merge can be specified, but not both");
            }
            params.operation = OP_MERGE;
        } else if (arg == "--split") {
            arg_found = true;
            if (params.operation != OP_NONE && params.operation != OP_SPLIT) {
                throw std::invalid_argument("error: either --split or --merge can be specified, but not both");
            }
            params.operation = OP_SPLIT;
        } else if (arg == "--split-max-tensors") {
            if (++arg_idx >= argc) {
                invalid_param = true;
                break;
            }
            arg_found = true;
            if (params.mode != MODE_NONE && params.mode != MODE_TENSOR) {
                throw std::invalid_argument("error: either --split-max-tensors or --split-max-size can be specified, but not both");
            }
            params.mode = MODE_TENSOR;
            params.n_split_tensors = atoi(argv[arg_idx]);
        } else if (arg == "--split-max-size") {
            if (++arg_idx >= argc) {
                invalid_param = true;
                break;
            }
            arg_found = true;
            if (params.mode != MODE_NONE && params.mode != MODE_SIZE) {
                throw std::invalid_argument("error: either --split-max-tensors or --split-max-size can be specified, but not both");
            }
            params.mode = MODE_SIZE;
            params.n_bytes_split = split_str_to_n_bytes(argv[arg_idx]);
        }

        if (!arg_found) {
            throw std::invalid_argument("error: unknown argument: " + arg);
        }
    }

    // the operation is split if not specified
    if (params.operation == OP_NONE) {
        params.operation = OP_SPLIT;
    }
    // the split mode is by tensor if not specified
    if (params.mode == MODE_NONE) {
        params.mode = MODE_TENSOR;
    }

    if (invalid_param) {
        throw std::invalid_argument("error: invalid parameter for argument: " + arg);
    }

    if (argc - arg_idx != 2) {
        throw std::invalid_argument("error: bad arguments");
    }

    params.input = argv[arg_idx++];
    params.output = argv[arg_idx++];
}

static bool split_params_parse(int argc, const char ** argv, split_params & params) {
    bool result = true;
    try {
        split_params_parse_ex(argc, argv, params);
    }
    catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        split_print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return result;
}

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

struct split_strategy {
    const split_params params;
    std::ifstream & f_input;
    struct gguf_context * ctx_gguf;
    struct ggml_context * ctx_meta = NULL;
    const int n_tensors;

    // one ctx_out per one output file
    std::vector<struct gguf_context *> ctx_outs;

    // temporary buffer for reading in tensor data
    std::vector<uint8_t> read_buf;

    split_strategy(const split_params & params,
            std::ifstream & f_input,
            struct gguf_context * ctx_gguf,
            struct ggml_context * ctx_meta) :
        params(params),
        f_input(f_input),
        ctx_gguf(ctx_gguf),
        ctx_meta(ctx_meta),
        n_tensors(gguf_get_n_tensors(ctx_gguf)) {

        // because we need to know list of tensors for each file in advance, we will build all the ctx_out for all output splits
        int i_split = -1;
        struct gguf_context * ctx_out = NULL;
        auto new_ctx_out = [&](bool allow_no_tensors) {
            i_split++;
            if (ctx_out != NULL) {
                if (gguf_get_n_tensors(ctx_out) == 0 && !allow_no_tensors) {
                    fprintf(stderr, "error: one of splits have 0 tensors. Maybe size or tensors limit is too small\n");
                    exit(EXIT_FAILURE);
                }
                ctx_outs.push_back(ctx_out);
            }
            ctx_out = gguf_init_empty();
            // Save all metadata in first split only
            if (i_split == 0) {
                gguf_set_kv(ctx_out, ctx_gguf);
            }
            gguf_set_val_u16(ctx_out, LLM_KV_SPLIT_NO, i_split);
            gguf_set_val_u16(ctx_out, LLM_KV_SPLIT_COUNT, 0); // placeholder
            gguf_set_val_i32(ctx_out, LLM_KV_SPLIT_TENSORS_COUNT, n_tensors);
        };

        // initialize ctx_out for the first split
        new_ctx_out(false);

        // skip first split if no_tensor_first_split is set
        if (params.no_tensor_first_split) {
            new_ctx_out(true);
        }

        // process tensors one by one
        size_t curr_tensors_size = 0; // current size by counting only tensors size (without metadata)
        for (int i = 0; i < n_tensors; ++i) {
            struct ggml_tensor * t = ggml_get_tensor(ctx_meta, gguf_get_tensor_name(ctx_gguf, i));
            // calculate the "imaginary" size = the current size + next tensor size
            size_t n_bytes = GGML_PAD(ggml_nbytes(t), GGUF_DEFAULT_ALIGNMENT);
            size_t next_tensors_size = curr_tensors_size + n_bytes;
            if (should_split(i, next_tensors_size)) {
                new_ctx_out(false);
                curr_tensors_size = n_bytes;
            } else {
                curr_tensors_size = next_tensors_size;
            }
            gguf_add_tensor(ctx_out, t);
        }

        // push the last ctx_out
        ctx_outs.push_back(ctx_out);

        // set the correct n_split for all ctx_out
        for (auto & ctx : ctx_outs) {
            gguf_set_val_u16(ctx, LLM_KV_SPLIT_COUNT, ctx_outs.size());
        }
    }

    ~split_strategy() {
        for (auto & ctx_out : ctx_outs) {
            gguf_free(ctx_out);
        }
    }

    bool should_split(int i_tensor, size_t next_size) {
        if (params.mode == MODE_SIZE) {
            // split by max size per file
            return next_size > params.n_bytes_split;
        } else if (params.mode == MODE_TENSOR) {
            // split by number of tensors per file
            return i_tensor > 0 && i_tensor < n_tensors && i_tensor % params.n_split_tensors == 0;
        }
        // should never happen
        GGML_ABORT("invalid mode");
    }

    void print_info() {
        printf("n_split: %zu\n", ctx_outs.size());
        int i_split = 0;
        for (auto & ctx_out : ctx_outs) {
            // re-calculate the real gguf size for each split (= metadata size + total size of all tensors)
            size_t total_size = gguf_get_meta_size(ctx_out);
            for (int i = 0; i < gguf_get_n_tensors(ctx_out); ++i) {
                struct ggml_tensor * t = ggml_get_tensor(ctx_meta, gguf_get_tensor_name(ctx_out, i));
                total_size += ggml_nbytes(t);
            }
            total_size = total_size / 1000 / 1000; // convert to megabytes
            printf("split %05d: n_tensors = %" PRIi64 ", total_size = %zuM\n", i_split + 1, gguf_get_n_tensors(ctx_out), total_size);
            i_split++;
        }
    }

    void write() {
        int i_split = 0;
        int n_split = ctx_outs.size();
        for (auto & ctx_out : ctx_outs) {
            // construct file path
            char split_path[PATH_MAX] = {0};
            llama_split_path(split_path, sizeof(split_path), params.output.c_str(), i_split, n_split);

            // open the output file
            printf("Writing file %s ... ", split_path);
            fflush(stdout);
            std::ofstream fout = std::ofstream(split_path, std::ios::binary);
            fout.exceptions(std::ofstream::failbit); // fail fast on write errors

            // write metadata
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_out));
            gguf_get_meta_data(ctx_out, data.data());
            fout.write((const char *)data.data(), data.size());

            // write tensors
            for (int i = 0; i < gguf_get_n_tensors(ctx_out); ++i) {
                // read tensor meta and prepare buffer
                const char * t_name = gguf_get_tensor_name(ctx_out, i);
                struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);
                auto n_bytes = ggml_nbytes(t);
                read_buf.resize(n_bytes);

                // calculate offset
                auto i_tensor_in = gguf_find_tensor(ctx_gguf, t_name); // idx of tensor in the input file
                auto offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i_tensor_in);

                // copy tensor from input to output file
                copy_file_to_file(f_input, fout, offset, n_bytes);
                zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
            }

            printf("done\n");
            // close the file
            fout.close();
            i_split++;
        }
    }

    void copy_file_to_file(std::ifstream & f_in, std::ofstream & f_out, const size_t in_offset, const size_t len) {
        // TODO: detect OS and use copy_file_range() here for better performance
        if (read_buf.size() < len) {
            read_buf.resize(len);
        }
        f_in.seekg(in_offset);
        f_in.read((char *)read_buf.data(), len);
        f_out.write((const char *)read_buf.data(), len);
    }
};

static void gguf_split(const split_params & split_params) {
    struct ggml_context * ctx_meta = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_meta,
    };

    std::ifstream f_input(split_params.input.c_str(), std::ios::binary);
    if (!f_input.is_open()) {
        fprintf(stderr, "%s:  failed to open input GGUF from %s\n", __func__, split_params.input.c_str());
        exit(EXIT_FAILURE);
    }

    auto * ctx_gguf = gguf_init_from_file(split_params.input.c_str(), params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s:  failed to load input GGUF from %s\n", __func__, split_params.input.c_str());
        exit(EXIT_FAILURE);
    }

    // prepare the strategy
    split_strategy strategy(split_params, f_input, ctx_gguf, ctx_meta);
    int n_split = strategy.ctx_outs.size();
    strategy.print_info();

    if (!split_params.dry_run) {
        // write all output splits
        strategy.write();
    }

    // done, clean up
    gguf_free(ctx_gguf);
    f_input.close();

    fprintf(stderr, "%s: %d gguf split written with a total of %d tensors.\n",
            __func__, n_split, strategy.n_tensors);
}

static void gguf_merge(const split_params & split_params) {
    fprintf(stderr, "%s: %s -> %s\n",
            __func__, split_params.input.c_str(),
            split_params.output.c_str());
    int n_split = 1;
    int total_tensors = 0;

    // avoid overwriting existing output file
    if (std::ifstream(split_params.output.c_str())) {
        fprintf(stderr, "%s: output file %s already exists\n", __func__, split_params.output.c_str());
        exit(EXIT_FAILURE);
    }

    std::ofstream fout(split_params.output.c_str(), std::ios::binary);
    fout.exceptions(std::ofstream::failbit); // fail fast on write errors

    auto * ctx_out = gguf_init_empty();

    std::vector<uint8_t> read_data;
    std::vector<ggml_context *> ctx_metas;
    std::vector<gguf_context *> ctx_ggufs;

    char split_path[PATH_MAX] = {0};
    strncpy(split_path, split_params.input.c_str(), sizeof(split_path) - 1);
    char split_prefix[PATH_MAX] = {0};

    // First pass to find KV and tensors metadata
    for (int i_split = 0; i_split < n_split; i_split++) {
        struct ggml_context * ctx_meta = NULL;

        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx_meta,
        };

        if (i_split > 0) {
            llama_split_path(split_path, sizeof(split_path), split_prefix, i_split, n_split);
        }
        fprintf(stderr, "%s: reading metadata %s ...", __func__, split_path);

        auto * ctx_gguf = gguf_init_from_file(split_path, params);
        if (!ctx_gguf) {
            fprintf(stderr, "\n%s:  failed to load input GGUF from %s\n", __func__, split_params.input.c_str());
            exit(EXIT_FAILURE);
        }
        ctx_ggufs.push_back(ctx_gguf);
        ctx_metas.push_back(ctx_meta);

        if (i_split == 0) {
            auto key_n_split = gguf_find_key(ctx_gguf, LLM_KV_SPLIT_COUNT);
            if (key_n_split < 0) {
                fprintf(stderr,
                        "\n%s: input file does not contain %s metadata\n",
                        __func__,
                        LLM_KV_SPLIT_COUNT);
                gguf_free(ctx_gguf);
                ggml_free(ctx_meta);
                gguf_free(ctx_out);
                fout.close();
                exit(EXIT_FAILURE);
            }

            n_split = gguf_get_val_u16(ctx_gguf, key_n_split);
            if (n_split < 1) {
                fprintf(stderr,
                        "\n%s: input file does not contain a valid split count %d\n",
                        __func__,
                        n_split);
                gguf_free(ctx_gguf);
                ggml_free(ctx_meta);
                gguf_free(ctx_out);
                fout.close();
                exit(EXIT_FAILURE);
            }

            // Verify the file naming and extract split_prefix
            if (!llama_split_prefix(split_prefix, sizeof (split_prefix), split_path, i_split, n_split)) {
                fprintf(stderr, "\n%s: unexpected input file name: %s"
                                " i_split=%d"
                                " n_split=%d\n", __func__,
                        split_path, i_split, n_split);
                gguf_free(ctx_gguf);
                ggml_free(ctx_meta);
                gguf_free(ctx_out);
                fout.close();
                exit(EXIT_FAILURE);
            }

            // Do not trigger merge if we try to merge again the output
            gguf_set_val_u16(ctx_gguf, LLM_KV_SPLIT_COUNT, 0);

            // Set metadata from the first split
            gguf_set_kv(ctx_out, ctx_gguf);
        }

        auto n_tensors = gguf_get_n_tensors(ctx_gguf);
        for (int i_tensor = 0; i_tensor < n_tensors; i_tensor++) {
            const char * t_name = gguf_get_tensor_name(ctx_gguf, i_tensor);
            struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);
            gguf_add_tensor(ctx_out, t);
        }
        total_tensors += n_tensors;

        fprintf(stderr, "\033[3Ddone\n");
    }

    // placeholder for the meta data
    {
        auto meta_size = gguf_get_meta_size(ctx_out);
        ::zeros(fout, meta_size);
    }

    // Write tensors data
    for (int i_split = 0; i_split < n_split; i_split++) {
        llama_split_path(split_path, sizeof(split_path), split_prefix, i_split, n_split);
        std::ifstream f_input(split_path, std::ios::binary);
        if (!f_input.is_open()) {
            fprintf(stderr, "%s:  failed to open input GGUF from %s\n", __func__, split_path);
            for (uint32_t i = 0; i < ctx_ggufs.size(); i++) {
                gguf_free(ctx_ggufs[i]);
                ggml_free(ctx_metas[i]);
            }
            gguf_free(ctx_out);
            fout.close();
            exit(EXIT_FAILURE);
        }
        fprintf(stderr, "%s: writing tensors %s ...", __func__, split_path);

        auto * ctx_gguf = ctx_ggufs[i_split];
        auto * ctx_meta = ctx_metas[i_split];

        auto n_tensors = gguf_get_n_tensors(ctx_gguf);
        for (int i_tensor = 0; i_tensor < n_tensors; i_tensor++) {
            const char * t_name = gguf_get_tensor_name(ctx_gguf, i_tensor);
            struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);

            auto n_bytes = ggml_nbytes(t);

            if (read_data.size() < n_bytes) {
                read_data.resize(n_bytes);
            }

            auto offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i_tensor);
            f_input.seekg(offset);
            f_input.read((char *)read_data.data(), n_bytes);

            // write tensor data + padding
            fout.write((const char *)read_data.data(), n_bytes);
            zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
        }

        gguf_free(ctx_gguf);
        ggml_free(ctx_meta);
        f_input.close();
        fprintf(stderr, "\033[3Ddone\n");
    }

    {
        // go back to beginning of file and write the updated metadata
        fout.seekp(0);
        std::vector<uint8_t> data(gguf_get_meta_size(ctx_out));
        gguf_get_meta_data(ctx_out, data.data());
        fout.write((const char *)data.data(), data.size());

        fout.close();
        gguf_free(ctx_out);
    }

    fprintf(stderr, "%s: %s merged from %d split with %d tensors.\n",
            __func__, split_params.output.c_str(), n_split, total_tensors);
}

int main(int argc, const char ** argv) {
    split_params params;
    split_params_parse(argc, argv, params);

    switch (params.operation) {
        case OP_SPLIT: gguf_split(params);
            break;
        case OP_MERGE: gguf_merge(params);
            break;
        default: split_print_usage(argv[0]);
            exit(EXIT_FAILURE);
    }

    return 0;
}
