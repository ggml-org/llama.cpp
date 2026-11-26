#include "ggml.h"
#include "gguf.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "llama.h"
#include "pca.hpp"
#include "mean.hpp"

#include <clocale>

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>


//////////////////////////////////////////////////
// utils

template <class Iter>
static std::string tokens_to_str(llama_context * ctx, Iter begin, Iter end) {
    std::string ret;
    for (; begin != end; ++begin) {
        ret += common_token_to_piece(ctx, *begin);
    }

    return ret;
}

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    CPU only:   %s -m ./llama-3.Q4_K_M.gguf\n", argv[0]);
    printf("\n    with GPU:   %s -m ./llama-3.Q4_K_M.gguf -ngl 99\n", argv[0]);
    printf("\n    advanced:   %s -m ./llama-3.Q4_K_M.gguf -ngl 99 --pca-iter 2000 --pca-batch 100\n", argv[0]);
    printf("\n    using mean: %s -m ./llama-3.Q4_K_M.gguf --method mean\n", argv[0]);
    printf("\n");
}

//////////////////////////////////////////////////


// cb_eval is reused for each pair of positive - negative prompt
struct callback_data {
    ggml_context * ctx_ggml = nullptr;   // holds v_pos, v_neg, v_diff_filtered

    int n_layers = 0;
    int n_embd = 0;
    int64_t n_tokens = 0;
    bool is_eval_pos = true;

    // Slot il - 1 stores the capture for block il.
    std::vector<struct ggml_tensor *> v_pos;
    std::vector<struct ggml_tensor *> v_neg;
    std::vector<struct ggml_tensor *> v_diff_filtered;   // vector of matrices of size [n_embd, n_nonzero_rows]. NOTE: n_nonzero_rows maybe different for each layer

    void save_tensor_for_layer(struct ggml_tensor * t, int il) {
        if (ctx_ggml == nullptr) {
            // alloc a new ctx_ggml if needed
            struct ggml_init_params params_ggml = {
                /*.mem_size   =*/ ggml_tensor_overhead() * n_layers * 3u,
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ctx_ggml = ggml_init(params_ggml);
        }

        auto & t_layer = (is_eval_pos ? v_pos : v_neg)[il - 1];
        if (!t_layer) {
            t_layer = ggml_new_tensor_2d(ctx_ggml, t->type, t->ne[0], t->ne[1]);
            t_layer->data = malloc(ggml_nbytes(t_layer));
            ggml_set_name(t_layer, ggml_get_name(t));
        }
        // BitNet can name both sides of the steering add. The later node is the block output.
        // Gemma 3n exposes [n_embd, n_tokens, n_altup]; copy only the first [n_embd, n_tokens] plane.
        GGML_ASSERT(t_layer->ne[0] == t->ne[0] && t_layer->ne[1] == t->ne[1]);
        ggml_backend_tensor_get(t, t_layer->data, 0, ggml_nbytes(t_layer));
    }

    // calculate diff (v_pos - v_neg) and place the result back to v_pos
    // all zero rows in the diff tensor will also be removed
    bool calc_diff() {
        for (size_t i = 0; i < v_pos.size(); ++i) {
            auto * pos = v_pos[i];
            auto * neg = v_neg[i];
            if (!pos || !neg || !ggml_are_same_shape(pos, neg)) {
                fprintf(stderr, "%s: missing or mismatched l_out-%zu captures\n", __func__, i + 1);
                return false;
            }
            float * a = (float *) pos->data;
            const float * b = (const float *) neg->data;
            const size_t n_elem = ggml_nelements(pos);
            for (size_t j = 0; j < n_elem; j++) {
                a[j] -= b[j];
            }
            v_diff_filtered.push_back(filter_nonzero_rows(pos));
        }
        return true;
    }

    // delete zero rows from a given 2D tensor
    struct ggml_tensor * filter_nonzero_rows(struct ggml_tensor * a) {
        //printf("filter_nonzero_rows\n");
        auto is_row_all_zeros = [](struct ggml_tensor * t, int row, float eps) -> bool {
            // check if given row containing all zero elements
            int n_cols = t->ne[0]; // hint: should be equal to n_embd
            for (int col = 0; col < n_cols; ++col) {
                if (std::fabs(ggml_get_f32_nd(t, col, row, 0, 0)) > eps) {
                    return false;
                }
            }
            return true;
        };
        std::vector<int> rows_to_copy; // the idx of non-zero cols (to be copied to row of diff_filtered)
        for (int i_row = 0; i_row < a->ne[1]; i_row++) {
            if (!is_row_all_zeros(a, i_row, 1e-6)) {
                rows_to_copy.push_back(i_row);
            }
        }

        // get "n_nonzero_rows" for the output "diff_filtered"
        int n_nonzero_rows = rows_to_copy.size();
        //printf("n_nonzero_rows: %d\n", n_nonzero_rows);
        int n_embd = a->ne[0];
        if (n_nonzero_rows == 0) {
            return nullptr;
        }

        // diff_filtered: [n_embd, n_nonzero_rows]
        struct ggml_tensor * diff_filtered = ggml_new_tensor_2d(
            ctx_ggml, GGML_TYPE_F32, n_embd, n_nonzero_rows);
        ggml_format_name(diff_filtered, "diff_filtered_%s", a->name);
        diff_filtered->data = malloc(ggml_nbytes(diff_filtered));

        // copy non-zero rows
        for (int dest_row = 0; dest_row < n_nonzero_rows; dest_row++) {
            int src_row = rows_to_copy[dest_row];
            for (int i = 0; i < n_embd; i++) {
                float src_elem = ggml_get_f32_nd(a, i, src_row, 0, 0);
                ggml_set_f32_nd(diff_filtered, i, dest_row, 0, 0, src_elem);
            }
        }

        //print_debug_tensor(diff_filtered);

        return diff_filtered;
    }

    // we don't implement destructor, because we want to reuse callback_data. we just want to free the tensors
    void reset() {
        for (auto ptr : v_pos)           if (ptr) free(ptr->data);
        for (auto ptr : v_neg)           if (ptr) free(ptr->data);
        for (auto ptr : v_diff_filtered) if (ptr) free(ptr->data);
        std::fill(v_pos.begin(), v_pos.end(), nullptr);
        std::fill(v_neg.begin(), v_neg.end(), nullptr);
        v_diff_filtered.clear();
        if (ctx_ggml) {
            ggml_free(ctx_ggml);
        }
        ctx_ggml = nullptr;
    }
};

/**
 * process_ctx is used to store the ggml context for pre-post processing the diff vectors
 * in short, input => v_diff and output => v_final
 */
struct train_context {
    ggml_context * ctx_ggml;
    int n_embd;
    int n_layers;

    /* pair of prompts to be used for generating final vector */
    std::vector<std::string> positive_entries;
    std::vector<std::string> negative_entries;

    // each element of the vector correspond to one layer
    // Layer 0 has no control vector, so slot il - 1 stores the data for layer il.
    // NOTE (2): v_diff is transposed from v_diff_tmp
    std::vector<struct ggml_tensor *> v_diff;  // vector of matrices of size [m, n_embd] where m ~ n_tokens * n_completions (v_diff contains no zero-rows)
    std::vector<struct ggml_tensor *> v_final; // vector of vectors of size [n_embd] to be written to file

    // to easily re-alloc when concat v_diff, we temporary store v_diff in a vector instead of a tensor
    // v_diff_tmp will get converted unto v_diff later on
    std::vector<std::vector<uint8_t>> v_diff_tmp;

    train_context(int n_embd_, int n_layers_) {
        n_embd = n_embd_;
        n_layers = n_layers_;
        struct ggml_init_params params_ggml = {
            /*.mem_size   =*/ ggml_tensor_overhead() * (n_layers - 1) * 2u,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx_ggml = ggml_init(params_ggml);
        for (int il = 0; il < n_layers - 1; il++) {
            std::vector<uint8_t> empty;
            v_diff_tmp.push_back(empty);
            auto t = ggml_new_tensor_1d(ctx_ggml, GGML_TYPE_F32, n_embd);
            t->data = malloc(ggml_nbytes(t)); // TODO: get rid of malloc if possible
            v_final.push_back(t);
        }
    }

    // add new rows into existing tensor in v_diff_tmp
    void concat_diff_tmp(const std::vector<struct ggml_tensor *> & diff_filtered) {
        GGML_ASSERT((int) diff_filtered.size() == n_layers - 1);
        for (int il = 0; il < n_layers - 1; il++) {
            auto t = diff_filtered[il];
            if (!t) {
                continue;
            }
            auto & diff_tmp = v_diff_tmp[il];
            size_t curr_size = diff_tmp.size();
            diff_tmp.resize(curr_size + ggml_nbytes(t));
            memcpy(diff_tmp.data() + curr_size, t->data, ggml_nbytes(t));
        }
    }

    // build the v_diff tensors from v_diff_tmp (v_diff need to be transposed)
    // TODO @ngxson : maybe add option NOT to transpose v_diff; will be useful for "mean" method
    bool build_v_diff(bool transpose) {
        printf("build_v_diff\n");
        for (int il = 0; il < n_layers - 1; il++) {
            auto & diff_tmp = v_diff_tmp[il];
            int n_elem = diff_tmp.size() / sizeof(float);
            GGML_ASSERT(n_elem % n_embd == 0);
            int n_rows = n_elem / n_embd;
            if (n_rows == 0) {
                fprintf(stderr, "%s: no nonzero differences for layer %d\n", __func__, il + 1);
                return false;
            }
            struct ggml_tensor * diff = transpose
                ? ggml_new_tensor_2d(ctx_ggml, GGML_TYPE_F32, n_rows, n_embd)
                : ggml_new_tensor_2d(ctx_ggml, GGML_TYPE_F32, n_embd, n_rows);
            ggml_set_name(diff, (std::string("diff_") + std::to_string(il)).c_str());
            diff->data = malloc(ggml_nbytes(diff)); // TODO: get rid of this malloc if possible
            if (transpose) {
                // copy data & transpose
                float * arr = (float *) diff_tmp.data();
                for (int ir = 0; ir < n_rows; ++ir) {
                    for (int ic = 0; ic < n_embd; ++ic) {
                        float f = arr[ir*n_embd + ic];
                        ggml_set_f32_nd(diff, ir, ic, 0, 0, f);
                    }
                }
            } else {
                // only copy
                memcpy(diff->data, diff_tmp.data(), ggml_nbytes(diff));
            }
            v_diff.push_back(diff);
            print_debug_tensor(diff);
            // free memory of diff_tmp
            diff_tmp.resize(0);
        }
        return true;
    }

    ~train_context() {
        for (auto ptr : v_final) free(ptr->data);
        for (auto ptr : v_diff) free(ptr->data);
        // no need to free v_diff_tmp, since we didn't use malloc
        ggml_free(ctx_ggml);
    }
};

struct tokenized_prompt {
    std::vector<llama_token> tokens_pos;
    std::vector<llama_token> tokens_neg;
    size_t n_tokens_pos;
    size_t n_tokens_neg;
    size_t max_seq_len;

    tokenized_prompt(llama_context * ctx, std::string pos, std::string neg) {
        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const bool add_bos = llama_vocab_get_add_bos(vocab);
        tokens_pos = common_tokenize(ctx, pos, add_bos, true);
        tokens_neg = common_tokenize(ctx, neg, add_bos, true);
        n_tokens_pos = tokens_pos.size();
        n_tokens_neg = tokens_neg.size();
        max_seq_len = std::max(n_tokens_pos, n_tokens_neg);
        padding_seq(ctx, tokens_pos, max_seq_len);
        padding_seq(ctx, tokens_neg, max_seq_len);
    }

    void padding_seq(llama_context * ctx, std::vector<llama_token> & tokens, size_t len) {
        if (tokens.size() >= len) {
            return;
        }
        // TODO: customize padding token
        std::vector<llama_token> pad_tokens = common_tokenize(ctx, " ", false);
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
        llama_token pad_tok = !pad_tokens.empty() ? pad_tokens.back() : llama_vocab_pad(vocab);
        if (pad_tok == LLAMA_TOKEN_NULL) {
            pad_tok = llama_vocab_eos(vocab);
        }
        while (tokens.size() < len) {
            tokens.push_back(pad_tok);
        }
    }
};

//////////////////////////////////////////////////

template <typename T>
static std::string to_string(const T & val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
}

static std::vector<std::string> ctrlvec_load_prompt_file(std::string path, bool skip_empty_lines) {
    std::vector<std::string> output;
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "error: unable to open file: %s\n", path.c_str());
        exit(1);
    }
    std::string line;
    while (std::getline(file, line)) {
        bool is_skip = skip_empty_lines && line.empty();
        if (!is_skip) {
            string_process_escapes(line);
            output.push_back(line);
        }
    }
    file.close();
    return output;
}

//////////////////////////////////////////////////

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;
    int il;
    char suffix;
    const bool capture = std::sscanf(t->name, "l_out-%d%c", &il, &suffix) == 1 &&
                         il > 0 && il < cb_data->n_layers &&
                         t->type == GGML_TYPE_F32 && t->ne[3] == 1 && ggml_is_contiguous(t) &&
                         t->ne[0] == cb_data->n_embd && (t->ne[1] == cb_data->n_tokens || t->ne[1] == 1);
    if (ask) {
        return capture;
    }
    if (capture) {
        cb_data->save_tensor_for_layer(t, il);
    }
    return true;
}

static bool get_hidden_layers(llama_context * ctx, const std::vector<llama_token> & tokens, size_t n_real_tokens) {
    llama_memory_clear(llama_get_memory(ctx), true);
    common_batch batch = common_batch_get_one(ctx, tokens);
    if (n_real_tokens > 0 && n_real_tokens < tokens.size()) {
        batch.set_output((int32_t) tokens.size() - 1, false);
        batch.set_output((int32_t) n_real_tokens - 1, true);
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        return false;
    }
    return true;
}

static void export_gguf(const std::vector<struct ggml_tensor *> & v_ctrl, const std::string fname, const std::string model_hint) {
    struct gguf_context * ctx = gguf_init_empty();

    const std::string arch = "controlvector";
    gguf_set_val_str(ctx, "general.architecture", arch.c_str());
    gguf_set_val_str(ctx, (arch + ".model_hint").c_str(), model_hint.c_str());
    gguf_set_val_i32(ctx, (arch + ".layer_count").c_str(), v_ctrl.size());

    for (size_t i = 0; i < v_ctrl.size(); ++i) {
        gguf_add_tensor(ctx, v_ctrl[i]);
        print_debug_tensor(v_ctrl[i]);
        printf("Added tensor: %s\n", v_ctrl[i]->name);
    }

    printf("%s: writing file...\n", __func__);
    gguf_write_to_file(ctx, fname.c_str(), false);
    printf("%s: wrote file '%s'\n", __func__, fname.c_str());
    gguf_free(ctx);
}

/**
 * Load prompt files and completion file.
 * Then format each pair of prompt + completion to make an entry.
 */
static int prepare_entries(common_params & params, train_context & ctx_train) {
    // load prompts
    std::vector<std::string> positive_prompts = ctrlvec_load_prompt_file(params.cvector_positive_file, true);
    std::vector<std::string> negative_prompts = ctrlvec_load_prompt_file(params.cvector_negative_file, true);
    if (positive_prompts.size() != negative_prompts.size()) {
        fprintf(stderr, "number of positive and negative prompts must be equal\n");
        return 1;
    }
    if (positive_prompts.empty()) {
        fprintf(stderr, "must provide at least one prompt pair\n");
        return 1;
    }
    ctx_train.positive_entries = positive_prompts;
    ctx_train.negative_entries = negative_prompts;
    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    params.out_file = "control_vector.gguf";

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CVECTOR_GENERATOR, print_usage)) {
        return 1;
    }

    if (params.n_pca_iterations % params.n_pca_batch != 0) {
        fprintf(stderr, "PCA iterations must by multiply of PCA batch size\n");
        return 1;
    }


    callback_data cb_data;

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    llama_print_build_info(llama_version());
    llama_backend_init();
    struct backend_guard {
        ~backend_guard() { llama_backend_free(); }
    } llama_guard;
    llama_numa_init(params.numa);

    // load the model to get hparams
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) {
        return 1;
    }

    // int n_ctx = llama_n_ctx(ctx);
    int n_layers = llama_model_n_layer(model);
    int n_embd = llama_model_n_embd(model);
    if (n_layers < 2) {
        fprintf(stderr, "control vector generation requires at least two layers\n");
        return 1;
    }

    cb_data.n_layers = n_layers;
    cb_data.n_embd = n_embd;
    cb_data.v_pos.resize(n_layers - 1);
    cb_data.v_neg.resize(n_layers - 1);

    // get model hint param (a.k.a model arch name)
    char model_hint[128];
    llama_model_meta_val_str(model, "general.architecture", model_hint, 128);

    // init train_context
    train_context ctx_train(n_embd, n_layers);

    // load and prepare entries for training
    if (prepare_entries(params, ctx_train) != 0) {
        return 1;
    }

    // we have to pretokenize everything because otherwise we don't know how much overhead to allocate ctx_diffs_wrapped
    std::vector<tokenized_prompt> tokenized_prompts;
    size_t n_total_tokens = 0;
    for (size_t i = 0; i < ctx_train.positive_entries.size(); ++i) {
        tokenized_prompt t(ctx, ctx_train.positive_entries[i], ctx_train.negative_entries[i]);
        if (t.n_tokens_pos == 0 || t.n_tokens_neg == 0) {
            fprintf(stderr, "prompt pair %zu produced an empty token sequence\n", i + 1);
            return 1;
        }
        n_total_tokens += 2 * t.max_seq_len;
        tokenized_prompts.push_back(std::move(t));
    }

    std::cout << "n_total_tokens: " << n_total_tokens << std::endl;

    for (size_t i = 0; i < ctx_train.positive_entries.size(); ++i) {
        const auto & t = tokenized_prompts[i];
        if (t.max_seq_len > llama_n_ubatch(ctx)) {
            fprintf(stderr, "prompt pair %zu exceeds the effective microbatch size (%u), increase -ub/-b/-c\n",
                    i + 1, llama_n_ubatch(ctx));
            return 1;
        }
        cb_data.n_tokens = t.max_seq_len;

        printf("Evaluating prompt[%d/%d]: \"%s\" - \"%s\" (%d tokens)\n",
            (int) i+1, (int) ctx_train.positive_entries.size(),
            tokens_to_str(ctx, t.tokens_pos.cbegin(), t.tokens_pos.cend()).c_str(),
            tokens_to_str(ctx, t.tokens_neg.cbegin(), t.tokens_neg.cend()).c_str(),
            (int) t.max_seq_len);

        cb_data.is_eval_pos = true;
        const bool pos_ok = get_hidden_layers(ctx, t.tokens_pos, t.n_tokens_pos);
        cb_data.is_eval_pos = false;
        if (!pos_ok || !get_hidden_layers(ctx, t.tokens_neg, t.n_tokens_neg) || !cb_data.calc_diff()) {
            cb_data.reset();
            return 1;
        }

        ctx_train.concat_diff_tmp(cb_data.v_diff_filtered);

        // reset for next iteration
        cb_data.reset();
    }

    // done with the model, we can now free it to make gain some memory
    printf("Done evaluate prompts, unload model...\n");
    llama_init.reset();

    bool use_pca = params.cvector_dimre_method == DIMRE_METHOD_PCA;

    // prepare ctx_train for PCA
    if (!ctx_train.build_v_diff(use_pca)) {
        return 1;
    }

    if (use_pca) {
        // run PCA
        PCA::pca_params pca_params;
        pca_params.n_threads    = params.cpuparams.n_threads;
        pca_params.n_batch      = params.n_pca_batch;
        pca_params.n_iterations = params.n_pca_iterations;
        PCA::run_pca(pca_params, ctx_train.v_diff, ctx_train.v_final);
    } else {
        // run mean
        if (!mean::run(ctx_train.v_diff, ctx_train.v_final)) {
            return 1;
        }
    }

    // write output vectors to gguf
    export_gguf(ctx_train.v_final, params.out_file, model_hint);

    return 0;
}
