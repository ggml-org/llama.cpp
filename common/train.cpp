#include "train.h"
#include "common.h"

#include <algorithm>
#include <random>
#include <sstream>
#include <functional>
#include <cstring>

struct random_normal_distribution {
    std::mt19937 gen;
    std::normal_distribution<float> rd;
    float min;
    float max;
};

struct random_uniform_distribution {
    std::mt19937 gen;
    std::uniform_real_distribution<float> rd;
};

struct train_state  * init_train_state() {
    struct train_state * state = new struct train_state;
    state->train_its     = 0;
    state->train_samples = 0;
    state->train_tokens  = 0;
    state->train_epochs  = 0;
    state->shuffle_samples_hash  = 0;
    state->shuffle_sample_count  = 0;
    state->shuffle_next_sample   = 0;
    state->shuffle_rng_state_current = "";
    state->shuffle_rng_state_next    = "";

    state->opt = new struct ggml_opt_context;
    state->opt->ctx = NULL;
    state->opt->params = ggml_opt_default_params(GGML_OPT_TYPE_ADAM);
    state->opt->params.graph_size = LLAMA_TRAIN_MAX_NODES;
    state->opt->loss_after = 0.0f;

    return state;
}

void free_train_state(struct train_state  * state) {
    delete state->opt;
    delete state;
}

struct random_normal_distribution * init_random_normal_distribution(
    int seed, float mean, float std, float min, float max
) {
    struct random_normal_distribution * rnd = (struct random_normal_distribution *) malloc(sizeof(struct random_normal_distribution));
    rnd->gen = std::mt19937(seed);
    rnd->rd = std::normal_distribution<float>{mean, std};
    rnd->min = min;
    rnd->max = max;
    return rnd;
}

struct random_uniform_distribution * init_random_uniform_distribution(int seed, float min, float max) {
    struct random_uniform_distribution * rnd = (struct random_uniform_distribution *) malloc(sizeof(struct random_uniform_distribution));
    rnd->gen = std::mt19937(seed);
    rnd->rd = std::uniform_real_distribution<float>{min, max};
    return rnd;
}

void free_random_normal_distribution (struct random_normal_distribution  * rnd) {
    free(rnd);
}

void free_random_uniform_distribution(struct random_uniform_distribution * rnd) {
    free(rnd);
}

struct ggml_tensor * randomize_tensor_normal(struct ggml_tensor * tensor, struct random_normal_distribution * rnd) {
    float scale = 1.0f; // xavier
    switch (ggml_n_dims(tensor)) {
        case 1:
            scale /= sqrtf((float) tensor->ne[0]);
            for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0]);
                *dst = scale * frand_normal(rnd);
            }
            break;
        case 2:
            scale /= sqrtf((float) tensor->ne[0]+tensor->ne[1]);
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
                    *dst = scale * frand_normal(rnd);
                }
            }
            break;
        case 3:
            scale /= sqrtf((float) tensor->ne[0]+tensor->ne[1]);
            for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                    for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                        float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2]);
                        *dst = scale * frand_normal(rnd);
                    }
                }
            }
            break;
        case 4:
            scale /= sqrtf((float) tensor->ne[0]+tensor->ne[1]);
            for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
                for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                    for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                        for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                            float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3]);
                            *dst = scale * frand_normal(rnd);
                        }
                    }
                }
            }
            break;
        default:
            die("Unsupported tensor->n_dims");
    };
    return tensor;
}

struct ggml_tensor * randomize_tensor_uniform(struct ggml_tensor * tensor, struct random_uniform_distribution * rnd) {
    switch (ggml_n_dims(tensor)) {
        case 1:
            for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0]);
                *dst = frand_uniform(rnd);
            }
            break;
        case 2:
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
                    *dst = frand_uniform(rnd);
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                    for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                        float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2]);
                        *dst = frand_uniform(rnd);
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
                for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                    for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                        for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                            float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3]);
                            *dst = frand_uniform(rnd);
                        }
                    }
                }
            }
            break;
        default:
            die("Unsupported tensor->n_dims");
    };
    return tensor;
}

float frand() {
    return (float)rand()/((float)(RAND_MAX) + 1.0f);
}

float frand_normal(struct random_normal_distribution * rnd) {
    return fclamp(rnd->rd(rnd->gen), rnd->min, rnd->max);
}

float frand_uniform(struct random_uniform_distribution * rnd) {
    return rnd->rd(rnd->gen);
}

int clamp(const int v, const int min, const int max) {
    return ((v < min) ? (min) : (v > max) ? (max) : v);
}

float fclamp(const float v, const float min, const float max) {
    return ((v < min) ? (min) : (v > max) ? (max) : v);
}

void assert_shape_1d(struct ggml_tensor * tensor, int64_t ne0) {
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == 1);
    GGML_ASSERT(tensor->ne[2] == 1);
    GGML_ASSERT(tensor->ne[3] == 1);
}

void assert_shape_2d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1) {
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == ne1);
    GGML_ASSERT(tensor->ne[2] == 1);
    GGML_ASSERT(tensor->ne[3] == 1);
}

void assert_shape_3d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2) {
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == ne1);
    GGML_ASSERT(tensor->ne[2] == ne2);
    GGML_ASSERT(tensor->ne[3] == 1);
}

void assert_shape_4d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == ne1);
    GGML_ASSERT(tensor->ne[2] == ne2);
    GGML_ASSERT(tensor->ne[3] == ne3);
}

int64_t get_example_targets_batch(
    struct llama_context * lctx,
    struct ggml_tensor   * tokens_input,
    struct ggml_tensor   * target_probs,
    int64_t                example_id,
    const size_t         * samples_offs,
    const size_t         * samples_begin,
    const size_t         * samples_size,
          size_t           samples_count,
    const llama_token    * train_data,
    size_t                 n_train_data,
    bool                   separate_with_eos,
    bool                   separate_with_bos,
    bool                   fill_with_next_samples,
    bool                   sample_random_offsets
) {
    GGML_ASSERT(samples_count > 0);
    GGML_ASSERT(ggml_is_matrix(tokens_input));
    GGML_ASSERT(ggml_is_3d(target_probs));
    int64_t n_vocab  = target_probs->ne[0];
    int64_t n_tokens = tokens_input->ne[0];
    int64_t n_batch  = tokens_input->ne[1];
    GGML_ASSERT(n_vocab  == target_probs->ne[0]);
    GGML_ASSERT(n_tokens == target_probs->ne[1]);
    GGML_ASSERT(n_batch  == target_probs->ne[2]);

    int64_t used_samples = 0;

    ggml_set_f32(target_probs, 0.0f);
    llama_token bos = llama_token_bos(llama_get_model(lctx));
    llama_token eos = llama_token_eos(llama_get_model(lctx));
    // printf("%s: example_id=%d n_batch=%d n_train_samples=%zu\n", __func__, example_id, n_batch, n_train_samples);
    for (int k=0; k<n_batch; ++k) {
        // printf("%s: batch %d\n", __func__, k);
        size_t sample_idx   = (example_id + used_samples) % samples_count;
        size_t sample_offs  = sample_random_offsets ? samples_offs[sample_idx] : 0;
        size_t sample_begin = samples_begin[sample_idx];
        size_t sample_size  = samples_size[sample_idx];
        ++used_samples;

        // printf("%s: sample_idx=%zu sample=%zu\n", __func__, sample_idx, sample);
        GGML_ASSERT(sample_begin+sample_size-1 < n_train_data);

        ggml_set_i32_nd(tokens_input, 0, k, 0, 0, bos);
        bool sample_separation_eos = !separate_with_eos;
        bool sample_separation_bos = !separate_with_bos;
        for (int64_t i=0; i<n_tokens; ++i) {
            llama_token token = eos;
            if (sample_offs >= sample_size && fill_with_next_samples) {
                if (!sample_separation_eos) {
                    // insert eos token to separate samples
                    sample_separation_eos = true;
                } else if (!sample_separation_bos) {
                    // insert bos token to separate samples
                    sample_separation_bos = true;
                    token = bos;
                } else {
                    // sample separation is done, continue with next sample
                    sample_separation_eos = !separate_with_eos;
                    sample_separation_bos = !separate_with_bos;
                    sample_offs  = 0;
                    sample_idx   = (example_id + used_samples) % samples_count;
                    sample_begin = samples_begin[sample_idx];
                    sample_size  = samples_size[sample_idx];
                    ++used_samples;
                }
            }
            // note: no else-if here
            if (sample_offs < sample_size) {
                token = clamp(train_data[sample_begin+sample_offs], 0, (llama_token) (n_vocab - 1));
                ++sample_offs;
            }
            ggml_set_f32_nd(target_probs,  token, (int) i, (int) k, 0, +1.0f);
            if (i+1<n_tokens) {
                ggml_set_i32_nd(tokens_input, (int) (i + 1), (int) k, 0, 0, token);
            }
        }
    }

    return used_samples;
}

void mt19937_set_state(std::mt19937& rng, const std::string& rng_state) {
    std::stringstream s_rng_state;
    s_rng_state.imbue(std::locale::classic());
    s_rng_state.exceptions(std::stringstream::failbit);
    s_rng_state.str(rng_state);
    s_rng_state >> rng;
}

std::string mt19937_get_state(const std::mt19937& rng) {
    std::stringstream s_rng_state;
    s_rng_state.imbue(std::locale::classic());
    s_rng_state << rng;
    return s_rng_state.str();
}

std::string mt19937_seed_to_state(unsigned seed) {
    std::mt19937 rng(seed);
    return mt19937_get_state(rng);
}

std::string shuffle_samples(
        const std::string & rng_state,
        size_t            * shuffled_offs,
        size_t            * shuffled_begins,
        size_t            * shuffled_sizes,
        const size_t      * begins,
        const size_t      * sizes,
        size_t              count) {
    if (count == 0) return rng_state;

    std::mt19937 rng;
    mt19937_set_state(rng, rng_state);

    // sort indices by random value for each index
    std::vector<size_t> idcs;
    {
        std::vector<unsigned> rnd;
        idcs.resize(count);
        rnd.resize(count);
        for (unsigned i=0; i<count; ++i) {
            idcs[i] = i;
            rnd[i]  = rng();
        }

        std::sort(idcs.begin(), idcs.end(), [&rnd](size_t a, size_t b){
            // stable sort for reproducibility
            return (rnd[a] == rnd[b]) ? (a < b) : (rnd[a] < rnd[b]);
        });
    }

    // create random offsets
    for (unsigned i=0; i<count; ++i) {
        shuffled_offs[i] = (size_t) ((sizes[idcs[i]] - 1) * ((double) rng() / (double) (rng.max()-1)));
    }

    // reorder begins and sizes by sorted indices
    for (unsigned i=0; i<count; ++i) {
        shuffled_begins[i] = begins[idcs[i]];
    }

    for (unsigned i=0; i<count; ++i) {
        shuffled_sizes[i] = sizes[idcs[i]];
    }

    return mt19937_get_state(rng);
}

size_t hash_combine(size_t h1, size_t h2) {
    return h1 ^ (h2 << 1);
}

size_t compute_samples_hash(const char* fn, const size_t* samples_begin, const size_t* samples_size, size_t sample_count) {
    std::hash<std::string> h_string;
    std::hash<unsigned long long> h_ull;
    size_t h = h_string(std::string(fn));
    h = hash_combine(h, h_ull((unsigned long long) sample_count));
    for (size_t i=0; i< sample_count; ++i) {
        h = hash_combine(h, h_ull((unsigned long long) samples_begin[i]));
        h = hash_combine(h, h_ull((unsigned long long) samples_size[i]));
    }
    return h;
}

std::string replace_str(const char * s, const char * needle, const char * replacement) {
    std::string str = s;
    size_t pos = str.find(needle);
    if (pos != std::string::npos) {
        str.replace(pos, strlen(needle), replacement);
    }
    return str;
}

void print_duration(double fmillis) {
    if (fmillis < 1000.0f) {
        printf("%.1fms", (float) fmillis);
        return;
    }
    const int64_t one_sec  = 1000;
    const int64_t one_min  = one_sec  * 60;
    const int64_t one_hour = one_min  * 60;
    const int64_t one_day  = one_hour * 24;

    int64_t millis  = (int64_t) fmillis;
    int64_t days    = millis/one_day;
    int64_t hours   = (millis - days*one_day)/one_hour;
    int64_t minutes = (millis - days*one_day - hours*one_hour)/one_min;
    int64_t seconds = (millis - days*one_day - hours*one_hour - minutes*one_min)/one_sec;

    // to print int64_t either cast to (long long int) or use macro PRId64 from <inttypes.h>
    if (days > 0) {
        printf("%lldd ", (long long int) days);
    }
    printf("%02lld:%02lld:%02lld", (long long int) hours, (long long int) minutes, (long long int) seconds);
}

float cosine_decay(int64_t step, int64_t decay_steps, float minimum) {
    if (step > decay_steps) {
        step = decay_steps;
    }
    const float cosine_decay = 0.50f*(1.0f + cosf(3.14159265359f*step/decay_steps));
    const float decay = (1 - minimum)*cosine_decay + minimum;
    return decay;
}

float cosine_decay_restart(int64_t step, int64_t decay_steps, float minimum, float restart_step_mult) {
    while (step > decay_steps) {
        step -= decay_steps;
        decay_steps = (int64_t) (restart_step_mult * decay_steps);
    }
    return cosine_decay(step, decay_steps, minimum);
}

float learning_schedule(
    int64_t step,
    int64_t warmup_steps,
    int64_t cos_decay_steps,
    float   learning_rate,
    float   overall_minimum,
    float   cos_decay_minimum,
    float   cos_decay_restart_step_mult,
    bool    enable_restart) {

    float result =
        (step < warmup_steps)
            ? (float) step / (float) warmup_steps
            : enable_restart
                ? cosine_decay_restart(
                    step - warmup_steps,
                    cos_decay_steps,
                    cos_decay_minimum,
                    cos_decay_restart_step_mult)
                : cosine_decay(
                    step,
                    cos_decay_steps,
                    cos_decay_minimum);

    float min = overall_minimum / learning_rate;
    result = min + result * (1.0f - min);
    return result;
}

static bool are_same_layout(struct ggml_tensor * a, struct ggml_tensor * b) {
    GGML_ASSERT(a != NULL);
    GGML_ASSERT(b != NULL);
    GGML_ASSERT(a->type == b->type);
    GGML_ASSERT(ggml_are_same_shape(a, b));
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(b));

    return true;
}

void copy_tensor_by_name(struct ggml_tensor * dst, struct ggml_context * ctx, const char * name) {
    if (dst == NULL) {
        return;
    }
    struct ggml_tensor * t  = ggml_get_tensor(ctx, name);
    GGML_ASSERT(are_same_layout(dst, t));
    memcpy(dst->data, t->data, ggml_nbytes(t));

    if (strlen(ggml_get_name(dst)) == 0) {
        ggml_set_name(dst, name);
    }
}

// gguf constants
static const char * LLM_KV_OPTIMIZER_TYPE = "optimizer.type";
static const char * LLM_KV_OPTIMIZER_TYPE_ADAM  = "adam";
static const char * LLM_KV_OPTIMIZER_TYPE_LBFGS = "lbfgs";
static const char * LLM_KV_OPTIMIZER_FILE_VERSION               = "optimizer.file_version";
static const char * LLM_KV_OPTIMIZER_CONVERGENCE_PAST_COUNT     = "optimizer.convergence_past_count";
static const char * LLM_KV_OPTIMIZER_PARAMETER_COUNT            = "optimizer.parameter_count";
static const char * LLM_KV_OPTIMIZER_ITERATION_COUNT            = "optimizer.iteration_count";
static const char * LLM_KV_OPTIMIZER_JUST_INITIALIZED           = "optimizer.just_initialized";
static const char * LLM_KV_OPTIMIZER_ADAM_BEST_LOSS             = "optimizer.adam.best_loss";
static const char * LLM_KV_OPTIMIZER_ADAM_PREVIOUS_LOSS         = "optimizer.adam.previous_loss";
static const char * LLM_KV_OPTIMIZER_ADAM_NO_IMPROVEMENT_COUNT  = "optimizer.adam.no_improvement_count";
static const char * LLM_KV_OPTIMIZER_LBFGS_APPROX_HESSIAN_COUNT = "optimizer.lbfgs.approx_hessian_count";
static const char * LLM_KV_OPTIMIZER_LBFGS_BEST_LOSS            = "optimizer.lbfgs.best_loss";
static const char * LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_STEP     = "optimizer.lbfgs.line_search_step";
static const char * LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_J        = "optimizer.lbfgs.line_search_j";
static const char * LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_K        = "optimizer.lbfgs.line_search_k";
static const char * LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_END      = "optimizer.lbfgs.line_search_end";
static const char * LLM_KV_OPTIMIZER_LBFGS_NO_IMPROVEMENT_COUNT = "optimizer.lbfgs.no_improvement_count";

static const char * LLM_TENSOR_OPTIMIZER_ADAM_FIRST_MOMENTS    = "optimizer.adam.first_moments";
static const char * LLM_TENSOR_OPTIMIZER_ADAM_SECOND_MOMENTS   = "optimizer.adam.second_moments";
static const char * LLM_TENSOR_OPTIMIZER_ADAM_PAST_LOSS_VALUES = "optimizer.adam.past_loss_values";

static const char * LLM_TENSOR_OPTIMIZER_LBFGS_CURRENT_PARAMETERS  = "optimizer.lbfgs.current_parameters";
static const char * LLM_TENSOR_OPTIMIZER_LBFGS_PREVIOUS_PARAMETERS = "optimizer.lbfgs.previous_parameters";
static const char * LLM_TENSOR_OPTIMIZER_LBFGS_CURRENT_GRADIENTS   = "optimizer.lbfgs.current_gradients";
static const char * LLM_TENSOR_OPTIMIZER_LBFGS_PREVIOUS_GRADIENTS  = "optimizer.lbfgs.previous_gradients";
static const char * LLM_TENSOR_OPTIMIZER_LBFGS_SEARCH_DIRECTION    = "optimizer.lbfgs.search_direction";
static const char * LLM_TENSOR_OPTIMIZER_LBFGS_PAST_LOSS_VALUES    = "optimizer.lbfgs.past_loss_values";
static const char * LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_ALPHA        = "optimizer.lbfgs.memory_alpha";
static const char * LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_YS           = "optimizer.lbfgs.memory_ys";
static const char * LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_S            = "optimizer.lbfgs.memory_s";
static const char * LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_Y            = "optimizer.lbfgs.memory_y";

static const char * LLM_KV_TRAINING_FILE_VERSION         = "training.file_version";
static const char * LLM_KV_TRAINING_ITERATION_COUNT      = "training.iteration_count";
static const char * LLM_KV_TRAINING_SAMPLE_COUNT         = "training.sample_count";
static const char * LLM_KV_TRAINING_TOKEN_COUNT          = "training.token_count";
static const char * LLM_KV_TRAINING_EPOCH_COUNT          = "training.epoch_count";
static const char * LLM_KV_TRAINING_SHUFFLE_SAMPLES_HASH = "training.shuffle.samples_hash";
static const char * LLM_KV_TRAINING_SHUFFLE_RNG_STATE    = "training.shuffle.rng_state";
static const char * LLM_KV_TRAINING_SHUFFLE_SAMPLE_COUNT = "training.shuffle.sample_count";
static const char * LLM_KV_TRAINING_SHUFFLE_NEXT_SAMPLE  = "training.shuffle.next_sample";

#define GGUF_GET_KEY(ctx, dst, func, type, req, key) \
{ \
    const std::string skey(key); \
    const int kid = gguf_find_key(ctx, skey.c_str()); \
    if (kid >= 0) { \
        enum gguf_type ktype = gguf_get_kv_type(ctx, kid); \
        if (ktype != (type)) { \
            die_fmt("key %s has wrong type: %s", skey.c_str(), gguf_type_name(ktype)); \
        } \
        (dst) = func(ctx, kid); \
    } else if (req) { \
        die_fmt("key not found in model: %s", skey.c_str()); \
    } \
}

void load_opt_context_gguf(struct gguf_context * fctx, struct ggml_context * f_ggml_ctx, struct ggml_opt_context * opt) {
    // NOTE: gguf_context must be initialized with f_ggml_ctx and no_alloc=false, otherwise tensor data can not be read

    uint32_t file_version;
    GGUF_GET_KEY(fctx, file_version, gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_OPTIMIZER_FILE_VERSION);
    GGML_ASSERT(file_version == 0);

    GGUF_GET_KEY(fctx, opt->params.past, gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_OPTIMIZER_CONVERGENCE_PAST_COUNT);
    GGUF_GET_KEY(fctx, opt->iter, gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_OPTIMIZER_ITERATION_COUNT);
    GGUF_GET_KEY(fctx, opt->just_initialized, gguf_get_val_bool, GGUF_TYPE_BOOL, true, LLM_KV_OPTIMIZER_JUST_INITIALIZED);

    uint64_t nx;
    GGUF_GET_KEY(fctx, nx, gguf_get_val_u64, GGUF_TYPE_UINT64, true, LLM_KV_OPTIMIZER_PARAMETER_COUNT);
    opt->nx = (size_t) nx;

    // don't call ggml_opt_init until optimizer type and optimizer specific parameters are know

    std::string opt_type;
    GGUF_GET_KEY(fctx, opt_type, gguf_get_val_str, GGUF_TYPE_STRING, true, LLM_KV_OPTIMIZER_TYPE);
    if (opt_type == LLM_KV_OPTIMIZER_TYPE_ADAM) {
        opt->params.type = GGML_OPT_TYPE_ADAM;

        GGUF_GET_KEY(fctx, opt->adam.fx_best,          gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, LLM_KV_OPTIMIZER_ADAM_BEST_LOSS);
        GGUF_GET_KEY(fctx, opt->adam.fx_prev,          gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, LLM_KV_OPTIMIZER_ADAM_PREVIOUS_LOSS);
        GGUF_GET_KEY(fctx, opt->adam.n_no_improvement, gguf_get_val_u32, GGUF_TYPE_UINT32,  true, LLM_KV_OPTIMIZER_ADAM_NO_IMPROVEMENT_COUNT);

        ggml_opt_init(opt->ctx, opt, opt->params, opt->nx);

        copy_tensor_by_name(opt->adam.m,  f_ggml_ctx, LLM_TENSOR_OPTIMIZER_ADAM_FIRST_MOMENTS);
        copy_tensor_by_name(opt->adam.v,  f_ggml_ctx, LLM_TENSOR_OPTIMIZER_ADAM_SECOND_MOMENTS);
        copy_tensor_by_name(opt->adam.pf, f_ggml_ctx, LLM_TENSOR_OPTIMIZER_ADAM_PAST_LOSS_VALUES);
    } else if (opt_type == LLM_KV_OPTIMIZER_TYPE_LBFGS) {
        opt->params.type = GGML_OPT_TYPE_LBFGS;

        GGUF_GET_KEY(fctx, opt->params.lbfgs.m,         gguf_get_val_u32, GGUF_TYPE_UINT32,  true, LLM_KV_OPTIMIZER_LBFGS_APPROX_HESSIAN_COUNT);
        GGUF_GET_KEY(fctx, opt->lbfgs.fx_best,          gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, LLM_KV_OPTIMIZER_LBFGS_BEST_LOSS);
        GGUF_GET_KEY(fctx, opt->lbfgs.step,             gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_STEP);
        GGUF_GET_KEY(fctx, opt->lbfgs.j,                gguf_get_val_i32, GGUF_TYPE_INT32,   true, LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_J);
        GGUF_GET_KEY(fctx, opt->lbfgs.k,                gguf_get_val_i32, GGUF_TYPE_INT32,   true, LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_K);
        GGUF_GET_KEY(fctx, opt->lbfgs.end,              gguf_get_val_i32, GGUF_TYPE_INT32,   true, LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_END);
        GGUF_GET_KEY(fctx, opt->lbfgs.n_no_improvement, gguf_get_val_u32, GGUF_TYPE_UINT32,  true, LLM_KV_OPTIMIZER_LBFGS_NO_IMPROVEMENT_COUNT);

        ggml_opt_init(opt->ctx, opt, opt->params, opt->nx);

        copy_tensor_by_name(opt->lbfgs.x,    f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_CURRENT_PARAMETERS);
        copy_tensor_by_name(opt->lbfgs.xp,   f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_PREVIOUS_PARAMETERS);
        copy_tensor_by_name(opt->lbfgs.g,    f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_CURRENT_GRADIENTS);
        copy_tensor_by_name(opt->lbfgs.gp,   f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_PREVIOUS_GRADIENTS);
        copy_tensor_by_name(opt->lbfgs.d,    f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_SEARCH_DIRECTION);
        copy_tensor_by_name(opt->lbfgs.pf,   f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_PAST_LOSS_VALUES);
        copy_tensor_by_name(opt->lbfgs.lmal, f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_ALPHA);
        copy_tensor_by_name(opt->lbfgs.lmys, f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_YS);
        copy_tensor_by_name(opt->lbfgs.lms,  f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_S);
        copy_tensor_by_name(opt->lbfgs.lmy,  f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_Y);
    } else {
        die("unknown optimizer type\n");
    }
}

void save_opt_context_gguf(struct gguf_context * fctx, struct ggml_opt_context * opt) {
    gguf_set_val_u32(fctx, LLM_KV_OPTIMIZER_FILE_VERSION, 0);
    gguf_set_val_u32(fctx, LLM_KV_OPTIMIZER_CONVERGENCE_PAST_COUNT, opt->params.past);
    gguf_set_val_u64(fctx, LLM_KV_OPTIMIZER_PARAMETER_COUNT, (uint64_t) opt->nx);
    gguf_set_val_u32(fctx, LLM_KV_OPTIMIZER_ITERATION_COUNT, opt->iter);
    gguf_set_val_bool(fctx, LLM_KV_OPTIMIZER_JUST_INITIALIZED, opt->just_initialized);

    switch (opt->params.type) {
        case GGML_OPT_TYPE_ADAM:
            {
                gguf_set_val_str(fctx, LLM_KV_OPTIMIZER_TYPE, LLM_KV_OPTIMIZER_TYPE_ADAM);
                gguf_set_val_f32(fctx, LLM_KV_OPTIMIZER_ADAM_BEST_LOSS,            opt->adam.fx_best);
                gguf_set_val_f32(fctx, LLM_KV_OPTIMIZER_ADAM_PREVIOUS_LOSS,        opt->adam.fx_prev);
                gguf_set_val_u32(fctx, LLM_KV_OPTIMIZER_ADAM_NO_IMPROVEMENT_COUNT, opt->adam.n_no_improvement);

                ggml_set_name(opt->adam.m, LLM_TENSOR_OPTIMIZER_ADAM_FIRST_MOMENTS);
                ggml_set_name(opt->adam.v, LLM_TENSOR_OPTIMIZER_ADAM_SECOND_MOMENTS);
                if (opt->adam.pf) {
                    ggml_set_name(opt->adam.pf, LLM_TENSOR_OPTIMIZER_ADAM_PAST_LOSS_VALUES);
                }

                gguf_add_tensor(fctx, opt->adam.m);
                gguf_add_tensor(fctx, opt->adam.v);
                if (opt->adam.pf) {
                    gguf_add_tensor(fctx, opt->adam.pf);
                }
            } break;
        case GGML_OPT_TYPE_LBFGS:
            {
                gguf_set_val_str(fctx, LLM_KV_OPTIMIZER_TYPE, LLM_KV_OPTIMIZER_TYPE_LBFGS);
                gguf_set_val_u32(fctx, LLM_KV_OPTIMIZER_LBFGS_APPROX_HESSIAN_COUNT, opt->params.lbfgs.m);
                gguf_set_val_f32(fctx, LLM_KV_OPTIMIZER_LBFGS_BEST_LOSS,            opt->lbfgs.fx_best);
                gguf_set_val_f32(fctx, LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_STEP,     opt->lbfgs.step);
                gguf_set_val_i32(fctx, LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_J,        opt->lbfgs.j);
                gguf_set_val_i32(fctx, LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_K,        opt->lbfgs.k);
                gguf_set_val_i32(fctx, LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_END,      opt->lbfgs.end);
                gguf_set_val_u32(fctx, LLM_KV_OPTIMIZER_LBFGS_NO_IMPROVEMENT_COUNT, opt->lbfgs.n_no_improvement);

                ggml_set_name(opt->lbfgs.x,    LLM_TENSOR_OPTIMIZER_LBFGS_CURRENT_PARAMETERS);
                ggml_set_name(opt->lbfgs.xp,   LLM_TENSOR_OPTIMIZER_LBFGS_PREVIOUS_PARAMETERS);
                ggml_set_name(opt->lbfgs.g,    LLM_TENSOR_OPTIMIZER_LBFGS_CURRENT_GRADIENTS);
                ggml_set_name(opt->lbfgs.gp,   LLM_TENSOR_OPTIMIZER_LBFGS_PREVIOUS_GRADIENTS);
                ggml_set_name(opt->lbfgs.d,    LLM_TENSOR_OPTIMIZER_LBFGS_SEARCH_DIRECTION);
                if (opt->lbfgs.pf) {
                    ggml_set_name(opt->lbfgs.pf, LLM_TENSOR_OPTIMIZER_LBFGS_PAST_LOSS_VALUES);
                }
                ggml_set_name(opt->lbfgs.lmal, LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_ALPHA);
                ggml_set_name(opt->lbfgs.lmys, LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_YS);
                ggml_set_name(opt->lbfgs.lms,  LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_S);
                ggml_set_name(opt->lbfgs.lmy,  LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_Y);

                gguf_add_tensor(fctx, opt->lbfgs.x);
                gguf_add_tensor(fctx, opt->lbfgs.xp);
                gguf_add_tensor(fctx, opt->lbfgs.g);
                gguf_add_tensor(fctx, opt->lbfgs.gp);
                gguf_add_tensor(fctx, opt->lbfgs.d);
                if (opt->lbfgs.pf) {
                    gguf_add_tensor(fctx, opt->lbfgs.pf);
                }
                gguf_add_tensor(fctx, opt->lbfgs.lmal);
                gguf_add_tensor(fctx, opt->lbfgs.lmys);
                gguf_add_tensor(fctx, opt->lbfgs.lms);
                gguf_add_tensor(fctx, opt->lbfgs.lmy);
            } break;
    }
}

bool load_train_state_gguf(struct gguf_context * fctx, struct ggml_context * f_ggml_ctx, struct train_state * train) {
    if (gguf_find_key(fctx, LLM_KV_TRAINING_FILE_VERSION) < 0) {
        return false;
    }

    uint32_t file_version;
    GGUF_GET_KEY(fctx, file_version,         gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_FILE_VERSION);
    GGML_ASSERT(file_version <= 1);

    if (file_version == 0) {

        GGUF_GET_KEY(fctx, train->train_its,     gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_ITERATION_COUNT);
        GGUF_GET_KEY(fctx, train->train_samples, gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_SAMPLE_COUNT);
        GGUF_GET_KEY(fctx, train->train_tokens,  gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_TOKEN_COUNT);

    } else if (file_version == 1) {

        GGUF_GET_KEY(fctx, train->train_its,     gguf_get_val_u64, GGUF_TYPE_UINT64, true, LLM_KV_TRAINING_ITERATION_COUNT);
        GGUF_GET_KEY(fctx, train->train_samples, gguf_get_val_u64, GGUF_TYPE_UINT64, true, LLM_KV_TRAINING_SAMPLE_COUNT);
        GGUF_GET_KEY(fctx, train->train_tokens,  gguf_get_val_u64, GGUF_TYPE_UINT64, true, LLM_KV_TRAINING_TOKEN_COUNT);
        GGUF_GET_KEY(fctx, train->train_epochs,  gguf_get_val_u64, GGUF_TYPE_UINT64, true, LLM_KV_TRAINING_EPOCH_COUNT);

        GGUF_GET_KEY(fctx, train->shuffle_samples_hash,      gguf_get_val_u64, GGUF_TYPE_UINT64, false, LLM_KV_TRAINING_SHUFFLE_SAMPLES_HASH);
        GGUF_GET_KEY(fctx, train->shuffle_rng_state_current, gguf_get_val_str, GGUF_TYPE_STRING, false, LLM_KV_TRAINING_SHUFFLE_RNG_STATE);
        GGUF_GET_KEY(fctx, train->shuffle_sample_count,      gguf_get_val_u64, GGUF_TYPE_UINT64, false, LLM_KV_TRAINING_SHUFFLE_SAMPLE_COUNT);
        GGUF_GET_KEY(fctx, train->shuffle_next_sample,       gguf_get_val_u64, GGUF_TYPE_UINT64, false, LLM_KV_TRAINING_SHUFFLE_NEXT_SAMPLE);
    }

    load_opt_context_gguf(fctx, f_ggml_ctx, train->opt);
    return true;
}

void save_train_state_gguf(struct gguf_context * fctx, struct train_state * train) {
    gguf_set_val_u32(fctx, LLM_KV_TRAINING_FILE_VERSION,    1);
    gguf_set_val_u64(fctx, LLM_KV_TRAINING_ITERATION_COUNT, train->train_its);
    gguf_set_val_u64(fctx, LLM_KV_TRAINING_SAMPLE_COUNT,    train->train_samples);
    gguf_set_val_u64(fctx, LLM_KV_TRAINING_TOKEN_COUNT,     train->train_tokens);
    gguf_set_val_u64(fctx, LLM_KV_TRAINING_EPOCH_COUNT,     train->train_epochs);

    gguf_set_val_u64(fctx, LLM_KV_TRAINING_SHUFFLE_SAMPLES_HASH, (uint64_t) train->shuffle_samples_hash);
    gguf_set_val_str(fctx, LLM_KV_TRAINING_SHUFFLE_RNG_STATE,    train->shuffle_rng_state_current.c_str());
    gguf_set_val_u64(fctx, LLM_KV_TRAINING_SHUFFLE_SAMPLE_COUNT, (uint64_t) train->shuffle_sample_count);
    gguf_set_val_u64(fctx, LLM_KV_TRAINING_SHUFFLE_NEXT_SAMPLE,  (uint64_t) train->shuffle_next_sample);

    save_opt_context_gguf(fctx, train->opt);
}


struct llama_file {
    // use FILE * so we don't have to re-open the file to mmap
    FILE * fp;
    size_t size;

    llama_file(const char * fname, const char * mode) {
        fp = std::fopen(fname, mode);
        if (fp == NULL) {
            size = 0;
        } else {
            seek(0, SEEK_END);
            size = tell();
            seek(0, SEEK_SET);
        }
    }

    size_t tell() const {
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        GGML_ASSERT(ret != -1); // this really shouldn't fail
        return (size_t) ret;
    }

    void seek(size_t offset, int whence) {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        GGML_ASSERT(ret == 0); // same
    }

    void read_raw(void * ptr, size_t size) {
        if (size == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, size, 1, fp);
        if (ferror(fp)) {
            die_fmt("read error: %s", strerror(errno));
        }
        if (ret != 1) {
            die("unexpectedly reached end of file");
        }
    }

    std::uint32_t read_u32() {
        std::uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    std::string read_string(std::uint32_t len) {
        std::vector<char> chars(len);
        read_raw(chars.data(), len);
        return std::string(chars.data(), len);
    }

    void write_raw(const void * ptr, size_t size) {
        if (size == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, size, 1, fp);
        if (ret != 1) {
            die_fmt("write error: %s", strerror(errno));
        }
    }

    void write_u32(std::uint32_t val) {
        write_raw(&val, sizeof(val));
    }

    ~llama_file() {
        if (fp) {
            std::fclose(fp);
        }
    }
};

static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

// mark each byte with its utf8 unit number.
// returns the number of utf8 characters.
// e.g. when bytes == '\x61\xD0\xB0\x62',
// then utf8_units will become [0,0,1,0]
// utf8_nunits will become [1,2,2,1] and 3 is returned.
// bytes where utf8_units is zero, are the begin of an utf8 character.
static size_t mark_utf8_units(const char* bytes, int * utf8_units, int * utf8_nunits, size_t count) {
    size_t offs = 0;
    size_t count_utf8 = 0;
    while(offs < count) {
        int len = (int) utf8_len(bytes[offs]);
        for (int i=0; i<len; ++i) {
            utf8_units[offs+i]  = i;
            utf8_nunits[offs+i] = len;
        }
        offs += len;
        ++count_utf8;
    }
    return count_utf8;
}

size_t tokenize_file(
        struct llama_context     * lctx,
        const char               * filename,
        const std::string        & sample_start,
        bool                       include_sample_start,
        bool                       overlapping_samples,
        unsigned                   context_length,
        std::vector<llama_token> & out_tokens,
        std::vector<size_t>      & out_samples_begin,
        std::vector<size_t>      & out_samples_size) {
    struct llama_file f(filename, "rb");

    if (f.size == 0) {
        out_tokens.clear();
        out_samples_begin.clear();
        out_samples_size.clear();
        printf("%s: warning: empty or not existing training data file '%s'\n",
            __func__, filename);
        return out_tokens.size();
    }

    // account for possible leading whitespace that will be added by tokenizer
    // e.g. '\t' will be tokenized by llama spm tokenizer to [29871, 12]
    const int n_max_tokens_overhead = 1;

    std::vector<char> buf;
    buf.resize(f.size);

    f.read_raw(buf.data(), f.size);

    std::vector<int> utf8_units;
    std::vector<int> utf8_nunits;
    utf8_units.resize(buf.size());
    utf8_nunits.resize(buf.size());
    mark_utf8_units(buf.data(), utf8_units.data(), utf8_nunits.data(), buf.size());

    if (sample_start.size() == 0) {
        // tokenize all data at once
        out_tokens.resize(buf.size() + n_max_tokens_overhead);

        int n_tokens = llama_tokenize(
            llama_get_model(lctx),
            buf.data(),
            (int) buf.size(),
            out_tokens.data(),
            (int) out_tokens.size(),
            false, false);
        if (n_tokens < 0) {
            out_tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(
                llama_get_model(lctx),
                buf.data(),
                (int) buf.size(),
                out_tokens.data(),
                (int) out_tokens.size(),
                false, false);
        }
        if (n_tokens >= 0) {
            out_tokens.resize(n_tokens);
        }

        // generate sample starts at all token positions
        out_samples_begin.clear();
        out_samples_begin.push_back(0);
        out_samples_size.push_back(std::min((size_t) context_length, out_tokens.size()));
        size_t end = (out_tokens.size() >= context_length) ? (out_tokens.size() - context_length) : 0;
        for (size_t sample_begin = 1; sample_begin < end; ++sample_begin) {
            out_samples_begin.push_back(sample_begin);
            out_samples_size.push_back(context_length);
        }
    } else {
        // split data into samples and tokenize each sample
        std::string data_str(buf.data(), buf.size());
        out_samples_begin.clear();
        out_samples_size.clear();
        out_tokens.clear();

        // find all positions of pattern sample_start
        size_t sample_begin = data_str.find(sample_start, 0);
        while (sample_begin != std::string::npos) {
            out_samples_begin.push_back(sample_begin);
            const size_t search_start = sample_begin + sample_start.size();
            sample_begin = data_str.find(sample_start, search_start);
        }
        if (out_samples_begin.size() == 0) {
            printf("%s: warning: sample start pattern '%s' not found. inserting single sample at data begin\n",
                __func__, sample_start.c_str());
            out_samples_begin.push_back(0);
        }

        out_samples_size.resize(out_samples_begin.size(), 0);

        std::vector<char>        buf_sample;
        std::vector<llama_token> tok_sample;

        const size_t sample_begin_offset = (include_sample_start ? 0 : sample_start.size());
        size_t found_too_big_sample   = 0;
        size_t found_too_small_sample = 0;
        size_t found_empty_sample     = 0;
        size_t found_min_sample_size  = SIZE_MAX;
        size_t found_max_sample_size  = 0;

        size_t max_token_text_size = 0;
        int n_vocab = llama_n_vocab(llama_get_model(lctx));
        for (llama_token token=0; token < n_vocab; ++token) {
            max_token_text_size = std::max(
                max_token_text_size,
                strlen(llama_token_get_text(llama_get_model(lctx), token)));
        }

        // upper bound of context byte length.
        // strings with this byte length should always tokenize to at least context_length tokens.
        size_t context_byte_len = max_token_text_size*context_length;

        for (unsigned i=0; i<out_samples_begin.size(); ++i) {
            // determine sample begin and end from pattern positions
            size_t sample_begin = out_samples_begin[i] + sample_begin_offset;
            size_t sample_end   = overlapping_samples
                                    ? std::min(
                                        data_str.size(),
                                        sample_begin + context_byte_len)
                                    : (i+1 < out_samples_begin.size()
                                        ? out_samples_begin[i+1]
                                        : data_str.size());
            if (sample_end < utf8_units.size() && utf8_units[sample_end] > 0) {
                // sample end is in the middle of an utf8 character.
                // advance sample_end to the begin of the next utf8 character.
                sample_end += utf8_nunits[sample_end] - utf8_units[sample_end];
            }
            size_t sample_size = sample_end - sample_begin;
            if (sample_size == 0) {
                ++found_empty_sample;
            }

            if (sample_size > 0) {
                // llama_tokenize expects zero terminated string,
                // copy sample into buffer and zero terminate it.
                buf_sample.resize(sample_size);
                memcpy(buf_sample.data(), data_str.data() + sample_begin, sample_size);

                // printf("sample: '%s'\n", buf_sample.data());

                // tokenize the sample
                tok_sample.resize(buf_sample.size() + n_max_tokens_overhead);
                int n_tokens = llama_tokenize(llama_get_model(lctx),
                    buf_sample.data(),
                    (int) buf_sample.size(),
                    tok_sample.data(),
                    (int) tok_sample.size(),
                    false, false);
                if (n_tokens < 0) {
                    tok_sample.resize(-n_tokens);
                    n_tokens = llama_tokenize(llama_get_model(lctx),
                        buf_sample.data(),
                        (int) buf_sample.size(),
                        tok_sample.data(),
                        (int) tok_sample.size(),
                        false, false);
                    GGML_ASSERT(n_tokens >= 0);
                }
                GGML_ASSERT(n_tokens <= (int) tok_sample.size());

                if ((size_t) n_tokens > context_length) {
                    ++found_too_big_sample;
                } else if ((size_t) n_tokens < context_length) {
                    ++found_too_small_sample;
                }
                found_max_sample_size = std::max(found_max_sample_size, (size_t) n_tokens);
                found_min_sample_size = std::min(found_min_sample_size, (size_t) n_tokens);

                // write out tokens, start and size of sample
                // overwrite the string start position with the token start position
                out_samples_begin[i] = out_tokens.size();
                out_samples_size[i] = (size_t) n_tokens;
                out_tokens.insert(out_tokens.end(), tok_sample.begin(), tok_sample.begin() + n_tokens);
            } else {
                out_samples_begin[i] = out_tokens.size();
                out_samples_size[i] = 0;
            }

        }
        if (found_too_big_sample > 0) {
            printf("%s: warning: found %zu samples (max length %zu) that exceed context length of %u. samples will be cut off.\n",
                __func__, found_too_big_sample, found_max_sample_size, context_length);
        }

        if (found_too_small_sample > 0) {
            printf("%s: warning: found %zu samples (min length %zu) that are shorter than context length of %u.\n",
                __func__, found_too_small_sample, found_min_sample_size, context_length);
        }

        if (found_empty_sample) {
            printf("%s: warning: found %zu empty samples.\n",
                __func__, found_empty_sample);
        }
    }
    printf("%s: total number of samples: %zu\n",
        __func__, out_samples_begin.size());

    GGML_ASSERT(out_samples_begin.size() == out_samples_size.size());

    return out_tokens.size();
}

std::string get_train_filename(const char * filename, const char * pattern_it, const char * latest, int64_t iteration) {
    std::string sit = (iteration >= 0) ? std::to_string(iteration) : std::string(latest);
    return replace_str(filename, pattern_it, sit.c_str());
}

struct train_params_common get_default_train_params_common() {
    struct train_params_common params;
    params.fn_train_data     = "shakespeare.txt";
    params.fn_checkpoint_in  = "checkpoint.gguf";
    params.fn_checkpoint_out = "checkpoint-ITERATION.gguf";
    params.pattern_fn_it     = "ITERATION";
    params.fn_latest         = "LATEST";

    params.print_usage = false;

    params.save_every = 10;

    params.seed       =   -1;

    params.n_ctx      =  128;
    params.n_threads  =    6;
    params.n_batch    =    8;
    params.n_gradient_accumulation = 1;
    params.n_epochs   = -1;
    params.n_gpu_layers = 0;

    params.custom_n_ctx = false;

    params.use_flash              = false;
    params.use_checkpointing      = true;

    params.sample_start           = "";
    params.include_sample_start   = false;
    params.escape                 = false;
    params.overlapping_samples    = false;
    params.fill_with_next_samples = false;
    params.separate_with_eos      = false;
    params.separate_with_bos      = true;
    params.sample_random_offsets  = false;
    params.force_reshuffle        = false;

    params.opt_past               = 0;
    params.opt_delta              = 1e-5f;
    params.opt_max_no_improvement = 0;

    params.warmup            =  100;
    params.cos_decay_steps   = 1000;
    params.cos_decay_restart = 1.1f;
    params.cos_decay_min     = 0.1f;
    params.enable_restart    = false;

    params.adam_n_iter         = 256;
    params.adam_alpha          = 1e-3f;
    params.adam_min_alpha      = 0;
    params.adam_decay          = 1e-1f;
    params.adam_decay_min_ndim = 2;
    params.adam_beta1          = 0.9f;
    params.adam_beta2          = 0.999f;
    params.adam_gclip          = 1.0f;
    params.adam_eps_f          = 0.0f;

    return params;
}

void print_common_train_usage(int /*argc*/, char ** /*argv*/, const struct train_params_common * params) {
    // fprintf(stderr, "usage: %s [options]\n", argv[0]);
    // fprintf(stderr, "\n");
    // fprintf(stderr, "options:\n");
    // fprintf(stderr, "  -h, --help                 show this help message and exit\n");
    fprintf(stderr, "  --train-data FNAME         path from which to load training data (default '%s')\n", params->fn_train_data);
    fprintf(stderr, "  --checkpoint-in FNAME      path from which to load training checkpoint (default '%s')\n", params->fn_checkpoint_in);
    fprintf(stderr, "  --checkpoint-out FNAME     path to save training checkpoint (default '%s')\n", params->fn_checkpoint_out);
    fprintf(stderr, "  --pattern-fn-it STR        pattern in output filenames to be replaced by iteration number (default '%s')\n", params->pattern_fn_it);
    fprintf(stderr, "  --fn-latest STR            string to use instead of iteration number for saving latest output (default '%s')\n", params->fn_latest);
    fprintf(stderr, "  --save-every N             save checkpoint and lora every N iterations. Disabled when N <= 0. (default '%d')\n", params->save_every);
    fprintf(stderr, "  -s SEED, --seed SEED       RNG seed (default: -1, use random seed for -1)\n");
    fprintf(stderr, "  -c N, --ctx N              Context size used during training (default %d)\n", params->n_ctx);
    fprintf(stderr, "  -t N, --threads N          Number of threads (default %d)\n", params->n_threads);
    fprintf(stderr, "  -b N, --batch N            Parallel batch size (default %d)\n", params->n_batch);
    fprintf(stderr, "  --grad-acc N               Number of gradient accumulation steps (simulates larger batch size of batch*gradacc) (default %d)\n", params->n_gradient_accumulation);
    fprintf(stderr, "  --sample-start STR         Sets the starting point for samples after the specified pattern. If empty use every token position as sample start. (default '%s')\n", params->sample_start.c_str());
    fprintf(stderr, "  --include-sample-start     Include the sample start in the samples. (default off)\n");
    fprintf(stderr, "  --escape                   process sample start escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\)\n");
    fprintf(stderr, "  --overlapping-samples      Samples may overlap, will include sample-start of second and following samples. When off, samples will end at begin of next sample. (default off)\n");
    fprintf(stderr, "  --fill-with-next-samples   Samples shorter than context length will be followed by the next (shuffled) samples. (default off)\n");
    fprintf(stderr, "  --separate-with-eos        When fill-with-next-samples, insert end-of-sequence token between samples.%s\n", params->separate_with_eos ? " (default)" : "");
    fprintf(stderr, "  --separate-with-bos        When fill-with-next-samples, insert begin-of-sequence token between samples.%s\n", params->separate_with_bos ? " (default)" : "");
    fprintf(stderr, "  --no-separate-with-eos     When fill-with-next-samples, don't insert end-of-sequence token between samples.%s\n", !params->separate_with_eos ? " (default)" : "");
    fprintf(stderr, "  --no-separate-with-bos     When fill-with-next-samples, don't insert begin-of-sequence token between samples.%s\n", !params->separate_with_bos ? " (default)" : "");
    fprintf(stderr, "  --sample-random-offsets    Use samples beginning at random offsets. Together with fill-with-next-samples this may help for training endless text generation.%s\n", params->sample_random_offsets ? " (default)" : "");
    fprintf(stderr, "  --force-reshuffle          Force a reshuffling of data at program start, otherwise the shuffling of loaded checkpoint is resumed.\n");
    fprintf(stderr, "  --no-flash                 Don't use flash attention \n");
    fprintf(stderr, "  --use-flash                Use flash attention (default)\n");
    fprintf(stderr, "  --no-checkpointing         Don't use gradient checkpointing\n");
    fprintf(stderr, "  --use-checkpointing        Use gradient checkpointing (default)\n");
    fprintf(stderr, "  --warmup N                 Only for Adam optimizer. Number of warmup steps (default %d)\n", params->warmup);
    fprintf(stderr, "  --cos-decay-steps N        Only for Adam optimizer. Number of cosine decay steps (default %d)\n", params->cos_decay_steps);
    fprintf(stderr, "  --cos-decay-restart N      Only for Adam optimizer. Increase of cosine decay steps after restart (default %f)\n", params->cos_decay_restart);
    fprintf(stderr, "  --cos-decay-min N          Only for Adam optimizer. Cosine decay minimum (default %f)\n", params->cos_decay_min);
    fprintf(stderr, "  --enable-restart N         Only for Adam optimizer. Enable restarts of cos-decay %s\n", params->enable_restart ? "(default)" : "");
    fprintf(stderr, "  --disable-restart N        Only for Adam optimizer. Disable restarts of cos-decay %s\n", !params->enable_restart ? "(default)" : "");
    fprintf(stderr, "  --opt-past N               Number of optimization iterations to track for delta convergence test. Disabled when zero. (default %d)\n", params->opt_past);
    fprintf(stderr, "  --opt-delta N              Maximum delta for delta convergence test. Disabled when <= zero. (default %f)\n", params->opt_delta);
    fprintf(stderr, "  --opt-max-no-improvement N Maximum number of optimization iterations with no improvement. Disabled when <= zero. (default %d)\n", params->opt_max_no_improvement);
    fprintf(stderr, "  --epochs N                 Maximum number epochs to process. (default %d)\n", params->n_epochs);
    fprintf(stderr, "  --adam-iter N              Maximum number of Adam optimization iterations for each batch (default %d)\n", params->adam_n_iter);
    fprintf(stderr, "  --adam-alpha N             Adam learning rate alpha (default %f)\n", params->adam_alpha);
    fprintf(stderr, "  --adam-min-alpha N         Adam minimum learning rate alpha - including warmup phase (default %f)\n", params->adam_min_alpha);
    fprintf(stderr, "  --adam-decay N             AdamW weight decay. Values greater zero enable AdamW instead of regular Adam. (default %f)\n", params->adam_decay);
    fprintf(stderr, "  --adam-decay-min-ndim N    Minimum number of tensor dimensions to apply AdamW weight decay. Weight decay is not applied to tensors with less n_dims. (default %d)\n", params->adam_decay_min_ndim);
    fprintf(stderr, "  --adam-beta1 N             AdamW beta1 in interval [0,1). How much to smooth the first moment of gradients. (default %f)\n", params->adam_beta1);
    fprintf(stderr, "  --adam-beta2 N             AdamW beta2 in interval [0,1). How much to smooth the second moment of gradients. (default %f)\n", params->adam_beta2);
    fprintf(stderr, "  --adam-gclip N             AdamW gradient clipping. Disabled when zero. (default %f)\n", params->adam_gclip);
    fprintf(stderr, "  --adam-epsf N              AdamW epsilon for convergence test. Disabled when <= zero. (default %f)\n", params->adam_eps_f);
    fprintf(stderr, "  -ngl N, --n-gpu-layers N   Number of model layers to offload to GPU (default %d)", params->n_gpu_layers);
    fprintf(stderr, "\n");
}

bool consume_common_train_arg(
    int argc, char ** argv, int * idx, struct train_params_common * params, bool * invalid_param
) {
    int& i = *idx;
    std::string arg = argv[i];
    const std::string arg_prefix = "--";
    if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
        std::replace(arg.begin(), arg.end(), '_', '-');
    }
    if (arg == "--train-data") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->fn_train_data = argv[i];
    } else if (arg == "--checkpoint-in") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->fn_checkpoint_in = argv[i];
    } else if (arg == "--checkpoint-out") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->fn_checkpoint_out = argv[i];
    } else if (arg == "--pattern-fn-it") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->pattern_fn_it = argv[i];
    } else if (arg == "--fn-latest") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->fn_latest = argv[i];
    } else if (arg == "--save-every") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->save_every = std::stoi(argv[i]);
    } else if (arg == "-s" || arg == "--seed") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->seed = std::stoi(argv[i]);
    } else if (arg == "-c" || arg == "--ctx") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->n_ctx = std::stoi(argv[i]);
        params->custom_n_ctx = true;
    } else if (arg == "-t" || arg == "--threads") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->n_threads = std::stoi(argv[i]);
    } else if (arg == "-b" || arg == "--batch") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->n_batch = std::stoi(argv[i]);
    } else if (arg == "--grad-acc") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->n_gradient_accumulation = std::max(1, std::stoi(argv[i]));
    } else if (arg == "--sample-start") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->sample_start = std::string(argv[i]);
    } else if (arg == "--escape") {
        params->escape = true;
    } else if (arg == "--include-sample-start") {
        params->include_sample_start = true;
    } else if (arg == "--overlapping-samples") {
        params->overlapping_samples = true;
    } else if (arg == "--fill-with-next-samples") {
        params->fill_with_next_samples = true;
    } else if (arg == "--separate-with-eos") {
        params->separate_with_eos = true;
    } else if (arg == "--separate-with-bos") {
        params->separate_with_bos = true;
    } else if (arg == "--no-separate-with-eos") {
        params->separate_with_eos = false;
    } else if (arg == "--no-separate-with-bos") {
        params->separate_with_bos = false;
    } else if (arg == "--sample-random-offsets") {
        params->sample_random_offsets = true;
    } else if (arg == "--force-reshuffle") {
        params->force_reshuffle = true;
    } else if (arg == "--no-flash") {
        params->use_flash = false;
    } else if (arg == "--use-flash") {
        params->use_flash = true;
    } else if (arg == "--no-checkpointing") {
        params->use_checkpointing = false;
    } else if (arg == "--use-checkpointing") {
        params->use_checkpointing = true;
    } else if (arg == "--warmup") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->warmup = std::stoi(argv[i]);
    } else if (arg == "--cos-decay-steps") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->cos_decay_steps = std::stoi(argv[i]);
    } else if (arg == "--cos-decay-restart") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->cos_decay_restart = std::stof(argv[i]);
    } else if (arg == "--cos-decay-min") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->cos_decay_min = std::stof(argv[i]);
    } else if (arg == "--enable-restart") {
        params->enable_restart = true;
    } else if (arg == "--disable-restart") {
        params->enable_restart = false;
    } else if (arg == "--opt-past") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->opt_past = std::stoi(argv[i]);
    } else if (arg == "--opt-delta") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->opt_delta = std::stof(argv[i]);
    } else if (arg == "--opt-max-no-improvement") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->opt_max_no_improvement = std::stoi(argv[i]);
    } else if (arg == "--adam-epsf") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->adam_eps_f = std::stof(argv[i]);
    } else if (arg == "--epochs") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->n_epochs = std::stoi(argv[i]);
    } else if (arg == "--adam-iter") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->adam_n_iter = std::stoi(argv[i]);
    } else if (arg == "--adam-alpha") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->adam_alpha = std::stof(argv[i]);
    } else if (arg == "--adam-min-alpha") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->adam_min_alpha = std::stof(argv[i]);
    } else if (arg == "--adam-decay") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->adam_decay = std::stof(argv[i]);
    } else if (arg == "--adam-decay-min-ndim") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->adam_decay_min_ndim = std::stoi(argv[i]);
    } else if (arg == "--adam-beta1") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->adam_beta1 = std::stof(argv[i]);
    } else if (arg == "--adam-beta2") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->adam_beta2 = std::stof(argv[i]);
    } else if (arg == "--adam-gclip") {
        if (++i >= argc) {
            *invalid_param = true;
            return true;
        }
        params->adam_gclip = std::stof(argv[i]);
    } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            if (++i >= argc) {
                *invalid_param = true;
                return true;
            }
            if (llama_supports_gpu_offload()) {
                params->n_gpu_layers = std::stoi(argv[i]);
            } else {
                fprintf(stderr, "warning: not compiled with GPU offload support, --n-gpu-layers option will be ignored\n");
                fprintf(stderr, "warning: see main README.md for information on enabling GPU BLAS support\n");
            }
    } else if (arg == "-h" || arg == "--help") {
        params->print_usage = true;
        return true;
    } else {
        return false;
    }
    return true;
}

void finish_processing_train_args(struct train_params_common * params) {
    if (params->escape) {
        string_process_escapes(params->sample_start);
    }
}

void train_opt_callback(void * vdata, int accum_step, float * sched, bool * cancel) {
    struct train_opt_callback_data * data   = (struct train_opt_callback_data *) vdata;
    struct train_params_common     * params = data->params;
    struct train_state             * train  = data->train;
    struct ggml_opt_context        * opt    = train->opt;
    int n_batch = params->n_batch;
    int n_ctx = params->n_ctx;

    if (accum_step == 0) {
        // time measurement
        int64_t now = ggml_time_ms();
        if (now > data->last_time && opt->iter > data->first_iter) {
            double dt = (double) (now - data->last_time);
            if (data->millis_per_iter == 0.0) {
                data->millis_per_iter = dt;
            } else {
                const double gain = 0.7;
                data->millis_per_iter = data->millis_per_iter*(1.0-gain) + dt*gain;
            }
        }

        double remaining_millis = 0.0;
        if (data->millis_per_iter > 0.0) {
            const int n_iter = params->adam_n_iter;
            const int done_iter = opt->iter - data->first_iter;
            const int remaining_iter = n_iter - done_iter;
            remaining_millis = remaining_iter * data->millis_per_iter;
        }

        // file saving
        const bool save_now = (params->save_every > 0) && (opt->iter - data->last_save_iter >= params->save_every);
        if (save_now) {
            int new_iters = opt->iter - data->last_save_iter;
            train->train_its    += new_iters;
            train->train_tokens += new_iters * opt->params.n_gradient_accumulation * n_batch * n_ctx;

            if (data->save_cb) {
                data->save_cb(data->save_data, train);
            }

            data->last_save_iter = opt->iter;
        }

        // exclude file saving from time measurement, by measuring last_time after saving
        data->last_time = ggml_time_ms();

        *sched = learning_schedule(
            opt->iter,
            params->warmup,
            params->cos_decay_steps,
            params->adam_alpha,
            params->adam_min_alpha,
            params->cos_decay_min,
            params->cos_decay_restart,
            params->enable_restart);

        int impr_plot = -(int)(1 + (opt->loss_before - opt->loss_after) * 10.0f + 0.5f);
        if (impr_plot > 0) impr_plot = 0;
        if (std::isnan(opt->loss_before) || std::isnan(opt->loss_after)) impr_plot = 0;
        printf("%s: iter=%6d sample=%zu/%zu sched=%f loss=%f",
            __func__, opt->iter, std::min(1+train->shuffle_next_sample, train->shuffle_sample_count), train->shuffle_sample_count,
            *sched, opt->loss_after);


        if (data->millis_per_iter > 0) {
            printf(" dt=");
            print_duration(data->millis_per_iter);
            printf(" eta=");
            print_duration(remaining_millis);
        }

        float improvement = opt->loss_before - opt->loss_after;
        const float plot_scale = 10.0f;
        int bar_len = (int)(1 + improvement*plot_scale + 0.5);
        printf(" |");
        for (int i=0; i<bar_len; ++i) {
            printf("-");
        }
        printf(">");
        printf("\n");
    }

    int64_t used_samples = get_example_targets_batch(
        data->lctx,
        data->tokens_input,
        data->target_probs,
        train->shuffle_next_sample,
        data->shuffled_samples_offs,
        data->shuffled_samples_begin,
        data->shuffled_samples_size,
        data->samples_count,
        data->tokens_data,
        data->tokens_size,
        params->separate_with_eos,
        params->separate_with_bos,
        params->fill_with_next_samples,
        params->sample_random_offsets);

    train->train_samples += used_samples;
    train->shuffle_next_sample += used_samples;

    if (train->shuffle_next_sample >= train->shuffle_sample_count) {
        ++train->train_epochs;
        printf("%s: reshuffle samples. completed epochs: %llu\n", __func__, (long long unsigned) train->train_epochs);
        // note: we may have used some samples from the current shuffling more than once
        train->shuffle_rng_state_current = train->shuffle_rng_state_next;
        train->shuffle_rng_state_next = shuffle_samples(
            train->shuffle_rng_state_current,
            data->shuffled_samples_offs,
            data->shuffled_samples_begin,
            data->shuffled_samples_size,
            data->samples_begin,
            data->samples_size,
            data->samples_count);
        train->shuffle_next_sample = 0;
    }

    const bool last_epoch_reached = (params->n_epochs > 0 && (int64_t) train->train_epochs - data->first_epoch >= params->n_epochs);
    if (last_epoch_reached) {
        // allow optimization iteration at last epoch to be completed before canceling
        if (data->iter_at_last_epoch < 0) {
            data->iter_at_last_epoch = opt->iter;
        } else if (opt->iter > data->iter_at_last_epoch) {
            *cancel = true;
        }
    }
}
