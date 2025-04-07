#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <unistd.h>
#include <sampling.h>
#include "llama.h"
#include "common.h"

/**
 * Logging utils
 */
#define TAG "llama-android.cpp"
#define LOGv(...) __android_log_print(ANDROID_LOG_VERBOSE,  TAG, __VA_ARGS__)
#define LOGd(...) __android_log_print(ANDROID_LOG_DEBUG,    TAG, __VA_ARGS__)
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO,     TAG, __VA_ARGS__)
#define LOGw(...) __android_log_print(ANDROID_LOG_WARN,     TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR,    TAG, __VA_ARGS__)

/**
 * LLama resources: context, model, batch and sampler
 */
constexpr int   N_THREADS_MIN       = 1;
constexpr int   N_THREADS_MAX       = 8;
constexpr int   N_THREADS_HEADROOM  = 2;

constexpr int   CONTEXT_SIZE = 4096;
constexpr int   BATCH_SIZE   = 512;
constexpr float SAMPLER_TEMP = 0.3f;

static llama_model     * g_model;
static llama_context   * g_context;
static llama_batch     * g_batch;
static common_sampler  * g_sampler;

static void log_callback(ggml_log_level level, const char *fmt, void *data) {
    int priority;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR:
            priority = ANDROID_LOG_ERROR;
            break;
        case GGML_LOG_LEVEL_WARN:
            priority = GGML_LOG_LEVEL_WARN;
            break;
        case GGML_LOG_LEVEL_INFO:
            priority = GGML_LOG_LEVEL_INFO;
            break;
        case GGML_LOG_LEVEL_DEBUG:
            priority = GGML_LOG_LEVEL_DEBUG;
            break;
        default:
            priority = ANDROID_LOG_DEFAULT;
            break;
    }
    __android_log_print(priority, TAG, fmt, data);
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv* env;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }

    // Set llama log handler to Android
    llama_log_set(log_callback, nullptr);

    // Initialize backends
    llama_backend_init();
    LOGi("Backend initiated.");

    return JNI_VERSION_1_6;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_systemInfo(JNIEnv *env, jobject /*unused*/) {
    return env->NewStringUTF(llama_print_system_info());
}

extern "C"
JNIEXPORT jint JNICALL
Java_android_llama_cpp_LLamaAndroid_loadModel(JNIEnv *env, jobject, jstring filename) {
    llama_model_params model_params = llama_model_default_params();

    const auto *path_to_model = env->GetStringUTFChars(filename, 0);
    LOGd("Loading model from: %s", path_to_model);

    auto *model = llama_model_load_from_file(path_to_model, model_params);
    env->ReleaseStringUTFChars(filename, path_to_model);
    if (!model) {
        LOGe("load_model() failed");
        return -1;
    }
    g_model = model;
    return 0;
}

static int init_context(llama_model *model) {
    if (!model) {
        LOGe("init_context(): model cannot be null");
        return -1;
    }

    // Multi-threading setup
    int n_threads = std::max(N_THREADS_MIN, std::min(N_THREADS_MAX,
                                                     (int) sysconf(_SC_NPROCESSORS_ONLN) -
                                                     N_THREADS_HEADROOM));
    LOGi("Using %d threads", n_threads);

    // Context parameters setup
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = CONTEXT_SIZE;
    ctx_params.n_batch = BATCH_SIZE;
    ctx_params.n_ubatch = BATCH_SIZE;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    auto *context = llama_init_from_model(g_model, ctx_params);
    if (!context) {
        LOGe("llama_new_context_with_model() returned null)");
        return -2;
    }
    g_context = context;
    return 0;
}

static void new_batch(int n_tokens, bool embd = false, int n_seq_max = 1) {
    // Source: Copy of llama.cpp:llama_batch_init but heap-allocated.
    auto *batch = new llama_batch{
            0,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
    };

    if (embd) {
        batch->embd = (float *) malloc(sizeof(float) * n_tokens * embd);
    } else {
        batch->token = (llama_token *) malloc(sizeof(llama_token) * n_tokens);
    }

    batch->pos = (llama_pos *) malloc(sizeof(llama_pos) * n_tokens);
    batch->n_seq_id = (int32_t *) malloc(sizeof(int32_t) * n_tokens);
    batch->seq_id = (llama_seq_id **) malloc(sizeof(llama_seq_id *) * n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        batch->seq_id[i] = (llama_seq_id *) malloc(sizeof(llama_seq_id) * n_seq_max);
    }
    batch->logits = (int8_t *) malloc(sizeof(int8_t) * n_tokens);
    g_batch = batch;
}

void new_sampler(float temp) {
    common_params_sampling sparams;
    sparams.temp = temp;
    g_sampler = common_sampler_init(g_model, sparams);
}

extern "C"
JNIEXPORT jint JNICALL
Java_android_llama_cpp_LLamaAndroid_initContext(JNIEnv * /*env*/, jobject /*unused*/) {
    int ret = init_context(g_model);
    if (ret != 0) { return ret; }
    new_batch(BATCH_SIZE);
    new_sampler(SAMPLER_TEMP);
    return 0;
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_cleanUp(JNIEnv * /*unused*/, jobject /*unused*/) {
    llama_model_free(g_model);
    llama_free(g_context);
    delete g_batch;
    common_sampler_free(g_sampler);
    llama_backend_free();
}

static std::string get_backend() {
    std::vector<std::string> backends;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto *      reg  = ggml_backend_reg_get(i);
        std::string name = ggml_backend_reg_name(reg);
        if (name != "CPU") {
            backends.push_back(ggml_backend_reg_name(reg));
        }
    }
    return backends.empty() ? "CPU" : join(backends, ",");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_benchModel(JNIEnv *env, jobject /*unused*/, jint pp, jint tg, jint pl, jint nr) {
    auto pp_avg = 0.0;
    auto tg_avg = 0.0;
    auto pp_std = 0.0;
    auto tg_std = 0.0;

    const uint32_t n_ctx = llama_n_ctx(g_context);

    LOGi("n_ctx = %d", n_ctx);

    int i, j;
    int nri;
    for (nri = 0; nri < nr; nri++) {
        LOGi("Benchmark prompt processing (pp)");

        common_batch_clear(*g_batch);

        const int n_tokens = pp;
        for (i = 0; i < n_tokens; i++) {
            common_batch_add(*g_batch, 0, i, {0}, false);
        }

        g_batch->logits[g_batch->n_tokens - 1] = true;
        llama_memory_clear(llama_get_memory(g_context), false);

        const auto t_pp_start = ggml_time_us();
        if (llama_decode(g_context, *g_batch) != 0) {
            LOGe("llama_decode() failed during prompt processing");
        }
        const auto t_pp_end = ggml_time_us();

        // bench text generation

        LOGi("Benchmark text generation (tg)");

        llama_memory_clear(llama_get_memory(g_context), false);
        const auto t_tg_start = ggml_time_us();
        for (i = 0; i < tg; i++) {
            common_batch_clear(*g_batch);
            for (j = 0; j < pl; j++) {
                common_batch_add(*g_batch, 0, i, {j}, true);
            }

            if (llama_decode(g_context, *g_batch) != 0) {
                LOGe("llama_decode() failed during text generation");
            }
        }
        const auto t_tg_end = ggml_time_us();

        llama_memory_clear(llama_get_memory(g_context), false);

        const auto t_pp = double(t_pp_end - t_pp_start) / 1000000.0;
        const auto t_tg = double(t_tg_end - t_tg_start) / 1000000.0;

        const auto speed_pp = double(pp) / t_pp;
        const auto speed_tg = double(pl * tg) / t_tg;

        pp_avg += speed_pp;
        tg_avg += speed_tg;

        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;

        LOGi("pp %f t/s, tg %f t/s", speed_pp, speed_tg);
    }

    pp_avg /= double(nr);
    tg_avg /= double(nr);

    if (nr > 1) {
        pp_std = sqrt(pp_std / double(nr - 1) - pp_avg * pp_avg * double(nr) / double(nr - 1));
        tg_std = sqrt(tg_std / double(nr - 1) - tg_avg * tg_avg * double(nr) / double(nr - 1));
    } else {
        pp_std = 0;
        tg_std = 0;
    }

    char model_desc[128];
    llama_model_desc(g_model, model_desc, sizeof(model_desc));

    const auto model_size = double(llama_model_size(g_model)) / 1024.0 / 1024.0 / 1024.0;
    const auto model_n_params = double(llama_model_n_params(g_model)) / 1e9;

    const auto *const backend = "(Android)"; // TODO: What should this be?

    std::stringstream result;
    result << std::setprecision(2);
    result << "| model | size | params | backend | test | t/s |\n";
    result << "| --- | --- | --- | --- | --- | --- |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | "
           << backend << " | pp " << pp << " | " << pp_avg << " ± " << pp_std << " |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | "
           << backend << " | tg " << tg << " | " << tg_avg << " ± " << tg_std << " |\n";

    return env->NewStringUTF(result.str().c_str());
}


/**
 * Prediction loop's long-term and short-term states
 */
static llama_pos current_position;

static llama_pos token_predict_budget;
static std::string cached_token_chars;

static int decode_tokens_in_batches(
        llama_context *context,
        const llama_tokens& tokens,
        const llama_pos start_pos,
        bool compute_last_logit = false,
        llama_batch *batch = g_batch) {
    // Process tokens in batches using the global batch
    LOGd("Decode %d tokens starting at position %d", tokens.size(), start_pos);
    for (int i = 0; i < (int) tokens.size(); i += BATCH_SIZE) {
        int cur_batch_size = std::min((int) tokens.size() - i, BATCH_SIZE);
        common_batch_clear(*batch);
        LOGv("Preparing a batch size of %d starting at: %d", cur_batch_size, i);

        // Add tokens to the batch with proper positions
        for (int j = 0; j < cur_batch_size; j++) {
            llama_token token_id = tokens[i + j];
            llama_pos position = start_pos + i + j;
            bool want_logit = compute_last_logit && (i + j == tokens.size() - 1);
            common_batch_add(*batch, token_id, position, {0}, want_logit);
        }

        // Decode this batch
        int decode_result = llama_decode(context, *batch);
        if (decode_result) {
            LOGe("llama_decode failed w/ %d", decode_result);
            return -1;
        }
    }

    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_android_llama_cpp_LLamaAndroid_processSystemPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jsystem_prompt
) {
    // Reset long-term states and reset KV cache
    current_position = 0;
    llama_memory_clear(llama_get_memory(g_context), false);

    // Reset short-term states
    token_predict_budget = 0;
    cached_token_chars.clear();

    // Obtain and tokenize system prompt
    const auto *const system_text = env->GetStringUTFChars(jsystem_prompt, nullptr);
    LOGd("System prompt received: \n%s", system_text);
    const auto system_tokens = common_tokenize(g_context, system_text, true, true);
    env->ReleaseStringUTFChars(jsystem_prompt, system_text);

    // Print each token in verbose mode
    for (auto id : system_tokens) {
        LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
    }

    // Decode system tokens in batches
    if (decode_tokens_in_batches(g_context, system_tokens, current_position)) {
        LOGe("llama_decode() failed!");
        return -1;
    }

    // Update position
    current_position = (int) system_tokens.size();
    return 0;
}

// TODO-hyin: support KV cache backtracking
extern "C"
JNIEXPORT jint JNICALL
Java_android_llama_cpp_LLamaAndroid_processUserPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring juser_prompt,
        jint n_predict
) {
    // Reset short-term states
    token_predict_budget = 0;
    cached_token_chars.clear();

    // Obtain and tokenize user prompt
    const auto *const user_text = env->GetStringUTFChars(juser_prompt, nullptr);
    LOGd("User prompt received: \n%s", user_text);
    const auto user_tokens = common_tokenize(g_context, user_text, true, true);
    env->ReleaseStringUTFChars(juser_prompt, user_text);

    // Print each token in verbose mode
    for (auto id : user_tokens) {
        LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
    }

    // Check if context space is enough for desired tokens
    int desired_budget = current_position + (int) user_tokens.size() + n_predict;
    if (desired_budget > llama_n_ctx(g_context)) {
        LOGe("error: total tokens exceed context size");
        return -1;
    }
    token_predict_budget = desired_budget;

    // Decode user tokens in batches
    if (decode_tokens_in_batches(g_context, user_tokens, current_position, true)) {
        LOGe("llama_decode() failed!");
        return -2;
    }

    // Update position
    current_position += (int) user_tokens.size(); // Update position
    return 0;
}

static bool is_valid_utf8(const char *string) {
    if (!string) { return true; }

    const auto *bytes = (const unsigned char *) string;
    int num;

    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }
    return true;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_predictLoop(
        JNIEnv *env,
        jobject /*unused*/
) {
    // Stop if running out of token budget
    if (current_position >= token_predict_budget) {
        LOGw("STOP: current position (%d) exceeds budget (%d)", current_position, token_predict_budget);
        return nullptr;
    }

    // Sample next token
    const auto new_token_id = common_sampler_sample(g_sampler, g_context, -1);
    common_sampler_accept(g_sampler, new_token_id, true);

    // Populate the batch with new token, then decode
    common_batch_clear(*g_batch);
    common_batch_add(*g_batch, new_token_id, current_position, {0}, true);
    if (llama_decode(g_context, *g_batch) != 0) {
        LOGe("llama_decode() failed for generated token");
        return nullptr;
    }

    // Update position
    current_position++;

    // Stop if next token is EOG
    if (llama_vocab_is_eog(llama_model_get_vocab(g_model), new_token_id)) {
        LOGd("id: %d,\tIS EOG!\nSTOP.", new_token_id);
        return nullptr;
    }

    // If not EOG, convert to text
    auto new_token_chars = common_token_to_piece(g_context, new_token_id);
    cached_token_chars += new_token_chars;

    // Create and return Java string
    jstring result = nullptr;
    if (is_valid_utf8(cached_token_chars.c_str())) {
        result = env->NewStringUTF(cached_token_chars.c_str());
        LOGv("id: %d,\tcached: `%s`,\tnew: `%s`", new_token_id, cached_token_chars.c_str(), new_token_chars.c_str());
        cached_token_chars.clear();
    } else {
        LOGv("id: %d,\tappend to cache", new_token_id);
        result = env->NewStringUTF("");
    }
    return result;
}
