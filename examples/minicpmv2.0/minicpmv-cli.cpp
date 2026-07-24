#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "clip.h"
#include "minicpmv.h"
#include "llama.h"
#include "ggml.h"

#include "base64.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

#undef LOG_TEE
#define LOG_TEE(...) fprintf(stderr, __VA_ARGS__) // convert to normal fprintf

struct llava_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG_TEE("\n example usage: %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    LOG_TEE("  note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past) {
    int N = (int) tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        // if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0))) {
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval))) {
            LOG_TEE("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context * ctx_llama, int id, int * n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

static bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, bool add_bos){
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = common_tokenize(ctx_llama, str2, add_bos, true);
    eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
    return true;
}

static void process_image(struct llava_context * ctx_llava, struct llava_image_embed * image_embed, std::vector<std::vector<struct llava_image_embed *>> image_embed_slices, common_params * params, int &n_past) {
    std::string system_prompt;
    system_prompt = "<User>";

    eval_string(ctx_llava->ctx_llama, (system_prompt+"<image>").c_str(), params->n_batch, &n_past, true);
    llava_eval_image_embed(ctx_llava->ctx_llama, image_embed, params->n_batch, &n_past);
    if (!image_embed_slices.empty()) {
        eval_string(ctx_llava->ctx_llama, std::string("</image><slice>").c_str(), params->n_batch, &n_past, false);
        for (int i = 0; i < image_embed_slices.size(); ++i) {
            eval_string(ctx_llava->ctx_llama, std::string("<image>").c_str(), params->n_batch, &n_past, false);
            for (int j = 0; j < image_embed_slices[i].size(); ++j) {
                llava_eval_image_embed(ctx_llava->ctx_llama, image_embed_slices[i][j], params->n_batch, &n_past);
                if (j != image_embed_slices[i].size() - 1) {
                    eval_string(ctx_llava->ctx_llama, std::string("</image><image>").c_str(), params->n_batch, &n_past, false);
                } else {
                    if (i != image_embed_slices.size() - 1) {
                        eval_string(ctx_llava->ctx_llama, std::string("</image>\n").c_str(), params->n_batch, &n_past, false);
                    } else {
                        eval_string(ctx_llava->ctx_llama, std::string("</image>").c_str(), params->n_batch, &n_past, false);
                    }
                }
            }
        }
        eval_string(ctx_llava->ctx_llama, std::string("</slice>\n").c_str(), params->n_batch, &n_past, false);

    } else {
        eval_string(ctx_llava->ctx_llama, std::string("</image>\n").c_str(), params->n_batch, &n_past, false);
    }
}

static const char * sample(struct common_sampler * smpl,
                           struct llama_context * ctx_llama,
                           int * n_past) {
    const llama_token id = common_sampler_sample(smpl, ctx_llama, -1);
    common_sampler_accept(smpl, id, true);

    const llama_model * model = llama_get_model(ctx_llama);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    static std::string ret;
    if (llama_token_is_eog(vocab, id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past);
    return ret.c_str();
}

static const char* IMG_BASE64_TAG_BEGIN = "<img src=\"data:image/jpeg;base64,";
static const char* IMG_BASE64_TAG_END = "\">";

static void find_image_tag_in_prompt(const std::string& prompt, size_t& begin_out, size_t& end_out) {
    begin_out = prompt.find(IMG_BASE64_TAG_BEGIN);
    end_out = prompt.find(IMG_BASE64_TAG_END, (begin_out == std::string::npos) ? 0UL : begin_out);
}

static bool prompt_contains_image(const std::string& prompt) {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    return (begin != std::string::npos);
}

// replaces the base64 image tag in the prompt with `replacement`
static llava_image_embed * llava_image_embed_make_with_prompt_base64(struct clip_ctx * ctx_clip, int n_threads, const std::string& prompt) {
    size_t img_base64_str_start, img_base64_str_end;
    find_image_tag_in_prompt(prompt, img_base64_str_start, img_base64_str_end);
    if (img_base64_str_start == std::string::npos || img_base64_str_end == std::string::npos) {
        LOG_TEE("%s: invalid base64 image tag. must be %s<base64 byte string>%s\n", __func__, IMG_BASE64_TAG_BEGIN, IMG_BASE64_TAG_END);
        return NULL;
    }

    auto base64_bytes_start = img_base64_str_start + strlen(IMG_BASE64_TAG_BEGIN);
    auto base64_bytes_count = img_base64_str_end - base64_bytes_start;
    auto base64_str = prompt.substr(base64_bytes_start, base64_bytes_count );

    auto required_bytes = base64::required_encode_size(base64_str.size());
    auto img_bytes = std::vector<unsigned char>(required_bytes);
    base64::decode(base64_str.begin(), base64_str.end(), img_bytes.begin());

    auto embed = llava_image_embed_make_with_bytes(ctx_clip, n_threads, img_bytes.data(), img_bytes.size());
    if (!embed) {
        LOG_TEE("%s: could not load image from base64 string.\n", __func__);
        return NULL;
    }

    return embed;
}

static std::string remove_image_from_prompt(const std::string& prompt, const char * replacement = "") {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    if (begin == std::string::npos || end == std::string::npos) {
        return prompt;
    }
    auto pre = prompt.substr(0, begin);
    auto post = prompt.substr(end + strlen(IMG_BASE64_TAG_END));
    return pre + replacement + post;
}

static void process_user_input(struct llava_context * ctx_llava, common_params * params, std::string prompt, int &n_past, bool is_first = false) {
    std::string user_prompt = prompt + "<AI>";
    if (!is_first) user_prompt = "<User>" + prompt;
    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;

    eval_string(ctx_llava->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, false);

    LOG_TEE("\n");

    // generate the response

    struct common_sampler * smpl = common_sampler_init(ctx_llava->model, params->sampling);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        exit(1);
    }
    // struct llama_sampling_context * ctx_sampling = llama_sampling_init(params->sparams);
    std::string response = "";
    for (int i = 0; i < max_tgt_len; i++) {
        const char * tmp = sample(smpl, ctx_llava->ctx_llama, &n_past);
        response += tmp;
        if (strcmp(tmp, "</s>") == 0) break;
        if (strstr(tmp, "###")) break; // Yi-VL behavior
        printf("%s", tmp);
        if (strstr(response.c_str(), "<User>")) break; // minicpm-v 2.0 - for some reason those decode not as the correct token (tokenizer works)

        fflush(stdout);
    }

    common_sampler_free(smpl);
    // printf("\n");
    LOG("\n");
}

static struct llava_context * llava_init(common_params * params) {
    const char * clip_path = params->mmproj.c_str();

    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }

    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);

    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = common_model_params_to_llama(*params);

    llama_model * model = llama_load_model_from_file(params->model.c_str(), model_params);
    if (model == NULL) {
        LOG_TEE("%s: error: unable to load model\n" , __func__);
        return NULL;
    }

    // below in func 'llava_init_context' in the lastest code
    llama_context_params ctx_params = common_context_params_to_llama(*params);
    
    // ctx_params.n_ctx           = params->n_ctx < 2048 ? 2048 : params->n_ctx; // we need a longer context size to process image embeddings
    if (params->n_ctx < 2048) {
        // warn user here, "Image processing requires at least 2048 context, setting context to 2048"
        LOG_WRN("%s: Image processing requires at least 2048 context, setting context to 2048\n" , __func__);
        ctx_params.n_ctx = 2048;
    } else {
        ctx_params.n_ctx = params->n_ctx;
    }

    // llama_context * ctx_llama = llama_new_context_with_model(model, ctx_params);
    llama_context * ctx_llama = llama_init_from_model(model, ctx_params);

    if (ctx_llama == NULL) {
        LOG_TEE("%s: error: failed to create the llama_context\n" , __func__);
        return NULL;
    }

    auto ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));

    ctx_llava->ctx_llama = ctx_llama;
    ctx_llava->ctx_clip = ctx_clip;
    ctx_llava->model = model;
    return ctx_llava;
}

static void llava_free(struct llava_context * ctx_llava) {
    if (ctx_llava->ctx_clip) {
        clip_free(ctx_llava->ctx_clip);
        ctx_llava->ctx_clip = NULL;
    }

    llama_free(ctx_llava->ctx_llama);
    llama_free_model(ctx_llava->model);
    llama_backend_free();
}

static void llama_log_callback_logTee(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    LOG_TEE("%s", text);
}

int main(int argc, char ** argv) {
    ggml_time_init();

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LLAVA, show_additional_info)) {
        return 1;
    }

    common_init();

// #ifndef LOG_DISABLE_LOGS
//     log_set_target(log_filename_generator("llava", "log"));
//     LOG_TEE("Log start\n");
//     log_dump_cmdline(argc, argv);
//     llama_log_set(llama_log_callback_logTee, nullptr);
// #endif // LOG_DISABLE_LOGS

    if (params.mmproj.empty() || (params.image.empty() && !prompt_contains_image(params.prompt))) {
        // gpt_print_usage(argc, argv, params);
        show_additional_info(argc, argv);
        return 1;
    }

    auto ctx_llava = llava_init(&params);
    if (ctx_llava == NULL) {
        LOG_TEE("%s: error: failed to init llava\n", __func__);
        return 1;
    }

    auto image_embed_and_slices = llava_image_embed_make_with_filename_slice(ctx_llava->ctx_clip, params.cpuparams.n_threads, params.image[0].c_str());
    auto image_embed = image_embed_and_slices.first;
    auto image_embed_slices = image_embed_and_slices.second;

    // process the prompt
    if (params.prompt.empty() && params.interactive == false) {
        LOG_TEE("prompt should be given or interactive mode should be on");
        return 1;
    }

    int n_past = 0;
    if (!image_embed) {
        LOG_TEE("%s: is %s really an image file? No image.\n", __func__, params.image[0].c_str());
    } else {
        process_image(ctx_llava, image_embed, image_embed_slices, &params, n_past);
    }
    
    if (!params.prompt.empty()) {
        LOG_TEE("<User>%s\n", params.prompt.c_str());
        LOG_TEE("<AI>");
        process_user_input(ctx_llava, &params, params.prompt, n_past, true);
    } else {
        LOG_TEE("No prompt given!\n");
        // while (true) {
        //     LOG_TEE("<User>");
        //     std::string prompt;
        //     std::getline(std::cin, prompt);
        //     LOG_TEE("<AI>");
        //     process_user_input(ctx_llava, &params, prompt, n_past, true);
        // }
    }

    llama_perf_context_print(ctx_llava->ctx_llama);

    llava_image_embed_free(image_embed);
    llava_free(ctx_llava);
    return 0;
}
