#include "mtmd-ios.h"
#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <vector>
#include <string>
#include <limits.h>
#include <cinttypes>
#include <memory>
#include <cstring>
#include <cstdlib>

struct mtmd_ios_context {
    mtmd::context_ptr ctx_vision;
    common_init_result llama_init;
    
    llama_model* model;
    llama_context* lctx;
    const llama_vocab* vocab;
    common_sampler* smpl;
    llama_batch batch;
    
    mtmd::bitmaps bitmaps;
    common_chat_templates_ptr tmpls;
    
    int n_threads;
    llama_pos n_past;
    int n_predict;
    
    std::string last_error;
    
    ~mtmd_ios_context() {
        if (batch.token) {
            llama_batch_free(batch);
        }
        if (smpl) {
            common_sampler_free(smpl);
        }
    }
};

void mtmd_ios_string_free(char* str) {
    if (str) {
        free(str);
    }
}

static void set_error(mtmd_ios_context* ctx, const std::string& error) {
    ctx->last_error = error;
}

static bool load_media_from_buffer(mtmd_ios_context* ctx, const unsigned char* buffer, size_t size) {
    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(ctx->ctx_vision.get(), buffer, size));
    if (!bmp.ptr) {
        return false;
    }
    ctx->bitmaps.entries.push_back(std::move(bmp));
    return true;
}

static int eval_message_internal(mtmd_ios_context* ctx, const common_chat_msg& msg, bool add_bos = false) {
    common_chat_templates_inputs tmpl_inputs;
    tmpl_inputs.messages = {msg};
    tmpl_inputs.add_generation_prompt = true;
    tmpl_inputs.use_jinja = false;
    
    auto formatted_chat = common_chat_templates_apply(ctx->tmpls.get(), tmpl_inputs);
    
    mtmd_input_text text;
    text.text = formatted_chat.prompt.c_str();
    text.add_special = add_bos;
    text.parse_special = true;
    
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = ctx->bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(ctx->ctx_vision.get(),
                               chunks.ptr.get(),
                               &text,
                               bitmaps_c_ptr.data(),
                               bitmaps_c_ptr.size());
    if (res != 0) {
        set_error(ctx, "Unable to tokenize prompt, res = " + std::to_string(res));
        return 1;
    }
    
    ctx->bitmaps.entries.clear();
    
    llama_pos new_n_past;
    if (mtmd_helper_eval_chunks(ctx->ctx_vision.get(),
                               ctx->lctx,
                               chunks.ptr.get(),
                               ctx->n_past,
                               0,
                               2048,
                               true,
                               &new_n_past)) {
        set_error(ctx, "Unable to eval prompt");
        return 1;
    }
    
    ctx->n_past = new_n_past;
    return 0;
}

mtmd_ios_params mtmd_ios_params_default(void) {
    mtmd_ios_params params = {};
    params.model_path = nullptr;
    params.mmproj_path = nullptr;
    params.n_predict = -1;
    params.n_ctx = 4096;
    params.n_threads = 4;
    params.temperature = 0.2f;
    params.use_gpu = true;
    params.mmproj_use_gpu = true;
    return params;
}

mtmd_ios_context* mtmd_ios_init(const mtmd_ios_params* params) {
    if (!params || !params->model_path || !params->mmproj_path) {
        return nullptr;
    }
    
    ggml_time_init();
    common_init();
    
    auto ctx = std::make_unique<mtmd_ios_context>();
    
    ctx->n_predict = params->n_predict;
    ctx->n_threads = params->n_threads;
    ctx->n_past = 0;
    
    common_params common_params;
    common_params.model.path = params->model_path;
    common_params.mmproj.path = params->mmproj_path;
    common_params.n_ctx = params->n_ctx;
    common_params.n_batch = 2048;
    common_params.cpuparams.n_threads = params->n_threads;
    common_params.sampling.temp = params->temperature;
    common_params.mmproj_use_gpu = params->mmproj_use_gpu;
    
    ctx->llama_init = common_init_from_params(common_params);
    ctx->model = ctx->llama_init.model.get();
    ctx->lctx = ctx->llama_init.context.get();
    ctx->vocab = llama_model_get_vocab(ctx->model);
    ctx->smpl = common_sampler_init(ctx->model, common_params.sampling);
    ctx->batch = llama_batch_init(1, 0, 1);
    
    if (!ctx->model || !ctx->lctx) {
        set_error(ctx.get(), "Failed to load model or create context");
        return nullptr;
    }
    
    if (!llama_model_chat_template(ctx->model, nullptr)) {
        set_error(ctx.get(), "Model does not have chat template");
        return nullptr;
    }
    
    ctx->tmpls = common_chat_templates_init(ctx->model, "");
    
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = params->mmproj_use_gpu;
    mparams.print_timings = false;
    mparams.n_threads = params->n_threads;
    mparams.verbosity = GGML_LOG_LEVEL_INFO;
    
    ctx->ctx_vision.reset(mtmd_init_from_file(params->mmproj_path, ctx->model, mparams));
    if (!ctx->ctx_vision.get()) {
        set_error(ctx.get(), "Failed to load vision model from " + std::string(params->mmproj_path));
        return nullptr;
    }
    
    return ctx.release();
}

void mtmd_ios_free(mtmd_ios_context* ctx) {
    if (ctx) {
        delete ctx;
    }
}

char* mtmd_ios_generate(mtmd_ios_context* ctx, const mtmd_ios_message* message) {
    if (!ctx || !message) {
        return nullptr;
    }
    
    for (int i = 0; i < message->n_images; i++) {
        if (!load_media_from_buffer(ctx, message->image_buffers[i], message->image_sizes[i])) {
            set_error(ctx, "Failed to load image");
            return nullptr;
        }
    }
    
    for (int i = 0; i < message->n_audios; i++) {
        if (!load_media_from_buffer(ctx, message->audio_buffers[i], message->audio_sizes[i])) {
            set_error(ctx, "Failed to load audio");
            return nullptr;
        }
    }
    
    std::string prompt = message->content;
    if (prompt.find(mtmd_default_marker()) == std::string::npos) {
        for (int i = 0; i < message->n_images + message->n_audios; i++) {
            prompt += mtmd_default_marker();
        }
    }
    
    common_chat_msg msg;
    msg.role = message->role;
    msg.content = prompt;
    
    if (eval_message_internal(ctx, msg, true)) {
        return nullptr;
    }
    
    std::string response;
    int n_predict = ctx->n_predict < 0 ? INT_MAX : ctx->n_predict;
    
    for (int i = 0; i < n_predict; i++) {
        llama_token token_id = common_sampler_sample(ctx->smpl, ctx->lctx, -1);
        common_sampler_accept(ctx->smpl, token_id, true);
        
        if (llama_vocab_is_eog(ctx->vocab, token_id)) {
            break;
        }
        
        std::string token_str = common_token_to_piece(ctx->lctx, token_id);
        response += token_str;
        
        common_batch_clear(ctx->batch);
        common_batch_add(ctx->batch, token_id, ctx->n_past++, {0}, true);
        if (llama_decode(ctx->lctx, ctx->batch)) {
            set_error(ctx, "failed to decode token");
            return nullptr;
        }
    }
    
    char* result_cstr = (char*)malloc(response.length() + 1);
    if (result_cstr) {
        strcpy(result_cstr, response.c_str());
    }
    
    return result_cstr;
}

const char* mtmd_ios_get_last_error(mtmd_ios_context* ctx) {
    return ctx ? ctx->last_error.c_str() : nullptr;
}
