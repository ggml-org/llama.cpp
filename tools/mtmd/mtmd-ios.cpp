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
#include <iostream>

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

mtmd_ios_params mtmd_ios_params_default(void) {
    mtmd_ios_params params = {};
    params.model_path = "";
    params.mmproj_path = "";
    params.n_predict = -1;
    params.n_ctx = 4096;
    params.n_threads = 4;
    params.temperature = 0.2f;
    params.use_gpu = true;
    return params;
}

mtmd_ios_context* mtmd_ios_init(const mtmd_ios_params* params) {
    if (!params || params->model_path.empty() || params->mmproj_path.empty()) {
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
    common_params.n_batch = 2048;  // 增加batch大小，与标准mtmd保持一致
    common_params.cpuparams.n_threads = params->n_threads;
    common_params.sampling.temp = params->temperature;
    common_params.mmproj_use_gpu = params->mmproj_use_gpu;

    ctx->llama_init = common_init_from_params(common_params);
    
    ctx->model = ctx->llama_init.model.get();
    ctx->lctx = ctx->llama_init.context.get();
    
    if (!ctx->model || !ctx->lctx) {
        set_error(ctx.get(), "Failed to load model or create context");
        return nullptr;
    }
    
    ctx->vocab = llama_model_get_vocab(ctx->model);
    
    ctx->smpl = common_sampler_init(ctx->model, common_params.sampling);
    if (!ctx->smpl) {
        set_error(ctx.get(), "Failed to initialize sampler");
        return nullptr;
    }
    
    ctx->batch = llama_batch_init(2048, 0, 1);
    if (!ctx->batch.token) {
        set_error(ctx.get(), "Failed to initialize batch");
        return nullptr;
    }
    
    std::string chat_template = "";
    if (!llama_model_chat_template(ctx->model, nullptr)) {
        chat_template = "chatml";
    }
    
    ctx->tmpls = common_chat_templates_init(ctx->model, chat_template);
    if (!ctx->tmpls) {
        set_error(ctx.get(), "Failed to initialize chat templates");
        return nullptr;
    }
    
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = params->mmproj_use_gpu;
    mparams.print_timings = false;
    mparams.n_threads = params->n_threads;
    mparams.verbosity = GGML_LOG_LEVEL_INFO;
    
    ctx->ctx_vision.reset(mtmd_init_from_file(params->mmproj_path.c_str(), ctx->model, mparams));
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

int mtmd_ios_prefill_image(mtmd_ios_context* ctx, const std::string& image_path) {
    if (!ctx || image_path.empty()) {
        return -1;
    }
    
    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx->ctx_vision.get(), image_path.c_str()));
    if (!bmp.ptr) {
        set_error(ctx, "Failed to load image from file: " + image_path);
        return -1;
    }
    ctx->bitmaps.entries.push_back(std::move(bmp));
    
    mtmd_input_text text;
    text.text = mtmd_default_marker();
    text.add_special = ctx->n_past == 0;
    text.parse_special = true;
    
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = ctx->bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(ctx->ctx_vision.get(),
                        chunks.ptr.get(),
                        &text,
                        bitmaps_c_ptr.data(),
                        bitmaps_c_ptr.size());
    if (res != 0) {
        set_error(ctx, "Failed to tokenize image");
        return -1;
    }
    
    ctx->bitmaps.entries.clear();
    
    llama_pos new_n_past;
    if (mtmd_helper_eval_chunks(ctx->ctx_vision.get(),
                ctx->lctx,
                chunks.ptr.get(),
                ctx->n_past,
                0,
                1024,
                false,
                &new_n_past)) {
        set_error(ctx, "Failed to eval image");
        return -1;
    }
    
    ctx->n_past = new_n_past;
    
    return 0;
}



int mtmd_ios_prefill_text(mtmd_ios_context* ctx, const std::string& text, const std::string& role) {
    if (!ctx || text.empty() || role.empty()) {
        return -1;
    }
    
    common_chat_msg msg;
    msg.role = role.c_str();
    msg.content = text.c_str();
    
    common_chat_templates_inputs tmpl_inputs;
    tmpl_inputs.messages = {msg};
    tmpl_inputs.add_generation_prompt = false;
    tmpl_inputs.use_jinja = false;
    auto formatted_chat = common_chat_templates_apply(ctx->tmpls.get(), tmpl_inputs);
    
    mtmd_input_text input_text;
    input_text.text = formatted_chat.prompt.c_str();
    input_text.add_special = ctx->n_past == 0;
    input_text.parse_special = true;
    
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    int32_t res = mtmd_tokenize(ctx->ctx_vision.get(),
                        chunks.ptr.get(),
                        &input_text,
                        nullptr,
                        0);
    if (res != 0) {
        set_error(ctx, "Failed to tokenize text");
        return -1;
    }
    
    llama_pos new_n_past;
    if (mtmd_helper_eval_chunks(ctx->ctx_vision.get(),
                ctx->lctx,
                chunks.ptr.get(),
                ctx->n_past,
                0,
                1024,
                true,
                &new_n_past)) {
        set_error(ctx, "Failed to eval text");
        return -1;
    }
    
    ctx->n_past = new_n_past;
    return 0;
}



mtmd_ios_token mtmd_ios_loop(mtmd_ios_context* ctx) {
    mtmd_ios_token result = {nullptr, true};
    
    if (!ctx) {
        return result;
    }
    
    llama_token token_id = common_sampler_sample(ctx->smpl, ctx->lctx, -1);
    common_sampler_accept(ctx->smpl, token_id, true);
    
    if (llama_vocab_is_eog(ctx->vocab, token_id)) {
        result.is_end = true;
        return result;
    }
    
    std::string token_str = common_token_to_piece(ctx->lctx, token_id);
    
    common_batch_clear(ctx->batch);
    common_batch_add(ctx->batch, token_id, ctx->n_past, {0}, true);
    
    if (ctx->batch.n_tokens > 0) {
        ctx->batch.logits[ctx->batch.n_tokens - 1] = true;
    }
    
    ctx->n_past++;
    if (llama_decode(ctx->lctx, ctx->batch)) {
        set_error(ctx, "failed to decode token");
        result.is_end = true;
        return result;
    }
    
    result.token = (char*)malloc(token_str.length() + 1);
    if (result.token) {
        strcpy(result.token, token_str.c_str());
    }
    result.is_end = false;
    
    return result;
}

const char* mtmd_ios_get_last_error(mtmd_ios_context* ctx) {
    return ctx ? ctx->last_error.c_str() : nullptr;
}
