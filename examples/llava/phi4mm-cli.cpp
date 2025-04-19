#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "clip.h"
#include "stb_image.h"
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>

struct phi4mm_context {
    struct clip_ctx    * ctx_clip = NULL;
    common_init_result   llama_init;

    llama_model        * model;
    llama_context      * lctx;
    llama_adapter_lora * vision_lora;

    phi4mm_context(common_params & params) : llama_init(common_init_from_params(params)) {
        model = llama_init.model.get();
        lctx = llama_init.context.get();
        vision_lora = llama_init.lora[0].get();
        llama_clear_adapter_lora(lctx);
        init_clip_model(params);
    }

    void init_clip_model(common_params & params) {
        const char * clip_path = params.mmproj.c_str();
        ctx_clip = clip_model_load(clip_path, params.verbosity > 1);
    }

    ~phi4mm_context() {
        clip_free(ctx_clip);
    }
};

struct decode_embd_batch {
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch;
    decode_embd_batch(float * embd, int32_t n_tokens, llama_pos pos_0, llama_seq_id seq_id) {
        pos     .resize(n_tokens);
        n_seq_id.resize(n_tokens);
        seq_ids .resize(n_tokens + 1);
        logits  .resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0] = seq_id;
        seq_ids [n_tokens] = nullptr;
        batch = {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ embd,
            /*pos            =*/ pos.data(),
            /*n_seq_id       =*/ n_seq_id.data(),
            /*seq_id         =*/ seq_ids.data(),
            /*logits         =*/ logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            batch.pos     [i] = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }
};

struct inp_bitmap {
    int nx;
    int ny;
    std::vector<unsigned char> data;
};

static void show_additional_info(int /*argc*/, char ** argv) {
    GGML_UNUSED(argv);
    LOG("TODO\n");
}

static void eval_text(phi4mm_context & ctx, int & n_past, std::string input, bool logits_last = false) {
    llama_tokens tokens = common_tokenize(ctx.lctx, input, false, true);
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (llama_token & t : tokens) {
        common_batch_add(batch, t, n_past++, {0}, false);
    }
    if (logits_last) {
        batch.logits[batch.n_tokens - 1] = true;
    }
    LOG("eval_text (n_tokens = %d): %s\n", (int)tokens.size(), input.c_str());
    if (llama_decode(ctx.lctx, batch)) {
        GGML_ABORT("Failed to decode\n");
    }
}

int main(int argc, char ** argv) {
    ggml_time_init();

    common_params params;

    // default values
    params.prompt = "<|user|>$what did you see?<|end|><|assistant|>";
    params.n_predict = 64;
    params.sampling.temp = 0.0f;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LLAVA, show_additional_info)) {
        return 1;
    }

    common_init();

    if (params.mmproj.empty() || (params.image.empty())) {
        show_additional_info(argc, argv);
        return 1;
    }

    if (params.lora_adapters.empty()) {
        LOG_ERR("error: no vision lora adapters specified\n");
        return 1;
    }

    phi4mm_context ctx(params);
    printf("%s: %s\n", __func__, params.model.c_str());

    int n_threads = params.cpuparams.n_threads;
    int n_past = 0;

    std::vector<std::string> prompt_parts = string_split<std::string>(params.prompt, '$');
    GGML_ASSERT(prompt_parts.size() == 2);
    eval_text(ctx, n_past, prompt_parts[0], false);

    // process images
    for (auto & image : params.image) {
        //break;
        std::vector<float> image_embd_v;
        int n_embd = llama_model_n_embd(ctx.model);
        int n_tokens = 256;
        image_embd_v.resize(n_tokens * n_embd);

        bool ok;
        struct clip_image_u8 * img_u8 = clip_image_u8_init();
        ok = clip_image_load_from_file(image.c_str(), img_u8);
        if (!ok) {
            LOG_ERR("Unable to load image %s\n", image.c_str());
            return 1;
        }

        clip_image_f32_batch batch_f32;
        ok = clip_image_preprocess(ctx.ctx_clip, img_u8, &batch_f32);
        if (!ok) {
            LOG_ERR("Unable to preprocess image\n");
            return 1;
        }

        LOG("Encoding image %s\n", image.c_str());
        ok = clip_image_batch_encode(ctx.ctx_clip, n_threads, &batch_f32, image_embd_v.data());
        if (!ok) {
            LOG_ERR("Unable to encode image\n");
            return 1;
        }

        // debug
        // for (int i = 0; i < 10; i++) {
        //     LOG("embd[%d] = %f, %f, %f\n", i, image_embd_v[i*n_embd], image_embd_v[i*n_embd+1], image_embd_v[i*n_embd+2]);
        // }

        clip_image_f32_batch_free(&batch_f32);
        clip_image_u8_free(img_u8);

        // decode image embeddings
        llama_set_adapter_lora(ctx.lctx, ctx.vision_lora, 1.0f);
        decode_embd_batch batch_img(image_embd_v.data(), n_tokens, n_past, 0);
        if (llama_decode(ctx.lctx, batch_img.batch)) {
            LOG_ERR("failed to decode image\n");
            return 1;
        }
        llama_clear_adapter_lora(ctx.lctx);
        n_past += n_tokens;
    }

    eval_text(ctx, n_past, prompt_parts[1], true);

    // generate text
    struct common_sampler * smpl = common_sampler_init(ctx.model, params.sampling);
    const llama_vocab * vocab    = llama_model_get_vocab(ctx.model);
    int n_prompt = n_past;
    llama_batch batch = llama_batch_init(1, 0, 1);
    while (true) {
        int n_generated = n_past - n_prompt;
        if (n_generated > params.n_predict) {
            printf("\n");
            break;
        }

        llama_token token_id = common_sampler_sample(smpl, ctx.lctx, -1);
        common_sampler_accept(smpl, token_id, true);
        printf("%s", common_token_to_piece(ctx.lctx, token_id).c_str());
        fflush(stdout);

        if (llama_vocab_is_eog(vocab, token_id)) {
            printf("\n");
            break;
        }

        // eval the token
        common_batch_clear(batch);
        common_batch_add(batch, token_id, n_past++, {0}, true);
        if (llama_decode(ctx.lctx, batch)) {
            LOG_ERR("failed to decode token\n");
            break;
        }
    }

    llama_batch_free(batch);

    return 0;
}
