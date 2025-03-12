#include "arg.h"
#include "base64.hpp"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cogagent.h"

cogagent_ctx cogagent_global;

// This function is mostly copied from cogagent cli
static bool eval_string_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past) {
    int N = (int) tokens.size();

    //// Processing the input tokens in batches
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }

        std::vector<int> pos;
        pos.resize(n_eval);
        for (int i=0; i<n_eval; i++) {
            pos[i] = *n_past + i;
        }

        llama_batch batch = llama_batch_get_one(&tokens[i], n_eval);
        batch.cross_embd = cogagent_global.cross_vision_image_tensor;
        batch.pos = pos.data();
        if (llama_decode(ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

static bool eval_image_tokens(llama_context * ctx_llama, std::vector<float> &img_data,
        int n_batch, int * n_past) {
    int n_embd = 4096;
    int num_tokens = 258;
    int positions[258];

    positions[0] = *n_past;
    for (int i=0; i<num_tokens-2; i++) {
        positions[i + 1] = *n_past + 1;
    }
    positions[num_tokens - 1] = *n_past + 2;

    float * data_ptr = img_data.data();

    for (int i = 0; i < num_tokens; i += n_batch) {
        int n_eval = num_tokens - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        llama_batch batch = {int32_t(n_eval), nullptr, data_ptr, positions, nullptr, nullptr, nullptr, nullptr, nullptr, };
        batch.cross_embd = cogagent_global.cross_vision_image_tensor;
        if (llama_decode(ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        data_ptr += i * n_embd;
    }
    *n_past += 3;
    return true;
}

static void print_usage(int, char ** argv) {
    LOG("\n example usage:\n");
    LOG("\n     %s -m <cogagent-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <cogagent-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> --image <path/to/another/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    LOG("\n note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static const char * sample(struct common_sampler * smpl,
                           struct llama_context * ctx_llama,
                           int * n_past) {
    const llama_token id = common_sampler_sample(smpl, ctx_llama, -1);
    common_sampler_accept(smpl, id, true);

    const llama_model * model = llama_get_model(ctx_llama);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    static std::string ret;
    if (llama_vocab_is_eog(vocab, id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_llama, id);
    }
    // Give the new token to the model. I'm not sure how it is stored.
    // Perhaps it is stored in the KV cache.
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    eval_string_tokens(ctx_llama, tokens, 1, n_past);

    return ret.c_str();
}

static bool run_vision_encoders(const char* vision_encoder_path, const char* image_path) {
    // Load image and resize for the encoders
    std::vector<float> small_image_data;  // For vision encoder
    std::vector<float> large_image_data;  // For cross vision encoder
    if (!load_and_stretch_image(image_path, cogagent_global.vision_encoder_img_size,
        small_image_data, cogagent_global.norm_mean, cogagent_global.norm_deviation)) {
        printf("Failed to load the specified image file.\n");
        return false;
    }
    if (!load_and_stretch_image(image_path, cogagent_global.cross_vision_img_size,
        large_image_data, cogagent_global.norm_mean, cogagent_global.norm_deviation)) {
        printf("Failed to load the specified image file.\n");
        return false;
    }

    // For debugging purposes
    const char * vision_encoder_resized_image = "cogagent_encoders/llama_vision_encoder_input.gguf";
    int dims[3] = {cogagent_global.vision_encoder_img_size,
                     cogagent_global.vision_encoder_img_size, 3};
    save_tensor_from_data(small_image_data, dims, vision_encoder_resized_image);
    const char * cross_vision_resized_image = "cogagent_encoders/llama_cross_vision_input.gguf";
    dims[0] = cogagent_global.cross_vision_img_size;
    dims[1] = cogagent_global.cross_vision_img_size;
    save_tensor_from_data(large_image_data, dims, cross_vision_resized_image);

    // const char * reference_vision_encoder_input = "/home/tianyue/myworkspace"
    //     "/vlm_intermediate/vision_encoder_input.gguf";
    // const char * reference_cross_vision_input = "/home/tianyue/myworkspace"
    //     "/vlm_intermediate/cross_vision_input.gguf";
    // // Load the reference input
    // if (get_input(small_image_data, reference_vision_encoder_input) < 0) {
    //     printf("Failed to load small image input\n");
    //     return false;
    // }
    // if (get_input(large_image_data, reference_cross_vision_input) < 0) {
    //     printf("Failed to load big image input\n");
    //     return false;
    // }
    printf("Loaded and resized the specified image.\n");

    // Load the vision encoder weights
    if (!vision_encoder_init_load(vision_encoder_path)) {
        printf("Failed to load vision encoder model file.\n");
        return false;
    }
    printf("Vision encoder weights loaded.\n");

    // Run the vision encoder
    run_vision_encoder(small_image_data);
    printf("Completed vision encoder run on image file.\n");

    free_vision_encoder_ctx();

    // Load and run the cross vision encoder
    if (!cross_vision_init_load(vision_encoder_path)) {
        printf("Failed to load cross vision encoder model file.\n");
        return false;
    }
    printf("Cross vision encoder weights loaded.\n");

    run_cross_vision(large_image_data);
    printf("Completed cross vision encoder run on image file.\n");

    free_cross_vision_ctx();
    return true;
}

int main(int argc, char ** argv) {
    ggml_time_init();
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COGAGENT, print_usage)) {
        return 1;
    }
    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    // Initialize a GGML context to store the encoded image tensors
    struct ggml_init_params token_ctx_params = {
        size_t(40000000),
        NULL,
        false,
    };
    cogagent_global.token_ctx = ggml_init(token_ctx_params);
    if (!cogagent_global.token_ctx) {
        printf("Failed to initialize token storage context.\n");
        return 1;
    }
    // Allocate the tensor for cross vision encoded image
    cogagent_global.cross_vision_image_tensor = ggml_new_tensor_2d(
        cogagent_global.token_ctx, GGML_TYPE_F32, 1024, 6400
    );

    // Load the images and the encoder models
    // Then run the encoder models
    if (!run_vision_encoders(params.mmproj.c_str(), params.image[0].c_str())) {
        return 1;
    }

    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.c_str(), model_params);
    if (model == nullptr) {
        printf("Failed to load decoder model\n");
        return 1;
    }

    llama_context_params ctx_params = common_context_params_to_llama(params);
    printf("Context size is %d tokens\n", ctx_params.n_ctx);
    llama_context * ctx_llama = llama_init_from_model(model, ctx_params);

    if (ctx_llama == nullptr) {
        printf("Failed to create the llama context\n");
        return 1;
    }

    cogagent_global.ctx_llama = ctx_llama;
    cogagent_global.cogvlm_model = model;

    // At the moment I can't figure out how the llama kv cache
    // keeps its information across runs.
    // It seems to me that the graph is allocated for each batch,
    // which would invalidate any tensors stored in the kv cache.
    // I don't spot logic for separately allocating the kv cache
    // tensors to avoid this, so it doesn't make sense.
    // Maybe the graph isn't actually allocated for each batch?
    // Perhaps that is why a worst case graph is allocated.

    // TODO: Check if system prompt is compatible
    std::vector<llama_token> begin_token;
    const llama_vocab * vocab = llama_model_get_vocab(cogagent_global.cogvlm_model);
    begin_token.push_back(llama_vocab_bos(vocab));

    int n_past = 0;
    printf("Run model with bos token.\n");
    eval_string_tokens(cogagent_global.ctx_llama,
        begin_token, params.n_batch, &n_past);
    printf("Run model with image tokens.\n");
    eval_image_tokens(cogagent_global.ctx_llama, cogagent_global.vision_encoder_image,
        params.n_batch, &n_past);
    // Tokenize user prompt
    // Third option set to false to that the tokenizer doesn't add
    // beginning of sentence and end of sentence
    std::vector<llama_token> user_prompt_tokens = common_tokenize(
        cogagent_global.ctx_llama, params.prompt, false, true
    );
    printf("Run model with user entered text tokens.\n");
    eval_string_tokens(cogagent_global.ctx_llama, user_prompt_tokens,
        params.n_batch, &n_past);

    printf("Parsed maximum sampling length %d.\n", params.n_predict);
    int max_len = params.n_predict < 0 ? 256 : params.n_predict;

    struct common_sampler * smpl = common_sampler_init(cogagent_global.cogvlm_model, params.sampling);
    if (!smpl) {
        printf("Failed to initialize sampler.\n");
        return 1;
    }
    printf("\nReprinting entered prompt.\n %s \n", params.prompt.c_str());
    printf("\n\n Beginning of response.\n");
    std::string response = "";
    for (int i=0; i<max_len; ++i) {
        const char * tmp = sample(smpl, cogagent_global.ctx_llama, &n_past);
        response += tmp;
        if (strcmp(tmp, "</s>") == 0) {
            if (i < 10) {
                continue;
            }
            break;
        }
        printf("%s", tmp);
        fflush(stdout);
    }
    common_sampler_free(smpl);

    llama_model_free(model);
    ggml_free(cogagent_global.token_ctx);
    return 0;
}