#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"
#include "sampling.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [--image img_path] [-p prompt]\n", argv[0]);
    printf("\n");
}

static llama_vision_bitmap * load_image_from_file(const char * fname) {
    std::ifstream file(fname, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open file");
    }
    std::vector<char> image_bytes = std::vector<char>(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>());
    // decode image to byte array
    int nx, ny, nc;
    auto * bytes = (unsigned char *) image_bytes.data();
    auto * img = stbi_load_from_memory(bytes, image_bytes.size(), &nx, &ny, &nc, 3);
    if (!img) {
        throw std::runtime_error("failed to decode image bytes");
    }
    // printf("nx=%d ny=%d nc=%d\n", nx, ny, nc);
    // GGML_ASSERT(nc == 3);
    // for (int y = 0; y < ny; y++) {
    //     for (int x = 0; x < nx; x++) {
    //         unsigned char * pix = img + x*nc + y*nc*nx;
    //         printf("%02x%02x%02x ", pix[0], pix[1], pix[2]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    llama_vision_bitmap * result = llama_vision_bitmap_init(nx, ny);
    memcpy(result->data, img, nx*ny*3);
    stbi_image_free(img);
    return result;
}

// split string by a `std::string delim` instead of `char delim`
static std::vector<std::string> string_split_str(std::string s, const std::string & delimiter) {
    std::vector<std::string> tokens;
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        tokens.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    tokens.push_back(s);
    return tokens;
}

struct tokenized_part {
    llama_tokens tokens;
    bool is_image;
};

// TODO: this function is hacky, need to be improved
// static const llama_token TOKEN_IMG_PLACEMENT = -1000;
static const std::string IMG_PLACEMENT = "<img_placement>";
static std::vector<tokenized_part> tokenize_with_img_placement(
        const llama_vocab * vocab,
        const std::string & text,
        bool   add_special,
        bool   parse_special) {
    std::vector<std::string> parts = string_split_str(text, IMG_PLACEMENT);
    std::vector<tokenized_part> output;
    for (const auto & part : parts) {
        //printf("tokenizing part: %s\n", part.c_str());
        bool add_bos = &parts.front() == &part;
        auto tokens = common_tokenize(vocab, part, add_special && add_bos, parse_special);
        if (tokens.empty()) {
            continue;
        }
        output.push_back({std::move(tokens), false});
        if (&parts.back() != &part) {
            // add image token to middle of 2 parts
            output.push_back({{}, true});
        }
    }
    return output;
}

int main(int argc, char ** argv) {
    common_params params;

    // default prompt for llava 1.5
    //params.prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:<img_placement>\nwhat did you see?\nASSISTANT:";
    // default prompt for minicpmv 2.6
    params.prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<img_placement>\nwhat do you see?<|im_end|>\n<|im_start|>assistant\n";
    params.n_predict = 64;
    params.n_batch = 2048;
    params.n_ubatch = 1024;
    params.n_gpu_layers = 99;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_VISION, print_usage)) {
        return 1;
    }

    common_init();
    common_init_result llama_init = common_init_from_params(params);
    llama_context * ctx = llama_init.context.get();
    const llama_model * model = llama_init.model.get();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!model) {
        LOG_ERR("failed to load model\n");
        return 1;
    }

    llama_vision_context_params vparams = llama_vision_context_default_params();
    vparams.n_threads = llama_n_threads(ctx);
    llama_vision_context * vctx = llama_vision_init_from_model(model, vparams);
    if (!vctx) {
        LOG_ERR("model does not have vision encoder\n");
        return 1;
    }

    struct common_sampler * smpl = common_sampler_init(model, params.sampling);

    llama_batch batch = llama_batch_init(llama_n_batch(ctx), 0, 1);
    int n_past = 0;
    int n_prompt = 0;

    // process image
    llama_vision_tokens * img_tokens = nullptr;
    {
        const char * img_path = params.image[0].c_str();
        if (params.image[0].empty()) {
            LOG_ERR("no image path provided\n");
            return 1;
        }
        llama_vision_bitmap * img = load_image_from_file(img_path);
        LOG_INF("loaded image %s, size = %d x %d\n", img_path, img->nx, img->ny);
        img_tokens = llama_vision_tokenize(vctx, img);
        if (!img_tokens) {
            LOG_ERR("failed to create image tokens\n");
            return 1;
        }
        if (llama_vision_encode(vctx, img_tokens)) {
            LOG_ERR("failed to encode image\n");
            return 1;
        }
        LOG_INF("encoded image\n");
    }

    // process prompt
    {
        std::vector<tokenized_part> parts = tokenize_with_img_placement(vocab, params.prompt, true, true);
        for (const tokenized_part & part : parts) {
            if (!part.is_image) {
                for (const llama_token & token : part.tokens) {
                    //LOG_INF("%d -> %s\n", token, common_token_to_piece(ctx, token).c_str());
                    common_batch_add(batch, token, n_past++, {0}, &part == &parts.back());
                }
                LOG_INF("eval text batch (%d tokens)\n", batch.n_tokens);
                if (llama_decode(ctx, batch)) {
                    LOG_ERR("failed to decode text prompt\n");
                    return 1;
                }
            } else {
                auto * img_embd = llama_vision_get_output_tensor(vctx);
                // std::vector<float> output_debug(ggml_nelements(img_embd));
                // ggml_backend_tensor_get(img_embd, output_debug.data(), 0, ggml_nbytes(img_embd));
                // for (int row = 0; row < 10; row++) {
                //     int off = row * img_embd->ne[0];
                //     printf("... %f %f %f\n", output_debug[off], output_debug[off+1], output_debug[off+2]);
                // }
                // exit(1);
                llama_batch batch_img = llama_batch_get_one_from_tensor(img_embd, n_past, 0);
                n_past += batch_img.n_tokens;
                LOG_INF("eval image batch (%d embeddings)\n", batch_img.n_tokens);
                if (llama_decode(ctx, batch_img)) {
                    LOG_ERR("failed to decode image prompt\n");
                    return 1;
                }
                llama_batch_free(batch_img);
            }
        }
        n_prompt = n_past;
        LOG_INF("prompt processed, %d tokens\n", n_prompt);
    }

    // generate response
    while (true){
        int n_generated = n_past - n_prompt;
        if (n_generated > params.n_predict) {
            printf("\n");
            break;
        }

        llama_token token_id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, token_id, true);
        printf("%s", common_token_to_piece(ctx, token_id).c_str());
        fflush(stdout);

        if (llama_vocab_is_eog(vocab, token_id)) {
            printf("\n");
            break;
        }

        // eval the token
        common_batch_clear(batch);
        common_batch_add(batch, token_id, n_past++, {0}, true);
        if (llama_decode(ctx, batch)) {
            LOG_ERR("failed to decode token\n");
            break;
        }
    }

    return 0;
}
