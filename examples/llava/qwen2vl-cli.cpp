#include "arg.h"
#include "base64.hpp"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "clip.h"
#include "llava.h"
#include "llama.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef NDEBUG
#include "ggml-alloc.h"
#include "ggml-backend.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>


static bool qwen2vl_eval_image_embed(llama_context * ctx_llama, const struct llava_image_embed * image_embed,
                                     int n_batch, int * n_past, int * st_pos_id, struct clip_image_size * image_size) {
    int n_embd  = llama_n_embd(llama_get_model(ctx_llama));
    const int patch_size = 14 * 2;
    const int ph = image_size->height / patch_size + (image_size->height % patch_size > 0);
    const int pw = image_size->width / patch_size + (image_size->width % patch_size > 0);
    auto img_tokens = image_embed->n_image_pos;
    // llama_pos mrope_pos[img_tokens * 4];
    std::vector<llama_pos> mrope_pos;
    mrope_pos.resize(img_tokens * 4);

    for (int y = 0; y < ph; y++)
    {
        for (int x = 0; x < pw; x++)
        {
            int i = y * pw + x;
            mrope_pos[i] = *st_pos_id;
            mrope_pos[i + img_tokens] = *st_pos_id + y;
            mrope_pos[i + img_tokens * 2] = *st_pos_id + x;
            mrope_pos[i + img_tokens * 3] = 0;
        }
    }
    *st_pos_id += std::max(pw, ph);

    int processed = 0;
    std::vector<llama_pos> batch_mrope_pos;
    batch_mrope_pos.resize(img_tokens * 4);

    for (int i = 0; i < img_tokens; i += n_batch) {
        int n_eval = img_tokens - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }

        // llama_pos batch_mrope_pos[n_eval * 4];
        std::fill(batch_mrope_pos.begin(), batch_mrope_pos.end(), 0);
        memcpy(batch_mrope_pos.data(), &mrope_pos[processed], n_eval * sizeof(llama_pos));
        memcpy(&batch_mrope_pos[n_eval * 1], &mrope_pos[img_tokens * 1 + processed], n_eval * sizeof(llama_pos));
        memcpy(&batch_mrope_pos[n_eval * 2], &mrope_pos[img_tokens * 2 + processed], n_eval * sizeof(llama_pos));
        memcpy(&batch_mrope_pos[n_eval * 3], &mrope_pos[img_tokens * 3 + processed], n_eval * sizeof(llama_pos));

        llama_batch batch = {
            int32_t(n_eval),                // n_tokens
            nullptr,                        // token
            (image_embed->embed+i*n_embd),  // embed
            batch_mrope_pos.data(),         // pos
            nullptr,  // n_seq_id
            nullptr,  // seq_id
            nullptr,  // logits
        };

        if (llama_decode(ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
        processed += n_eval;
    }
    return true;
}


static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past, int * st_pos_id) {
    int N = (int) tokens.size();
    std::vector<llama_pos> pos;
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        auto batch = llama_batch_get_one(&tokens[i], n_eval);
        // TODO: add mrope pos ids somewhere else
        pos.resize(batch.n_tokens * 4);
        std::fill(pos.begin(), pos.end(), 0);
        for (int j = 0; j < batch.n_tokens * 3; j ++) {
            pos[j] = *st_pos_id + (j % batch.n_tokens);
        }
        batch.pos = pos.data();

        if (llama_decode(ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
        *st_pos_id += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context * ctx_llama, int id, int * n_past, int * st_pos_id) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past, st_pos_id);
}

static bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, int * st_pos_id, bool add_bos){
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = common_tokenize(ctx_llama, str2, add_bos, true);
    eval_tokens(ctx_llama, embd_inp, n_batch, n_past, st_pos_id);
    return true;
}

static const char * sample(struct common_sampler * smpl,
                           struct llama_context * ctx_llama,
                           int * n_past, int * st_pos_id) {
    const llama_token id = common_sampler_sample(smpl, ctx_llama, -1);
    common_sampler_accept(smpl, id, true);
    static std::string ret;
    if (llama_token_is_eog(llama_get_model(ctx_llama), id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past, st_pos_id);
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
        LOG_ERR("%s: invalid base64 image tag. must be %s<base64 byte string>%s\n", __func__, IMG_BASE64_TAG_BEGIN, IMG_BASE64_TAG_END);
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
        LOG_ERR("%s: could not load image from base64 string.\n", __func__);
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

struct llava_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

static void print_usage(int, char ** argv) {
    LOG("\n example usage:\n");
    LOG("\n     %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> --image <path/to/another/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    LOG("\n note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static struct llava_image_embed * load_image(llava_context * ctx_llava, common_params * params, const std::string & fname) {

    // load and preprocess the image
    llava_image_embed * embed = NULL;
    auto prompt = params->prompt;
    if (prompt_contains_image(prompt)) {
        if (!params->image.empty()) {
            LOG_INF("using base64 encoded image instead of command line image path\n");
        }
        embed = llava_image_embed_make_with_prompt_base64(ctx_llava->ctx_clip, params->cpuparams.n_threads, prompt);
        if (!embed) {
            LOG_ERR("%s: can't load image from prompt\n", __func__);
            return NULL;
        }
        params->prompt = remove_image_from_prompt(prompt);
    } else {
        embed = llava_image_embed_make_with_filename(ctx_llava->ctx_clip, params->cpuparams.n_threads, fname.c_str());
        if (!embed) {
            fprintf(stderr, "%s: is %s really an image file?\n", __func__, fname.c_str());
            return NULL;
        }
    }

    return embed;
}

static void process_prompt(struct llava_context * ctx_llava, struct llava_image_embed * image_embed, common_params * params, const std::string & prompt) {
    int n_past = 0;
    int cur_pos_id = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;

    std::string system_prompt, user_prompt;
    size_t image_pos = prompt.find("<|vision_start|>");
    if (image_pos != std::string::npos) {
        // new templating mode: Provide the full prompt including system message and use <image> as a placeholder for the image
        system_prompt = prompt.substr(0, image_pos);
        user_prompt = prompt.substr(image_pos + std::string("<|vision_pad|>").length());
        LOG_INF("system_prompt: %s\n", system_prompt.c_str());
        if (params->verbose_prompt) {
            auto tmp = common_tokenize(ctx_llava->ctx_llama, system_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
        LOG_INF("user_prompt: %s\n", user_prompt.c_str());
        if (params->verbose_prompt) {
            auto tmp = common_tokenize(ctx_llava->ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
    } else {
        // llava-1.5 native mode
        system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>";
        user_prompt = "<|vision_end|>" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        if (params->verbose_prompt) {
            auto tmp = common_tokenize(ctx_llava->ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
    }

    eval_string(ctx_llava->ctx_llama, system_prompt.c_str(), params->n_batch, &n_past, &cur_pos_id, true);
    if (image_embed != nullptr) {
        auto image_size = clip_get_load_image_size(ctx_llava->ctx_clip);
        qwen2vl_eval_image_embed(ctx_llava->ctx_llama, image_embed, params->n_batch, &n_past, &cur_pos_id, image_size);
    }
    eval_string(ctx_llava->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, &cur_pos_id, false);

    // generate the response

    LOG("\n");

    struct common_sampler * smpl = common_sampler_init(ctx_llava->model, params->sampling);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        exit(1);
    }

    std::string response = "";
    for (int i = 0; i < max_tgt_len; i++) {
        const char * tmp = sample(smpl, ctx_llava->ctx_llama, &n_past, &cur_pos_id);
        response += tmp;
        if (strcmp(tmp, "</s>") == 0) break;
        if (strstr(tmp, "###")) break; // Yi-VL behavior
        LOG("%s", tmp);
        if (strstr(response.c_str(), "<|im_end|>")) break; // Yi-34B llava-1.6 - for some reason those decode not as the correct token (tokenizer works)
        if (strstr(response.c_str(), "<|im_start|>")) break; // Yi-34B llava-1.6
        if (strstr(response.c_str(), "USER:")) break; // mistral llava-1.6

        fflush(stdout);
    }

    common_sampler_free(smpl);
    LOG("\n");
}

static struct llama_model * llava_init(common_params * params) {
    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = common_model_params_to_llama(*params);

    llama_model * model = llama_model_load_from_file(params->model.c_str(), model_params);
    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n" , __func__);
        return NULL;
    }
    return model;
}

static struct llava_context * llava_init_context(common_params * params, llama_model * model) {
    const char * clip_path = params->mmproj.c_str();

    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }

    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);


    llama_context_params ctx_params = common_context_params_to_llama(*params);
    ctx_params.n_ctx           = params->n_ctx < 2048 ? 2048 : params->n_ctx; // we need a longer context size to process image embeddings

    llama_context * ctx_llama = llama_new_context_with_model(model, ctx_params);

    if (ctx_llama == NULL) {
        LOG_ERR("%s: failed to create the llama_context\n" , __func__);
        return NULL;
    }

    auto * ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));

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
    llama_model_free(ctx_llava->model);
    llama_backend_free();
}

#ifndef NDEBUG

static void debug_test_mrope_2d() {
    // 1. Initialize backend
    ggml_backend_t backend = NULL;
    std::string backend_name = "";
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    backend = ggml_backend_cuda_init(0); // init device 0
    backend_name = "cuda";
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif
    // if there aren't GPU Backends fallback to CPU backend
    if (!backend) {
        backend = ggml_backend_cpu_init();
        backend_name = "cpu";
    }

    // Calculate the size needed to allocate
    size_t ctx_size = 0;
    ctx_size += 2 * ggml_tensor_overhead(); // tensors
    // no need to allocate anything else!

    // 2. Allocate `ggml_context` to store tensor data
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * inp_raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 12, 30);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 30 * 4);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);

    std::vector<float> dummy_q;
    dummy_q.resize(128 * 12 * 30);
    std::fill(dummy_q.begin(), dummy_q.end(), 0.1);
    // memcpy(inp_raw->data, dummy_q.data(), 128 * 12 * 30 * ggml_element_size(inp_raw));

    std::vector<int> pos_id;
    pos_id.resize(30 * 4);
    for (int i = 0; i < 30; i ++) {
        pos_id[i] = i;
        pos_id[i + 30] = i + 10;
        pos_id[i + 60] = i + 20;
        pos_id[i + 90] = i + 30;
    }
    int sections[4] = {32, 32, 0, 0};

    // 4. Allocate a `ggml_backend_buffer` to store all tensors
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // 5. Copy tensor data from main memory (RAM) to backend buffer
    ggml_backend_tensor_set(inp_raw, dummy_q.data(), 0, ggml_nbytes(inp_raw));
    ggml_backend_tensor_set(pos, pos_id.data(), 0, ggml_nbytes(pos));

    // 6. Create a `ggml_cgraph` for mul_mat operation
    struct ggml_cgraph * gf = NULL;
    struct ggml_context * ctx_cgraph = NULL;

    // create a temporally context to build the graph
    struct ggml_init_params params0 = {
        /*.mem_size   =*/ ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };
    ctx_cgraph = ggml_init(params0);
    gf = ggml_new_graph(ctx_cgraph);

    struct ggml_tensor * result0 = ggml_rope_multi(
        ctx_cgraph, inp_raw, pos, nullptr,
        128/2, sections, LLAMA_ROPE_TYPE_VISION, 32768, 1000000, 1,
        0, 1, 32, 1);

    // Add "result" tensor and all of its dependencies to the cgraph
    ggml_build_forward_expand(gf, result0);

    // 7. Create a `ggml_gallocr` for cgraph computation
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // 9. Run the computation
    int n_threads = 1; // Optional: number of threads to perform some operations with multi-threading
    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }
    ggml_backend_graph_compute(backend, gf);

    // 10. Retrieve results (output tensors)
    // in this example, output tensor is always the last tensor in the graph
    struct ggml_tensor * result = result0;
    // struct ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    float * result_data = (float *)malloc(ggml_nbytes(result));
    // because the tensor data is stored in device buffer, we need to copy it back to RAM
    ggml_backend_tensor_get(result, result_data, 0, ggml_nbytes(result));
    const std::string bin_file = "mrope_2d_" + backend_name +".bin";
    std::ofstream outFile(bin_file, std::ios::binary);

    if (outFile.is_open()) {
        outFile.write(reinterpret_cast<const char*>(result_data), ggml_nbytes(result));
        outFile.close();
        std::cout << "Data successfully written to " + bin_file << std::endl;
    } else {
        std::cerr << "Error opening file!" << std::endl;
    }

    free(result_data);
    // 11. Free memory and exit
    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
}

static void debug_dump_img_embed(struct llava_context * ctx_llava) {
    int n_embd  = llama_n_embd(llama_get_model(ctx_llava->ctx_llama));
    int ne = n_embd * 4;
    float vals[56 * 56 * 3];
    // float embd[ne];
    std::vector<float> embd;
    embd.resize(ne);

    for (int i = 0; i < 56*56; i++)
    {
        for (int c = 0; c < 3; c++)
            vals[i * 3 + c] = (float)(i % (56 * 56)) / (56*56);
    }

    clip_encode_float_image(ctx_llava->ctx_clip, 16, vals, 56, 56, embd.data());

    std::ofstream outFile("img_embed.bin", std::ios::binary);
    if (outFile.is_open()) {
        outFile.write(reinterpret_cast<const char*>(embd.data()), ne * sizeof(float));

        outFile.close();
        std::cout << "Data successfully written to mrope.bin" << std::endl;
    } else {
        std::cerr << "Error opening file!" << std::endl;
    }
}

#endif


int main(int argc, char ** argv) {
    ggml_time_init();

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LLAVA, print_usage)) {
        return 1;
    }

    common_init();

    if (params.mmproj.empty() || (params.image.empty() && !prompt_contains_image(params.prompt))) {
        print_usage(argc, argv);
        return 1;
    }

    auto * model = llava_init(&params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to init llava model\n", __func__);
        return 1;
    }

    if (prompt_contains_image(params.prompt)) {
        auto * ctx_llava = llava_init_context(&params, model);

        auto * image_embed = load_image(ctx_llava, &params, "");

        // process the prompt
        process_prompt(ctx_llava, image_embed, &params, params.prompt);

        llama_perf_context_print(ctx_llava->ctx_llama);
        llava_image_embed_free(image_embed);
        ctx_llava->model = NULL;
        llava_free(ctx_llava);
#ifndef NDEBUG
    } else if (params.image[0].empty()) {
        auto ctx_llava = llava_init_context(&params, model);

        debug_test_mrope_2d();
        debug_dump_img_embed(ctx_llava);

        llama_perf_context_print(ctx_llava->ctx_llama);
        ctx_llava->model = NULL;
        llava_free(ctx_llava);
#endif
    } else {
        for (auto & image : params.image) {
            auto * ctx_llava = llava_init_context(&params, model);

            auto * image_embed = load_image(ctx_llava, &params, image);
            if (!image_embed) {
                LOG_ERR("%s: failed to load image %s. Terminating\n\n", __func__, image.c_str());
                return 1;
            }

            // process the prompt
            process_prompt(ctx_llava, image_embed, &params, params.prompt);

            llama_perf_context_print(ctx_llava->ctx_llama);
            llava_image_embed_free(image_embed);
            ctx_llava->model = NULL;
            llava_free(ctx_llava);
        }
    }

    llama_model_free(model);

    return 0;
}
