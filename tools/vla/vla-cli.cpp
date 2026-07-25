#include "vla.h"

#include "common.h"
#include "llama.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

struct cli_params {
    std::string model_path;
    std::string mmproj_path;
    std::string vla_path;
    std::string prompt;
    std::vector<std::string> images;
    std::string state_path;
    std::string noise_path;
    std::string output_path;
    int32_t embodiment_id = 0;
    int32_t n_ctx = 4096;
    int32_t n_batch = 512;
    bool use_gpu = true;
    bool mmproj_use_gpu = true;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
            "Usage: %s -m model.gguf --mmproj mmproj.gguf --vla vla.gguf \\\n"
            "  --image image.jpg --prompt '<image>\\n...' --state state.bin [options]\n\n"
            "Options:\n"
            "  --noise FILE          optional f32 [horizon, action_dim] noise\n"
            "  --embodiment-id N      embodiment index (default: 0)\n"
            "  --output FILE          write actions as raw f32; otherwise print them\n"
            "  --ctx-size N           libllama context size (default: 4096)\n"
            "  --batch-size N         prompt batch size (default: 512)\n"
            "  --cpu                  run libllama and VLA on CPU\n"
            "  --no-mmproj-offload    run mtmd on CPU\n",
            argv0);
}

bool parse_args(int argc, char ** argv, cli_params & params) {
    auto value = [&](int & i) -> const char * {
        if (i + 1 >= argc) {
            return nullptr;
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        const char * val = nullptr;
        if ((std::strcmp(arg, "-m") == 0 || std::strcmp(arg, "--model") == 0) &&
                (val = value(i))) {
            params.model_path = val;
        } else if (std::strcmp(arg, "--mmproj") == 0 && (val = value(i))) {
            params.mmproj_path = val;
        } else if (std::strcmp(arg, "--vla") == 0 && (val = value(i))) {
            params.vla_path = val;
        } else if ((std::strcmp(arg, "-p") == 0 || std::strcmp(arg, "--prompt") == 0) &&
                (val = value(i))) {
            params.prompt = val;
        } else if (std::strcmp(arg, "--image") == 0 && (val = value(i))) {
            params.images.emplace_back(val);
        } else if (std::strcmp(arg, "--state") == 0 && (val = value(i))) {
            params.state_path = val;
        } else if (std::strcmp(arg, "--noise") == 0 && (val = value(i))) {
            params.noise_path = val;
        } else if (std::strcmp(arg, "--output") == 0 && (val = value(i))) {
            params.output_path = val;
        } else if (std::strcmp(arg, "--embodiment-id") == 0 && (val = value(i))) {
            params.embodiment_id = std::atoi(val);
        } else if (std::strcmp(arg, "--ctx-size") == 0 && (val = value(i))) {
            params.n_ctx = std::atoi(val);
        } else if (std::strcmp(arg, "--batch-size") == 0 && (val = value(i))) {
            params.n_batch = std::atoi(val);
        } else if (std::strcmp(arg, "--cpu") == 0) {
            params.use_gpu = false;
        } else if (std::strcmp(arg, "--no-mmproj-offload") == 0) {
            params.mmproj_use_gpu = false;
        } else if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            return false;
        } else {
            std::fprintf(stderr, "unknown or incomplete argument: %s\n", arg);
            return false;
        }
    }

    return !params.model_path.empty() && !params.mmproj_path.empty() &&
        !params.vla_path.empty() && !params.prompt.empty() &&
        !params.state_path.empty() && params.n_ctx > 0 && params.n_batch > 0;
}

bool read_f32(const std::string & path, std::vector<float> & data) {
    FILE * file = std::fopen(path.c_str(), "rb");
    if (!file) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return false;
    }
    std::fseek(file, 0, SEEK_END);
    const long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    if (size <= 0 || size % (long) sizeof(float) != 0) {
        std::fprintf(stderr, "invalid f32 file size for %s\n", path.c_str());
        std::fclose(file);
        return false;
    }
    data.resize((size_t) size / sizeof(float));
    const bool ok = std::fread(data.data(), 1, (size_t) size, file) == (size_t) size;
    std::fclose(file);
    return ok;
}

bool write_f32(const std::string & path, const std::vector<float> & data) {
    FILE * file = std::fopen(path.c_str(), "wb");
    if (!file) {
        return false;
    }
    const size_t size = data.size() * sizeof(float);
    const bool ok = std::fwrite(data.data(), 1, size, file) == size;
    std::fclose(file);
    return ok;
}

struct hidden_collector {
    llama_context * lctx;
    int64_t n_embd;
    std::vector<float> data;
};

int32_t collect_hidden(llama_batch batch, void * user_data) {
    auto * collector = static_cast<hidden_collector *>(user_data);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const float * embedding = llama_get_embeddings_ith(collector->lctx, i);
        if (!embedding) {
            std::fprintf(stderr, "failed to get hidden state for batch token %d\n", i);
            return 1;
        }
        collector->data.insert(
                collector->data.end(), embedding, embedding + collector->n_embd);
    }
    return 0;
}

int32_t decode_text_chunk(
        llama_context * lctx,
        const mtmd_input_chunk * chunk,
        llama_pos & n_past,
        int32_t n_batch,
        hidden_collector & collector) {
    size_t n_tokens = 0;
    const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    size_t offset = 0;
    int32_t result = 0;
    while (offset < n_tokens) {
        batch.n_tokens = 0;
        while (offset < n_tokens && batch.n_tokens < n_batch) {
            const int32_t i = batch.n_tokens++;
            batch.token[i] = tokens[offset++];
            batch.pos[i] = n_past++;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
        }
        result = llama_decode(lctx, batch);
        if (result != 0 || collect_hidden(batch, &collector) != 0) {
            result = result != 0 ? result : 1;
            break;
        }
    }
    llama_batch_free(batch);
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    cli_params cli;
    if (!parse_args(argc, argv, cli)) {
        usage(argv[0]);
        return 1;
    }

    common_init();
    common_params params;
    params.model.path = cli.model_path;
    params.n_ctx = cli.n_ctx;
    params.n_batch = cli.n_batch;
    params.n_ubatch = cli.n_batch;
    params.n_gpu_layers = cli.use_gpu ? -2 : 0;
    params.embedding = true;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    params.warmup = false;

    common_init_result_ptr llama_init = common_init_from_params(params);
    if (!llama_init || !llama_init->model() || !llama_init->context()) {
        std::fprintf(stderr, "failed to initialize libllama\n");
        return 1;
    }
    llama_model * model = llama_init->model();
    llama_context * lctx = llama_init->context();

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = cli.mmproj_use_gpu;
    mparams.n_threads = params.cpuparams.n_threads;
    mparams.warmup = false;
    mtmd::context_ptr mtmd_ctx(mtmd_init_from_file(cli.mmproj_path.c_str(), model, mparams));
    if (!mtmd_ctx) {
        std::fprintf(stderr, "failed to initialize mtmd\n");
        return 1;
    }

    vla_context_params vparams = vla_context_params_default();
    vparams.use_gpu = cli.use_gpu;
    vparams.n_threads = params.cpuparams.n_threads;
    vla_context * raw_vla = vla_init_from_file(cli.vla_path.c_str(), model, vparams);
    if (!raw_vla) {
        return 1;
    }
    std::unique_ptr<vla_context, decltype(&vla_free)> vla_ctx(raw_vla, vla_free);

    std::vector<float> state;
    std::vector<float> noise;
    if (!read_f32(cli.state_path, state) ||
            (!cli.noise_path.empty() && !read_f32(cli.noise_path, noise))) {
        return 1;
    }

    mtmd::bitmaps bitmaps;
    std::vector<mtmd_helper::video_ptr> videos;
    for (const std::string & image : cli.images) {
        auto media = mtmd_helper_bitmap_init_from_file(mtmd_ctx.get(), image.c_str(), false);
        if (!media.bitmap) {
            std::fprintf(stderr, "failed to load media %s\n", image.c_str());
            return 1;
        }
        bitmaps.entries.emplace_back(media.bitmap);
        if (media.video_ctx) {
            videos.emplace_back(media.video_ctx);
        }
    }

    mtmd_input_text text = {
        /*.text          =*/ cli.prompt.data(),
        /*.text_len      =*/ cli.prompt.size(),
        /*.add_special   =*/ true,
        /*.parse_special =*/ true,
    };
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmap_ptrs = bitmaps.c_ptr();
    if (mtmd_tokenize(
                mtmd_ctx.get(),
                chunks.ptr.get(),
                &text,
                bitmap_ptrs.data(),
                bitmap_ptrs.size()) != 0) {
        std::fprintf(stderr, "failed to tokenize multimodal prompt\n");
        return 1;
    }

    hidden_collector collector = {
        /*.lctx   =*/ lctx,
        /*.n_embd =*/ llama_model_n_embd_out(model),
        /*.data   =*/ {},
    };
    llama_pos n_past = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
        const mtmd_input_chunk * chunk = chunks[i];
        const mtmd_input_chunk_type type = mtmd_input_chunk_get_type(chunk);
        int32_t result = 0;
        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            result = decode_text_chunk(lctx, chunk, n_past, cli.n_batch, collector);
        } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE ||
                type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            result = mtmd_encode_chunk(mtmd_ctx.get(), chunk);
            if (result == 0) {
                llama_pos new_n_past = n_past;
                result = mtmd_helper_decode_image_chunk(
                        mtmd_ctx.get(),
                        lctx,
                        chunk,
                        mtmd_get_output_embd(mtmd_ctx.get()),
                        n_past,
                        0,
                        cli.n_batch,
                        &new_n_past,
                        true,
                        collect_hidden,
                        &collector);
                n_past = new_n_past;
            }
        } else {
            result = 1;
        }
        if (result != 0) {
            std::fprintf(stderr, "failed to evaluate prompt chunk %zu\n", i);
            return 1;
        }
    }

    const int64_t n_embd = vla_conditioning_dim(vla_ctx.get());
    if ((int64_t) collector.data.size() % n_embd != 0) {
        std::fprintf(stderr, "hidden-state buffer has an invalid size\n");
        return 1;
    }
    const int64_t n_tokens = (int64_t) collector.data.size() / n_embd;
    std::vector<float> actions(
            (size_t) vla_action_horizon(vla_ctx.get()) * vla_action_dim(vla_ctx.get()));
    vla_input input = {
        /*.embeddings    =*/ collector.data.data(),
        /*.n_tokens      =*/ n_tokens,
        /*.n_embd        =*/ n_embd,
        /*.state         =*/ state.data(),
        /*.n_state       =*/ (int64_t) state.size(),
        /*.noise         =*/ noise.empty() ? nullptr : noise.data(),
        /*.n_noise       =*/ (int64_t) noise.size(),
        /*.embodiment_id =*/ cli.embodiment_id,
    };
    vla_output output = {
        /*.actions  =*/ actions.data(),
        /*.capacity =*/ (int64_t) actions.size(),
    };
    if (!vla_predict(vla_ctx.get(), &input, &output)) {
        return 1;
    }

    if (!cli.output_path.empty()) {
        if (!write_f32(cli.output_path, actions)) {
            std::fprintf(stderr, "failed to write %s\n", cli.output_path.c_str());
            return 1;
        }
    } else {
        const int64_t action_dim = vla_action_dim(vla_ctx.get());
        for (int64_t t = 0; t < vla_action_horizon(vla_ctx.get()); ++t) {
            std::printf("%lld", (long long) t);
            for (int64_t d = 0; d < action_dim; ++d) {
                std::printf(" %.9g", actions[(size_t) t * action_dim + d]);
            }
            std::printf("\n");
        }
    }
    return 0;
}
