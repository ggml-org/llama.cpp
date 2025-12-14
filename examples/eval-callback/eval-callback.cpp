#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

/**
 * This the arbitrary data which will be passed to each callback.
 * Later on we can for example add operation or tensor name filter from the CLI arg, or a file descriptor to dump the tensor.
 */
struct callback_data {
    std::vector<uint8_t> data;
};

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static inline float ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    union {
        float f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    return u.f;
}

static float ggml_get_float_value(const uint8_t * data, ggml_type type, const size_t * nb, size_t i0, size_t i1, size_t i2, size_t i3) {
    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    float v;
    if (type == GGML_TYPE_F16) {
        v = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[i]);
    } else if (type == GGML_TYPE_F32) {
        v = *(const float *) &data[i];
    } else if (type == GGML_TYPE_I64) {
        v = (float) *(const int64_t *) &data[i];
    } else if (type == GGML_TYPE_I32) {
        v = (float) *(const int32_t *) &data[i];
    } else if (type == GGML_TYPE_I16) {
        v = (float) *(const int16_t *) &data[i];
    } else if (type == GGML_TYPE_I8) {
        v = (float) *(const int8_t *) &data[i];
    } else if (type == GGML_TYPE_BF16) {
        v = ggml_compute_bf16_to_fp32(*(const ggml_bf16_t *) &data[i]);
    } else {
        GGML_ABORT("fatal error");
    }
    return v;
}

static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    const float v = ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    sum += v;
                }
            }
        }
    }
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                LOG("                                      ..., \n");
                i2 = ne[2] - n;
            }
            LOG("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    LOG("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                LOG("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        LOG("..., ");
                        i0 = ne[0] - n;
                    }
                    const float v = ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    LOG("%12.4f", v);
                    if (i0 < ne[0] - 1) LOG(", ");
                }
                LOG("],\n");
            }
            LOG("                                      ],\n");
        }
        LOG("                                     ]\n");
        LOG("                                     sum = %f\n", sum);
    }

    // TODO: make this abort configurable/optional?
    if (std::isnan(sum)) {
        LOG_ERR("encountered NaN - aborting\n");
        exit(0);
    }
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        return true; // Always retrieve data
    }

    char src1_str[128] = {0};
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    LOG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
         t->name, ggml_type_name(t->type), ggml_op_desc(t),
         src0->name, ggml_ne_string(src0).c_str(),
         src1 ? src1_str : "",
         ggml_ne_string(t).c_str());


    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if (!ggml_is_quantized(t->type)) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
        ggml_print_tensor(data, t->type, t->ne, t->nb, 3);
    }

    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return false;
    }

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    callback_data cb_data;

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ggml_debug;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
