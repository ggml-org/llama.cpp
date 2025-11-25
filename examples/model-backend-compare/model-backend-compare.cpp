#include "arg.h"
#include "common.h"
#include "log.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <vector>
#include <cstdint>
#include <unordered_map>
#include <string>
#include <sstream>
#include <cstring>
#include <cmath>

namespace {
constexpr double nmse_threshold = 1e-2;

struct callback_data {
    std::vector<uint8_t> data;
    std::vector<float> device_results;
    std::unordered_map<std::string, std::vector<float>> cpu_results;
};

bool gather = true;

// normalized mean squared error = mse(a, b) / mse(a, 0)
double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

void ggml_print_tensor(const ggml_tensor * t, const std::vector<float> data, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    const float v = data[i3 * t->ne[2] * t->ne[1] * t->ne[0] + i2 * t->ne[1] * t->ne[0] + i1 * t->ne[0] + i0];
                    sum += v;
                }
            }
        }
    }
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        LOG("                                     [\n");
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            if (i2 == n && t->ne[2] > 2*n) {
                LOG("                                      ..., \n");
                i2 = t->ne[2] - n;
            }
            LOG("                                      [\n");
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                if (i1 == n && t->ne[1] > 2*n) {
                    LOG("                                       ..., \n");
                    i1 = t->ne[1] - n;
                }
                LOG("                                       [");
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    if (i0 == n && t->ne[0] > 2*n) {
                        LOG("..., ");
                        i0 = t->ne[0] - n;
                    }
                    const float v = data[i3 * t->ne[2] * t->ne[1] * t->ne[0] + i2 * t->ne[1] * t->ne[0] + i1 * t->ne[0] + i0];
                    LOG("%12.4f", v);
                    if (i0 < t->ne[0] - 1) LOG(", ");
                }
                LOG("],\n");
            }
            LOG("                                      ],\n");
        }
        LOG("                                     ]\n");
        LOG("                                     sum = %f\n", sum);
    }

    if (std::isnan(sum)) {
        LOG_ERR("encountered NaN - aborting\n");
        exit(0);
    }
}

inline float ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    union {
        float f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    return u.f;
}

float to_float(const uint8_t * ptr, ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return *(const float *)ptr;
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*(const ggml_fp16_t *)ptr);
        case GGML_TYPE_BF16:
            return ggml_compute_bf16_to_fp32(*(const ggml_bf16_t *)ptr);
        case GGML_TYPE_I8:
            return static_cast<float>(*(const int8_t *)ptr);
        case GGML_TYPE_I16:
            return static_cast<float>(*(const int16_t *)ptr);
        case GGML_TYPE_I32:
            return static_cast<float>(*(const int32_t *)ptr);
        case GGML_TYPE_I64:
            return static_cast<float>(*(const int64_t *)ptr);
        default:
            GGML_ABORT("unsupported ggml_type %d in to_float", type);
    }
    return 0.0f;
}

void tensor_to_float_array(const ggml_tensor * t, const void * data, std::vector<float> & out) {
    const size_t n_elements = ggml_nelements(t);
    out.resize(n_elements);

    // convert to float
    size_t idx = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                if (!ggml_is_quantized(t->type)) {
                    for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                        const uint8_t * ptr = ((const uint8_t *)data) + i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + i0 * t->nb[0];

                        out[idx] = to_float(ptr, t->type);
                        idx++;
                    }
                } else {
                    GGML_ABORT("quantized types are not supported in tensor_to_float_array");
                }
            }
        }
    }
}

bool tensor_is_empty(ggml_tensor * node) {
    return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
}

std::string remove_device_from_name(const std::string & name) {
    // Remove prefix and suffix
    // Example: Vulkan0#inp_embd#0 -> inp_embd
    size_t start = name.find_first_of('#');
    size_t end = name.find_last_of('#');
    if (start != std::string::npos && end != std::string::npos &&
        end > start) {
        return name.substr(start + 1, end - start - 1);
    }
    return name;
}

std::string tensor_name(ggml_tensor * t) {
    const std::string tname(t->name, strnlen(t->name, GGML_MAX_NAME));

    std::stringstream ss;
    ss << tname << "[";
    // Get last source
    size_t last_src = 0;
    for (size_t i = 0; i < GGML_MAX_SRC; i++) {
        if (t->src[i]) {
            last_src = i;
        }
    }
    for (size_t i = 0; i < GGML_MAX_SRC; i++) {
        if (t->src[i]) {
            const std::string src_name(t->src[i]->name, strnlen(t->src[i]->name, GGML_MAX_NAME));
            ss << remove_device_from_name(src_name);
            if (i < last_src) {
                ss << ", ";
            }
        }
    }
    ss << "]";
    return ss.str();
}

bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    if (ask || tensor_is_empty(t)) {
        return true; // Always retrieve data
    }

    const std::string name = tensor_name(t);

    if (gather) {
        // CPU data should be host-visible
        GGML_ASSERT(ggml_backend_buffer_is_host(t->buffer));

        // Make sure this tensor does not exist yet
        if (cb_data->cpu_results.find(name) != cb_data->cpu_results.end())
        {
            LOG_ERR("%s : tensor '%s' already exists in CPU reference data\n", __func__, name.c_str());
            GGML_ABORT("fatal error");
        }

        std::vector<float>& result = cb_data->cpu_results[name];

        // LOG("gathering CPU reference data for tensor '%s'\n", name.c_str());
        // for (size_t i = 0; i < GGML_MAX_DIMS; i++) {
        //     LOG("  ne[%zu] = %lld\n", i, t->ne[i]);
        // }
        // for (size_t i = 0; i < GGML_MAX_SRC; i++) {
        //     if (t->src[i]) {
        //         const std::string src_name(t->src[i]->name, strnlen(t->src[i]->name, GGML_MAX_NAME));
        //         LOG("  src[%zu] = %s\n", i, src_name.c_str());
        //     }
        // }

        tensor_to_float_array(t, t->data, result);

        return true;
    }

    // Compare with CPU data if available
    auto it = cb_data->cpu_results.find(name);
    if (it == cb_data->cpu_results.end()) {
        LOG_ERR("no CPU reference data for tensor '%s'\n", name.c_str());
        return true;
    }

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const size_t n_bytes = ggml_nbytes(t);

    const uint8_t * data;

    if (!is_host) {
        if (cb_data->data.size() < n_bytes) {
            cb_data->data.resize(n_bytes);
        }

        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
        data = cb_data->data.data();
    } else {
        data = (const uint8_t *) t->data;
    }

    tensor_to_float_array(t, data, cb_data->device_results);

    const std::vector<float>& ref_data = it->second;

    double error = nmse(ref_data.data(), cb_data->device_results.data(), ref_data.size());

    if (error > nmse_threshold) {
        LOG_ERR("nmse = %.12f tensor '%s' op=%s\n", error, name.c_str(), ggml_op_name(t->op));
        LOG_ERR("  ne: ");
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            LOG_ERR("%ld ", t->ne[i]);
        }
        LOG_ERR("\n  nb: ");
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            LOG_ERR("%zu ", t->nb[i]);
        }
        LOG_ERR("\n\n");
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (t->src[i]) {
                const std::string src_name(t->src[i]->name, strnlen(t->src[i]->name, GGML_MAX_NAME));
                LOG_ERR("  src%d: %s\n", i, src_name.c_str());
            }
        }

        LOG_ERR("CPU reference data for tensor '%s':\n", name.c_str());
        ggml_print_tensor(t, ref_data, 2);

        LOG_ERR("Device data for tensor '%s':\n", name.c_str());
        ggml_print_tensor(t, cb_data->device_results, 2);
        return false;
    } else {
        LOG("nmse = %.12f tensor '%s' op = %s\n", error, name.c_str(), ggml_op_name(t->op));
    }

    return true;
}

bool run(llama_context * ctx, const common_params & params) {
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
} // namespace

int main(int argc, char ** argv) {
    callback_data cb_data;

    common_params params;
    params.prompt = "The quick brown fox";
    params.sampling.seed = 1234;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();

    llama_backend_init();

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ggml_debug;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    params.split_mode = LLAMA_SPLIT_MODE_NONE;

    const size_t n_dev = ggml_backend_dev_count();

    for (size_t i = 0; i < n_dev * 2; i++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i % ggml_backend_dev_count());

        // Run CPU-only first to gather reference results
        if ((i < n_dev && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) ||
            (i >= n_dev && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU)) {
            continue;
        }

        params.devices.clear();
        params.devices.push_back(device);

        if (i < n_dev) {
            LOG_INF("=== Running on device %zu (gathering reference results) ===\n", i);
            gather = true;
        } else {
            LOG_INF("=== Running on device %zu ===\n", i - n_dev);
            gather = false;
        }

        // init
        common_init_result llama_init = common_init_from_params(params);

        llama_model * model = llama_init.model.get();
        llama_context * ctx = llama_init.context.get();

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
    }

    return 0;
}