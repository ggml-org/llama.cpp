#include "fit.h"

#include "log.h"

#include "../src/llama-ext.h"

#include <array>
#include <cassert>
#include <stdexcept>
#include <cinttypes>
#include <set>
#include <string>
#include <vector>

// this enum is only used in llama_params_fit_impl but needs to be defined outside of it to fix a Windows compilation issue
// enum to identify part of a layer for distributing its tensors:
enum common_layer_fraction_t {
    LAYER_FRACTION_NONE = 0, // nothing
    LAYER_FRACTION_ATTN = 1, // attention
    LAYER_FRACTION_UP   = 2, // attention + up
    LAYER_FRACTION_GATE = 3, // attention + up + gate
    LAYER_FRACTION_MOE  = 4, // everything but sparse MoE weights
};

class common_params_fit_exception : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

static std::vector<llama_device_memory_data> common_get_device_memory_data(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc  = true;
    mparams_copy.use_mmap  = false;
    mparams_copy.use_mlock = false;

    llama_model * model = llama_model_load_from_file(path_model, mparams_copy);
    if (model == nullptr) {
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to load model");
    }

    llama_context * ctx = llama_init_from_model(model, *cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to create llama_context from model");
    }

    const size_t nd = llama_model_n_devices(model);
    std::vector<llama_device_memory_data> ret(nd + 1);

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    for (const auto & [buft, mb] : memory_breakdown) {
        if (ggml_backend_buft_is_host(buft)) {
            ret.back().mb.model   += mb.model;
            ret.back().mb.context += mb.context;
            ret.back().mb.compute += mb.compute;
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            continue;
        }
        for (size_t i = 0; i < nd; i++) {
            if (dev == llama_model_get_device(model, i)) {
                ret[i].mb.model   += mb.model;
                ret[i].mb.context += mb.context;
                ret[i].mb.compute += mb.compute;
                break;
            }
        }
    }

    {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error("no CPU backend found");
        }
        size_t free;
        size_t total;
        ggml_backend_dev_memory(cpu_dev, &free, &total);
        ret.back().free  = free;
        ret.back().total = total;
    }
    for (size_t i = 0; i < nd; i++) {
        ggml_backend_dev_t dev = llama_model_get_device(model, i);

        size_t free;
        size_t total;
        ggml_backend_dev_memory(dev, &free, &total);

        // Some non-GPU accelerator backends, such as BLAS, report 0/0 and rely on
        // the host-memory fallback. For GPU-like backends, keep 0/0 so --fit does
        // not assign anything to a device with an unknown memory budget.
        if (free == 0 && total == 0) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                LOG_WRN("%s: device %s did not report memory; --fit will not use it\n",
                        __func__, ggml_backend_dev_name(dev));
            } else {
                free  = ret.back().free;
                total = ret.back().total;
            }
        }
        ret[i].free  = free;
        ret[i].total = total;
    }

    devs.clear();
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devs.push_back(llama_model_get_device(model, i));
    }

    hp_ngl         = llama_model_n_layer(model);
    hp_n_ctx_train = llama_model_n_ctx_train(model);
    hp_n_expert    = llama_model_n_expert(model);

    common_memory_breakdown_print(ctx);

    llama_free(ctx);
    llama_model_free(model);
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);

    return ret;
}

static void common_params_fit_impl(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        float * tensor_split, struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins_s, uint32_t n_ctx_min, enum ggml_log_level log_level) {
    if (mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        throw common_params_fit_exception("llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort");
    }
    constexpr int64_t MiB = 1024*1024;
    typedef std::vector<llama_device_memory_data> dmds_t;
    const llama_model_params default_mparams = llama_model_default_params();

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    // step 1: get data for default parameters and check whether any changes are necessary in the first place

    LOG_TRC("%s: getting device memory data for initial parameters:\n", __func__);
    const dmds_t dmds_full = common_get_device_memory_data(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
    const size_t nd = devs.size(); // number of devices

    std::vector<int64_t> margins; // this function uses int64_t rather than size_t for memory sizes to more conveniently handle deficits
    margins.reserve(nd);
    if (nd == 0) {
        margins.push_back(margins_s[0]);
    } else {
        for (size_t id = 0; id < nd; id++) {
            margins.push_back(margins_s[id]);
        }
    }

    std::vector<std::string> dev_names;
    {
        dev_names.reserve(nd);
        size_t max_length = 0;
        for (const auto & dev : devs) {
            std::string name = ggml_backend_dev_name(dev);
            name += " (";
            name += ggml_backend_dev_description(dev);
            name += ")";
            dev_names.push_back(name);
            max_length = std::max(max_length, name.length());
        }
        for (std::string & dn : dev_names) {
            dn.insert(dn.end(), max_length - dn.length(), ' ');
        }
    }

    int64_t sum_free            = 0;
    int64_t sum_projected_free  = 0;
    int64_t sum_projected_used  = 0;
    int64_t sum_projected_model = 0;
    std::vector<int64_t> projected_free_per_device;
    projected_free_per_device.reserve(nd);

    if (nd == 0) {
        sum_projected_used = dmds_full.back().mb.total();
        sum_free           = dmds_full.back().total;
        sum_projected_free = sum_free - sum_projected_used;
        LOG_INF("%s: projected to use %" PRId64 " MiB of host memory vs. %" PRId64 " MiB of total host memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (sum_projected_free >= margins[0]) {
            LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of system memory, no changes needed\n",
                __func__, sum_projected_free/MiB, margins[0]/MiB);
            return;
        }
    } else {
        if (nd > 1) {
            LOG_TRC("%s: projected memory use with initial parameters [MiB]:\n", __func__);
        }
        for (size_t id = 0; id < nd; id++) {
            const llama_device_memory_data & dmd = dmds_full[id];

            const int64_t projected_used = dmd.mb.total();
            const int64_t projected_free = dmd.free - projected_used;
            projected_free_per_device.push_back(projected_free);

            sum_free            += dmd.free;
            sum_projected_used  += projected_used;
            sum_projected_free  += projected_free;
            sum_projected_model += dmd.mb.model;

            if (nd > 1) {
                LOG_TRC("%s:   - %s: %6" PRId64 " total, %6" PRId64 " used, %6" PRId64 " free vs. target of %6" PRId64 "\n",
                    __func__, dev_names[id].c_str(), dmd.total/MiB, projected_used/MiB, projected_free/MiB, margins[id]/MiB);
            }
        }
        assert(sum_free >= 0 && sum_projected_used >= 0);
        LOG_TRC("%s: projected to use %" PRId64 " MiB of device memory vs. %" PRId64 " MiB of free device memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (nd == 1) {
            if (projected_free_per_device[0] >= margins[0]) {
                LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of free device memory, no changes needed\n",
                    __func__, projected_free_per_device[0]/MiB, margins[0]/MiB);
                return;
            }
        } else {
            bool changes_needed = false;
            for (size_t id = 0; id < nd; id++) {
                if (projected_free_per_device[id] < margins[id]) {
                    changes_needed = true;
                    break;
                }
            }
            if (!changes_needed) {
                LOG_TRC("%s: targets for free memory can be met on all devices, no changes needed\n", __func__);
                return;
            }
        }
    }

    // step 2: try reducing memory use by reducing the context size

    {
        int64_t global_surplus = sum_projected_free;
        if (nd == 0) {
            global_surplus -= margins[0];
        } else {
            for (size_t id = 0; id < nd; id++) {
                global_surplus -= margins[id];
            }
        }
        if (global_surplus < 0) {
            if (nd <= 1) {
                LOG_TRC("%s: cannot meet free memory target of %" PRId64 " MiB, need to reduce device memory by %" PRId64 " MiB\n",
                    __func__, margins[0]/MiB, -global_surplus/MiB);
            } else {
                LOG_TRC(
                    "%s: cannot meet free memory targets on all devices, need to use %" PRId64 " MiB less in total\n",
                    __func__, -global_surplus/MiB);
            }
            if (cparams->n_ctx == 0) {
                if (hp_nct > n_ctx_min) {
                    int64_t sum_used_target = sum_free;
                    if (nd == 0) {
                        sum_used_target -= margins[0];
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_used_target -= margins[id];
                        }
                    }
                    if (nd > 1) {
                        // for multiple devices we need to be more conservative in terms of how much context we think can fit:
                        //   - for dense models only whole layers can be assigned to devices
                        //   - for MoE models only whole tensors can be assigned to devices, which we estimate to be <= 1/3 of a layer
                        //   - on average we expect a waste of 0.5 layers/tensors per device
                        //   - use slightly more than the expected average for nd devices to be safe
                        const int64_t model_per_layer = sum_projected_model / std::min(uint32_t(mparams->n_gpu_layers), hp_ngl);
                        sum_used_target -= (nd + 1) * model_per_layer / (hp_nex == 0 ? 2 : 6);
                    }

                    int64_t sum_projected_used_min_ctx = 0;
                    cparams->n_ctx = n_ctx_min;
                    const dmds_t dmds_min_ctx = common_get_device_memory_data(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                    if (nd == 0) {
                        sum_projected_used_min_ctx = dmds_min_ctx.back().mb.total();
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_projected_used_min_ctx += dmds_min_ctx[id].mb.total();
                        }
                    }
                    if (sum_used_target > sum_projected_used_min_ctx) {
                        // linear interpolation between minimum and maximum context size:
                        cparams->n_ctx += (hp_nct - n_ctx_min) * (sum_used_target - sum_projected_used_min_ctx)
                            / (sum_projected_used - sum_projected_used_min_ctx);
                        cparams->n_ctx = std::max(cparams->n_ctx - cparams->n_ctx % 256, n_ctx_min); // round down context for CUDA backend

                        const int64_t bytes_per_ctx = (sum_projected_used - sum_projected_used_min_ctx) / (hp_nct - n_ctx_min);
                        const int64_t memory_reduction = (hp_nct - cparams->n_ctx) * bytes_per_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, hp_nct, cparams->n_ctx, memory_reduction/MiB);
                        if (nd <= 1) {
                            LOG_TRC("%s: entire model can be fit by reducing context\n", __func__);
                            return;
                        }
                        LOG_TRC("%s: entire model should be fit across devices by reducing context\n", __func__);
                    } else {
                        const int64_t memory_reduction = sum_projected_used - sum_projected_used_min_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, hp_nct, cparams->n_ctx, memory_reduction/MiB);
                    }
                } else {
                    if (n_ctx_min == UINT32_MAX) {
                        LOG_TRC("%s: user has requested full context size of %" PRIu32 " -> no change\n", __func__, hp_nct);
                    } else {
                        LOG_TRC("%s: default model context size is %" PRIu32 " which is <= the min. context size of %" PRIu32 " -> no change\n",
                            __func__, hp_nct, n_ctx_min);
                    }
                }
            } else {
                LOG_TRC("%s: context size set by user to %" PRIu32 " -> no change\n", __func__, cparams->n_ctx);
            }
        }
    }
    if (nd == 0) {
        throw common_params_fit_exception("was unable to fit model into system memory by reducing context, abort");
    }

    if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
        throw common_params_fit_exception("n_gpu_layers already set by user to " + std::to_string(mparams->n_gpu_layers) + ", abort");
    }
    if (nd > 1) {
        if (!tensor_split) {
            throw common_params_fit_exception("did not provide a buffer to write the tensor_split to, abort");
        }
        if (mparams->tensor_split) {
            for (size_t id = 0; id < nd; id++) {
                if (mparams->tensor_split[id] != 0.0f) {
                    throw common_params_fit_exception("model_params::tensor_split already set by user, abort");
                }
            }
        }
        if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
            throw common_params_fit_exception("changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented, abort");
        }
    }
    if (!tensor_buft_overrides) {
        throw common_params_fit_exception("did not provide buffer to set tensor_buft_overrides, abort");
    }
    if (mparams->tensor_buft_overrides && (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft)) {
        throw common_params_fit_exception("model_params::tensor_buft_overrides already set by user, abort");
    }

    // step 3: iteratively fill the back to front with "dense" layers
    //   - for a dense model simply fill full layers, giving each device a contiguous slice of the model
    //   - for a MoE model, same as dense model but with all MoE tensors in system memory

    // utility function that returns a static C string matching the tensors for a specific layer index and layer fraction:
    auto get_overflow_pattern = [&](const size_t il, const common_layer_fraction_t lf) -> const char * {
        constexpr size_t n_strings = 1000;
        if (il >= n_strings) {
            throw std::runtime_error("at most " + std::to_string(n_strings) + " model layers are supported");
        }
        switch (lf) {
            case LAYER_FRACTION_ATTN: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|up|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_UP: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_GATE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_down.*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_MOE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|down|gate_up|gate)_(ch|)exps";
                }
                return patterns[il].c_str();
            }
            default:
                GGML_ABORT("fatal error");
        }
    };

    struct ngl_t {
        uint32_t n_layer = 0; // number of total layers
        uint32_t n_part  = 0; // number of partial layers, <= n_layer

        // for the first partial layer varying parts can overflow, all further layers use LAYER_FRACTION_MOE:
        common_layer_fraction_t overflow_type = LAYER_FRACTION_MOE;

        uint32_t n_full() const {
            assert(n_layer >= n_part);
            return n_layer - n_part;
        }
    };

    const size_t ntbo = llama_max_tensor_buft_overrides();

    // utility function to set n_gpu_layers and tensor_split
    auto set_ngl_tensor_split_tbo = [&](
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts,
            llama_model_params & mparams) {
        mparams.n_gpu_layers = 0;
        for (size_t id = 0; id < nd; id++) {
            mparams.n_gpu_layers += ngl_per_device[id].n_layer;
            if (nd > 1) {
                tensor_split[id] = ngl_per_device[id].n_layer;
            }
        }
        assert(uint32_t(mparams.n_gpu_layers) <= hp_ngl + 1);
        uint32_t il0 = hp_ngl + 1 - mparams.n_gpu_layers; // start index for tensor buft overrides

        mparams.tensor_split = tensor_split;

        size_t itbo = 0;
        for (size_t id = 0; id < nd; id++) {
            il0 += ngl_per_device[id].n_full();
            for (uint32_t il = il0; il < il0 + ngl_per_device[id].n_part; il++) {
                if (itbo + 1 >= ntbo) {
                    tensor_buft_overrides[itbo].pattern = nullptr;
                    tensor_buft_overrides[itbo].buft    = nullptr;
                    itbo++;
                    mparams.tensor_buft_overrides = tensor_buft_overrides;
                    throw common_params_fit_exception("llama_max_tensor_buft_overrides() == "
                        + std::to_string(ntbo) + " is insufficient for model");
                }
                tensor_buft_overrides[itbo].pattern = get_overflow_pattern(il, il == il0 ? ngl_per_device[id].overflow_type : LAYER_FRACTION_MOE);
                tensor_buft_overrides[itbo].buft = il == il0 ? overflow_bufts[id] : ggml_backend_cpu_buffer_type();
                itbo++;
            }
            il0 += ngl_per_device[id].n_part;
        }
        tensor_buft_overrides[itbo].pattern = nullptr;
        tensor_buft_overrides[itbo].buft    = nullptr;
        itbo++;
        mparams.tensor_buft_overrides = tensor_buft_overrides;
    };

    // utility function that returns the memory use per device for given numbers of layers per device
    auto get_memory_for_layers = [&](
            const char * func_name,
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts) -> std::vector<int64_t> {
        llama_model_params mparams_copy = *mparams;
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, mparams_copy);

        const dmds_t dmd_nl = common_get_device_memory_data(
            path_model, &mparams_copy, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

        LOG_TRC("%s: memory for test allocation by device:\n", func_name);
        for (size_t id = 0; id < nd; id++) {
            const ngl_t & n = ngl_per_device[id];
            LOG_TRC(
                "%s: id=%zu, n_layer=%2" PRIu32 ", n_part=%2" PRIu32 ", overflow_type=%d, mem=%6" PRId64 " MiB\n",
                func_name, id, n.n_layer, n.n_part, int(n.overflow_type), dmd_nl[id].mb.total()/MiB);
        }

        std::vector<int64_t> ret;
        ret.reserve(nd);
        for (size_t id = 0; id < nd; id++) {
            ret.push_back(dmd_nl[id].mb.total());
        }
        return ret;
    };

    int64_t global_surplus_cpu_moe = 0;
    if (hp_nex > 0) {
        const static std::string pattern_moe_all = "blk\\.\\d+\\.ffn_(up|down|gate_up|gate)_(ch|)exps"; // matches all MoE tensors
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        tensor_buft_overrides[0] = {pattern_moe_all.c_str(), cpu_buft};
        tensor_buft_overrides[1] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;

        LOG_TRC("%s: getting device memory data with all MoE tensors moved to system memory:\n", __func__);
        const dmds_t dmds_cpu_moe = common_get_device_memory_data(
            path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

        for (size_t id = 0; id < nd; id++) {
            global_surplus_cpu_moe += dmds_cpu_moe[id].free;
            global_surplus_cpu_moe -= int64_t(dmds_cpu_moe[id].mb.total()) + margins[id];
        }

        if (global_surplus_cpu_moe > 0) {
            LOG_TRC("%s: with only dense weights in device memory there is a total surplus of %" PRId64 " MiB\n",
                __func__, global_surplus_cpu_moe/MiB);
        } else {
            LOG_TRC("%s: with only dense weights in device memory there is still a total deficit of %" PRId64 " MiB\n",
                __func__, -global_surplus_cpu_moe/MiB);
        }

        // reset
        tensor_buft_overrides[0] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;
    }

    std::vector<int64_t> targets; // maximum acceptable memory use per device
    targets.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        targets.push_back(dmds_full[id].free - margins[id]);
        LOG_TRC("%s: id=%zu, target=%" PRId64 " MiB\n", __func__, id, targets[id]/MiB);
    }

    std::vector<ggml_backend_buffer_type_t> overflow_bufts; // which bufts the first partial layer of a device overflows to:
    overflow_bufts.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        overflow_bufts.push_back(ggml_backend_cpu_buffer_type());
    }

    std::vector<ngl_t> ngl_per_device(nd);
    std::vector<int64_t> mem = get_memory_for_layers(__func__, ngl_per_device, overflow_bufts);

    // optimize the number of layers per device using the method of false position:
    //   - ngl_per_device has 0 layers for each device, lower bound
    //   - try a "high" configuration where a device is given all unassigned layers
    //   - interpolate the memory use / layer between low and high linearly to get a guess where it meets our target
    //   - check memory use of our guess, replace either the low or high bound
    //   - once we only have a difference of a single layer, stop and return the lower bound that just barely still fits
    //   - the last device has the output layer, which cannot be a partial layer
    if (hp_nex == 0) {
        LOG_TRC("%s: filling dense layers back-to-front:\n", __func__);
    } else {
        LOG_TRC("%s: filling dense-only layers back-to-front:\n", __func__);
    }
    for (int id = nd - 1; id >= 0; id--) {
        uint32_t n_unassigned = hp_ngl + 1;
        for (size_t jd = id + 1; jd < nd; ++jd) {
            assert(n_unassigned >= ngl_per_device[jd].n_layer);
            n_unassigned -= ngl_per_device[jd].n_layer;
        }

        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        ngl_per_device_high[id].n_layer = n_unassigned;
        if (hp_nex > 0) {
            ngl_per_device_high[id].n_part = size_t(id) < nd - 1 ? ngl_per_device_high[id].n_layer : ngl_per_device_high[id].n_layer - 1;
        }
        if (ngl_per_device_high[id].n_layer > 0) {
            std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);
            if (mem_high[id] > targets[id]) {
                assert(ngl_per_device_high[id].n_layer > ngl_per_device[id].n_layer);
                uint32_t delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                LOG_TRC("%s: start filling device %" PRIu32 ", delta=%" PRIu32 "\n", __func__, id, delta);
                while (delta > 1) {
                    uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                    step_size = std::max(step_size, uint32_t(1));
                    step_size = std::min(step_size, delta - 1);

                    std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                    ngl_per_device_test[id].n_layer += step_size;
                    if (hp_nex) {
                        ngl_per_device_test[id].n_part += size_t(id) == nd - 1 && ngl_per_device_test[id].n_part == 0 ?
                            step_size - 1 : step_size; // the first layer is the output layer which must always be full
                    }
                    const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                    if (mem_test[id] <= targets[id]) {
                        ngl_per_device = ngl_per_device_test;
                        mem            = mem_test;
                        LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
                    } else {
                        ngl_per_device_high = ngl_per_device_test;
                        mem_high            = mem_test;
                        LOG_TRC("%s: set ngl_per_device_high[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device_high[id].n_layer);
                    }
                    delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                }
            } else {
                assert(ngl_per_device_high[id].n_layer == n_unassigned);
                ngl_per_device = ngl_per_device_high;
                mem            = mem_high;
                LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers, %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, mem[id]/MiB, projected_margin/MiB);
    }
    if (hp_nex == 0 || global_surplus_cpu_moe <= 0) {
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
        return;
    }

    // step 4: for a MoE model where all dense tensors fit,
    //     convert the dense-only layers in the back to full layers in the front until all devices are full
    // essentially the same procedure as for the dense-only layers except front-to-back
    // also, try fitting at least part of one more layer to reduce waste for "small" GPUs with e.g. 24 GiB VRAM

    size_t id_dense_start = nd;
    for (int id = nd - 1; id >= 0; id--) {
        if (ngl_per_device[id].n_layer > 0) {
            id_dense_start = id;
            continue;
        }
        break;
    }
    assert(id_dense_start < nd);

    LOG_TRC("%s: converting dense-only layers to full layers and filling them front-to-back with overflow to next device/system memory:\n", __func__);
    for (size_t id = 0; id <= id_dense_start && id_dense_start < nd; id++) {
        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        for (size_t jd = id_dense_start; jd < nd; jd++) {
            const uint32_t n_layer_move = jd < nd - 1 ? ngl_per_device_high[jd].n_layer : ngl_per_device_high[jd].n_layer - 1;
            ngl_per_device_high[id].n_layer += n_layer_move;
            ngl_per_device_high[jd].n_layer -= n_layer_move;
            ngl_per_device_high[jd].n_part = 0;
        }
        size_t id_dense_start_high = nd - 1;
        std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);

        if (mem_high[id] > targets[id]) {
            assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
            uint32_t delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            while (delta > 1) {
                uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                step_size = std::max(step_size, uint32_t(1));
                step_size = std::min(step_size, delta - 1);

                std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                size_t id_dense_start_test = id_dense_start;
                uint32_t n_converted_test = 0;
                for (;id_dense_start_test < nd; id_dense_start_test++) {
                    const uint32_t n_convert_jd = std::min(step_size - n_converted_test, ngl_per_device_test[id_dense_start_test].n_part);
                    ngl_per_device_test[id_dense_start_test].n_layer -= n_convert_jd;
                    ngl_per_device_test[id_dense_start_test].n_part -= n_convert_jd;
                    ngl_per_device_test[id].n_layer += n_convert_jd;
                    n_converted_test += n_convert_jd;

                    if (ngl_per_device_test[id_dense_start_test].n_part > 0) {
                        break;
                    }
                }
                const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                if (mem_test[id] <= targets[id]) {
                    ngl_per_device = ngl_per_device_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                } else {
                    ngl_per_device_high = ngl_per_device_test;
                    mem_high            = mem_test;
                    id_dense_start_high = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device_high[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start_high=%zu\n",
                        __func__, id, ngl_per_device_high[id].n_layer, ngl_per_device_high[id].n_part, id_dense_start_high);
                }
                assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
                delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            }
        } else {
            ngl_per_device = ngl_per_device_high;
            mem            = mem_high;
            id_dense_start = id_dense_start_high;
            LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
        }

        // try to fit at least part of one more layer
        if (ngl_per_device[id_dense_start].n_layer > (id < nd - 1 ? 0 : 1)) {
            std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
            size_t id_dense_start_test = id_dense_start;
            ngl_per_device_test[id_dense_start_test].n_layer--;
            ngl_per_device_test[id_dense_start_test].n_part--;
            ngl_per_device_test[id].n_layer++;
            ngl_per_device_test[id].n_part++;
            if (ngl_per_device_test[id_dense_start_test].n_part == 0) {
                id_dense_start_test++;
            }
            ngl_per_device_test[id].overflow_type = LAYER_FRACTION_UP;
            std::vector<ggml_backend_buffer_type_t> overflow_bufts_test = overflow_bufts;
            if (id < nd - 1) {
                overflow_bufts_test[id] = ggml_backend_dev_buffer_type(devs[id + 1]);
            }
            LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_UP\n", __func__);
            std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
            if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                ngl_per_device = ngl_per_device_test;
                overflow_bufts = overflow_bufts_test;
                mem            = mem_test;
                id_dense_start = id_dense_start_test;
                LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", UP), id_dense_start=%zu\n",
                    __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);

                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_GATE;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_GATE\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", GATE), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            } else {
                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_ATTN;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_ATTN\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", ATTN), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    // print info for devices that were not changed during the conversion from dense only to full layers:
    for (size_t id = id_dense_start + 1; id < nd; id++) {
        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
}

enum common_params_fit_status common_fit_params(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        float * tensor_split,
        llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins,
        uint32_t n_ctx_min,
        ggml_log_level log_level) {
    const int64_t t0_us = llama_time_us();
    common_params_fit_status status = COMMON_PARAMS_FIT_STATUS_SUCCESS;
    try {
        common_params_fit_impl(path_model, mparams, cparams, tensor_split, tensor_buft_overrides, margins, n_ctx_min, log_level);
        LOG_TRC("%s: successfully fit params to free device memory\n", __func__);
    } catch (const common_params_fit_exception & e) {
        LOG_WRN("%s: failed to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_FAILURE;
    } catch (const std::runtime_error & e) {
        LOG_ERR("%s: encountered an error while trying to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_ERROR;
    }
    const int64_t t1_us = llama_time_us();
    LOG_TRC("%s: fitting params to free memory took %.2f seconds\n", __func__, (t1_us - t0_us) * 1e-6);
    return status;
}

void common_memory_breakdown_print(const struct llama_context * ctx) {
    //const auto & devices = ctx->get_model().devices;
    const auto * model = llama_get_model(ctx);

    std::vector<ggml_backend_dev_t> devices;
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devices.push_back(llama_model_get_device(model, i));
    }

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    std::vector<std::array<std::string, 9>> table_data;
    table_data.reserve(devices.size());
    const std::string template_header = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
    const std::string template_gpu    = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
    const std::string template_other  = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

    table_data.push_back({template_header, "memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute", "unaccounted"});

    constexpr size_t MiB = 1024 * 1024;
    const std::vector<std::string> desc_prefixes_strip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

    // track seen buffer types to avoid double counting:
    std::set<ggml_backend_buffer_type_t> seen_buffer_types;

    // accumulative memory breakdown for each device and for host:
    std::vector<llama_memory_breakdown_data> mb_dev(devices.size());
    llama_memory_breakdown_data              mb_host;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (ggml_backend_buft_is_host(buft)) {
            mb_host.model   += mb.model;
            mb_host.context += mb.context;
            mb_host.compute += mb.compute;
            seen_buffer_types.insert(buft);
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            int i_dev = -1;
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i] == dev) {
                    i_dev = i;
                    break;
                }
            }
            if (i_dev != -1) {
                mb_dev[i_dev].model   += mb.model;
                mb_dev[i_dev].context += mb.context;
                mb_dev[i_dev].compute += mb.compute;
                seen_buffer_types.insert(buft);
                continue;
            }
        }
    }

    // print memory breakdown for each device:
    for (size_t i = 0; i < devices.size(); i++) {
        ggml_backend_dev_t dev = devices[i];
        llama_memory_breakdown_data mb = mb_dev[i];

        const std::string name = ggml_backend_dev_name(dev);
        std::string desc = ggml_backend_dev_description(dev);
        for (const std::string & prefix : desc_prefixes_strip) {
            if (desc.length() >= prefix.length() && desc.substr(0, prefix.length()) == prefix) {
                desc = desc.substr(prefix.length());
            }
        }

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        const size_t self = mb.model + mb.context + mb.compute;
        const int64_t unaccounted = static_cast<int64_t>(total) - static_cast<int64_t>(free) - static_cast<int64_t>(self);

        table_data.push_back({
            template_gpu,
            "  - " + name + " (" + desc + ")",
            std::to_string(total / MiB),
            std::to_string(free / MiB),
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            std::to_string(unaccounted / static_cast<int64_t>(MiB))});
    }

    // print memory breakdown for host:
    {
        const size_t self = mb_host.model + mb_host.context + mb_host.compute;
        table_data.push_back({
            template_other,
            "  - Host",
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb_host.model / MiB),
            std::to_string(mb_host.context / MiB),
            std::to_string(mb_host.compute / MiB),
            ""}); // unaccounted
    }

    // print memory breakdown for all remaining buffer types:
    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (seen_buffer_types.count(buft) == 1) {
            continue;
        }
        const std::string name = ggml_backend_buft_name(buft);
        const size_t self = mb.model + mb.context + mb.compute;
        table_data.push_back({
            template_other,
            "  - " + name,
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            ""}); // unaccounted
        seen_buffer_types.insert(buft);
    }

    for (size_t j = 1; j < table_data[0].size(); j++) {
        size_t max_len = 0;
        for (const auto & td : table_data) {
            max_len = std::max(max_len, td[j].length());
        }
        for (auto & td : table_data) {
            td[j].insert(j == 1 ? td[j].length() : 0, max_len - td[j].length(), ' ');
        }
    }
    for (const auto & td : table_data) {
        LOG_TRC(td[0].c_str(),
            __func__, td[1].c_str(), td[2].c_str(), td[3].c_str(), td[4].c_str(), td[5].c_str(),
            td[6].c_str(), td[7].c_str(), td[8].c_str());
    }
}

void common_fit_print(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams) {
    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    auto dmd = common_get_device_memory_data(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);
    GGML_ASSERT(dmd.size() == devs.size() + 1);

    for (size_t id = 0; id < devs.size(); id++) {
        printf("%s ",  ggml_backend_dev_name(devs[id]));
        printf("%zu ", dmd[id].mb.model/1024/1024);
        printf("%zu ", dmd[id].mb.context/1024/1024);
        printf("%zu ", dmd[id].mb.compute/1024/1024);
        printf("\n");
    }

    printf("Host ");
    printf("%zu ", dmd.back().mb.model/1024/1024);
    printf("%zu ", dmd.back().mb.context/1024/1024);
    printf("%zu ", dmd.back().mb.compute/1024/1024);
    printf("\n");
}
