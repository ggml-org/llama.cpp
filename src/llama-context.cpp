#include "llama-context.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-io.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <cinttypes>

//
// llama_context
//

llama_context::llama_context(
        const llama_model & model,
        const llama_context_params & params) :
    model     (model),
    t_start_us(model.t_start_us),
    t_load_us (model.t_load_us) {
    const auto & hparams = model.hparams;

    cparams.n_seq_max        = std::max(1u, params.n_seq_max);
    cparams.n_threads        = params.n_threads;
    cparams.n_threads_batch  = params.n_threads_batch;
    cparams.yarn_ext_factor  = params.yarn_ext_factor;
    cparams.yarn_attn_factor = params.yarn_attn_factor;
    cparams.yarn_beta_fast   = params.yarn_beta_fast;
    cparams.yarn_beta_slow   = params.yarn_beta_slow;
    cparams.defrag_thold     = params.defrag_thold;
    cparams.embeddings       = params.embeddings;
    cparams.offload_kqv      = params.offload_kqv;
    cparams.flash_attn       = params.flash_attn;
    cparams.no_perf          = params.no_perf;
    cparams.pooling_type     = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch          = hparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    // the batch has to be at least GGML_KQ_MASK_PAD because we will be padding the KQ_mask
    // this is required by GPU kernels in order to avoid out-of-bounds accesses (e.g. ggml_flash_attn_ext)
    // ref: https://github.com/ggerganov/llama.cpp/pull/5021
    if (cparams.n_batch < GGML_KQ_MASK_PAD) {
        LLAMA_LOG_WARN("%s: n_batch is less than GGML_KQ_MASK_PAD - increasing to %d\n", __func__, GGML_KQ_MASK_PAD);
        cparams.n_batch = GGML_KQ_MASK_PAD;
    }

    cparams.n_ubatch         = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    const uint32_t n_ctx_per_seq = cparams.n_ctx / cparams.n_seq_max;

    LLAMA_LOG_INFO("%s: n_seq_max     = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx         = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_per_seq = %u\n",   __func__, n_ctx_per_seq);
    LLAMA_LOG_INFO("%s: n_batch       = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch      = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: flash_attn    = %d\n",   __func__, cparams.flash_attn);
    LLAMA_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale    = %g\n",   __func__, cparams.rope_freq_scale);

    if (n_ctx_per_seq < hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_per_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, n_ctx_per_seq, hparams.n_ctx_train);
    }

    if (n_ctx_per_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_pre_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, n_ctx_per_seq, hparams.n_ctx_train);
    }

    logits_all = params.logits_all;

    if (!hparams.vocab_only) {
        // GPU backends
        for (auto * dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to initialize %s backend\n", __func__, ggml_backend_dev_name(dev));
                throw std::runtime_error("failed to initialize backend");
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    LLAMA_LOG_ERROR("%s: failed to initialize %s backend\n", __func__, ggml_backend_dev_name(dev));
                    throw std::runtime_error("failed to initialize backend");
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to initialize CPU backend\n", __func__);
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            // resized during inference when a batch uses more outputs
            if ((uint32_t) output_reserve(params.n_seq_max) < params.n_seq_max) {
                LLAMA_LOG_ERROR("%s: failed to reserve initial output buffer\n", __func__);
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }
}

llama_context::~llama_context() = default;

void llama_context::init() {
    const auto & hparams = model.hparams;

    if (hparams.vocab_only) {
        LLAMA_LOG_WARN("%s: model is vocab-only -- no computation will be performed\n", __func__);
        return;
    }

    // buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<ggml_backend_t>             backend_ptrs;
    for (auto & backend : backends) {
        auto * buft = ggml_backend_get_default_buffer_type(backend.get());
        auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
        if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
            // use the host buffer of the first device CPU for faster transfer of the intermediate state
            auto * dev = model.devices[0];
            auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
            if (host_buft) {
                buft = host_buft;
            }
        }
        backend_buft.push_back(buft);
        backend_ptrs.push_back(backend.get());
    }

    const size_t max_nodes = this->max_nodes();

    // buffer used to store the computation graph and the tensor meta data
    // TODO: move to base class
    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    // TODO: move these checks to ggml_backend_sched
    // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
    bool pipeline_parallel =
        model.n_devices() > 1 &&
        model.params.n_gpu_layers > (int) model.hparams.n_layer &&
        model.params.split_mode == LLAMA_SPLIT_MODE_LAYER &&
        cparams.offload_kqv;

    // pipeline parallelism requires support for async compute and events in all devices
    if (pipeline_parallel) {
        for (auto & backend : backends) {
            auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                // ignore CPU backend
                continue;
            }
            auto * dev = ggml_backend_get_device(backend.get());
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev, &props);
            if (!props.caps.async || !props.caps.events) {
                // device does not support async compute or events
                pipeline_parallel = false;
                break;
            }
        }
    }

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, pipeline_parallel));

    if (pipeline_parallel) {
        LLAMA_LOG_INFO("%s: pipeline parallelism enabled (n_copies=%d)\n", __func__, ggml_backend_sched_get_n_copies(sched.get()));
    }

    // initialize scheduler with the worst-case graph
    {
        uint32_t n_seqs = 1; // TODO: worst-case number of sequences
        uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);
        llama_token token = model.vocab.token_bos(); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph

        int n_splits_pp = -1;
        int n_nodes_pp  = -1;

        int n_splits_tg = -1;
        int n_nodes_tg  = -1;

        // reserve pp graph first so that buffers are only allocated once
        {
            llama_ubatch ubatch_pp = { true, n_tokens, n_tokens / n_seqs, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};
            auto * gf = graph_init();
            graph_build(ctx_compute.get(), gf, ubatch_pp, true);
            if (!ggml_backend_sched_reserve(sched.get(), gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute pp buffers\n", __func__);
                throw std::runtime_error("failed to allocate compute buffers");
            }

            n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
            n_nodes_pp  = ggml_graph_n_nodes(gf);
        }

        // reserve with tg graph to get the number of splits and nodes
        {
            llama_ubatch ubatch_tg = { true, 1, 1, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};
            auto * gf = graph_init();
            graph_build(ctx_compute.get(), gf, ubatch_tg, true);
            if (!ggml_backend_sched_reserve(sched.get(), gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute tg buffers\n", __func__);
                throw std::runtime_error("failed to allocate compute buffers");
            }
            n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
            n_nodes_tg  = ggml_graph_n_nodes(gf);
        }

        // reserve again with pp graph to avoid ggml-alloc reallocations during inference
        {
            llama_ubatch ubatch_pp = { true, n_tokens, n_tokens / n_seqs, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};
            auto * gf = graph_init();
            graph_build(ctx_compute.get(), gf, ubatch_pp, true);
            if (!ggml_backend_sched_reserve(sched.get(), gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute pp buffers\n", __func__);
                throw std::runtime_error("failed to allocate compute buffers");
            }
        }

        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft = backend_buft[i];
            size_t size = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size > 1) {
                LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                        ggml_backend_buft_name(buft),
                        size / 1024.0 / 1024.0);
            }
        }

        if (n_nodes_pp == n_nodes_tg) {
            LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
        } else {
            LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
        }

        if (n_splits_pp == n_splits_tg) {
            LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
        } else {
            LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
        }
    }
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_per_seq() const {
    return cparams.n_ctx / cparams.n_seq_max;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

int32_t llama_context::max_nodes() const {
    return std::max<int32_t>(8192, 5*model.n_tensors());
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    // reorder logits for backward compatibility
    output_reorder();

    return logits;
}

float * llama_context::get_logits_ith(int32_t i) {
    int32_t j = -1;

    try {
        if (logits == nullptr) {
            throw std::runtime_error("no logits");
        }

        if (i < 0) {
            j = n_outputs + i;
            if (j < 0) {
                throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
            }
        } else if ((size_t) i >= output_ids.size()) {
            throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
        } else {
            j = output_ids[i];
        }

        if (j < 0) {
            throw std::runtime_error(format("batch.logits[%d] != true", i));
        }
        if (j >= n_outputs) {
            // This should not happen
            throw std::runtime_error(format("corrupt output buffer (j=%d, n_outputs=%d)", j, n_outputs));
        }

        return logits + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    // reorder embeddings for backward compatibility
    output_reorder();

    return embd;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    int32_t j = -1;

    try {
        if (embd == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        if (i < 0) {
            j = n_outputs + i;
            if (j < 0) {
                throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
            }
        } else if ((size_t) i >= output_ids.size()) {
            throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
        } else {
            j = output_ids[i];
        }

        if (j < 0) {
            throw std::runtime_error(format("batch.logits[%d] != true", i));
        }
        if (j >= n_outputs) {
            // This should not happen
            throw std::runtime_error(format("corrupt output buffer (j=%d, n_outputs=%d)", j, n_outputs));
        }

        return embd + j*model.hparams.n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

int64_t llama_context::n_pos_per_token() const {
    return model.arch == LLM_ARCH_QWEN2VL ? 4 : 1;
}

void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
        if (set_abort_callback_fn) {
            set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
        }
    }
}

void llama_context::set_embeddings(bool value) {
    cparams.embeddings = value;
}

void llama_context::set_causal_attn(bool value) {
    cparams.causal_attn = value;
}

void llama_context::set_adapter_lora(
            struct llama_adapter_lora * adapter,
            float scale) {
    loras[adapter] = scale;
}

bool llama_context::rm_adapter_lora(
            struct llama_adapter_lora * adapter) {
    auto pos = loras.find(adapter);
    if (pos != loras.end()) {
        loras.erase(pos);
        return true;
    }

    return false;
}

void llama_context::clear_adapter_lora() {
    loras.clear();
}

bool llama_context::apply_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    return cvec.apply(model, data, len, n_embd, il_start, il_end);
}

void llama_context::synchronize() {
    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

ggml_cgraph * llama_context::graph_init() {
    inp_tokens  = nullptr;
    inp_embd    = nullptr;
    inp_pos     = nullptr;
    inp_out_ids = nullptr;
    inp_mean    = nullptr;
    inp_cls     = nullptr;

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    return ggml_new_graph_custom(ctx_compute.get(), max_nodes(), false);
}

llama_graph_result llama_context::graph_build(
            ggml_context * ctx,
             ggml_cgraph * gf,
      const llama_ubatch & ubatch,
                    bool   worst_case) {
    return model.build_graph(ctx, gf, this, cparams, ubatch, worst_case);
}

enum ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        set_threadpool_fn(backend_cpu, tp);
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

void llama_context::input_set(const llama_ubatch & ubatch) {
    const llama_hparams & hparams = model.hparams;

    if (ubatch.token) {
        const int64_t n_tokens = ubatch.n_tokens;

        ggml_backend_tensor_set(inp_tokens, ubatch.token, 0, n_tokens*ggml_element_size(inp_tokens));
    }

    if (ubatch.embd) {
        const int64_t n_embd   = hparams.n_embd;
        const int64_t n_tokens = ubatch.n_tokens;

        ggml_backend_tensor_set(inp_embd, ubatch.embd, 0, n_tokens*n_embd*ggml_element_size(inp_embd));
    }

    if (ubatch.pos && inp_pos) {
        const int64_t n_tokens = ubatch.n_tokens;

        ggml_backend_tensor_set(inp_pos, ubatch.pos, 0, n_tokens*n_pos_per_token()*ggml_element_size(inp_pos));
    }

    if (hparams.causal_attn || cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        //GGML_ASSERT(inp_out_ids && "every model that can must skip unused outputs");

        if (!inp_out_ids) {
            LLAMA_LOG_WARN("%s: 'inp_out_ids' is not created\n", __func__);
        } else {
            const int64_t n_tokens = ubatch.n_tokens;

            GGML_ASSERT(ggml_backend_buffer_is_host(inp_out_ids->buffer));
            int32_t * data = (int32_t *) inp_out_ids->data;

            if (n_outputs == n_tokens) {
                for (int i = 0; i < n_tokens; ++i) {
                    data[i] = i;
                }
            } else if (ubatch.output) {
                int32_t n_outputs = 0;
                for (int i = 0; i < n_tokens; ++i) {
                    if (ubatch.output[i]) {
                        data[n_outputs++] = i;
                    }
                }
                // the graph needs to have been passed the correct number of outputs
                GGML_ASSERT(n_outputs == n_outputs);
            } else if (n_outputs == 1) {
                // only keep last output
                data[0] = n_tokens - 1;
            } else {
                GGML_ASSERT(n_outputs == 0);
            }
        }
    }

    if (cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN) {
        const int64_t n_tokens     = ubatch.n_tokens;
        const int64_t n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t n_seqs       = ubatch.n_seqs;

        GGML_ASSERT(inp_mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_mean->buffer));

        float * data = (float *) inp_mean->data;
        memset(inp_mean->data, 0, n_tokens * n_tokens * ggml_element_size(inp_mean));

        std::vector<uint64_t> sum(n_tokens, 0);

        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[s][0];

            // TODO: adapt limits to n_seqs when ubatch.equal_seqs is true
            GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == MEAN");

            sum[seq_id] += ubatch.n_seq_tokens;
        }

        std::vector<float> div(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const uint64_t s = sum[i];
            if (s > 0) {
                div[i] = 1.0f/float(s);
            }
        }

        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[s][0];

            for (int i = 0; i < n_seq_tokens; ++i) {
                data[seq_id*n_tokens + s*n_seq_tokens + i] = div[seq_id];
            }
        }
    }

    if (cparams.embeddings && (
                cparams.pooling_type == LLAMA_POOLING_TYPE_CLS ||
                cparams.pooling_type == LLAMA_POOLING_TYPE_RANK)) {
        const int64_t n_tokens     = ubatch.n_tokens;
        const int64_t n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t n_seqs       = ubatch.n_seqs;

        GGML_ASSERT(inp_cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_cls->buffer));

        uint32_t * data = (uint32_t *) inp_cls->data;
        memset(inp_cls->data, 0, n_tokens * ggml_element_size(inp_cls));

        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[s][0];

            // TODO: adapt limits to n_seqs when ubatch.equal_seqs is true
            GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == CLS or RANK");

            for (int i = 0; i < n_seq_tokens; ++i) {
                const llama_pos pos = ubatch.pos[s*n_seq_tokens + i];

                if (pos == 0) {
                    data[seq_id] = s*n_seq_tokens + i;
                }
            }
        }
    }

    if (cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_LAST) {
        const int64_t n_tokens     = ubatch.n_tokens;
        const int64_t n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t n_seqs       = ubatch.n_seqs;

        GGML_ASSERT(inp_cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_cls->buffer));

        uint32_t * data = (uint32_t *) inp_cls->data;
        memset(inp_cls->data, 0, n_tokens * ggml_element_size(inp_cls));

        std::vector<int> last_pos(n_tokens, -1);
        std::vector<int> last_row(n_tokens, -1);

        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[s][0];

            // TODO: adapt limits to n_seqs when ubatch.equal_seqs is true
            GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == LAST");

            for (int i = 0; i < n_seq_tokens; ++i) {
                const llama_pos pos = ubatch.pos[s*n_seq_tokens + i];

                if (pos >= last_pos[seq_id]) {
                    last_pos[seq_id] = pos;
                    last_row[seq_id] = s*n_seq_tokens + i;
                }
            }
        }

        for (int i = 0; i < n_tokens; ++i) {
            if (last_row[i] >= 0) {
                data[i] = last_row[i];
            }
        }
    }

    GGML_ASSERT(
            // (!a || b) is a logical implication (a -> b)
            // !hparams.causal_attn -> !cparams.causal_attn
            (hparams.causal_attn || !cparams.causal_attn) &&
            "causal attention is not supported by this model"
            );
}

int32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, cparams.n_seq_max);

    const auto n_batch = cparams.n_batch;
    const auto n_vocab = vocab.n_tokens();
    const auto n_embd  = hparams.n_embd;

    // TODO: use a per-batch flag for logits presence instead
    const bool has_logits = !cparams.embeddings;
    const bool has_embd   =  cparams.embeddings && (cparams.pooling_type == LLAMA_POOLING_TYPE_NONE);

    logits_size = has_logits ? n_vocab*n_outputs_max : 0;
    embd_size   = has_embd   ?  n_embd*n_outputs_max : 0;

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  = (logits_size + embd_size) * sizeof(float);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_INFO("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            buf_output = nullptr;
            logits = nullptr;
            embd = nullptr;
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    logits = has_logits ? output_base               : nullptr;
    embd   = has_embd   ? output_base + logits_size : nullptr;

    output_size = n_outputs_max;

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    ggml_backend_buffer_clear(buf_output.get(), 0);

    n_outputs = 0;

    return n_outputs_max;
}

void llama_context::output_reorder() {
    auto & out_ids = sbatch.out_ids;
    if (!out_ids.empty()) {
        const uint32_t n_vocab = model.vocab.n_tokens();
        const uint32_t n_embd  = model.hparams.n_embd;

        GGML_ASSERT((size_t) n_outputs == out_ids.size());

        // TODO: is there something more efficient which also minimizes swaps?
        // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
        for (int32_t i = 0; i < n_outputs - 1; ++i) {
            int32_t j_min = i;
            for (int32_t j = i + 1; j < n_outputs; ++j) {
                if (out_ids[j] < out_ids[j_min]) {
                    j_min = j;
                }
            }
            if (j_min == i) { continue; }
            std::swap(out_ids[i], out_ids[j_min]);
            if (logits_size > 0) {
                for (uint32_t k = 0; k < n_vocab; k++) {
                    std::swap(logits[i*n_vocab + k], logits[j_min*n_vocab + k]);
                }
            }
            if (embd_size > 0) {
                for (uint32_t k = 0; k < n_embd; k++) {
                    std::swap(embd[i*n_embd + k], embd[j_min*n_embd + k]);
                }
            }
        }
        std::fill(output_ids.begin(), output_ids.end(), -1);
        for (int32_t i = 0; i < n_outputs; ++i) {
            output_ids[out_ids[i]] = i;
        }
        out_ids.clear();
    }
}

void llama_context::build_cb(
         ggml_tensor * cur,
          const char * name,
  const llama_ubatch & ubatch,
                 int   il) {
    if (il >= 0) {
        ggml_format_name(cur, "%s-%d", name, il);
    } else {
        ggml_set_name(cur, name);
    }

    if (!cparams.offload_kqv) {
        if (strcmp(name, "kqv_merged_cont") == 0) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend_cpu);
        }
    }

    // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
    // FIXME: fix in ggml_backend_sched
    const bool full_offload = model.params.n_gpu_layers > (int) model.hparams.n_layer;
    if (ubatch.n_tokens < 32 || full_offload) {
        if (il != -1 && strcmp(name, "norm") == 0) {
            const auto & dev_layer = model.dev_layer(il);
            for (auto & backend : backends) {
                if (ggml_backend_get_device(backend.get()) == dev_layer) {
                    if (ggml_backend_supports_op(backend.get(), cur)) {
                        ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                    }
                }
            }
        }
    }
}

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);

    return data;
}

ggml_tensor * llama_context::build_cvec(
        ggml_context * ctx0,
         ggml_tensor * cur,
                 int   il) {
    return cvec.apply_to(ctx0, cur, il);
}

ggml_tensor * llama_context::build_lora_mm(
        ggml_context * ctx0,
         ggml_tensor * w,
         ggml_tensor * cur) {
    struct ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    for (const auto & lora : loras) {
        struct llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        struct ggml_tensor * ab_cur = ggml_mul_mat(
            ctx0, lw->b,
            ggml_mul_mat(ctx0, lw->a, cur)
        );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llama_context::build_lora_mm_id(
        ggml_context * ctx0,
         ggml_tensor * w,
         ggml_tensor * cur,
         ggml_tensor * ids) {
    struct ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
    for (const auto & lora : loras) {
        struct llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        struct ggml_tensor * ab_cur = ggml_mul_mat_id(
            ctx0, lw->b,
            ggml_mul_mat_id(ctx0, lw->a, cur, ids),
            ids
        );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llama_context::build_rope_factors(int il) {
    const auto & hparams = model.hparams;

    // choose long/short freq factors based on the context size
    const auto n_ctx_per_seq = cparams.n_ctx / cparams.n_seq_max;

    if (model.layers[il].rope_freqs != nullptr) {
        return model.layers[il].rope_freqs;
    }

    if (n_ctx_per_seq > hparams.n_ctx_orig_yarn) {
        return model.layers[il].rope_long;
    }

    return model.layers[il].rope_short;
}

ggml_tensor * llama_context::build_rope_shift(
        ggml_context * ctx0,
        ggml_tensor * cur,
        ggml_tensor * shift,
        ggml_tensor * factors,
        ggml_backend_buffer * bbuf) {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;
    const auto & freq_base  = cparams.rope_freq_base;
    const auto & freq_scale = cparams.rope_freq_scale;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;

    const auto & n_rot     = model.hparams.n_rot;
    const auto & rope_type = model.hparams.rope_type;

    struct ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx0, cur, GGML_TYPE_F32);

        if (bbuf) {
            for (auto & backend : backends) {
                // Figure out which backend KV cache belongs to
                if (ggml_backend_supports_buft(backend.get(), ggml_backend_buffer_get_type(bbuf))) {
                    ggml_backend_sched_set_tensor_backend(sched.get(), tmp, backend.get());
                    break;
                }
            }
        }

        tmp = ggml_rope_ext_inplace(ctx0, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        tmp = ggml_cpy(ctx0, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx0, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

ggml_tensor * llama_context::build_inp_embd(
        ggml_context * ctx0,
         ggml_tensor * tok_embd,
  const llama_ubatch & ubatch) {
    const auto & hparams = model.hparams;

    const int64_t n_embd = hparams.n_embd;

    struct ggml_tensor * inpL;

    if (ubatch.token) {
        inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        //cb(inp_tokens, "inp_tokens", -1);
        ggml_set_input(inp_tokens);

        inpL = ggml_get_rows(ctx0, tok_embd, inp_tokens);

        // apply lora for embedding tokens if needed
        for (const auto & lora : loras) {
            struct llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            struct ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, inp_tokens)
                        ), scale);

            inpL = ggml_add(ctx0, inpL, inpL_delta);
        }
    } else {
        inp_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, ubatch.n_tokens);
        inpL = inp_embd;
        ggml_set_input(inp_embd);
    }

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f) {
        inpL = ggml_scale(ctx0, inpL, hparams.f_embedding_scale);
    }

    //cb(inpL, "inp_embd", -1);

    return inpL;
}

ggml_tensor * llama_context::build_inp_pos(
        ggml_context * ctx0,
             int32_t   n_tokens) {
    inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens*n_pos_per_token());
    ggml_set_input(inp_pos);

    return inp_pos;
}

ggml_tensor * llama_context::build_inp_out_ids(
        ggml_context * ctx0,
             int32_t   n_tokens,
                bool   worst_case) {
    const int32_t n_out_ids = worst_case ? n_tokens : n_outputs;

    inp_out_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_out_ids);
    ggml_set_input(inp_out_ids);

    return inp_out_ids;
}

ggml_tensor * llama_context::build_inp_mean(
        ggml_context * ctx0,
             int32_t   n_tokens) {
    inp_mean = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_input(inp_mean);

    return inp_mean;
}

ggml_tensor * llama_context::build_inp_cls(
        ggml_context * ctx0,
             int32_t   n_tokens) {
    inp_cls = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_cls);

    return inp_cls;
}

//
// state
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy() = default;

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(const ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    size_t size_written = 0;
};

class llama_io_write_buffer : public llama_io_write_i {
public:
    llama_io_write_buffer(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ggml_backend_tensor_get(tensor, ptr, offset, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;
};

class llama_io_read_buffer : public llama_io_read_i {
public:
    llama_io_read_buffer(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    const uint8_t * read(size_t size) override {
        const uint8_t * base_ptr = ptr;
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ptr += size;
        size_read += size;
        buf_size -= size;
        return base_ptr;
    }

    void read_to(void * dst, size_t size) override {
        memcpy(dst, read(size), size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read_to(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    const uint8_t * read(size_t size) override {
        temp_buffer.resize(size);
        read_to(temp_buffer.data(), size);
        return temp_buffer.data();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io;
    try {
        return state_get_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_get_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_buffer io(src, size);
    try {
        return state_set_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_size(llama_seq_id seq_id) {
    llama_io_write_dummy io;
    try {
        return state_seq_get_data(io, seq_id);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_seq_get_data(io, seq_id);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size) {
    llama_io_read_buffer io(src, size);
    try {
        return state_seq_set_data(io, seq_id);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_set_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_get_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_set_data(io, seq_id);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_get_data(io, seq_id);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_get_data(llama_io_write_i & io) {
    // write model info
    {
        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    // write output ids
    {
        output_reorder();

        const auto n_outputs    = this->n_outputs;
        const auto & output_ids = this->output_ids;

        std::vector<int32_t> w_output_pos;

        GGML_ASSERT(n_outputs <= output_size);

        w_output_pos.resize(n_outputs);

        // build a more compact representation of the output ids
        for (size_t i = 0; i < n_batch(); ++i) {
            // map an output id to a position in the batch
            int32_t pos = output_ids[i];
            if (pos >= 0) {
                GGML_ASSERT(pos < n_outputs);
                w_output_pos[pos] = i;
            }
        }

        io.write(&n_outputs, sizeof(n_outputs));

        if (n_outputs) {
            io.write(w_output_pos.data(), n_outputs * sizeof(int32_t));
        }
    }

    // write logits
    {
        const uint64_t logits_size = std::min((uint64_t) this->logits_size, (uint64_t) n_outputs * model.vocab.n_tokens());

        io.write(&logits_size, sizeof(logits_size));

        if (logits_size) {
            io.write(logits, logits_size * sizeof(float));
        }
    }

    // write embeddings
    {
        const uint64_t embd_size = std::min((uint64_t) this->embd_size, (uint64_t) n_outputs * model.hparams.n_embd);

        io.write(&embd_size, sizeof(embd_size));

        if (embd_size) {
            io.write(embd, embd_size * sizeof(float));
        }
    }

    return io.n_bytes();
}

size_t llama_context::state_set_data(llama_io_read_i & io) {
    // read model info
    {
        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    // read output ids
    {
        auto n_outputs = this->n_outputs;
        io.read_to(&n_outputs, sizeof(n_outputs));

        if (n_outputs > output_reserve(n_outputs)) {
            throw std::runtime_error("could not reserve outputs");
        }

        std::vector<int32_t> output_pos;

        if (n_outputs) {
            output_pos.resize(n_outputs);
            io.read_to(output_pos.data(), n_outputs * sizeof(int32_t));

            for (int32_t i = 0; i < (int32_t) output_pos.size(); ++i) {
                int32_t id = output_pos[i];
                if ((uint32_t) id >= n_batch()) {
                    throw std::runtime_error(format("invalid output id, %d does not fit in batch size of %u", id, n_batch()));
                }
                this->output_ids[id] = i;
            }

            this->n_outputs = n_outputs;
        }
    }

    // read logits
    {
        uint64_t logits_size;
        io.read_to(&logits_size, sizeof(logits_size));

        if (this->logits_size < logits_size) {
            throw std::runtime_error("logits buffer too small");
        }

        if (logits_size) {
            io.read_to(this->logits, logits_size * sizeof(float));
        }
    }

    // read embeddings
    {
        uint64_t embd_size;
        io.read_to(&embd_size, sizeof(embd_size));

        if (this->embd_size < embd_size) {
            throw std::runtime_error("embeddings buffer too small");
        }

        if (embd_size) {
            io.read_to(this->embd, embd_size * sizeof(float));
        }
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_get_data(llama_io_write_i & io, llama_seq_id seq_id) {
    GGML_UNUSED(seq_id);

    return io.n_bytes();
}

size_t llama_context::state_seq_set_data(llama_io_read_i & io, llama_seq_id seq_id) {
    GGML_UNUSED(seq_id);

    return io.n_bytes();
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
}

//
// llama_context_kv_self
//

llama_context_kv_self::llama_context_kv_self(
        const llama_model & model,
        const llama_context_params & params) :
    llama_context(model, params),
    kv_self(model.hparams) {
    const auto & hparams = model.hparams;

    LLAMA_LOG_DEBUG("%s: n_ctx = %u\n", __func__, cparams.n_ctx);

    cparams.n_ctx = GGML_PAD(cparams.n_ctx, get_ctx_padding(cparams));

    LLAMA_LOG_DEBUG("%s: n_ctx = %u (padded)\n", __func__, cparams.n_ctx);

    // build worst-case graph for encoder if a model contains encoder
    is_encoding = llama_model_has_encoder(&model); // TODO: model.has_encoder()

    uint32_t kv_size = cparams.n_ctx;
    ggml_type type_k = params.type_k;
    ggml_type type_v = params.type_v;

    // Mamba only needs a constant number of KV cache cells per sequence
    if (llama_model_is_recurrent(&model)) {
        // Mamba needs at least as many KV cells as there are sequences kept at any time
        kv_size = std::max((uint32_t) 1, params.n_seq_max);
        // it's probably best to keep as much precision as possible for the states
        type_k = GGML_TYPE_F32; // required by ggml_ssm_conv for Mamba's conv_states
        type_v = GGML_TYPE_F32; // required by ggml_ssm_scan for Mamba's ssm_states
    }

    GGML_ASSERT(hparams.n_embd_head_k % ggml_blck_size(type_k) == 0);
    GGML_ASSERT(hparams.n_embd_head_v % ggml_blck_size(type_v) == 0);

    if (!hparams.vocab_only) {
        if (!kv_self.init(model, cparams, type_k, type_v, kv_size, cparams.offload_kqv)) {
            LLAMA_LOG_ERROR("%s: llama_kv_cache_init() failed for self-attention cache\n", __func__);
            throw std::runtime_error("failed to initialize self-attention cache");
        }

        {
            const size_t memory_size_k = kv_self.size_k_bytes();
            const size_t memory_size_v = kv_self.size_v_bytes();

            LLAMA_LOG_INFO("%s: KV self size  = %7.2f MiB, K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                      (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f),
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
        }
    }
}

llama_context_kv_self::~llama_context_kv_self() = default;

uint32_t llama_context_kv_self::n_seq_max() const {
    // TODO: add notion of n_seq_max to llama_kv_cache and use it here
    return kv_self.size;
}

llama_kv_cache * llama_context_kv_self::get_kv_self() {
    return &kv_self;
}

const llama_kv_cache * llama_context_kv_self::get_kv_self() const {
    return &kv_self;
}

ggml_cgraph * llama_context_kv_self::graph_init() {
    inp_KQ_mask         = nullptr;
    inp_KQ_mask_cnv     = nullptr;
    inp_KQ_mask_swa     = nullptr;
    inp_KQ_mask_swa_cnv = nullptr;
    inp_KQ_mask_cross   = nullptr;
    inp_k_shift         = nullptr;
    inp_s_copy          = nullptr;
    inp_s_mask          = nullptr;
    inp_embd_enc        = nullptr;
    inp_pos_bucket      = nullptr;

    return llama_context::graph_init();
}

int llama_context_kv_self::encode(llama_batch & inp_batch) {
    is_encoding = true;

    if (inp_batch.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    // temporary allocate memory for the input batch if needed
    // TODO: this is incorrect for multiple sequences because pos_max() is the maximum across all sequences
    llama_batch_allocr batch_allocr(inp_batch, inp_batch.pos ? -1 : pos_max() + 1);

    const llama_batch & batch = batch_allocr.batch;
    const int32_t n_tokens = batch.n_tokens;

    const auto & hparams = model.hparams;

    GGML_ASSERT((!batch.token && batch.embd) || (batch.token && !batch.embd)); // NOLINT

    if (batch.token) {
        for (int32_t i = 0; i < n_tokens; ++i) {
            if (batch.token[i] < 0 || (uint32_t) batch.token[i] >= model.vocab.n_tokens()) {
                LLAMA_LOG_ERROR("%s: invalid token[%d] = %d\n", __func__, i, batch.token[i]);
                return -1;
            }
        }
    }

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= (uint32_t) n_tokens && "encoder requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    n_queued_tokens += n_tokens;

    const int64_t n_embd = hparams.n_embd;

    sbatch.from_batch(batch, n_embd, /* simple_split */ true, /* logits_all */ true);

    const llama_ubatch ubatch = sbatch.split_simple(n_tokens);

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (int32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    inp_embd_enc = NULL;
    n_outputs = n_tokens;

    //batch_manager->prepare(ubatch);

    // TODO: do reserve
    GGML_ASSERT(need_reserve == false);

    ggml_backend_sched_reset(sched.get());
    ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

    auto * gf = graph_init();
    auto res = graph_build(ctx_compute.get(), gf, ubatch, false);

    ggml_backend_sched_alloc_graph(sched.get(), gf);

    input_set(ubatch);

    const auto compute_status = graph_compute(gf, n_tokens > 1);
    switch (compute_status) {
        case GGML_STATUS_SUCCESS:
            break;
        case GGML_STATUS_ABORTED:
            return 2;
        case GGML_STATUS_ALLOC_FAILED:
            return -2;
        case GGML_STATUS_FAILED:
        default:
            return -3;
    }

    auto * t_embd = res.t_embd_pooled ? res.t_embd_pooled : res.t_embd;

    // extract embeddings
    if (t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        if (llama_model_has_decoder(&model)) {
            embd_enc.resize(n_tokens*n_embd);
            float * embd_out = embd_enc.data();

            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_tokens*n_embd*sizeof(float));
            GGML_ASSERT(!ubatch.equal_seqs); // TODO: handle equal splits

            // remember the sequence ids used during the encoding - needed for cross attention later
            seq_ids_enc.resize(n_tokens);
            for (int32_t i = 0; i < n_tokens; i++) {
                for (int s = 0; s < ubatch.n_seq_id[i]; s++) {
                    llama_seq_id seq_id = ubatch.seq_id[i][s];
                    seq_ids_enc[i].insert(seq_id);
                }
            }
        } else {
            GGML_ASSERT(embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd != nullptr);
                        float * embd_out = embd;

                        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_size);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_tokens*n_embd*sizeof(float));
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings
                        auto & embd_seq_out = embd_seq;
                        embd_seq_out.clear();

                        GGML_ASSERT(!ubatch.equal_seqs); // TODO: handle equal splits

                        for (int32_t i = 0; i < n_tokens; i++) {
                            const llama_seq_id seq_id = ubatch.seq_id[i][0];
                            if (embd_seq_out.find(seq_id) != embd_seq_out.end()) {
                                continue;
                            }
                            embd_seq_out[seq_id].resize(n_embd);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd*seq_id)*sizeof(float), n_embd*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // TODO: this likely should be the same logic as in llama_decoder_internal, but better to
                        //       wait for an encoder model that requires this pooling type in order to test it
                        //       https://github.com/ggerganov/llama.cpp/pull/9510
                        GGML_ABORT("RANK pooling not implemented yet");
                    }
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }
    }

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(sched.get());

    return 0;
}

int llama_context_kv_self::decode(llama_batch & inp_batch) {
    is_encoding = false;

    if (inp_batch.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    // temporary allocate memory for the input batch if needed
    // TODO: this is incorrect for multiple sequences because pos_max() is the maximum across all sequences
    llama_batch_allocr batch_allocr(inp_batch, inp_batch.pos ? -1 : pos_max() + 1);

    const llama_batch & batch = batch_allocr.batch;

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int32_t n_vocab = vocab.n_tokens();

    const int64_t n_tokens_all = batch.n_tokens;
    const int64_t n_embd       = hparams.n_embd;

    // TODO: remove this stuff
    class batch_guard {
    public:
        batch_guard(llama_kv_cache & kv_self) : kv_slot_restorer(kv_self) {
        }

        ~batch_guard() {
            if (!is_done) {
                kv_slot_restorer.restore();
            }
        }

        void done() {
            is_done = true;
        }

        void save(const llama_kv_cache_slot_info & slot_info) {
            kv_slot_restorer.save(slot_info);
        }

    private:
        bool is_done = false;

        llama_kv_slot_restorer kv_slot_restorer;
    };

    batch_guard bg(kv_self);

    GGML_ASSERT((!batch.token && batch.embd) || (batch.token && !batch.embd)); // NOLINT

    if (batch.token) {
        for (int64_t i = 0; i < n_tokens_all; ++i) {
            if (batch.token[i] < 0 || (uint32_t) batch.token[i] >= model.vocab.n_tokens()) {
                LLAMA_LOG_ERROR("%s: invalid token[%" PRId64 "] = %d\n", __func__, i, batch.token[i]);
                throw std::runtime_error("invalid token");
            }
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    // this indicates we are doing pooled embedding, so we ignore batch.logits and output all tokens
    const bool embd_pooled = cparams.embeddings && cparams.pooling_type != LLAMA_POOLING_TYPE_NONE;

    embd_seq.clear();

    int64_t n_outputs_all = 0;

    // count outputs
    if (batch.logits && !embd_pooled) {
        for (uint32_t i = 0; i < n_tokens_all; ++i) {
            n_outputs_all += batch.logits[i] != 0;
        }
    } else if (logits_all || embd_pooled) {
        n_outputs_all = n_tokens_all;
    } else {
        // keep last output only
        n_outputs_all = 1;
    }

    const bool logits_all = n_outputs_all == n_tokens_all;

    sbatch.from_batch(batch, n_embd,
            /* simple_split */ !kv_self.recurrent,
            /* logits_all   */ logits_all);

    // reserve output buffer
    // TODO: move to batch manager?
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %" PRId64 " outputs\n", __func__, n_outputs_all);
        return -2;
    };

    int64_t n_outputs_prev = 0;

    while (sbatch.n_tokens > 0) {
        llama_ubatch ubatch = llama_ubatch();

        const auto & n_ubatch = cparams.n_ubatch;

        const bool embd_pooled = cparams.embeddings && cparams.pooling_type != LLAMA_POOLING_TYPE_NONE;

        if (kv_self.recurrent) {
            if (embd_pooled) {
                // Pooled embeddings cannot be split across ubatches (yet)
                ubatch = sbatch.split_seq(n_ubatch);
            } else {
                // recurrent model architectures are easier to implement
                // with equal-length sequences
                ubatch = sbatch.split_equal(n_ubatch);
            }
        } else {
            ubatch = sbatch.split_simple(n_ubatch);
        }

        // count the outputs in this u_batch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                GGML_ASSERT(ubatch.output);
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        // non-causal masks do not use the KV cache
        if (hparams.causal_attn) {
            kv_self_update();

            // if we have enough unused cells before the current head ->
            //   better to start searching from the beginning of the cache, hoping to fill it
            if (kv_self.head > kv_self.used + 2*ubatch.n_tokens) {
                kv_self.head = 0;
            }

            const auto slot_info = kv_self.find_slot(ubatch);
            if (!slot_info) {
                LLAMA_LOG_ERROR("%s: failed to prepare ubatch\n", __func__);
                return -3;
            }

            bg.save(slot_info);

            if (!kv_self.recurrent) {
                // a heuristic, to avoid attending the full cache if it is not yet utilized
                // after enough generations, the benefit from this heuristic disappears
                // if we start defragmenting the cache, the benefit from this will be more important
                const uint32_t pad = kv_self.get_padding(cparams);
                kv_self.n = std::min(kv_self.size, std::max(pad, GGML_PAD(kv_self.cell_max(), pad)));
                //kv_self.n = llama_kv_cache_cell_max(kv_self);
            }
        }

        //printf("kv_self.n = %5d, kv_self.used = %5d, kv_self.head = %5d\n", kv_self.n, kv_self.used, kv_self.head);

        // reserve a worst case graph if needed
        if (need_reserve) {
            LLAMA_LOG_DEBUG("%s: reserving a worst case graph\n", __func__);

            // build worst-case graph
            uint32_t n_seqs = 1; // TODO: worst-case number of sequences
            uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

            llama_token token = model.vocab.token_bos(); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph
            llama_ubatch ubatch = { true, n_tokens, n_tokens / n_seqs, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};

            auto * gf = graph_init();
            graph_build(ctx_compute.get(), gf, ubatch, true);

            // initialize scheduler with the worst-case graph
            ggml_backend_sched_reset(sched.get());
            if (!ggml_backend_sched_reserve(sched.get(), gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
            }

            need_reserve = false;
        }

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        auto * gf = graph_init();
        auto res = graph_build(ctx_compute.get(), gf, ubatch, false);

        // LLAMA_LOG_INFO("graph build time: %.3f ms (%d nodes, %d leafs)\n", (ggml_time_us() - t_start_us)/1000.0, gf->n_nodes, gf->n_leafs);

        ggml_backend_sched_alloc_graph(sched.get(), gf);

        input_set(ubatch);

        const auto compute_status = graph_compute(gf, ubatch.n_tokens > 1);
        if (compute_status != GGML_STATUS_SUCCESS) {
            switch (compute_status) {
                case GGML_STATUS_ABORTED:
                    return 2;
                case GGML_STATUS_ALLOC_FAILED:
                    return -2;
                case GGML_STATUS_FAILED:
                default:
                    return -3;
            }
        }

        // update the kv ring buffer
        {
            kv_self.head += ubatch.n_tokens;

            // Ensure kv cache head points to a valid index.
            if (kv_self.head >= kv_self.size) {
                kv_self.head = 0;
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits = cparams.embeddings ? nullptr    : res.t_logits;
        auto * t_embd   = cparams.embeddings ? res.t_embd : nullptr;

        if (t_embd && res.t_embd_pooled) {
            t_embd = res.t_embd_pooled;
        }

        // extract logits
        if (t_logits && n_outputs > 0) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits != nullptr);

            float * logits_out = logits + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits_size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd != nullptr);
                        float * embd_out = embd + n_outputs_prev*n_embd;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd <= (int64_t) embd_size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
                            const llama_seq_id seq_id = ubatch.seq_id[s][0];
                            if (embd_seq_out.find(seq_id) != embd_seq_out.end()) {
                                continue;
                            }
                            embd_seq_out[seq_id].resize(n_embd);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd*seq_id)*sizeof(float), n_embd*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - a single float per sequence
                        auto & embd_seq_out = embd_seq;

                        for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
                            const llama_seq_id seq_id = ubatch.seq_id[s][0];
                            if (embd_seq_out.find(seq_id) != embd_seq_out.end()) {
                                continue;
                            }
                            embd_seq_out[seq_id].resize(1);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (seq_id)*sizeof(float), sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        n_outputs_prev += n_outputs;
    }

    // finalize the batch processing
    bg.done();

    // set output mappings
    {
        bool sorted_output = true;

        GGML_ASSERT(sbatch.out_ids.size() == (size_t) n_outputs_all);

        for (int64_t i = 0; i < n_outputs_all; ++i) {
            int64_t out_id = sbatch.out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        if (sorted_output) {
            sbatch.out_ids.clear();
        }
    }

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    // decide if we need to defrag the kv cache
    if (cparams.causal_attn && cparams.defrag_thold > 0.0f) {
        // - do not defrag small contexts (i.e. < 2048 tokens)
        // - count the padding towards the number of used tokens
        const float fragmentation = kv_self.n >= 2048 ? std::max(0.0f, 1.0f - float(kv_self.used + get_ctx_padding(cparams))/float(kv_self.n)) : 0.0f;

        // queue defragmentation for next llama_kv_cache_update
        if (fragmentation > cparams.defrag_thold) {
            LLAMA_LOG_DEBUG("%s: fragmentation: %.2f - requesting defrag\n", __func__, fragmentation);

            kv_self.defrag();
        }
    }

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(sched.get());

    return 0;
}

llama_pos llama_context_kv_self::pos_max() const {
    return kv_self.pos_max();
}

uint32_t llama_context_kv_self::get_ctx_padding(const llama_cparams & cparams) const {
    return kv_self.get_padding(cparams);
}

// llama input

void llama_context_kv_self::input_set(const llama_ubatch & ubatch) {
    const llama_hparams & hparams = model.hparams;

    if (inp_k_shift) {
        assert(ggml_backend_buffer_is_host(inp_k_shift->buffer));

        int32_t * data = (int32_t *) inp_k_shift->data;

        for (uint32_t i = 0; i < kv_self.size; ++i) {
            data[i] = kv_self.cells[i].delta;
        }

        // the K-shift graph requires just this input
        return;
    }

    // call base functionality
    llama_context::input_set(ubatch);

    if (inp_KQ_mask || inp_KQ_mask_swa) {
        // NOTE: hparams.causal_attn indicates the model is capable of generation and uses the kv cache.
        if (cparams.causal_attn && !is_encoding) {
            const int64_t n_kv         = kv_self.n;
            const int64_t n_tokens     = ubatch.n_tokens;
            const int64_t n_seq_tokens = ubatch.n_seq_tokens;
            const int64_t n_seqs       = ubatch.n_seqs;

            float * data     = nullptr;
            float * data_swa = nullptr;

            if (inp_KQ_mask) {
                GGML_ASSERT(ggml_backend_buffer_is_host(inp_KQ_mask->buffer));
                data = (float *) inp_KQ_mask->data;
            }

            if (inp_KQ_mask_swa) {
                GGML_ASSERT(ggml_backend_buffer_is_host(inp_KQ_mask_swa->buffer));
                data_swa = (float *) inp_KQ_mask_swa->data;
            }

            // For causal attention, use only the previous KV cells
            // of the correct sequence for each token of the ubatch.
            // It's assumed that if a token in the batch has multiple sequences, they are equivalent.
            for (int h = 0; h < 1; ++h) {
                for (int s = 0; s < n_seqs; ++s) {
                    const llama_seq_id seq_id = ubatch.seq_id[s][0];

                    for (int j = 0; j < n_seq_tokens; ++j) {
                        const llama_pos pos = ubatch.pos[s*n_seq_tokens + j];

                        for (int i = 0; i < n_kv; ++i) {
                            float f;
                            if (!kv_self.cells[i].has_seq_id(seq_id) || kv_self.cells[i].pos > pos) {
                                f = -INFINITY;
                            } else {
                                if (hparams.use_alibi) {
                                    f = -std::abs(kv_self.cells[i].pos - pos);
                                } else {
                                    f = 0.0f;
                                }
                            }

                            if (data) {
                                data[h*(n_kv*n_tokens) + s*(n_kv*n_seq_tokens) + j*n_kv + i] = f;
                            }

                            // may need to cut off old tokens for sliding window
                            if (data_swa) {
                                if (pos - kv_self.cells[i].pos >= (int32_t)hparams.n_swa) {
                                    f = -INFINITY;
                                }
                                data_swa[h*(n_kv*n_tokens) + s*(n_kv*n_seq_tokens) + j*n_kv + i] = f;
                            }
                        }
                    }
                }

                if (data) {
                    for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                        for (int j = 0; j < n_kv; ++j) {
                            data[h*(n_kv*n_tokens) + i*n_kv + j] = -INFINITY;
                        }
                    }
                }

                if (data_swa) {
                    for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                        for (int j = 0; j < n_kv; ++j) {
                            data_swa[h*(n_kv*n_tokens) + i*n_kv + j] = -INFINITY;
                        }
                    }
                }
            }
        } else {
            const int64_t n_tokens     = ubatch.n_tokens;
            const int64_t n_seq_tokens = ubatch.n_seq_tokens;
            const int64_t n_seqs       = ubatch.n_seqs;
            // when using kv cache, the mask needs to match the kv cache size
            const int64_t n_stride = hparams.causal_attn && !is_encoding ? kv_self.n : n_tokens;

            GGML_ASSERT(ggml_backend_buffer_is_host(inp_KQ_mask->buffer));

            float * data = (float *) inp_KQ_mask->data;

            for (int h = 0; h < 1; ++h) {
                for (int s1 = 0; s1 < n_seqs; ++s1) {
                    const llama_seq_id seq_id = ubatch.seq_id[s1][0];

                    for (int j = 0; j < n_seq_tokens; ++j) {
                        const int32_t tj = s1*n_seq_tokens + j;

                        for (int s0 = 0; s0 < n_seqs; ++s0) {
                            for (int i = 0; i < n_seq_tokens; ++i) {
                                const int32_t ti = s0*n_seq_tokens + i;
                                float f = -INFINITY;

                                for (int s = 0; s < ubatch.n_seq_id[s0]; ++s) {
                                    if (ubatch.seq_id[s0][s] == seq_id) {
                                        if (hparams.use_alibi) {
                                            f = -std::abs(ubatch.pos[ti] - ubatch.pos[tj]);
                                        } else {
                                            f = 0.0f;
                                        }
                                        break;
                                    }
                                }

                                data[h*(n_tokens*n_tokens) + tj*n_stride + ti] = f;
                            }
                        }

                        for (int i = n_tokens; i < n_stride; ++i) {
                            data[h*(n_tokens*n_tokens) + tj*n_stride + i] = -INFINITY;
                        }
                    }
                }
            }
        }
    }

    if (kv_self.recurrent) {
        const int64_t n_kv = kv_self.n;

        if (inp_s_mask) {
            GGML_ASSERT(ggml_backend_buffer_is_host(inp_s_mask->buffer));
            float * data = (float *) inp_s_mask->data;

            // clear unused states
            for (int i = 0; i < n_kv; ++i) {
                const uint32_t  cell_id = i + kv_self.head;
                llama_kv_cell & kv_cell = kv_self.cells[cell_id];

                data[i] = (float) (kv_cell.src >= 0);

                // TODO: do not mutate the KV cache
                // only clear once
                if (kv_cell.src < 0) {
                    kv_cell.src = cell_id;
                }
            }
        }

        if (inp_s_copy) {
            GGML_ASSERT(ggml_backend_buffer_is_host(inp_s_copy->buffer));
            int32_t * data = (int32_t *) inp_s_copy->data;

            // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
            for (uint32_t i = 0; i < n_kv; ++i) {
                const uint32_t  cell_id = i + kv_self.head;
                llama_kv_cell & kv_cell = kv_self.cells[cell_id];

                // prevent out-of-bound sources
                if (kv_cell.src < 0 || (uint32_t) kv_cell.src >= kv_self.size) {
                    kv_cell.src = cell_id;
                }

                data[i] = kv_cell.src;

                // TODO: do not mutate the KV cache
                // ensure copy only happens once
                if (kv_cell.src != (int32_t) cell_id) {
                    kv_cell.src = cell_id;
                }
            }
        }
    }

    if (inp_pos_bucket) {
        const int64_t n_tokens = ubatch.n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(inp_pos_bucket->buffer));
        GGML_ASSERT(!ubatch.equal_seqs); // TODO: use ubatch.n_seqs instead of failing

        static const auto relative_position_bucket = [](llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
            // TODO move to hparams if a T5 variant appears that uses a different value
            const int64_t max_distance = 128;

            if (bidirectional) {
                n_buckets >>= 1;
            }

            const int64_t max_exact = n_buckets >> 1;

            int32_t relative_position = x - y;
            int32_t relative_bucket = 0;
            if (bidirectional) {
                relative_bucket += (relative_position > 0) * n_buckets;
                relative_position = abs(relative_position);
            } else {
                relative_position = -std::min<int32_t>(relative_position, 0);
            }
            int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
            relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
            relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);
            return relative_bucket;
        };

        int32_t * data = (int32_t *) inp_pos_bucket->data;

        if (!is_encoding) {
            const int64_t n_kv = kv_self.n;
            for (int h = 0; h < 1; ++h) {
                for (int j = 0; j < n_tokens; ++j) {
                    for (int i = 0; i < n_kv; ++i) {
                        data[h*(n_kv*n_tokens) + j*n_kv + i] = relative_position_bucket(kv_self.cells[i].pos, ubatch.pos[j], hparams.n_rel_attn_bkts, is_encoding);
                    }
                }
            }
        } else {
            for (int h = 0; h < 1; ++h) {
                for (int j = 0; j < n_tokens; ++j) {
                    for (int i = 0; i < n_tokens; ++i) {
                        data[h*(n_tokens*n_tokens) + j*n_tokens + i] = relative_position_bucket(ubatch.pos[i], ubatch.pos[j], hparams.n_rel_attn_bkts, is_encoding);
                    }
                }
            }
        }
    }

    if (!is_encoding && inp_embd_enc) {
        assert(inp_embd_enc->type == GGML_TYPE_F32);
        assert((size_t) ggml_nelements(inp_embd_enc) == embd_enc.size());

        ggml_backend_tensor_set(inp_embd_enc, embd_enc.data(), 0, ggml_nbytes(inp_embd_enc));
    }

    if (!is_encoding && inp_KQ_mask_cross) {
        const int64_t n_output_enc = embd_enc.size() / hparams.n_embd;
        const int64_t n_tokens = ubatch.n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(inp_KQ_mask_cross->buffer));
        GGML_ASSERT(!ubatch.equal_seqs); // TODO: use ubatch.n_seqs instead of failing

        float * data = (float *) inp_KQ_mask_cross->data;

        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                for (int i = 0; i < n_output_enc; ++i) {
                    float f = -INFINITY;
                    for (int s = 0; s < ubatch.n_seq_id[j]; ++s) {
                        const llama_seq_id seq_id = ubatch.seq_id[j][s];
                        if (seq_ids_enc[i].find(seq_id) != seq_ids_enc[i].end()) {
                            f = 0.0f;
                        }
                    }
                    data[h*(n_output_enc*n_tokens) + j*n_output_enc + i] = f;
                }
            }

            for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                for (int j = 0; j < n_output_enc; ++j) {
                    data[h*(n_output_enc*n_tokens) + i*n_output_enc + j] = -INFINITY;
                }
            }
        }
    }
}

void llama_context_kv_self::kv_self_update() {
    auto & kv = kv_self;

    if (kv.has_shift) {
        if (!kv.can_shift) {
            GGML_ABORT("The current context does not support K-shift");
        }

        // apply K-shift if needed
        if (model.hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched.get());

            auto * gf = graph_init();

            kv_self.build_shift(ctx_compute.get(), gf, this);

            ggml_backend_sched_alloc_graph(sched.get(), gf);

            input_set({});

            graph_compute(gf, false);

            need_reserve = true;
        }

        {
            kv.has_shift = false;

            for (uint32_t i = 0; i < kv.size; ++i) {
                kv.cells[i].delta = 0;
            }
        }
    }

    // defragment the KV cache if needed
    if (kv.do_defrag) {
        ggml_backend_sched_reset(sched.get());

        auto * gf = graph_init();

        kv_self.build_defrag(ctx_compute.get(), gf, max_nodes(), !cparams.flash_attn);

        ggml_backend_sched_alloc_graph(sched.get(), gf);

        // no input
        //input_set({});

        graph_compute(gf, false);

        kv.do_defrag = false;

        need_reserve = true;
    }
}

ggml_tensor * llama_context_kv_self::build_inp_k_shift(ggml_context * ctx0) {
    inp_k_shift = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ctx());
    ggml_set_input(inp_k_shift);

    return inp_k_shift;
}

void llama_context_kv_self::build_attn_inp(
        ggml_context * ctx0,
             int32_t   n_tokens,
                bool   causal,
                bool   swa,
                bool   worst_case) {
    const auto & hparams = model.hparams;

    const auto n_kv = worst_case ? kv_self.size : kv_self.n;

    inp_KQ_mask = causal
        ? ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv,     GGML_PAD(n_tokens, GGML_KQ_MASK_PAD))
        : ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
    //cb(inp_KQ_mask, "KQ_mask", -1);
    ggml_set_input(inp_KQ_mask);

    inp_KQ_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp_KQ_mask, GGML_TYPE_F16) : inp_KQ_mask;

    if (swa) {
        GGML_ASSERT(hparams.n_swa > 0);

        inp_KQ_mask_swa = causal
            ? ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv,     GGML_PAD(n_tokens, GGML_KQ_MASK_PAD))
            : ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
        //cb(inp_KQ_mask_swa, "KQ_mask_swa", -1);
        ggml_set_input(inp_KQ_mask_swa);

        inp_KQ_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp_KQ_mask_swa, GGML_TYPE_F16) : inp_KQ_mask_swa;
    }
}

void llama_context_kv_self::build_attn_kv_store(
        ggml_context * ctx0,
         ggml_cgraph * graph,
         ggml_tensor * k_cur,
         ggml_tensor * v_cur,
             int32_t   n_tokens,
             int64_t   il,
             bool      worst_case) {
    const auto & hparams = model.hparams;

    const auto & n_ctx = cparams.n_ctx;

    const auto kv_head = worst_case ? (kv_self.recurrent ? 0 : kv_self.size - n_tokens) : kv_self.head;

    const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

    GGML_ASSERT(kv_self.size == n_ctx);

    struct ggml_tensor * k_cache_view = ggml_view_1d(ctx0, kv_self.k_l[il], n_tokens*n_embd_k_gqa, ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa)*kv_head);
    //cb(k_cache_view, "k_cache_view", il);

    // note: storing RoPE-ed version of K in the KV cache
    ggml_build_forward_expand(graph, ggml_cpy(ctx0, k_cur, k_cache_view));

    assert(v_cur->ne[0] == n_embd_v_gqa && v_cur->ne[1] == n_tokens);

    struct ggml_tensor * v_cache_view = nullptr;

    if (cparams.flash_attn) {
        v_cache_view = ggml_view_1d(ctx0, kv_self.v_l[il], n_tokens*n_embd_v_gqa, ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa)*kv_head);
    } else {
        // note: the V cache is transposed when not using flash attention
        v_cache_view = ggml_view_2d(ctx0, kv_self.v_l[il], n_tokens, n_embd_v_gqa,
                (  n_ctx)*ggml_element_size(kv_self.v_l[il]),
                (kv_head)*ggml_element_size(kv_self.v_l[il]));

        v_cur = ggml_transpose(ctx0, v_cur);
    }
    //cb(v_cache_view, "v_cache_view", il);

    ggml_build_forward_expand(graph, ggml_cpy(ctx0, v_cur, v_cache_view));
}

ggml_tensor * llama_context_kv_self::build_attn_qkv(
        ggml_context * ctx0,
         ggml_cgraph * graph,
         ggml_tensor * wo,
         ggml_tensor * wo_b,
         ggml_tensor * q_cur,
             int32_t   n_tokens,
             float     kq_scale,
             int       il,
             bool      worst_case) {
    const auto & hparams = model.hparams;

    const auto & n_ctx         = cparams.n_ctx;
    const auto & n_embd_head_k = hparams.n_embd_head_k;
    const auto & n_embd_head_v = hparams.n_embd_head_v;

    // TODO: improve
    bool is_sliding = false;

    switch (model.arch) {
        case LLM_ARCH_COHERE2:
            {
                const int32_t sliding_window_pattern = 4;
                is_sliding = il % sliding_window_pattern < (sliding_window_pattern - 1);
            } break;
        case LLM_ARCH_GEMMA2:
            {
                const int32_t sliding_window_pattern = 2;
                is_sliding = il % sliding_window_pattern < (sliding_window_pattern - 1);
            } break;
        case LLM_ARCH_PHI3:
            {
                is_sliding = hparams.n_swa > 0;
            } break;
        default:
            {
                is_sliding = false;
            }
    };

    const auto & kq_mask = is_sliding ? inp_KQ_mask_swa_cnv : inp_KQ_mask_cnv;

    const auto n_kv = worst_case ? kv_self.size : kv_self.n;

    const int64_t n_head       = hparams.n_head(il);
    const int64_t n_head_kv    = hparams.n_head_kv(il);
    const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

    struct ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    //cb(q, "q", il);

    struct ggml_tensor * k =
        ggml_view_3d(ctx0, kv_self.k_l[il],
                n_embd_head_k, n_kv, n_head_kv,
                ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                ggml_row_size(kv_self.k_l[il]->type, n_embd_head_k),
                0);
    //cb(k, "k", il);

    struct ggml_tensor * cur;

    if (cparams.flash_attn) {
        GGML_UNUSED(model);
        GGML_UNUSED(n_ctx);

        // split cached v into n_head heads (not transposed)
        struct ggml_tensor * v =
            ggml_view_3d(ctx0, kv_self.v_l[il],
                    n_embd_head_v, n_kv, n_head_kv,
                    ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa),
                    ggml_row_size(kv_self.v_l[il]->type, n_embd_head_v),
                    0);
        //cb(v, "v", il);

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);

        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

        cur = ggml_reshape_2d(ctx0, cur, n_embd_head_v*n_head, n_tokens);
    } else {
        struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        //cb(kq, "kq", il);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        if (model.arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplyer of 0.08838834764831845
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, 0.08838834764831845f/30.0f));
            kq = ggml_scale(ctx0, kq, 30);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            kq = ggml_tanh(ctx0, kq);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        //cb(kq, "kq_soft_max_ext", il);

        GGML_ASSERT(kv_self.size == n_ctx);

        // split cached v into n_head heads
        struct ggml_tensor * v =
            ggml_view_3d(ctx0, kv_self.v_l[il],
                    n_kv, n_embd_head_v, n_head_kv,
                    ggml_element_size(kv_self.v_l[il])*n_ctx,
                    ggml_element_size(kv_self.v_l[il])*n_ctx*n_embd_head_v,
                    0);
        //cb(v, "v", il);

        struct ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        //cb(kqv, "kqv", il);

        struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        //cb(kqv_merged, "kqv_merged", il);

        cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_head_v*n_head, n_tokens);
        //cb(cur, "kqv_merged_cont", il);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend_cpu);
        }
    }

    ggml_build_forward_expand(graph, cur);

    if (wo) {
        cur = build_lora_mm(ctx0, wo, cur);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llama_context_kv_self::build_attn_soft_max(
        ggml_context * ctx0,
         ggml_tensor * kq,
             float     kq_scale) {
    const auto & hparams = model.hparams;

    return ggml_soft_max_ext(ctx0, kq, inp_KQ_mask_cnv, kq_scale, hparams.f_max_alibi_bias);
}

ggml_tensor * llama_context_kv_self::build_inp_embd_enc(
        ggml_context * ctx0,
             int32_t   n_tokens,
                bool   worst_case) {
    const auto & hparams = model.hparams;
    const int64_t n_embd = hparams.n_embd;

    // TODO: not sure if this is correct
    const int32_t n_outputs_enc = worst_case ? n_tokens : embd_enc.size() / n_embd;

    inp_embd_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_outputs_enc);
    ggml_set_input(inp_embd_enc);

    return inp_embd_enc;
}

ggml_tensor * llama_context_kv_self::build_inp_KQ_mask_cross(
            ggml_context * ctx0,
                 int32_t   n_tokens,
                 bool   worst_case) {
    const auto & hparams = model.hparams;
    const int64_t n_embd = hparams.n_embd;

    // TODO: not sure if this is correct
    const int32_t n_outputs_enc = worst_case ? n_tokens : embd_enc.size() / n_embd;

    inp_KQ_mask_cross = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_outputs_enc, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
    ggml_set_input(inp_KQ_mask_cross);

    return inp_KQ_mask_cross;
}

ggml_tensor * llama_context_kv_self::build_inp_s_copy(
        ggml_context * ctx0,
                bool   worst_case) {
    const auto n_kv    = worst_case ? kv_self.size : kv_self.n;

    inp_s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_kv);
    //cb(inp_s_copy, "inp_s_copy", -1);
    ggml_set_input(inp_s_copy);
    return inp_s_copy;
}

ggml_tensor * llama_context_kv_self::build_inp_s_mask(
        ggml_context * ctx0,
                bool   worst_case) {
    const auto n_kv    = worst_case ? kv_self.size : kv_self.n;
    inp_s_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_kv);
    //cb(inp_s_mask, "inp_s_mask", -1);
    ggml_set_input(inp_s_mask);
    return inp_s_mask;
}

ggml_tensor * llama_context_kv_self::build_copy_mask_state(
        ggml_context * ctx0,
         ggml_cgraph * gf,
         ggml_tensor * s,
         ggml_tensor * state_copy,
         ggml_tensor * state_mask,
             int32_t   n_tokens,
             int32_t   n_state,
             int32_t   n_seqs,
                bool   worst_case) {
    const auto n_kv    = worst_case ? kv_self.size : kv_self.n;
    const auto kv_head = worst_case ? (kv_self.recurrent ? 0 : kv_self.size - n_tokens) : kv_self.head;

    struct ggml_tensor * states = ggml_reshape_2d(ctx0, s, n_state, kv_self.size);

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between kv_head and kv_head + n_kv
    // this shrinks the tensors's ne[1] to n_kv
    states = ggml_get_rows(ctx0, states, state_copy);

    // clear states of sequences which are starting at the beginning of this batch
    // FIXME: zero-out NANs?
    states = ggml_mul(ctx0, states, state_mask);

    // copy states which won't be changed further (between n_seqs and n_kv)
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            ggml_view_1d(ctx0, states, n_state*(n_kv - n_seqs), (n_seqs          )*n_state*ggml_element_size(states)),
            ggml_view_1d(ctx0, s,      n_state*(n_kv - n_seqs), (kv_head + n_seqs)*n_state*ggml_element_size(s))));

    // the part of the states that will be used and modified
    return ggml_view_2d(ctx0, states, n_state, n_seqs, states->nb[1], 0);
}

// TODO: split
ggml_tensor * llama_context_kv_self::build_mamba_layer(
        ggml_context * ctx0,
         ggml_cgraph * gf,
         ggml_tensor * cur,
         ggml_tensor * state_copy,
         ggml_tensor * state_mask,
  const llama_ubatch & ubatch,
                 int   il,
                bool   worst_case) {
    const auto & hparams = model.hparams;

    const auto & n_tokens = ubatch.n_tokens;

    const auto kv_head = worst_case ? (kv_self.recurrent ? 0 : kv_self.size - n_tokens) : kv_self.head;

    const int64_t d_conv  = hparams.ssm_d_conv;
    const int64_t d_inner = hparams.ssm_d_inner;
    const int64_t d_state = hparams.ssm_d_state;
    const int64_t dt_rank = hparams.ssm_dt_rank;
    const int64_t n_seqs  = ubatch.n_seqs;
    // Some variants of Mamba arch (e.g. FalconMamba do apply layer norm on B and Dt layers)
    const bool ssm_dt_b_c_rms = hparams.ssm_dt_b_c_rms;
    // Use the same RMS norm as the final layer norm
    const float norm_rms_eps = hparams.f_norm_rms_eps;

    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs);
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    struct ggml_tensor * conv_states_all = kv_self.k_l[il];
    struct ggml_tensor * ssm_states_all  = kv_self.v_l[il];

    // (ab)using the KV cache to store the states
    struct ggml_tensor * conv = build_copy_mask_state(
            ctx0, gf, conv_states_all, state_copy, state_mask,
            n_tokens, hparams.n_embd_k_s(), n_seqs, worst_case);
    conv = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner, n_seqs);
    struct ggml_tensor * ssm = build_copy_mask_state(
            ctx0, gf, ssm_states_all, state_copy, state_mask,
            n_tokens, hparams.n_embd_v_s(), n_seqs, worst_case);
    ssm = ggml_reshape_3d(ctx0, ssm, d_state, d_inner, n_seqs);

    // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

    // {n_embd, 2*d_inner} @ {n_embd, n_seq_tokens, n_seqs} => {2*d_inner, n_seq_tokens, n_seqs}
    struct ggml_tensor * xz = build_lora_mm(ctx0, model.layers[il].ssm_in, cur);
    // split the above in two
    // => {d_inner, n_seq_tokens, n_seqs}
    struct ggml_tensor * x = ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], 0);
    struct ggml_tensor * z = ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], d_inner*ggml_element_size(xz));

    // conv
    {
        // => {d_conv - 1 + n_seq_tokens, d_inner, n_seqs}
        struct ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, x), 0);

        // copy last (d_conv - 1) columns back into the state cache
        struct ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs, conv_x->nb[1], conv_x->nb[2], n_seq_tokens*(conv_x->nb[0]));

        ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, last_conv,
                ggml_view_1d(ctx0, conv_states_all,
                    (d_conv - 1)*(d_inner)*(n_seqs),
                    kv_head*(d_conv - 1)*(d_inner)*ggml_element_size(conv_states_all))));

        // 1D convolution
        // The equivalent is to make a self-overlapping view of conv_x
        // over d_conv columns at each stride in the 3rd dimension,
        // then element-wise multiply that with the conv1d weight,
        // then sum the elements of each row,
        // (the last two steps are a dot product over rows (also doable with mul_mat))
        // then permute away the ne[0] dimension,
        // and then you're left with the resulting x tensor.
        // For simultaneous sequences, all sequences need to have the same length.
        x = ggml_ssm_conv(ctx0, conv_x, model.layers[il].ssm_conv1d);

        // bias
        x = ggml_add(ctx0, x, model.layers[il].ssm_conv1d_b);

        x = ggml_silu(ctx0, x);
    }

    // ssm
    {
        // {d_inner, dt_rank + 2*d_state} @ {d_inner, n_seq_tokens, n_seqs} => {dt_rank + 2*d_state, n_seq_tokens, n_seqs}
        struct ggml_tensor * x_db = build_lora_mm(ctx0, model.layers[il].ssm_x, x);
        // split
        struct ggml_tensor * dt = ggml_view_3d(ctx0, x_db, dt_rank, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], 0);
        struct ggml_tensor * B  = ggml_view_3d(ctx0, x_db, d_state, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], ggml_element_size(x_db)*dt_rank);
        struct ggml_tensor * C  = ggml_view_3d(ctx0, x_db, d_state, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], ggml_element_size(x_db)*(dt_rank+d_state));

        // Some Mamba variants (e.g. FalconMamba) apply RMS norm in B, C & Dt layers
        if (ssm_dt_b_c_rms) {
            dt = ggml_rms_norm(ctx0, dt, norm_rms_eps);
            B = ggml_rms_norm(ctx0, B, norm_rms_eps);
            C = ggml_rms_norm(ctx0, C, norm_rms_eps);
        }

        // {dt_rank, d_inner} @ {dt_rank, n_seq_tokens, n_seqs} => {d_inner, n_seq_tokens, n_seqs}
        dt = build_lora_mm(ctx0, model.layers[il].ssm_dt, dt);
        dt = ggml_add(ctx0, dt, model.layers[il].ssm_dt_b);

        // Custom operator to optimize the parallel associative scan
        // as described in the Annex D of the Mamba paper.
        // => {d_inner, n_seq_tokens, n_seqs} and {d_state, d_inner, n_seqs}
        struct ggml_tensor * y_ssm = ggml_ssm_scan(ctx0, ssm, x, dt, model.layers[il].ssm_a, B, C);

        // store last states
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx0,
                ggml_view_1d(ctx0, y_ssm, d_state*d_inner*n_seqs, x->nb[3]),
                ggml_view_1d(ctx0, ssm_states_all, d_state*d_inner*n_seqs, kv_head*d_state*d_inner*ggml_element_size(ssm_states_all))));

        struct ggml_tensor * y = ggml_view_3d(ctx0, y_ssm, d_inner, n_seq_tokens, n_seqs, x->nb[1], x->nb[2], 0);

        // TODO: skip computing output earlier for unused tokens

        // {d_inner, n_seq_tokens, n_seqs} * {d_inner} => {d_inner, n_seq_tokens, n_seqs}
        y = ggml_add(ctx0, y, ggml_mul(ctx0, x, model.layers[il].ssm_d));
        y = ggml_mul(ctx0, y, ggml_silu(ctx0, ggml_cont(ctx0, z)));

        // {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
        cur = build_lora_mm(ctx0, model.layers[il].ssm_out, y);
    }

    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
    //cb(cur, "mamba_out", il);

    return cur;
}


ggml_tensor * llama_context_kv_self::build_rwkv_token_shift_load(
        ggml_context * ctx0,
         ggml_cgraph * gf,
         ggml_tensor * state_copy,
         ggml_tensor * state_mask,
  const llama_ubatch & ubatch,
                 int   il,
                bool   worst_case) {
    const auto & hparams = model.hparams;

    const auto token_shift_count = hparams.token_shift_count;

    const auto & n_tokens = ubatch.n_tokens;
    const int64_t n_seqs  = ubatch.n_seqs;

    struct ggml_tensor * token_shift_all = kv_self.k_l[il];

    struct ggml_tensor * token_shift = build_copy_mask_state(
            ctx0, gf, token_shift_all, state_copy, state_mask,
            n_tokens, hparams.n_embd_k_s(), n_seqs, worst_case);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}


ggml_tensor * llama_context_kv_self::build_rwkv_token_shift_store(
        ggml_context * ctx0,
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il,
                bool   worst_case) {
    const auto & hparams = model.hparams;

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const auto & n_tokens = ubatch.n_tokens;
    const int64_t n_seqs  = ubatch.n_seqs;

    const auto kv_head = worst_case ? (kv_self.recurrent ? 0 : kv_self.size - n_tokens) : kv_self.head;

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, kv_self.k_l[il], hparams.n_embd_k_s() * n_seqs, hparams.n_embd_k_s() * kv_head * ggml_element_size(kv_self.k_l[il]))
    );
}


ggml_tensor * llama_context_kv_self::build_rwkv6_time_mix(
        ggml_context * ctx0,
         ggml_cgraph * gf,
         ggml_tensor * cur,
         ggml_tensor * x_prev,
         ggml_tensor * state_copy,
         ggml_tensor * state_mask,
  const llama_ubatch & ubatch,
                 int   il,
                bool   worst_case) {
    const auto & hparams = model.hparams;

    const auto n_tokens = ubatch.n_tokens;
    const auto n_seqs = ubatch.n_seqs;
    const auto n_embd = hparams.n_embd;
    const auto head_size = hparams.wkv_head_size;
    const auto n_head = n_embd / head_size;
    const auto n_head_kv = hparams.n_head_kv(il);

    const auto kv_head = worst_case ? (kv_self.recurrent ? 0 : kv_self.size - n_tokens) : kv_self.head;

    const auto & layer = model.layers[il];

    bool is_qrwkv = layer.time_mix_first == nullptr;

    struct ggml_tensor * sx = ggml_sub(ctx0, x_prev, cur);
    struct ggml_tensor * xxx = ggml_add(ctx0, ggml_mul(ctx0, sx, layer.time_mix_lerp_x), cur);

    xxx = ggml_reshape_4d(
        ctx0,
        ggml_tanh(
            ctx0,
            ggml_mul_mat(ctx0, layer.time_mix_w1, xxx)
        ),
        layer.time_mix_w1->ne[1] / 5, 1, 5, n_tokens
    );

    xxx = ggml_cont(ctx0, ggml_permute(ctx0, xxx, 0, 1, 3, 2));

    xxx = ggml_mul_mat(
        ctx0,
        ggml_reshape_4d(
            ctx0,
            layer.time_mix_w2,
            layer.time_mix_w2->ne[0], layer.time_mix_w2->ne[1], 1, 5
        ),
        xxx
    );

    struct ggml_tensor *xw, *xk, *xv, *xr, *xg;
    if (layer.time_mix_lerp_fused) {
        // fusing these weights makes some performance improvement
        sx  = ggml_reshape_3d(ctx0, sx,  n_embd, 1, n_tokens);
        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
        xxx = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xxx, layer.time_mix_lerp_fused), sx), cur);
        xw = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], 0);
        xk = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * sizeof(float));
        xv = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 2 * sizeof(float));
        xr = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 3 * sizeof(float));
        xg = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 4 * sizeof(float));
    } else {
        // for backward compatibility
        xw = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], 0);
        xk = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * sizeof(float));
        xv = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 2 * sizeof(float));
        xr = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 3 * sizeof(float));
        xg = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 4 * sizeof(float));

        xw = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xw, layer.time_mix_lerp_w), sx), cur);
        xk = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xk, layer.time_mix_lerp_k), sx), cur);
        xv = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xv, layer.time_mix_lerp_v), sx), cur);
        xr = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xr, layer.time_mix_lerp_r), sx), cur);
        xg = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xg, layer.time_mix_lerp_g), sx), cur);
    }

    struct ggml_tensor * r = build_lora_mm(ctx0, layer.time_mix_receptance, xr);
    struct ggml_tensor * k = build_lora_mm(ctx0, layer.time_mix_key,        xk);
    struct ggml_tensor * v = build_lora_mm(ctx0, layer.time_mix_value,      xv);
    if (layer.time_mix_receptance_b) {
        r = ggml_add(ctx0, r, layer.time_mix_receptance_b);
    }
    if (layer.time_mix_key_b) {
        k = ggml_add(ctx0, k, layer.time_mix_key_b);
    }
    if (layer.time_mix_value_b) {
        v = ggml_add(ctx0, v, layer.time_mix_value_b);
    }

    struct ggml_tensor * g = build_lora_mm(ctx0, layer.time_mix_gate, xg);
    if (is_qrwkv) {
        g = ggml_sigmoid(ctx0, g);
    } else {
        g = ggml_silu(ctx0, g);
    }

    if (n_head_kv != 0 && n_head_kv != n_head) {
        GGML_ASSERT(n_head % n_head_kv == 0);
        k = ggml_reshape_4d(ctx0, k, head_size, 1, n_head_kv, n_tokens);
        v = ggml_reshape_4d(ctx0, v, head_size, 1, n_head_kv, n_tokens);
        struct ggml_tensor * tmp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_size, n_head / n_head_kv, n_head_kv, n_tokens);
        k = ggml_repeat(ctx0, k, tmp);
        v = ggml_repeat(ctx0, v, tmp);
    }

    k = ggml_reshape_3d(ctx0, k, head_size, n_head, n_tokens);
    v = ggml_reshape_3d(ctx0, v, head_size, n_head, n_tokens);
    r = ggml_reshape_3d(ctx0, r, head_size, n_head, n_tokens);

    struct ggml_tensor * w = ggml_mul_mat(
        ctx0,
        layer.time_mix_decay_w2,
        ggml_tanh(
            ctx0,
            ggml_mul_mat(ctx0, layer.time_mix_decay_w1, xw)
        )
    );

    w = ggml_add(ctx0, w, layer.time_mix_decay);
    w = ggml_exp(ctx0, ggml_neg(ctx0, ggml_exp(ctx0, w)));
    w = ggml_reshape_3d(ctx0, w, head_size, n_head, n_tokens);

    if (is_qrwkv) {
        // k = k * (1 - w)
        k = ggml_sub(ctx0, k, ggml_mul(ctx0, k, w));
    }

    struct ggml_tensor * wkv_state = build_copy_mask_state(
            ctx0, gf, kv_self.v_l[il], state_copy, state_mask,
            n_tokens, hparams.n_embd_v_s(), n_seqs, worst_case);

    struct ggml_tensor * wkv_output;
    if (is_qrwkv) {
        wkv_output = ggml_gated_linear_attn(ctx0, k, v, r, w, wkv_state, pow(head_size, -0.5f));
    } else {
        wkv_output = ggml_rwkv_wkv6(ctx0, k, v, r, layer.time_mix_first, w, wkv_state);
    }
    cur = ggml_view_1d(ctx0, wkv_output, n_embd * n_tokens, 0);
    wkv_state = ggml_view_1d(ctx0, wkv_output, n_embd * head_size * n_seqs, n_embd * n_tokens * sizeof(float));

    ggml_build_forward_expand(
        gf,
        ggml_cpy(
            ctx0,
            wkv_state,
            ggml_view_1d(
                ctx0,
                kv_self.v_l[il],
                hparams.n_embd_v_s() * n_seqs,
                hparams.n_embd_v_s() * kv_head * ggml_element_size(kv_self.v_l[il])
            )
        )
    );

    if (!is_qrwkv) {
        // group norm with head_count groups
        cur = ggml_reshape_3d(ctx0, cur, n_embd / n_head, n_head, n_tokens);
        cur = ggml_norm(ctx0, cur, 64e-5f);

        // Convert back to regular vectors.
        cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.time_mix_ln), layer.time_mix_ln_b);
    } else {
        cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
    }

    cur = ggml_mul(ctx0, cur, g);
    cur = build_lora_mm(ctx0, layer.time_mix_output, cur);

    return cur;
}

// state save/load

size_t llama_context_kv_self::state_get_data(llama_io_write_i & io) {
    llama_context::state_get_data(io);

    kv_self.state_write(io);

    return io.n_bytes();
}

size_t llama_context_kv_self::state_set_data(llama_io_read_i & io) {
    llama_context::state_set_data(io);

    kv_self.state_read(io);

    return io.n_bytes();
}

size_t llama_context_kv_self::state_seq_get_data(llama_io_write_i & io, llama_seq_id seq_id) {
    llama_context::state_seq_get_data(io, seq_id);

    kv_self.state_write(io, seq_id);

    return io.n_bytes();
}

size_t llama_context_kv_self::state_seq_set_data(llama_io_read_i & io, llama_seq_id seq_id) {
    llama_context::state_seq_set_data(io, seq_id);

    kv_self.state_read(io, seq_id);

    return io.n_bytes();
}

//
// interface implementation
//

void llama_free(struct llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const struct llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_batch(const struct llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const struct llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const struct llama_context * ctx) {
    return ctx->n_seq_max();
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

llama_kv_cache * llama_get_kv_self(llama_context * ctx) {
    return ctx->get_kv_self();
}

void llama_kv_self_update(llama_context * ctx) {
    ctx->kv_self_update();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
        struct llama_context * ctx,
           ggml_threadpool_t   threadpool,
           ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(struct llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(struct llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(struct llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(struct llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(struct llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(struct llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(struct llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_synchronize(struct llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(struct llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(struct llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_logits_ith(i);
}

float * llama_get_embeddings(struct llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

// llama adapter API

int32_t llama_set_adapter_lora(
            struct llama_context * ctx,
            struct llama_adapter_lora * adapter,
            float scale) {
    ctx->set_adapter_lora(adapter, scale);

    return 0;
}

int32_t llama_rm_adapter_lora(
            struct llama_context * ctx,
            struct llama_adapter_lora * adapter) {
    bool res = ctx->rm_adapter_lora(adapter);

    return res ? 0 : -1;
}

void llama_clear_adapter_lora(struct llama_context * ctx) {
    ctx->clear_adapter_lora();
}

int32_t llama_apply_adapter_cvec(
        struct llama_context * ctx,
                 const float * data,
                      size_t   len,
                     int32_t   n_embd,
                     int32_t   il_start,
                     int32_t   il_end) {
    bool res = ctx->apply_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// kv cache view
//

struct llama_kv_cache_view llama_kv_cache_view_init(const llama_context * ctx, int32_t n_seq_max) {
    return llama_kv_cache_view_init(*ctx->get_kv_self(), n_seq_max);
}

void llama_kv_cache_view_update(const llama_context * ctx, llama_kv_cache_view * view) {
    llama_kv_cache_view_update(view, *ctx->get_kv_self());
}

//
// kv cache
//

// deprecated
int32_t llama_get_kv_cache_token_count(const llama_context * ctx) {
    return llama_kv_self_n_tokens(ctx);
}

int32_t llama_kv_self_n_tokens(const llama_context * ctx) {
    return llama_kv_cache_n_tokens(ctx->get_kv_self());
}

// deprecated
int32_t llama_get_kv_cache_used_cells(const llama_context * ctx) {
    return llama_kv_self_used_cells(ctx);
}

int32_t llama_kv_self_used_cells(const llama_context * ctx) {
    return llama_kv_cache_used_cells(ctx->get_kv_self());
}

// deprecated
void llama_kv_cache_clear(llama_context * ctx) {
    llama_kv_self_clear(ctx);
}

void llama_kv_self_clear(llama_context * ctx) {
    llama_kv_cache_clear(ctx->get_kv_self());
}

// deprecated
bool llama_kv_cache_seq_rm(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1) {
    return llama_kv_self_seq_rm(ctx, seq_id, p0, p1);
}

bool llama_kv_self_seq_rm(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1) {
    return llama_kv_cache_seq_rm(ctx->get_kv_self(), seq_id, p0, p1);
}

// deprecated
void llama_kv_cache_seq_cp(
        llama_context * ctx,
         llama_seq_id   seq_id_src,
         llama_seq_id   seq_id_dst,
            llama_pos   p0,
            llama_pos   p1) {
    return llama_kv_self_seq_cp(ctx, seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_self_seq_cp(
        llama_context * ctx,
         llama_seq_id   seq_id_src,
         llama_seq_id   seq_id_dst,
            llama_pos   p0,
            llama_pos   p1) {
    return llama_kv_cache_seq_cp(ctx->get_kv_self(), seq_id_src, seq_id_dst, p0, p1);
}

// deprecated
void llama_kv_cache_seq_keep(
        llama_context * ctx,
         llama_seq_id   seq_id) {
    return llama_kv_self_seq_keep(ctx, seq_id);
}

void llama_kv_self_seq_keep(llama_context * ctx, llama_seq_id seq_id) {
    return llama_kv_cache_seq_keep(ctx->get_kv_self(), seq_id);
}

// deprecated
void llama_kv_cache_seq_add(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1,
            llama_pos   delta) {
    return llama_kv_self_seq_add(ctx, seq_id, p0, p1, delta);
}

void llama_kv_self_seq_add(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1,
            llama_pos   delta) {
    return llama_kv_cache_seq_add(ctx->get_kv_self(), seq_id, p0, p1, delta);
}

// deprecated
void llama_kv_cache_seq_div(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1,
                  int   d) {
    return llama_kv_self_seq_div(ctx, seq_id, p0, p1, d);
}

void llama_kv_self_seq_div(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1,
                  int   d) {
    return llama_kv_cache_seq_div(ctx->get_kv_self(), seq_id, p0, p1, d);
}

// deprecated
llama_pos llama_kv_cache_seq_pos_max(llama_context * ctx, llama_seq_id seq_id) {
    return llama_kv_self_seq_pos_max(ctx, seq_id);
}

llama_pos llama_kv_self_seq_pos_max(llama_context * ctx, llama_seq_id seq_id) {
    return llama_kv_cache_seq_pos_max(ctx->get_kv_self(), seq_id);
}

// deprecated
void llama_kv_cache_defrag(llama_context * ctx) {
    return llama_kv_self_defrag(ctx);
}

void llama_kv_self_defrag(llama_context * ctx) {
    return llama_kv_cache_defrag(ctx->get_kv_self());
}

// deprecated
bool llama_kv_cache_can_shift(const llama_context * ctx) {
    return llama_kv_self_can_shift(ctx);
}

bool llama_kv_self_can_shift(const llama_context * ctx) {
    return llama_kv_cache_can_shift(ctx->get_kv_self());
}

// deprecated
void llama_kv_cache_update(llama_context * ctx) {
    llama_kv_self_update(ctx);
}

// llama state API

// deprecated
size_t llama_get_state_size(struct llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(struct llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(struct llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(struct llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(struct llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(struct llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(struct llama_context * ctx, llama_seq_id seq_id) {
    return ctx->state_seq_get_size(seq_id);
}

size_t llama_state_seq_get_data(struct llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size);
}

size_t llama_state_seq_set_data(struct llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size);
}

size_t llama_state_seq_save_file(struct llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(struct llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        struct llama_context * ctx,
          struct llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        struct llama_context * ctx,
          struct llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}


const std::vector<std::pair<std::string, struct ggml_tensor *>> & llama_internal_get_tensor_map(
    struct llama_context * ctx
) {
    return ctx->get_model().tensors_by_name;
}
