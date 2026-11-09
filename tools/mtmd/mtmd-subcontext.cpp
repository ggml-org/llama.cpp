#include "mtmd-subcontext.h"
#include "mtmd-graph.h"
#include "models/models.h"
#include "mtmd-audio.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

// ggml_backend_cpu_set_n_threads(backend_cpu, n_threads);
static void mtmd_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend_cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) {
        auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            ggml_backend_set_n_threads_fn(backend_cpu, n_threads);
        }
    }
}

std::shared_ptr<ggml_threadpool> mtmd_threadpool_init(int n_threads) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg || n_threads < 2) {
        return nullptr;
    }
    auto * new_fn  = (decltype(ggml_threadpool_new) *)  ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto * free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
    if (!new_fn || !free_fn) {
        return nullptr;
    }
    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    tpp.poll = 0; // the pool is idle most of the time, don't take CPU away from the text model
    ggml_threadpool_t threadpool = new_fn(&tpp);
    if (!threadpool) {
        throw std::runtime_error("failed to create threadpool");
    }
    return std::shared_ptr<ggml_threadpool>(threadpool, free_fn);
}

ggml_backend_dev_t mtmd_select_device(bool use_gpu, ggml_backend_dev_t device) {
    if (!use_gpu) {
        return nullptr;
    }
    if (device != nullptr) {
        return device;
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    return dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
}

mtmd_subcontext::mtmd_subcontext(const mtmd_model & model, const mtmd_subctx_params & params) :
        model(model),
        threadpool(params.threadpool),
        max_nodes(model.max_nodes),
        flash_attn_type(params.flash_attn_type) {
    if (flash_attn_type == MTMD_FLASH_ATTN_TYPE_AUTO && !model.flash_attn_auto) {
        flash_attn_type = MTMD_FLASH_ATTN_TYPE_DISABLED;
    }

    backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend_cpu) {
        throw std::runtime_error("failed to initialize CPU backend");
    }
    if (threadpool) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, threadpool.get());
        }
    }
    if (params.device != nullptr) {
        backend = ggml_backend_dev_init(params.device, nullptr);
        if (!backend) {
            ggml_backend_free(backend_cpu);
            throw std::runtime_error(string_format("%s: failed to initialize \"%s\" backend\n",
                                                   __func__, ggml_backend_dev_name(params.device)));
        }
    }

    if (backend) {
        LOG_INF("%s: using %s backend\n", __func__, ggml_backend_name(backend));
        backend_ptrs.push_back(backend);
        backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
    } else {
        backend = backend_cpu;
        LOG_INF("%s: using CPU backend\n", __func__);
    }

    backend_ptrs.push_back(backend_cpu);
    backend_buft.push_back(ggml_backend_get_default_buffer_type(backend_cpu));

    sched.reset(
        ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, true)
    );

    if (params.cb_eval != nullptr) {
        ggml_backend_sched_set_eval_callback(sched.get(), params.cb_eval, params.cb_eval_user_data);
    }

    debug_output_embeddings = std::getenv("MTMD_DEBUG_EMBEDDINGS") != nullptr;

    buf_compute_meta.resize(max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false));

    // TODO: fix warmup for audio gen
    if (model.modality != MTMD_MODALITY_GEN_AUDIO) {
        // check batching support
        auto batch = get_dummy_batch();
        support_batch = mtmd_get_graph_builder(this, batch)->support_batch();

        if (params.warmup) {
            warmup(batch);
        }
    }
}

mtmd_subcontext::~mtmd_subcontext() {
    // the scheduler references the backends
    sched.reset();
    ggml_backend_free(backend);
    if (backend != backend_cpu) {
        ggml_backend_free(backend_cpu);
    }
}

mtmd_image_f32_batch mtmd_subcontext::get_dummy_batch() const {
    // create a fake batch
    const auto & hparams = model.hparams;
    mtmd_image_f32_batch batch;
    mtmd_image_f32 img;
    if (model.modality == MTMD_MODALITY_VISION) {
        const int sz = hparams.warmup_image_size;
        img.set_size({sz, sz}, false, false);
        LOG_INF("%s: warmup with image size = %d x %d\n", __func__, sz, sz);
    } else {
        // GEMMA4UA uses n_mel_bins as a raw-waveform frame size (640), not a mel-bin count,
        // so the [1, 256] bound only applies to FFT-based models.
        const bool fft_based = model.proj_type != PROJECTOR_TYPE_GEMMA4UA;
        if (hparams.n_mel_bins <= 0 || (fft_based && hparams.n_mel_bins > 256)) {
            throw std::runtime_error(string_format("%s: invalid n_mel_bins (%d), must be in [1, 256]\n", __func__, hparams.n_mel_bins));
        }
        img.set_size({hparams.warmup_audio_size, hparams.n_mel_bins}, false, false);
        LOG_INF("%s: warmup with audio size = %d\n", __func__, hparams.warmup_audio_size);
    }
    batch.entries.push_back(img);
    return batch;
}

void mtmd_subcontext::warmup(const mtmd_image_f32_batch & batch) {
    support_info_graph info;

    if (flash_attn_type == MTMD_FLASH_ATTN_TYPE_AUTO) {
        // try to enable flash attention to see if it's supported
        flash_attn_type = MTMD_FLASH_ATTN_TYPE_ENABLED;
        info = reserve_compute_meta(batch);
        if (!info.fattn && info.fattn_op) {
            auto op = info.fattn_op;
            LOG_WRN("%s: *****************************************************************\n", __func__);
            LOG_WRN("%s: WARNING: flash attention not supported by %s, memory usage will increase\n", __func__, ggml_backend_name(backend));
            LOG_WRN("%s: op params: \n", __func__);
            static auto print_shape = [](const char * fn, const char * name, ggml_tensor * t) {
                LOG_WRN("%s:   %s: type = %s, ne = [%d %d %d %d], nb = [%d %d %d %d]\n", fn,
                        name, ggml_type_name(t->type),
                        t->ne[0], t->ne[1], t->ne[2], t->ne[3],
                        t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
            };
            print_shape(__func__, " dst", op);
            print_shape(__func__, "src0", op->src[0]);
            print_shape(__func__, "src1", op->src[1]);
            print_shape(__func__, "src2", op->src[2]);
            LOG_WRN("%s: please report this on github as an issue\n", __func__);
            LOG_WRN("%s: *****************************************************************\n", __func__);
            flash_attn_type = MTMD_FLASH_ATTN_TYPE_DISABLED;
            reserve_compute_meta(batch);
        }
    } else {
        info = reserve_compute_meta(batch);
        if (!info.fattn && flash_attn_type == MTMD_FLASH_ATTN_TYPE_ENABLED) {
            LOG_WRN("%s: flash attention is not supported by the current backend; falling back to CPU (performance will be degraded)\n", __func__);
        }
    }

    is_allocated = true; // mark buffers as allocated

    LOG_INF("%s: flash attention is %s\n", __func__,
        (flash_attn_type == MTMD_FLASH_ATTN_TYPE_ENABLED) ? "enabled" : "disabled");

    // print ops that are not supported by the GPU backend (if there is one)
    if (backend && backend != backend_cpu) {
        std::vector<support_info_op> unsupported_ops;
        for (const auto & op : info.ops) {
            if (!op.is_accel) {
                unsupported_ops.push_back(op);
            }
        }
        if (!unsupported_ops.empty()) {
            LOG_WRN("%s: *****************************************************************\n", __func__);
            LOG_WRN("%s: WARNING: the mmproj graph uses unsupported operators by the backend\n", __func__);
            LOG_WRN("%s:          the performance will be suboptimal                      \n", __func__);
            LOG_WRN("%s:          list of unsupported ops (backend=%s):\n", __func__, ggml_backend_name(backend));
            for (const auto & op : unsupported_ops) {
                LOG_WRN("%s: %16s: type = %s, ne = [%d %d %d %d]\n", __func__,
                        ggml_op_name(op.op->op),
                        ggml_type_name(op.op->type),
                        op.op->ne[0], op.op->ne[1], op.op->ne[2], op.op->ne[3]);
            }
            LOG_WRN("%s: flash attention is %s\n", __func__,
                (flash_attn_type == MTMD_FLASH_ATTN_TYPE_ENABLED) ? "enabled" : "disabled");
            LOG_WRN("%s: please report this on github as an issue\n", __func__);
            LOG_WRN("%s: ref: https://github.com/ggml-org/llama.cpp/pull/16837#issuecomment-3461676118\n", __func__);
            LOG_WRN("%s: *****************************************************************\n", __func__);
        }
    }
}

mtmd_subcontext::support_info_graph mtmd_subcontext::reserve_compute_meta(const mtmd_image_f32_batch & batch) {
    ggml_cgraph * gf = mtmd_get_graph_builder(this, batch)->build();
    ggml_backend_sched_reserve(sched.get(), gf);

    mem_compute.clear();
    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft = backend_buft[i];
        size_t size = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        if (size > 1) {
            LOG_INF("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    size / 1024.0 / 1024.0);
        }
        mem_compute[ggml_backend_get_device(backend)] += size;
    }

    const int n_splits = ggml_backend_sched_get_n_splits(sched.get());
    const int n_nodes  = ggml_graph_n_nodes(gf);

    LOG_INF("%s: graph splits = %d, nodes = %d\n", __func__,  n_splits, n_nodes);

    support_info_graph res {
        /*.fattn    = */ true,
        /*.fattn_op = */ nullptr,
        /*.ops      = */ {},
    };

    // check op support
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        res.ops.push_back({node, true});
        if (!ggml_backend_supports_op(backend, node)) {
            res.ops.back().is_accel = false;
            if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                res.fattn    = false;
                res.fattn_op = node;
            }
        }
    }

    return res;
}

// TODO: remove this and use ggml_mul_mat with GGML_PREC_F32 once all backends honor it for F32 x F32
// ggml_mul_mat for 2D F32 tensors, always in F32 on CPU: some GPU backends round the inputs to F16 or TF32, which loses the quiet bins of a spectrum
static void mtmd_mul_mat_prec_op(ggml_tensor * dst, int ith, int nth, void * /*userdata*/) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    const int64_t n_k = a->ne[0];

    for (int64_t n = ith; n < b->ne[1]; n += nth) {
        const float * b_row = (const float *) ((const char *) b->data + n * b->nb[1]);
        float * dst_row = (float *) ((char *) dst->data + n * dst->nb[1]);
        for (int64_t m = 0; m < a->ne[1]; m++) {
            const float * a_row = (const float *) ((const char *) a->data + m * a->nb[1]);
            // independent partial sums, so that the compiler can vectorize the loop
            float acc[16] = {};
            int64_t k = 0;
            for (; k + 16 <= n_k; k += 16) {
                for (int l = 0; l < 16; l++) {
                    acc[l] += a_row[k + l] * b_row[k + l];
                }
            }
            float sum = 0.0f;
            for (; k < n_k; k++) {
                sum += a_row[k] * b_row[k];
            }
            for (int l = 0; l < 16; l++) {
                sum += acc[l];
            }
            dst_row[m] = sum;
        }
    }
}

static ggml_tensor * mtmd_mul_mat_prec(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_n_dims(a) <= 2 && ggml_n_dims(b) <= 2 && a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->nb[0] == sizeof(float) && b->nb[0] == sizeof(float));
    ggml_tensor * args[] = { a, b };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, a->ne[1], b->ne[1], 1, 1, args, 2, mtmd_mul_mat_prec_op, GGML_N_TASKS_MAX, nullptr);
}

bool mtmd_audio_log_mel(const mtmd_audio_mel_config & config, const float * samples, const float * window, int64_t n_frames, float * out) {
    ggml_backend_sched_t sched = config.sched;
    ggml_backend_t backend_cpu = ggml_backend_sched_get_backend(sched, ggml_backend_sched_get_n_backends(sched) - 1);
    mtmd_backend_cpu_set_n_threads(backend_cpu, std::max(1, config.n_threads));

    const auto &  cache      = *config.cache;
    const auto &  filters    = cache.filters.data;
    const int64_t n_fft_bins = config.n_fft_bins;
    const int64_t frame_size = (n_fft_bins - 1) * 2;
    const int64_t n_mel      = config.n_mel;
    GGML_ASSERT(filters.size() == (size_t) (n_mel * n_fft_bins));
    GGML_ASSERT(cache.cos_vals.size() == (size_t) frame_size && cache.sin_vals.size() == (size_t) frame_size);

    // DFT matrix with the window folded in, the first n_fft_bins rows give the real part
    std::vector<float> dft((size_t) 2 * n_fft_bins * frame_size);
    for (int64_t k = 0; k < n_fft_bins; k++) {
        for (int64_t j = 0; j < frame_size; j++) {
            const int64_t q = (j * k) % frame_size;
            dft[k * frame_size + j]                = window[j] * cache.cos_vals[q];
            dft[(k + n_fft_bins) * frame_size + j] = window[j] * cache.sin_vals[q];
        }
    }

    // the spectrum is several times larger than the result, so long audio is processed in blocks
    const int64_t n_frames_max = 8192;

    for (int64_t off = 0; off < n_frames; off += n_frames_max) {
        const int64_t n_cur = std::min(n_frames_max, n_frames - off);

        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx_ptr(ggml_init(params));
        ggml_context * ctx0 = ctx_ptr.get();

        ggml_tensor * inp_samples = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, (n_cur - 1) * config.hop_length + frame_size);
        ggml_tensor * inp_dft     = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, frame_size, 2 * n_fft_bins);
        ggml_tensor * inp_filters = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_fft_bins, n_mel);
        ggml_set_input(inp_samples);
        ggml_set_input(inp_dft);
        ggml_set_input(inp_filters);

        ggml_tensor * cur = ggml_cont(ctx0, ggml_view_2d(ctx0, inp_samples, frame_size, n_cur, config.hop_length * ggml_element_size(inp_samples), 0));
        cur = mtmd_mul_mat_prec(ctx0, inp_dft, cur); // [2 * n_fft_bins, n_cur]

        cur = ggml_sqr(ctx0, cur);
        cur = ggml_add(ctx0,
                ggml_view_2d(ctx0, cur, n_fft_bins, n_cur, cur->nb[1], 0),
                ggml_view_2d(ctx0, cur, n_fft_bins, n_cur, cur->nb[1], n_fft_bins * ggml_element_size(cur)));
        if (config.use_magnitude) {
            cur = ggml_sqrt(ctx0, cur);
        }

        cur = mtmd_mul_mat_prec(ctx0, cur, inp_filters); // [n_cur, n_mel]
        cur = config.mel_floor_add
            ? ggml_scale_bias(ctx0, cur, 1.0f, config.mel_floor)
            : ggml_clamp(ctx0, cur, config.mel_floor, FLT_MAX);
        cur = ggml_log(ctx0, cur);
        if (!config.use_natural_log) {
            cur = ggml_scale(ctx0, cur, 1.0f / logf(10.0f));
        }
        ggml_set_output(cur);

        ggml_cgraph * gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, cur);

        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            LOG_ERR("%s: failed to allocate compute graph\n", __func__);
            return false;
        }

        ggml_backend_tensor_set(inp_samples, samples + off * config.hop_length, 0, ggml_nbytes(inp_samples));
        ggml_backend_tensor_set(inp_dft,     dft.data(),                        0, ggml_nbytes(inp_dft));
        ggml_backend_tensor_set(inp_filters, filters.data(),                    0, ggml_nbytes(inp_filters));

        auto status = ggml_backend_sched_graph_compute(sched, gf);
        if (status != GGML_STATUS_SUCCESS) {
            LOG_ERR("%s: ggml_backend_sched_graph_compute failed with error %d\n", __func__, status);
            return false;
        }

        for (int64_t j = 0; j < n_mel; j++) {
            ggml_backend_tensor_get(cur, out + j * n_frames + off, j * cur->nb[1], n_cur * ggml_element_size(cur));
        }
    }

    return true;
}

bool mtmd_subcontext::encode_image(int n_threads, const mtmd_image_f32 * img, std::vector<float> & out_vec) {
    mtmd_image_f32_batch imgs;
    mtmd_image_f32 img_copy = *img;
    imgs.entries.push_back(std::move(img_copy));

    return encode_batch(n_threads, &imgs, out_vec);
}

bool mtmd_subcontext::encode_batch(int n_threads, const mtmd_image_f32_batch * imgs_c_ptr, std::vector<float> & out_batch_embd) {
    mtmd_subctx_encode_params params;
    params.imgs = imgs_c_ptr;
    params.n_threads = n_threads;
    params.out_embd = &out_batch_embd;

    if (model.proj_type == PROJECTOR_TYPE_NEMOTRON_TRANSCRIBE) {
        return mtmd_graph_nemotron_transcribe::encode_windows(this, model, &params);
    }
    return encode(&params);
}

// persisted state slots of the gen-audio decoders and the streaming audio encoders
static std::vector<c2w_state_slot> list_gen_state_slots(const mtmd_hparams & hparams, const mtmd_model & model) {
    switch (model.proj_type) {
        case PROJECTOR_TYPE_QWEN3TTS_GEN:  return list_c2w_state_slots(hparams, model);
        case PROJECTOR_TYPE_DRAMA3_GEN:    return list_drama3_state_slots(hparams, model);
        case PROJECTOR_TYPE_POCKETTTS_GEN: return list_pockettts_state_slots(hparams, model);
        case PROJECTOR_TYPE_LFM3A_GEN:     return list_lfm3a_gen_state_slots(hparams, model);
        default:                           return {};
    }
}

// state_out can have another shape than its slot, only the byte size must match
static void copy_gen_state_out(ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_nbytes(src) == ggml_nbytes(dst));
    if (src->type == dst->type && ggml_are_same_shape(src, dst) && ggml_is_contiguous(src)) {
        ggml_backend_tensor_copy(src, dst);
        return;
    }
    std::vector<uint8_t> tmp(ggml_nbytes(src));
    ggml_backend_tensor_get(src, tmp.data(), 0, tmp.size());
    ggml_backend_tensor_set(dst, tmp.data(), 0, tmp.size());
}

mtmd_subctx_gen_state * mtmd_subcontext::gen_state_init(uint32_t seed) {
    auto st = std::make_unique<mtmd_subctx_gen_state>();
    st->rng.seed(seed == UINT32_MAX ? std::random_device{}() : seed);

    const auto slots = list_gen_state_slots(model.hparams, model);
    if (slots.empty()) {
        return st.release();
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * slots.size(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    st->ctx_data.reset(ggml_init(params));
    for (const auto & slot : slots) {
        ggml_tensor * t = ggml_new_tensor_2d(st->ctx_data.get(), GGML_TYPE_F32, slot.ne0, slot.ne1);
        ggml_format_name(t, "gen_state_%s", slot.name.c_str());
        st->slots.push_back(t);
    }
    st->buf.reset(ggml_backend_alloc_ctx_tensors(st->ctx_data.get(), backend));
    if (!st->buf) {
        LOG_ERR("%s: failed to allocate the gen state\n", __func__);
        return nullptr;
    }
    // a zero state is the cold start
    ggml_backend_buffer_clear(st->buf.get(), 0);

    return st.release();
}

mtmd_subctx_enc_state * mtmd_subcontext::enc_state_init(int32_t n_frames_max) {
    const auto & hparams = model.hparams;
    if (model.proj_type == PROJECTOR_TYPE_LFM3A) {
        // downsampler context, short-conv taps and attention window, see list_lfm3a_state_slots()
        const auto slots = list_lfm3a_state_slots(hparams, model);
        auto st = std::make_unique<mtmd_subctx_enc_state>();
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * slots.size(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        st->ctx_data.reset(ggml_init(params));
        for (const auto & slot : slots) {
            ggml_tensor * t = ggml_new_tensor_2d(st->ctx_data.get(), GGML_TYPE_F32, slot.ne0, slot.ne1);
            ggml_format_name(t, "enc_state_%s", slot.name.c_str());
            st->slots.push_back(t);
        }
        st->buf.reset(ggml_backend_alloc_ctx_tensors(st->ctx_data.get(), backend));
        if (!st->buf) {
            LOG_ERR("%s: failed to allocate the encoder state\n", __func__);
            return nullptr;
        }
        // a zero state is the cold start
        ggml_backend_buffer_clear(st->buf.get(), 0);
        return st.release();
    }
    if (model.proj_type != PROJECTOR_TYPE_VOXTRAL_RT || hparams.attn_window_size <= 0) {
        LOG_ERR("%s: the encoder does not support streaming input\n", __func__);
        return nullptr;
    }
    if (n_frames_max <= 0) {
        LOG_ERR("%s: invalid n_frames_max %d\n", __func__, n_frames_max);
        return nullptr;
    }

    const int n_layer   = hparams.n_layer;
    const int d_head    = hparams.n_embd_head > 0 ? hparams.n_embd_head : hparams.n_embd / hparams.n_head;
    const int n_embd_kv = hparams.n_head_kv * d_head;

    auto st = std::make_unique<mtmd_subctx_enc_state>();
    st->n_ctx = hparams.attn_window_size + n_frames_max;
    st->slot_pos.assign(st->n_ctx, -1);

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2 * n_layer,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    st->ctx_data.reset(ggml_init(params));
    for (int il = 0; il < n_layer; il++) {
        ggml_tensor * k = ggml_new_tensor_2d(st->ctx_data.get(), GGML_TYPE_F16, n_embd_kv, st->n_ctx);
        ggml_tensor * v = ggml_new_tensor_2d(st->ctx_data.get(), GGML_TYPE_F16, n_embd_kv, st->n_ctx);
        ggml_format_name(k, "enc_state_k_%d", il);
        ggml_format_name(v, "enc_state_v_%d", il);
        st->k.push_back(k);
        st->v.push_back(v);
    }
    st->buf.reset(ggml_backend_alloc_ctx_tensors(st->ctx_data.get(), backend));
    if (!st->buf) {
        LOG_ERR("%s: failed to allocate the encoder state\n", __func__);
        return nullptr;
    }
    // empty slots are masked out, but they must not hold NaN
    ggml_backend_buffer_clear(st->buf.get(), 0);

    LOG_INF("%s: encoder state: %d slots, %.2f MiB\n", __func__, st->n_ctx, ggml_backend_buffer_get_size(st->buf.get()) / 1024.0 / 1024.0);

    return st.release();
}

bool mtmd_subcontext::encode(mtmd_subctx_encode_params * params) {
    const mtmd_image_f32_batch & imgs = *params->imgs;
    int n_batch_cur = imgs.entries.size();

    // [QWEN_VIDEO] for video models, the batch dimension is used as temporal dimension for merged frames
    if (!support_batch && n_batch_cur > model.n_temporal_merge()) {
        LOG_ERR("%s: batch size %d exceeds maximum supported batch/temporal-merge size %d\n", __func__, n_batch_cur, model.n_temporal_merge());
        return false;
    }

    // if buffers are not allocated, we need to do a warmup run to allocate them
    if (!is_allocated) {
        warmup(*params->imgs);
    }

    const bool needs_gen_state = params->gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_CODE ||
                                 params->gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_WAV ||
                                 params->gen_process == MTMD_SUBCTX_GEN_PROCESS_TTS;
    if (needs_gen_state && params->gen_state == nullptr) {
        LOG_ERR("%s: audio gen requires a gen state\n", __func__);
        return false;
    }
    if (model.proj_type == PROJECTOR_TYPE_DRAMA3_GEN && params->gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_CODE) {
        float pos = 0.0f;
        ggml_backend_tensor_get(params->gen_state->slots[0], &pos, 0, sizeof(float));
        if ((int) pos >= mtmd_graph_drama3_gen::n_pos_max) {
            LOG_ERR("%s: the codebook tower holds at most %d rows\n", __func__, mtmd_graph_drama3_gen::n_pos_max);
            return false;
        }
    }

    // build the inference graph
    ggml_backend_sched_reset(sched.get());
    ggml_cgraph * gf = mtmd_get_graph_builder(this, imgs, params)->build();

    // the last node is the embedding tensor, code2wav has no out_embd
    // read it before scheduling: backends may reorder the nodes of graphs with several outputs (Metal)
    ggml_tensor * embeddings = params->out_embd ? ggml_graph_node(gf, -1) : nullptr;

    if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
        LOG_ERR("%s: failed to allocate compute graph\n", __func__);
        return false;
    }

    // set inputs
    const auto & hparams = model.hparams;

    const int image_size_width  = imgs.entries[0].nx();
    const int image_size_height = imgs.entries[0].ny();

    const int patch_size    = hparams.patch_size;
    const int num_patches   = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int n_pos = num_patches + (model.class_embedding ? 1 : 0);
    const int pos_w = image_size_width  / patch_size;
    const int pos_h = image_size_height / patch_size;


    auto get_inp_tensor = [&gf](const char * name) {
        ggml_tensor * inp = ggml_graph_get_tensor(gf, name);
        if (inp == nullptr) {
            GGML_ABORT("Failed to get tensor %s", name);
        }
        if (!(inp->flags & GGML_TENSOR_FLAG_INPUT)) {
            GGML_ABORT("Tensor %s is not an input tensor", name);
        }
        return inp;
    };

    auto set_input_f32 = [&get_inp_tensor](const char * name, const std::vector<float> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    // mask from build_inp_attn_mask(), f16 if flash attn is enabled
    auto set_input_attn_mask = [&get_inp_tensor](const std::vector<float> & values, int idx = 0) {
        ggml_tensor * cur = get_inp_tensor(get_attn_mask_name(idx).c_str());
        GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
        if (cur->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> values_f16(values.size());
            ggml_fp32_to_fp16_row(values.data(), values_f16.data(), values.size());
            ggml_backend_tensor_set(cur, values_f16.data(), 0, ggml_nbytes(cur));
        } else {
            GGML_ASSERT(cur->type == GGML_TYPE_F32);
            ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
        }
    };

    auto set_input_i32 = [&get_inp_tensor](const char * name, std::vector<int32_t> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    // state of the previous call, zero on a cold start
    auto set_gen_state_in = [&]() {
        const auto & slots = params->gen_state->slots;
        size_t i = 0;
        for (const auto & slot : list_gen_state_slots(hparams, model)) {
            GGML_ASSERT(i < slots.size());
            if (slot.in_place) {
                i++;
                continue;
            }
            ggml_backend_tensor_copy(slots[i++], get_inp_tensor(("state_in_" + slot.name).c_str()));
        }
    };

    // lfm3a encoder and codec: causal sliding window over [cached window | this call]
    auto set_lfm3a_window_inputs = [&](int base, int n_pos, int window) {
        std::vector<int32_t> positions(n_pos);
        for (int i = 0; i < n_pos; i++) {
            positions[i] = base + i;
        }
        set_input_i32("positions", positions);

        // key j holds position base - prefix + j
        const int prefix = window - 1;
        const int n_kv   = prefix + n_pos;
        std::vector<float> mask((size_t) n_kv * n_pos, -INFINITY);
        for (int q = 0; q < n_pos; q++) {
            for (int j = std::max(0, q + prefix - window + 1); j <= q + prefix; j++) {
                if (base - prefix + j >= 0) {
                    mask[(size_t) q * n_kv + j] = 0.0f;
                }
            }
        }
        set_input_attn_mask(mask);
    };

    // rope positions and attention mask of the mimi transformers (pocket-tts).
    // the mask is causal with a sliding window, see _build_attention_mask() in the reference
    auto set_pockettts_tfm_inputs = [&]() {
        const int64_t n_pos = ggml_nelements(get_inp_tensor("inp_pos"));
        GGML_ASSERT(n_pos > 0);
        std::vector<int32_t> positions((size_t) n_pos);
        for (int64_t i = 0; i < n_pos; i++) {
            positions[(size_t) i] = (int32_t) i;
        }
        set_input_i32("inp_pos", positions);

        // the preprocessor truncates the waveform to keep this mask bounded
        const int64_t max_pos = (int64_t) mtmd_hparams::pockettts_max_spk_seconds * hparams.audio_sample_rate / 120;
        GGML_ASSERT(n_pos <= max_pos && "pocket-tts speaker reference too long for a dense mask");

        const int64_t context = hparams.mimi_tfm_context;
        std::vector<float> mask((size_t) n_pos * n_pos, -INFINITY);
        for (int64_t q = 0; q < n_pos; q++) {
            for (int64_t k = 0; k < n_pos; k++) {
                const int64_t delta = q - k;
                if (delta >= 0 && delta < context) {
                    mask[(size_t) q * n_pos + k] = 0.0f;
                }
            }
        }
        set_input_attn_mask(mask);
    };

    // set input pixel values
    if (!imgs.is_audio) {
        size_t nelem = 0;
        for (const auto & img : imgs.entries) {
            nelem += img.nx() * img.ny() * 3;
        }
        std::vector<float> inp_raw(nelem);

        // layout of data (note: the channel dim is unrolled to better visualize the layout):
        //
        // ┌──W──┐
        // │     H │  channel = R
        // ├─────┤ │
        // │     H │  channel = G
        // ├─────┤ │
        // │     H │  channel = B
        // └─────┘ │
        //   ──────┘ x B

        // IMPORTANT: [QWEN_VIDEO] the batch dim is currently used for temporal dim in Qwen-VL models
        // All entries must have the same spatial size (enforced by can_batch_with() during merging)
        {
            const int nx = imgs.entries[0].nx();
            const int ny = imgs.entries[0].ny();
            const int n  = nx * ny;

            for (int b = 0; b < n_batch_cur; b++) {
                LOG_DBG("%s: copying image %d/%d to input buffer (nx=%d, ny=%d)\n", __func__, b+1, n_batch_cur, nx, ny);
                const auto & buf = imgs.entries[b].get_ro_buf();
                float * batch_entry = inp_raw.data() + b * (3*n);
                for (int y = 0; y < ny; y++) {
                    for (int x = 0; x < nx; x++) {
                        size_t base_src = 3*(y * nx + x);
                        size_t base_dst =    y * nx + x;
                        batch_entry[      base_dst] = buf[base_src    ];
                        batch_entry[1*n + base_dst] = buf[base_src + 1];
                        batch_entry[2*n + base_dst] = buf[base_src + 2];
                    }
                }
            }
        }
        set_input_f32("inp_raw", inp_raw);

    } else if (params->gen_process != MTMD_SUBCTX_GEN_PROCESS_GEN_WAV &&
               params->gen_process != MTMD_SUBCTX_GEN_PROCESS_TTS &&
               params->gen_process != MTMD_SUBCTX_GEN_PROCESS_TTS_VOCODE &&
               ggml_graph_get_tensor(gf, "inp_raw") != nullptr) {
        // audio input. GEN_WAV and the chatterbox TTS stages are not here: they take codes or feats, set in the switch below.
        // a GEN_CODE graph may have no hidden-state input (lfm3a_gen feedback embedding)
        GGML_ASSERT(imgs.entries.size() == 1);

        const auto & mel_inp = imgs.entries[0];
        const auto & buf = mel_inp.get_ro_buf();
        const int n_step = mel_inp.nx();
        const int n_mel  = mel_inp.ny();
        GGML_ASSERT((size_t)n_step * n_mel == buf.size());

        set_input_f32("inp_raw", buf);
    }

    // set input per projector
    switch (model.proj_type) {
        case PROJECTOR_TYPE_VOXTRAL_RT:
            {
                mtmd_subctx_enc_state * st = params->enc_state;
                if (!st) {
                    LOG_ERR("%s: this encoder needs a mtmd_subctx_enc_state\n", __func__);
                    return false;
                }
                const int n_window = hparams.attn_window_size;
                const int n_new    = (imgs.entries[0].nx() - 2) / 2;
                GGML_ASSERT(n_new <= st->n_ctx - n_window);

                std::vector<int32_t> pos(n_new);
                std::vector<int64_t> idx(n_new);
                std::vector<int32_t> slot_pos = st->slot_pos; // including the frames written by this call
                for (int i = 0; i < n_new; i++) {
                    pos[i] = st->n_past + i;
                    idx[i] = pos[i] % st->n_ctx;
                    slot_pos[idx[i]] = pos[i];
                }
                set_input_i32("inp_pos", pos);
                ggml_backend_tensor_set(get_inp_tensor("inp_kv_idx"), idx.data(), 0, idx.size() * sizeof(int64_t));

                // each frame sees itself and the n_window - 1 frames before it
                std::vector<float> mask((size_t) st->n_ctx * n_new, -INFINITY);
                for (int i = 0; i < n_new; i++) {
                    for (int s = 0; s < st->n_ctx; s++) {
                        const int32_t p = slot_pos[s];
                        if (p >= 0 && p <= pos[i] && p > pos[i] - n_window) {
                            mask[(size_t) i * st->n_ctx + s] = 0.0f;
                        }
                    }
                }
                set_input_attn_mask(mask);

                std::vector<float> tail = st->stem_tail;
                tail.resize(hparams.n_embd, 0.0f);
                set_input_f32("inp_stem_tail", tail);
            } break;
        case PROJECTOR_TYPE_MUSE_GLIMMER:
            {
                const int grid_w = pos_w;            // image_size_width  / patch_size
                const int grid_h = pos_h;            // image_size_height / patch_size
                const int n_tok  = grid_w * grid_h;
                const int pgrid  = (int) std::sqrt((double) model.position_embeddings->ne[1]); // 32
                const int f      = hparams.n_merge;  // downsample 2

                // pixel patchify runs inside the graph via build_inp() (ggml_conv_2d);
                // pos-emb bilinear interp via resize_position_embeddings().

                // --- sparse window grouping (pgrid x pgrid windows) ---
                const int win = pgrid;
                const int nwin_h = (grid_h + win - 1) / win;
                const int nwin_w = (grid_w + win - 1) / win;
                std::vector<int32_t> sp_perm; sp_perm.reserve(n_tok);
                std::vector<int>     sp_slens;
                for (int wy = 0; wy < nwin_h; wy++) {
                    for (int wx = 0; wx < nwin_w; wx++) {
                        int cnt = 0;
                        for (int hh = 0; hh < win; hh++) {
                            for (int ww = 0; ww < win; ww++) {
                                const int gy = wy * win + hh;
                                const int gx = wx * win + ww;
                                if (gy < grid_h && gx < grid_w) { sp_perm.push_back(gy * grid_w + gx); cnt++; }
                            }
                        }
                        if (cnt > 0) sp_slens.push_back(cnt);
                    }
                }
                std::vector<int32_t> rpos_w(n_tok), rpos_h(n_tok), inv_perm(n_tok);
                for (int i = 0; i < n_tok; i++) {
                    const int orig = sp_perm[i];
                    rpos_w[i] = (orig % grid_w) + 1; // 1-indexed
                    rpos_h[i] = (orig / grid_w) + 1;
                    inv_perm[orig] = i;
                }
                set_input_i32("muse_glimmer_sp_perm",  sp_perm);
                set_input_i32("muse_glimmer_inv_perm", inv_perm);
                set_input_i32("muse_glimmer_pos_w",    rpos_w);
                set_input_i32("muse_glimmer_pos_h",    rpos_h);

                // block-diagonal window mask (permuted order)
                std::vector<float> sp_mask((size_t) n_tok * n_tok, -INFINITY);
                {
                    int off = 0;
                    for (int s : sp_slens) {
                        for (int a = 0; a < s; a++)
                            for (int b = 0; b < s; b++)
                                sp_mask[(size_t) (off + a) * n_tok + (off + b)] = 0.0f;
                        off += s;
                    }
                }
                set_input_attn_mask(sp_mask);

                // pixel-shuffle gather (original order): f*f spatial neighbours grouped
                std::vector<int32_t> dsp; dsp.reserve(n_tok);
                for (int oy = 0; oy < grid_h / f; oy++)
                    for (int ox = 0; ox < grid_w / f; ox++)
                        for (int ry = 0; ry < f; ry++)
                            for (int rx = 0; rx < f; rx++)
                                dsp.push_back((oy * f + ry) * grid_w + (ox * f + rx));
                set_input_i32("muse_glimmer_ds_perm", dsp);
            } break;
        case PROJECTOR_TYPE_MINICPMV:
            {
                // inspired from siglip:
                //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit
                //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit/blob/d66538faeba44480d0bfaa42145eef26f9423199/modeling_siglip.py#L316
                std::vector<int32_t> positions(pos_h * pos_w);
                // note: sized by the actual patch counts; a tall/wide image produces more
                // than 1024 patches per side and a fixed [1024] array would be overrun
                std::vector<int> bucket_coords_h(pos_h);
                std::vector<int> bucket_coords_w(pos_w);
                for (int i = 0; i < pos_h; i++){
                    bucket_coords_h[i] = std::floor(70.0*i/pos_h);
                }
                for (int i = 0; i < pos_w; i++){
                    bucket_coords_w[i] = std::floor(70.0*i/pos_w);
                }
                for (int i = 0, id = 0; i < pos_h; i++){
                    for (int j = 0; j < pos_w; j++){
                        positions[id++] = bucket_coords_h[i]*70 + bucket_coords_w[j];
                    }
                }
                set_input_i32("positions", positions);

                // inputs for resampler projector
                // set the 2D positions (using float for sinusoidal embedding)
                int n_patches_per_col = image_size_width / patch_size;
                std::vector<float> pos_data(n_pos);
                // dimension H
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = static_cast<float>(i / n_patches_per_col);
                }
                set_input_f32("pos_h", pos_data);
                // dimension W
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = static_cast<float>(i % n_patches_per_col);
                }
                set_input_f32("pos_w", pos_data);
                // base frequency omega
                const float base_freq   = 10000.0f;
                const int   n_embd_proj = model.n_mmproj_embd();
                std::vector<float> omega(n_embd_proj / 4);
                for (int i = 0; i < n_embd_proj / 4; ++i) {
                    omega[i] = 1.0f / std::pow(base_freq, static_cast<float>(i) / (n_embd_proj / 4));
                }
                set_input_f32("omega", omega);
            } break;
        case PROJECTOR_TYPE_MINICPMV4_6:
        case PROJECTOR_TYPE_MINICPMV4_7:
        case PROJECTOR_TYPE_MINICPM_OCR_1_0:
            {
                const bool is_4x = hparams.n_merge == 2;

                // SigLIP position buckets (same as resampler path)
                std::vector<int32_t> positions(pos_h * pos_w);
                std::vector<int> bucket_coords_h(pos_h);
                std::vector<int> bucket_coords_w(pos_w);
                for (int i = 0; i < pos_h; i++){
                    bucket_coords_h[i] = std::floor(70.0*i/pos_h);
                }
                for (int i = 0; i < pos_w; i++){
                    bucket_coords_w[i] = std::floor(70.0*i/pos_w);
                }
                for (int i = 0, id = 0; i < pos_h; i++){
                    for (int j = 0; j < pos_w; j++){
                        positions[id++] = bucket_coords_h[i]*70 + bucket_coords_w[j];
                    }
                }
                set_input_i32("positions", positions);

                const int half_h = pos_h / 2;
                const int half_w = pos_w / 2;

                auto make_ds_idx = [](int off_r, int off_c, int ds_h, int ds_w, int stride_w) {
                    std::vector<int32_t> idx(ds_h * ds_w);
                    for (int i = 0; i < ds_h; i++) {
                        for (int j = 0; j < ds_w; j++) {
                            idx[i * ds_w + j] = (2*i + off_r) * stride_w + (2*j + off_c);
                        }
                    }
                    return idx;
                };

                if (!is_4x) {
                    // window reorder indices for 2x2 windows
                    std::vector<int32_t> window_idx(n_pos);
                    std::vector<int32_t> inv_window_idx(n_pos);
                    {
                        int k = 0;
                        for (int wi = 0; wi < half_h; wi++) {
                            for (int wj = 0; wj < half_w; wj++) {
                                window_idx[k++] = (2*wi    ) * pos_w + (2*wj    );
                                window_idx[k++] = (2*wi    ) * pos_w + (2*wj + 1);
                                window_idx[k++] = (2*wi + 1) * pos_w + (2*wj    );
                                window_idx[k++] = (2*wi + 1) * pos_w + (2*wj + 1);
                            }
                        }
                        for (int i = 0; i < n_pos; i++) {
                            inv_window_idx[window_idx[i]] = i;
                        }
                    }
                    set_input_i32("vit_merger_window_idx",     window_idx);
                    set_input_i32("vit_merger_inv_window_idx", inv_window_idx);

                    // block-diagonal attention mask: tokens in the same 4-token
                    // window attend to each other (mask = 0), all other positions
                    // are masked out (-inf). matches the window-major reorder above.
                    std::vector<float> window_mask_data(n_pos * n_pos, std::numeric_limits<float>::lowest());
                    for (int wi = 0; wi < n_pos / 4; wi++) {
                        for (int i = 0; i < 4; i++) {
                            for (int j = 0; j < 4; j++) {
                                window_mask_data[(wi*4 + i) * n_pos + (wi*4 + j)] = 0.0f;
                            }
                        }
                    }
                    set_input_attn_mask(window_mask_data);

                    // ViT merger 2x2 downsample indices
                    auto vit_merger_ds_0 = make_ds_idx(0, 0, half_h, half_w, pos_w);
                    auto vit_merger_ds_1 = make_ds_idx(0, 1, half_h, half_w, pos_w);
                    auto vit_merger_ds_2 = make_ds_idx(1, 0, half_h, half_w, pos_w);
                    auto vit_merger_ds_3 = make_ds_idx(1, 1, half_h, half_w, pos_w);
                    set_input_i32("vit_merger_ds_idx_0", vit_merger_ds_0);
                    set_input_i32("vit_merger_ds_idx_1", vit_merger_ds_1);
                    set_input_i32("vit_merger_ds_idx_2", vit_merger_ds_2);
                    set_input_i32("vit_merger_ds_idx_3", vit_merger_ds_3);
                }

                const int merger_h = is_4x ? pos_h : half_h;
                const int merger_w = is_4x ? pos_w : half_w;
                auto m_ds_0 = make_ds_idx(0, 0, merger_h / 2, merger_w / 2, merger_w);
                auto m_ds_1 = make_ds_idx(0, 1, merger_h / 2, merger_w / 2, merger_w);
                auto m_ds_2 = make_ds_idx(1, 0, merger_h / 2, merger_w / 2, merger_w);
                auto m_ds_3 = make_ds_idx(1, 1, merger_h / 2, merger_w / 2, merger_w);
                set_input_i32("merger_ds_idx_0", m_ds_0);
                set_input_i32("merger_ds_idx_1", m_ds_1);
                set_input_i32("merger_ds_idx_2", m_ds_2);
                set_input_i32("merger_ds_idx_3", m_ds_3);
            } break;
        case PROJECTOR_TYPE_LFM3VL:
            {
                // patches in raster-scan order (no interleave before encoder layers)
                const int pw = image_size_width  / patch_size;
                const int ph = image_size_height / patch_size;
                std::vector<int> positions(n_pos * 4);
                int ptr = 0;
                for (int y = 0; y < ph; y++) {
                    for (int x = 0; x < pw; x++) {
                        positions[                  ptr] = y;
                        positions[    num_patches + ptr] = x;
                        positions[2 * num_patches + ptr] = y;
                        positions[3 * num_patches + ptr] = x;
                        ptr++;
                    }
                }
                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_MAGEVL:
            {
                // t, h, w in 2x2 block order, t is always 0 for an image
                const int merge_ratio = hparams.n_merge;
                const int pw = image_size_width  / patch_size / merge_ratio;
                const int ph = image_size_height / patch_size / merge_ratio;
                std::vector<int> positions(n_pos * 4, 0);
                int ptr = 0;
                for (int ty = 0; ty < ph; ty++) {
                    for (int tx = 0; tx < pw; tx++) {
                        for (int dy = 0; dy < merge_ratio; dy++) {
                            for (int dx = 0; dx < merge_ratio; dx++) {
                                positions[    num_patches + ptr] = ty * merge_ratio + dy;
                                positions[2 * num_patches + ptr] = tx * merge_ratio + dx;
                                ptr++;
                            }
                        }
                    }
                }
                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_LING3VL:
        case PROJECTOR_TYPE_GLM4V:
        case PROJECTOR_TYPE_GLM5V:
            {
                const int merge_ratio = hparams.n_merge;
                const int pw = image_size_width  / patch_size / merge_ratio;
                const int ph = image_size_height / patch_size / merge_ratio;
                std::vector<int> positions(n_pos * 4);
                int ptr = 0;
                for (int ty = 0; ty < ph; ty++) {
                    for (int tx = 0; tx < pw; tx++) {
                        for (int dy = 0; dy < merge_ratio; dy++) {
                            for (int dx = 0; dx < merge_ratio; dx++) {
                                const int y = ty * merge_ratio + dy;
                                const int x = tx * merge_ratio + dx;
                                positions[                  ptr] = y;
                                positions[    num_patches + ptr] = x;
                                positions[2 * num_patches + ptr] = y;
                                positions[3 * num_patches + ptr] = x;
                                ptr++;
                            }
                        }
                    }
                }

                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_STEP3VL:
            {
                std::vector<int32_t> pos_data(n_pos);
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i / pos_w;
                }
                set_input_i32("pos_h", pos_data);
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i % pos_w;
                }
                set_input_i32("pos_w", pos_data);
            } break;
        case PROJECTOR_TYPE_PADDLEOCR:
            {
                const int merge_ratio = hparams.n_merge;
                const int pw = image_size_width  / patch_size / merge_ratio;
                const int ph = image_size_height / patch_size / merge_ratio;
                std::vector<int> positions(n_pos * 4);
                int ptr = 0;
                // NOTE: same as Qwen-VL, but x and y are swapped
                for (int ty = 0; ty < ph; ty++) {
                    for (int dy = 0; dy < merge_ratio; dy++) {
                        for (int tx = 0; tx < pw; tx++) {
                            for (int dx = 0; dx < merge_ratio; dx++) {
                                const int y = ty * merge_ratio + dy;
                                const int x = tx * merge_ratio + dx;
                                positions[                  ptr] = y;
                                positions[    num_patches + ptr] = x;
                                positions[2 * num_patches + ptr] = y;
                                positions[3 * num_patches + ptr] = x;
                                ptr++;
                            }
                        }
                    }
                }

                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_MINIMAX_M3:
            {
                const int n_merge = hparams.n_merge;
                const int gh = image_size_height / patch_size;
                const int gw = image_size_width  / patch_size;
                std::vector<int32_t> pos_h, pos_w;
                pos_h.reserve(gh * gw);
                pos_w.reserve(gh * gw);
                for (int bh = 0; bh < gh / n_merge; bh++)
                for (int bw = 0; bw < gw / n_merge; bw++)
                for (int mh = 0; mh < n_merge; mh++)
                for (int mw = 0; mw < n_merge; mw++) {
                    pos_h.push_back(bh * n_merge + mh);
                    pos_w.push_back(bw * n_merge + mw);
                }
                set_input_i32("minimax_pos_h", pos_h);
                set_input_i32("minimax_pos_w", pos_w);
            } break;
        case PROJECTOR_TYPE_DOTS_OCR:
        case PROJECTOR_TYPE_DOTS3NOTE_V:
            {
                const int pw = image_size_width / patch_size;
                const int ph = image_size_height / patch_size;
                const int n_pos = ph * pw;
                std::vector<int> positions(n_pos * 4);
                int ptr = 0;

                // flat layout: [h, w, h, w] for each patch
                // patches are in raster order (matching conv2d output)
                for (int y = 0; y < ph; y++) {
                    for (int x = 0; x < pw; x++) {
                        positions[          ptr] = y;
                        positions[  n_pos + ptr] = x;
                        positions[2*n_pos + ptr] = y;
                        positions[3*n_pos + ptr] = x;
                        ptr++;
                    }
                }

                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_EXAONE4_5:
        case PROJECTOR_TYPE_YOUTUVL:
            {
                // pw * ph = number of tokens output by ViT after apply patch merger
                // ipw * ipw = number of vision token been processed inside ViT
                const bool use_window_attn =
                    (model.proj_type == PROJECTOR_TYPE_QWEN25VL || model.proj_type == PROJECTOR_TYPE_EXAONE4_5)
                        ? hparams.n_wa_pattern > 0
                        : !hparams.wa_layer_indexes.empty();
                const int merge_ratio = 2;
                const int pw  = image_size_width  / patch_size / merge_ratio;
                const int ph  = image_size_height / patch_size / merge_ratio;
                const int ipw = image_size_width  / patch_size;
                const int iph = image_size_height / patch_size;

                std::vector<int> idx    (ph * pw);
                std::vector<int> inv_idx(ph * pw);

                if (use_window_attn) {
                    const int attn_window_size = hparams.attn_window_size > 0 ? hparams.attn_window_size : 112;
                    const int grid_window = attn_window_size / patch_size / merge_ratio;
                    int dst = 0;
                    // [num_vision_tokens, num_vision_tokens] attention mask tensor
                    std::vector<float> mask(pow(ipw * iph, 2), std::numeric_limits<float>::lowest());
                    int mask_row = 0;

                    for (int y = 0; y < ph; y += grid_window) {
                        for (int x = 0; x < pw; x += grid_window) {
                            const int win_h = std::min(grid_window, ph - y);
                            const int win_w = std::min(grid_window, pw - x);
                            const int dst_0 = dst;
                            // group all tokens belong to the same window togather (to a continue range)
                            for (int dy = 0; dy < win_h; dy++) {
                                for (int dx = 0; dx < win_w; dx++) {
                                    const int src = (y + dy) * pw + (x + dx);
                                    GGML_ASSERT(src < (int)idx.size());
                                    GGML_ASSERT(dst < (int)inv_idx.size());
                                    idx    [src] = dst;
                                    inv_idx[dst] = src;
                                    dst++;
                                }
                            }

                            for (int r=0; r < win_h * win_w * merge_ratio * merge_ratio; r++) {
                                int row_offset = mask_row * (ipw * iph);
                                std::fill(
                                    mask.begin() + row_offset + (dst_0 * merge_ratio * merge_ratio),
                                    mask.begin() + row_offset + (dst   * merge_ratio * merge_ratio),
                                    0.0);
                                mask_row++;
                            }
                        }
                    }

                    set_input_i32("window_idx",     idx);
                    set_input_i32("inv_window_idx", inv_idx);
                    set_input_attn_mask(mask);
                } else {
                    for (int i = 0; i < ph * pw; i++) {
                        idx[i] = i;
                    }
                }

                const int mpow = merge_ratio * merge_ratio;
                std::vector<int> positions(n_pos * 4);

                int ptr = 0;
                for (int ty = 0; ty < ph; ty++) {
                    for (int tx = 0; tx < pw; tx++) {
                        for (int dy = 0; dy < merge_ratio; dy++) {
                            for (int dx = 0; dx < merge_ratio; dx++) {
                                auto remap = idx[ptr / mpow];
                                remap = (remap * mpow) + (ptr % mpow);

                                const int y = ty * merge_ratio + dy;
                                const int x = tx * merge_ratio + dx;
                                positions[                  remap] = y;
                                positions[    num_patches + remap] = x;
                                positions[2 * num_patches + remap] = y;
                                positions[3 * num_patches + remap] = x;
                                ptr++;
                            }
                        }
                    }
                }

                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_MIMOVL:
            {
                const int merge      = hparams.n_merge;        // 2
                const int merge_unit = merge * merge;          // 4
                const int patch      = hparams.patch_size;     // 16
                const int H          = image_size_height / patch;
                const int W          = image_size_width  / patch;
                const int n_pos_full = H * W;
                const int llm_h      = H / merge;
                const int llm_w      = W / merge;
                const int n_units    = llm_h * llm_w;          // n_pos / merge_unit

                // Row-major merge-tile-ordered (h, w) positions
                std::vector<int32_t> pos_h_row(n_pos_full);
                std::vector<int32_t> pos_w_row(n_pos_full);
                {
                    int idx = 0;
                    for (int ty = 0; ty < llm_h; ty++) {
                        for (int tx = 0; tx < llm_w; tx++) {
                            for (int dy = 0; dy < merge; dy++) {
                                for (int dx = 0; dx < merge; dx++) {
                                    pos_h_row[idx] = ty * merge + dy;
                                    pos_w_row[idx] = tx * merge + dx;
                                    idx++;
                                }
                            }
                        }
                    }
                }

                // Col-major merge-unit permutation
                std::vector<float> idx_col(n_units);
                for (int r = 0; r < llm_h; r++) {
                    for (int c = 0; c < llm_w; c++) {
                        int u_row = r * llm_w + c;
                        int u_col = c * llm_h + r;
                        idx_col[u_col] = (float) u_row;
                    }
                }

                // Col-mode positions: permute pos_*_row by idx_col
                std::vector<int32_t> pos_h_col(n_pos_full);
                std::vector<int32_t> pos_w_col(n_pos_full);
                for (int u = 0; u < n_units; u++) {
                    int src = (int) idx_col[u];
                    for (int k = 0; k < merge_unit; k++) {
                        pos_h_col[u * merge_unit + k] = pos_h_row[src * merge_unit + k];
                        pos_w_col[u * merge_unit + k] = pos_w_row[src * merge_unit + k];
                    }
                }

                // Pack into ggml_rope_multi VISION-mode layout. The non-CPU kernels
                // only read slots 0 and 1, so pack h in slot 0, w in slot 1:
                //   positions[0..n_pos)         = h
                //   positions[n_pos..2*n_pos)   = w
                //   positions[2*n_pos..3*n_pos) = 0
                //   positions[3*n_pos..4*n_pos) = 0
                std::vector<int32_t> positions_row(static_cast<size_t>(n_pos_full) * 4, 0);
                std::vector<int32_t> positions_col(static_cast<size_t>(n_pos_full) * 4, 0);
                for (int i = 0; i < n_pos_full; i++) {
                    positions_row[0 * n_pos_full + i] = pos_h_row[i];
                    positions_row[1 * n_pos_full + i] = pos_w_row[i];
                    positions_col[0 * n_pos_full + i] = pos_h_col[i];
                    positions_col[1 * n_pos_full + i] = pos_w_col[i];
                }

                // Banded 1D sliding-window mask
                const int window = hparams.attn_window_size;
                GGML_ASSERT(window > 0);
                std::vector<float> mask(static_cast<size_t>(n_pos_full) * n_pos_full, std::numeric_limits<float>::lowest());
                for (int q = 0; q < n_pos_full; q++) {
                    int lo = std::max(0, q - window);
                    int hi = std::min(n_pos_full - 1, q + window);
                    for (int k = lo; k <= hi; k++) {
                        mask[static_cast<size_t>(q) * n_pos_full + k] = 0.0f;
                    }
                }

                set_input_i32("mimovl_positions_row", positions_row);
                set_input_i32("mimovl_positions_col", positions_col);
                set_input_f32("mimovl_idx_col",       idx_col);
                set_input_attn_mask(mask);
            } break;
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_KIMIVL:
        case PROJECTOR_TYPE_KIMIK25:
        case PROJECTOR_TYPE_LIGHTONOCR:
            {
                // set the 2D positions
                int n_patches_per_col = image_size_width / patch_size;
                std::vector<int> pos_data(n_pos);
                // dimension H
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i / n_patches_per_col;
                }
                set_input_i32("pos_h", pos_data);
                // dimension W
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i % n_patches_per_col;
                }
                set_input_i32("pos_w", pos_data);
            } break;
        case PROJECTOR_TYPE_SENSENOVA_U1:
            {
                const int n_patches_per_row = image_size_width / patch_size;
                std::vector<int> pos_data(n_pos);
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i / n_patches_per_row;
                }
                set_input_i32("pos_y", pos_data);
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i % n_patches_per_row;
                }
                set_input_i32("pos_x", pos_data);
            } break;
        case PROJECTOR_TYPE_DEEPSEEK4V:
            {
                // set the 2D positions (mrope layout, only the first 2 channels are used)
                int n_patches_per_row = image_size_width / patch_size;
                std::vector<int32_t> positions(n_pos * 4, 0);
                for (int i = 0; i < n_pos; i++) {
                    positions[i]         = i / n_patches_per_row; // row
                    positions[n_pos + i] = i % n_patches_per_row; // col
                }
                set_input_i32("positions", positions);

                // token block layout index (see mtmd_graph_deepseek4v::build)
                // rows [0, n_grid) are the aligner output, the sentinels follow
                const int n_merge = hparams.n_merge;
                const int n_llm_w = MTMD_ALIGN(pos_w, n_merge) / n_merge;
                const int n_llm_h = MTMD_ALIGN(pos_h, n_merge) / n_merge;
                const int n_grid  = n_llm_w * n_llm_h;
                const int idx_start   = n_grid;
                const int idx_end     = n_grid + 1;
                const int idx_newline = n_grid + 2;
                const int idx_pad     = n_grid + 3;

                const int lead_pad = imgs.entries[0].lead_pad;
                const auto bl = dsv4_get_block_layout(n_llm_w, n_llm_h, lead_pad);

                std::vector<int32_t> idx;
                idx.reserve(bl.n_out);
                for (int i = 0; i < lead_pad; i++) {
                    idx.push_back(idx_pad);
                }
                idx.push_back(idx_start);
                // pairs of adjacent rows are interleaved column-wise ("N-layout")
                // ref: build_image_block in inference/image_processor.py
                for (int t = 0; t < bl.rows * bl.row_len; t++) {
                    const int g   = t / (2 * bl.row_len);
                    const int rem = t % (2 * bl.row_len);
                    const int c   = rem / 2; // column
                    const int r   = 2 * g + rem % 2; // row
                    if (r >= n_llm_h) {
                        idx.push_back(idx_pad);
                    } else if (c == n_llm_w) {
                        idx.push_back(idx_newline);
                    } else {
                        idx.push_back(r * n_llm_w + c);
                    }
                }
                for (int i = 0; i < bl.pad_last; i++) {
                    idx.push_back(idx_pad);
                }
                idx.push_back(idx_end);
                set_input_i32("layout_idx", idx);
            } break;
        case PROJECTOR_TYPE_DEEPSEEK41V:
            {
                // mrope layout, only the first 2 channels are used
                const int n_patches_per_row = image_size_width / patch_size;
                std::vector<int32_t> positions(n_pos * 4, 0);
                for (int i = 0; i < n_pos; i++) {
                    positions[i]         = i / n_patches_per_row;
                    positions[n_pos + i] = i % n_patches_per_row;
                }
                set_input_i32("positions", positions);

                // reading-order block: rows [0, n_grid) are the aligner output, then start / end / newline
                const int n_merge = hparams.n_merge;
                const int n_llm_w = MTMD_ALIGN(pos_w, n_merge) / n_merge;
                const int n_llm_h = MTMD_ALIGN(pos_h, n_merge) / n_merge;
                const int n_grid  = n_llm_w * n_llm_h;
                std::vector<int32_t> idx;
                idx.reserve(dsv41_n_block_tokens(n_llm_w, n_llm_h));
                idx.push_back(n_grid);         // IMAGE_START
                for (int r = 0; r < n_llm_h; r++) {
                    for (int c = 0; c < n_llm_w; c++) {
                        idx.push_back(r * n_llm_w + c);
                    }
                    idx.push_back(n_grid + 2); // IMAGE_NEW_LINE
                }
                idx.push_back(n_grid + 1);     // IMAGE_END
                set_input_i32("layout_idx", idx);
            } break;
        case PROJECTOR_TYPE_GLM_EDGE:
        {
            // llava and other models
            std::vector<int32_t> positions(n_pos);
            for (int i = 0; i < n_pos; i++) {
                positions[i] = i;
            }
            set_input_i32("positions", positions);
        } break;
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
            {
                // llava and other models
                std::vector<int32_t> positions(n_pos);
                for (int i = 0; i < n_pos; i++) {
                    positions[i] = i;
                }
                set_input_i32("positions", positions);

                // The patches vector is used to get rows to index into the embeds with;
                // we should skip dim 0 only if we have CLS to avoid going out of bounds
                // when retrieving the rows.
                int patch_offset = model.class_embedding ? 1 : 0;
                std::vector<int32_t> patches(num_patches);
                for (int i = 0; i < num_patches; i++) {
                    patches[i] = i + patch_offset;
                }
                set_input_i32("patches", patches);
            } break;
        case PROJECTOR_TYPE_POCKETTTS_SPKENC:
            {
                set_pockettts_tfm_inputs();
            } break;
        case PROJECTOR_TYPE_POCKETTTS_GEN:
            {
                if (params->gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_WAV) {
                    GGML_ASSERT(params->feats != nullptr);
                    set_input_f32("inp_feats", *params->feats);
                    // positions and mask are derived in-graph from the persisted counter
                    set_gen_state_in();
                } else {
                    // flow matching starts from gaussian noise, std = sqrt(temp), no noise at temp 0
                    ggml_tensor * t = get_inp_tensor("inp_noise");
                    std::vector<float> noise(ggml_nelements(t), 0.0f);
                    if (params->temp > 0.0f) {
                        std::normal_distribution<float> dist(0.0f, std::sqrt(params->temp));
                        for (auto & v : noise) {
                            v = dist(params->gen_state->rng);
                        }
                    }
                    set_input_f32("inp_noise", noise);
                }
            } break;
        case PROJECTOR_TYPE_SOPRANO:
            {
                set_input_f32("inp_feats", *params->feats);
            } break;
        case PROJECTOR_TYPE_GEMMA4V:
        case PROJECTOR_TYPE_GEMMA4UV:
            {
                // set (col, row) patch positions for learned positional embedding
                const int n_cols = image_size_width  / patch_size;
                std::vector<int> pos_x(num_patches), pos_y(num_patches);
                for (int i = 0; i < num_patches; i++) {
                    pos_x[i] = i % n_cols;
                    pos_y[i] = i / n_cols;
                }
                set_input_i32("pos_x", pos_x);
                set_input_i32("pos_y", pos_y);
            } break;
        case PROJECTOR_TYPE_DEEPSEEKOCR:
        case PROJECTOR_TYPE_DEEPSEEKOCR2:
            {
                GGML_ASSERT(
                    (pos_w == pos_h) // overview image
                    || (pos_h >= pos_w && pos_h % pos_w == 0) // tile images
                );

                const int window = hparams.attn_window_size;
                const int pos = pos_w;
                std::vector<int32_t> rel_pos_indices_local(window * window);
                std::vector<int32_t> rel_pos_indices_global(pos * pos);

                for (int q = 0; q < window; q++) {
                    for (int k = 0; k < window; k++) {
                        rel_pos_indices_local[q * window + k] = q - k + window - 1;
                    }
                }

                for (int q = 0; q < pos; q++) {
                    for (int k = 0; k < pos; k++) {
                        rel_pos_indices_global[q * pos + k] = q - k + pos - 1;
                    }
                }

                set_input_i32("rel_pos_indices_local", rel_pos_indices_local);
                set_input_i32("rel_pos_indices_global", rel_pos_indices_global);

                if (model.proj_type == PROJECTOR_TYPE_DEEPSEEKOCR2) {

                    // qwen2 encoder attention mask

                    // num_image_tokens = num_patches / 16
                    //   256 for 1024 global view
                    //   144 for 768 tile views
                    const int   num_image_tokens = num_patches / 16;
                    const int   seq_len          = num_image_tokens * 2;
                    std::vector qwen2_mask(static_cast<size_t>(seq_len) * seq_len, 0.0f);

                    // attention mask layout
                    //  +--------------+---------------+
                    //  |    all 0     |   all -inf    |
                    //  +--------------+---------------+
                    //  |    all 0     |  lower tri 0  |
                    //  +--------------+---------------+
                    for (int i = 0; i < seq_len; i++) {
                        for (int j = 0; j < seq_len; j++) {
                            const bool zero = i < num_image_tokens ?
                                                     j < num_image_tokens :
                                                     j < num_image_tokens || j <= i;
                            qwen2_mask[static_cast<size_t>(i) * seq_len + j] = zero ? 0.0f : -1e9f;
                        }
                    }
                    set_input_attn_mask(qwen2_mask);
                }
            } break;
        case PROJECTOR_TYPE_CHATTERBOX:
            {
                if (params->gen_process == MTMD_SUBCTX_GEN_PROCESS_TTS_VOCODE) {
                    set_input_f32("inp_mel",   *params->mel_in);
                    set_input_f32("inp_sstft", *params->sstft_in);
                    break;
                }
                if (params->gen_process == MTMD_SUBCTX_GEN_PROCESS_TOKENIZE) {
                    const int T1 = (imgs.entries[0].nx() - 1) / 2 + 1;
                    const int T2 = (T1 - 1) / 2 + 1;
                    std::vector<int32_t> pos(T2);
                    for (int i = 0; i < T2; i++) {
                        pos[(size_t) i] = i;
                    }
                    set_input_i32("inp_pos", pos);
                    break;
                }
                if (params->gen_process != MTMD_SUBCTX_GEN_PROCESS_TTS) {
                    break;
                }
                const int n_gen = (int) params->codes->size();
                GGML_ASSERT(params->ref);
                const int n_prompt = (int) params->ref->tokens.size();
                std::vector<int32_t> tokens = params->ref->tokens;
                tokens.resize((size_t) n_prompt + n_gen);
                memcpy(tokens.data() + n_prompt, params->codes->data(), (size_t) n_gen * sizeof(int32_t));
                const int64_t n_codes = model.cbx.input_embedding_w->ne[1];
                for (int32_t t : tokens) {
                    if (t < 0 || t >= n_codes) {
                        LOG_ERR("%s: speech token out of range (%d, codebook size %lld)\n", __func__, t, (long long) n_codes);
                        return false;
                    }
                }
                const int T1 = n_prompt + n_gen;
                const int T2 = 2 * T1;
                set_input_i32("inp_tokens", tokens);

                // mel-rate reference conditioning of the voice
                set_input_f32("inp_prompt_feat", params->ref->feat);
                set_input_f32("inp_xvec", params->ref->spk);

                // espnet relative positional encoding, entry k holds the sinusoid of relative position (T-1) - k
                auto fill_pos = [&](const char * name, int T) {
                    const int d = 512;
                    std::vector<float> pos((size_t) (2 * T - 1) * d);
                    for (int k = 0; k < 2 * T - 1; k++) {
                        const double rel = (double) (T - 1 - k);
                        for (int i = 0; i < d / 2; i++) {
                            const double div = exp(-(double) (2 * i) * log(10000.0) / d);
                            pos[(size_t) k * d + 2 * i    ] = (float) sin(rel * div);
                            pos[(size_t) k * d + 2 * i + 1] = (float) cos(rel * div);
                        }
                    }
                    set_input_f32(name, pos);
                };
                fill_pos("inp_pos1", T1);
                fill_pos("inp_pos2", T2);

                // noise and the time embeddings of the solver span points, matcha layout (sines then cosines, scale 1000)
                std::vector<float> noise((size_t) 80 * T2);
                std::normal_distribution<float> nd(0.0f, 1.0f);
                for (auto & f : noise) f = nd(params->gen_state->rng);
                set_input_f32("inp_noise", noise);

                const bool meanflow = model.cbx.time_embed_mixer_w != nullptr;
                const int n_steps = meanflow ? 2 : 10;
                std::vector<float> temb((size_t) 320 * (n_steps + 1));
                for (int s = 0; s <= n_steps; s++) {
                    const double u = (double) s / n_steps;
                    const double t = meanflow ? u : 1.0 - cos(u * M_PI / 2.0);
                    for (int i = 0; i < 160; i++) {
                        const double div = exp(-(double) i * log(10000.0) / 159.0);
                        temb[(size_t) s * 320 + i      ] = (float) sin(1000.0 * t * div);
                        temb[(size_t) s * 320 + i + 160] = (float) cos(1000.0 * t * div);
                    }
                }
                set_input_f32("inp_temb", temb);
            } break;
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_GEMMA3NV:
        case PROJECTOR_TYPE_IDEFICS3:
        case PROJECTOR_TYPE_COHERE2V:
        case PROJECTOR_TYPE_INTERNVL:
        case PROJECTOR_TYPE_NEMOTRON_V2_VL:
        case PROJECTOR_TYPE_NEMOTRON_OMNI:
        case PROJECTOR_TYPE_QWEN2A:
        case PROJECTOR_TYPE_QWEN3A:
        case PROJECTOR_TYPE_GLMA:
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_D1OMNI_V:
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_MERALION:
        case PROJECTOR_TYPE_MUSIC_FLAMINGO:
        case PROJECTOR_TYPE_JANUS_PRO:
        case PROJECTOR_TYPE_PHI4:
        case PROJECTOR_TYPE_COGVLM:
        case PROJECTOR_TYPE_YASA2:
        case PROJECTOR_TYPE_GEMMA4UA:
        case PROJECTOR_TYPE_MUSE_SPARK_AUDIO:
        case PROJECTOR_TYPE_QWEN3TTS_SPKENC:
            {
                // do nothing
            } break;
        case PROJECTOR_TYPE_CHATTERBOX_SPKENC:
            {
                // the ceil-mode seg pooling divides by the full segment length, rescale the last partial segment
                const int T1 = (imgs.entries[0].nx() - 1) / 2 + 1;
                const int S  = (T1 + 99) / 100;
                std::vector<float> segfix((size_t) S, 1.0f);
                const int last = T1 - (S - 1) * 100;
                segfix[(size_t) S - 1] = 100.0f / (float) last;
                set_input_f32("inp_segfix", segfix);
            } break;
        case PROJECTOR_TYPE_LFM3A_GEN:
            {
                const int64_t n_cb    = model.gen_code_embd_w->ne[2];
                const int64_t n_vocab = model.gen_code_embd_w->ne[1];
                if (params->gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_WAV) {
                    GGML_ASSERT(params->codes != nullptr);
                    // codes index the FSQ table via ggml_get_rows, the end-of-audio code has no waveform
                    const int64_t n_codes = model.c2w.quant_first_cb_w->ne[1];
                    for (int32_t c : *params->codes) {
                        if (c < 0 || c >= n_codes) {
                            LOG_ERR("%s: code out of range (%d, codebook size %lld)\n", __func__, c, (long long) n_codes);
                            return false;
                        }
                    }
                    std::vector<int32_t> codes = *params->codes;
                    set_input_i32("inp_codes", codes);
                    set_gen_state_in();
                    // the "pos" slot leads the state
                    float base = 0.0f;
                    ggml_backend_tensor_get(params->gen_state->slots[0], &base, 0, sizeof(float));
                    set_lfm3a_window_inputs((int) base, (int) ggml_nelements(get_inp_tensor("positions")), hparams.wav_tfm_swa);
                } else {
                    // rows of the merged per-codebook tables; a masked codebook (-1) takes row 0, the mask drops it
                    std::vector<int32_t> codes(n_cb, -1);
                    if (params->codes) {
                        GGML_ASSERT(params->codes->size() == (size_t) n_cb);
                        codes = *params->codes;
                    }
                    std::vector<float> mask(n_cb, 0.0f);
                    for (int64_t i = 0; i < n_cb; i++) {
                        if (codes[i] >= n_vocab) {
                            LOG_ERR("%s: code out of range (%d, vocab %lld)\n", __func__, codes[i], (long long) n_vocab);
                            return false;
                        }
                        if (codes[i] < 0) {
                            codes[i] = 0;
                        } else {
                            mask[i] = 1.0f;
                        }
                        codes[i] += (int32_t) (i * n_vocab);
                    }
                    set_input_i32("inp_codes", codes);
                    if (ggml_graph_get_tensor(gf, "inp_mask") != nullptr) {
                        set_input_f32("inp_mask", mask);
                    }
                }
            } break;
        case PROJECTOR_TYPE_DRAMA3_GEN:
            {
                const int64_t n_cb      = model.gen_code_embd_w->ne[2];
                const int64_t n_vocab   = model.gen_code_embd_w->ne[1];
                if (params->gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_WAV) {
                    // frame-major to group-major, clamp: codec residual codebooks are smaller than the tower vocab
                    GGML_ASSERT(params->codes != nullptr && params->codes->size() % n_cb == 0);
                    const int64_t n_frames = (int64_t) params->codes->size() / n_cb;
                    const int64_t n_first  = model.c2w.quant_first_cb_w->ne[1];
                    const int64_t n_rest   = model.c2w.quant_rest_cb_w->ne[1];
                    std::vector<int32_t> codes(params->codes->size());
                    for (int64_t f = 0; f < n_frames; f++) {
                        for (int64_t g = 0; g < n_cb; g++) {
                            const int64_t n = g == 0 ? n_first : n_rest;
                            codes[g * n_frames + f] = std::clamp<int32_t>((*params->codes)[f * n_cb + g], 0, (int32_t) n - 1);
                        }
                    }
                    set_input_i32("inp_codes", codes);
                    set_gen_state_in();
                } else {
                    // code0 < 0: next row is text (prompt), the tower only fills its KV
                    // codes: input frame codes for a speech row, empty for a text row
                    const bool next_speech = params->code0 >= 0;
                    const bool cur_speech  = params->codes && (int64_t) params->codes->size() == n_cb;
                    if (params->code0 >= n_vocab) {
                        LOG_ERR("%s: code0 out of range (%d, vocab %lld)\n", __func__, params->code0, (long long) n_vocab);
                        return false;
                    }
                    std::vector<int32_t> code0 = { std::max(params->code0, 0) };
                    set_input_i32("inp_code0", code0);

                    std::vector<int32_t> extra(n_cb - 2);
                    for (int64_t g = 2; g < n_cb; g++) {
                        const int32_t c = cur_speech ? std::clamp<int32_t>((*params->codes)[g], 0, (int32_t) n_vocab - 1) : 0;
                        extra[g - 2] = c + (int32_t) (g * n_vocab);
                    }
                    set_input_i32("inp_extra", extra);
                    set_input_f32("inp_scale", { next_speech ? 1.0f : 0.0f, cur_speech ? 1.0f : 0.0f });

                    float pos = 0.0f;
                    ggml_backend_tensor_get(params->gen_state->slots[0], &pos, 0, sizeof(float));
                    std::vector<int32_t> inp_pos = { (int32_t) pos };
                    set_input_i32("inp_pos", inp_pos);

                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                    std::vector<float> r(model.gen_code_head_w->ne[2]);
                    for (auto & v : r) {
                        v = dist(params->gen_state->rng);
                    }
                    set_input_f32("inp_rand", r);
                }
            } break;
        case PROJECTOR_TYPE_QWEN3TTS_GEN:
            {
                if (params->gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_WAV) {
                    GGML_ASSERT(params->codes != nullptr);

                    // frame-major input to group-major, rear-padded with code 0 up to one window
                    const int64_t n_codes  = model.gen_code_head_w->ne[2] + 1;
                    const int64_t n_frames_w = hparams.wav_tfm_swa;
                    const int64_t n_frames   = (int64_t) params->codes->size() / n_codes;
                    GGML_ASSERT(n_frames > 0 && n_frames <= n_frames_w);

                    // codes are used as ggml_get_rows indices, so check them against the codebook vocab
                    const int64_t vocab_first = model.c2w.quant_first_cb_w->ne[1];
                    const int64_t vocab_rest  = model.c2w.quant_rest_cb_w->ne[1];
                    for (int64_t f = 0; f < n_frames; f++) {
                        for (int64_t g = 0; g < n_codes; g++) {
                            const int32_t c = (*params->codes)[f * n_codes + g];
                            const int64_t vocab = (g == 0) ? vocab_first : vocab_rest;
                            if (c < 0 || (int64_t) c >= vocab) {
                                LOG_ERR("%s: code out of range (frame %lld, group %lld, code %d, vocab %lld)\n",
                                        __func__, (long long) f, (long long) g, c, (long long) vocab);
                                return false;
                            }
                        }
                    }

                    std::vector<int32_t> codes(n_frames_w * n_codes, 0);
                    for (int64_t f = 0; f < n_frames; f++) {
                        for (int64_t g = 0; g < n_codes; g++) {
                            codes[g * n_frames_w + f] = (*params->codes)[f * n_codes + g];
                        }
                    }
                    set_input_i32("inp_codes", codes);
                    set_gen_state_in();
                } else {
                    // code0 indexes gen_code_out_embd_w via ggml_get_rows; bound it
                    const int64_t vocab0 = model.gen_code_out_embd_w->ne[1];
                    if (params->code0 < 0 || (int64_t) params->code0 >= vocab0) {
                        LOG_ERR("%s: code0 out of range (%d, vocab %lld)\n", __func__, params->code0, (long long) vocab0);
                        return false;
                    }
                    std::vector<int32_t> code0 = { params->code0 };
                    set_input_i32("inp_code0", code0);

                    // one uniform(0,1) draw per codebook, used by do_sampling()
                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                    const int64_t n_acoustic = model.gen_code_head_w->ne[2];
                    for (int64_t g = 0; g < n_acoustic; g++) {
                        std::vector<float> r = { dist(params->gen_state->rng) };
                        set_input_f32(("inp_rand_" + std::to_string(g)).c_str(), r);
                    }
                }
            } break;
        case PROJECTOR_TYPE_HUNYUANVL:
            {
                // Compute the HunyuanVL 2D position embedding on CPU (with the
                // custom sf=(target+0.1)/n_grid bilinear sampling that the
                // reference implementation uses) and upload it to the graph
                // input declared in mtmd_graph_hunyuanvl::build().
                GGML_ASSERT(model.position_embeddings != nullptr);
                ggml_tensor * src_t   = model.position_embeddings;
                const int64_t n_embd  = src_t->ne[0];
                const int64_t n_pos   = src_t->ne[1];            // = n_grid * n_grid
                const int     n_grid  = (int)std::lround(std::sqrt((double)n_pos));
                GGML_ASSERT((int64_t)n_grid * n_grid == n_pos);
                const int     out_w   = pos_w;                    // pw
                const int     out_h   = pos_h;                    // ph

                // Pull weight to host.
                std::vector<float> src(n_embd * n_pos);
                ggml_backend_tensor_get(src_t, src.data(), 0, ggml_nbytes(src_t));

                // Output layout matches ggml_new_tensor_2d(F32, n_embd, out_h*out_w):
                //   ne[0] = n_embd (fastest), ne[1] = out_h*out_w
                //   dst[(y*out_w + x) * n_embd + c]
                std::vector<float> dst((size_t)n_embd * out_h * out_w);

                const float sx = (float)(out_w + 0.1f) / (float)n_grid;
                const float sy = (float)(out_h + 0.1f) / (float)n_grid;

                for (int y = 0; y < out_h; ++y) {
                    // Match ggml_compute_forward_upscale_f32 pixel-center
                    // convention (align_corners=False): src_y = (y+0.5)/sy - 0.5.
                    const float fy = ((float)y + 0.5f) / sy - 0.5f;
                    int y0 = (int)std::floor(fy);
                    int y1 = y0 + 1;
                    y0 = std::clamp(y0, 0, n_grid - 1);
                    y1 = std::clamp(y1, 0, n_grid - 1);
                    float wy1 = std::clamp(fy - (float)y0, 0.0f, 1.0f);
                    const float wy0 = 1.0f - wy1;
                    for (int x = 0; x < out_w; ++x) {
                        const float fx = ((float)x + 0.5f) / sx - 0.5f;
                        int x0 = (int)std::floor(fx);
                        int x1 = x0 + 1;
                        x0 = std::clamp(x0, 0, n_grid - 1);
                        x1 = std::clamp(x1, 0, n_grid - 1);
                        float wx1 = std::clamp(fx - (float)x0, 0.0f, 1.0f);
                        const float wx0 = 1.0f - wx1;

                        const float w00 = wy0 * wx0;
                        const float w01 = wy0 * wx1;
                        const float w10 = wy1 * wx0;
                        const float w11 = wy1 * wx1;

                        const float * s00 = &src[((size_t)y0 * n_grid + x0) * n_embd];
                        const float * s01 = &src[((size_t)y0 * n_grid + x1) * n_embd];
                        const float * s10 = &src[((size_t)y1 * n_grid + x0) * n_embd];
                        const float * s11 = &src[((size_t)y1 * n_grid + x1) * n_embd];
                        float * d         = &dst[((size_t)y * out_w + x) * n_embd];
                        for (int c = 0; c < n_embd; ++c) {
                            d[c] = w00 * s00[c] + w01 * s01[c] + w10 * s10[c] + w11 * s11[c];
                        }
                    }
                }

                set_input_f32("hunyuanvl_pos_embd", dst);
            } break;
        case PROJECTOR_TYPE_LLAMA4:
            {
                // set the 2D positions
                int n_patches_per_col = image_size_width / patch_size;
                std::vector<int> pos_data(num_patches + 1, 0); // +1 for the [CLS] token
                // last pos is always kept 0, it's for CLS
                // dimension H
                for (int i = 0; i < num_patches; i++) {
                    pos_data[i] = (i / n_patches_per_col) + 1;
                }
                set_input_i32("pos_h", pos_data);
                // dimension W
                for (int i = 0; i < num_patches; i++) {
                    pos_data[i] = (i % n_patches_per_col) + 1;
                }
                set_input_i32("pos_w", pos_data);
            } break;
        case PROJECTOR_TYPE_DOTS3NOTE_A:
            {
                GGML_ASSERT(imgs.entries.size() == 1);
                const int n_pos = (imgs.entries.front().nx() + 7) / 8; // 3x stride-2 conv2d
                std::vector<int32_t> positions(n_pos);
                for (int i = 0; i < n_pos; i++) {
                    positions[i] = i;
                }
                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_GEMMA4A:
            {
                GGML_ASSERT(imgs.entries.size() == 1);
                const auto & img0 = imgs.entries.front();
                // Compute n_pos matching SSCP output: two stride-2 convs
                int n_pos = img0.nx();
                for (int i = 0; i < 2; i++) { n_pos = (n_pos - 1) / 2 + 1; }

                // Chunked local attention: blocked causal mask and RPE
                const int chunk_size   = 12;
                const int max_past     = 12;
                const int context_size = chunk_size + max_past;
                const int num_blocks   = (n_pos + chunk_size - 1) / chunk_size;

                // Blocked causal attention mask: [context_size, chunk_size, num_blocks]
                {
                    std::vector<float> mask(context_size * chunk_size * num_blocks, -1e9f);
                    for (int b = 0; b < num_blocks; b++) {
                        for (int q = 0; q < chunk_size; q++) {
                            int gq = b * chunk_size + q;
                            for (int k = 0; k < context_size; k++) {
                                int gk = b * chunk_size - max_past + k;
                                if (gq < n_pos && gk >= 0 && gk < n_pos && gk <= gq && (gq - gk) < max_past) {
                                    mask[k + q * context_size + b * context_size * chunk_size] = 0.0f;
                                }
                            }
                        }
                    }
                    set_input_f32("kq_mask", mask);
                }

                // Sinusoidal RPE: 13 positions [12, 11, ..., 0]
                {
                    const int n_embd = model.hparams.n_embd;
                    const int num_timescales = n_embd / 2;
                    const float log_timescale_increment = logf(10000.0f) / std::max(num_timescales - 1, 1);
                    const int rpe_len = max_past + 1;
                    std::vector<float> pos_emb(n_embd * rpe_len, 0.0f);
                    for (int p = 0; p < rpe_len; p++) {
                        float position = (float)(max_past - p);
                        for (int i = 0; i < num_timescales; i++) {
                            float inv_ts = expf(-(float)i * log_timescale_increment);
                            float scaled = position * inv_ts;
                            pos_emb[p * n_embd + i]                 = sinf(scaled);
                            pos_emb[p * n_embd + i + num_timescales] = cosf(scaled);
                        }
                    }
                    set_input_f32("pos_emb", pos_emb);
                }
            } break;
        case PROJECTOR_TYPE_MIMO_AUDIO:
            {
                GGML_ASSERT(imgs.entries.size() == 1);
                const int n_frames = imgs.entries.front().nx();
                const int n_pos    = (n_frames - 1) / 2 + 1; // matches conv1(s=1)+conv2(s=2) output length

                std::vector<int32_t> positions(n_pos);
                for (int i = 0; i < n_pos; i++) {
                    positions[i] = i;
                }
                set_input_i32("mimo_audio_positions", positions);

                const int window = hparams.attn_window_size;
                GGML_ASSERT(window > 0);

                const float neg_inf = std::numeric_limits<float>::lowest();
                std::vector<float> full_mask((size_t) n_pos * n_pos);
                std::vector<float> window_mask((size_t) n_pos * n_pos);
                for (int q = 0; q < n_pos; q++) {
                    for (int k = 0; k < n_pos; k++) {
                        const bool causal_ok = k <= q;
                        full_mask[(size_t) q * n_pos + k]   = causal_ok ? 0.0f : neg_inf;
                        window_mask[(size_t) q * n_pos + k] = (causal_ok && (q - k) <= window) ? 0.0f : neg_inf;
                    }
                }
                set_input_attn_mask(full_mask, 0);
                set_input_attn_mask(window_mask, 1);

                // input_local_transformer: block-diagonal mask + in-group positions
                {
                    const int n_pos_ds   = (n_pos - 2) / 2 + 1; // matches downsample conv (k=2,s=2,p=0)
                    const int group_size = hparams.audio_local_group_size;
                    GGML_ASSERT(group_size > 0);
                    const int n_groups = (n_pos_ds + group_size - 1) / group_size;
                    const int n_padded = n_groups * group_size;

                    std::vector<int32_t> local_positions(n_padded);
                    for (int i = 0; i < n_padded; i++) {
                        local_positions[i] = i % group_size;
                    }
                    set_input_i32("mimo_audio_local_positions", local_positions);

                    std::vector<float> local_mask((size_t) n_padded * n_padded);
                    for (int q = 0; q < n_padded; q++) {
                        for (int k = 0; k < n_padded; k++) {
                            const bool same_group = (q / group_size) == (k / group_size);
                            local_mask[(size_t) q * n_padded + k] = same_group ? 0.0f : neg_inf;
                        }
                    }
                    set_input_attn_mask(local_mask, 2);
                }
            } break;
        case PROJECTOR_TYPE_LFM2A:
        case PROJECTOR_TYPE_D1OMNI_A:
            {
                GGML_ASSERT(imgs.entries.size() == 1);
                const auto n_frames = model.n_output_tokens(&imgs.entries.front());

                auto d_model = 512;
                auto seq_len = n_frames * 2 - 1;
                std::vector<float> pos_emb(d_model*seq_len);
                std::vector<double> inv_freq(d_model / 2);
                for (size_t i = 0; i < inv_freq.size(); ++i) {
                    inv_freq[i] = std::exp(-(std::log(10000.0) / (float)d_model) * (2.0f * (float)(i)));
                }
                for (int64_t pos = 0; pos < seq_len; ++pos) {
                    for (size_t i = 0; i < inv_freq.size(); ++i) {
                        const float ang = (n_frames - pos - 1) * inv_freq[i];
                        pos_emb[pos*d_model + 2*i + 0] = sinf(ang);  // even
                        pos_emb[pos*d_model + 2*i + 1] = cosf(ang);  // odd
                    }
                }
                set_input_f32("pos_emb", pos_emb);
            } break;
        case PROJECTOR_TYPE_LFM3A:
            {
                GGML_ASSERT(imgs.entries.size() == 1);
                const int n_pos = model.n_output_tokens(&imgs.entries.front());

                // the "pos" slot leads the state; without a stream the state is zero, the clip is encoded from its start
                int base = 0;
                if (params->enc_state != nullptr) {
                    const auto & slots = params->enc_state->slots;
                    float base_f = 0.0f;
                    ggml_backend_tensor_get(slots[0], &base_f, 0, sizeof(float));
                    base = (int) base_f;
                    size_t i = 0;
                    for (const auto & slot : list_lfm3a_state_slots(hparams, model)) {
                        ggml_backend_tensor_copy(slots[i++], get_inp_tensor(("state_in_" + slot.name).c_str()));
                    }
                } else {
                    for (const auto & slot : list_lfm3a_state_slots(hparams, model)) {
                        ggml_tensor * t = get_inp_tensor(("state_in_" + slot.name).c_str());
                        ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
                    }
                }

                set_lfm3a_window_inputs(base, n_pos, hparams.attn_window_size);
            } break;
        case PROJECTOR_TYPE_NEMOTRON_TRANSCRIBE:
            {
                GGML_ASSERT(imgs.entries.size() == 1);
                std::vector<int32_t> positions(ggml_nelements(get_inp_tensor("positions")));
                for (size_t i = 0; i < positions.size(); i++) {
                    positions[i] = (int32_t) i;
                }
                set_input_i32("positions", positions);
                GGML_ASSERT(params->enc_state != nullptr);
                set_input_f32("inp_norm", params->enc_state->mel_norm);
                if (ggml_graph_get_tensor(gf, "spk_prev") != nullptr) {
                    set_input_f32("spk_prev", params->enc_state->spk_prev);
                }
            } break;
        case PROJECTOR_TYPE_PARAKEET:
            {
                GGML_ASSERT(imgs.entries.size() == 1);
                struct ggml_tensor * attn_mask = ggml_graph_get_tensor(gf, "attn_mask");
                const int n_q = attn_mask->ne[1];
                const int n_k = attn_mask->ne[0];
                const int n_frames = imgs.entries.front().nx();
                const int n_tokens_real = (n_frames + hparams.subsampling_factor-1) / hparams.subsampling_factor;
                const float mask_value  = -1e30f;

                std::vector<float> mask_data(n_q * n_k);
                if (n_k == n_q) {
                    // full attention: mask keys that are padding
                    for (int q = 0; q < n_q; ++q) {
                        for (int k = 0; k < n_k; ++k) {
                            mask_data[q * n_k + k] = (k >= n_tokens_real) ? mask_value : 0.0f;
                        }
                    }
                } else {
                    // local attention: mask keys outside the valid window
                    const int att_left = n_k / 2;
                    for (int q = 0; q < n_q; ++q) {
                        for (int k = 0; k < n_k; ++k) {
                            const int key = q - att_left + k;
                            mask_data[q * n_k + k] = (key >= 0 && key < n_tokens_real) ? 0.0f : mask_value;
                        }
                    }
                }
                set_input_f32(attn_mask->name, mask_data);

                // local attention skew mask: zeroes out the probs that were
                // computed for keys outside the valid sliding window.
                if (struct ggml_tensor * local_mask = ggml_graph_get_tensor(gf, "local_mask")) {
                    const int lm_k = local_mask->ne[0];
                    const int lm_q = local_mask->ne[1];
                    const int window_size = lm_k - lm_q + 1;
                    std::vector<float> lm_data(lm_q * lm_k);
                    for (int q = 0; q < lm_q; ++q) {
                        for (int k = 0; k < lm_k; ++k) {
                            const int rel = k - q;
                            lm_data[q * lm_k + k] = (rel >= 0 && rel < window_size) ? 1.0f : 0.0f;
                        }
                    }
                    set_input_f32(local_mask->name, lm_data);
                }

                // Generate rotation frequencies for relative positional encoding.
                {
                    const int n_state     = hparams.n_embd;
                    const int d_half      = n_state / 2;
                    const float log_10000 = logf(10000.0f);
                    std::vector<float> freqs(d_half);
                    for (int k = 0; k < d_half; ++k) {
                        freqs[k] = expf(-(float(k * 2) * log_10000 / float(n_state)));
                    }
                    set_input_f32("pos_freqs", freqs);
                }

                // Generate relative positional distance values which scaled by
                // the frequency to produce the angles for sin/cos.
                {
                    // window_size is only known after graph construction since it depends on
                    // n_time from the conv output, so we read it back from the graph tensor.
                    struct ggml_tensor * rel_pos = ggml_graph_get_tensor(gf, "rel_positions");
                    const int window_size = rel_pos->ne[1];
                    std::vector<float> pos(window_size);
                    // local attention: window is fixed at [att_left, att_right]
                    // full attention: window covers the full sequence, centered
                    if (ggml_graph_get_tensor(gf, "local_mask")) {
                        const int att_left = window_size / 2;
                        for (int t = 0; t < window_size; ++t) {
                            pos[t] = float(att_left - t);
                        }
                    } else {
                        const int n_time = (window_size + 1) / 2;
                        for (int t = 0; t < window_size; ++t) {
                            pos[t] = float(n_time - 1 - t);
                        }
                    }
                    set_input_f32(rel_pos->name, pos);
                }
            } break;
        case PROJECTOR_TYPE_GRANITE_SPEECH:
            {
                const int context_size = model.hparams.audio_chunk_size;
                const int max_pos_emb  = model.hparams.audio_max_pos_emb;

                std::vector<int32_t> dists((size_t) context_size * (size_t) context_size);
                for (int i = 0; i < context_size; i++) {
                    for (int j = 0; j < context_size; j++) {
                        int d = i - j;
                        if (d < -context_size) d = -context_size;
                        if (d >  context_size) d =  context_size;
                        dists[(size_t) i * (size_t) context_size + (size_t) j] = d + max_pos_emb;
                    }
                }
                set_input_i32("attn_dists", dists);

                const int n_frames   = image_size_width;
                const int remainder  = n_frames % context_size;
                if (remainder > 0) {
                    const int num_blocks = (n_frames + context_size - 1) / context_size;
                    std::vector<float> mask((size_t) context_size * (size_t) context_size * (size_t) num_blocks, 0.0f);
                    const float neg_inf = -INFINITY;
                    const size_t last_block_offset = (size_t) (num_blocks - 1) * (size_t) context_size * (size_t) context_size;
                    for (int q = 0; q < context_size; q++) {
                        for (int k = 0; k < context_size; k++) {
                            if (q >= remainder || k >= remainder) {
                                mask[last_block_offset + (size_t) q * (size_t) context_size + (size_t) k] = neg_inf;
                            }
                        }
                    }
                    set_input_f32("attn_mask", mask);
                }
            } break;
        case PROJECTOR_TYPE_GRANITE4_VISION:
            {
                // Granite Vision 4.1 uses precomputed permutation index
                // tensors to express the _win / _unwin / spatial sampling
                // reshapes as ggml_get_rows gathers. The names are set
                // by g4v_gather() in models/granite4-vision.cpp.
                const int patch_size  = model.hparams.patch_size;
                const int image_side  = imgs.entries.front().nx() / patch_size;
                const int window_side = hparams.downsample_window_side;
                const int query_side  = hparams.downsample_query_side;
                const int n           = image_side / window_side;
                const int new_side    = n * query_side;

                // Builds the raster→window permutation indices for a
                // (side, side) grid split into (n × n) windows of (win × win)
                // tokens each.  dst[w * win*win + p] = source raster index.
                auto make_win_idx = [](int side, int win) {
                    const int nn = side / win;
                    std::vector<int32_t> idx(static_cast<size_t>(side) * side);
                    for (int wy = 0; wy < nn; ++wy) {
                        for (int wx = 0; wx < nn; ++wx) {
                            for (int iy = 0; iy < win; ++iy) {
                                for (int ix = 0; ix < win; ++ix) {
                                    const int w  = wy * nn + wx;
                                    const int p  = iy * win + ix;
                                    const int y  = wy * win + iy;
                                    const int x  = wx * win + ix;
                                    idx[static_cast<size_t>(w) * (win*win) + p] = y * side + x;
                                }
                            }
                        }
                    }
                    return idx;
                };

                auto make_unwin_idx = [&](int side, int win) {
                    const std::vector<int32_t> fwd = make_win_idx(side, win);
                    std::vector<int32_t> inv(fwd.size());
                    for (size_t i = 0; i < fwd.size(); ++i) {
                        inv[fwd[i]] = static_cast<int32_t>(i);
                    }
                    return inv;
                };

                auto make_spatial_idx = [](int side, int offset) {
                    const int off_y = (offset >> 1) & 1;
                    const int off_x = offset & 1;
                    const int new_s = side / 2;
                    std::vector<int32_t> idx(static_cast<size_t>(new_s) * new_s);
                    for (int y = 0; y < new_s; ++y) {
                        for (int x = 0; x < new_s; ++x) {
                            idx[y * new_s + x] = (y * 2 + off_y) * side + (x * 2 + off_x);
                        }
                    }
                    return idx;
                };

                // the same permutation is applied to every tile of the stacked image
                auto upload = [&](const std::string & name, const std::vector<int32_t> & idx) {
                    ggml_tensor * t = ggml_graph_get_tensor(gf, name.c_str());
                    GGML_ASSERT(t);
                    GGML_ASSERT(ggml_nelements(t) % (int64_t) idx.size() == 0);
                    const int n_rep = ggml_nelements(t) / idx.size();
                    std::vector<int32_t> buf;
                    buf.reserve(idx.size() * n_rep);
                    for (int i = 0; i < n_rep; ++i) {
                        buf.insert(buf.end(), idx.begin(), idx.end());
                    }
                    ggml_backend_tensor_set(t, buf.data(), 0, ggml_nbytes(t));
                };

                // Stage 1b only uses block 0's permutations; future stages
                // will upload all blocks.
                for (size_t bid = 0; bid < hparams.feature_layers.size(); ++bid) {
                    const std::string prefix = "g4v_blk" + std::to_string(bid) + "_";
                    upload(prefix + "win_idx",     make_win_idx(image_side, window_side));
                    upload(prefix + "qwin_idx",    make_win_idx(new_side, query_side));
                    upload(prefix + "unwin_idx",   make_unwin_idx(new_side, query_side));
                    const auto spatial_offset = hparams.proj_spatial_offsets[bid];
                    if (spatial_offset >= 0) {
                        upload(prefix + "spatial_idx", make_spatial_idx(image_side,spatial_offset));
                    }
                }
            } break;
        default:
            GGML_ABORT("Unknown projector type");
    }

    mtmd_backend_cpu_set_n_threads(backend_cpu, params->n_threads);

    auto status = ggml_backend_sched_graph_compute(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("%s: ggml_backend_sched_graph_compute failed with error %d\n", __func__, status);
        return false;
    }

    if (embeddings != nullptr) {
        // sanity check (assuming that all images in batch have the same number of tokens, so we only check the first one)
        const int n_tokens_out = embeddings->ne[1];
        const int expected_n_tokens_out = model.n_output_tokens(&imgs.entries[0]);
        if (n_tokens_out != expected_n_tokens_out) {
            LOG_ERR("%s: expected output %d tokens, got %d\n", __func__, expected_n_tokens_out, n_tokens_out);
            GGML_ABORT("Invalid number of output tokens");
        }

        LOG_DBG("%s: output embedding shape [%d, %d, %d]\n", __func__,
            (int)embeddings->ne[0], (int)embeddings->ne[1], (int)embeddings->ne[2]);

        // copy output to user buffer if provided
        // if output is empty, skip the copy
        auto & out_batch_embd = *params->out_embd;
        if (!out_batch_embd.empty()) {
            if (out_batch_embd.size() != (size_t)ggml_nelements(embeddings)) {
                LOG_ERR("%s: output buffer has %zu elements but expected %zu\n", __func__, out_batch_embd.size(), (size_t)ggml_nelements(embeddings));
                GGML_ABORT("Output buffer size mismatch");
            }
            ggml_backend_tensor_get(embeddings, out_batch_embd.data(), 0, ggml_nbytes(embeddings));
        } else {
            LOG_WRN("%s: output buffer is empty, skipping copy\n", __func__);
        }
    }

    //
    // for audio gen models
    //

    if (model.proj_type == PROJECTOR_TYPE_CHATTERBOX && params->gen_process == MTMD_SUBCTX_GEN_PROCESS_TOKENIZE) {
        ggml_tensor * fsq = ggml_graph_get_tensor(gf, "out_fsq");
        GGML_ASSERT(fsq != nullptr && params->out_codes);
        const int n_tok = (int) fsq->ne[1];
        std::vector<float> h((size_t) 8 * n_tok);
        ggml_backend_tensor_get(fsq, h.data(), 0, ggml_nbytes(fsq));

        // fsq round to base 3: h in (-1, 1) maps to digits {0, 1, 2}
        params->out_codes->resize(n_tok);
        for (int t = 0; t < n_tok; t++) {
            int32_t code = 0;
            for (int i = 7; i >= 0; i--) {
                const int32_t d = (int32_t) roundf(h[(size_t) t * 8 + i] * 0.9990000128746033f) + 1;
                code = code * 3 + d;
            }
            (*params->out_codes)[(size_t) t] = code;
        }

        // speech table rows of the codes, the multilingual variant adds its learned positions as its perceiver expects
        if (params->out_code_embd) {
            const auto & tensors = model.cbx_tensors;
            auto tab_it = tensors.find("a.gen.code.out_embd.weight");
            GGML_ASSERT(tab_it != tensors.end());
            ggml_tensor * tab = tab_it->second;
            const int n_e = (int) tab->ne[0];
            auto pos_it = tensors.find("a.gen.t3.speech_pos_emb");
            ggml_tensor * pos = pos_it == tensors.end() ? nullptr : pos_it->second;
            if ((tab->type != GGML_TYPE_F16 && tab->type != GGML_TYPE_F32) ||
                (pos && (pos->type != GGML_TYPE_F32 || pos->ne[0] != n_e || pos->ne[1] < n_tok))) {
                LOG_ERR("%s: unexpected speech table or positions\n", __func__);
                return false;
            }

            params->out_code_embd->resize((size_t) n_tok * n_e);
            std::vector<ggml_fp16_t> h16(n_e);
            std::vector<float> pr(n_e);
            for (int t = 0; t < n_tok; t++) {
                float * dst = params->out_code_embd->data() + (size_t) t * n_e;
                if ((*params->out_codes)[(size_t) t] >= tab->ne[1]) {
                    LOG_ERR("%s: speech code beyond the speech table\n", __func__);
                    return false;
                }
                const size_t r = (size_t) (*params->out_codes)[(size_t) t] * n_e;
                if (tab->type == GGML_TYPE_F16) {
                    ggml_backend_tensor_get(tab, h16.data(), r * sizeof(ggml_fp16_t), (size_t) n_e * sizeof(ggml_fp16_t));
                    for (int j = 0; j < n_e; j++) {
                        dst[j] = ggml_fp16_to_fp32(h16[j]);
                    }
                } else {
                    ggml_backend_tensor_get(tab, dst, r * sizeof(float), (size_t) n_e * sizeof(float));
                }
                if (pos) {
                    ggml_backend_tensor_get(pos, pr.data(), (size_t) t * n_e * sizeof(float), (size_t) n_e * sizeof(float));
                    for (int j = 0; j < n_e; j++) {
                        dst[j] += pr[j];
                    }
                }
            }
        }
        return true;
    }

    if (model.proj_type == PROJECTOR_TYPE_CHATTERBOX && params->gen_process == MTMD_SUBCTX_GEN_PROCESS_TTS_VOCODE) {
        ggml_tensor * sp = ggml_graph_get_tensor(gf, "out_spec");
        GGML_ASSERT(sp != nullptr && params->out_spec);
        params->out_spec->resize(ggml_nelements(sp));
        ggml_backend_tensor_get(sp, params->out_spec->data(), 0, ggml_nbytes(sp));
        return true;
    }

    if (model.proj_type == PROJECTOR_TYPE_CHATTERBOX && params->gen_process == MTMD_SUBCTX_GEN_PROCESS_TTS) {
        ggml_tensor * mel = ggml_graph_get_tensor(gf, "out_mel");
        GGML_ASSERT(mel != nullptr);
        std::vector<float> mel_data(ggml_nelements(mel));
        ggml_backend_tensor_get(mel, mel_data.data(), 0, ggml_nbytes(mel));

        // hift: f0 -> harmonic source -> source stft on the host, then the vocoder graph and the istft
        ggml_tensor * f0_t = ggml_graph_get_tensor(gf, "out_f0");
        GGML_ASSERT(f0_t != nullptr);
        const int n_mel_out = (int) ggml_nelements(f0_t);
        std::vector<float> f0(n_mel_out);
        ggml_backend_tensor_get(f0_t, f0.data(), 0, ggml_nbytes(f0_t));

        // SineGen: f0 upsampled x480, 9 harmonics with cumulative phase, uv gating and noise, merged by l_linear + tanh
        const int ups_total = 480;
        const double sr = 24000.0;
        const int64_t n_wav = (int64_t) n_mel_out * ups_total;
        std::vector<float> lw(9);
        float lb = 0.0f;
        if (model.cbx_read_tensor("a.gen.hift.m_source.l_linear.weight", lw.data(), lw.size()) != 9 ||
            model.cbx_read_tensor("a.gen.hift.m_source.l_linear.bias", &lb, 1) != 1) {
            LOG_ERR("%s: unexpected hift source module\n", __func__);
            return false;
        }

        std::mt19937 & srng = params->gen_state->rng;
        std::uniform_real_distribution<double> ud(-M_PI, M_PI);
        std::normal_distribution<float> snd(0.0f, 1.0f);
        double phase[9];
        for (int h = 0; h < 9; h++) {
            phase[h] = h == 0 ? 0.0 : ud(srng);
        }
        std::vector<float> src((size_t) n_wav);
        double cum[9] = {0.0};
        for (int64_t t = 0; t < n_wav; t++) {
            const float f = f0[(size_t) (t / ups_total)];
            const float uv = f > 10.0f ? 1.0f : 0.0f; // nsf_voiced_threshold
            const float namp = uv * 0.003f + (1.0f - uv) * 0.1f / 3.0f;
            float merged = lb;
            for (int h = 0; h < 9; h++) {
                cum[h] += (double) f * (h + 1) / sr;
                cum[h] -= floor(cum[h]);
                float sine = 0.1f * (float) sin(2.0 * M_PI * cum[h] + phase[h]);
                sine = sine * uv + namp * snd(srng);
                merged += lw[(size_t) h] * sine;
            }
            src[(size_t) t] = tanhf(merged);
        }

        // stft of the source: n_fft 16, hop 4, hann window, centered
        const int n_fft = 16, hop = 4, n_bins = 9;
        std::vector<float> win(n_fft);
        for (int i = 0; i < n_fft; i++) win[(size_t) i] = 0.5f - 0.5f * cosf(2.0f * (float) M_PI * i / n_fft);
        const int n_stft = (int) (n_wav / hop) + 1;
        std::vector<float> sstft((size_t) n_stft * 18);
        for (int fr = 0; fr < n_stft; fr++) {
            const int64_t c0 = (int64_t) fr * hop - n_fft / 2; // centered, reflect padded
            for (int k = 0; k < n_bins; k++) {
                double re = 0.0, im = 0.0;
                for (int i = 0; i < n_fft; i++) {
                    int64_t idx = c0 + i;
                    if (idx < 0) idx = -idx;
                    if (idx >= n_wav) idx = 2 * (n_wav - 1) - idx;
                    const double v = (double) src[(size_t) idx] * win[(size_t) i];
                    const double a = 2.0 * M_PI * k * i / n_fft;
                    re += v * cos(a);
                    im -= v * sin(a);
                }
                sstft[(size_t) fr * 18 + k    ] = (float) re;
                sstft[(size_t) fr * 18 + 9 + k] = (float) im;
            }
        }

        // vocoder graph on mel + source stft
        std::vector<float> spec;
        {
            mtmd_subctx_encode_params vp = *params;
            vp.gen_process   = MTMD_SUBCTX_GEN_PROCESS_TTS_VOCODE;
            vp.mel_in        = &mel_data;
            vp.sstft_in      = &sstft;
            vp.vocode_n_mel  = n_mel_out;
            vp.vocode_n_stft = n_stft;
            vp.out_audio     = nullptr;
            vp.out_spec      = &spec;
            if (!encode(&vp)) {
                LOG_ERR("%s: vocoder stage failed\n", __func__);
                return false;
            }
        }

        // istft: mag = clipped exp, phase = sin, hann overlap-add
        const int n_frames_out = (int) (spec.size() / 18);
        const int64_t n_out = (int64_t) (n_frames_out - 1) * hop;
        std::vector<double> acc((size_t) n_out + n_fft, 0.0);
        std::vector<double> wsum((size_t) n_out + n_fft, 0.0);
        for (int fr = 0; fr < n_frames_out; fr++) {
            double frame[16];
            for (int i = 0; i < n_fft; i++) {
                double v = 0.0;
                for (int k = 0; k < n_bins; k++) {
                    const double mag = fmin(exp((double) spec[(size_t) fr * 18 + k]), 1e2);
                    const double ph  = sin((double) spec[(size_t) fr * 18 + 9 + k]);
                    const double re  = mag * cos(ph), im = mag * sin(ph);
                    const double a   = 2.0 * M_PI * k * i / n_fft;
                    const double w   = (k == 0 || k == n_fft / 2) ? 1.0 : 2.0;
                    v += w * (re * cos(a) - im * sin(a));
                }
                frame[i] = v / n_fft;
            }
            const int64_t o = (int64_t) fr * hop;
            for (int i = 0; i < n_fft; i++) {
                acc [(size_t) (o + i)] += frame[i] * win[(size_t) i];
                wsum[(size_t) (o + i)] += (double) win[(size_t) i] * win[(size_t) i];
            }
        }
        GGML_ASSERT(params->out_audio);
        auto & out_audio = *params->out_audio;
        out_audio.resize((size_t) std::min<int64_t>(n_out - n_fft / 2, n_wav));
        for (size_t i = 0; i < out_audio.size(); i++) {
            const size_t j = i + n_fft / 2; // drop the centering pad
            const double v = wsum[j] > 1e-11 ? acc[j] / wsum[j] : 0.0;
            out_audio[(size_t) i] = (float) fmax(-0.99, fmin(0.99, v));
        }

        // trim_fade of the reference: silence 20 ms then fade in 20 ms, hides the artifact of the prompt boundary
        const size_t n_trim = 24000 / 50;
        for (size_t i = 0; i < 2 * n_trim && i < out_audio.size(); i++) {
            const double g = i < n_trim ? 0.0
                : (cos(M_PI * (1.0 - (double) (i - n_trim) / n_trim)) + 1.0) / 2.0;
            out_audio[(size_t) i] *= (float) g;
        }
        return true;
    }

    // optional outputs: a pipeline yields codes or feats, and not all have an eos head
    if (params->out_codes != nullptr) {
        ggml_tensor * codes = ggml_graph_get_tensor(gf, "out_codes");
        if (codes != nullptr) {
            auto & out_codes = *params->out_codes;
            out_codes.resize(ggml_nelements(codes));
            ggml_backend_tensor_get(codes, out_codes.data(), 0, ggml_nbytes(codes));
        }
    }
    if (params->out_feats != nullptr) {
        ggml_tensor * feats = ggml_graph_get_tensor(gf, "out_feats");
        if (feats != nullptr) {
            auto & out_feats = *params->out_feats;
            out_feats.resize(ggml_nelements(feats));
            ggml_backend_tensor_get(feats, out_feats.data(), 0, ggml_nbytes(feats));
        }
    }
    if (params->out_is_eos != nullptr) {
        ggml_tensor * eos = ggml_graph_get_tensor(gf, "out_eos_score");
        if (eos != nullptr) {
            GGML_ASSERT(ggml_nelements(eos) == 1);
            float score = 0.0f;
            ggml_backend_tensor_get(eos, &score, 0, sizeof(float));
            *params->out_is_eos = score > hparams.gen_eos_threshold;
        }
    }
    if (params->out_audio != nullptr) {
        ggml_tensor * audio = ggml_graph_get_tensor(gf, "out_audio");
        if (audio == nullptr) {
            GGML_ABORT("out_audio requested but graph has no \"out_audio\" tensor");
        }
        auto & out_audio = *params->out_audio;
        out_audio.resize(ggml_nelements(audio));
        ggml_backend_tensor_get(audio, out_audio.data(), 0, ggml_nbytes(audio));

        if (model.proj_type == PROJECTOR_TYPE_SOPRANO) {
            std::vector<float> spectrum = std::move(out_audio);
            mtmd_graph_soprano::decode_spectrum(model, spectrum, out_audio);
        }

        // drop the tail audio that comes from the code-0 rear padding, drama3 has no padding
        if (model.proj_type != PROJECTOR_TYPE_DRAMA3_GEN) {
            const int64_t n_codes    = params->codes ? model.gen_code_head_w->ne[2] + 1 : 0;
            const int64_t n_frames_w = hparams.wav_tfm_swa;
            const int64_t n_frames   = params->codes ? (int64_t) params->codes->size() / n_codes : n_frames_w;
            if (n_frames < n_frames_w) {
                const size_t hop = out_audio.size() / n_frames_w;
                out_audio.resize((size_t) n_frames * hop);
            }
        }
    }
    if (params->enc_state != nullptr && model.proj_type == PROJECTOR_TYPE_LFM3A) {
        const auto & slots = params->enc_state->slots;
        size_t i = 0;
        for (const auto & slot : list_lfm3a_state_slots(hparams, model)) {
            ggml_tensor * t = ggml_graph_get_tensor(gf, ("state_out_" + slot.name).c_str());
            GGML_ASSERT(t != nullptr);
            copy_gen_state_out(t, slots[i++]);
        }
    } else if (params->enc_state != nullptr && model.proj_type == PROJECTOR_TYPE_NEMOTRON_TRANSCRIBE) {
        mtmd_subctx_enc_state * st = params->enc_state;
        ggml_tensor * chunk = ggml_graph_get_tensor(gf, "spk_chunk");
        ggml_tensor * preds = ggml_graph_get_tensor(gf, "spk_preds");
        GGML_ASSERT(chunk != nullptr && preds != nullptr);
        st->spk_chunk.resize(ggml_nelements(chunk));
        st->spk_preds.resize(ggml_nelements(preds));
        ggml_backend_tensor_get(chunk, st->spk_chunk.data(), 0, ggml_nbytes(chunk));
        ggml_backend_tensor_get(preds, st->spk_preds.data(), 0, ggml_nbytes(preds));
    } else if (params->enc_state != nullptr) {
        mtmd_subctx_enc_state * st = params->enc_state;
        ggml_tensor * tail = ggml_graph_get_tensor(gf, "out_stem_tail");
        GGML_ASSERT(tail != nullptr);
        st->stem_tail.resize(ggml_nelements(tail));
        ggml_backend_tensor_get(tail, st->stem_tail.data(), 0, ggml_nbytes(tail));

        const int n_new = (imgs.entries[0].nx() - 2) / 2;
        for (int i = 0; i < n_new; i++) {
            st->slot_pos[(st->n_past + i) % st->n_ctx] = st->n_past + i;
        }
        st->n_past += n_new;
    }

    if (params->gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_WAV) {
        const auto & slots = params->gen_state->slots;
        size_t i = 0;
        for (const auto & slot : list_gen_state_slots(hparams, model)) {
            if (slot.in_place) {
                i++;
                continue;
            }
            ggml_tensor * t = ggml_graph_get_tensor(gf, ("state_out_" + slot.name).c_str());
            if (t == nullptr) {
                GGML_ABORT("graph has no \"state_out_%s\" tensor", slot.name.c_str());
            }
            GGML_ASSERT(i < slots.size());
            copy_gen_state_out(t, slots[i++]);
        }
    }

    //
    // Debug: dump final embeddings if MTMD_DEBUG_EMBEDDINGS is set
    //

    if (debug_output_embeddings && embeddings != nullptr) {
        const int64_t n_embd = embeddings->ne[0];
        const int64_t n_tokens = embeddings->ne[1];
        std::vector<float> emb_data(ggml_nelements(embeddings));
        ggml_backend_tensor_get(embeddings, emb_data.data(), 0, ggml_nbytes(embeddings));

        LOG_INF("\n=== MTMD_DEBUG_EMBEDDINGS ===\n");
        LOG_INF("Shape: [%lld, %lld]\n", (long long)n_embd, (long long)n_tokens);

        // TEMP debugging (parity validation), will be removed before merge
        // when the env var holds a path, dump the raw data: [int32 n_tokens][int32 n_embd][f32 data]
        const char * dump_path = std::getenv("MTMD_DEBUG_EMBEDDINGS");
        if (dump_path && strcmp(dump_path, "1") != 0) {
            FILE * f = fopen(dump_path, "wb");
            if (f) {
                const int32_t hdr[2] = { (int32_t)n_tokens, (int32_t)n_embd };
                fwrite(hdr, sizeof(hdr), 1, f);
                fwrite(emb_data.data(), sizeof(float), emb_data.size(), f);
                fclose(f);
            }
        }

        // Print first few values of first token
        LOG_INF("Token 0 (first 16 values): ");
        for (int i = 0; i < std::min((int64_t)16, n_embd); i++) {
            LOG_INF("%.6f ", emb_data[i]);
        }
        LOG_INF("\n");

        // Print last few values of first token
        if (n_embd > 16) {
            LOG_INF("Token 0 (last 16 values):  ");
            for (int64_t i = n_embd - 16; i < n_embd; i++) {
                LOG_INF("%.6f ", emb_data[i]);
            }
            LOG_INF("\n");
        }

        // Compute and print statistics
        float sum = 0.0f, sum_sq = 0.0f, min_val = emb_data[0], max_val = emb_data[0];
        for (size_t i = 0; i < emb_data.size(); i++) {
            sum += emb_data[i];
            sum_sq += emb_data[i] * emb_data[i];
            min_val = std::min(min_val, emb_data[i]);
            max_val = std::max(max_val, emb_data[i]);
        }
        float mean = sum / emb_data.size();
        float variance = (sum_sq / emb_data.size()) - (mean * mean);
        LOG_INF("Stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f, sum=%.6f\n",
                mean, sqrtf(variance), min_val, max_val, sum);
        LOG_INF("=== END MTMD_DEBUG_EMBEDDINGS ===\n\n");
    }

    return true;
}

std::map<ggml_backend_dev_t, size_t> mtmd_subcontext::get_mem_usage() const {
    std::map<ggml_backend_dev_t, size_t> result = model.mem_weights;
    for (auto & [dev, size] : mem_compute) {
        result[dev] += size;
    }
    return result;
}
