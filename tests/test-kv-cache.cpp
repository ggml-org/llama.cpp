#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

static void init_model(llama_model & model, uint32_t n_layer) {
    model.hparams = {};
    model.hparams.n_layer_all = n_layer;
    model.hparams.n_embd_head_k_full = 4;
    model.hparams.n_embd_head_v_full = 4;
    model.hparams.n_head_arr.fill(1);
    model.hparams.n_head_kv_arr.fill(1);
    std::iota(model.hparams.kv_source_impl.begin(), model.hparams.kv_source_impl.end(), 0);
    model.layers.resize(n_layer);
}

struct state_writer : llama_io_write_i {
    std::vector<uint8_t> data;
    std::vector<std::pair<size_t, size_t>> tensors;

    void write(const void * src, size_t size) override {
        const auto * ptr = static_cast<const uint8_t *>(src);
        data.insert(data.end(), ptr, ptr + size);
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        tensors.emplace_back(data.size(), size);
        const size_t start = data.size();
        data.resize(start + size);
        ggml_backend_tensor_get(tensor, data.data() + start, offset, size);
    }

    size_t n_bytes() override { return data.size(); }
};

struct state_reader : llama_io_read_i {
    const std::vector<uint8_t> & data;
    size_t pos = 0;
    struct tensor_read { ggml_tensor * tensor; size_t offset; size_t size; };
    std::vector<tensor_read> tensors;

    explicit state_reader(const std::vector<uint8_t> & data) : data(data) {}

    void read(void * dst, size_t size) override {
        if (size > data.size() - pos) {
            throw std::runtime_error("truncated state");
        }
        memcpy(dst, data.data() + pos, size);
        pos += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        std::vector<uint8_t> bytes(size);
        read(bytes.data(), size);
        ggml_backend_tensor_set(tensor, bytes.data(), offset, size);
        tensors.push_back({tensor, offset, size});
    }

    size_t n_bytes() override { return pos; }
};

static void test_iswa_restore_failure() {
    for (bool variable : { false, true }) {
        for (bool v_trans : { false, true }) {
            std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
            init_model(*model, 2);
            model->hparams.n_head_arr.fill(2);
            model->hparams.n_embd_head_k_swa = 4;
            model->hparams.n_embd_head_v_swa = 4;
            model->hparams.n_swa = 16;
            model->hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
            model->hparams.is_swa_impl[1] = true;
            model->hparams.n_head_kv_arr[1] = variable ? 2 : 1;
            llama_kv_cache_iswa cache(*model, false, GGML_TYPE_F32, GGML_TYPE_F32,
                    v_trans, false, true, true, 256, 2, 32, nullptr, nullptr, nullptr, nullptr);
            cache.clear(true);

            llama_batch_allocr balloc(1);
            auto ubatch = balloc.ubatch_reserve(1, 1);
            llama_seq_id seq_id = 0;
            ubatch.seq_id_unq[0] = seq_id;
            ubatch.seq_id[0] = &seq_id;
            ubatch.n_seq_id[0] = 1;
            ubatch.pos[0] = 0;
            for (auto * kv : { cache.get_base(), cache.get_swa() }) {
                kv->apply_ubatch(kv->find_slot(ubatch, false), ubatch);
            }

            state_writer saved;
            cache.get_base()->state_write(saved, 0);
            const size_t swa_offset = saved.data.size();
            cache.get_swa()->state_write(saved, 0);
            cache.clear(true);

            // Keep a live sequence while the other sequence fails to restore.
            seq_id = 1;
            ubatch.seq_id_unq[0] = seq_id;
            for (auto * kv : { cache.get_base(), cache.get_swa() }) {
                kv->apply_ubatch(kv->find_slot(ubatch, false), ubatch);
            }
            state_writer before;
            cache.state_write(before, 1);
            for (const auto & range : before.tensors) {
                std::fill_n(before.data.begin() + range.first, range.second, 0x3e);
            }
            state_reader live(before.data);
            cache.state_read(live, 1);

            for (const auto & range : saved.tensors) {
                std::fill_n(saved.data.begin() + range.first, range.second, 0xff);
            }
            std::fill_n(saved.data.begin() + swa_offset, sizeof(uint32_t), 0xff);
            state_reader reader(saved.data);
            bool failed = false;
            try {
                cache.state_read(reader, 0);
            } catch (const std::exception &) {
                failed = true;
            }
            GGML_ASSERT(failed);
            GGML_ASSERT(cache.get_base()->seq_pos_max(0) == -1);
            GGML_ASSERT(cache.get_swa()->seq_pos_max(0) == -1);
            state_writer after;
            cache.state_write(after, 1);
            GGML_ASSERT(before.data == after.data);

            GGML_ASSERT(!reader.tensors.empty());
            for (const auto & range : reader.tensors) {
                std::vector<uint8_t> bytes(range.size);
                ggml_backend_tensor_get(range.tensor, bytes.data(), range.offset, range.size);
                for (uint8_t byte : bytes) {
                    GGML_ASSERT(byte == 0);
                }
            }
        }
    }
}

int main() {
    for (int bit = 0; bit < 64; ++bit) {
        GGML_ASSERT(ctz64(uint64_t{1} << bit) == bit);
        GGML_ASSERT(ctz64(UINT64_MAX << bit) == bit);
    }

    llama_backend_init();

    {
        std::string errors;
        llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
            if (level == GGML_LOG_LEVEL_ERROR) {
                *static_cast<std::string *>(user_data) += text;
            }
        }, &errors);

        for (llm_arch arch : { LLM_ARCH_DEEPSEEK2, LLM_ARCH_DEEPSEEK32, LLM_ARCH_DEEPSEEK4,
                LLM_ARCH_DEEPSEEK41, LLM_ARCH_GLM_DSA, LLM_ARCH_GLM5_NEXT, LLM_ARCH_DOTS3NOTE,
                LLM_ARCH_MISTRAL_LARGE4, LLM_ARCH_HY_V4, LLM_ARCH_MINIMAX_M3, LLM_ARCH_QWEN4EXP,
                LLM_ARCH_DFLASH }) {
            std::unique_ptr<llama_model> model(llama_model_create(arch, llama_model_default_params()));
            init_model(*model, 1);
            if (arch == LLM_ARCH_DEEPSEEK2) {
                model->hparams.n_embd_head_k_mla_impl = 4;
                model->hparams.n_embd_head_v_mla_impl = 4;
            }

            for (ggml_type type_k : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                for (ggml_type type_v : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                    if (type_k == GGML_TYPE_F16 && type_v == GGML_TYPE_F16) {
                        continue;
                    }
                    auto params = llama_context_default_params();
                    params.type_k = type_k;
                    params.type_v = type_v;
                    errors.clear();
                    llama_context * ctx = llama_init_from_model(model.get(), params);
                    GGML_ASSERT(ctx == nullptr);
                    if (type_k == type_v || (!model->hparams.is_mla() && arch != LLM_ARCH_DEEPSEEK4 && arch != LLM_ARCH_DEEPSEEK41)) {
                        GGML_ASSERT(errors.find("FP8 cache is not supported for architecture") != std::string::npos);
                    } else {
                        GGML_ASSERT(errors.find("model does not support different K") != std::string::npos);
                    }
                }
            }
        }

        llama_log_set(nullptr, nullptr);
    }

    {
        std::unique_ptr<llama_model> target(llama_model_create(LLM_ARCH_GEMMA4, llama_model_default_params()));
        std::unique_ptr<llama_model> assistant(llama_model_create(LLM_ARCH_GEMMA4_ASSISTANT, llama_model_default_params()));
        init_model(*target, 3);
        init_model(*assistant, 2);

        ggml_context_ptr ctx(ggml_init({ 4096, nullptr, false }));
        target->layers[2].k_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        target->layers[2].v_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[0].k_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[0].v_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[1].k_cache_s = assistant->layers[0].k_cache_s;
        assistant->layers[1].v_cache_s = assistant->layers[0].v_cache_s;
        *static_cast<float *>(target->layers[2].k_cache_s->data) = 0.5f;
        *static_cast<float *>(target->layers[2].v_cache_s->data) = 0.25f;
        *static_cast<float *>(assistant->layers[0].k_cache_s->data) = 2.0f;
        *static_cast<float *>(assistant->layers[0].v_cache_s->data) = 4.0f;

        for (ggml_type type_k : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
            for (ggml_type type_v : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                llama_kv_cache source(*target, false, target->hparams, type_k, type_v,
                        false, false, true, 32, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr,
                        [](uint32_t il) { return il == 0 ? 2 : -1; }, nullptr);
                llama_kv_cache shared(*assistant, false, assistant->hparams, type_k, type_v,
                        false, false, true, 32, 1, 0, LLAMA_SWA_TYPE_NONE, &source, nullptr, nullptr,
                        [](int32_t il) { return il == 0 ? 2 : 1; });

                GGML_ASSERT(source.get_k_scale(2) == (type_k == GGML_TYPE_F8_E4M3 ? target->layers[2].k_cache_s : nullptr));
                GGML_ASSERT(source.get_v_scale(2) == (type_v == GGML_TYPE_F8_E4M3 ? target->layers[2].v_cache_s : nullptr));
                GGML_ASSERT(source.get_k_scale(0) == source.get_k_scale(2));
                GGML_ASSERT(source.get_v_scale(0) == source.get_v_scale(2));
                GGML_ASSERT(shared.get_k_storage(0) == source.get_k_storage(2));
                GGML_ASSERT(shared.get_k_scale(0) == source.get_k_scale(2));
                GGML_ASSERT(shared.get_v_scale(0) == source.get_v_scale(2));
                GGML_ASSERT(shared.get_k_scale(1) == nullptr);
                GGML_ASSERT(shared.get_v_scale(1) == nullptr);
            }
        }
    }

    test_iswa_restore_failure();

    llama_backend_free();
    return 0;
}
