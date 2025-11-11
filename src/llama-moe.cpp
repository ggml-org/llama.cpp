#include "llama-moe.h"

#ifdef LLAMA_MOE_ENABLE

#include "llama-impl.h"
#include "llama-context.h"

#ifdef GGML_USE_CUDA
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>

static inline void llama_cuda_try(cudaError_t result, const char * expr) {
    if (result != cudaSuccess) {
        LLAMA_LOG_ERROR("%s: CUDA call failed: %s (%d)\n", __func__, cudaGetErrorString(result), result);
        GGML_ABORT("%s", expr);
    }
}
#endif

#ifdef GGML_USE_CUDA
void llama_moe_dispatch_cuda(const llama_moe_dispatch_desc & desc,
        ggml_tensor * dst,
        ggml_tensor * input,
        ggml_tensor * selected,
        ggml_tensor * weights);
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace {
constexpr uint32_t kDefaultMaxExperts = 16;
constexpr float kInvSqrt2 = 0.70710678118654752440f;
constexpr float kSwigluOaiAlpha = 1.702f;
constexpr float kSwigluOaiLimit = 7.0f;

[[nodiscard]] size_t llama_moe_required_device_bytes(const llama_moe_expert_handle & handle) {
    return handle.bytes;
}

static void llama_moe_copy_to_staging(const llama_moe_expert_handle & handle, void * dst) {
    if (handle.host_ptr == nullptr) {
        GGML_ABORT("expert handle host data unavailable");
    }

    std::memcpy(dst, handle.host_ptr, handle.bytes);
}
}

struct llama_moe_dispatch_userdata {
    llama_moe_dispatch_desc desc;
    std::vector<float> input_scaled;
    std::vector<float> up_buf;
    std::vector<float> gate_buf;
    std::vector<float> hidden_buf;
    std::vector<float> down_buf;
};
namespace {

[[nodiscard]] inline float act_silu(float x) {
    return x / (1.0f + std::exp(-x));
}

[[nodiscard]] inline float act_gelu(float x) {
    const float cdf = 0.5f * (1.0f + std::erf(x * kInvSqrt2));
    return x * cdf;
}

[[nodiscard]] inline float act_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

bool llama_moe_is_supported_handle(const llama_moe_expert_handle * h) {
    if (h == nullptr) {
        return false;
    }
    if (h->host_ptr == nullptr) {
        LLAMA_LOG_ERROR("%s: expert handle missing host data (id=%d)\n", __func__, h->id);
        return false;
    }
    if (h->is_quantized) {
        LLAMA_LOG_ERROR("%s: quantized experts not yet supported in CPU dispatcher (id=%d)\n", __func__, h->id);
        return false;
    }
    if (h->type != GGML_TYPE_F32) {
        LLAMA_LOG_ERROR("%s: unsupported tensor type %d for expert id=%d (only F32 supported)\n", __func__, (int) h->type, h->id);
        return false;
    }
    if (h->rows <= 0 || h->cols <= 0) {
        LLAMA_LOG_ERROR("%s: invalid tensor shape for expert id=%d (rows=%lld cols=%lld)\n", __func__, h->id, (long long) h->rows, (long long) h->cols);
        return false;
    }
    GGML_ASSERT(h->nb[0] == sizeof(float));
    return true;
}

void llama_moe_add_bias(const llama_moe_expert_handle * bias, float * vec, int64_t len) {
    if (!bias) {
        return;
    }
    if (!llama_moe_is_supported_handle(bias)) {
        GGML_ABORT("unsupported bias handle");
    }
    GGML_ASSERT(bias->rows == len || (bias->rows == 1 && len == bias->cols));
    const float * data = reinterpret_cast<const float *>(bias->host_ptr);
    for (int64_t i = 0; i < len; ++i) {
        vec[i] += data[i];
    }
}

bool llama_moe_matvec(const llama_moe_expert_handle * weight, int64_t cols, const float * input, float * output) {
    if (!llama_moe_is_supported_handle(weight)) {
        return false;
    }
    GGML_ASSERT(weight->cols == cols);
    const char * base = static_cast<const char *>(weight->host_ptr);
    const size_t row_stride = weight->nb[1];
    for (int64_t row = 0; row < weight->rows; ++row) {
        const float * row_ptr = reinterpret_cast<const float *>(base + row_stride * row);
        float sum = 0.0f;
        for (int64_t col = 0; col < cols; ++col) {
            sum += row_ptr[col] * input[col];
        }
        output[row] = sum;
    }
    return true;
}

void llama_moe_apply_activation(
        llm_ffn_op_type type,
        const float * gate,
        const float * up,
        float * hidden,
        int64_t n,
        bool has_gate) {
    switch (type) {
        case LLM_FFN_SILU:
        case LLM_FFN_SWIGLU:
            if (has_gate) {
                for (int64_t i = 0; i < n; ++i) {
                    hidden[i] = act_silu(gate[i]) * up[i];
                }
            } else {
                for (int64_t i = 0; i < n; ++i) {
                    hidden[i] = act_silu(up[i]);
                }
            }
            break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            if (has_gate) {
                for (int64_t i = 0; i < n; ++i) {
                    const float x = std::min(gate[i], kSwigluOaiLimit);
                    const float y = std::clamp(up[i], -kSwigluOaiLimit, kSwigluOaiLimit);
                    const float out_glu = x / (1.0f + std::exp(kSwigluOaiAlpha * (-x)));
                    hidden[i] = out_glu * (y + 1.0f);
                }
            } else {
                for (int64_t i = 0; i < n; ++i) {
                    hidden[i] = act_silu(up[i]);
                }
            }
            break;
        case LLM_FFN_GELU:
        case LLM_FFN_GEGLU:
            if (has_gate) {
                for (int64_t i = 0; i < n; ++i) {
                    hidden[i] = act_gelu(gate[i]) * up[i];
                }
            } else {
                for (int64_t i = 0; i < n; ++i) {
                    hidden[i] = act_gelu(up[i]);
                }
            }
            break;
        case LLM_FFN_RELU:
        case LLM_FFN_REGLU:
            if (has_gate) {
                for (int64_t i = 0; i < n; ++i) {
                    hidden[i] = act_relu(gate[i]) * up[i];
                }
            } else {
                for (int64_t i = 0; i < n; ++i) {
                    hidden[i] = act_relu(up[i]);
                }
            }
            break;
        case LLM_FFN_RELU_SQR:
            for (int64_t i = 0; i < n; ++i) {
                const float r = act_relu(up[i]);
                hidden[i] = r * r;
            }
            break;
        default:
            LLAMA_LOG_ERROR("%s: unsupported activation type %d\n", __func__, (int) type);
            GGML_ABORT("unsupported activation type");
    }
}

} // namespace

static void llama_moe_dispatch_kernel(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * user) {
    GGML_UNUSED(nth);

    if (ith != 0) {
        return;
    }

    auto * ud = static_cast<llama_moe_dispatch_userdata *>(user);
    GGML_ASSERT(ud != nullptr);
    const auto & desc = ud->desc;

    ExpertCache * cache = desc.cache;
    if (cache == nullptr && desc.ctx != nullptr) {
        cache = desc.ctx->get_expert_cache();
    }

    if (cache == nullptr) {
        LLAMA_LOG_ERROR("%s: ExpertCache unavailable\n", __func__);
        GGML_ABORT("missing expert cache");
    }

    ggml_tensor * input = dst->src[0];
    ggml_tensor * selected = dst->src[1];
    ggml_tensor * weights = dst->src[2];

    GGML_ASSERT(input != nullptr && selected != nullptr && weights != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(selected->type == GGML_TYPE_I32);

#ifdef GGML_USE_CUDA
    if (desc.use_cuda) {
        llama_moe_dispatch_cuda(desc, dst, input, selected, weights);
        return;
    }
#endif

    const int64_t n_embd = input->ne[0];
    const int64_t n_tokens = input->ne[1];
    const int64_t top_k = selected->ne[0];

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    GGML_ASSERT(weights->ne[1] == top_k);
    GGML_ASSERT(weights->ne[2] == n_tokens);

    const char * input_data = static_cast<const char *>(input->data);
    const char * selected_data = static_cast<const char *>(selected->data);
    const char * weight_data = static_cast<const char *>(weights->data);
    float * output_data = static_cast<float *>(dst->data);

    auto & input_scaled = ud->input_scaled;
    auto & up_buf = ud->up_buf;
    auto & gate_buf = ud->gate_buf;
    auto & hidden_buf = ud->hidden_buf;
    auto & down_buf = ud->down_buf;

    input_scaled.resize(n_embd);

    for (int64_t t = 0; t < n_tokens; ++t) {
        float * out = reinterpret_cast<float *>(reinterpret_cast<char *>(output_data) + t * dst->nb[1]);
        std::fill(out, out + n_embd, 0.0f);

        const float * input_vec = reinterpret_cast<const float *>(input_data + t * input->nb[1]);

        for (int64_t k = 0; k < top_k; ++k) {
            const int32_t expert_index = *reinterpret_cast<const int32_t *>(selected_data + k * selected->nb[0] + t * selected->nb[1]);
            if (expert_index < 0) {
                continue;
            }

            const float weight = *reinterpret_cast<const float *>(weight_data + k * weights->nb[1] + t * weights->nb[2]);
            if (!desc.weight_before_ffn && std::abs(weight) < std::numeric_limits<float>::min()) {
                continue;
            }

            const float * expert_input = input_vec;
            if (desc.weight_before_ffn) {
                for (int64_t i = 0; i < n_embd; ++i) {
                    input_scaled[i] = input_vec[i] * weight;
                }
                expert_input = input_scaled.data();
            }

            const auto fetch = [&](llama_moe_weight_kind kind) -> const llama_moe_expert_handle * {
                const int32_t id = llama_moe_compose_id(desc.layer, expert_index, kind);
                return cache->find(id);
            };

            const llama_moe_expert_handle * up_w = fetch(llama_moe_weight_kind::UP_WEIGHT);
            const llama_moe_expert_handle * gate_w = fetch(llama_moe_weight_kind::GATE_WEIGHT);
            const llama_moe_expert_handle * down_w = fetch(llama_moe_weight_kind::DOWN_WEIGHT);
            const llama_moe_expert_handle * up_b = desc.has_up_bias ? fetch(llama_moe_weight_kind::UP_BIAS) : nullptr;
            const llama_moe_expert_handle * gate_b = desc.has_gate_bias ? fetch(llama_moe_weight_kind::GATE_BIAS) : nullptr;
            const llama_moe_expert_handle * down_b = desc.has_down_bias ? fetch(llama_moe_weight_kind::DOWN_BIAS) : nullptr;

            if (!llama_moe_is_supported_handle(up_w) || !llama_moe_is_supported_handle(down_w)) {
                LLAMA_LOG_ERROR("%s: missing expert weights for layer %d expert %d\n", __func__, desc.layer, expert_index);
                GGML_ABORT("missing expert weights");
            }

            const int64_t n_ff = up_w->rows;
            if ((int64_t) up_w->cols != n_embd || (int64_t) down_w->cols != n_ff || (int64_t) down_w->rows != n_embd) {
                LLAMA_LOG_ERROR("%s: unexpected expert dimension mismatch (layer %d expert %d)\n", __func__, desc.layer, expert_index);
                GGML_ABORT("expert dim mismatch");
            }

            if ((int64_t) up_buf.size() < n_ff) up_buf.resize(n_ff);
            if ((int64_t) hidden_buf.size() < n_ff) hidden_buf.resize(n_ff);
            if ((int64_t) gate_buf.size() < n_ff) gate_buf.resize(n_ff);
            if ((int64_t) down_buf.size() < n_embd) down_buf.resize(n_embd);

            std::fill_n(up_buf.begin(), n_ff, 0.0f);
            std::fill_n(hidden_buf.begin(), n_ff, 0.0f);
            std::fill_n(gate_buf.begin(), n_ff, 0.0f);
            std::fill_n(down_buf.begin(), n_embd, 0.0f);

            if (!llama_moe_matvec(up_w, n_embd, expert_input, up_buf.data())) {
                GGML_ABORT("failed to compute up branch");
            }
            llama_moe_add_bias(up_b, up_buf.data(), n_ff);

            bool has_gate = gate_w != nullptr && llama_moe_is_supported_handle(gate_w);
            if (has_gate) {
                if (gate_w->rows != n_ff) {
                    LLAMA_LOG_ERROR("%s: gate rows mismatch (layer %d expert %d)\n", __func__, desc.layer, expert_index);
                    GGML_ABORT("gate dim mismatch");
                }
                if (!llama_moe_matvec(gate_w, n_embd, expert_input, gate_buf.data())) {
                    GGML_ABORT("failed to compute gate branch");
                }
                llama_moe_add_bias(gate_b, gate_buf.data(), n_ff);
            }

            llama_moe_apply_activation(desc.activation, gate_buf.data(), up_buf.data(), hidden_buf.data(), n_ff, has_gate);

            if (!llama_moe_matvec(down_w, n_ff, hidden_buf.data(), down_buf.data())) {
                GGML_ABORT("failed to compute down branch");
            }
            llama_moe_add_bias(down_b, down_buf.data(), n_embd);

            if (desc.weight_before_ffn) {
                for (int64_t i = 0; i < n_embd; ++i) {
                    out[i] += down_buf[i];
                }
            } else {
                for (int64_t i = 0; i < n_embd; ++i) {
                    out[i] += weight * down_buf[i];
                }
            }
        }
    }
}

ggml_tensor * llama_moe_build_dispatch(
        ggml_context * ctx,
        ggml_tensor * input,
        ggml_tensor * selected_experts,
        ggml_tensor * weights,
        const llama_moe_dispatch_desc & desc,
        llm_graph_result * owner) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(input != nullptr);
    GGML_ASSERT(selected_experts != nullptr);
    GGML_ASSERT(weights != nullptr);

    const int64_t ne0 = input->ne[0];
    const int64_t ne1 = input->ne[1];

    GGML_ASSERT(owner != nullptr);

    llama_moe_dispatch_desc local = desc;
    local.n_embd = static_cast<int32_t>(ne0);
    local.n_tokens = static_cast<int32_t>(ne1);
    local.n_expert_used = static_cast<int32_t>(selected_experts->ne[0]);

    if (local.n_expert == 0) {
        local.n_expert = local.n_expert_used;
    }

    auto * userdata = new llama_moe_dispatch_userdata{};
    userdata->desc = local;

    owner->add_cleanup([userdata]() { delete userdata; });

    ggml_tensor * srcs[3] = {input, selected_experts, weights};

    ggml_tensor * out = ggml_custom_4d(
            ctx,
            GGML_TYPE_F32,
            ne0,
            ne1,
            1,
            1,
            srcs,
            3,
            llama_moe_dispatch_kernel,
            1,
            userdata);

    return out;
}

ExpertCache::ExpertCache() {
    configure(Config{});
}

ExpertCache::ExpertCache(const Config & cfg) {
    configure(cfg);
}

ExpertCache::~ExpertCache() {
    clear();
    release_pool();
}

void ExpertCache::configure(const Config & cfg) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = cfg;
    if (config_.max_resident_experts == 0) {
        config_.max_resident_experts = kDefaultMaxExperts;
    }
    device_policies_ = cfg.device_policies;
    device_policy_by_id_.clear();
    device_policy_total_weight_ = 0.0;
    for (const auto & policy : device_policies_) {
        if (policy.device < 0) {
            continue;
        }
        device_policy_by_id_[policy.device] = policy;
        const double weight = policy.weight > 0.0f ? policy.weight : 1.0;
        device_policy_total_weight_ += weight;
    }
    auto_assign_devices_ = cfg.auto_assign_devices;
    auto_assign_devices_ = cfg.auto_assign_devices;
    release_pool();
    allocate_pool();
    reset_stats();
}

void ExpertCache::register_expert(const llama_moe_expert_handle & handle) {
    if (handle.tensor == nullptr || handle.bytes == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    llama_moe_expert_handle stored = handle;
    if (stored.id < 0 && stored.layer >= 0 && stored.expert_index >= 0 && stored.kind != llama_moe_weight_kind::UNKNOWN) {
        stored.id = llama_moe_compose_id(stored.layer, stored.expert_index, stored.kind);
    }

    if (stored.id < 0) {
        LLAMA_LOG_WARN("%s: skipping expert registration with invalid id (layer=%d expert=%d kind=%d)\n",
            __func__, stored.layer, stored.expert_index, static_cast<int>(stored.kind));
        return;
    }

    auto & slot = experts_[stored.id];
    slot.handle = stored;
    slot.slot_by_device.clear();
}

void ExpertCache::register_experts(const std::vector<llama_moe_expert_handle> & handles) {
    for (const auto & h : handles) {
        register_expert(h);
    }
}

void ExpertCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    release_pool();
    for (auto & kv : experts_) {
        kv.second.slot_by_device.clear();
    }
    timestamp_ = 0;
    allocate_pool();
}

bool ExpertCache::has_resident(int32_t expert_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = experts_.find(expert_id);
    if (it == experts_.end()) {
        return false;
    }
    return !it->second.slot_by_device.empty();
}

const llama_moe_expert_handle * ExpertCache::find(int32_t expert_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = experts_.find(expert_id);
    if (it == experts_.end()) {
        return nullptr;
    }
    return &it->second.handle;
}

llama_moe_cache_stats ExpertCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    llama_moe_cache_stats result = stats_;
    result.per_device.clear();
    size_t resident = 0;
    size_t capacity = 0;
    for (const auto & pool_entry : device_pools_) {
        const int device = pool_entry.first;
        const DevicePool & pool = pool_entry.second;
        size_t device_resident = 0;
        for (const auto & slot : pool_entry.second.slots) {
            if (slot.expert_id != -1) {
                ++resident;
                ++device_resident;
            }
        }
        capacity += pool.pool_bytes;
        llama_moe_cache_stats::device_stats dev_stats{};
        dev_stats.device = device;
        dev_stats.resident = device_resident;
        dev_stats.capacity_bytes = pool.pool_bytes;
        dev_stats.loads = pool.stats.loads;
        dev_stats.hits = pool.stats.hits;
        dev_stats.evictions = pool.stats.evictions;
        result.per_device.push_back(dev_stats);
    }
    result.resident = resident;
    result.capacity_bytes = capacity;
    return result;
}

void ExpertCache::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = {};
    for (auto & pool_entry : device_pools_) {
        pool_entry.second.stats = {};
    }
}

#ifdef GGML_USE_CUDA
void ExpertCache::attach_stream(cudaStream_t stream, int device) {
    std::lock_guard<std::mutex> lock(mutex_);
    stream_ = stream;
    device_streams_[device] = stream;
    current_device_ = device;
}

cudaStream_t ExpertCache::stream() const {
    return stream_;
}
#endif

void * ExpertCache::ensure_loaded(int32_t expert_id, int device, ggml_backend_buffer_t device_buffer) {
    GGML_UNUSED(device_buffer);
#ifndef GGML_USE_CUDA
    GGML_UNUSED(expert_id);
    GGML_UNUSED(device);
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.hits++;
    return nullptr;
#else
    std::lock_guard<std::mutex> lock(mutex_);

    if (device < 0) {
        device = current_device_;
    }
    device = select_device_for_expert(expert_id, device);

    if (device < 0) {
        LLAMA_LOG_ERROR("%s: device id not set for expert cache load\n", __func__);
        return nullptr;
    }

    auto it = experts_.find(expert_id);
    if (it == experts_.end()) {
        LLAMA_LOG_WARN("%s: expert %d not registered\n", __func__, expert_id);
        return nullptr;
    }

    auto & record = it->second;
    timestamp_++;

    DevicePool & pool = get_or_create_pool(device);

    if (record.slot_by_device.count(device) != 0) {
        size_t slot_idx = record.slot_by_device[device];
        DeviceSlot & slot = pool.slots[slot_idx];
        slot.last_used = timestamp_;
        slot.hits++;
        stats_.hits++;
        pool.stats.hits++;
        return slot.device_ptr;
    }

    if (record.handle.host_ptr == nullptr && record.handle.tensor) {
        record.handle.host_ptr = record.handle.tensor->data;
    }

    if (record.handle.host_ptr == nullptr) {
        LLAMA_LOG_WARN("%s: expert %d host data unavailable\n", __func__, expert_id);
        return nullptr;
    }

    DeviceSlot * target = nullptr;
    size_t target_idx = 0;
    for (size_t i = 0; i < pool.slots.size(); ++i) {
        if (pool.slots[i].expert_id == -1) {
            target = &pool.slots[i];
            target_idx = i;
            break;
        }
    }
    if (!target) {
        target = find_lru(device);
        if (target) {
            target_idx = static_cast<size_t>(target - pool.slots.data());
        }
    }

    if (!target) {
        LLAMA_LOG_ERROR("%s: unable to evict expert for expert_id=%d\n", __func__, expert_id);
        return nullptr;
    }

    const size_t required_bytes = llama_moe_required_device_bytes(record.handle);

    if (target->device_ptr == nullptr || target->bytes < required_bytes) {
        if (target->device_ptr != nullptr) {
            cudaFree(target->device_ptr);
        }
        llama_cuda_try(cudaMalloc(&target->device_ptr, required_bytes), "cudaMalloc expert cache");
        target->bytes = required_bytes;
    }

    if (target->host_staging == nullptr || target->staging_capacity < required_bytes) {
        if (target->host_staging != nullptr) {
            cudaFreeHost(target->host_staging);
        }
        llama_cuda_try(cudaMallocHost(&target->host_staging, required_bytes), "cudaMallocHost expert staging");
        target->staging_capacity = required_bytes;
    }

    llama_moe_copy_to_staging(record.handle, target->host_staging);

    bool evicting = target->expert_id != -1 && target->expert_id != expert_id;
    if (evicting) {
        auto evicted_it = experts_.find(target->expert_id);
        if (evicted_it != experts_.end()) {
            evicted_it->second.slot_by_device.erase(device);
        }
        stats_.evictions++;
        pool.stats.evictions++;
    }

    cudaStream_t stream = stream_;
    auto stream_it = device_streams_.find(device);
    if (stream_it != device_streams_.end()) {
        stream = stream_it->second;
    }

    llama_cuda_try(cudaMemcpyAsync(
        target->device_ptr,
        target->host_staging,
        required_bytes,
        cudaMemcpyHostToDevice,
        stream),
        "cudaMemcpyAsync expert upload");

    stats_.loads++;
    pool.stats.loads++;

    target->expert_id = expert_id;
    target->last_used = timestamp_;
    target->hits = 1;

    record.slot_by_device[device] = target_idx;

    return target->device_ptr;
#endif
}

void ExpertCache::prefetch(const std::vector<int32_t> & expert_ids) {
#ifdef GGML_USE_CUDA
    if (!config_.enable_prefetch || expert_ids.empty()) {
        return;
    }
    int device = -1;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        device = current_device_;
        if (device < 0) {
            return;
        }
        stats_.prefetch_requests += expert_ids.size();
    }
    for (int32_t id : expert_ids) {
        ensure_loaded(id, device);
    }
#else
    GGML_UNUSED(expert_ids);
#endif
}

size_t ExpertCache::resident_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto & pool_entry : device_pools_) {
        for (const auto & slot : pool_entry.second.slots) {
            if (slot.expert_id != -1) {
                ++count;
            }
        }
    }
    return count;
}

size_t ExpertCache::capacity_bytes() const {
    return pool_bytes_;
}

void ExpertCache::allocate_pool() {
    pool_bytes_ = 0;
    if (!device_policies_.empty()) {
        for (const auto & policy : device_policies_) {
            if (policy.device < 0) {
                continue;
            }
            DevicePool pool;
            pool.pool_bytes = capacity_for_device(policy.device);
            const size_t slot_count = max_slots_for_device(policy.device);
            pool.slots.resize(slot_count);
            for (auto & slot : pool.slots) {
                slot.expert_id = -1;
                slot.device_ptr = nullptr;
                slot.bytes = 0;
                slot.last_used = 0;
                slot.hits = 0;
                slot.host_staging = nullptr;
                slot.staging_capacity = 0;
            }
            pool.stats = {};
            device_pools_.emplace(policy.device, std::move(pool));
            pool_bytes_ += capacity_for_device(policy.device);
        }
    } else {
        pool_bytes_ = config_.vram_pool_bytes;
    }
}

void ExpertCache::release_pool() {
#ifdef GGML_USE_CUDA
    for (auto & pool_entry : device_pools_) {
        for (auto & slot : pool_entry.second.slots) {
            if (slot.device_ptr != nullptr) {
                cudaFree(slot.device_ptr);
                slot.device_ptr = nullptr;
            }
            if (slot.host_staging != nullptr) {
                cudaFreeHost(slot.host_staging);
                slot.host_staging = nullptr;
                slot.staging_capacity = 0;
            }
            slot.expert_id = -1;
            slot.bytes = 0;
            slot.last_used = 0;
            slot.hits = 0;
        }
        pool_entry.second.stats = {};
    }
#endif
    device_pools_.clear();
    pool_bytes_ = 0;
}

ExpertCache::DeviceSlot * ExpertCache::find_lru(int device) {
#ifdef GGML_USE_CUDA
    auto it = device_pools_.find(device);
    if (it == device_pools_.end()) {
        return nullptr;
    }
    DeviceSlot * candidate = nullptr;
    for (auto & slot : it->second.slots) {
        if (slot.expert_id == -1) {
            return &slot;
        }
        if (candidate == nullptr || slot.last_used < candidate->last_used) {
            candidate = &slot;
        }
    }
    return candidate;
#else
    GGML_UNUSED(device);
    return nullptr;
#endif
}

ExpertCache::DevicePool & ExpertCache::get_or_create_pool(int device) {
    auto it = device_pools_.find(device);
    if (it == device_pools_.end()) {
        DevicePool pool;
        pool.pool_bytes = capacity_for_device(device);
        const size_t slot_count = max_slots_for_device(device);
        pool.slots.resize(slot_count);
        for (auto & slot : pool.slots) {
            slot.expert_id = -1;
            slot.device_ptr = nullptr;
            slot.bytes = 0;
            slot.last_used = 0;
            slot.hits = 0;
            slot.host_staging = nullptr;
            slot.staging_capacity = 0;
        }
        pool.stats = {};
        it = device_pools_.emplace(device, std::move(pool)).first;
        pool_bytes_ += capacity_for_device(device);
    }
    return it->second;
}

size_t ExpertCache::capacity_for_device(int device) const {
    auto it = device_policy_by_id_.find(device);
    if (it != device_policy_by_id_.end() && it->second.capacity_bytes > 0) {
        return it->second.capacity_bytes;
    }
    return config_.vram_pool_bytes;
}

size_t ExpertCache::max_slots_for_device(int device) const {
    auto it = device_policy_by_id_.find(device);
    uint32_t slots = config_.max_resident_experts;
    if (it != device_policy_by_id_.end() && it->second.max_resident_experts > 0) {
        slots = it->second.max_resident_experts;
    }
    if (slots == 0) {
        slots = kDefaultMaxExperts;
    }
    return slots;
}

int ExpertCache::select_device_for_expert(int32_t expert_id, int device_hint) const {
    if (device_hint >= 0) {
        return device_hint;
    }
    if (!device_policies_.empty() && device_policy_total_weight_ > 0.0) {
        const uint64_t hash = std::hash<int32_t>{}(expert_id);
        const double normalized = static_cast<double>(hash) / static_cast<double>(std::numeric_limits<uint64_t>::max());
        const double target = normalized * device_policy_total_weight_;
        double accum = 0.0;
        for (const auto & policy : device_policies_) {
            if (policy.device < 0) {
                continue;
            }
            const double weight = policy.weight > 0.0f ? policy.weight : 1.0;
            accum += weight;
            if (target <= accum) {
                return policy.device;
            }
        }
        return device_policies_.back().device;
    }

    if (!device_pools_.empty()) {
        const size_t index = static_cast<size_t>(std::abs(expert_id)) % device_pools_.size();
        auto it = device_pools_.begin();
        std::advance(it, index);
        return it->first;
    }

    if (current_device_ >= 0) {
        return current_device_;
    }

    return device_hint;
}

#endif // LLAMA_MOE_ENABLE
