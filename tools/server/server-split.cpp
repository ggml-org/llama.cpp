#include "server-split.h"
#include "log.h"

#include "llama.h"
#include "llama-cpp.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "../../ggml/src/ggml-backend-impl.h"
#include "../../ggml/src/ggml-cuda/arena.h"
#include "../../src/llama-model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <future>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

static const ggml_cuda_arena_api * arena_api = nullptr;

static void check_arena(const char * error) {
    if (error) {
        throw std::runtime_error(error);
    }
}

static constexpr size_t block_bytes = 4096;
static int phase = 0;
using from_ptr_fn = ggml_backend_buffer_t (*)(ggml_backend_buffer_type_t, void *, size_t, void (*)(void *), void *);
static from_ptr_fn from_ptr = nullptr;
static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct pinned_bytes {
    void * data = nullptr;
    size_t size = 0;
    void resize(size_t bytes) {
        if (bytes <= size) return;
        reset();
        check_arena(arena_api->host_alloc(&data, bytes));
        size = bytes;
    }
    void reset() {
        if (data) arena_api->host_free(data);
        data = nullptr;
        size = 0;
    }
    ~pinned_bytes() { reset(); }
};

struct copy_plan {
    pinned_bytes packed_uploads;
    size_t moved = 0;
    size_t uploaded = 0;
    std::vector<ggml_cuda_arena_copy> local_rows;
    std::vector<ggml_cuda_arena_copy> upload_rows;
    std::vector<ggml_cuda_arena_span> gaps;
    size_t scratch_bytes = 0;
};

struct device_arena {
    int device;
    ggml_backend_buffer_type_t buft;
    decltype(ggml_backend_buffer_type::iface) original_iface;
    ggml_cuda_arena * handle = nullptr;
    uintptr_t base = 0;
    size_t mapped = 0;
    size_t used[2] = {};
    std::vector<std::pair<size_t, size_t>> free_spans[2];
    size_t weights[2] = {};
    pinned_bytes shadow[2];
    copy_plan plans[2];

    device_arena(int device, ggml_backend_buffer_type_t buft, size_t reserved = 0) : device(device), buft(buft), original_iface(buft->iface) {
        check_arena(arena_api->create(device, reserved, &handle, &base));
    }

    void * alloc(size_t bytes) {
        if (bytes > SIZE_MAX - (block_bytes - 1)) throw std::runtime_error("CUDA arena allocation size overflow");
        bytes = (bytes + block_bytes - 1) / block_bytes * block_bytes;
        auto & spans = free_spans[phase];
        for (size_t i = 0; i < spans.size(); i++) {
            if (spans[i].second >= bytes) {
                size_t offset = spans[i].first;
                spans[i].first += bytes;
                spans[i].second -= bytes;
                if (!spans[i].second) spans.erase(spans.begin() + i);
                return (void *) (base + offset);
            }
        }
        size_t offset = (used[phase] + block_bytes - 1) / block_bytes * block_bytes;
        if (bytes > SIZE_MAX - offset) throw std::runtime_error("CUDA arena allocation size overflow");
        size_t end = offset + bytes;
        if (end > mapped) {
            check_arena(arena_api->grow(handle, end, &mapped));
        }
        used[phase] = end;
        return (void *) (base + offset);
    }

    void release(int mode, size_t offset, size_t size) {
        auto & spans = free_spans[mode];
        spans.emplace_back(offset, size);
        std::sort(spans.begin(), spans.end());
        for (size_t i = 1; i < spans.size();) {
            if (spans[i - 1].first + spans[i - 1].second == spans[i].first) {
                spans[i - 1].second += spans[i].second;
                spans.erase(spans.begin() + i);
            } else i++;
        }
        if (!spans.empty() && spans.back().first + spans.back().second == used[mode]) {
            used[mode] = spans.back().first;
            spans.pop_back();
        }
    }

    ~device_arena() {
        buft->iface = original_iface;
        arena_api->destroy(handle);
    }
};

static std::vector<std::unique_ptr<device_arena>> arenas;
struct allocation {
    device_arena * arena;
    int mode;
    size_t offset;
    size_t size;
};
static void arena_release(void * data) {
    auto * block = (allocation *) data;
    block->arena->release(block->mode, block->offset, block->size);
    delete block;
}
static size_t arena_alignment(ggml_backend_buffer_type_t) { return block_bytes; }
static ggml_backend_buffer_t arena_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    for (auto & arena : arenas) {
        if (arena->buft == buft) {
            void * ptr = arena->alloc(size);
            auto * block = new allocation{arena.get(), phase, size_t((uintptr_t) ptr - arena->base),
                (size + block_bytes - 1) / block_bytes * block_bytes};
            return from_ptr(buft, ptr, size, arena_release, block);
        }
    }
    return nullptr;
}

static void build_tensor_plan(llama_model * tensor_model, llama_model * layer_model, int dst_phase) {
    const int n_devices = arenas.size();
    std::vector<std::vector<std::pair<size_t, size_t>>> occupied(n_devices);
    std::vector<const ggml_tensor *> parts(n_devices);
    std::vector<size_t> part_offset(n_devices);
    struct layer_weight {
        int device;
        size_t offset;
        size_t size;
    };
    std::multimap<std::string, layer_weight> layer_weights;
    for (const auto & [name, tensor] : layer_model->tensors_by_name) {
        for (const auto & arena : arenas) {
            if (!tensor->buffer || tensor->buffer->buft != arena->buft) continue;
            const size_t offset = (uintptr_t) tensor->data - arena->base;
            const size_t size = ggml_nbytes(tensor);
            if (offset > arena->weights[1] || size > arena->weights[1] - offset) throw std::runtime_error("layer weight exceeds CUDA arena: " + name);
            layer_weights.emplace(name, layer_weight{arena->device, offset, size});
            if (dst_phase) occupied[arena->device].emplace_back(offset, size);
            break;
        }
    }
    for (const auto & named : tensor_model->tensors_by_name) {
        const ggml_tensor * full = named.second;
        if (!full->buffer || !ggml_backend_buffer_is_meta(full->buffer)) continue;
        auto it = layer_weights.find(named.first);
        if (it == layer_weights.end()) throw std::runtime_error("missing CUDA layer weight: " + named.first);
        const int owner = it->second.device;
        const size_t layer_offset = it->second.offset;
        if (ggml_nbytes(full) != it->second.size) throw std::runtime_error("layer weight size mismatch: " + named.first);
        layer_weights.erase(it);
        auto ss = llama_meta_device_get_split_state(full, &tensor_model->get_split_state_ud);
        for (int j = 0; j < n_devices; j++) {
            parts[j] = ggml_backend_meta_buffer_simple_tensor(full, j);
            if (!ggml_nbytes(parts[j])) {
                part_offset[j] = 0;
                continue;
            }
            part_offset[j] = (uintptr_t) parts[j]->data - arenas[j]->base;
            if (part_offset[j] > arenas[j]->weights[0] || ggml_nbytes(parts[j]) > arenas[j]->weights[0] - part_offset[j]) {
                throw std::runtime_error("tensor shard exceeds CUDA arena");
            }
            if (!dst_phase) {
                size_t bytes = ggml_nbytes(parts[j]);
                occupied[j].emplace_back(part_offset[j], bytes);
                if (j != owner) {
                    auto & plan = arenas[j]->plans[0];
                    plan.upload_rows.push_back({part_offset[j], part_offset[j], bytes, bytes, bytes, 1, 0});
                    plan.uploaded += bytes;
                }
            }
        }
        auto add_local = [&](int j, size_t global, size_t local, size_t full_pitch, size_t part_pitch, size_t width, size_t rows) {
            if (!width || !rows) return;
            auto & plan = arenas[owner]->plans[dst_phase];
            if (rows > (SIZE_MAX - plan.scratch_bytes) / width) throw std::runtime_error("weight staging size overflow");
            ggml_cuda_arena_copy copy = {part_offset[j] + local, layer_offset + global, part_pitch, full_pitch, width, rows, plan.scratch_bytes};
            if (!dst_phase) {
                std::swap(copy.src, copy.dst);
                std::swap(copy.src_pitch, copy.dst_pitch);
            }
            const char * src = (const char *) arenas[owner]->shadow[1 - dst_phase].data;
            const char * dst = (const char *) arenas[owner]->shadow[dst_phase].data;
            auto check_range = [&](size_t offset, size_t pitch, size_t size) {
                if (offset > size || width > size - offset || !pitch || rows - 1 > (size - offset - width) / pitch) {
                    throw std::runtime_error("weight copy exceeds CUDA arena");
                }
            };
            check_range(copy.src, copy.src_pitch, arenas[owner]->weights[1 - dst_phase]);
            check_range(copy.dst, copy.dst_pitch, arenas[owner]->weights[dst_phase]);
            for (size_t row = 0; row < rows; row++) {
                if (memcmp(src + copy.src + row * copy.src_pitch, dst + copy.dst + row * copy.dst_pitch, width)) {
                    throw std::runtime_error("tensor split mapping differs from loaded weights: " + named.first);
                }
            }
            plan.local_rows.push_back(copy);
            plan.scratch_bytes += width * rows;
            plan.moved += width * rows;
        };
        if (ss.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            add_local(owner, 0, 0, ggml_nbytes(full), ggml_nbytes(full), ggml_nbytes(full), 1);
        } else if (ss.axis >= 0 && ss.axis < 3) {
            size_t full_pitch = full->nb[ss.axis + 1];
            if (!full_pitch) throw std::runtime_error("empty tensor split pitch");
            size_t rows = ggml_nbytes(full) / full_pitch;
            size_t global = 0, local = 0;
            for (size_t s = 0; s < ss.n_segments; s++) {
                for (size_t r = 0; r < ss.nr[s]; r++) {
                    for (int j = 0; j < n_devices; j++) {
                        size_t width = ss.axis == 0 ? ggml_row_size(full->type, ss.ne[n_devices * s + j]) : full->nb[ss.axis] * ss.ne[n_devices * s + j];
                        if (global > full_pitch || width > full_pitch - global) throw std::runtime_error("invalid tensor split extent");
                        if (j == owner) {
                            add_local(j, global, local, full_pitch, parts[j]->nb[ss.axis + 1], width, rows);
                            local += width;
                        } else if (dst_phase && width) {
                            auto & plan = arenas[owner]->plans[1];
                            plan.upload_rows.push_back({layer_offset + global, layer_offset + global, full_pitch, full_pitch, width, rows, 0});
                            plan.uploaded += width * rows;
                        }
                        global += width;
                    }
                }
            }
            if (global != full_pitch) throw std::runtime_error("invalid tensor split extent");
        } else throw std::runtime_error("unsupported weight split for tensor copy plan");
    }
    if (dst_phase) {
        // Layer mode can have a separate copy of the same weight on each GPU.
        for (const auto & entry : layer_weights) {
            const auto & [device, offset, size] = entry.second;
            if (!size) continue;
            auto & plan = arenas[device]->plans[1];
            plan.upload_rows.push_back({offset, offset, size, size, size, 1, 0});
            plan.uploaded += size;
        }
    }
    for (int j = 0; j < n_devices; j++) {
        auto & arena = *arenas[j];
        auto & plan = arena.plans[dst_phase];
        std::sort(occupied[j].begin(), occupied[j].end());
        size_t end = 0;
        for (auto span : occupied[j]) {
            if (span.first > end) plan.gaps.push_back({end, span.first - end});
            end = std::max(end, span.first + span.second);
        }
        if (arena.weights[dst_phase] > end) plan.gaps.push_back({end, arena.weights[dst_phase] - end});
        for (auto span : plan.gaps) memset((char *) arena.shadow[dst_phase].data + span.offset, 0, span.size);
        LOG_INF("hybrid: tensor plan GPU%d -> %s: local %.1f MiB, upload %.1f MiB, %zu local spans, %zu upload spans\n", j,
            dst_phase ? "layer" : "tensor", plan.moved / 1048576.0, plan.uploaded / 1048576.0, plan.local_rows.size(), plan.upload_rows.size());
    }
}

static void compact_uploads(device_arena & arena) {
    for (int mode = 0; mode < 2; mode++) {
        auto & plan = arena.plans[mode];
        plan.packed_uploads.resize(plan.uploaded);
        auto * packed = (char *) plan.packed_uploads.data;
        auto * full = (const char *) arena.shadow[mode].data;
        size_t offset = 0;
        for (auto & copy : plan.upload_rows) {
            if (!copy.width || !copy.rows || copy.rows > (plan.uploaded - offset) / copy.width ||
                    copy.src > arena.shadow[mode].size || copy.width > arena.shadow[mode].size - copy.src ||
                    (copy.rows > 1 && (!copy.src_pitch || copy.rows - 1 > (arena.shadow[mode].size - copy.src - copy.width) / copy.src_pitch))) {
                throw std::runtime_error("packed upload range is out of bounds");
            }
            for (size_t row = 0; row < copy.rows; row++) {
                memcpy(packed + offset + row * copy.width, full + copy.src + row * copy.src_pitch, copy.width);
            }
            copy.src = offset;
            copy.src_pitch = copy.width;
            offset += copy.width * copy.rows;
        }
        if (offset != plan.uploaded) throw std::runtime_error("invalid packed upload extent");
        arena.shadow[mode].reset();
    }
}

static void restore_weights(device_arena & arena, int dst_phase) {
    const auto & plan = arena.plans[dst_phase];
    check_arena(arena_api->restore(arena.handle, std::max(arena.weights[0], arena.weights[1]),
                plan.local_rows.data(), plan.local_rows.size(), plan.packed_uploads.data,
                plan.upload_rows.data(), plan.upload_rows.size(), plan.gaps.data(), plan.gaps.size()));
}

struct server_split_impl final : server_split {
    common_init_result_ptr loaded[2];
    common_speculative_init_result_ptr draft_loaded[2];
    common_speculative_ptr specs[2];
    pinned_bytes state;
    pinned_bytes state_draft;
    double prefill_ms[2] = {};
    double weight_ms[2] = {};
    double state_ms_per_byte = 1.0 / (4 * 1024 * 1024);
    size_t state_bytes_per_token = 0;
    bool broken = false;
    bool owns_arenas = false;
    bool had_pipeline_env = false;
    std::string pipeline_env;
    void (*shared_pool)(bool) = nullptr;

    ~server_split_impl() override {
        if (!owns_arenas) return;
        specs[0].reset();
        specs[1].reset();
        draft_loaded[0].reset();
        draft_loaded[1].reset();
        loaded[0].reset();
        loaded[1].reset();
        arenas.clear();
        shared_pool(false);
#ifdef _WIN32
        _putenv_s("LLAMA_PIPELINE_PARALLEL", pipeline_env.c_str());
#else
        if (had_pipeline_env) setenv("LLAMA_PIPELINE_PARALLEL", pipeline_env.c_str(), 1);
        else unsetenv("LLAMA_PIPELINE_PARALLEL");
#endif
    }

    llama_model * model() const override { return loaded[0]->model(); }
    llama_context * context() const override { return loaded[phase]->context(); }

    llama_context * draft_context() const override { return draft_loaded[phase] ? draft_loaded[phase]->context() : nullptr; }
    common_speculative * speculative() const override { return specs[phase].get(); }

    size_t state_size() const {
        return llama_state_seq_get_size(context(), 0) + (draft_context() ? llama_state_seq_get_size(draft_context(), 0) : 0);
    }

    void snapshot_model() {
        for (auto & arena : arenas) {
            size_t end = 0;
            auto add = [&](const ggml_tensor * tensor) {
                if (!tensor) return;
                auto * buffer = tensor->buffer;
                if (buffer && buffer->buft == arena->buft) {
                    size_t offset = (uintptr_t) ggml_backend_buffer_get_base(buffer) - arena->base;
                    size_t size = ggml_backend_buffer_get_size(buffer);
                    if (offset > arena->used[phase] || size > arena->used[phase] - offset) {
                        throw std::runtime_error("weight buffer exceeds CUDA arena");
                    }
                    end = std::max(end, offset + size);
                }
            };
            for (const auto & named : loaded[phase]->model()->tensors_by_name) {
                auto * tensor = named.second;
                if (tensor->buffer && ggml_backend_buffer_is_meta(tensor->buffer)) {
                    add(ggml_backend_meta_buffer_simple_tensor(tensor, arena->device));
                } else add(tensor);
            }
            if (!end) continue;
            if (end > arena->used[phase]) throw std::runtime_error("invalid hybrid weight extent");
            arena->weights[phase] = (end + block_bytes - 1) / block_bytes * block_bytes;
            arena->shadow[phase].resize(arena->weights[phase]);
            check_arena(arena_api->snapshot(arena->handle, arena->shadow[phase].data, arena->weights[phase]));
        }
    }

    void switch_to(int dst) {
        if (phase == dst) return;
        // A failed migration cannot leave either context available for inference.
        broken = true;
        double start = now_ms();
        llama_synchronize(context());
        size_t bytes = llama_state_seq_get_size(context(), 0);
        if (bytes > state.size) state.resize(std::max(bytes, state.size * 2));
        if (llama_state_seq_get_data(context(), (uint8_t *) state.data, bytes, 0) != bytes) throw std::runtime_error("hybrid state export failed");
        size_t bytes_draft = 0;
        std::vector<uint8_t> spec_state;
        if (draft_context()) {
            llama_synchronize(draft_context());
            bytes_draft = llama_state_seq_get_size(draft_context(), 0);
            if (bytes_draft > state_draft.size) state_draft.resize(std::max(bytes_draft, state_draft.size * 2));
            if (llama_state_seq_get_data(draft_context(), (uint8_t *) state_draft.data, bytes_draft, 0) != bytes_draft ||
                    !common_speculative_get_state(speculative(), 0, spec_state)) throw std::runtime_error("hybrid MTP state export failed");
        }
        double exported = now_ms();
        std::vector<std::future<void>> workers;
        for (size_t i = 1; i < arenas.size(); i++) {
            workers.push_back(std::async(std::launch::async, restore_weights, std::ref(*arenas[i]), dst));
        }
        restore_weights(*arenas.front(), dst);
        for (auto & worker : workers) worker.get();
        double copied = now_ms();
        phase = dst;
        llama_memory_clear(llama_get_memory(context()), true);
        if (llama_state_seq_set_data(context(), (uint8_t *) state.data, bytes, 0) != bytes) throw std::runtime_error("hybrid state import failed");
        llama_synchronize(context());
        if (draft_context()) {
            llama_memory_clear(llama_get_memory(draft_context()), true);
            if (llama_state_seq_set_data(draft_context(), (uint8_t *) state_draft.data, bytes_draft, 0) != bytes_draft) throw std::runtime_error("hybrid MTP state import failed");
            llama_synchronize(draft_context());
            common_speculative_set_state(speculative(), 0, spec_state);
            if (!dst) common_speculative_begin(speculative(), 0, {});
        }
        bytes += bytes_draft;
        double end = now_ms();
        weight_ms[dst] = copied - exported;
        if (bytes > 1024 * 1024) state_ms_per_byte = std::max(state_ms_per_byte, (exported - start + end - copied) / bytes);
        broken = false;
        LOG_INF("hybrid: switched to %s in %.1f ms (weights %.1f ms, state %.1f MiB)\n", dst ? "layer" : "tensor", end - start, weight_ms[dst], bytes / 1048576.0);
    }

    void external_begin() override {
        llama_synchronize(context());
        if (draft_context()) llama_synchronize(draft_context());
        for (auto & arena : arenas) arena->buft->iface = arena->original_iface;
        shared_pool(false);
    }

    void external_end() override {
        for (auto & arena : arenas) {
            arena->buft->iface.alloc_buffer = arena_alloc;
            arena->buft->iface.get_alignment = arena_alignment;
        }
        shared_pool(true);
    }

    void prepare(bool prefill, size_t remaining) override {
        if (broken) throw std::runtime_error("hybrid migration failed; reload the model");
        if (!prefill) { switch_to(0); return; }
        if (phase == 1) return;
        double saving = remaining * std::max(0.0, prefill_ms[0] - prefill_ms[1]);
        size_t bytes = llama_state_seq_get_size(context(), 0);
        if (draft_context()) bytes += llama_state_seq_get_size(draft_context(), 0);
        // Account for growing KV state and require headroom over the measured break-even point.
        double cost = weight_ms[0] + weight_ms[1] + 50 + (2.0 * bytes + double(remaining) * state_bytes_per_token) * state_ms_per_byte;
        LOG_INF("hybrid: prefill remaining %zu, estimated saving %.1f ms, round trip %.1f ms: %s\n", remaining, saving, cost, saving > 1.25 * cost ? "layer" : "tensor");
        if (saving > 1.25 * cost) switch_to(1);
    }

    void observe(size_t tokens, double ms) override {
        if (tokens >= 128) prefill_ms[phase] = 0.9 * prefill_ms[phase] + 0.1 * ms / tokens;
    }

    void init(common_params & params, const std::map<ggml_backend_dev_t, size_t> & reserved) {
        if (!arenas.empty()) throw std::runtime_error("only one hybrid model can be active in a process");
        const bool mtp = std::find(params.speculative.types.begin(), params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
        const bool other_spec = std::any_of(params.speculative.types.begin(), params.speculative.types.end(), [](common_speculative_type type) {
            return type != COMMON_SPECULATIVE_TYPE_NONE && type != COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
        });
        if (params.n_parallel != 1 || params.split_mode != LLAMA_SPLIT_MODE_TENSOR || params.embedding ||
                !params.lora_adapters.empty() || !params.control_vectors.empty() ||
                (!params.tensor_buft_overrides.empty() && params.tensor_buft_overrides.front().pattern) || params.moe_cache_size != 0 ||
                params.sampling.backend_sampling || params.speculative.has_dft() ||
                other_spec) {
            throw std::runtime_error("hybrid prefill requires tensor mode, -np 1, text generation, CPU sampling, and no adapters, tensor overrides, or MoE cache; only draft-mtp speculation is supported");
        }
        auto * reg = ggml_backend_reg_by_name("CUDA");
        const size_t n_devices = reg ? ggml_backend_reg_dev_count(reg) : 0;
        if (n_devices < 2 || n_devices > llama_max_devices()) throw std::runtime_error("hybrid prefill requires at least two supported CUDA GPUs");
        std::vector<ggml_backend_dev_t> devices;
        std::vector<float> layer_split(n_devices);
        std::istringstream placement(params.hybrid_prefill);
        for (size_t i = 0; i < n_devices; i++) {
            devices.push_back(ggml_backend_reg_dev_get(reg, i));
            char comma;
            if ((i && (!(placement >> comma) || comma != ',')) || !(placement >> layer_split[i]) ||
                    !std::isfinite(layer_split[i]) || layer_split[i] <= 0) {
                throw std::runtime_error("hybrid prefill expects one positive layer proportion per CUDA GPU");
            }
        }
        if (!(placement >> std::ws).eof()) throw std::runtime_error("unexpected data after hybrid prefill layer proportions");
        devices.push_back(nullptr);
        if (!params.devices.empty() && params.devices != devices) throw std::runtime_error("hybrid prefill requires all CUDA devices in backend order");
        auto get_api = (decltype(&ggml_backend_cuda_get_arena_api)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_get_arena_api_v1");
        if (!get_api) throw std::runtime_error("hybrid prefill requires a CUDA backend with VMM support on all devices in physical device order");
        arena_api = get_api();
        from_ptr = (from_ptr_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_buffer_from_ptr");
        shared_pool = (void (*)(bool)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_use_shared_pool");
        if (!from_ptr || !shared_pool) throw std::runtime_error("CUDA backend lacks hybrid arena support");
        const char * old_pipeline = getenv("LLAMA_PIPELINE_PARALLEL");
        had_pipeline_env = old_pipeline != nullptr;
        if (old_pipeline) pipeline_env = old_pipeline;
        owns_arenas = true;
        shared_pool(true);
#ifdef _WIN32
        _putenv_s("LLAMA_PIPELINE_PARALLEL", "0");
#else
        setenv("LLAMA_PIPELINE_PARALLEL", "0", 1);
#endif
        for (size_t i = 0; i < n_devices; i++) {
            auto * buft = ggml_backend_dev_buffer_type(devices[i]);
            auto found = reserved.find(devices[i]);
            size_t extra = found == reserved.end() ? 0 : found->second;
            LOG_INF("hybrid: reserving %.1f MiB on GPU%zu outside language-model arenas\n", extra / 1048576.0, i);
            arenas.emplace_back(new device_arena(i, buft, extra));
            buft->iface.alloc_buffer = arena_alloc;
            buft->iface.get_alignment = arena_alignment;
        }
        for (phase = 0; phase < 2; phase++) {
            common_params p = params;
            p.fit_params = false;
            p.devices = devices;
            p.n_gpu_layers = -2;
            p.warmup = false;
            p.split_mode = phase ? LLAMA_SPLIT_MODE_LAYER : LLAMA_SPLIT_MODE_TENSOR;
            if (phase) std::copy(layer_split.begin(), layer_split.end(), p.tensor_split);
            loaded[phase] = common_init_from_params(p);
            if (!loaded[phase]->model() || !loaded[phase]->context()) throw std::runtime_error("hybrid model/context load failed");
            if (mtp) {
                auto draft_params = common_base_params_to_speculative(p);
                draft_loaded[phase] = common_speculative_init_from_params(draft_params, loaded[phase]->model(), context());
                if (!draft_context()) throw std::runtime_error("hybrid MTP context load failed");
                if (llama_get_ctx_other(draft_context()) || llama_model_n_layer_nextn(loaded[phase]->model()) != 1) {
                    throw std::runtime_error("hybrid MTP requires one draft head with independent KV memory");
                }
                p.speculative.draft.ctx_tgt = context();
                p.speculative.draft.ctx_dft = draft_context();
                specs[phase].reset(common_speculative_init(p.speculative, 1));
                if (!speculative()) throw std::runtime_error("hybrid MTP initialization failed");
            }
            snapshot_model();
            auto * ctx = context();
            int n = std::min<int>(llama_n_ubatch(ctx), llama_n_ctx(ctx) / 4);
            if (n < 128) throw std::runtime_error("hybrid prefill calibration requires ubatch and context capacity >= 128 and 512");
            std::vector<llama_token> tokens(n, 0);
            auto clear = [&]() {
                llama_memory_clear(llama_get_memory(ctx), true);
                if (draft_context()) {
                    llama_memory_clear(llama_get_memory(draft_context()), true);
                    std::vector<uint8_t> empty;
                    if (!common_speculative_get_state(speculative(), 0, empty)) throw std::runtime_error("hybrid MTP state unavailable");
                    std::fill(empty.begin(), empty.end(), 0);
                    common_speculative_set_state(speculative(), 0, empty);
                }
            };
            auto warm = [&](int count) {
                auto batch = common_batch_get_one(ctx, tokens.data(), count);
                if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) throw std::runtime_error("hybrid calibration failed");
                llama_synchronize(ctx);
                if (speculative() && !common_speculative_process(speculative(), batch)) throw std::runtime_error("hybrid MTP calibration failed");
                if (draft_context()) llama_synchronize(draft_context());
            };
            for (int run = 0; run < 2; run++) {
                clear();
                double start = now_ms();
                warm(n);
                prefill_ms[phase] = (now_ms() - start) / n;
            }
            size_t bytes_prefill = state_size();
            warm(1);
            size_t bytes_decode = state_size();
            if (bytes_decode >= bytes_prefill) state_bytes_per_token = std::max(state_bytes_per_token, bytes_decode - bytes_prefill);
            if (mtp) warm(std::min(n, 1 + p.speculative.draft.n_max));
            clear();
            if (!phase) {
                params.sampling = p.sampling;
                params.ctx_shift = p.ctx_shift;
            }
        }
        build_tensor_plan(loaded[0]->model(), loaded[1]->model(), 0);
        build_tensor_plan(loaded[0]->model(), loaded[1]->model(), 1);
        size_t retained = 0;
        for (auto & arena : arenas) {
            const size_t weights = std::max(arena->weights[0], arena->weights[1]);
            const size_t scratch = std::max(arena->plans[0].scratch_bytes, arena->plans[1].scratch_bytes);
            if (scratch > SIZE_MAX - weights) throw std::runtime_error("weight staging size overflow");
            if (weights + scratch > arena->mapped) {
                LOG_INF("hybrid: GPU%d needs %.1f MiB more arena space for weight staging\n",
                    arena->device, (weights + scratch - arena->mapped) / 1048576.0);
                check_arena(arena_api->grow(arena->handle, weights + scratch, &arena->mapped));
            }
            LOG_INF("hybrid: GPU%d arena %.1f MiB, tensor %.1f MiB, layer %.1f MiB\n", arena->device,
                arena->mapped / 1048576.0, arena->used[0] / 1048576.0, arena->used[1] / 1048576.0);
            compact_uploads(*arena);
            retained += arena->plans[0].packed_uploads.size + arena->plans[1].packed_uploads.size;
        }
        LOG_INF("hybrid: retained upload buffers %.1f MiB pinned; full weight snapshots released\n", retained / 1048576.0);
        phase = 1;
        switch_to(0);
        switch_to(1);
        LOG_INF("hybrid: calibrated tensor %.3f ms/token, layer %.3f ms/token; initial prefill uses layer\n", prefill_ms[0], prefill_ms[1]);
    }
};
}

std::unique_ptr<server_split> server_split_init(common_params & params, const std::map<ggml_backend_dev_t, size_t> & reserved) {
    auto result = std::make_unique<server_split_impl>();
    result->init(params, reserved);
    return result;
}
