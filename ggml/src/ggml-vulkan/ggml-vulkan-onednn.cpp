#define GGML_VULKAN_ONEDNN_BUILD
#include "ggml-vulkan-onednn.h"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>
#if defined(_WIN32)
#include <level_zero/ze_api.h>
#endif

#include <cmath>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <chrono>
#include <algorithm>

using namespace dnnl;
using namespace dnnl::graph;
using dt = logical_tensor::data_type;
using dims = logical_tensor::dims;

struct ids_t { size_t q, k, ks, divisor, mask, v, vs, out; };
struct graph_t { dnnl::graph::graph g; ids_t ids; };

struct graph_key {
    int q, kv, h, hkv, d;
    bool operator==(const graph_key & other) const { return q == other.q && kv == other.kv && h == other.h && hkv == other.hkv && d == other.d; }
};

struct graph_key_hash {
    size_t operator()(const graph_key & k) const {
        size_t h = (size_t) k.q;
        h = h * 131 + (size_t) k.kv;
        h = h * 131 + (size_t) k.h;
        h = h * 131 + (size_t) k.hkv;
        return h * 131 + (size_t) k.d;
    }
};

struct cached_graph {
    std::shared_ptr<graph_t> graph;
    std::shared_ptr<dnnl::graph::compiled_partition> partition;
    std::vector<logical_tensor> inputs;
    std::vector<logical_tensor> outputs;
};

static sycl::device pick_device() {
    for (const auto & dev : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        const auto name = dev.get_info<sycl::info::device::name>();
        if (name.find("A750") == std::string::npos) continue;
        if (dev.get_backend() == sycl::backend::ext_oneapi_level_zero) return dev;
    }
    throw std::runtime_error("Intel Arc A750 Level Zero GPU not found");
}

static graph_t make_graph(int q, int kv) {
    size_t id = 0;
    const int hkv = 4, rep = 4, d = 256, ng = 8;
    const dims qd = {hkv, rep, q, d};
    const dims kd = {hkv, 1, d, kv};
    const dims ksd = {hkv, 1, ng, kv};
    const dims vd = {hkv, 1, kv, d};
    const dims vsd = {hkv, 1, kv, ng};
    const dims scored = {hkv, rep, q, kv};

    auto Q = logical_tensor(id++, dt::f16, qd, logical_tensor::layout_type::strided);
    auto K = logical_tensor(id++, dt::s8, kd, logical_tensor::layout_type::strided);
    auto KS = logical_tensor(id++, dt::f16, ksd, logical_tensor::layout_type::strided);
    auto divisor = logical_tensor(id++, dt::f16, dims{1}, logical_tensor::layout_type::strided);
    auto mask = logical_tensor(id++, dt::f16, dims{1, 1, q, kv}, logical_tensor::layout_type::strided);
    auto V = logical_tensor(id++, dt::s8, vd, logical_tensor::layout_type::strided);
    auto VS = logical_tensor(id++, dt::f16, vsd, logical_tensor::layout_type::strided);
    auto KF = logical_tensor(id++, dt::f16, kd, logical_tensor::layout_type::strided);
    auto score = logical_tensor(id++, dt::f32, scored, logical_tensor::layout_type::strided);
    auto scaled = logical_tensor(id++, dt::f32, scored, logical_tensor::layout_type::strided);
    auto masked = logical_tensor(id++, dt::f32, scored, logical_tensor::layout_type::strided);
    auto probs = logical_tensor(id++, dt::f16, scored, logical_tensor::layout_type::strided);
    auto VF = logical_tensor(id++, dt::f16, vd, logical_tensor::layout_type::strided);
    auto out = logical_tensor(id++, dt::f16, qd, logical_tensor::layout_type::strided);

    op dqk(id++, op::kind::DynamicDequantize, "dq_k_g32_s8");
    dqk.set_attr<std::string>(op::attr::qtype, "per_group");
    dqk.set_attr<dims>(op::attr::group_shape, dims{1, 1, 32, 1});
    dqk.set_attr<int64_t>(op::attr::axis, 2);
    dqk.add_inputs({K, KS}); dqk.add_outputs({KF});
    op qk(id++, op::kind::MatMul, "qk_matmul"); qk.add_inputs({Q, KF}); qk.add_outputs({score});
    op div(id++, op::kind::Divide, "scale_div"); div.add_inputs({score, divisor}); div.add_outputs({scaled});
    op add(id++, op::kind::Add, "mask_add"); add.add_inputs({scaled, mask}); add.add_outputs({masked});
    op sm(id++, op::kind::SoftMax, "softmax"); sm.set_attr<int64_t>(op::attr::axis, -1); sm.set_attr<std::string>(op::attr::mode, "inf_as_zero"); sm.add_inputs({masked}); sm.add_outputs({probs});
    op dqv(id++, op::kind::DynamicDequantize, "dq_v_g32_s8");
    dqv.set_attr<std::string>(op::attr::qtype, "per_group");
    dqv.set_attr<dims>(op::attr::group_shape, dims{1, 1, 1, 32});
    dqv.set_attr<int64_t>(op::attr::axis, 3);
    dqv.add_inputs({V, VS}); dqv.add_outputs({VF});
    op pv(id++, op::kind::MatMul, "pv_matmul"); pv.add_inputs({probs, VF}); pv.add_outputs({out});

    dnnl::graph::graph g(engine::kind::gpu); g.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    g.add_op(dqk); g.add_op(qk); g.add_op(div); g.add_op(add); g.add_op(sm); g.add_op(dqv); g.add_op(pv); g.finalize();
    return {std::move(g), {Q.get_id(), K.get_id(), KS.get_id(), divisor.get_id(), mask.get_id(), V.get_id(), VS.get_id(), out.get_id()}};
}

static uint16_t float_to_half(float x) {
    const sycl::half h = static_cast<sycl::half>(x);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

struct allocation { sycl::queue *q = nullptr; void *p = nullptr; allocation() = default; allocation(sycl::queue &qq, size_t n) : q(&qq), p(sycl::malloc_device(n, qq)) { if (!p) throw std::bad_alloc(); } allocation(const allocation &) = delete; allocation &operator=(const allocation &) = delete; allocation(allocation &&x) noexcept : q(x.q), p(x.p) { x.p = nullptr; } ~allocation() { if (p) sycl::free(p, *q); } };

#if defined(_WIN32)
struct imported_allocation {
    ze_context_handle_t context = nullptr;
    void *ptr = nullptr;
    void *handle = nullptr;
    size_t size = 0;
    imported_allocation() = default;
    imported_allocation(ze_context_handle_t c, ze_device_handle_t d, void *h, size_t n) : context(c), handle(h), size(n) {
        ze_external_memory_import_win32_handle_t ext{};
        ext.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32;
        ext.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32;
        ext.handle = h;
        ze_device_mem_alloc_desc_t desc{};
        desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
        desc.pNext = &ext;
        const ze_result_t result = zeMemAllocDevice(context, &desc, n, 0, d, &ptr);
        if (result != ZE_RESULT_SUCCESS || !ptr) throw std::runtime_error("Level Zero Vulkan handle import failed");
    }
    imported_allocation(const imported_allocation &) = delete;
    imported_allocation & operator=(const imported_allocation &) = delete;
    imported_allocation(imported_allocation && other) noexcept : context(other.context), ptr(other.ptr), handle(other.handle), size(other.size) { other.ptr = nullptr; }
    ~imported_allocation() { if (ptr) zeMemFree(context, ptr); }
};
#endif

struct runtime {
    sycl::device device;
    sycl::context context;
    sycl::queue queue;
    engine engine_obj;
    stream stream_obj;
    std::mutex mutex;
    std::unordered_map<graph_key, std::shared_ptr<cached_graph>, graph_key_hash> graphs;
#if defined(_WIN32)
    std::unordered_map<uint64_t, std::shared_ptr<imported_allocation>> imported_allocations;
#endif

    runtime() : device(pick_device()), context(device), queue(context, device, sycl::property::queue::in_order{}),
        engine_obj(dnnl::sycl_interop::make_engine(device, context)), stream_obj(dnnl::sycl_interop::make_stream(engine_obj, queue)) {}
};

static runtime & get_runtime() {
    static runtime value;
    return value;
}

extern "C" GGML_VULKAN_ONEDNN_API int ggml_vulkan_onednn_sdpa(int q, int kv, const uint16_t *query, const int8_t *key, const uint16_t *key_scale, const int8_t *value, const uint16_t *value_scale, const uint16_t *mask, float divisor, float *output) {
    try {
        if (q <= 0 || kv <= 0 || !query || !key || !key_scale || !value || !value_scale || !mask || !output) throw std::invalid_argument("invalid SDPA arguments");
        runtime & rt = get_runtime();
        std::shared_ptr<cached_graph> cached;
        {
            std::lock_guard<std::mutex> lock(rt.mutex);
            const graph_key key{q, kv, 16, 4, 256};
            auto it = rt.graphs.find(key);
            if (it == rt.graphs.end()) {
                auto item = std::make_shared<cached_graph>();
                item->graph = std::make_shared<graph_t>(make_graph(q, kv));
                auto parts = item->graph->g.get_partitions();
                if (parts.size() != 1 || !parts[0].is_supported()) throw std::runtime_error("oneDNN graph partition unsupported");
                item->inputs = parts[0].get_input_ports();
                item->outputs = parts[0].get_output_ports();
                item->partition = std::make_shared<dnnl::graph::compiled_partition>(parts[0].compile(item->inputs, item->outputs, rt.engine_obj));
                for (auto & x : item->outputs) x = item->partition->query_logical_tensor(x.get_id());
                rt.graphs.emplace(key, item);
                std::cerr << "ggml-vulkan-onednn: compiled shape q=" << q << " kv=" << kv << " h=16 hkv=4 d=256\n";
                cached = std::move(item);
            } else {
                cached = it->second;
                std::cerr << "ggml-vulkan-onednn: reused shape q=" << q << " kv=" << kv << " h=16 hkv=4 d=256\n";
            }
        }
        auto & inputs = cached->inputs;
        auto & outputs = cached->outputs;
        const auto & ids = cached->graph->ids;
        std::vector<allocation> mem; std::vector<tensor> ins, outs; std::vector<void *> inptr, outptr;
        mem.reserve(inputs.size() + outputs.size());
        ins.reserve(inputs.size());
        outs.reserve(outputs.size());
        inptr.reserve(inputs.size());
        outptr.reserve(outputs.size());
        for (const auto & x : inputs) { mem.emplace_back(rt.queue, x.get_mem_size()); inptr.push_back(mem.back().p); ins.emplace_back(x, rt.engine_obj, mem.back().p); }
        for (const auto & x : outputs) { mem.emplace_back(rt.queue, x.get_mem_size()); outptr.push_back(mem.back().p); outs.emplace_back(x, rt.engine_obj, mem.back().p); }
        // TODO: replace staging with Vulkan/L0 shared memory.
        auto copy = [&](size_t id, const void *src, size_t bytes) { for (size_t i = 0; i < inputs.size(); ++i) if (inputs[i].get_id() == id) { rt.queue.memcpy(inptr[i], src, bytes).wait_and_throw(); return; } throw std::runtime_error("graph input not found"); };
        const auto t_h2d = std::chrono::steady_clock::now();
        copy(ids.q, query, size_t(16) * q * 256 * 2); copy(ids.k, key, size_t(4) * 256 * kv); copy(ids.ks, key_scale, size_t(4) * 8 * kv * 2); copy(ids.v, value, size_t(4) * kv * 256); copy(ids.vs, value_scale, size_t(4) * kv * 8 * 2); copy(ids.mask, mask, size_t(q) * kv * 2);
        const uint16_t divisor_bits = float_to_half(divisor); copy(ids.divisor, &divisor_bits, sizeof(divisor_bits));
        const double h2d_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_h2d).count();
        rt.queue.wait_and_throw(); std::cerr << "ggml-vulkan-onednn: dispatch A750 q=" << q << " kv=" << kv << "\n";
        const auto t_exec = std::chrono::steady_clock::now();
        dnnl::graph::sycl_interop::execute(*cached->partition, rt.stream_obj, ins, outs); rt.stream_obj.wait();
        const double exec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_exec).count();
        const auto t_d2h = std::chrono::steady_clock::now();
        std::vector<uint16_t> host(size_t(16) * q * 256); bool found_output = false; for (size_t i = 0; i < outputs.size(); ++i) if (outputs[i].get_id() == ids.out) { rt.queue.memcpy(host.data(), outptr[i], host.size() * sizeof(uint16_t)); found_output = true; break; }
        if (!found_output) throw std::runtime_error("graph output not found");
        rt.queue.wait_and_throw();
        const double d2h_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_d2h).count();
        std::cerr << "ggml-vulkan-onednn: timing h2d_ms=" << h2d_ms << " onednn_exec_ms=" << exec_ms << " d2h_ms=" << d2h_ms << "\n";
        for (size_t i = 0; i < host.size(); ++i) { sycl::half h; std::memcpy(&h, &host[i], sizeof(h)); output[i] = static_cast<float>(h); if (!std::isfinite(output[i])) throw std::runtime_error("oneDNN output contains NaN/Inf"); }
        return 1;
    } catch (const std::exception &e) { std::cerr << "ggml-vulkan-onednn: failure: " << e.what() << "\n"; return 0; }
}

extern "C" GGML_VULKAN_ONEDNN_API int ggml_vulkan_onednn_sdpa_win32(int q, int kv, const ggml_vulkan_onednn_win32_allocation * shared, float divisor) {
#if !defined(_WIN32)
    (void) q; (void) kv; (void) shared; (void) divisor;
    return 0;
#else
    try {
        if (q <= 0 || kv <= 0 || !shared || !shared->handle || !shared->allocation_size || !shared->allocation_id) throw std::invalid_argument("invalid Win32 SDPA arguments");
        runtime & rt = get_runtime();
        const auto native_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(rt.device);
        const auto native_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(rt.context);
        if (!native_device || !native_context) throw std::runtime_error("oneDNN queue is not Level Zero");
        std::shared_ptr<cached_graph> cached;
        {
            std::lock_guard<std::mutex> lock(rt.mutex);
            const graph_key key{q, kv, 16, 4, 256};
            auto it = rt.graphs.find(key);
            if (it == rt.graphs.end()) {
                auto item = std::make_shared<cached_graph>();
                item->graph = std::make_shared<graph_t>(make_graph(q, kv));
                auto parts = item->graph->g.get_partitions();
                if (parts.size() != 1 || !parts[0].is_supported()) throw std::runtime_error("oneDNN graph partition unsupported");
                item->inputs = parts[0].get_input_ports(); item->outputs = parts[0].get_output_ports();
                item->partition = std::make_shared<dnnl::graph::compiled_partition>(parts[0].compile(item->inputs, item->outputs, rt.engine_obj));
                for (auto & x : item->outputs) x = item->partition->query_logical_tensor(x.get_id());
                rt.graphs.emplace(key, item); cached = std::move(item);
            } else cached = it->second;
        }
        const auto & ids = cached->graph->ids;
        std::shared_ptr<imported_allocation> imported;
        {
            std::lock_guard<std::mutex> lock(rt.mutex);
            auto it = rt.imported_allocations.find(shared->allocation_id);
            if (it == rt.imported_allocations.end()) {
                imported = std::make_shared<imported_allocation>(native_context, native_device, shared->handle, shared->allocation_size);
                rt.imported_allocations.emplace(shared->allocation_id, imported);
            } else {
                imported = it->second;
                if (imported->handle != shared->handle || imported->size != shared->allocation_size) throw std::invalid_argument("Level Zero allocation identity mismatch");
            }
        }
        std::vector<tensor> ins, outs;
        (void) divisor;
        auto imported_ptr = [&](size_t id, size_t expected_size) -> void * {
            size_t offset = 0;
            size_t plane_size = 0;
            if (id == ids.q) { offset = shared->query_offset; plane_size = shared->query_size; }
            else if (id == ids.k) { offset = shared->key_offset; plane_size = shared->key_size; }
            else if (id == ids.ks) { offset = shared->key_scale_offset; plane_size = shared->key_scale_size; }
            else if (id == ids.mask) { offset = shared->mask_offset; plane_size = shared->mask_size; }
            else if (id == ids.v) { offset = shared->value_offset; plane_size = shared->value_size; }
            else if (id == ids.vs) { offset = shared->value_scale_offset; plane_size = shared->value_scale_size; }
            else if (id == ids.divisor) { offset = shared->divisor_offset; plane_size = shared->divisor_size; }
            else return nullptr;
            if (plane_size < expected_size || offset > shared->allocation_size || plane_size > shared->allocation_size - offset) throw std::invalid_argument("Vulkan plane exceeds allocation");
            return static_cast<uint8_t *>(imported->ptr) + offset;
        };
        for (const auto & x : cached->inputs) {
            void * ptr = imported_ptr(x.get_id(), x.get_mem_size());
            if (!ptr) throw std::invalid_argument("missing Vulkan plane");
            ins.emplace_back(x, rt.engine_obj, ptr);
        }
        const auto out_it = std::find_if(cached->outputs.begin(), cached->outputs.end(), [&](const logical_tensor &x) { return x.get_id() == ids.out; });
        if (out_it == cached->outputs.end()) throw std::invalid_argument("missing graph output");
        if (shared->output_offset > shared->allocation_size || shared->output_size < out_it->get_mem_size() || shared->output_size > shared->allocation_size - shared->output_offset) throw std::invalid_argument("Vulkan output exceeds allocation");
        outs.emplace_back(*out_it, rt.engine_obj, static_cast<uint8_t *>(imported->ptr) + shared->output_offset);
        dnnl::graph::sycl_interop::execute(*cached->partition, rt.stream_obj, ins, outs);
        rt.stream_obj.wait();
        return 1;
    } catch (const std::exception & e) { std::cerr << "ggml-vulkan-onednn: Win32 import failure: " << e.what() << "\n"; return 0; }
#endif
}

extern "C" GGML_VULKAN_ONEDNN_API int ggml_vulkan_onednn_release_win32(uint64_t allocation_id) {
#if !defined(_WIN32)
    (void) allocation_id;
    return 1;
#else
    if (!allocation_id) return 1;
    try {
        runtime & rt = get_runtime();
        std::shared_ptr<imported_allocation> imported;
        {
            std::lock_guard<std::mutex> lock(rt.mutex);
            auto it = rt.imported_allocations.find(allocation_id);
            if (it == rt.imported_allocations.end()) return 1;
            imported = std::move(it->second);
            rt.imported_allocations.erase(it);
        }
        imported.reset();
        std::cerr << "ggml-vulkan-onednn: released allocation_id=" << allocation_id << "\n";
        return 1;
    } catch (const std::exception & e) {
        std::cerr << "ggml-vulkan-onednn: Win32 release failure: " << e.what() << "\n";
        return 0;
    }
#endif
}
