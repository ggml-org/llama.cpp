#include "../ggml/src/ggml-backend-execution.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "testing.h"
#include <cstring>
#include <stdexcept>
#include <vector>
#include <climits>

struct read_probe {
    inline static decltype(ggml_backend_buffer_i::get_tensor) original = nullptr;
    inline static int calls = 0;
    static void read(ggml_backend_buffer_t b,const ggml_tensor * t,void * p,size_t offset,size_t bytes) {
        if (++calls > 1) throw std::runtime_error("recursive proxy byte routing");
        original(b,t,p,offset,bytes);
    }
};

struct execution_probe { int writes = 0, computes = 0, frees = 0; };
static const ggml_backend_execution_ops ops{
    [](void *,const ggml_tensor * t) { return t->op == GGML_OP_SCALE; },
    [](void * p,ggml_backend_t,ggml_tensor *) { ++static_cast<execution_probe *>(p)->computes; return GGML_STATUS_SUCCESS; },
    [](void *) { return true; },
    [](void * p) { ++static_cast<execution_probe *>(p)->writes; },
    [](void * p) { ++static_cast<execution_probe *>(p)->frees; }
};
int main(int argc, char ** argv) {
    testing t;
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    t.test("compact_mask_preserves_bits_and_rejects_noninterval_rows", [&](testing & t) {
        for (size_t tokens : {size_t(1), size_t(255), size_t(256), size_t(257), size_t(1024)}) {
            for (size_t begin : {size_t(0), tokens/2, tokens}) {
                for (size_t end : {begin, tokens}) {
                    std::vector<ggml_fp16_t> dense(tokens, 0xfc00);
                    for (size_t i=begin; i<end; ++i) dense[i]=0;
                    ggml_backend_execution_mask_interval interval{7,9,11,13};
                    if (!t.assert_true(ggml_backend_execution_mask_interval_encode(dense.data(), tokens, interval))) return;
                    t.assert_equal(int32_t(tokens), interval.tokens);
                    t.assert_equal(int32_t(1), interval.version);
                    for (size_t i=0; i<tokens; ++i)
                        if (!t.assert_equal(dense[i], ggml_backend_execution_mask_interval_value(interval, int32_t(i)))) return;
                }
            }
        }
        for (auto dense : {std::vector<ggml_fp16_t>{0,0xfc00,0}, std::vector<ggml_fp16_t>{0x8000},
                std::vector<ggml_fp16_t>{0x3c00}, std::vector<ggml_fp16_t>{0x7e00}, std::vector<ggml_fp16_t>{0x7c00}}) {
            ggml_backend_execution_mask_interval interval{7,9,11,13}, previous=interval;
            t.assert_true(!ggml_backend_execution_mask_interval_encode(dense.data(),dense.size(),interval));
            t.assert_true(!std::memcmp(&interval,&previous,sizeof(interval)));
        }
        ggml_backend_execution_mask_interval interval{7,9,11,13}, previous=interval;
        const ggml_fp16_t value=0;
        t.assert_true(!ggml_backend_execution_mask_interval_encode(nullptr,1,interval));
        t.assert_true(!ggml_backend_execution_mask_interval_encode(&value,0,interval));
        t.assert_true(!ggml_backend_execution_mask_interval_encode(&value,size_t(INT32_MAX)+1,interval));
        t.assert_true(!std::memcmp(&interval,&previous,sizeof(interval)));
    });
    t.test("compact_mask_is_typed_bounded_and_only_attached_to_private_attention", [&](testing & t) {
        ggml_context_ptr ctx(ggml_init({16384,nullptr,true}));
        auto * q=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F32,256,129,12,1);
        auto * k=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_Q8_0,256,32768,2,1);
        auto * v=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_Q4_0,256,32768,2,1);
        auto * bounds=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,4,129);
        auto * op=ggml_flash_attn_ext(ctx.get(),q,k,v,nullptr,1.f/16,0,0);
        int32_t arithmetic[5]; std::memcpy(arithmetic,op->op_params,sizeof(arithmetic));
        t.assert_true(ggml_backend_execution_set_mask_bounds(op,bounds));
        t.assert_true(op->src[3] == nullptr && op->src[5] == bounds);
        ggml_backend_buffer_t owner=nullptr;
        t.assert_true(!ggml_backend_execution_owner(op,owner) && !owner);
        t.assert_true(!ggml_backend_supports_op(backend.get(),op));
        t.assert_equal(size_t(129*16),ggml_nbytes(bounds));
        t.assert_true(!std::memcmp(arithmetic,op->op_params,sizeof(arithmetic)));
        ggml_tensor logical{}, mask{};
        t.assert_true(ggml_backend_execution_mask_metadata(op,logical,mask));
        t.assert_true(logical.src[3] == &mask && logical.src[5] == bounds && mask.type == GGML_TYPE_F16);
        t.assert_true(mask.ne[0] == k->ne[1] && mask.ne[1] == q->ne[1]);
        t.assert_true(mask.data == nullptr && mask.buffer == nullptr);
        t.assert_true(op->src[3] == nullptr && op->src[5] == bounds);
        const auto saved=*bounds;
        bounds->type=GGML_TYPE_F16;
        t.assert_true(!ggml_backend_execution_mask_metadata(op,logical,mask));
        *bounds=saved; bounds->ne[0]=3;
        t.assert_true(!ggml_backend_execution_mask_metadata(op,logical,mask));
        *bounds=saved; bounds->ne[1]=128;
        t.assert_true(!ggml_backend_execution_mask_metadata(op,logical,mask));
        *bounds=saved; bounds->nb[1]+=4;
        t.assert_true(!ggml_backend_execution_mask_metadata(op,logical,mask));
        *bounds=saved; op->src[3]=&mask;
        t.assert_true(!ggml_backend_execution_mask_metadata(op,logical,mask));
        t.assert_true(!ggml_backend_execution_set_mask_bounds(op,bounds));
        op->src[3]=nullptr; op->src[4]=q;
        t.assert_true(!ggml_backend_execution_mask_metadata(op,logical,mask));
        t.assert_true(!ggml_backend_execution_set_mask_bounds(op,bounds));
        t.assert_true(!ggml_backend_execution_set_mask_bounds(q,bounds));
        execution_probe probe;
        ggml_backend_buffer_ptr backing(ggml_backend_alloc_buffer(backend.get(),1024));
        ggml_backend_buffer_ptr managed(ggml_backend_execution_buffer_new(ggml_backend_get_device(backend.get()),backing.get(),ops,&probe));
        if (!t.assert_true(bool(managed))) return;
        op->src[4]=nullptr;
        t.assert_true(!ggml_backend_execution_owner(op,owner) && !owner);
        k->buffer=managed.get();
        t.assert_true(ggml_backend_execution_owner(op,owner) && owner == managed.get());
        t.assert_true(!ggml_backend_execution_supports(owner,ggml_backend_get_device(backend.get()),op));
    });
    t.test("attention_workspace_hint_preserves_arithmetic_parameters", [&](testing & t) {
        ggml_context_ptr ctx(ggml_init({8192, nullptr, true}));
        auto * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 32, 4, 2, 1);
        auto * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, 32, 256, 2, 1);
        auto * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, 32, 256, 2, 1);
        auto * attention = ggml_flash_attn_ext(ctx.get(), q, k, v, nullptr, .125f, 0, 0);
        ggml_prec_set_acc(attention, GGML_PREC_F32);
        ggml_flash_attn_ext_set_n_kv_max(attention, 128);
        int32_t arithmetic[5];
        std::memcpy(arithmetic, attention->op_params, sizeof(arithmetic));
        t.assert_true(!ggml_backend_execution_has_external_workspace(attention));
        t.assert_true(!ggml_backend_execution_has_external_workspace(q));
        t.assert_true(!ggml_backend_execution_has_external_workspace(nullptr));
        ggml_backend_execution_set_external_workspace(attention, true);
        t.assert_true(ggml_backend_execution_has_external_workspace(attention));
        t.assert_true(std::memcmp(arithmetic, attention->op_params, sizeof(arithmetic)) == 0);
        ggml_backend_execution_set_external_workspace(attention, false);
        t.assert_true(!ggml_backend_execution_has_external_workspace(attention));
        t.assert_true(std::memcmp(arithmetic, attention->op_params, sizeof(arithmetic)) == 0);
        ggml_flash_attn_ext_set_n_kv_max(attention, 1);
        t.assert_true(!ggml_backend_execution_has_external_workspace(attention));
    });
    t.test("execution_buffer_retains_backing_and_dispatches_only_its_owner", [&](testing & t) {
        execution_probe probe;
        ggml_backend_buffer_ptr backing(ggml_backend_alloc_buffer(backend.get(),1024));
        ggml_backend_buffer_ptr buffer(ggml_backend_execution_buffer_new(ggml_backend_get_device(backend.get()),backing.get(),ops,&probe));
        if (!t.assert_true(bool(buffer))) return;
        backing.reset();
        t.assert_true(ggml_backend_buft_is_execution(ggml_backend_buffer_get_type(buffer.get())));
        t.assert_true(!ggml_backend_buffer_is_host(buffer.get()));
        ggml_context_ptr ctx(ggml_init({4096,nullptr,true}));
        auto * a = ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        t.assert_true(ggml_backend_tensor_alloc(buffer.get(),a,ggml_backend_buffer_get_base(buffer.get())) == GGML_STATUS_SUCCESS);
        const float data[4] = {1,2,3,4}; float copied[4] = {};
        ggml_backend_tensor_set(a,data,0,sizeof(data)); ggml_backend_tensor_get(a,copied,0,sizeof(copied));
        t.assert_true(!std::memcmp(data,copied,sizeof(data)) && probe.writes == 1);
        auto * operation = ggml_scale(ctx.get(),a,2);
        ggml_backend_buffer_t owner = nullptr;
        t.assert_true(ggml_backend_execution_owner(operation,owner) && owner == buffer.get());
        t.assert_true(ggml_backend_execution_supports(owner,ggml_backend_get_device(backend.get()),operation));
        t.assert_true(ggml_backend_execution_compute(owner,backend.get(),operation) == GGML_STATUS_SUCCESS && probe.computes == 1);
        operation->op = GGML_OP_ADD;
        t.assert_true(!ggml_backend_execution_supports(owner,ggml_backend_get_device(backend.get()),operation));
        t.assert_true(ggml_backend_execution_compute(owner,backend.get(),operation) == GGML_STATUS_FAILED);
        auto * view = ggml_view_1d(ctx.get(),a,2,sizeof(float));
        t.assert_true(ggml_backend_view_init(view) == GGML_STATUS_SUCCESS);
        read_probe::original = buffer->iface.get_tensor; read_probe::calls = 0;
        buffer->iface.get_tensor = read_probe::read;
        bool caught = false; float values[2] = {};
        try { ggml_backend_tensor_get(view,values,0,sizeof(values)); }
        catch (const std::runtime_error &) { caught = true; }
        buffer->iface.get_tensor = read_probe::original;
        t.assert_true(!caught && read_probe::calls == 1 && values[0] == 2 && values[1] == 3);
        const float replacement[2] = {5,6};
        ggml_backend_tensor_set(view,replacement,0,sizeof(replacement));
        ggml_backend_tensor_get(a,copied,0,sizeof(copied));
        t.assert_true(copied[0] == 1 && copied[1] == 5 && copied[2] == 6 && copied[3] == 4);
        ggml_backend_tensor_memset(view,0,0,sizeof(float));
        ggml_backend_tensor_get(view,values,0,sizeof(values));
        t.assert_true(values[0] == 0 && values[1] == 6 && probe.writes == 3);
        buffer.reset(); t.assert_equal(1,probe.frees);
    });
    t.test("stateful_host_storage_is_not_silently_wrapped", [&](testing & t) {
        execution_probe probe;
        ggml_backend_buffer_ptr backing(ggml_backend_alloc_buffer(backend.get(),1024));
        const auto reset = backing->iface.reset;
        backing->iface.reset = [](ggml_backend_buffer_t) {};
        ggml_backend_buffer_ptr buffer(ggml_backend_execution_buffer_new(ggml_backend_get_device(backend.get()),backing.get(),ops,&probe));
        t.assert_true(!buffer); buffer.reset();
        t.assert_equal(0,probe.frees);
        backing->iface.reset = reset;
    });
    t.test("cpu_graph_dispatches_publication_marker_between_ordinary_nodes", [&](testing & t) {
        execution_probe probe;
        auto marker_ops=ops;
        marker_ops.supports=[](void *,const ggml_tensor * op) { return op->op == GGML_OP_CPY; };
        marker_ops.compute=[](void * p,ggml_backend_t,ggml_tensor * op) -> ggml_status {
            ++static_cast<execution_probe *>(p)->computes;
            float values[4];
            ggml_backend_tensor_get(op->src[0],values,0,sizeof(values));
            ggml_backend_tensor_set(op,values,0,sizeof(values));
            return GGML_STATUS_SUCCESS;
        };
        ggml_backend_buffer_ptr backing(ggml_backend_alloc_buffer(backend.get(),1024));
        ggml_backend_buffer_ptr marker(ggml_backend_execution_buffer_new(
            ggml_backend_get_device(backend.get()),backing.get(),marker_ops,&probe));
        if (!t.assert_true(bool(marker))) return;
        ggml_context_ptr ctx(ggml_init({16384,nullptr,true}));
        auto * input=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * destination=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        ggml_backend_buffer_ptr input_buffer(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),64));
        if (!t.assert_true(bool(input_buffer))) return;
        if (!t.assert_true(ggml_backend_tensor_alloc(input_buffer.get(),input,
                ggml_backend_buffer_get_base(input_buffer.get())) == GGML_STATUS_SUCCESS &&
                ggml_backend_tensor_alloc(marker.get(),destination,
                    ggml_backend_buffer_get_base(marker.get())) == GGML_STATUS_SUCCESS)) return;
        const float values[4]={1,2,3,4};
        ggml_backend_tensor_set(input,values,0,sizeof(values));
        auto * copied=ggml_cpy(ctx.get(),input,destination);
        auto * graph=ggml_new_graph_custom(ctx.get(),16,false);
        ggml_build_forward_expand(graph,copied);
        t.assert_true(ggml_backend_graph_compute(backend.get(),graph) == GGML_STATUS_SUCCESS);
        float result[4]={};
        ggml_backend_tensor_get(destination,result,0,sizeof(result));
        t.assert_true(!std::memcmp(values,result,sizeof(values)));
        t.assert_equal(1,probe.computes);
    });

    if (argc > 1 && !std::strcmp(argv[1],"--cuda")) t.test("cuda_scheduler_dispatches_managed_nodes_between_native_segments", [&](testing & t) {
        ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr cuda(ggml_backend_dev_init(dev,nullptr));
        execution_probe probe;
        auto native_ops = ops;
        native_ops.compute = [](void * p,ggml_backend_t,ggml_tensor * op) -> ggml_status {
            ++static_cast<execution_probe *>(p)->computes;
            float data[4]; ggml_backend_tensor_get(op->src[0],data,0,sizeof(data));
            float scale; std::memcpy(&scale,op->op_params,sizeof(scale));
            for (auto & value : data) value *= scale;
            ggml_backend_tensor_set(op,data,0,sizeof(data)); return GGML_STATUS_SUCCESS;
        };
        ggml_backend_buffer_ptr backing(ggml_backend_alloc_buffer(backend.get(),1024));
        ggml_backend_buffer_ptr buffer(ggml_backend_execution_buffer_new(dev,backing.get(),native_ops,&probe));
        if (!t.assert_true(bool(buffer))) return;
        ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
        auto * input = ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        t.assert_true(ggml_backend_tensor_alloc(buffer.get(),input,ggml_backend_buffer_get_base(buffer.get())) == GGML_STATUS_SUCCESS);
        const float data[4] = {1,2,3,4}; ggml_backend_tensor_set(input,data,0,sizeof(data));
        auto * middle = ggml_scale(ctx.get(),input,2);
        auto * output = ggml_add(ctx.get(),middle,middle);
        auto * graph = ggml_new_graph_custom(ctx.get(),64,false); ggml_build_forward_expand(graph,output);
        ggml_backend_t backends[] = {cuda.get(),backend.get()};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends,nullptr,2,64,false,true));
        if (!t.assert_true(ggml_backend_sched_alloc_graph(sched.get(),graph))) return;
        for (int run = 0; run < 4; ++run) {
            t.assert_true(ggml_backend_sched_graph_compute(sched.get(),graph) == GGML_STATUS_SUCCESS);
            float values[4]; ggml_backend_tensor_get(output,values,0,sizeof(values));
            for (int i = 0; i < 4; ++i) t.assert_equal(data[i]*4,values[i]);
        }
        t.assert_equal(4,probe.computes);
    });
    if (argc > 1 && !std::strcmp(argv[1],"--cuda")) t.test("cuda_managed_attention_respects_native_and_external_workspace_contracts", [&](testing & t) {
        ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        auto * buft = ggml_backend_dev_buffer_type(dev);
        execution_probe probe;
        auto managed_ops = ops;
        managed_ops.supports = [](void *, const ggml_tensor * op) { return op->op == GGML_OP_FLASH_ATTN_EXT; };
        ggml_backend_buffer_ptr backing(ggml_backend_alloc_buffer(backend.get(), 1 << 20));
        ggml_backend_buffer_ptr managed(ggml_backend_execution_buffer_new(dev, backing.get(), managed_ops, &probe));
        if (!t.assert_true(bool(managed))) return;
        ggml_context_ptr ctx(ggml_init({16384, nullptr, true}));
        auto * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 256, 4, 24, 1);
        auto * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_Q8_0, 256, 512, 4, 1);
        auto * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_Q4_0, 256, 512, 4, 1);
        auto * mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, 512, 4, 1, 1);
        auto * attention = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, 1.0f/16, 0, 0);
        auto * prefill_q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 256, 256, 24, 1);
        auto * prefill_mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, 512, 256, 1, 1);
        auto * prefill_attention = ggml_flash_attn_ext(ctx.get(), prefill_q, k, v, prefill_mask, 1.0f/16, 0, 0);
        const size_t stock = ggml_backend_buft_get_alloc_size(buft, attention);
        const size_t stock_prefill = ggml_backend_buft_get_alloc_size(buft, prefill_attention);
        t.assert_true(stock > ggml_nbytes(attention));
        t.assert_true(stock_prefill > ggml_nbytes(prefill_attention));
        auto * base = static_cast<char *>(ggml_backend_buffer_get_base(managed.get()));
        if (!t.assert_true(ggml_backend_tensor_alloc(managed.get(), k, base) == GGML_STATUS_SUCCESS &&
                ggml_backend_tensor_alloc(managed.get(), v, base + ggml_nbytes(k)) == GGML_STATUS_SUCCESS)) return;
        // Short prefill uses native MMA conversion scratch even with a managed KV owner.
        t.assert_equal(stock, ggml_backend_buft_get_alloc_size(buft, attention));
        t.assert_equal(stock_prefill, ggml_backend_buft_get_alloc_size(buft, prefill_attention));
        ggml_prec_set_acc(attention, GGML_PREC_F32);
        ggml_backend_execution_set_external_workspace(attention, true);
        t.assert_true(ggml_backend_execution_has_external_workspace(attention));
        t.assert_equal(GGML_PREC_F32, ggml_flash_attn_ext_get_prec(attention));
        t.assert_equal(ggml_nbytes(attention), ggml_backend_buft_get_alloc_size(buft, attention));
        ggml_backend_execution_set_external_workspace(attention, false);
        t.assert_equal(stock, ggml_backend_buft_get_alloc_size(buft, attention));
        for (int64_t queries : {1, 2, 3}) {
            auto * short_q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 256, queries, 24, 1);
            auto * short_mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, 512, queries, 1, 1);
            auto * short_attention = ggml_flash_attn_ext(ctx.get(), short_q, k, v, short_mask, 1.0f/16, 0, 0);
            k->buffer = v->buffer = nullptr;
            const size_t native_size = ggml_backend_buft_get_alloc_size(buft, short_attention);
            k->buffer = v->buffer = managed.get();
            t.assert_equal(native_size, ggml_backend_buft_get_alloc_size(buft, short_attention));
            ggml_backend_execution_set_external_workspace(short_attention, true);
            t.assert_equal(ggml_nbytes(short_attention), ggml_backend_buft_get_alloc_size(buft, short_attention));
        }
        ggml_backend_execution_set_external_workspace(prefill_attention, true);
        t.assert_equal(stock_prefill, ggml_backend_buft_get_alloc_size(buft, prefill_attention));
        ggml_backend_execution_set_verified_workspace(prefill_attention);
        t.assert_true(ggml_backend_execution_has_verified_workspace(prefill_attention));
        t.assert_equal(ggml_nbytes(prefill_attention),ggml_backend_buft_get_alloc_size(buft,prefill_attention));
        managed_ops.supports = [](void *, const ggml_tensor *) { return false; };
        ggml_backend_buffer_ptr unsupported(ggml_backend_execution_buffer_new(dev, backing.get(), managed_ops, &probe));
        if (!t.assert_true(bool(unsupported))) return;
        k->buffer = v->buffer = unsupported.get();
        t.assert_equal(stock, ggml_backend_buft_get_alloc_size(buft, attention));
        k->buffer = v->buffer = nullptr;
        unsupported.reset(); managed.reset();
        t.assert_equal(2, probe.frees);
    });
    return t.summary();
}
