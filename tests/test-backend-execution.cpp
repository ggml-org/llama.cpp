#include "../ggml/src/ggml-backend-execution.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "testing.h"
#include <cstring>
#include <stdexcept>

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
    return t.summary();
}
