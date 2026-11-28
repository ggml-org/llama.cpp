// A custom operator: saxpy, dst = a*x + y
//
// The op is a list of kernels, one for each backend that can run it (see ggml_custom_op_init in ggml.h).
// Each kernel also tells how it runs a node: its threads or its workgroups.
// A backend without a kernel leaves the node to the CPU backend.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

// the params of a node, every kernel gets a copy
struct saxpy_params {
    float   a;
    int32_t n;
};

// CPU: thread ith of nth computes its part
static void saxpy_cpu(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    const saxpy_params * p = (const saxpy_params *) userdata;

    const float * x = (const float *) dst->src[0]->data;
    const float * y = (const float *) dst->src[1]->data;
    float       * d = (float *) dst->data;

    for (int i = p->n*ith/nth; i < p->n*(ith + 1)/nth; i++) {
        d[i] = p->a*x[i] + y[i];
    }
}

// CPU: on all the threads
static void saxpy_cpu_sched(const struct ggml_tensor * node, const void * params, struct ggml_custom_sched * sched) {
    (void) node;
    (void) params;
    ggml_custom_sched_set_n_tasks(sched, GGML_N_TASKS_MAX);
}

static const int n = 8;

static const float data_x[n] = { 1, 2, 3, 4, 5, 6, 7, 8 };
static const float data_y[n] = { 10, 20, 30, 40, 50, 60, 70, 80 };

// computes 2*x + y with the backend and the CPU backend, the scheduler gives the node to the first one that can run it
static std::vector<float> compute(ggml_backend_t backend, const struct ggml_custom_op * op) {
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    ggml_backend_t backends[2] = { backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true, // the tensors will be allocated later
    };
    struct ggml_context * ctx = ggml_init(params0);
    struct ggml_cgraph  * gf  = ggml_new_graph(ctx);

    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    // the node: its arguments and its params
    struct ggml_tensor * args[2] = { x, y };
    const saxpy_params p = { 2.0f, n };
    struct ggml_tensor * result = ggml_custom_op_4d(ctx, GGML_TYPE_F32, n, 1, 1, 1, args, 2, op, &p, sizeof(p));

    ggml_build_forward_expand(gf, result);

    ggml_backend_sched_alloc_graph(sched, gf);

    ggml_backend_tensor_set(x, data_x, 0, ggml_nbytes(x));
    ggml_backend_tensor_set(y, data_y, 0, ggml_nbytes(y));

    ggml_backend_sched_graph_compute(sched, gf);

    printf("%s computed by %s:\n[", ggml_custom_op_name(op), ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched, result)));

    std::vector<float> out(n);
    ggml_backend_tensor_get(result, out.data(), 0, ggml_nbytes(result));
    for (float v : out) {
        printf(" %.2f", v);
    }
    printf(" ]\n");

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);
    ggml_free(ctx);

    return out;
}

int main(void) {
    ggml_backend_load_all();

    // the op: its kernels, by the name of the backend (see ggml_backend_reg_name)
    struct ggml_custom_op * op = ggml_custom_op_init("saxpy");

    struct ggml_custom_kernel * cpu = ggml_custom_op_add_kernel(op, "CPU");
    ggml_custom_kernel_set_fun  (cpu, saxpy_cpu);
    ggml_custom_kernel_set_sched(cpu, saxpy_cpu_sched);

    // expected result:
    // [ 12.00 24.00 36.00 48.00 60.00 72.00 84.00 96.00 ]

    // the op has no kernel for the best backend: the CPU computes the node
    ggml_backend_t backend = ggml_backend_init_best();
    const std::vector<float> out = compute(backend, op);
    ggml_backend_free(backend);

    ggml_custom_op_free(op);

    for (int i = 0; i < n; i++) {
        if (out[i] != 2.0f*data_x[i] + data_y[i]) {
            printf("wrong result\n");
            return 1;
        }
    }
    return 0;
}
