// kernels of the user in GGML_OP_CUSTOM nodes of the CUDA backend (see ggml_cuda_kernel in ggml-cuda.h):
// arguments, parameters, grid, dynamic shared memory, pinned host memory, side streams with a join, next to regular nodes

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cuda.h"

#include "test-cuda-kernel-op.h"
#include "test-cuda-kernel-op.ptx.hpp" // test-cuda-kernel-op.cu compiled by nvcc

#include <cstdio>
#include <string>
#include <vector>

static constexpr int n      = 4*TEST_BLOCK_SIZE;
static constexpr int n_iter = 6; // the CUDA graph of a graph is captured on its second use, then replayed

// the side stream kernels write 20 ms late, and their input is ready 40 ms late:
// a side stream that does not wait for the nodes before it, or a join that does not wait for the side streams, reads old data
static constexpr uint64_t delay_ns = 20*1000*1000;

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t dev = nullptr;
    ggml_custom_op_t kernel_op = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count() && dev == nullptr; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        kernel_op = (ggml_custom_op_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(d), "ggml_backend_cuda_kernel_op");
        if (kernel_op != nullptr) {
            dev = d;
        }
    }
    if (dev == nullptr) {
        printf("no device with kernel ops, skipping\n");
        return 0;
    }
    printf("device: %s\n", ggml_backend_dev_description(dev));

    ggml_backend_ptr cuda(ggml_backend_dev_init(dev, nullptr));
    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!cuda || !cpu) {
        printf("failed to init the backends\n");
        return 1;
    }

    // the kernels read pinned host memory at its host address
    ggml_backend_buffer_type_t buft_host = ggml_backend_dev_host_buffer_type(dev);
    ggml_backend_buffer_ptr buf_host(buft_host ? ggml_backend_buft_alloc_buffer(buft_host, n*sizeof(float)) : nullptr);
    if (!buf_host || ggml_backend_buffer_get_type(buf_host.get()) != buft_host) {
        printf("no pinned host memory, skipping\n");
        return 0;
    }
    float * host = (float *) ggml_backend_buffer_get_base(buf_host.get());
    for (int i = 0; i < n; ++i) {
        host[i] = 0.25f*(i % 5);
    }

    const std::string image((const char *) test_cuda_kernel_op_ptx, test_cuda_kernel_op_ptx_len);

    auto params = [&](float scale, float add, uint64_t delay) {
        test_kernel_params p = {};
        p.host     = host;
        p.scale    = scale;
        p.add      = add;
        p.n        = n;
        p.delay_ns = delay;
        return p;
    };
    auto kernel = [&](const char * name, const test_kernel_params * p, unsigned int smem, int stream, bool join) {
        ggml_cuda_kernel k = {};
        k.image    = image.c_str();
        k.name     = name;
        k.grid[0]  = n/TEST_BLOCK_SIZE;
        k.grid[1]  = 1;
        k.grid[2]  = 1;
        k.block[0] = TEST_BLOCK_SIZE;
        k.block[1] = 1;
        k.block[2] = 1;
        k.smem     = smem;
        k.params   = p;
        k.stream   = stream;
        k.join     = join;
        return k;
    };

    // the kernels and their parameters outlive the graph
    const test_kernel_params p_a  = params(2.0f, 1.0f, 2*delay_ns);
    const test_kernel_params p_s1 = params(0.5f, 2.0f, delay_ns);
    const test_kernel_params p_s2 = params(0.0f, 0.0f, delay_ns);
    const test_kernel_params p_0  = params(0.0f, 0.0f, 0);
    ggml_cuda_kernel k_a  = kernel("test_affine",   &p_a,  0, 0, false);
    ggml_cuda_kernel k_s1 = kernel("test_affine",   &p_s1, 0, 1, false);
    ggml_cuda_kernel k_s2 = kernel("test_add_host", &p_s2, 0, 2, false);
    ggml_cuda_kernel k_r  = kernel("test_reverse",  &p_0,  TEST_BLOCK_SIZE*sizeof(float), 0, false);
    ggml_cuda_kernel k_j  = kernel("test_sum3",     &p_0,  0, 0, true);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 64*ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr(ggml_init(iparams));
    ggml_context * ctx = ctx_ptr.get();

    auto node = [&](ggml_cuda_kernel & k, std::vector<ggml_tensor *> args) {
        return ggml_custom_4d(ctx, GGML_TYPE_F32, n, 1, 1, 1, args.data(), (int) args.size(), kernel_op, 1, &k);
    };

    // a kernel that is not in the module and a module that does not load are not supported
    {
        ggml_cuda_kernel k_missing = kernel("test_no_such_kernel", &p_0, 0, 0, false);
        ggml_cuda_kernel k_bad     = kernel("test_affine", &p_0, 0, 0, false);
        k_bad.image = "not a module";
        if (ggml_backend_dev_supports_op(dev, node(k_missing, {})) || ggml_backend_dev_supports_op(dev, node(k_bad, {}))) {
            printf("FAIL: a kernel that cannot be loaded is supported\n");
            return 1;
        }
    }

    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_input(x);

    ggml_tensor * a   = node(k_a, { x });             // a  = 2x + 1
    ggml_tensor * t   = ggml_scale(ctx, a, 3.0f);     // only the side stream reads it: stays allocated until the join
    ggml_tensor * s1  = node(k_s1, { t });            // s1 = t/2 + 2      (side stream 1)
    ggml_tensor * s2  = node(k_s2, { a });            // s2 = a + host     (side stream 2)
    ggml_tensor * u   = ggml_sqr(ctx, a);             // regular nodes and a kernel next to the side streams
    ggml_tensor * v   = ggml_add(ctx, u, a);
    ggml_tensor * r   = node(k_r, { v });             // r  = v, each block reversed
    ggml_tensor * j   = node(k_j, { r, s1, s2 });     // j  = r + s1 + s2  (join)
    ggml_tensor * out = ggml_add(ctx, j, x);
    ggml_set_output(out);

    // the side streams start before the nodes they run next to
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, s1);
    ggml_build_forward_expand(gf, s2);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backends[] = { cuda.get(), cpu.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true));

    for (ggml_tensor * k : { a, s1, s2, r, j }) {
        ggml_backend_sched_set_tensor_backend(sched.get(), k, cuda.get());
    }
    if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
        printf("FAIL: cannot allocate the graph\n");
        return 1;
    }

    std::vector<float> x_host(n);
    std::vector<float> out_host(n);
    std::vector<float> v_host(n);
    for (int it = 0; it < n_iter; ++it) {
        for (int i = 0; i < n; ++i) {
            x_host[i] = (float) ((i + 3*it + 1) % 17);
        }
        ggml_backend_tensor_set(x, x_host.data(), 0, ggml_nbytes(x));
        if (ggml_backend_sched_graph_compute(sched.get(), gf) != GGML_STATUS_SUCCESS) {
            printf("FAIL: cannot compute the graph\n");
            return 1;
        }
        ggml_backend_tensor_get(out, out_host.data(), 0, ggml_nbytes(out));

        // all the values are exact in float
        for (int i = 0; i < n; ++i) {
            const float av = 2.0f*x_host[i] + 1.0f;
            v_host[i] = av*av + av;
        }
        for (int i = 0; i < n; ++i) {
            const float av = 2.0f*x_host[i] + 1.0f;
            const float s1v = 3.0f*av*0.5f + 2.0f;
            const float s2v = av + host[i];
            const float rv  = v_host[i - i % TEST_BLOCK_SIZE + TEST_BLOCK_SIZE - 1 - i % TEST_BLOCK_SIZE];
            const float want = rv + s1v + s2v + x_host[i];
            if (out_host[i] != want) {
                printf("FAIL: iteration %d, element %d: %f, expected %f\n", it, i, out_host[i], want);
                return 1;
            }
        }
    }

    printf("OK: %d iterations\n", n_iter);
    return 0;
}
