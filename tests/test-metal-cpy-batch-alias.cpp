#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <vector>

// src and dst share one address, so a batch of two copies must keep their order
static constexpr int     n_rounds    = 100;
static constexpr int     n_tensors   = 64;
static constexpr int     graph_size  = 32;
static constexpr int64_t n_data      = 8;
static constexpr size_t  buffer_size = n_data * sizeof(float);
static constexpr float   first_value = 3.0f;

// a 2D view of a 1D f32 tensor, in elements
struct view_desc {
    int64_t ne0;
    int64_t ne1;
    int64_t stride;
    int64_t offset;
};

struct copy_desc {
    view_desc src;
    view_desc dst;
};

struct alias_case {
    const char * name;
    copy_desc    copies[2];
};

static ggml_tensor * make_view(ggml_context * ctx, ggml_tensor * t, const view_desc & v) {
    return ggml_view_2d(ctx, t, v.ne0, v.ne1, v.stride * sizeof(float), v.offset * sizeof(float));
}

static ggml_cgraph * make_graph(ggml_context * ctx, ggml_tensor * src, ggml_tensor * dst, const alias_case & c) {
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
    for (const copy_desc & copy : c.copies) {
        ggml_build_forward_expand(graph, ggml_cpy(ctx, make_view(ctx, src, copy.src), make_view(ctx, dst, copy.dst)));
    }
    // views first, so the two copies are adjacent
    ggml_tensor ** nodes = ggml_graph_nodes(graph);
    std::stable_sort(nodes, nodes + ggml_graph_n_nodes(graph), [](const ggml_tensor * a, const ggml_tensor * b) {
        return (a->op == GGML_OP_VIEW) > (b->op == GGML_OP_VIEW);
    });
    return graph;
}

static bool init_views(ggml_context * ctx) {
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        if (tensor->view_src && ggml_backend_view_init(tensor) != GGML_STATUS_SUCCESS) {
            return false;
        }
    }
    return true;
}

static std::vector<float> read_view(const std::vector<float> & data, const view_desc & v) {
    std::vector<float> values;
    for (int64_t i = 0; i < v.ne0 * v.ne1; ++i) {
        values.push_back(data[v.offset + (i / v.ne0) * v.stride + i % v.ne0]);
    }
    return values;
}

static void write_view(std::vector<float> & data, const view_desc & v, const std::vector<float> & values) {
    for (int64_t i = 0; i < v.ne0 * v.ne1; ++i) {
        data[v.offset + (i / v.ne0) * v.stride + i % v.ne0] = values[i];
    }
}

// the result of running the copies one after the other
static std::vector<float> reference_copies(std::vector<float> data, const alias_case & c) {
    for (const copy_desc & copy : c.copies) {
        write_view(data, copy.dst, read_view(data, copy.src));
    }
    return data;
}

static bool check_case(ggml_backend_t backend, ggml_tensor * data, ggml_cgraph * graph, const alias_case & c) {
    std::vector<float> input(n_data);
    std::iota(input.begin(), input.end(), first_value);
    const std::vector<float> expected = reference_copies(input, c);

    std::vector<float> output(n_data);
    for (int i = 0; i < n_rounds; ++i) {
        ggml_backend_tensor_set(data, input.data(), 0, buffer_size);
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            return false;
        }
        ggml_backend_tensor_get(data, output.data(), 0, buffer_size);
        if (output != expected) {
            std::fprintf(stderr, "%s: round %d differs from the sequential copies\n", c.name, i);
            return false;
        }
    }
    std::printf("%s: OK\n", c.name);
    return true;
}

static bool run_cases(ggml_backend_t backend, ggml_context * ctx, ggml_tensor * src, ggml_tensor * dst,
                      const std::vector<alias_case> & cases) {
    std::vector<ggml_cgraph *> graphs;
    for (const alias_case & c : cases) {
        graphs.push_back(make_graph(ctx, src, dst, c));
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend, buffer_size);
    void * base = ggml_backend_buffer_get_base(buffer);
    bool ok = ggml_backend_tensor_alloc(buffer, src, base) == GGML_STATUS_SUCCESS &&
              ggml_backend_tensor_alloc(buffer, dst, base) == GGML_STATUS_SUCCESS &&
              init_views(ctx);
    for (size_t i = 0; ok && i < cases.size(); ++i) {
        ok = check_case(backend, src, graphs[i], cases[i]);
    }

    ggml_backend_buffer_free(buffer);
    return ok;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name("MTL0", nullptr);
    if (!backend) {
        std::fprintf(stderr, "Metal backend unavailable\n");
        return 1;
    }

    // {ne0, ne1, stride, offset} of the source and destination views of each copy
    const std::vector<alias_case> cases = {
        { "forward", { { { 1, 1, 1, 0 }, { 1, 1, 1, 1 } }, { { 1, 1, 1, 1 }, { 1, 1, 1, 2 } } } },
        { "reverse", { { { 1, 1, 1, 1 }, { 1, 1, 1, 0 } }, { { 1, 1, 1, 2 }, { 1, 1, 1, 1 } } } },
        // the second copy writes element 4, which the strided source of the first copy reads past its destination size
        { "strided", { { { 1, 2, 4, 0 }, { 1, 2, 1, 6 } }, { { 1, 2, 4, 1 }, { 1, 2, 1, 3 } } } },
    };

    const size_t mem_size = n_tensors * ggml_tensor_overhead() + cases.size() * ggml_graph_overhead_custom(graph_size, false);
    ggml_init_params params = { mem_size, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_data);
    ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_data);

    const bool ok = run_cases(backend, ctx, src, dst, cases);

    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
