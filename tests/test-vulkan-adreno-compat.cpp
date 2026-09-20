// Guard boundaries and real backend tensor checks for the bounded opt-in rule.
#include "../ggml/src/ggml-vulkan/ggml-vulkan-adreno-compat.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
struct selection {
    unsigned vendor = 0x5143, driver = 8, version = 2150604839u;
    const char * name = "Adreno (TM) 750";
    bool shader = true;
    unsigned block = 64, rows = 1, cols = 5, subgroup = 64;
    bool full = true;
    unsigned memory = 32768;
    bool matches() const {
        return ggml_vk_adreno_750_matvec_shmem(vendor, driver, version, name, shader,
                block, rows, cols, subgroup, full, memory);
    }
};
static void guards() {
    selection s;
    require(s.matches(), "observed tuple rejected");
    for (unsigned cols = 0; cols <= 17; ++cols) {
        s = selection{}; s.cols = cols;
        require(s.matches() == (cols == 5), "column boundary widened");
    }
    for (unsigned block : {0u, 32u, 63u, 128u}) {
        s = selection{}; s.block = block;
        require(!s.matches(), "block boundary widened");
    }
    for (unsigned rows : {0u, 2u, 4u}) {
        s = selection{}; s.rows = rows;
        require(!s.matches(), "row boundary widened");
    }
#define REJECT(field, value) s = selection{}; s.field = value; require(!s.matches(), #field " boundary widened")
    REJECT(vendor, 0x10de); REJECT(vendor, 0x1002); REJECT(vendor, 0x8086);
    REJECT(driver, 1); REJECT(version, 2150604838u); REJECT(version, 2150604840u);
    REJECT(name, "Adreno (TM) 740"); REJECT(name, "Adreno (TM) 750 extra"); REJECT(name, nullptr);
    REJECT(shader, false); REJECT(subgroup, 32); REJECT(full, false); REJECT(memory, 1279);
#undef REJECT
    s = selection{}; s.memory = 1280;
    require(s.matches(), "shared-memory boundary rejected");
    std::puts("PASS: exact guard and other-vendor/driver/shader/shape/subgroup/capacity negatives");
}
static void tensors(ggml_backend_t backend, int k, int rows, int cols, int biases) {
    ggml_init_params params = {1024 * 1024, nullptr, true};
    auto ctx = ggml_init(params);
    require(ctx != nullptr, "context allocation");
    auto a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
    auto b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, cols);
    auto result = ggml_mul_mat(ctx, a, b);
    ggml_tensor * bias[2] = {nullptr, nullptr};
    for (int i = 0; i < biases; ++i) {
        bias[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, cols);
        result = ggml_add(ctx, result, bias[i]);
    }
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        require(ggml_backend_supports_op(backend, ggml_graph_node(graph, i)), "requested backend does not support graph; no CPU fallback");
    }
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "backend allocation");
    std::vector<float> av(k * rows), bv(k * cols), out(rows * cols), bias_values(rows * cols);
    for (size_t i = 0; i < av.size(); ++i) av[i] = (int(i * 13 % 31) - 15) / 16.0f;
    for (size_t i = 0; i < bv.size(); ++i) bv[i] = (int(i * 7 % 23) - 11) / 8.0f;
    ggml_backend_tensor_set(a, av.data(), 0, av.size() * sizeof(float));
    ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * sizeof(float));
    for (int i = 0; i < biases; ++i) {
        std::fill(bias_values.begin(), bias_values.end(), i == 0 ? 0.125f : -0.25f);
        ggml_backend_tensor_set(bias[i], bias_values.data(), 0, bias_values.size() * sizeof(float));
    }
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "backend graph compute");
    ggml_backend_tensor_get(result, out.data(), 0, out.size() * sizeof(float));
    for (int col = 0; col < cols; ++col) for (int row = 0; row < rows; ++row) {
        double ref = 0, magnitude = 0;
        for (int i = 0; i < k; ++i) {
            double product = double(av[row * k + i]) * bv[col * k + i];
            ref += product; magnitude += std::fabs(product);
        }
        if (biases > 0) ref += 0.125;
        if (biases > 1) ref -= 0.25;
        const double tolerance = 1e-5 + 4e-6 * magnitude;
        if (!std::isfinite(out[col * rows + row]) || std::fabs(out[col * rows + row] - ref) > tolerance) {
            std::fprintf(stderr, "k=%d rows=%d cols=%d biases=%d row=%d col=%d actual=%g expected=%g tolerance=%g\n",
                    k, rows, cols, biases, row, col, out[col * rows + row], ref, tolerance);
            require(false, "tensor mismatch");
        }
    }
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}
int main(int argc, char ** argv) {
    guards();
    if (argc > 1 && std::strcmp(argv[1], "--guard-only") == 0) return 0;
    const char * requested = argc > 1 ? argv[1] : "CPU";
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name(requested, nullptr);
    require(backend != nullptr, "requested backend unavailable; no fallback");
    std::printf("Requested backend=%s actual=%s\n", requested, ggml_backend_name(backend));
    int cases = 0;
    for (int k : {1, 3, 4, 63, 64, 65, 255, 256, 257, 288, 512, 768, 1024, 1536})
        for (int rows : {1, 7}) for (int cols : {1, 2, 4, 5, 6, 8, 16})
            for (int biases : {0, 1, 2}) { tensors(backend, k, rows, cols, biases); ++cases; }
    std::printf("PASS: %d actual backend tensor cases; no performance or other-backend claim\n", cases);
    ggml_backend_free(backend);
}
