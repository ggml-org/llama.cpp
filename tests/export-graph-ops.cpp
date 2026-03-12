#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "../src/llama-ext.h"
#include "ggml.h"

#include <array>
#include <vector>
#include <set>
#include <fstream>
#include <iostream>

struct input_tensor {
    ggml_type type;
    std::array<int64_t, 4> ne;
    std::array<size_t, 4> nb;

    input_tensor(ggml_type type, int64_t * ne, size_t * nb): type(type) {
        memcpy(this->ne.data(), ne, 4 * sizeof(int64_t));
        memcpy(this->nb.data(), nb, 4 * sizeof(size_t));
    }

    bool operator<(const input_tensor &b) const {
        return std::tie(type, ne, nb) <
               std::tie(b.type, b.ne, b.nb);
    }

    void serialize(std::ostream& out) const {
        out << type << ' ';
        for (size_t i = 0; i < 4; i++) {
            out << ne[i] << ' ';
        }
        for (size_t i = 0; i < 4; i++) {
            out << nb[i] << ' ';
        }
    }
};

struct test_object {
    ggml_op op;
    ggml_type type;
    std::array<int64_t, 4> ne;
    std::vector<int32_t> op_params;
    std::vector<input_tensor> sources;
    std::string name;

    void serialize(std::ostream& out) const {
        out << op << ' ' << type << ' ';
        for (size_t i = 0; i < 4; i++) {
            out << ne[i] << ' ';
        }

        out << op_params.size() << ' ';
        for (size_t i = 0; i < op_params.size(); i++) {
            out << op_params[i] << ' ';
        }

        out << sources.size() << ' ';
        for (size_t s = 0; s < sources.size(); s++) {
            sources[s].serialize(out);
        }

        if (!name.empty()) {
            out << name;
        } else {
            out << '-';
        }

        out << '\n';
    }

    bool operator<(const test_object &b) const {
        return std::tie(op, type, ne, op_params, sources) <
               std::tie(b.op, b.type, b.ne, b.op_params, b.sources);
    }
};

static void extract_graph_ops(ggml_cgraph * cgraph, const char * label, std::set<test_object> & tests) {
    int n_nodes = ggml_graph_n_nodes(cgraph);
    int n_skipped = 0;
    int n_before = (int) tests.size();
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = ggml_graph_node(cgraph, i);

        if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
            n_skipped++;
            continue;
        }

        test_object test;

        test.op = node->op;
        test.type = node->type;
        memcpy(&test.ne, node->ne, 4 * sizeof(int64_t));

        test.op_params.resize(GGML_MAX_OP_PARAMS / sizeof(int32_t));
        memcpy(test.op_params.data(), node->op_params, GGML_MAX_OP_PARAMS);

        for (size_t s = 0; s < GGML_MAX_SRC; s++) {
            if (node->src[s] == nullptr) {
                break;
            }

            test.sources.emplace_back(node->src[s]->type, node->src[s]->ne, node->src[s]->nb);
        }

        test.name = node->name;
        tests.insert(test);
    }

    int n_new = (int) tests.size() - n_before;
    LOG_INF("%s: %d unique ops, %d total nodes, %d skipped (view ops)\n",
            label, n_new, n_nodes, n_skipped);
}

int main(int argc, char ** argv) {
    common_params params;
    params.out_file = "tests.txt";

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EXPORT_GRAPH_OPS)) {
        return 1;
    }

    common_init();

    // Load CPU-only
    ggml_backend_dev_t cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    params.devices = { cpu_device, nullptr };
    params.fit_params = false;
    params.n_gpu_layers = 0;

    params.warmup = false;

    auto init_result = common_init_from_params(params);

    llama_context * ctx = init_result->context();

    const uint32_t n_seqs  = llama_n_seq_max(ctx);
    const uint32_t n_tokens = std::min(llama_n_ctx(ctx), llama_n_ubatch(ctx));

    std::set<test_object> tests;

    auto * gf_pp = llama_graph_reserve(ctx, n_tokens, n_seqs, n_tokens);
    if (!gf_pp) {
        throw std::runtime_error("failed to reserve prompt processing graph");
    }
    extract_graph_ops(gf_pp, "pp", tests);

    auto * gf_tg = llama_graph_reserve(ctx, n_seqs, n_seqs, n_seqs);
    if (!gf_tg) {
        throw std::runtime_error("failed to reserve token generation graph");
    }
    extract_graph_ops(gf_tg, "tg", tests);

    LOG_INF("%d unique ops total\n", (int) tests.size());

    std::ofstream f(params.out_file);

    if (!f.is_open()) {
        throw std::runtime_error("Unable to open output file");
    }

    for (const auto& test : tests) {
        test.serialize(f);
    }

    return 0;
}
