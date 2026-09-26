#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"
#include "../src/llama-ext.h"
#include "ggml.h"
#include "gguf-model-data.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "download.h"

#include <array>
#include <vector>
#include <map>
#include <set>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>

// Noop because weights are not needed
static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
}

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

static bool is_turbo_kv_type(ggml_type type) {
    return type == GGML_TYPE_TURBO2_0 ||
           type == GGML_TYPE_TURBO3_0 ||
           type == GGML_TYPE_TURBO4_0;
}

static int count_turbo_wht_on_query_layout_chain(const ggml_tensor * tensor, int direction) {
    std::set<const ggml_tensor *> visited;
    int count = 0;

    while (tensor && visited.insert(tensor).second) {
        switch (tensor->op) {
            case GGML_OP_TURBO_WHT: {
                int tensor_direction;
                memcpy(&tensor_direction, tensor->op_params, sizeof(tensor_direction));
                count += tensor_direction == direction;
                tensor = tensor->src[0];
            } break;
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_CONT:
            case GGML_OP_PAD:
                tensor = tensor->src[0];
                break;
            default:
                return count;
        }
    }

    return count;
}

// Walks backward through pure layout ops (view/reshape/permute/transpose/
// cont/pad) to the first op outside that set. Used to find which node (an
// attention node, or nothing) produced the value now feeding a TURBO_WHT op.
static const ggml_tensor * trace_layout_origin(const ggml_tensor * tensor) {
    std::set<const ggml_tensor *> visited;

    while (tensor && visited.insert(tensor).second) {
        switch (tensor->op) {
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_CONT:
            case GGML_OP_PAD:
                tensor = tensor->src[0];
                break;
            default:
                return tensor;
        }
    }

    return tensor;
}

// Finds every GGML_OP_FLASH_ATTN_EXT node (the graph's attention paths across
// both turbo and non-turbo KV types -- these 5 DSA/MLA architectures and
// plain llama all route attention through FA in this harness's config) and,
// for each, checks that the query layout chain carries exactly one forward
// WHT when K is turbo-typed (zero otherwise), and that the attention output
// feeds exactly one inverse WHT when V is turbo-typed (zero otherwise). This
// scopes the forward/inverse pairing per attention path rather than summing
// GGML_OP_TURBO_WHT nodes graph-wide: a DSA lightning indexer's own forward
// WHT on its query (see build_attn_pad_turbo_query()) legitimately has no
// inverse partner -- it feeds a score, not a value output -- and must not be
// counted here. Emits a GRAPH_ATTENTION_COUNTS line consumed by
// tests/test-turbo-attention-architectures.sh, and (when
// LLAMA_TEST_GRAPH_EXPECTATION is "turbo" or "non-turbo") enforces that the
// graph actually has the turbo coverage the caller expected.
static bool check_and_report_turbo_attention(ggml_cgraph * graph, const char * label, const char * expectation) {
    std::vector<const ggml_tensor *> attn_nodes;

    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            attn_nodes.push_back(node);
        }
    }

    std::set<const ggml_tensor *> attn_set(attn_nodes.begin(), attn_nodes.end());
    std::map<const ggml_tensor *, int> inverse_hits;

    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op != GGML_OP_TURBO_WHT) {
            continue;
        }

        int direction;
        memcpy(&direction, node->op_params, sizeof(direction));
        if (direction != 1) {
            continue;
        }

        const ggml_tensor * origin = trace_layout_origin(node->src[0]);
        if (attn_set.count(origin)) {
            inverse_hits[origin]++;
        }
    }

    int k_turbo = 0, v_turbo = 0, forward_wht = 0, inverse_wht = 0;

    for (size_t idx = 0; idx < attn_nodes.size(); ++idx) {
        const ggml_tensor * node = attn_nodes[idx];
        const bool this_k_turbo = node->src[1] && is_turbo_kv_type(node->src[1]->type);
        const bool this_v_turbo = node->src[2] && is_turbo_kv_type(node->src[2]->type);
        k_turbo += this_k_turbo;
        v_turbo += this_v_turbo;

        const int n_forward = count_turbo_wht_on_query_layout_chain(node->src[0], 0);
        const int expect_forward = this_k_turbo ? 1 : 0;
        if (n_forward != expect_forward) {
            LOG_ERR("%s: attention node %d has %d forward WHT ops on its query chain, expected %d (k_turbo=%d)\n",
                    label, (int) idx, n_forward, expect_forward, (int) this_k_turbo);
            return false;
        }
        forward_wht += n_forward;

        const auto it = inverse_hits.find(node);
        const int n_inverse = it != inverse_hits.end() ? it->second : 0;
        const int expect_inverse = this_v_turbo ? 1 : 0;
        if (n_inverse != expect_inverse) {
            LOG_ERR("%s: attention node %d has %d inverse WHT ops on its output chain, expected %d (v_turbo=%d)\n",
                    label, (int) idx, n_inverse, expect_inverse, (int) this_v_turbo);
            return false;
        }
        inverse_wht += n_inverse;
    }

    LOG_INF("GRAPH_ATTENTION_COUNTS label=%s attention=%d k_turbo=%d v_turbo=%d forward_wht=%d inverse_wht=%d\n",
            label, (int) attn_nodes.size(), k_turbo, v_turbo, forward_wht, inverse_wht);

    if (expectation && strcmp(expectation, "turbo") == 0) {
        if (k_turbo == 0 && v_turbo == 0) {
            LOG_ERR("%s: LLAMA_TEST_GRAPH_EXPECTATION=turbo but no turbo-typed attention node found\n", label);
            return false;
        }
    } else if (expectation && strcmp(expectation, "non-turbo") == 0) {
        if (k_turbo != 0 || v_turbo != 0 || forward_wht != 0 || inverse_wht != 0) {
            LOG_ERR("%s: LLAMA_TEST_GRAPH_EXPECTATION=non-turbo but found turbo attention coverage "
                    "(k_turbo=%d v_turbo=%d forward_wht=%d inverse_wht=%d)\n",
                    label, k_turbo, v_turbo, forward_wht, inverse_wht);
            return false;
        }
    }

    return true;
}

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

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EXPORT_GRAPH_OPS)) {
        return 1;
    }

    // Load CPU-only
    ggml_backend_dev_t cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    params.devices = { cpu_device, nullptr };
    params.fit_params = false;
    params.n_gpu_layers = 0;

    params.warmup = false;

    llama_context * ctx;
    common_init_result_ptr init_result;
    llama_context_ptr ctx2;
    llama_model_ptr model;

    if (params.model.hf_repo.empty()) {
        init_result = common_init_from_params(params);

        ctx = init_result->context();
        if (!ctx) {
            LOG_ERR("failed to initialize params\n");
            return 1;
        }
    } else {
#ifdef LLAMA_HF_FETCH
        auto [hf_repo, hf_quant] = common_download_split_repo_tag(params.model.hf_repo);
        if (hf_quant.empty() || hf_quant == "latest") {
            hf_quant = "Q4_K_M";
        }

        gguf_context_ptr gguf_ctx = gguf_fetch_gguf_ctx(hf_repo, hf_quant);
        if (!gguf_ctx) {
            LOG_ERR("failed to fetch GGUF metadata from %s\n", hf_repo.c_str());
            return 1;
        }

        llama_model_params model_params = llama_model_default_params();
        model_params.devices = params.devices.data();
        model_params.no_alloc = true;

        model.reset(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, nullptr, model_params));

        if (!model) {
            LOG_ERR("failed to create llama_model from %s\n", hf_repo.c_str());
            return 1;
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx2.reset(llama_init_from_model(model.get(), ctx_params));
        ctx = ctx2.get();

        if (!ctx) {
            LOG_ERR("failed to create llama_context\n");
            return 1;
        }
#else
        LOG_ERR("test-export-graph-ops compiled without HF fetch support\n");
        return 1;
#endif
    }

    const uint32_t n_seqs  = llama_n_seq_max(ctx);
    const uint32_t n_tokens = std::min(llama_n_ctx(ctx), llama_n_ubatch(ctx));

    const char * graph_expectation = getenv("LLAMA_TEST_GRAPH_EXPECTATION");

    std::set<test_object> tests;

    auto * gf_pp = llama_graph_reserve(ctx, n_tokens, n_seqs, n_tokens);
    if (!gf_pp) {
        LOG_ERR("failed to reserve prompt processing graph\n");
        return 1;
    }
    if (!check_and_report_turbo_attention(gf_pp, "pp", graph_expectation)) {
        return 1;
    }
    extract_graph_ops(gf_pp, "pp", tests);

    auto * gf_tg = llama_graph_reserve(ctx, n_seqs, n_seqs, n_seqs);
    if (!gf_tg) {
        LOG_ERR("failed to reserve token generation graph\n");
        return 1;
    }
    if (!check_and_report_turbo_attention(gf_tg, "tg", graph_expectation)) {
        return 1;
    }
    extract_graph_ops(gf_tg, "tg", tests);

    LOG_INF("%d unique ops total\n", (int) tests.size());

    std::ofstream f(params.out_file);

    if (!f.is_open()) {
        LOG_ERR("unable to open output file: %s\n", params.out_file.c_str());
        return 1;
    }

    for (const auto& test : tests) {
        test.serialize(f);
    }

    common_log_flush(common_log_main());
    return 0;
}
