#include "ggml-vulkan-common.h"

bool vk_perf_logger_enabled = false;

bool vk_perf_logger_concurrent = false;

uint32_t vk_perf_logger_frequency = 1;

void vk_perf_logger::print_timings(bool force) {
    if (timings.empty()) {
        return;
    }
    print_count++;
    if ((print_count % vk_perf_logger_frequency) != 0 && !force) {
        return;
    }
    print_count = 0;
    uint64_t total_all_op_times = 0;
    std::cerr << "----------------\nVulkan Timings:" << std::endl;
    for (const auto & t : timings) {
        uint64_t total_op_times = 0;
        for (const auto & time : t.second) {
            total_op_times += time;
        }
        std::cerr << t.first << ": " << t.second.size() << " x " << (total_op_times / t.second.size() / 1000.0)
                  << " us = " << (total_op_times / 1000.0) << " us";

        // If we have as many flops entries as timing entries for the op, then compute and log the flops/S.
        auto it = flops.find(t.first);
        if (it != flops.end() && (it->second).size() == t.second.size()) {
            uint64_t total_op_flops = 0;
            for (const auto & elem : it->second) {
                total_op_flops += elem;
            }
            std::cerr << " ("
                      << (double(total_op_flops) / (1000.0 * 1000.0 * 1000.0)) /
                             (double(total_op_times) / (1000.0 * 1000.0 * 1000.0))
                      << " GFLOPS/s)";
        }

        total_all_op_times += total_op_times;

        std::cerr << std::endl;
    }

    if (timings.size() > 0) {
        std::cerr << "Total time: " << total_all_op_times / 1000.0 << " us." << std::endl;
    }

    timings.clear();
    flops.clear();
}

std::string vk_perf_logger::get_node_fusion_name(const ggml_tensor * node, const char *fusion_name, uint64_t *n_flops) {
    *n_flops = 0;
    std::string fusion_str;
    if (fusion_name) {
        fusion_str = fusion_name + std::string(" ");
    }
    if (node->op == GGML_OP_UNARY) {
        return fusion_str + ggml_unary_op_name(ggml_get_unary_op(node));
    }
    if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
        const uint64_t m     = node->ne[0];
        const uint64_t n     = node->ne[1];
        const uint64_t k     = node->src[1]->ne[0];
        const uint64_t batch = node->ne[2] * node->ne[3];
        std::string    name  = ggml_op_name(node->op);
        if ((node->op == GGML_OP_MUL_MAT && n <= mul_mat_vec_max_cols) ||
            (node->op == GGML_OP_MUL_MAT_ID && node->src[2]->ne[1] == 1)) {
            name += "_VEC";
        }
        name += " ";
        name += ggml_type_name(node->src[0]->type);
        name += " m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k);
        if (node->op == GGML_OP_MUL_MAT_ID) {
            name += " n_expert=" + std::to_string(node->src[0]->ne[2]);
        }
        if (batch > 1) {
            name += " batch=" + std::to_string(batch);
        }
        name = fusion_str + name;
        *n_flops = m * n * (k + (k - 1)) * batch;
        return name;
    }
    if (node->op == GGML_OP_CONV_2D || node->op == GGML_OP_CONV_TRANSPOSE_2D) {
        std::string   name    = ggml_op_name(node->op);
        ggml_tensor * knl     = node->src[0];
        uint64_t      OW      = node->ne[0];
        uint64_t      OH      = node->ne[1];
        uint64_t      N       = node->ne[3];
        uint64_t      Cout    = node->ne[2];
        uint64_t      KW      = knl->ne[0];
        uint64_t      KH      = knl->ne[1];
        uint64_t      Cin     = node->src[1]->ne[2];
        // KxCRS @ CRSxNPQ = KxNPQ -> M=K, K=CRS, N=NPQ
        uint64_t      size_M  = Cout;
        uint64_t      size_K  = Cin * KW * KH;
        uint64_t      size_N  = N * OW * OH;
        *n_flops = size_M * size_N * (size_K + (size_K - 1));
        name += " M=Cout=" + std::to_string(size_M) + ", K=Cin*KW*KH=" + std::to_string(size_K) +
                ", N=N*OW*OH=" + std::to_string(size_N);
        name = fusion_str + name;
        return name;
    }
    if (node->op == GGML_OP_RMS_NORM) {
        std::string   name    = ggml_op_name(node->op);
        name += "(" + std::to_string(node->ne[0]) + "," + std::to_string(node->ne[1]) + "," + std::to_string(node->ne[2]) + "," + std::to_string(node->ne[3]) + ")";
        name = fusion_str + name;
        return name;
    }
    if (node->op == GGML_OP_FLASH_ATTN_EXT) {
        const ggml_tensor * dst = node;
        const ggml_tensor * q = node->src[0];
        const ggml_tensor * k = node->src[1];
        const ggml_tensor * v = node->src[2];
        const ggml_tensor * m = node->src[3];
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(node->op) <<
            " dst(" << dst->ne[0] << "," << dst->ne[1] << "," << dst->ne[2] << "," << dst->ne[3] << "), " <<
            " q(" << q->ne[0] << "," << q->ne[1] << "," << q->ne[2] << "," << q->ne[3] << "), " <<
            " k(" << k->ne[0] << "," << k->ne[1] << "," << k->ne[2] << "," << k->ne[3] << "), " <<
            " v(" << v->ne[0] << "," << v->ne[1] << "," << v->ne[2] << "," << v->ne[3] << "), " <<
            " m(" << (m?m->ne[0]:0) << "," << (m?m->ne[1]:0) << "," << (m?m->ne[2]:0) << "," << (m?m->ne[3]:0) << ")";
        *n_flops = 2ull * q->ne[1] * q->ne[2] * (k->ne[0] + v->ne[0]) * k->ne[1] * q->ne[3];
        return name.str();
    }
    if (node->op == GGML_OP_TOP_K) {
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(node->op) <<
            " K=" << node->ne[0] <<
            " (" << node->src[0]->ne[0] << "," << node->src[0]->ne[1] << "," << node->src[0]->ne[2] << "," << node->src[0]->ne[3] << ")";
        return name.str();
    }
    return fusion_str + ggml_op_name(node->op);
}

void vk_perf_logger::log_timing(const ggml_tensor * node, const char *fusion_name, uint64_t time) {
    uint64_t n_flops;
    std::string name = get_node_fusion_name(node, fusion_name, &n_flops);
    if (n_flops) {
        flops[name].push_back(n_flops);
    }
    timings[name].push_back(time);
}

void vk_perf_logger::log_timing(const std::vector<ggml_tensor *> &nodes, const std::vector<const char *> &names, uint64_t time) {
    uint64_t total_flops = 0;
    std::string name;
    for (size_t n = 0; n < nodes.size(); ++n) {
        uint64_t n_flops = 0;
        name += get_node_fusion_name(nodes[n], names[n], &n_flops);
        total_flops += n_flops;

        if (n != nodes.size() - 1) {
            name += ", ";
        }
    }
    if (total_flops) {
        flops[name].push_back(total_flops);
    }
    timings[name].push_back(time);
}
