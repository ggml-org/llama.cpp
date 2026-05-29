#include "llama-benchmark.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

llama_op_metrics llama_op_metrics_compute(const ggml_tensor * node) {
    llama_op_metrics m = {};

    switch (node->op) {
        case GGML_OP_MUL_MAT: {
            m.N    = node->ne[0];
            m.M    = node->ne[1];
            m.K    = node->src[0]->ne[0];
            m.ops  = 2.0 * m.N * m.K * m.M;
            m.bytes = ggml_nbytes(node->src[0]) + ggml_nbytes(node->src[1]) + ggml_nbytes(node);
            m.quant_type = ggml_type_name(node->src[0]->type);
            break;
        }
        case GGML_OP_MUL_MAT_ID: {
            m.N              = node->ne[0];
            m.n_experts_used = node->ne[1];
            m.M              = node->ne[2];
            m.K              = node->src[0]->ne[0];
            const int64_t total_experts = node->src[0]->ne[2];
            m.ops  = 2.0 * m.N * m.K * m.M * m.n_experts_used;
            m.bytes = (ggml_nbytes(node->src[0]) * m.n_experts_used / total_experts)
                    + ggml_nbytes(node->src[1]) + ggml_nbytes(node);
            m.quant_type = ggml_type_name(node->src[0]->type);
            break;
        }
        case GGML_OP_FLASH_ATTN_EXT: {
            m.head_dim  = node->ne[0];
            m.n_q_heads = node->ne[1];
            m.n_tokens  = node->ne[2];
            m.ctx_len   = node->src[1]->ne[1];
            m.n_kv_heads = node->src[1]->ne[2];
            m.ops  = 2.0 * m.n_tokens * m.head_dim * m.ctx_len * m.n_q_heads * 2.0;
            m.bytes = ggml_nbytes(node->src[0]) + ggml_nbytes(node->src[1])
                    + ggml_nbytes(node->src[2]) + ggml_nbytes(node);
            break;
        }
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            m.n_elements = ggml_nelements(node);
            m.ops  = 6.0 * m.n_elements;
            m.bytes = ggml_nbytes(node->src[0]) + ggml_nbytes(node);
            m.quant_type = ggml_type_name(node->type);
            break;
        }
        case GGML_OP_RMS_NORM: {
            m.n_elements = ggml_nelements(node);
            m.ops  = 3.0 * m.n_elements;
            m.bytes = ggml_nbytes(node->src[0]) + ggml_nbytes(node);
            m.quant_type = ggml_type_name(node->type);
            break;
        }
        case GGML_OP_GLU: {
            m.n_elements = ggml_nelements(node);
            m.ops  = 2.0 * m.n_elements;
            m.bytes = ggml_nbytes(node->src[0]);
            if (node->src[1]) { m.bytes += ggml_nbytes(node->src[1]); }
            m.bytes += ggml_nbytes(node);
            m.quant_type = ggml_type_name(node->type);
            break;
        }
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SUB:
        case GGML_OP_DIV: {
            m.n_elements = ggml_nelements(node);
            m.ops  = m.n_elements;
            m.bytes = ggml_nbytes(node->src[0]);
            if (node->src[1]) { m.bytes += ggml_nbytes(node->src[1]); }
            m.bytes += ggml_nbytes(node);
            m.quant_type = ggml_type_name(node->type);
            break;
        }
        case GGML_OP_GET_ROWS: {
            m.n_elements = ggml_nelements(node);
            m.bytes = ggml_nbytes(node);
            m.quant_type = ggml_type_name(node->src[0]->type);
            break;
        }
        case GGML_OP_SET_ROWS:
        case GGML_OP_CPY: {
            m.n_elements = ggml_nelements(node);
            m.bytes = ggml_nbytes(node->src[0]) + ggml_nbytes(node);
            m.quant_type = ggml_type_name(node->src[0]->type);
            break;
        }
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NONE:
            break;
        default: {
            m.n_elements = ggml_nelements(node);
            m.bytes = ggml_nbytes(node);
            if (node->src[0]) { m.bytes += ggml_nbytes(node->src[0]); }
            m.quant_type = ggml_type_name(node->type);
            break;
        }
    }
    return m;
}

std::string llama_benchmark_predictor::make_key(
        const std::string & op, const std::string & quant,
        int64_t N, int64_t K, int64_t batch) {
    return op + "|" + quant + "|" + std::to_string(N) + "|"
             + std::to_string(K) + "|" + std::to_string(batch);
}

std::string llama_benchmark_predictor::make_attn_key(
        int64_t ctx_len, int64_t n_kv_heads, int64_t n_tokens) {
    return "FLASH_ATTN|" + std::to_string(ctx_len) + "|"
             + std::to_string(n_kv_heads) + "|" + std::to_string(n_tokens);
}

std::string llama_benchmark_predictor::make_elem_key(
        const std::string & op, int64_t n_elements) {
    return op + "|" + std::to_string(n_elements);
}

static void build_entry_map(
        std::unordered_map<std::string, const llama_benchmark_entry *> & map,
        const std::vector<llama_benchmark_entry> & entries) {
    map.clear();
    map.reserve(entries.size());
    for (const auto & e : entries) {
        std::string key;
        if (e.op_name == "MUL_MAT" || e.op_name == "MUL_MAT_ID") {
            key = llama_benchmark_predictor::make_key(e.op_name, e.quant, e.N, e.K, e.B);
        } else if (e.op_name.compare(0, 10, "FLASH_ATTN") == 0) {
            // n_heads stores n_kv_heads for attention benchmarks (written by profiler)
            key = llama_benchmark_predictor::make_attn_key(e.ctx_len, e.n_heads, e.n_tokens);
        } else {
            key = llama_benchmark_predictor::make_elem_key(e.op_name, e.n_elements);
        }
        map[key] = &e;
    }
}

void llama_benchmark_predictor::build_maps() {
    build_entry_map(cpu_map, cpu_entries);
    build_entry_map(gpu_map, gpu_entries);
}

bool llama_benchmark_predictor::load_cpu(const char * filepath, int n_threads) {
    cpu_entries.clear();
    cpu_map.clear();

    FILE * f = fopen(filepath, "r");
    if (!f) {
        LLAMA_LOG_WARN("%s: could not open CPU benchmark file: %s\n", __func__, filepath);
        return false;
    }

    double max_dram_bw = 0.0;

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            int tc = 0;
            double dram_bw = 0.0, pcie_standalone = 0.0, pcie_concurrent = 0.0, cpu_eff = 0.0;

            if (sscanf(line,
                       "#   Threads=%d: DRAM_BW=%lf GB/s, PCIe_Standalone=%lf GB/s,"
                       " PCIe_Concurrent=%lf GB/s (CPU_Eff=%lf%%)",
                       &tc, &dram_bw, &pcie_standalone, &pcie_concurrent, &cpu_eff) == 5) {
                if (dram_bw > max_dram_bw) {
                    max_dram_bw = dram_bw;
                }
                if (tc == n_threads) {
                    stats.peak_system_bw = dram_bw;
                    stats.peak_pcie_bw   = pcie_standalone;
                    stats.eff_system_bw  = dram_bw * (cpu_eff / 100.0);
                    stats.eff_pcie_bw    = std::min(pcie_concurrent, dram_bw);
                }
            } else if (sscanf(line, "#   Threads=%d: DRAM_BW=%lf GB/s", &tc, &dram_bw) == 2) {
                if (dram_bw > max_dram_bw) {
                    max_dram_bw = dram_bw;
                }
                if (tc == n_threads) {
                    stats.peak_system_bw = dram_bw;
                }
            }
            continue;
        }

        if (line[0] == '\n') continue;

        llama_benchmark_entry e;
        char op_name[64], quant[16];

        int parsed = sscanf(line,
                "%63s %15s %d %lf %lf %lf %lf %lf %lf"
                " %lld %lld %lld %lld %lld %lld %lld %lld",
                op_name, quant, &e.threads,
                &e.ai, &e.bw_gb_s, &e.peak_gflops, &e.ridge,
                &e.eff_gflops, &e.eff_pcie_bw,
                &e.N, &e.K, &e.B, &e.n_tokens, &e.ctx_len,
                &e.n_heads, &e.head_dim, &e.n_elements);

        if (parsed >= 9 && e.threads == n_threads) {
            e.op_name = op_name;
            e.quant   = quant;
            cpu_entries.push_back(std::move(e));
        }
    }
    fclose(f);

    // PCIe can't exceed system DRAM BW
    if (max_dram_bw > 0.0) {
        stats.peak_pcie_bw = std::min(stats.peak_pcie_bw, max_dram_bw);
    } else {
        stats.peak_pcie_bw = std::min(stats.peak_pcie_bw, stats.peak_system_bw);
    }

    if (stats.eff_system_bw == 0.0) { stats.eff_system_bw = stats.peak_system_bw; }
    if (stats.eff_pcie_bw   == 0.0) { stats.eff_pcie_bw   = stats.peak_pcie_bw;   }

    build_maps();

    LLAMA_LOG_INFO("%s: loaded %zu CPU entries for %d threads"
                   " (peak: DRAM=%.1f GB/s, PCIe=%.1f GB/s"
                   " | concurrent: DRAM=%.1f GB/s, PCIe=%.1f GB/s)\n",
                   __func__, cpu_entries.size(), n_threads,
                   stats.peak_system_bw, stats.peak_pcie_bw,
                   stats.eff_system_bw, stats.eff_pcie_bw);

    return !cpu_entries.empty();
}

bool llama_benchmark_predictor::load_gpu(const char * filepath) {
    gpu_entries.clear();
    gpu_map.clear();

    FILE * f = fopen(filepath, "r");
    if (!f) {
        LLAMA_LOG_WARN("%s: could not open GPU benchmark file: %s\n", __func__, filepath);
        return false;
    }

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            const char * bw_str = strstr(line, "GPU_Memory_BW=");
            const char * cp_str = strstr(line, "GPU_Peak_Compute=");
            if (bw_str && cp_str) {
                double mem_bw = 0.0, compute = 0.0;
                sscanf(bw_str, "GPU_Memory_BW=%lf", &mem_bw);
                sscanf(cp_str, "GPU_Peak_Compute=%lf", &compute);
                stats.peak_gpu_mem_bw  = mem_bw;
                stats.peak_gpu_compute = compute;
            }
            continue;
        }

        if (line[0] == '\n') continue;

        llama_benchmark_entry e;
        char op_name[64], quant[16];

        int parsed = sscanf(line,
                "%63s %15s %lf %lf %lf %lf"
                " %lld %lld %lld %lld %lld %lld %lld %lld",
                op_name, quant,
                &e.ai, &e.bw_gb_s, &e.peak_gflops, &e.ridge,
                &e.N, &e.K, &e.B, &e.n_tokens, &e.ctx_len,
                &e.n_heads, &e.head_dim, &e.n_elements);

        if (parsed == 14) {
            e.op_name     = op_name;
            e.quant       = quant;
            e.threads     = -1;
            e.eff_gflops  = e.peak_gflops;
            e.eff_pcie_bw = 0.0;
            gpu_entries.push_back(std::move(e));
        }
    }
    fclose(f);

    build_maps();

    LLAMA_LOG_INFO("%s: loaded %zu GPU entries (mem_bw=%.1f GB/s, compute=%.1f GFLOP/s)\n",
                   __func__, gpu_entries.size(),
                   stats.peak_gpu_mem_bw, stats.peak_gpu_compute);

    return !gpu_entries.empty();
}

std::string llama_benchmark_predictor::make_timing_key(
        bool is_gpu, bool async_copy, const char * op_name,
        const llama_op_metrics & m, int32_t batch_size) {
    // mode prefix: GPU vs CPU with/without concurrent PCIe (async_copy uses eff_* bandwidths)
    std::string key = is_gpu ? "GPU|" : (async_copy ? "CPU_ASYNC|" : "CPU_SYNC|");
    key += op_name;
    key += "|";

    if (m.quant_type) {
        key += m.quant_type;
        key += "|";
    }

    if (strstr(op_name, "MUL_MAT_ID")) {
        key += std::to_string(m.N) + "|" + std::to_string(m.K) + "|"
             + std::to_string(batch_size) + "|" + std::to_string(m.M) + "|"
             + std::to_string(m.n_experts_used);
    } else if (strstr(op_name, "MUL_MAT")) {
        key += std::to_string(m.N) + "|" + std::to_string(m.K) + "|"
             + std::to_string(batch_size) + "|" + std::to_string(m.M);
    } else if (strstr(op_name, "FLASH_ATTN")) {
        key += std::to_string(m.head_dim) + "|" + std::to_string(m.n_q_heads) + "|"
             + std::to_string(batch_size) + "|" + std::to_string(m.n_tokens) + "|"
             + std::to_string(m.ctx_len) + "|" + std::to_string(m.n_kv_heads);
    } else {
        key += std::to_string(m.n_elements);
    }

    return key;
}

llama_split_timing llama_benchmark_predictor::predict_split(
        struct ggml_tensor ** nodes, int n_nodes,
        bool is_gpu, int32_t batch_size, bool async_copy,
        timing_cache_t * timing_cache) const {

    llama_split_timing result = {};
    timing_cache_t local_timing_cache;
    timing_cache_t & cache = timing_cache ? *timing_cache : local_timing_cache;

    const auto & map     = is_gpu ? gpu_map : cpu_map;
    const auto & entries = is_gpu ? gpu_entries : cpu_entries;

    double pcie_sum = 0.0;

    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = nodes[i];
        llama_op_metrics m = llama_op_metrics_compute(node);

        if (m.ops == 0.0 && m.bytes == 0.0) continue;

        const char * op_name = ggml_op_name(node->op);
        double op_time_ms  = 0.0;
        double pcie_contrib = 0.0;

        std::string tkey = make_timing_key(is_gpu, async_copy, op_name, m, batch_size);
        auto cache_it = cache.find(tkey);
        if (cache_it != cache.end()) {
            op_time_ms   = cache_it->second.first;
            pcie_contrib = cache_it->second.second;
            result.time_ms += op_time_ms;
            pcie_sum       += pcie_contrib;
            result.n_cache_hit++;
            result.op_count++;
            continue;
        }

        // build hash key for exact lookup
        std::string hkey;
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                hkey = make_key("MUL_MAT", m.quant_type ? m.quant_type : "", m.N, m.K, batch_size);
                break;
            case GGML_OP_MUL_MAT_ID:
                hkey = make_key("MUL_MAT_ID", m.quant_type ? m.quant_type : "", m.N, m.K, batch_size);
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                hkey = make_attn_key(m.ctx_len, m.n_kv_heads, batch_size);
                break;
            case GGML_OP_ROPE:
            case GGML_OP_RMS_NORM:
            case GGML_OP_GLU:
            case GGML_OP_ADD:
            case GGML_OP_MUL:
            case GGML_OP_SUB:
            case GGML_OP_DIV:
                hkey = make_elem_key(op_name, m.n_elements);
                break;
            default:
                break;
        }

        const llama_benchmark_entry * match = nullptr;
        bool exact = false;

        // try exact hash match
        if (!hkey.empty()) {
            auto it = map.find(hkey);
            if (it != map.end()) {
                match = it->second;
                exact = true;
                result.n_exact++;
            }
        }

        // fall back to nearest-neighbor
        if (!match) {
            switch (node->op) {
                case GGML_OP_MUL_MAT:
                    match = find_nearest(entries, "MUL_MAT", m.quant_type, m.N, m.K, 0, 0, -1, batch_size);
                    break;
                case GGML_OP_MUL_MAT_ID:
                    match = find_nearest(entries, "MUL_MAT_ID", m.quant_type, m.N, m.K, 0, 0, -1, batch_size);
                    break;
                case GGML_OP_FLASH_ATTN_EXT:
                    match = find_nearest(entries, "FLASH_ATTN", nullptr, 0, 0, m.ctx_len, 0, m.n_kv_heads, batch_size);
                    break;
                default:
                    match = find_nearest(entries, op_name, nullptr, 0, 0, 0, m.n_elements, -1, batch_size);
                    break;
            }
            if (match) {
                result.n_nearest++;
            }
        }

        // compute timing from match or fall back to peak BW
        if (match) {
            // GPU always peak; CPU uses eff (concurrent) when async_copy, peak otherwise
            const double gflops = is_gpu ? match->peak_gflops
                                         : (async_copy ? match->eff_gflops : match->peak_gflops);

            if (exact) {
                if (gflops > 0.0) {
                    op_time_ms = (m.ops / 1e9) / gflops * 1000.0;
                }
            } else {
                const double ai = m.bytes > 0.0 ? (m.ops / m.bytes) : 0.0;
                if (ai < match->ridge && match->bw_gb_s > 0.0) {
                    op_time_ms = (m.bytes / 1e9) / match->bw_gb_s * 1000.0;
                } else if (gflops > 0.0) {
                    op_time_ms = (m.ops / 1e9) / gflops * 1000.0;
                }
            }

            pcie_contrib = op_time_ms * match->eff_pcie_bw;
        } else {
            result.n_fallback++;
            if (m.bytes > 0.0) {
                const double bw = is_gpu ? stats.peak_gpu_mem_bw
                                         : (async_copy ? stats.eff_system_bw : stats.peak_system_bw);
                if (bw > 0.0) {
                    op_time_ms = (m.bytes / 1e9) / bw * 1000.0;
                }
                if (!is_gpu && stats.eff_pcie_bw > 0.0) {
                    pcie_contrib = op_time_ms * stats.eff_pcie_bw;
                }
            }
        }

        result.time_ms += op_time_ms;
        pcie_sum       += pcie_contrib;
        result.op_count++;

        cache[tkey] = {op_time_ms, pcie_contrib};
    }

    // convert accumulated pcie_sum to time-weighted average BW
    if (!is_gpu) {
        result.eff_pcie_bw = (result.time_ms > 0.0) ? (pcie_sum / result.time_ms)
                                                      : stats.eff_pcie_bw;
    }

    return result;
}

static double get_bits_per_weight(const std::string & quant) {
    static const std::unordered_map<std::string, double> table = {
        {"q2_K",  2.6}, {"q3_K",  3.4}, {"q4_0",  4.5}, {"q4_1",  5.0}, {"q4_K",  4.5},
        {"q5_0",  5.5}, {"q5_1",  6.0}, {"q5_K",  5.5}, {"q6_K",  6.6},
        {"q8_0",  8.5}, {"q8_1",  9.0}, {"f16",  16.0}, {"f32",  32.0},
    };
    auto it = table.find(quant);
    return it != table.end() ? it->second : -1.0;
}

const llama_benchmark_entry * llama_benchmark_predictor::find_nearest(
        const std::vector<llama_benchmark_entry> & entries,
        const char * op_name, const char * quant,
        int64_t N, int64_t K, int64_t ctx_len, int64_t n_elements,
        int64_t n_kv_heads, int64_t target_batch) {

    if (!op_name || entries.empty()) {
        return nullptr;
    }

    const llama_benchmark_entry * best = nullptr;
    double best_score = 1e20;

    const double target_bpw = quant ? get_bits_per_weight(quant) : -1.0;

    // FLASH_ATTN entries have suffixed names (e.g. FLASH_ATTN_MHA, FLASH_ATTN_GQA-8)
    const bool attn_query = (strncmp(op_name, "FLASH_ATTN", 10) == 0);
    const bool matmul_query = (strcmp(op_name, "MUL_MAT") == 0
                            || strcmp(op_name, "MUL_MAT_ID") == 0);

    for (const auto & b : entries) {
        if (attn_query) {
            if (b.op_name.compare(0, 10, "FLASH_ATTN") != 0) continue;
        } else {
            if (b.op_name != op_name) continue;
        }

        double dim_score   = 0.0;
        double batch_score = 0.0;
        double quant_score = 0.0;

        if (matmul_query) {
            const double n_diff = std::abs((double)b.N - N) / std::max(N, (int64_t)1);
            const double k_diff = std::abs((double)b.K - K) / std::max(K, (int64_t)1);
            dim_score = n_diff + k_diff;
        } else if (attn_query) {
            const double ctx_diff = std::abs((double)b.ctx_len - ctx_len) / std::max(ctx_len, (int64_t)1);
            double kv_diff = 0.0;
            if (n_kv_heads > 0 && b.n_heads > 0) {
                kv_diff = std::abs((double)b.n_heads - n_kv_heads) / std::max(n_kv_heads, (int64_t)1);
            }
            dim_score = ctx_diff + kv_diff * 0.5;
        } else {
            const double elem_diff = std::abs((double)b.n_elements - n_elements) / std::max(n_elements, (int64_t)1);
            dim_score = elem_diff;
        }

        if (target_batch > 0 && b.B > 0) {
            batch_score = std::abs((double)b.B - target_batch) / std::max(target_batch, (int64_t)1);
        }

        if (target_bpw > 0) {
            const double bench_bpw = get_bits_per_weight(b.quant);
            if (bench_bpw > 0) {
                quant_score = std::abs(bench_bpw - target_bpw) / target_bpw;
            } else {
                quant_score = 1.0;
            }
        } else if (quant && b.quant != quant) {
            continue;
        }

        // batch most important (memory vs compute regime), then dims, then quant
        const double score = batch_score * 1.0 + dim_score * 0.5 + quant_score * 0.3;

        if (score < best_score) {
            best_score = score;
            best       = &b;
        }
    }

    return best;
}

double llama_benchmark_predictor::predict_tps(
        ggml_backend_sched_t sched,
        int cpu_backend_id,
        uint32_t kv_size,
        int32_t batch_size,
        uint32_t n_outputs,
        bool has_rs) const {

    const int n_splits = ggml_backend_sched_get_n_splits(sched);
    if (n_splits <= 0) return 0.0;

    LLAMA_LOG_DEBUG("%s: n_splits=%d, bs=%d, kv_size=%u, n_outputs=%u, has_rs=%d\n",
        __func__, n_splits, batch_size, kv_size, n_outputs, (int)has_rs);

    const double pcie_bw = stats.peak_pcie_bw;
    // KV transfer ratios vs full writeback_bytes the sched reports:
    //   download: only newly written cells -> batch_size/kv_size (decode bs=1/kv=1024 -> 0.1%)
    //   upload:   all used cells (cache fills over time) -> 1.0 (conservative, assumes full cache)
    // RS models use FULL mode (entire tensor every batch), so both are 1.0.
    const double dl_ratio = has_rs ? 1.0
        : ((kv_size > 0) ? std::min(1.0, (double)batch_size / kv_size) : 1.0);
    const double ul_ratio = 1.0;
    double total_ms = 0.0;
    bool copy_prefetched = false;
    timing_cache_t timing_cache;

    for (int i = 0; i < n_splits; i++) {
        struct ggml_backend_sched_split_info si = {};
        if (!ggml_backend_sched_get_split_info(sched, i, &si)) continue;

        const bool is_gpu = (si.backend_id != cpu_backend_id);

        double input_copy_ms = 0.0;
        double input_copy_bytes = 0.0;
        if (is_gpu && !copy_prefetched && pcie_bw > 0.0) {
            input_copy_bytes = (double)si.input_weight_copy_bytes
                             + (double)si.writeback_bytes * ul_ratio;
            input_copy_ms = (input_copy_bytes / 1e9 / pcie_bw) * 1000.0;
        }

        // peek at next split to determine if async prefetch will overlap with this split
        double prefetch_bytes = 0.0;
        bool next_copy_prefetched = false;
        if (i + 1 < n_splits) {
            struct ggml_backend_sched_split_info next_si = {};
            if (ggml_backend_sched_get_split_info(sched, i + 1, &next_si) &&
                next_si.can_prefetch_weights) {
                prefetch_bytes = (double)next_si.input_weight_bytes
                               + (double)next_si.writeback_bytes * ul_ratio;
                next_copy_prefetched = prefetch_bytes > 0.0;
            }
        }

        const bool async_copy = (prefetch_bytes > 0.0);

        // compute cost (CPU splits use eff_gflops when async_copy due to PCIe contention)
        struct ggml_tensor ** nodes = ggml_graph_nodes(si.graph);
        int n_nodes = ggml_graph_n_nodes(si.graph);
        llama_split_timing t = predict_split(nodes, n_nodes, is_gpu, batch_size, async_copy, &timing_cache);

        // prefetch cost: use concurrent PCIe BW for CPU splits (bus shared with DRAM),
        // peak PCIe BW for GPU splits (GPU compute doesn't contend with PCIe DMA)
        double prefetch_ms = 0.0;
        if (prefetch_bytes > 0.0) {
            const double eff_bw = (!is_gpu && t.eff_pcie_bw > 0.0) ? t.eff_pcie_bw : pcie_bw;
            prefetch_ms = (prefetch_bytes / 1e9 / eff_bw) * 1000.0;
        }

        // output scaling: use the output rows in the reserved graph, then scale to
        // the runtime number of logits. This keeps the memory-probe graph as the
        // source of truth and avoids double-scaling when it was already reduced.
        if (i == n_splits - 1 && n_outputs > 0) {
            for (int j = 0; j < n_nodes; j++) {
                if (nodes[j]->name && strstr(nodes[j]->name, "result_output")) {
                    const llama_op_metrics out_m = llama_op_metrics_compute(nodes[j]);
                    const int32_t graph_outputs = (int32_t) std::max<int64_t>(1, out_m.M);
                    if (graph_outputs != batch_size || n_outputs < (uint32_t) graph_outputs) {
                        llama_split_timing out_included = predict_split(&nodes[j], 1, is_gpu, batch_size, async_copy, &timing_cache);
                        llama_split_timing out_graph    = predict_split(&nodes[j], 1, is_gpu, graph_outputs, async_copy, &timing_cache);
                        if (out_included.time_ms > 0.0 || out_graph.time_ms > 0.0) {
                            const double scale = n_outputs < (uint32_t) graph_outputs
                                               ? (double)n_outputs / (double)graph_outputs
                                               : 1.0;
                            const double scaled_ms = out_graph.time_ms * scale;
                            t.time_ms -= out_included.time_ms;
                            t.time_ms += scaled_ms;
                            LLAMA_LOG_DEBUG("%s: out_t: included=%f graph=%f scaled=%f rows=%d->%u\n",
                                    __func__, out_included.time_ms, out_graph.time_ms, scaled_ms, graph_outputs, n_outputs);
                        }
                    }
                    break;
                }
            }
        }

        // KV/RS download cost (GPU splits, after compute, on compute stream)
        double kv_dl_ms = 0.0;
        if (is_gpu && si.writeback_bytes > 0 && pcie_bw > 0.0) {
            double dl_bytes = (double)si.writeback_bytes * dl_ratio;
            kv_dl_ms = (dl_bytes / 1e9 / pcie_bw) * 1000.0;
        }

        // activation copy cost (synchronous, between splits on different backends)
        double activ_copy_ms = 0.0;
        if (si.input_activ_bytes > 0 && pcie_bw > 0.0) {
            activ_copy_ms = ((double)si.input_activ_bytes / 1e9 / pcie_bw) * 1000.0;
        }

        // Prefetch is enqueued after current inputs/precompute and before graph compute,
        // so whatever remains can overlap the current split compute. CPU splits use the
        // effective PCIe BW above because DMA contends with CPU memory traffic.
        double split_ms = input_copy_ms + activ_copy_ms;
        split_ms += std::max(t.time_ms + kv_dl_ms, prefetch_ms);
        total_ms += split_ms;
        copy_prefetched = next_copy_prefetched;

        double dl_bytes = is_gpu ? (double)si.writeback_bytes * dl_ratio : 0.0;
        LLAMA_LOG_DEBUG("%s:   split %d/%d [%s] input_copy=%.3f (%.2f MiB) compute=%.3f kv_dl=%.3f (%.2f MiB) prefetch=%.3f (%.2f MiB) activ=%.3f (%.2f MiB) -> %.3f ms"
            " (exact=%d near=%d fall=%d cache=%d)\n",
            __func__, i, n_splits, is_gpu ? "GPU" : "CPU",
            input_copy_ms, input_copy_bytes / (1024.0 * 1024.0),
            t.time_ms,
            kv_dl_ms, dl_bytes / (1024.0 * 1024.0),
            prefetch_ms, prefetch_bytes / (1024.0 * 1024.0),
            activ_copy_ms, (double)si.input_activ_bytes / (1024.0 * 1024.0),
            split_ms,
            t.n_exact, t.n_nearest, t.n_fallback, t.n_cache_hit);
    }

    double tps = (total_ms > 0.0) ? (batch_size * 1000.0 / total_ms) : 0.0;
    LLAMA_LOG_DEBUG("%s: total=%.3f ms, dl_ratio=%.4f, ul_ratio=%.1f, pcie_bw=%.1f GB/s -> %.1f tps\n",
        __func__, total_ms, dl_ratio, ul_ratio, pcie_bw, tps);
    return tps;
}
