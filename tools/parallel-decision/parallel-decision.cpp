// Parallel constrained decisions from prepared request files (see decision-engine.h).
//
// Protocol (compatible with llama-mojo's decide-worker and its tools/prepare_decisions.py):
//   llama-parallel-decision MODEL N_CPU_MOE --worker N_CTX N_BATCH [STREAM_MIN [GEMM_MIN [THREADS]]]
//   stdin : one request per line, tab separated: @shared|@fresh STATIC CONTEXT FIELD...
//           each FIELD file: a count N, N allowed-value lines, then the field's suffix
//   stdout: prefill_ms .. suffix_scoring_ms .. decision_rounds .., RESULT_JSON, <json>, WORKER_DONE
// STREAM_MIN and GEMM_MIN are llama-mojo tuning knobs; they are accepted and ignored here.
// Environment: DECIDE_TREE (1 = tree, 0 = greedy, unset = auto), DECIDE_TREE_MAX (default 128),
// DECIDE_SPLIT_BOUNDARY=1 (legacy tokenisation), DECIDE_NSEQ (sequences, default 24).

#include "decision-engine.h"

#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string env_str(const char * name) {
    const char * v = std::getenv(name);
    return v ? v : "";
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static void decide_files(llama_decision::engine & eng, const std::vector<std::string> & files) {
    if (files.size() < 4 || (files[0] != "@shared" && files[0] != "@fresh")) {
        throw std::runtime_error("request: @shared|@fresh STATIC CONTEXT FIELD...");
    }
    llama_decision::options opt;
    const std::string tree = env_str("DECIDE_TREE");
    opt.mode           = tree == "1" ? "tree" : tree == "0" ? "greedy" : "auto";
    opt.tree_max       = env_str("DECIDE_TREE_MAX").empty() ? 128 : std::stoul(env_str("DECIDE_TREE_MAX"));
    opt.split_boundary = env_str("DECIDE_SPLIT_BOUNDARY") == "1";
    opt.allow_cache    = files[0] == "@shared";

    std::vector<llama_decision::field_input> fields;
    for (size_t f = 3; f < files.size(); ++f) {
        std::vector<std::string> lines;
        std::stringstream ss(read_file(files[f]));
        for (std::string line; std::getline(ss, line);) {
            lines.push_back(line);
        }
        const int n = lines.empty() ? 0 : std::stoi(lines[0]);
        if (n < 1 || (int) lines.size() < n + 2) {
            throw std::runtime_error("field file needs a count, the allowed values and a suffix");
        }
        llama_decision::field_input in;
        in.candidates.assign(lines.begin() + 1, lines.begin() + 1 + n);
        for (size_t j = n + 1; j < lines.size(); ++j) {
            in.suffix += (j > (size_t) n + 1 ? "\n" : "") + lines[j];
        }
        fields.push_back(in);
    }

    const auto r = eng.decide(read_file(files[1]), read_file(files[2]), fields, opt);

    std::ostringstream out;
    out << "prefill_ms " << r.prefill_ms << " suffix_scoring_ms " << r.scoring_ms << " decision_rounds " << r.rounds << "\n";
    out << "RESULT_JSON\n";
    out << "{\"mode\":\"" << opt.mode << "\",\"fused\":0,\"rows\":" << r.rows << ",\"prefix_cache_hit\":" << (r.cache_hit ? 1 : 0)
        << ",\"shared_tokens\":" << r.shared_tokens << ",\"context_tokens\":" << r.context_tokens
        << ",\"decision_rounds\":" << r.rounds << ",\"fields\":[";
    for (size_t f = 0; f < r.fields.size(); ++f) {
        const auto & fd = r.fields[f];
        out << (f ? "," : "") << "{\"field_index\":" << f << ",\"candidate_index\":" << fd.winner
            << ",\"path_score\":" << fd.path_score << ",\"scored_nodes\":" << fd.scored_nodes
            << ",\"tree\":" << (fd.tree ? 1 : 0) << ",\"probs\":[";
        for (size_t i = 0; i < fd.probs.size(); ++i) {
            out << (i ? "," : "") << fd.probs[i];
        }
        out << "]}";
    }
    out << "]}\n";
    std::fputs(out.str().c_str(), stdout);
    std::fflush(stdout);
}

static void quiet_log(ggml_log_level level, const char * text, void * /*user*/) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s MODEL N_CPU_MOE --worker N_CTX N_BATCH [STREAM_MIN [GEMM_MIN [THREADS]]]\n"
                             "       %s MODEL N_CPU_MOE @shared|@fresh STATIC CONTEXT FIELD...\n", argv[0], argv[0]);
        return 1;
    }
    const bool worker = std::string(argv[3]) == "--worker";

    common_params params;
    params.model.path   = argv[1];
    params.n_gpu_layers = 999;
    params.fit_params   = false;
    llm_add_n_cpu_ffn_overrides(std::stoi(argv[2]), LLM_FFN_EXPS_REGEX, params.tensor_buft_overrides);
    if (!params.tensor_buft_overrides.empty()) {
        params.tensor_buft_overrides.push_back({ nullptr, nullptr });
    }
    params.n_ctx      = worker ? std::stoi(argv[4]) : 4096;
    params.n_batch    = worker ? std::stoi(argv[5]) : 1024;
    params.n_ubatch   = params.n_batch;
    params.kv_unified = true; // branches share the trunk's cells instead of copying them
    params.n_parallel = env_str("DECIDE_NSEQ").empty() ? 24 : std::stoi(env_str("DECIDE_NSEQ"));
    const int threads = (worker && argc >= 9) ? std::stoi(argv[8]) : 6;
    params.cpuparams.n_threads       = threads;
    params.cpuparams_batch.n_threads = threads;

    llama_backend_init();
    common_init_result_ptr init = common_init_from_params(params);
    if (!init || !init->model() || !init->context()) {
        std::fprintf(stderr, "failed to load model or create context\n");
        return 1;
    }
    llama_log_set(quiet_log, nullptr); // keep stdout parseable once loaded

    try {
        llama_decision::engine eng(init->context(), 0, (int) llama_n_seq_max(init->context()));
        if (!worker) {
            decide_files(eng, std::vector<std::string>(argv + 3, argv + argc));
            return 0;
        }
        std::printf("WORKER_READY\n");
        std::fflush(stdout);
        for (std::string line; std::getline(std::cin, line);) {
            if (line.empty() || line == "QUIT") {
                break;
            }
            std::vector<std::string> files;
            std::stringstream ss(line);
            for (std::string part; std::getline(ss, part, '\t');) {
                files.push_back(part);
            }
            decide_files(eng, files);
            std::printf("WORKER_DONE\n");
            std::fflush(stdout);
        }
    } catch (const std::exception & e) {
        // Fatal request errors exit the worker; the supervisor restarts it with clean state.
        std::printf("ERROR %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
    return 0;
}
