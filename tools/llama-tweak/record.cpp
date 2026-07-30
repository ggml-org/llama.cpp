#include "llama-tweak.h"

#include "ggml-backend.h"
#include "llama-bench-api.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using json     = nlohmann::ordered_json;

struct tweak_case {
    std::string tag;
    std::string backend_kind;
    std::string ggml_device;
    std::string openvino_device;
    int         openvino_stateful = 0;
    bool        openvino_phase_split = false;
    std::string openvino_prefill;
    std::string openvino_decode;
    std::string sycl_selector;
    std::string vulkan_device;
    std::string ov_cache_subdir;
};

static std::vector<tweak_case> default_cases() {
    const char * igpu = std::getenv("LL_OPENVINO_IGPU_DEVICE");
    const std::string ig = igpu ? igpu : "GPU.0";
    return {
        {"openvino_CPU_sf0", "openvino", "OPENVINO0", "CPU", 0, false, {}, {}, {}, {}, "CPU_sf0"},
        {"openvino_igpu_sf0", "openvino", "OPENVINO0", ig, 0, false, {}, {}, {}, {}, "igpu_sf0"},
        {"openvino_igpu_sf1", "openvino", "OPENVINO0", ig, 1, false, {}, {}, {}, {}, "igpu_sf1"},
        {"openvino_NPU_sf0", "openvino", "OPENVINO0", "NPU", 0, false, {}, {}, {}, {}, "NPU_sf0"},
        {"openvino_split_CPU_igpu", "openvino", "OPENVINO0", "CPU", 1, true, "CPU", ig, {}, {}, "split_CPU_igpu"},
        {"openvino_split_igpu_CPU", "openvino", "OPENVINO0", ig, 1, true, ig, "CPU", {}, {}, "split_igpu_CPU"},
        {"sycl_dgpu_l0", "sycl", "SYCL0", {}, 0, false, {}, {}, "level_zero:0", {}, "sycl_l0_0"},
        {"sycl_igpu_l1", "sycl", "SYCL0", {}, 0, false, {}, {}, "level_zero:1", {}, "sycl_l0_1"},
        {"sycl_cpu", "sycl", "SYCL0", {}, 0, false, {}, {}, "*:cpu", {}, "sycl_cpu"},
        {"vulkan_dgpu", "vulkan", "Vulkan0", {}, 0, false, {}, {}, {}, "Vulkan0", "vk0"},
        {"vulkan_igpu", "vulkan", "Vulkan1", {}, 0, false, {}, {}, {}, "Vulkan1", "vk1"},
    };
}

static void apply_case_env(const tweak_case & c, int pp, int tg) {
    unsetenv("GGML_OPENVINO_PHASE_SPLIT");
    unsetenv("GGML_OPENVINO_PREFILL_DEVICE");
    unsetenv("GGML_OPENVINO_DECODE_DEVICE");
    unsetenv("GGML_OPENVINO_DEVICE");
    unsetenv("GGML_OPENVINO_STATEFUL_EXECUTION");
    unsetenv("ONEAPI_DEVICE_SELECTOR");

    std::string cache = "/tmp/llama_tweak_bench/" + c.ov_cache_subdir + "_" + std::to_string(pp) + "_" +
                        std::to_string(tg);
    setenv("GGML_OPENVINO_CACHE_DIR", cache.c_str(), 1);

    if (c.backend_kind == "openvino") {
        if (c.openvino_phase_split) {
            setenv("GGML_OPENVINO_PHASE_SPLIT", "1", 1);
            setenv("GGML_OPENVINO_PREFILL_DEVICE", c.openvino_prefill.c_str(), 1);
            setenv("GGML_OPENVINO_DECODE_DEVICE", c.openvino_decode.c_str(), 1);
            setenv("GGML_OPENVINO_DEVICE", c.openvino_device.c_str(), 1);
        } else if (!c.openvino_device.empty()) {
            setenv("GGML_OPENVINO_DEVICE", c.openvino_device.c_str(), 1);
        }
        setenv("GGML_OPENVINO_STATEFUL_EXECUTION", c.openvino_stateful ? "1" : "0", 1);
    } else if (c.backend_kind == "sycl" && !c.sycl_selector.empty()) {
        setenv("ONEAPI_DEVICE_SELECTOR", c.sycl_selector.c_str(), 1);
    }
}

static bool run_bench_capture(const std::string & model, int pp, int tg, const tweak_case & c, double & out_tps) {
    apply_case_env(c, pp, tg);

    const std::string dev = c.backend_kind == "vulkan" ? c.vulkan_device : c.ggml_device;

    std::vector<std::string> args_s = {
        "llama-bench", "-m", model, "-r", "1", "--no-warmup", "-p", "0", "-n", "0",
        "-pg", std::to_string(pp) + "," + std::to_string(tg), "-o", "jsonl", "--device", dev, "-ngl", "999"};
    std::vector<char *> argv;
    for (auto & s : args_s) {
        argv.push_back(s.data());
    }
    argv.push_back(nullptr);

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        return false;
    }

    FILE * orig_out = stdout;
    fflush(stdout);
    stdout = fdopen(pipefd[1], "w");
    if (!stdout) {
        close(pipefd[0]);
        close(pipefd[1]);
        stdout = orig_out;
        return false;
    }

    const int rc = llama_bench((int) argv.size() - 1, argv.data());
    fflush(stdout);
    fclose(stdout);
    stdout = orig_out;

    std::string line;
    {
        char    chunk[4096];
        ssize_t n;
        while ((n = read(pipefd[0], chunk, sizeof(chunk))) > 0) {
            line.append(chunk, (size_t) n);
        }
    }
    close(pipefd[0]);

    if (rc != 0) {
        return false;
    }

    const fs::path model_base = fs::path(model).filename();

    std::istringstream stream(line);
    std::string        one;
    while (std::getline(stream, one)) {
        const auto pos = one.find('{');
        if (pos == std::string::npos) {
            continue;
        }
        try {
            json j = json::parse(one.substr(pos));
            if (!j.contains("build_commit") || !j.contains("avg_ts")) {
                continue;
            }
            if (j.value("n_prompt", -1) != pp || j.value("n_gen", -1) != tg) {
                continue;
            }
            const std::string mf = j.value("model_filename", "");
            if (!mf.empty() && fs::path(mf).filename() != model_base) {
                continue;
            }
            out_tps = j.value("avg_ts", 0.0);
            if (out_tps > 0.0 && out_tps < 100000.0) {
                return true;
            }
        } catch (...) {
        }
    }
    return false;
}

static void parse_pg_list(const char * s, std::vector<int> & out) {
    out.clear();
    std::stringstream ss(s);
    std::string       item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(std::atoi(item.c_str()));
        }
    }
}

static double mean_vec(const std::vector<double> & v) {
    if (v.empty()) {
        return 0.0;
    }
    double s = 0;
    for (double x : v) {
        s += x;
    }
    return s / (double) v.size();
}

static double stdev_vec(const std::vector<double> & v) {
    if (v.size() < 2) {
        return 0.0;
    }
    const double m = mean_vec(v);
    double       s = 0;
    for (double x : v) {
        const double d = x - m;
        s += d * d;
    }
    return std::sqrt(s / (double) (v.size() - 1));
}

static void usage() {
    fprintf(stderr,
            "usage: llama-tweak record -m model.gguf [--pp 128,512] [--tg 128] [--runs 3]\n"
            "       [--output path.json]  (default cache: ./llama-tweak-<stem>.json, env LLAMA_TWEAK_CACHE)\n"
            "       llama-tweak explain -m model.gguf [--pp N] [--tg N]\n");
}

int llama_tweak_record_main(int argc, char ** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    std::string cmd = argv[1];
    std::string model;
    std::string pp_list = "512";
    std::string tg_val  = "128";
    std::string out_path;
    int         runs    = 3;

    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            model = argv[++i];
        } else if (!strcmp(argv[i], "--pp") && i + 1 < argc) {
            pp_list = argv[++i];
        } else if (!strcmp(argv[i], "--tg") && i + 1 < argc) {
            tg_val = argv[++i];
        } else if ((!strcmp(argv[i], "--output") || !strcmp(argv[i], "-o")) && i + 1 < argc) {
            out_path = argv[++i];
        } else if (!strcmp(argv[i], "--runs") && i + 1 < argc) {
            runs = std::max(1, std::atoi(argv[++i]));
        }
    }

    if (!out_path.empty()) {
        llama_tweak_set_cache_path(out_path);
    } else if (const char * env = std::getenv("LLAMA_TWEAK_CACHE")) {
        if (env[0] != '\0') {
            llama_tweak_set_cache_path(env);
        }
    }

    if (cmd == "explain") {
        if (model.empty()) {
            usage();
            return 1;
        }
        int pp = std::getenv("LLAMA_TWEAK_PP") ? std::atoi(std::getenv("LLAMA_TWEAK_PP")) : 512;
        int tg = std::getenv("LLAMA_TWEAK_TG") ? std::atoi(std::getenv("LLAMA_TWEAK_TG")) : 128;
        for (int j = 2; j < argc; ++j) {
            if (!strcmp(argv[j], "--pp") && j + 1 < argc) {
                pp = std::atoi(argv[++j]);
            } else if (!strcmp(argv[j], "--tg") && j + 1 < argc) {
                tg = std::atoi(argv[++j]);
            }
        }
        llama_tweak_plan plan;
        if (!llama_tweak_resolve(model, pp, tg, plan)) {
            fprintf(stderr, "llama-tweak: no cache for %s (pp=%d tg=%d). Run: llama-tweak record -m ...\n", model.c_str(),
                    pp, tg);
            return 1;
        }
        fprintf(stderr, "best: %s backend=%s ggml_dev=%s cache_pp=%d cache_tg=%d expected=%.2f tok/s\n",
                plan.selected_tag.c_str(), plan.backend_kind.c_str(), plan.ggml_device.c_str(), plan.resolved_pp,
                plan.resolved_tg, plan.expected_tps);
        return 0;
    }

    if (cmd != "record" || model.empty()) {
        usage();
        return 1;
    }

    std::vector<int> pps;
    parse_pg_list(pp_list.c_str(), pps);
    const int tg = std::max(0, std::atoi(tg_val.c_str()));
    if (pps.empty()) {
        pps.push_back(512);
    }

    json doc = llama_tweak_load_or_empty(model);

    for (int pp : pps) {
        for (const auto & c : default_cases()) {
                if (c.tag == "openvino_NPU_sf0") {
                    double    probe = 0;
                    tweak_case pc   = c;
                    if (!run_bench_capture(model, 8, 0, pc, probe)) {
                        fprintf(stderr, "skip %s (probe failed)\n", c.tag.c_str());
                        continue;
                    }
                }
                std::vector<double> samples;
                fprintf(stderr, "=== %s pp=%d tg=%d (%d runs) ===\n", c.tag.c_str(), pp, tg, runs);
                for (int r = 0; r < runs; ++r) {
                    double tps = 0;
                    if (!run_bench_capture(model, pp, tg, c, tps)) {
                        fprintf(stderr, "  run %d: FAIL\n", r + 1);
                        continue;
                    }
                    samples.push_back(tps);
                    fprintf(stderr, "  run %d: %.2f tok/s\n", r + 1, tps);
                }
                if (samples.empty()) {
                    continue;
                }
                json e;
                e["tag"]                   = c.tag;
                e["pp"]                    = pp;
                e["tg"]                    = tg;
                e["backend_kind"]          = c.backend_kind;
                e["ggml_device"]           = c.backend_kind == "vulkan" ? c.vulkan_device : c.ggml_device;
                e["openvino_device"]       = c.openvino_device;
                e["openvino_stateful"]     = c.openvino_stateful;
                e["openvino_phase_split"]    = c.openvino_phase_split;
                e["openvino_prefill_device"] = c.openvino_prefill;
                e["openvino_decode_device"]  = c.openvino_decode;
                e["sycl_device_selector"]    = c.sycl_selector;
                e["mean_tps"]              = mean_vec(samples);
                e["stddev_tps"]            = stdev_vec(samples);
                e["runs"]                  = (int) samples.size();
                llama_tweak_merge_entry(doc, e);
        }
    }

    if (!llama_tweak_save_cache_file(model, doc)) {
        fprintf(stderr, "failed to write %s\n", llama_tweak_json_path_for_model(model).c_str());
        return 1;
    }
    fprintf(stderr, "wrote %s\n", llama_tweak_json_path_for_model(model).c_str());
    return 0;
}
