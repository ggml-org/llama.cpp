#pragma once

#include <cstdint>
#include <string>

#include "nlohmann/json_fwd.hpp"

// llama-tweak: offline bench cache + runtime routing (Intel backends MVP).

struct llama_tweak_plan {
    std::string ggml_device;
    std::string backend_kind;
    std::string openvino_device;
    int         openvino_stateful = 0;
    bool        openvino_phase_split = false;
    std::string openvino_prefill_device;
    std::string openvino_decode_device;
    std::string sycl_device_selector;
    std::string selected_tag;
    double      expected_tps = 0.0;
    int         resolved_pp  = 0;
    int         resolved_tg  = 0;
};

// Cache file path (CLI override > LLAMA_TWEAK_CACHE > ./llama-tweak-<model-stem>.json).
void llama_tweak_set_cache_path(const std::string & path);
std::string llama_tweak_json_path_for_model(const std::string & model_path);

void llama_tweak_set_active_model(const std::string & model_path);

bool llama_tweak_runtime_enabled();

// Nearest benchmarked pp (and tg) in cache; picks fastest backend for that bucket.
bool llama_tweak_resolve(const std::string & model_path, int pp, int tg, llama_tweak_plan & out);

void llama_tweak_apply_env(const llama_tweak_plan & plan);

int llama_tweak_record_main(int argc, char ** argv);

bool llama_tweak_save_cache_file(const std::string & model_path, const nlohmann::ordered_json & doc);
void llama_tweak_merge_entry(nlohmann::ordered_json & doc, const nlohmann::ordered_json & entry);
nlohmann::ordered_json llama_tweak_load_or_empty(const std::string & model_path);

struct common_params;
bool common_llama_tweak_prepare(common_params & params);
