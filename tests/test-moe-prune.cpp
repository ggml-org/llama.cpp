#include "moe-prune.h"
#include "dataset.h"
#include "hard-prune.h"

#include "chat.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>

static void make_fixture(const std::string & path) {
    gguf_context * gguf = gguf_init_empty();
    gguf_set_val_str(gguf, "general.architecture", "gemma4");
    gguf_set_val_u32(gguf, "gemma4.block_count", 30);
    gguf_set_val_u32(gguf, "gemma4.expert_count", 8);
    gguf_set_val_u32(gguf, "gemma4.expert_used_count", 2);

    ggml_context * tensors = ggml_init({ 1024 * 1024, nullptr, false });
    ggml_tensor * router = ggml_new_tensor_2d(tensors, GGML_TYPE_F32, 32, 8);
    ggml_set_name(router, "blk.0.ffn_gate_inp.weight");
    gguf_add_tensor(gguf, router);
    ggml_tensor * gate_up = ggml_new_tensor_3d(tensors, GGML_TYPE_Q4_0, 32, 4, 8);
    ggml_set_name(gate_up, "blk.0.ffn_gate_up_exps.weight");
    gguf_add_tensor(gguf, gate_up);
    ggml_tensor * down = ggml_new_tensor_3d(tensors, GGML_TYPE_Q4_0, 32, 32, 8);
    ggml_set_name(down, "blk.0.ffn_down_exps.weight");
    gguf_add_tensor(gguf, down);

    for (int expert = 0; expert < 8; ++expert) {
        memset((char *) router->data + expert * ggml_row_size(router->type, router->ne[0]), expert, ggml_row_size(router->type, router->ne[0]));
        memset((char *) gate_up->data + expert * ggml_row_size(gate_up->type, gate_up->ne[0]) * gate_up->ne[1], expert, ggml_row_size(gate_up->type, gate_up->ne[0]) * gate_up->ne[1]);
        memset((char *) down->data + expert * ggml_row_size(down->type, down->ne[0]) * down->ne[1], expert, ggml_row_size(down->type, down->ne[0]) * down->ne[1]);
    }
    if (!gguf_write_to_file(gguf, path.c_str(), false)) throw std::runtime_error("failed to write synthetic GGUF fixture");
    ggml_free(tensors);
    gguf_free(gguf);
}

static void expect_failure(const std::function<void()> & fn) {
    bool failed = false;
    try {
        fn();
    } catch (const std::exception &) {
        failed = true;
    }
    if (!failed) throw std::runtime_error("expected failure");
}

static void require(bool value) {
    if (!value) throw std::runtime_error("test assertion failed");
}

static void test_dataset() {
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model * model = llama_model_load_from_file(AIKAR_TEST_GEMMA_VOCAB, params);
    require(model != nullptr);
    common_chat_templates_ptr templates = common_chat_templates_init(model, "");

    const std::string path = "/tmp/llama-moe-prune-test-data.jsonl";
    {
        FILE * file = fopen(path.c_str(), "wb");
        require(file != nullptr);
        fputs("{\"messages\":[{\"role\":\"user\",\"content\":\"Question\"},{\"role\":\"assistant\",\"content\":\"Answer\"}]}\n", file);
        fputs("{\"messages\":[{\"role\":\"user\",\"content\":\"Solve\"},{\"role\":\"assistant\",\"reasoning\":\"Work\",\"content\":\"Final\"}]}\n", file);
        fputs("{\"messages\":[{\"role\":\"user\",\"content\":\"Q1\"},{\"role\":\"assistant\",\"content\":\"A1\"},{\"role\":\"user\",\"content\":\"Q2\"},{\"role\":\"assistant\",\"content\":\"A2\"},{\"role\":\"user\",\"content\":\"Q3\"},{\"role\":\"assistant\",\"content\":\"A3\"},{\"role\":\"user\",\"content\":\"Q4\"},{\"role\":\"assistant\",\"content\":\"A4\"},{\"role\":\"user\",\"content\":\"Q5\"},{\"role\":\"assistant\",\"content\":\"A5\"}]}\n", file);
        fclose(file);
    }
    const aikar_dataset dataset = aikar_dataset_load(path, model, templates.get(), 1);
    const aikar_dataset parallel_dataset = aikar_dataset_load(path, model, templates.get(), 4);
    require(dataset.records.size() == 3);
    require(parallel_dataset.records.size() == dataset.records.size());
    require(parallel_dataset.total_tokens == dataset.total_tokens);
    for (size_t i = 0; i < dataset.records.size(); ++i) {
        require(parallel_dataset.records[i].line == dataset.records[i].line);
        require(parallel_dataset.records[i].tokens == dataset.records[i].tokens);
        require(parallel_dataset.records[i].token_fields == dataset.records[i].token_fields);
    }
    size_t assistant = 0;
    size_t reasoning = 0;
    size_t content = 0;
    for (const aikar_dataset_record & record : dataset.records) {
        for (size_t i = 0; i < record.tokens.size(); ++i) {
            assistant += aikar_token_is_evaluated(record, i, aikar_ppl_mask::ASSISTANT);
            reasoning += aikar_token_is_evaluated(record, i, aikar_ppl_mask::REASONING);
            content += aikar_token_is_evaluated(record, i, aikar_ppl_mask::CONTENT);
        }
    }
    require(assistant > 0 && reasoning > 0 && content > 0);

    {
        FILE * file = fopen(path.c_str(), "wb");
        require(file != nullptr);
        fputs("{\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}]}\n", file);
        fputs("{\"messages\":42}\n", file);
        fclose(file);
    }
    bool line_error = false;
    try {
        (void) aikar_dataset_load(path, model, templates.get());
    } catch (const std::exception & e) {
        line_error = std::string(e.what()).find("line 2") != std::string::npos;
    }
    require(line_error);
    std::remove(path.c_str());
    llama_model_free(model);
}

int main() {
    test_dataset();
    common_moe_prune_model_info model;
    model.architecture = "gemma4";
    model.model_hash = "sha256:model";
    model.expert_tensor_hash = "sha256:experts";
    model.layer_count = 30;
    model.expert_count = 8;
    model.experts_used = 2;
    model.moe_layers = { 1, 3 };

    common_moe_prune_stats stats;
    for (int32_t layer : model.moe_layers) {
        stats[layer].resize(model.expert_count);
        for (int32_t expert = 0; expert < model.expert_count; ++expert) {
            auto & value = stats[layer][expert];
            value.selection_count = 10;
            value.probability_sum = expert + 1;
            value.output_norm_sum = 2 * (expert + 1);
            value.weighted_output_sum = expert + layer;
        }
    }

    const auto profiles = common_moe_prune_make_profiles(
        model, stats, { 0.25, 0.50 }, 0.50, "sha256:dataset", "assistant", "router-output", 100);
    require(profiles.size() == 2);
    for (int32_t layer : model.moe_layers) {
        const auto & small = profiles[0].layers.at(layer).disabled_experts;
        const auto & large = profiles[1].layers.at(layer).disabled_experts;
        require(small.size() == 2);
        require(large.size() == 4);
        for (int32_t expert : small) require(std::find(large.begin(), large.end(), expert) != large.end());
    }

    const std::string path = "/tmp/llama-moe-prune-test-profile.json";
    common_moe_prune_profile_write(profiles[0], path);
    const common_moe_prune_profile loaded = common_moe_prune_profile_load(path);
    common_moe_prune_profile_validate(loaded, model);
    require(loaded.layers.at(1).disabled_experts == profiles[0].layers.at(1).disabled_experts);
    std::remove(path.c_str());

    common_moe_prune_profile invalid = loaded;
    invalid.model_hash = "sha256:wrong";
    expect_failure([&]() { common_moe_prune_profile_validate(invalid, model); });
    invalid = loaded;
    invalid.layers.at(1).disabled_experts = { 0, 0 };
    expect_failure([&]() { common_moe_prune_profile_validate(invalid, model); });
    invalid = loaded;
    invalid.layers.at(1).disabled_experts = { 0, 8 };
    expect_failure([&]() { common_moe_prune_profile_validate(invalid, model); });
    invalid = loaded;
    invalid.layers.at(1).disabled_experts = { 0, 1, 2, 3, 4, 5, 6 };
    invalid.layers.at(3).disabled_experts = invalid.layers.at(1).disabled_experts;
    expect_failure([&]() { common_moe_prune_profile_validate(invalid, model); });

    require(std::abs(stats[1][2].mean_probability() - 0.3) < 1e-12);
    require(std::abs(stats[1][2].mean_output_norm() - 0.6) < 1e-12);
    require(std::abs(stats[1][2].importance() - 0.3) < 1e-12);

    const std::string source_path = "/tmp/llama-moe-prune-test-source.gguf";
    const std::string output_path = "/tmp/llama-moe-prune-test-output.gguf";
    make_fixture(source_path);
    const common_moe_prune_model_info fixture_info = common_moe_prune_inspect_model(source_path);
    require(fixture_info.model_hash == common_moe_prune_sha256_file(source_path));
    common_moe_prune_profile fixture_profile;
    fixture_profile.architecture = fixture_info.architecture;
    fixture_profile.model_hash = fixture_info.model_hash;
    fixture_profile.expert_tensor_hash = fixture_info.expert_tensor_hash;
    fixture_profile.expert_count = fixture_info.expert_count;
    fixture_profile.experts_used = fixture_info.experts_used;
    fixture_profile.layers[0].disabled_experts = { 1, 3 };
    const aikar_hard_prune_report report = aikar_hard_prune_gemma4_q4_0(source_path, fixture_profile, fixture_info, output_path);
    require(report.original_to_new.at(0).at(0) == 0);
    require(report.original_to_new.at(0).at(2) == 1);
    require(report.expert_bytes_removed > 0);

    ggml_context * pruned_tensors = nullptr;
    gguf_context * pruned = gguf_init_from_file(output_path.c_str(), { false, &pruned_tensors });
    require(pruned != nullptr);
    const int64_t expert_key = gguf_find_key(pruned, "gemma4.expert_count");
    require(expert_key >= 0 && gguf_get_val_u32(pruned, expert_key) == 6);
    const int64_t router_id = gguf_find_tensor(pruned, "blk.0.ffn_gate_inp.weight");
    const int64_t expert_id = gguf_find_tensor(pruned, "blk.0.ffn_gate_up_exps.weight");
    require(router_id >= 0 && gguf_get_tensor_ne(pruned, router_id)[1] == 6);
    require(expert_id >= 0 && gguf_get_tensor_ne(pruned, expert_id)[2] == 6);
    const int expected[] = { 0, 2, 4, 5, 6, 7 };
    const ggml_tensor * pruned_router = ggml_get_tensor(pruned_tensors, "blk.0.ffn_gate_inp.weight");
    const ggml_tensor * pruned_experts = ggml_get_tensor(pruned_tensors, "blk.0.ffn_gate_up_exps.weight");
    for (int expert = 0; expert < 6; ++expert) {
        require(*((const unsigned char *) pruned_router->data + expert * ggml_row_size(pruned_router->type, pruned_router->ne[0])) == expected[expert]);
        require(*((const unsigned char *) pruned_experts->data + expert * ggml_row_size(pruned_experts->type, pruned_experts->ne[0]) * pruned_experts->ne[1]) == expected[expert]);
    }
    ggml_free(pruned_tensors);
    gguf_free(pruned);
    std::remove(source_path.c_str());
    std::remove(output_path.c_str());
    std::remove((output_path + ".report.json").c_str());
    return 0;
}
