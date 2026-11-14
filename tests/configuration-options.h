#pragma once

#include "arg.h"

#include <functional>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

struct configuration_option_test {
    std::vector<std::string> args;
    std::string ini;
    llama_example example;
    bool valid;
    std::function<void(const common_params &)> check;
};

static const std::vector<configuration_option_test> configuration_options = {
    {{"-m"}, "model", LLAMA_EXAMPLE_COMMON, false, {}},
    {{"-ngl", "hello"}, "gpu-layers = hello", LLAMA_EXAMPLE_COMMON, false, {}},
    {{"-sm", "hello"}, "split-mode = hello", LLAMA_EXAMPLE_COMMON, false, {}},
    {{"--repeat-last-n", "-1"}, "repeat-last-n = -1", LLAMA_EXAMPLE_COMMON, false, {}},
    {{"--dry-penalty-last-n", "-1"}, "dry-penalty-last-n = -1", LLAMA_EXAMPLE_COMMON, false, {}},
    {{"--repeat-penalty", "0"}, "repeat-penalty = 0", LLAMA_EXAMPLE_COMMON, false, {}},
    {{"--repeat-penalty", "-1"}, "repeat-penalty = -1", LLAMA_EXAMPLE_COMMON, false, {}},
    {{"--repeat-penalty", "nan"}, "repeat-penalty = nan", LLAMA_EXAMPLE_COMMON, false, {}},
    {{"--repeat-penalty", "inf"}, "repeat-penalty = inf", LLAMA_EXAMPLE_COMMON, false, {}},
    {{"--repeat-penalty", "-inf"}, "repeat-penalty = -inf", LLAMA_EXAMPLE_COMMON, false, {}},
    {{"--frequency-penalty", "nan"}, "frequency-penalty = nan", LLAMA_EXAMPLE_COMMON, false, {}},
    {{"--frequency-penalty", "inf"}, "frequency-penalty = inf", LLAMA_EXAMPLE_COMMON, false, {}},
    {{"--frequency-penalty", "-inf"}, "frequency-penalty = -inf", LLAMA_EXAMPLE_COMMON, false, {}},
    {{"--presence-penalty", "nan"}, "presence-penalty = nan", LLAMA_EXAMPLE_COMMON, false, {}},
    {{"--presence-penalty", "inf"}, "presence-penalty = inf", LLAMA_EXAMPLE_COMMON, false, {}},
    {{"--presence-penalty", "-inf"}, "presence-penalty = -inf", LLAMA_EXAMPLE_COMMON, false, {}},
    {{"--draft", "123"}, "draft = 123", LLAMA_EXAMPLE_EMBEDDING, false, {}},
    {{"-lm", "hello"}, "load-mode = hello", LLAMA_EXAMPLE_COMMON, false, {}},
    {{"--spec-synth-len", "3.4x"}, "spec-synth-len = 3.4x", LLAMA_EXAMPLE_SERVER, false, {}},
    {{"-m", "model_file.gguf"}, "model = model_file.gguf", LLAMA_EXAMPLE_COMMON, true,
        [](const common_params & p) { assert(p.model.path == "model_file.gguf"); }},
    {{"-t", "1234"}, "threads = 1234", LLAMA_EXAMPLE_COMMON, true,
        [](const common_params & p) { assert(p.cpuparams.n_threads == 1234); }},
    {{"--verbose"}, "verbose = true", LLAMA_EXAMPLE_COMMON, true,
        [](const common_params & p) { assert(p.verbosity > 1); }},
    {{"-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"}, "model = abc.gguf\npredict = 6789\nbatch-size = 9090", LLAMA_EXAMPLE_COMMON, true,
        [](const common_params & p) {
            assert(p.model.path == "abc.gguf");
            assert(p.n_predict == 6789);
            assert(p.n_batch == 9090);
        }},
    {{"--spec-draft-n-max", "123"}, "spec-draft-n-max = 123", LLAMA_EXAMPLE_SPECULATIVE, true,
        [](const common_params & p) { assert(p.speculative.draft.n_max == 123); }},
    {{"--spec-synth-len", "3.4"}, "spec-synth-len = 3.4", LLAMA_EXAMPLE_SERVER, true,
        [](const common_params & p) { assert(p.speculative.synth_len == 3.4); }},
    {{"--spec-synth-rates", "0.8,0.6,0.2"}, "spec-synth-rates = 0.8,0.6,0.2", LLAMA_EXAMPLE_SERVER, true,
        [](const common_params & p) { assert(p.speculative.synth_rates == std::vector<double>({0.8, 0.6, 0.2})); }},
    {{"-lm", "none"}, "load-mode = none", LLAMA_EXAMPLE_COMMON, true,
        [](const common_params & p) { assert(p.load_mode == LLAMA_LOAD_MODE_NONE); }},
    {{"-lm", "mmap"}, "load-mode = mmap", LLAMA_EXAMPLE_COMMON, true,
        [](const common_params & p) { assert(p.load_mode == LLAMA_LOAD_MODE_MMAP); }},
    {{"-lm", "mlock"}, "load-mode = mlock", LLAMA_EXAMPLE_COMMON, true,
        [](const common_params & p) { assert(p.load_mode == LLAMA_LOAD_MODE_MLOCK); }},
    {{"-lm", "mmap+mlock"}, "load-mode = mmap+mlock", LLAMA_EXAMPLE_COMMON, true,
        [](const common_params & p) { assert(p.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK); }},
    {{"-lm", "dio"}, "load-mode = dio", LLAMA_EXAMPLE_COMMON, true,
        [](const common_params & p) { assert(p.load_mode == LLAMA_LOAD_MODE_DIRECT_IO); }},
    {{"--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"}, "lora = file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf", LLAMA_EXAMPLE_COMMON, true,
        [](const common_params & p) {
            assert(p.lora_adapters.size() == 4);
            assert(p.lora_adapters[0].path == "file1.gguf");
            assert(p.lora_adapters[1].path == "file2,2.gguf");
            assert(p.lora_adapters[2].path == "file3\"3\".gguf");
            assert(p.lora_adapters[3].path == "file4\".gguf");
        }},
    {{"--prompt", "#literal ; also literal # still literal"}, "prompt = #literal ; also literal # still literal", LLAMA_EXAMPLE_COMMON, true,
        [](const common_params & p) { assert(p.prompt == "#literal ; also literal # still literal"); }},
    {{"--dry-sequence-breaker", ";,{,\\},[,],(,),.,/,\\,\\r,\\n,:,\\\",*,}"}, "dry-sequence-breaker = ;,{,\\},[,],(,),.,/,\\,\\r,\\n,:,\\\",*,}", LLAMA_EXAMPLE_COMMON, true,
        [](const common_params & p) { assert(p.sampling.dry_sequence_breakers.back() == ";,{,\\},[,],(,),.,/,\\,\r,\n,:,\",*,}"); }},
    {{"--samplers", "top_k;top_p;temperature"}, "samplers = top_k;top_p;temperature", LLAMA_EXAMPLE_COMMON, true,
        [](const common_params & p) { assert(p.sampling.samplers == std::vector<common_sampler_type>({COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_TEMPERATURE})); }},
};
