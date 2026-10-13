#include "arg.h"
#include "common.h"
#include "download.h"
#include "llama.h"
#include "speculative.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#undef NDEBUG
#include <cassert>

static void test(void) {
    common_params params;

    {
        common_params dflash;
        dflash.shared_device_memory_bytes=1536*1024*1024ULL;
        dflash.speculative.types={COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
        dflash.speculative.draft.mparams.path="draft.gguf";
        dflash.speculative.draft.n_max=7;
        dflash.n_parallel=1; dflash.n_batch=dflash.n_ubatch=256;
        dflash.fit_params=false; dflash.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        dflash.cache_type_k=GGML_TYPE_Q8_0; dflash.cache_type_v=GGML_TYPE_Q4_0;
        assert(common_params_uses_streamed_dflash(dflash));
        assert(!common_params_uses_streamed_mtp(dflash));
        assert(!common_streamed_dflash_config_error(dflash));
        const ggml_type cache_types[]={GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,
            GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_IQ4_NL,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1};
        for (auto key : cache_types) for (auto value : cache_types) {
            auto configured=dflash;
            configured.cache_type_k=key; configured.cache_type_v=value;
            configured.speculative.draft.cache_type_k=value; configured.speculative.draft.cache_type_v=key;
            assert(!common_streamed_dflash_config_error(configured));
        }
        for (auto limits : {std::pair{128,64}, {256,128}, {256,256}, {512,128}, {512,256},
                {512,512}, {1024,256}, {1024,512}, {1024,1024}, {513,129}, {128,256}, {256,0}}) {
            auto configured=dflash;
            configured.n_batch=limits.first; configured.n_ubatch=limits.second;
            assert(!common_streamed_dflash_config_error(configured));
        }
        for (int change=0; change<5; ++change) {
            auto invalid=dflash;
            switch (change) {
                case 0: invalid.n_batch=0; break;
                case 1: invalid.n_ubatch=-1; break;
                case 2: invalid.cache_type_k=GGML_TYPE_I32; break;
                case 3: invalid.speculative.draft.cache_type_v=GGML_TYPE_Q2_K; break;
                case 4: invalid.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED; break;
            }
            assert(common_streamed_dflash_config_error(invalid));
        }
        assert(common_context_params_to_llama(dflash).n_rs_seq == 7);
        assert(common_kv_stream_auxiliary_layers(dflash,1) == 0);
        const auto draft=common_base_params_to_speculative(dflash);
        assert(draft.cache_type_k == GGML_TYPE_F16 && draft.cache_type_v == GGML_TYPE_F16);
        assert(!draft.shared_device_memory_bytes && !draft.kv_stream_pool_bytes && !draft.kv_stream_auxiliary_layers);
        for (int change=0;change<7;++change) {
            auto invalid=dflash;
            switch (change) {
                case 0: invalid.n_parallel=2; break;
                case 1: invalid.speculative.draft.n_max=8; break;
                case 2: invalid.kv_stream_auxiliary_layers=1; break;
                case 3: invalid.fit_params=true; break;
                case 4: invalid.mmproj.path="vision.gguf"; break;
                case 5: invalid.n_ubatch=4; break;
                case 6: invalid.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP); break;
            }
            assert(common_streamed_dflash_config_error(invalid));
        }
    }

    {
        common_params automatic;
        automatic.shared_device_memory_bytes = 2240*1024*1024ULL;
        automatic.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        automatic.speculative.draft.n_max = 3;
        automatic.cache_type_k = GGML_TYPE_Q8_0;
        automatic.cache_type_v = GGML_TYPE_Q4_0;
        assert(common_params_uses_streamed_mtp(automatic));
        assert(common_kv_stream_auxiliary_layers(automatic, 1) == 1);
        assert(common_kv_stream_auxiliary_layers(automatic, 3) == 3);
        assert(automatic.kv_stream_auxiliary_layers == 0);
        const auto automatic_draft = common_base_params_to_speculative(automatic);
        assert(automatic_draft.cache_type_k == GGML_TYPE_Q8_0);
        assert(automatic_draft.cache_type_v == GGML_TYPE_Q4_0);
        assert(automatic_draft.kv_stream_auxiliary_layers == 0);
        assert(automatic_draft.kv_stream_pool_bytes == 0);
        assert(automatic_draft.shared_device_memory_bytes == 0);
        automatic.kv_stream_pool_bytes = automatic.shared_device_memory_bytes;
        automatic.shared_device_memory_bytes = 0;
        assert(common_kv_stream_auxiliary_layers(automatic, 3) == 3);
        automatic.kv_stream_auxiliary_layers = 1;
        assert(common_kv_stream_auxiliary_layers(automatic, 1) == 1);
        auto rejects = [](const common_params & params, int32_t count, const std::string & message) {
            try {
                common_kv_stream_auxiliary_layers(params, count);
            } catch (const std::invalid_argument & error) {
                return std::string(error.what()).find(message) != std::string::npos;
            }
            return false;
        };
        assert(rejects(automatic, 3, "does not match"));
        automatic.kv_stream_auxiliary_layers = 0;
        assert(rejects(automatic, 0, "no MTP"));
        assert(rejects(automatic, -1, "no MTP"));
        automatic.speculative.draft.mparams.path = "standalone.gguf";
        assert(rejects(automatic, 1, "embedded"));
        automatic.speculative.draft.mparams.path.clear();
        automatic.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3);
        assert(rejects(automatic, 1, "only draft-mtp"));
        automatic.speculative.types = {COMMON_SPECULATIVE_TYPE_NONE};
        assert(!common_params_uses_streamed_mtp(automatic));
        assert(common_kv_stream_auxiliary_layers(automatic, 3) == 0);
        automatic.kv_stream_auxiliary_layers = 1;
        assert(rejects(automatic, 1, "requires"));
        automatic.kv_stream_auxiliary_layers = 0;
        automatic.kv_stream_pool_bytes = 0;
        automatic.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        assert(!common_params_uses_streamed_mtp(automatic));
        assert(common_kv_stream_auxiliary_layers(automatic, 3) == 0);
    }

    auto assert_output_limits = [](int32_t n_batch, int32_t n_parallel, int32_t n_draft,
                                   int32_t total, int32_t per_seq) {
        const auto limits = common_speculative_get_output_limits(n_batch, n_parallel, n_draft);
        assert(limits.total == total);
        assert(limits.per_seq == per_seq);
    };

    assert_output_limits(16, 2,  3, 8, 4);
    assert_output_limits(16, 2, -1, 2, 1);
    assert_output_limits( 6, 2,  3, 6, 4);
    assert_output_limits( 2, 1,  3, 2, 2);
    assert_output_limits(
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max());

    {
        common_params_speculative spec;
        spec.synth_len = 3.4;

        auto assert_invalid = [](const common_params_speculative & value, int32_t n_max) {
            try {
                common_speculative_synth_rates_resolve(&value, n_max);
                assert(false);
            } catch (const std::invalid_argument &) {
            }
        };

        const auto rates = common_speculative_synth_rates_resolve(&spec, 4);
        assert(rates.size() == 4);
        assert(std::abs(rates[0] - 0.80581) < 1e-5);
        assert(std::abs(rates[1] - 0.64933) < 1e-5);
        assert(std::abs(rates[2] - 0.52323) < 1e-5);
        assert(std::abs(rates[3] - 0.42163) < 1e-5);
        assert(std::abs(1.0 + rates[0] + rates[1] + rates[2] + rates[3] - 3.4) < 1e-8);

        spec.synth_len = 1.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({0.0, 0.0, 0.0, 0.0}));

        spec.synth_len = 5.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({1.0, 1.0, 1.0, 1.0}));

        spec.synth_len = 5.1;
        assert_invalid(spec, 4);

        spec.synth_len = std::numeric_limits<double>::quiet_NaN();
        assert_invalid(spec, 4);

        spec.synth_len = 0.0;
        assert_invalid(spec, 4);

        spec.synth_len = -1.0;
        spec.synth_rates = {0.8, 0.6, 0.4};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        assert(common_speculative_synth_rates_resolve(&spec, 4) == spec.synth_rates);

        spec.synth_rates = {0.8, 0.9, 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, std::numeric_limits<double>::quiet_NaN(), 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, -0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        spec.synth_len = 3.0;
        assert_invalid(spec, 4);
    }

    {
        common_params base;
        base.n_parallel = 4;
        base.n_outputs_max_per_seq = 8;

        const auto draft = common_base_params_to_speculative(base);
        assert(draft.n_outputs_max == 4);
        assert(draft.n_outputs_max_per_seq == 1);
    }
    {
        common_params target;
        target.shared_device_memory_bytes = 4096*1024*1024ULL;
        target.kv_stream_auxiliary_layers = 1;
        target.cache_type_k = GGML_TYPE_Q8_0;
        target.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        target.speculative.draft.n_max = 3;
        assert(common_context_params_to_llama(target).n_rs_seq == 3);
        target.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3};
        assert(common_context_params_to_llama(target).n_rs_seq == 0);
        target.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        target.cache_type_v = GGML_TYPE_Q4_0;
        const auto draft = common_base_params_to_speculative(target);
        assert(draft.kv_stream_auxiliary_layers == 0);
        assert(draft.kv_stream_pool_bytes == 0);
        assert(draft.shared_device_memory_bytes == 0);
        assert(draft.cache_type_k == GGML_TYPE_Q8_0);
        assert(draft.cache_type_v == GGML_TYPE_Q4_0);
    }


    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            common_params_add_preset_options(ctx_arg.options);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.get_args()) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg.c_str());
                        exit(1);
                    }
                }
                // check for env var duplications
                for (const auto & env : opt.get_env()) {
                    if (seen_env_vars.find(env) == seen_env_vars.end()) {
                        seen_env_vars.insert(env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", env.c_str());
                        exit(1);
                    }
                }

                // exclude spec args from this check
                // ref: https://github.com/ggml-org/llama.cpp/pull/22397
                const bool skip = opt.is_spec;

                // ensure shorter argument precedes longer argument
                if (!skip && opt.args.size() > 1) {
                    const std::string first(opt.args.front());
                    const std::string last(opt.args.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }

                // same check for negated arguments
                if (opt.args_neg.size() > 1) {
                    const std::string first(opt.args_neg.front());
                    const std::string last(opt.args_neg.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter negated argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    {
        common_params shared;
        argv = {"binary_name","--shared-device-memory-mib","4096"};
        assert(common_params_parse(argv.size(),list_str_to_char(argv).data(),shared,LLAMA_EXAMPLE_SERVER));
        assert(shared.shared_device_memory_bytes == size_t(4096)*1024*1024);
        assert(shared.kv_stream_pool_bytes == 0);

        common_params alias;
        argv = {"binary_name","--kv-stream-arena-mib","3072"};
        assert(common_params_parse(argv.size(),list_str_to_char(argv).data(),alias,LLAMA_EXAMPLE_SERVER));
        assert(alias.shared_device_memory_bytes == size_t(3072)*1024*1024);
        assert(alias.kv_stream_pool_bytes == 0);
        common_params mtp;
        argv = {"binary_name","--shared-device-memory-mib","4096",
            "--kv-stream-auxiliary-layers","1"};
        assert(common_params_parse(argv.size(),list_str_to_char(argv).data(),mtp,LLAMA_EXAMPLE_SERVER));
        assert(mtp.kv_stream_auxiliary_layers == 1);
        common_params rollback;
        argv = {"binary_name","--shared-device-memory-mib","4096",
            "--kv-stream-auxiliary-layers","1","--spec-type","draft-mtp",
            "--spec-draft-n-max","3"};
        assert(common_params_parse(argv.size(),list_str_to_char(argv).data(),rollback,LLAMA_EXAMPLE_SERVER));
        assert(common_context_params_to_llama(rollback).n_rs_seq == 3);
        argv.push_back("--no-kv-stream-rs-rollback");
        common_params full_checkpoint;
        assert(common_params_parse(argv.size(),list_str_to_char(argv).data(),full_checkpoint,LLAMA_EXAMPLE_SERVER));
        assert(common_context_params_to_llama(full_checkpoint).n_rs_seq == 0);
        common_params automatic_mtp;
        argv = {"binary_name","--kv-stream-arena-mib","2240",
            "--spec-type","draft-mtp","--spec-draft-n-max","3"};
        assert(common_params_parse(argv.size(),list_str_to_char(argv).data(),automatic_mtp,LLAMA_EXAMPLE_SERVER));
        assert(automatic_mtp.kv_stream_auxiliary_layers == 0);
        assert(common_kv_stream_auxiliary_layers(automatic_mtp,1) == 1);
        assert(common_context_params_to_llama(automatic_mtp).n_rs_seq == 3);
        argv.push_back("--no-kv-stream-rs-rollback");
        common_params automatic_full_checkpoint;
        assert(common_params_parse(argv.size(),list_str_to_char(argv).data(),automatic_full_checkpoint,LLAMA_EXAMPLE_SERVER));
        assert(common_context_params_to_llama(automatic_full_checkpoint).n_rs_seq == 0);
        common_params multiple_heads;
        argv = {"binary_name","--kv-stream-arena-mib","2240",
            "--kv-stream-auxiliary-layers","3","--spec-type","draft-mtp"};
        assert(common_params_parse(argv.size(),list_str_to_char(argv).data(),multiple_heads,LLAMA_EXAMPLE_SERVER));
        assert(common_kv_stream_auxiliary_layers(multiple_heads,3) == 3);
        for (const std::vector<std::string> & invalid_rollback : {
                std::vector<std::string>{"binary_name","--shared-device-memory-mib","4096",
                    "--kv-stream-auxiliary-layers","1","--spec-type","draft-mtp",
                    "--spec-draft-n-max","4"},
                std::vector<std::string>{"binary_name","--kv-stream-arena-mib","2240",
                    "--spec-type","draft-mtp","--spec-draft-n-max","4"},
                std::vector<std::string>{"binary_name","--spec-type","draft-mtp",
                    "--no-kv-stream-rs-rollback"}}) {
            common_params rejected;
            argv = invalid_rollback;
            assert(!common_params_parse(argv.size(),list_str_to_char(argv).data(),rejected,LLAMA_EXAMPLE_SERVER));
        }
        for (const std::vector<std::string> & invalid_aux : {
                std::vector<std::string>{"binary_name","--kv-stream-auxiliary-layers","1"},
                std::vector<std::string>{"binary_name","--shared-device-memory-mib","4096",
                    "--kv-stream-auxiliary-layers","-1"}}) {
            common_params rejected;
            argv = invalid_aux;
            assert(!common_params_parse(argv.size(),list_str_to_char(argv).data(),rejected,LLAMA_EXAMPLE_SERVER));
        }


        common_params legacy;
        argv = {"binary_name","--kv-stream-pool-mib","64"};
        assert(common_params_parse(argv.size(),list_str_to_char(argv).data(),legacy,LLAMA_EXAMPLE_SERVER));
        assert(legacy.kv_stream_pool_bytes == size_t(64)*1024*1024);
        assert(legacy.shared_device_memory_bytes == 0);

        for (const std::vector<std::string> & conflicting : {
                std::vector<std::string>{"binary_name","--kv-stream-pool-mib","64",
                    "--shared-device-memory-mib","3072"},
                std::vector<std::string>{"binary_name","--kv-stream-arena-mib","3072",
                    "--kv-stream-pool-mib","64"}}) {
            common_params rejected;
            argv = conflicting;
            assert(!common_params_parse(
                argv.size(),list_str_to_char(argv).data(),rejected,LLAMA_EXAMPLE_SERVER));
            assert(rejected.kv_stream_pool_bytes == 0);
            assert(rejected.shared_device_memory_bytes == 0);
        }

        common_params disabled;
        argv = {"binary_name","--kv-stream-pool-mib","0",
            "--shared-device-memory-mib","0"};
        assert(common_params_parse(
            argv.size(),list_str_to_char(argv).data(),disabled,LLAMA_EXAMPLE_SERVER));
        assert(disabled.kv_stream_pool_bytes == 0);
        assert(disabled.shared_device_memory_bytes == 0);

        common_params invalid;
        argv = {"binary_name","--shared-device-memory-mib","-1"};
        assert(!common_params_parse(
            argv.size(),list_str_to_char(argv).data(),invalid,LLAMA_EXAMPLE_SERVER));
    }

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    {
        common_params penalty_params;
        assert(penalty_params.sampling.penalty_last_n == 64);
        assert(penalty_params.sampling.dry_penalty_last_n == 64);

        argv = {"binary_name", "--repeat-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--dry-penalty-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "0"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "nan"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        const char * penalty_options[] = {"--frequency-penalty", "--presence-penalty"};
        const char * nonfinite_values[] = {"nan", "inf", "-inf"};
        for (const char * option : penalty_options) {
            for (const char * value : nonfinite_values) {
                argv = {"binary_name", option, value};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));
            }
        }
    }

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));

    argv = {"binary_name", "-lm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--spec-draft-n-max", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_max == 123);

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_len == 3.4);
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-rates", "0.8,0.6,0.2"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_rates == std::vector<double>({0.8, 0.6, 0.2}));
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4x"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
    }

    argv = {"binary_name", "-lm", "none"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);

    argv = {"binary_name", "-lm", "mmap"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    argv = {"binary_name", "-lm", "mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    argv = {"binary_name", "-lm", "mmap+mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    argv = {"binary_name", "-lm", "dio"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    // multi-value args (CSV)
    argv = {"binary_name", "--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.lora_adapters.size() == 4);
    assert(params.lora_adapters[0].path == "file1.gguf");
    assert(params.lora_adapters[1].path == "file2,2.gguf");
    assert(params.lora_adapters[2].path == "file3\"3\".gguf");
    assert(params.lora_adapters[3].path == "file4\".gguf");

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);

    setenv("LLAMA_ARG_LOAD_MODE", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_LOAD_MODE", "mmap", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    setenv("LLAMA_ARG_LOAD_MODE", "mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "mmap+mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "dio", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    printf("test-arg-parser: test negated environment variables\n\n");

    setenv("LLAMA_ARG_LOAD_MODE", "none", true);
    setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);
    assert(params.no_perf == true);

    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);
#endif // _WIN32

    printf("test-arg-parser: test download functions\n\n");
    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    {
        printf("test-arg-parser: test good URL\n\n");
        auto res = common_remote_get_content(GOOD_URL, {});
        assert(res.first == 200);
        assert(res.second.size() > 0);
        std::string str(res.second.data(), res.second.size());
        assert(str.find("llama.cpp") != std::string::npos);
    }

    {
        printf("test-arg-parser: test bad URL\n\n");
        auto res = common_remote_get_content(BAD_URL, {});
        assert(res.first == 404);
    }

    {
        printf("test-arg-parser: test max size error\n");
        common_remote_params params;
        params.max_size = 1;
        try {
            common_remote_get_content(GOOD_URL, params);
            assert(false && "it should throw an error");
        } catch (std::exception & e) {
            printf("  expected error: %s\n\n", e.what());
        }
    }

    printf("test-arg-parser: all tests OK\n\n");
}

int main(void) {
    try {
        test();
    } catch (std::exception & e) {
        fprintf(stderr, "test-arg-parser: exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
