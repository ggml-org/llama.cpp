#include "server-schema.h"

#include "json-schema-to-grammar.h"

namespace server_schema {

//
// llama.cpp-specific completion schema
//

std::vector<std::unique_ptr<field>> make_llama_cmpl_schema(const common_params & params_base, task_params & params) {
    std::vector<std::unique_ptr<field>> fields;
    auto add = [&](field * f) {
        fields.emplace_back(f);
    };

    add((new field_bool("verbose", params.verbose))
        ->set_desc("Include __verbose field in the response with additional debug information"));

    add((new field_bool("timings_per_token", params.timings_per_token))
        ->set_desc("Include prompt processing and text generation speed information in each response"));

    add((new field_bool("stream", params.stream))
        ->set_desc("Allows receiving each predicted token in real-time instead of waiting for the completion to finish"));

    add((new field_nested("stream_options"))
        ->add_subfield((new field_bool("include_usage", params.include_usage))
            ->set_desc("Whether to include usage information in the stream"))
        ->set_desc("Additional options for streaming responses"));

    add((new field_bool("cache_prompt", params.cache_prompt))
        ->set_desc("Re-use KV cache from a previous request if possible. This way the common prefix does not have to be re-processed, only the suffix that differs between the requests"));

    add((new field_bool("return_tokens", params.return_tokens))
        ->set_desc("Return the raw generated token ids in the `tokens` field"));

    add((new field_bool("return_progress", params.return_progress))
        ->set_desc("Include prompt processing progress events in stream mode"));

    add((new field_num("sse_ping_interval", params.sse_ping_interval))
        ->set_hard_limits(-1, INT32_MAX)
        ->set_desc("Interval in seconds between SSE comment pings emitted while the stream stays silent, -1 disables pings"));

    add((new field_num("n_predict", params.n_predict))
        ->set_hard_limits(-1, INT32_MAX)
        ->add_alias("max_completion_tokens")
        ->add_alias("max_tokens")
        ->set_desc("Set the maximum number of tokens to predict. When 0, no tokens will be generated but the prompt is evaluated into the cache"));

    add((new field_num("n_indent", params.n_indent))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Specify the minimum line indentation for the generated text in number of whitespace characters. Useful for code completion tasks"));

    add((new field_num("n_keep", params.n_keep))
        ->set_hard_limits(-1, INT32_MAX)
        ->set_desc("Specify the number of tokens from the initial prompt to retain when context size is exceeded. Use -1 to retain all tokens from the prompt"));

    add((new field_num("n_discard", params.n_discard))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Number of tokens after n_keep that may be discarded when shifting context (0 = half context)"));

    add((new field_num("n_cmpl", params.n_cmpl))
        ->set_hard_limits(1, params_base.n_parallel)
        ->add_alias("n") // alias "n" as fallback (OpenAI completions API)
        ->set_desc("Number of completions to generate. If the input has multiple prompts, total outputs will be N prompts times n_cmpl"));

    add((new field_num("n_cache_reuse", params.n_cache_reuse))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Min chunk size to attempt reusing from the cache via KV shifting. See --cache-reuse arg"));

    // TODO: implement t_max_prompt_ms
    // add((new field_num("t_max_prompt_ms", params.t_max_prompt_ms))

    add((new field_num("t_max_predict_ms", params.t_max_predict_ms))
        ->set_hard_limits(-1, std::numeric_limits<int64_t>::max())
        ->set_desc("Set a time limit in milliseconds for the prediction phase. The timeout triggers if generation exceeds this time (measured since the first token) and a newline has been generated. Useful for FIM applications"));

    add((new field_json("response_fields"))
        ->set_desc("A list of response fields to return. Missing fields are omitted without error. Fields with a slash are unnested (e.g. generation_settings/n_predict moves n_predict to the root)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.response_fields = json_value(data, "response_fields", std::vector<std::string>());
        }));


    //
    // Sampling params
    //

    add((new field_num("top_k", params.sampling.top_k))
        ->set_limits(0, INT32_MAX)
        ->set_desc("Limit the next token selection to the K most probable tokens (0 = disabled)"));

    add((new field_num("top_p", params.sampling.top_p))
        ->set_limits(0.0f, 1.0f)
        ->set_desc("Limit the next token selection to a subset of tokens with cumulative probability above threshold P (1.0 = disabled)"));

    add((new field_num("min_p", params.sampling.min_p))
        ->set_limits(0.0f, 1.0f)
        ->set_desc("The minimum probability for a token to be considered, relative to the probability of the most likely token (0 = disabled)"));

    add((new field_num("top_n_sigma", params.sampling.top_n_sigma))
        ->set_desc("Keep tokens within n standard deviations of the top token logit (< 0 = disabled)"));

    add((new field_num("xtc_probability", params.sampling.xtc_probability))
        ->set_limits(0.0f, 1.0f)
        ->set_desc("Set the chance for token removal via XTC sampler (0 = disabled)"));

    add((new field_num("xtc_threshold", params.sampling.xtc_threshold))
        ->set_limits(0.0f, 1.0f)
        ->set_desc("Set a minimum probability threshold for tokens to be removed via XTC sampler (> 0.5 disables XTC)"));

    add((new field_num("typical_p", params.sampling.typ_p))
        // ->set_limits(0.0f, 1.0f) // what's the valid range?
        ->set_desc("Enable locally typical sampling with parameter p (1.0 = disabled)"));

    add((new field_num("temperature", params.sampling.temp))
        ->set_limits(0.0f, std::numeric_limits<float>::infinity())
        ->set_desc("Adjust the randomness of the generated text (0 = greedy)"));

    add((new field_num("dynatemp_range", params.sampling.dynatemp_range))
        ->set_desc("Dynamic temperature range. The final temperature will be in [temperature - range, temperature + range] (0 = disabled)"));

    add((new field_num("dynatemp_exponent", params.sampling.dynatemp_exponent))
        ->set_desc("Dynamic temperature exponent, controls how entropy maps to temperature"));

    add((new field_num("repeat_last_n", params.sampling.penalty_last_n))
        ->set_hard_limits(-1, INT32_MAX)
        ->set_desc("Last n tokens to consider for penalizing repetition (0 = disabled, -1 = ctx-size)"));

    add((new field_num("repeat_penalty", params.sampling.penalty_repeat))
        ->set_desc("Control the repetition of token sequences in the generated text (1.0 = disabled)"));

    add((new field_num("frequency_penalty", params.sampling.penalty_freq))
        ->set_desc("Repeat alpha frequency penalty (0 = disabled)"));

    add((new field_num("presence_penalty", params.sampling.penalty_present))
        ->set_desc("Repeat alpha presence penalty (0 = disabled)"));

    add((new field_num("dry_multiplier", params.sampling.dry_multiplier))
        ->set_desc("Set the DRY (Don't Repeat Yourself) repetition penalty multiplier (0 = disabled)"));

    add((new field_num("dry_base", params.sampling.dry_base))
        ->set_desc("Set the DRY repetition penalty base value (must be >= 1.0, any values < 1.0 will be replaced with the default value)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            float v = data.at("dry_base").get<float>();
            ctx.params.sampling.dry_base = (v < 1.0f) ? params_base.sampling.dry_base : v;
        }));

    add((new field_num("dry_allowed_length", params.sampling.dry_allowed_length))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Tokens that extend repetition beyond this length receive exponentially increasing penalty: multiplier * base ^ (sequence_length - allowed_length)"));

    add((new field_num("dry_penalty_last_n", params.sampling.dry_penalty_last_n))
        ->set_hard_limits(-1, INT32_MAX)
        ->set_desc("How many tokens to scan for repetitions (0 = disabled, -1 = context size)"));

    add((new field_num("mirostat", params.sampling.mirostat))
        ->set_limits(0, 2)
        ->set_desc("Enable Mirostat sampling, controlling perplexity during text generation (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)"));

    add((new field_num("mirostat_tau", params.sampling.mirostat_tau))
        ->set_desc("Set the Mirostat target entropy, parameter tau"));

    add((new field_num("mirostat_eta", params.sampling.mirostat_eta))
        ->set_desc("Set the Mirostat learning rate, parameter eta"));

    add((new field_num("adaptive_target", params.sampling.adaptive_target))
        ->set_limits(-std::numeric_limits<float>::max(), 1.0f)
        ->set_desc("Adaptive sampling target entropy (valid range 0.0 to 1.0; negative = disabled)"));

    add((new field_num("adaptive_decay", params.sampling.adaptive_decay))
        ->set_hard_limits(0.0f, 0.99f)
        ->set_desc("EMA decay for adaptive sampling; history approximates 1/(1-decay) tokens"));

    // seed is uint32_t; field_num uses int32_t so use a handler
    add((new field_num("seed", params.sampling.seed))
        ->set_desc("Set the random number generator (RNG) seed (-1 = random)"));

    add((new field_num("n_probs", params.sampling.n_probs))
        ->add_alias("logprobs") // use "logprobs" if "n_probs" wasn't provided
        ->set_desc("If greater than 0, output the probabilities of top N tokens for each generated token"));

    add((new field_num("min_keep", params.sampling.min_keep))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("If greater than 0, force samplers to return at least N possible tokens"));

    add((new field_bool("backend_sampling", params.sampling.backend_sampling))
        ->set_desc("Use backend sampling instead of llama.cpp sampling"));

    add((new field_bool("post_sampling_probs", params.post_sampling_probs))
        ->set_desc("Return probabilities of top n_probs tokens after applying the sampling chain"));

    //
    // Speculative decoding params
    //

    // TODO: to keep things simple, we disable speculative parameter adjustments for now
#if 0
    // TODO: for now, be able to adjust only the draft-model based speculative parameters
    add((new field_num("speculative.n_max", params.speculative.draft.n_max))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Maximum number of tokens to draft during speculative decoding"));

    add((new field_num("speculative.n_min", params.speculative.draft.n_min))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Minimum number of draft tokens to use for speculative decoding");

    add((new field_num("speculative.p_min", params.speculative.draft.p_min))
        ->set_hard_limits(0.0f, 1.0f)
        ->set_desc("Minimum speculative decoding probability for draft tokens (0 = greedy)"));

    add((new field_str("speculative.type"))
        ->set_desc("Speculative decoding method (for debugging and research purposes)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.speculative.types = { common_speculative_type_from_name(data.at("speculative.type").get<std::string>()) };
        }));

    add((new field_num("speculative.ngram_size_n", params.speculative.ngram_simple.size_n))
        ->set_desc("Ngram size for lookup in ngram-based speculative decoding"));

    add((new field_num("speculative.ngram_size_m", params.speculative.ngram_simple.size_m))
        ->set_desc("Mgram size for speculative tokens in ngram-based speculative decoding"));

    add((new field_num("speculative.ngram_min_hits", params.speculative.ngram_simple.min_hits))
        ->set_desc("Minimum hits at ngram lookup for mgram to be proposed"));
#endif

    add((new field_json("lora"))
        ->set_desc("A list of LoRA adapters to apply to this request. Each entry must have `id` and `scale` fields. Adapters not listed default to scale 0.0")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const auto & lora = data.at("lora");
            if (!lora.is_array()) {
                throw std::runtime_error("Error: 'lora' must be an array of objects with 'id' and 'scale' fields");
            }
            ctx.params.lora = parse_lora_request(lora);
        }));

    // sequence breakers for DRY
    // Currently, this is not compatible with TextGen WebUI, Koboldcpp and SillyTavern format
    // Ref: https://github.com/oobabooga/text-generation-webui/blob/d1af7a41ade7bd3c3a463bfa640725edb818ebaf/extensions/openai/typing.py#L39
    add((new field_json("dry_sequence_breakers"))
        ->set_desc("Specify an array of sequence breakers for DRY sampling. Only a JSON array of strings is accepted")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.sampling.dry_sequence_breakers = json_value(data, "dry_sequence_breakers", std::vector<std::string>());
            if (ctx.params.sampling.dry_sequence_breakers.empty()) {
                throw std::runtime_error("Error: dry_sequence_breakers must be a non-empty array of strings");
            }
        }));

    // handle both "json_schema" and "grammar"
    add((new field_json("json_schema"))
        ->add_alias("grammar")
        ->set_desc("Set a JSON schema (json_schema) or GBNF grammar string (grammar) for constrained generation. json_schema takes precedence if both are provided")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            auto & params = ctx.params;
            if (data.contains("json_schema") && !data.contains("grammar")) {
                try {
                    auto schema                  = json_value(data, "json_schema", json::object());
                    SRV_DBG("JSON schema: %s\n", schema.dump(2).c_str());
                    std::string grammar_str      = json_schema_to_grammar(schema);
                    SRV_DBG("Converted grammar: %s\n", grammar_str.c_str());
                    params.sampling.grammar      = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, std::move(grammar_str)};
                } catch (const std::exception & e) {
                    throw std::runtime_error(std::string("\"json_schema\": ") + e.what());
                }
            } else {
                std::string grammar_str = json_value(data, "grammar", std::string());
                if (!grammar_str.empty()) {
                    // grammar_type key is set by the server when converting chat template grammars
                    std::string grammar_type = json_value(data, "grammar_type", std::string());
                    if (grammar_type == "tool_calls") {
                        params.sampling.grammar = {COMMON_GRAMMAR_TYPE_TOOL_CALLS, std::move(grammar_str)};
                    } else {
                        // explicit grammar from the user (API field "grammar")
                        params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, std::move(grammar_str)};
                    }
                    SRV_DBG("Grammar (%s): %s\n", grammar_type.c_str(), common_grammar_value(params.sampling.grammar).c_str());
                }
            }
        }));

    add((new field_bool("grammar_lazy", params.sampling.grammar_lazy))
        ->set_desc("Whether to apply grammar constraints lazily, only when triggered (instead of at every step)"));

    //
    // Chat parser params
    //

    // TODO: change this to string field instead
    add((new field_json("chat_format"))
        ->set_desc("Chat format used internally by the server")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.chat_parser_params.format = static_cast<common_chat_format>(data.at("chat_format").get<int>());
            SRV_TRC("chat format: %s\n", common_chat_format_name(ctx.params.chat_parser_params.format));
        }));

    add((new field_str("reasoning_format"))
        ->set_desc("Reasoning format for chain-of-thought models")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            auto reasoning_format = common_reasoning_format_from_name(data.at("reasoning_format").get<std::string>());
            ctx.params.chat_parser_params.reasoning_format = reasoning_format;
            ctx.params.chat_parser_params.reasoning_in_content = ctx.params.stream && (reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY);
        }));

    add((new field_str("generation_prompt"))
        ->set_desc("Generation prompt appended to the chat template output")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            std::string s = data.at("generation_prompt").get<std::string>();
            ctx.params.chat_parser_params.generation_prompt = s;
            ctx.params.sampling.generation_prompt = s;
        }));

    add((new field_bool("parse_tool_calls", params.chat_parser_params.parse_tool_calls))
        ->set_desc("Whether to parse tool calls from the generated output"));

    add((new field_str("chat_parser"))
        ->set_desc("Chat parser configuration string")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.chat_parser_params.parser.load(data.at("chat_parser").get<std::string>());
        }));

    add((new field_json("continue_final_message"))
        ->set_desc("Whether to continue the final message of the chat template")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            auto continuation = common_chat_continuation_parse(data.at("continue_final_message"));
            ctx.params.chat_parser_params.is_continuation = continuation != COMMON_CHAT_CONTINUATION_NONE;
        }));

    add((new field_bool("echo", params.chat_parser_params.echo))
        ->set_desc("Whether to echo the input tokens in the output"));

    //
    // Token-level fields (require vocab)
    //

    add((new field_json("preserved_tokens"))
        ->set_desc("List of token strings that must not be split during tokenization")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.vocab != nullptr);
            for (const auto & t : data.at("preserved_tokens")) {
                auto ids = common_tokenize(ctx.vocab, t.get<std::string>(), false, true);
                if (ids.size() == 1) {
                    ctx.params.sampling.preserved_tokens.insert(ids[0]);
                }
            }
        }));

    add((new field_json("grammar_triggers"))
        ->set_desc("List of strings or patterns that trigger grammar-constrained generation")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.vocab != nullptr);
            for (const auto & t : data.at("grammar_triggers")) {
                server_grammar_trigger ct(t);
                if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    const auto & word = ct.value.value;
                    auto ids = common_tokenize(ctx.vocab, word, false, true);
                    if (ids.size() == 1) {
                        auto token = ids[0];
                        if (std::find(ctx.params.sampling.preserved_tokens.begin(), ctx.params.sampling.preserved_tokens.end(), (llama_token) token) == ctx.params.sampling.preserved_tokens.end()) {
                            throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                        }
                        common_grammar_trigger trigger;
                        trigger.type  = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                        trigger.value = word;
                        trigger.token = token;
                        ctx.params.sampling.grammar_triggers.push_back(std::move(trigger));
                    } else {
                        ctx.params.sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
                    }
                } else {
                    ctx.params.sampling.grammar_triggers.emplace_back(std::move(ct.value));
                }
            }
            if (ctx.params.sampling.grammar_lazy && ctx.params.sampling.grammar_triggers.empty()) {
                throw std::runtime_error("Error: no triggers set for lazy grammar!");
            }
        }));

    add((new field_bool("reasoning_control", params.sampling.reasoning_control))
        ->set_desc("Create the budget sampler on demand so reasoning can be ended at runtime"));

    add((new field_num("reasoning_budget_tokens", params.sampling.reasoning_budget_tokens))
        ->set_hard_limits(-1, INT32_MAX)
        ->set_desc("Number of tokens in the reasoning budget (-1 = disabled)"));

    add((new field_str("reasoning_budget_start_tag"))
        ->set_desc("Token string marking the start of the reasoning budget section")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.vocab != nullptr);
            ctx.params.sampling.reasoning_budget_start = common_tokenize(ctx.vocab, data.at("reasoning_budget_start_tag").get<std::string>(), false, true);
        }));

    add((new field_str("reasoning_budget_end_tag"))
        ->set_desc("Token string marking the end of the reasoning budget section")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.vocab != nullptr);
            std::string end_tag = data.at("reasoning_budget_end_tag").get<std::string>();
            ctx.params.sampling.reasoning_budget_end = common_tokenize(ctx.vocab, end_tag, false, true);
        }));

    add((new field_str("reasoning_budget_message"))
        ->set_desc("Message to prepend to the reasoning budget end tag when forcing it")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.vocab != nullptr);
            std::string end_tag = json_value(data, "reasoning_budget_end_tag", std::string());
            std::string message = data.at("reasoning_budget_message").get<std::string>();
            ctx.params.sampling.reasoning_budget_forced = common_tokenize(ctx.vocab, message + end_tag, false, true);
        }));

    add((new field_json("logit_bias"))
        ->set_desc("Modify the likelihood of specific tokens. Accepts an array of [token, bias] pairs or an object mapping token to bias. Use false as bias to ban a token")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.vocab != nullptr);
            ctx.params.sampling.logit_bias.clear();
            const auto & logit_bias = data.at("logit_bias");
            const int n_vocab = llama_vocab_n_tokens(ctx.vocab);
            auto parse_bias = [](const json & v, float & bias) -> bool {
                if (v.is_number())                        { bias = v.get<float>(); return true; }
                if (v.is_boolean() && !v.get<bool>())     { bias = -INFINITY;      return true; }
                return false;
            };
            if (logit_bias.is_array()) {
                for (const auto & el : logit_bias) {
                    if (!el.is_array() || el.size() != 2) continue;
                    float bias;
                    if (!parse_bias(el[1], bias)) continue;
                    if (el[0].is_number_integer()) {
                        llama_token tok = el[0].get<llama_token>();
                        if (tok >= 0 && tok < n_vocab) ctx.params.sampling.logit_bias.push_back({tok, bias});
                    } else if (el[0].is_string()) {
                        for (auto tok : common_tokenize(ctx.vocab, el[0].get<std::string>(), false))
                            ctx.params.sampling.logit_bias.push_back({tok, bias});
                    }
                }
            } else if (logit_bias.is_object()) {
                for (const auto & el : logit_bias.items()) {
                    float bias;
                    if (!parse_bias(el.value(), bias)) continue;
                    char * end;
                    llama_token tok = strtol(el.key().c_str(), &end, 10);
                    if (*end == 0) {
                        if (tok >= 0 && tok < n_vocab) ctx.params.sampling.logit_bias.push_back({tok, bias});
                    } else {
                        for (auto t : common_tokenize(ctx.vocab, el.key(), false))
                            ctx.params.sampling.logit_bias.push_back({t, bias});
                    }
                }
            }
        }));

    add((new field_bool("ignore_eos", params.sampling.ignore_eos))
        ->set_desc("Ignore the end-of-sequence token and continue generating")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.logit_bias_eog != nullptr);
            ctx.params.sampling.ignore_eos = data.at("ignore_eos").get<bool>();
            if (ctx.params.sampling.ignore_eos && ctx.logit_bias_eog) {
                ctx.params.sampling.logit_bias.insert(
                    ctx.params.sampling.logit_bias.end(),
                    ctx.logit_bias_eog->begin(), ctx.logit_bias_eog->end());
            }
        }));

    add((new field_json("stop"))
        ->set_desc("Specify stopping strings. Generation stops when one is produced, and the string is not included in the output")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.antiprompt.clear();
            const auto & stop = data.at("stop");
            if (stop.is_array()) {
                for (const auto & word : stop) {
                    if (!word.empty()) ctx.params.antiprompt.push_back(word);
                }
            } else if (stop.is_string()) {
                ctx.params.antiprompt.push_back(stop.get<std::string>());
            }
            // fall back to CLI defaults if the request provided no effective stop strings
            if (ctx.params.antiprompt.empty()) {
                ctx.params.antiprompt = params_base.antiprompt;
            }
        }));

    add((new field_json("samplers"))
        ->set_desc("The order in which samplers are applied. An array of sampler type names, or a single string of sampler chars")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const auto & samplers = data.at("samplers");
            if (samplers.is_array()) {
                ctx.params.sampling.samplers = common_sampler_types_from_names(samplers);
            } else if (samplers.is_string()) {
                ctx.params.sampling.samplers = common_sampler_types_from_chars(samplers.get<std::string>());
            }
        }));

    return fields;
}

task_params eval_llama_cmpl_schema(
                const llama_vocab * vocab,
                const common_params & params_base,
                const int n_ctx_slot,
                const std::vector<llama_logit_bias> & logit_bias_eog,
                const json & data) {
    task_params params;

    // Sampling parameter defaults are loaded from the global server context (but individual requests can still them)
    params.sampling      = params_base.sampling;
    params.speculative   = params_base.speculative;
    params.n_keep        = params_base.n_keep;
    params.n_predict     = params_base.n_predict;
    params.n_cache_reuse = params_base.n_cache_reuse;
    params.cache_prompt  = params_base.cache_prompt;
    params.antiprompt    = params_base.antiprompt;
    params.sse_ping_interval = params_base.sse_ping_interval;

    // enabling this will output extra debug information in the HTTP responses from the server
    params.verbose       = params_base.verbosity > 9;

    params.chat_parser_params.reasoning_format = params_base.reasoning_format;

    // create context and schema
    field_eval_context ctx(params);
    ctx.vocab          = vocab;
    ctx.logit_bias_eog = &logit_bias_eog;

    auto schema = make_llama_cmpl_schema(params_base, params);

    // eval all fields in the schema
    for (const auto & f : schema) {
        f->eval(ctx, data);
    }

    // post-processing
    {
        if (params.sampling.penalty_last_n == -1) {
            // note: should be the slot's context and not the full context, but it's ok
            params.sampling.penalty_last_n = n_ctx_slot;
        }

        if (params.sampling.dry_penalty_last_n == -1) {
            params.sampling.dry_penalty_last_n = n_ctx_slot;
        }

        // if "reasoning_format" is not provided, its handler will not be called, we will need to handle it here
        auto reasoning_format = params.chat_parser_params.reasoning_format;
        params.chat_parser_params.reasoning_in_content = params.stream && (reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY);
    }

    // debugging
    {
        auto budget = params.sampling.reasoning_budget_tokens;
        SRV_DBG("reasoning budget: tokens=%d, generation_prompt='%s', start=%zu toks, end=%zu toks, forced=%zu toks\n",
                budget, params.sampling.generation_prompt.c_str(),
                params.sampling.reasoning_budget_start.size(),
                params.sampling.reasoning_budget_end.size(),
                params.sampling.reasoning_budget_forced.size());
    }

    return params;
}

//
// eval() implementations
//

static void handle_with_catch(const char * name, std::function<void()> func) {
    try {
        func();
    } catch (const std::exception & e) {
        throw std::invalid_argument(string_format("Field '%s': %s", name, e.what()));
    }
}

template <typename T>
void field_num<T>::eval(field_eval_context & ctx, const json & data) {
    for (const auto & n : name) {
        if (data.contains(n)) {
            handle_with_catch(n, [&]() {
                if (custom_handler) {
                custom_handler(ctx, data);
                } else if (!is_hard_limit) {
                    val = std::max(min, std::min(max, data.at(n).template get<T>()));
                } else {
                    T tmp = data.at(n).template get<T>();
                    if (tmp < min || tmp > max) {
                        throw std::invalid_argument(std::string("Value must be between ") + std::to_string(min) + " <= value <= " + std::to_string(max) + ", but got " + std::to_string(tmp));
                    }
                    val = tmp;
                }
            });
            return;
        }
    }
}

void field_str::eval(field_eval_context & ctx, const json & data) {
    GGML_ASSERT(custom_handler);
    for (const auto & n : name) {
        if (data.contains(n)) {
            handle_with_catch(n, [&]() {
                custom_handler(ctx, data);
            });
            return;
        }
    }
}

void field_bool::eval(field_eval_context & ctx, const json & data) {
    for (const auto & n : name) {
        if (data.contains(n)) {
            handle_with_catch(n, [&]() {
                if (custom_handler) {
                    custom_handler(ctx, data);
                } else {
                    val = data.at(n).get<bool>();
                }
            });
            return;
        }
    }
}

void field_json::eval(field_eval_context & ctx, const json & data) {
    GGML_ASSERT(custom_handler);
    for (const auto & n : name) {
        if (data.contains(n)) {
            handle_with_catch(n, [&]() {
                custom_handler(ctx, data);
            });
            return;
        }
    }
}

void field_nested::eval(field_eval_context & ctx, const json & data) {
    for (const auto & n : name) {
        if (data.contains(n) && data.at(n).is_object()) {
            for (auto & f : subfields) {
                f->eval(ctx, data.at(n));
            }
            return;
        }
    }
}

} // namespace server_schema
