#include "chat-peg-parser.h"
#include "ggml.h"
#include "log.h"
#include "nlohmann/json.hpp"
#include "tests.h"

#include <iostream>
#include <numeric>
#include <ostream>
#include <string>

static inline std::string join(const std::vector<std::string>& parts,
                        const std::string& sep = ", ") {
    if (parts.empty()) { return {}; }

    // Reserve an approximate size to avoid many reallocations.
    std::size_t total_len = sep.size() * (parts.size() - 1);
    for (const auto& s : parts) { total_len += s.size(); }

    std::string result;
    result.reserve(total_len);
    result += parts[0];

    for (std::size_t i = 1; i < parts.size(); ++i) {
        result += sep;
        result += parts[i];
    }
    return result;
}

void test_example_minimax_m2(testing &t) {
    auto helper_parser = build_peg_parser_helper([](common_chat_peg_parser_builder_helper & p) {
        auto thinking = p.reasoning();
        auto content = p.content_before_tools("<minimax:tool_call>");
        auto function = p.quasi_xml_attr("generate_joke",
            std::vector<std::string>({
                "category"
            }));
        auto tool_call = p.trigger(p.add_rule("tool-call",
            "<minimax:tool_call>" + p.one_or_more(function) + "</minimax:tool_call>"));

        return thinking + p.optional(p.space() + content) + p.zero_or_more(p.space() + tool_call);
    });


    t.test("minimax_m2_accumulation_test", [&](testing &t) {
        std::string input =
            "<think>"
            "To keep the reply light I’ll fetch a random joke using the `generate_joke` tool."
            "</think>"
            "<minimax:tool_call>"
            "<invoke name=\"generate_joke\">"
            "<parameter name=\"category\">funny</parameter></invoke>"
            "</minimax:tool_call>";

        std::vector<std::string> tokens = simple_tokenize(input);
        LOG_ERR("Tokens: %s\n", join(tokens).c_str());

        common_chat_msg prev;
        common_chat_parse_result last_result;
        t.test("helper_builder", [&](testing &t) {
            size_t token_cnt = 0;
            for (auto it = tokens.begin(); it != tokens.end(); it++) {
                token_cnt++;
                std::string in = std::accumulate(tokens.begin(), it + 1, std::string());
                LOG_ERR("Current input: %s\n", in.c_str());

                common_chat_parse_semantics semantics;
                common_chat_parse_context   ctx(in, &semantics, it + 1 == tokens.end());

                if (it + 1 == tokens.end()) {
                    common_log_set_verbosity_thold(LOG_DEFAULT_DEBUG);
                }

                ctx.event_handler = it + 1 == tokens.end() ? parser_semantic_handler_with_printout : parser_semantic_handler;

                auto result = helper_parser.parse(ctx);
                last_result = result;
                if (result.fail()) {
                    LOG_ERR("Parsing failure!");
                    break;
                }

                // This shouldn't emit any runtime errors
                auto msg   = semantics.to_msg();
                auto diffs = common_chat_msg_diff::compute_diffs(prev, msg);
                prev       = msg;
            }
            LOG_ERR("Last message: %s\n", prev.to_json_oaicompat<nlohmann::ordered_json>().dump().c_str());
            t.assert_true("last_result_should_be_success", last_result.success());
            t.assert_equal("should_parse_all_tokens_helper", tokens.size(), token_cnt);
        });
        common_log_set_verbosity_thold(LOG_DEFAULT_LLAMA);
    });
}
