#include "tests.h"

#include "json-schema-to-grammar.h"

void test_json_serialization(testing &t) {
    auto original = build_peg_parser([](common_peg_parser_builder & p) {
        return "<tool_call>" + p.json() + "</tool_call>";
    });

    auto json_serialized = original.to_json().dump();

    t.test("compare before/after", [&](testing &t) {
        auto deserialized = common_peg_arena::from_json(common_json::parse(json_serialized));

        // Test complex JSON
        std::string input = R"({"name": "test", "values": [1, 2, 3], "nested": {"a": true}})";
        common_peg_parse_context ctx1(input);
        common_peg_parse_context ctx2(input);

        auto result1 = original.parse(ctx1);
        auto result2 = deserialized.parse(ctx2);

        t.assert_equal("both_succeed", result1.success(), result2.success());
        t.assert_equal("same_end_pos", result1.end, result2.end);
    });

    t.test("until branches rebuild the same grammar", [](testing &t) {
        auto original = build_peg_parser([](common_peg_parser_builder & p) {
            return p.until(p.until("</tag>"), { { p.literal("</tag>"), p.literal("x") } }, true);
        });
        auto deserialized = common_peg_arena::from_json(common_json::parse(original.to_json().dump()));

        auto grammar = [](const common_peg_arena & arena) {
            return build_grammar([&](const common_grammar_builder & builder) {
                arena.build_grammar(builder);
            });
        };
        t.assert_equal("same_grammar", grammar(original), grammar(deserialized));
    });

    t.test("trigger rules rebuild the same scanning grammar", [](testing &t) {
        auto original = build_peg_parser([](common_peg_parser_builder & p) {
            auto start = p.literal("<tag>");
            std::vector<common_peg_trigger> triggers = { { start, p.literal("x") } };
            return p.until(start) + p.trigger_rule("tool", triggers);
        });
        auto deserialized = common_peg_arena::from_json(common_json::parse(original.to_json().dump()));

        auto grammar = [](const common_peg_arena & arena) {
            return build_grammar([&](const common_grammar_builder & builder) {
                arena.build_grammar(builder, true);
            });
        };
        t.assert_equal("same_grammar", grammar(original), grammar(deserialized));
    });

    t.bench("deserialize", [&]() {
        auto deserialized = common_peg_arena::from_json(common_json::parse(json_serialized));
    }, 100);
}
