#include "tests.h"

void test_json_serialization(testing &t) {
    auto original = build_peg_parser([](common_peg_parser_builder & p) {
        return "<tool_call>" + p.json() + "</tool_call>";
    });

    auto json_serialized = original.to_json().dump();

    t.test("compare before/after", [&](testing &t) {
        auto deserialized = common_peg_arena::from_json(nlohmann::json::parse(json_serialized));

        // Test complex JSON
        std::string input = R"({"name": "test", "values": [1, 2, 3], "nested": {"a": true}})";
        common_peg_parse_context ctx1(input);
        common_peg_parse_context ctx2(input);

        auto result1 = original.parse(ctx1);
        auto result2 = deserialized.parse(ctx2);

        t.assert_equal("both_succeed", result1.success(), result2.success());
        t.assert_equal("same_end_pos", result1.end, result2.end);
    });

    t.bench("deserialize", [&]() {
        auto deserialized = common_peg_arena::from_json(nlohmann::json::parse(json_serialized));
    }, 100);
}
