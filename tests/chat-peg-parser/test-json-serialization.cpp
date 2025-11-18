#include "tests.h"

void test_json_serialization(testing &t) {
    t.test("simple literal parser round-trip", [](testing &t) {
        auto original = build_peg_parser([](common_chat_peg_parser_builder & p) {
            return p.literal("hello");
        });

        auto json = original.to_json();
        auto deserialized = common_chat_peg_arena::from_json(json);

        // Test that both parsers produce identical results
        std::string input = "hello world";
        common_chat_parse_context ctx1(input);
        common_chat_parse_context ctx2(input);

        auto result1 = original.parse(ctx1);
        auto result2 = deserialized.parse(ctx2);

        t.assert_equal("both_succeed", result1.success(), result2.success());
        t.assert_equal("same_end_pos", result1.end, result2.end);
    });

    t.test("complex parser round-trip", [](testing &t) {
        auto original = build_peg_parser([](common_chat_peg_parser_builder & p) {
            return p.choice({
                p.sequence({p.literal("hello"), p.space(), p.literal("world")}),
                p.literal("goodbye")
            });
        });

        auto json = original.to_json();
        auto deserialized = common_chat_peg_arena::from_json(json);

        // Test both branches work identically
        std::string input1 = "hello world";
        common_chat_parse_context ctx1a(input1);
        common_chat_parse_context ctx1b(input1);

        auto result1a = original.parse(ctx1a);
        auto result1b = deserialized.parse(ctx1b);

        t.assert_equal("hello_both_succeed", result1a.success(), result1b.success());
        t.assert_equal("hello_same_end", result1a.end, result1b.end);

        std::string input2 = "goodbye";
        common_chat_parse_context ctx2a(input2);
        common_chat_parse_context ctx2b(input2);

        auto result2a = original.parse(ctx2a);
        auto result2b = deserialized.parse(ctx2b);

        t.assert_equal("goodbye_both_succeed", result2a.success(), result2b.success());
        t.assert_equal("goodbye_same_end", result2a.end, result2b.end);
    });

    // Test round-trip serialization of recursive grammar
    t.test("recursive grammar round-trip", [](testing &t) {
        auto original = build_peg_parser([](common_chat_peg_parser_builder & p) {
            auto expr = p.rule("expr", [&]() {
                return p.choice({
                    p.sequence({p.literal("("), p.space(), p.ref("expr"), p.space(), p.literal(")")}),
                    p.one("[a-z]+")
                });
            });
            return expr;
        });

        // Serialize
        auto json = original.to_json();

        // Deserialize
        auto deserialized = common_chat_peg_arena::from_json(json);

        // Test nested expressions
        std::string input = "(( abc ))";
        common_chat_parse_context ctx1(input);
        common_chat_parse_context ctx2(input);

        auto result1 = original.parse(ctx1);
        auto result2 = deserialized.parse(ctx2);

        t.assert_equal("both_succeed", result1.success(), result2.success());
        t.assert_equal("same_end_pos", result1.end, result2.end);
    });

    // Test round-trip serialization of JSON parser
    t.test("JSON parser round-trip", [](testing &t) {
        auto original = build_peg_parser([](common_chat_peg_parser_builder & p) {
            return p.json();
        });

        // Serialize
        auto json_serialized = original.to_json();

        // Deserialize
        auto deserialized = common_chat_peg_arena::from_json(json_serialized);

        // Test complex JSON
        std::string input = R"({"name": "test", "values": [1, 2, 3], "nested": {"a": true}})";
        common_chat_parse_context ctx1(input);
        common_chat_parse_context ctx2(input);

        auto result1 = original.parse(ctx1);
        auto result2 = deserialized.parse(ctx2);

        t.assert_equal("both_succeed", result1.success(), result2.success());
        t.assert_equal("same_end_pos", result1.end, result2.end);
    });

    // Test round-trip with captures
    t.test("parser with captures round-trip", [](testing &t) {
        auto original = build_peg_parser([](common_chat_peg_parser_builder & p) {
            return p.sequence({
                p.capture("greeting", p.literal("hello")),
                p.space(),
                p.capture("name", p.one("[a-z]+"))
            });
        });

        // Serialize
        auto json = original.to_json();

        // Deserialize
        auto deserialized = common_chat_peg_arena::from_json(json);

        // Test with semantics
        std::string input = "hello alice";
        common_chat_parse_semantics sem1;
        common_chat_parse_semantics sem2;
        common_chat_parse_context ctx1(input, &sem1);
        common_chat_parse_context ctx2(input, &sem2);

        auto result1 = original.parse(ctx1);
        auto result2 = deserialized.parse(ctx2);

        t.assert_equal("both_succeed", result1.success(), result2.success());
        t.assert_equal("both_capture_greeting", sem1.captures.count("greeting") > 0, sem2.captures.count("greeting") > 0);
        t.assert_equal("both_capture_name", sem1.captures.count("name") > 0, sem2.captures.count("name") > 0);
        if (sem1.captures.count("greeting") && sem2.captures.count("greeting")) {
            t.assert_equal("same_greeting", sem1.captures["greeting"], sem2.captures["greeting"]);
        }
        if (sem1.captures.count("name") && sem2.captures.count("name")) {
            t.assert_equal("same_name", sem1.captures["name"], sem2.captures["name"]);
        }
    });

    // Test serialization with repetitions
    t.test("parser with repetitions round-trip", [](testing &t) {
        auto original = build_peg_parser([](common_chat_peg_parser_builder & p) {
            return p.sequence({
                p.one_or_more(p.one("[a-z]")),
                p.optional(p.one("[0-9]")),
                p.zero_or_more(p.literal("!"))
            });
        });

        // Serialize
        auto json = original.to_json();

        // Deserialize
        auto deserialized = common_chat_peg_arena::from_json(json);

        // Test various inputs
        std::vector<std::string> test_inputs = {"abc", "abc5", "xyz!!!", "test9!"};

        for (const auto& input : test_inputs) {
            common_chat_parse_context ctx1(input);
            common_chat_parse_context ctx2(input);

            auto result1 = original.parse(ctx1);
            auto result2 = deserialized.parse(ctx2);

            t.assert_equal("input_" + input + "_both_succeed", result1.success(), result2.success());
            t.assert_equal("input_" + input + "_same_end", result1.end, result2.end);
        }
    });

    // Test serialization with chars parser
    t.test("chars parser round-trip", [](testing &t) {
        auto original = build_peg_parser([](common_chat_peg_parser_builder & p) {
            return p.chars("[a-zA-Z0-9_]", 3, 10);
        });

        // Serialize
        auto json = original.to_json();

        // Deserialize
        auto deserialized = common_chat_peg_arena::from_json(json);

        // Test
        std::string input = "hello123";
        common_chat_parse_context ctx1(input);
        common_chat_parse_context ctx2(input);

        auto result1 = original.parse(ctx1);
        auto result2 = deserialized.parse(ctx2);

        t.assert_equal("both_succeed", result1.success(), result2.success());
        t.assert_equal("same_end_pos", result1.end, result2.end);
    });

    // Test serialization with lookahead
    t.test("lookahead parser round-trip", [](testing &t) {
        auto original = build_peg_parser([](common_chat_peg_parser_builder & p) {
            return p.sequence({
                p.peek(p.literal("test")),
                p.literal("test")
            });
        });

        // Serialize
        auto json = original.to_json();

        // Deserialize
        auto deserialized = common_chat_peg_arena::from_json(json);

        // Test
        std::string input = "test";
        common_chat_parse_context ctx1(input);
        common_chat_parse_context ctx2(input);

        auto result1 = original.parse(ctx1);
        auto result2 = deserialized.parse(ctx2);

        t.assert_equal("both_succeed", result1.success(), result2.success());
        t.assert_equal("same_end_pos", result1.end, result2.end);
    });

    // Benchmark: deserialize JSON parser
    t.test("deserialize JSON parser", [&](testing & t) {
        auto parser = build_peg_parser([](common_chat_peg_parser_builder & p) {
            return p.json();
        });

        auto json = parser.to_json();
        auto json_str = json.dump();

        t.bench("deserialize json", [&]() {
            auto deserialized = common_chat_peg_arena::from_json(nlohmann::json::parse(json_str));
        }, 1000);
    });
}
