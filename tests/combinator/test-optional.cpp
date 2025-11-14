#include "tests.h"

class test_optional : public compound_test {
public:
    test_optional() : compound_test("test_optional") {
        // Full match with optional part present
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.literal("hello") + p.optional(p.literal(" world"));
            });
            
            auto ctx = parser_context("hello world");
            auto result = parser.parse(ctx);
            h.assert_equals("optional_present", true, result.is_success());
            int end_pos = result.end;
            h.assert_equals("optional_present_end", 11, end_pos);
        }, "optional_present");
        
        // Full match with optional part absent
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.literal("hello") + p.optional(p.literal(" world"));
            });
            
            auto ctx = parser_context("hello", true);
            auto result = parser.parse(ctx);
            h.assert_equals("optional_absent", true, result.is_success());
            int end_pos = result.end;
            h.assert_equals("optional_absent_end", 5, end_pos);
        }, "optional_absent");
        
        // Partial match - waiting for more input to determine if optional matches
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.literal("hello") + p.optional(p.literal(" world"));
            });
            
            auto ctx = parser_context("hello ", false);
            auto result = parser.parse(ctx);
            h.assert_equals("partial_match_need_more", true, result.is_need_more_input());
        }, "partial_match_need_more");
    }

    // Provide a convenient way to run all tests
    void run_all_tests() {
        run_all();
        summary();
    }
};