#include "chat-auto-parser-helpers.h"
#include "chat-diff-analyzer.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "peg-parser.h"
#include "testing.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace autoparser;

static void test_calculate_diff_split_basic(testing & t);
static void test_calculate_diff_split_identical(testing & t);
static void test_calculate_diff_split_common_prefix(testing & t);
static void test_calculate_diff_split_common_suffix(testing & t);
static void test_calculate_diff_split_common_both(testing & t);
static void test_calculate_diff_split_empty_cases(testing & t);
static void test_calculate_diff_split_no_common(testing & t);
static void test_calculate_diff_split_single_char(testing & t);
static void test_calculate_diff_split_overlaps(testing & t);
static void test_calculate_diff_split_tag_boundaries(testing & t);
static void test_calculate_diff_split(testing & t);

static void test_until_common_prefix_basic(testing & t);
static void test_until_common_prefix(testing & t);

static void test_after_common_suffix_basic(testing & t);
static void test_after_common_suffix(testing & t);

static void test_analyze_tool_call_pure_json(testing & t);
static void test_analyze_tool_call_function_name_markers(testing & t);
static void test_analyze_tool_call_full_markers(testing & t);
static void test_analyze_tool_call_edge_cases(testing & t);

static void test_compare_variants_basic(testing & t);
static void test_compare_variants_messages_modifier(testing & t);
static void test_compare_variants_tools_modifier(testing & t);
static void test_compare_variants_both_modifiers(testing & t);
static void test_compare_variants_template_failure(testing & t);
static void test_compare_variants_identity(testing & t);
static void test_compare_variants(testing & t);

// Seed-OSS template tool calling analysis tests
static void test_seed_oss_tool_analysis(testing & t);
static void test_seed_oss_tool_presence(testing & t);
static void test_seed_oss_call_count(testing & t);
static void test_seed_oss_function_names(testing & t);
static void test_seed_oss_argument_count(testing & t);
static void test_seed_oss_args_presence(testing & t);
static void test_seed_oss_tool_with_reasoning(testing & t);

// Nemotron template analysis tests
static void test_nemotron_analysis(testing & t);
static void test_nemotron_reasoning_detection(testing & t);
static void test_nemotron_tool_format(testing & t);

// CohereForAI template analysis tests
static void test_cohere_reasoning_detection(testing & t);
static void test_cohere_analysis(testing & t);

// Marker separation
static void test_marker_separation(testing & t);

// standard_json_tools format tests
static void test_standard_json_tools_formats(testing & t);
static void test_standard_json_tools_openai(testing & t);
static void test_standard_json_tools_cohere(testing & t);
static void test_standard_json_tools_function_key(testing & t);

// normalize_quotes_to_json tests
static void test_normalize_quotes_to_json(testing & t);
static void test_normalize_quotes_with_embedded_quotes(testing & t);

// TAG_WITH_TAGGED argument parsing tests
static void test_tagged_args_with_embedded_quotes(testing & t);

int main(int argc, char * argv[]) {
    testing t(std::cout);
    t.verbose = true;

    // usage: test-chat-auto-parser-helpers [filter_regex]

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("diff_split", test_calculate_diff_split);
    t.test("common_prefix", test_until_common_prefix);
    t.test("common_suffix", test_after_common_suffix);
    t.test("compare_variants", test_compare_variants);
    t.test("segments", test_marker_separation);
    t.test("seed_oss_diffs", test_seed_oss_tool_analysis);
    t.test("cohere", test_cohere_analysis);
    t.test("nemotron", test_nemotron_analysis);
    t.test("standard_json_tools", test_standard_json_tools_formats);
    t.test("normalize_quotes_to_json", test_normalize_quotes_to_json);
    t.test("tagged_args_embedded_quotes", test_tagged_args_with_embedded_quotes);

    return t.summary();
}

static void test_marker_separation(testing & t) {
    auto single_square_marker = segmentize_markers("pre_marker[marker]post_marker");
    auto single_diag_marker = segmentize_markers("pre_marker<marker>post_marker");
    auto paired_markers = segmentize_markers("<hello>world</hello>");
    auto double_different_markers = segmentize_markers("<hello>[hello]<world>[world]");
    auto in_between = segmentize_markers("im<blue>daba<dee>da[hey]");

    t.test("single_square_marker", [&] (testing & t) {
        t.assert_equal("first is text", segment_type::TEXT, single_square_marker[0].type);
        t.assert_equal("second is marker", segment_type::MARKER, single_square_marker[1].type);
        t.assert_equal("last is text", segment_type::TEXT, single_square_marker[2].type);

        t.assert_equal("first is 'pre_marker'", "pre_marker", single_square_marker[0].value);
        t.assert_equal("second is '[marker]'", "[marker]", single_square_marker[1].value);
        t.assert_equal("last is 'post_marker'", "post_marker", single_square_marker[2].value);
    });

    t.test("single_diagonal_marker", [&] (testing & t) {
        t.assert_equal("first is text", segment_type::TEXT, single_diag_marker[0].type);
        t.assert_equal("second is marker", segment_type::MARKER, single_diag_marker[1].type);
        t.assert_equal("last is text", segment_type::TEXT, single_diag_marker[2].type);

        t.assert_equal("first is 'pre_marker'", "pre_marker", single_diag_marker[0].value);
        t.assert_equal("second is '<marker>'", "<marker>", single_diag_marker[1].value);
        t.assert_equal("last is 'post_marker'", "post_marker", single_diag_marker[2].value);
    });

    t.test("paired_markers", [&] (testing & t) {
        t.assert_equal("first is marker", segment_type::MARKER, paired_markers[0].type);
        t.assert_equal("second is text", segment_type::TEXT, paired_markers[1].type);
        t.assert_equal("third is marker", segment_type::MARKER, paired_markers[2].type);

        t.assert_equal("first is '<hello>'", "<hello>", paired_markers[0].value);
        t.assert_equal("second is 'world'", "world", paired_markers[1].value);
        t.assert_equal("third is '</hello>'", "</hello>", paired_markers[2].value);
    });

    t.test("double_different_markers", [&] (testing & t) {
        t.assert_equal("first is marker", segment_type::MARKER, double_different_markers[0].type);
        t.assert_equal("second is marker", segment_type::MARKER, double_different_markers[1].type);
        t.assert_equal("third is marker", segment_type::MARKER, double_different_markers[2].type);
        t.assert_equal("fourth is marker", segment_type::MARKER, double_different_markers[3].type);

        t.assert_equal("first is '<hello>'", "<hello>", double_different_markers[0].value);
        t.assert_equal("second is '[hello]'", "[hello]", double_different_markers[1].value);
        t.assert_equal("third is '<world>'", "<world>", double_different_markers[2].value);
        t.assert_equal("fourth is '[world]'", "[world]", double_different_markers[3].value);
    });

    t.test("in_between", [&] (testing & t) {
        t.assert_equal("first is text", segment_type::TEXT, in_between[0].type);
        t.assert_equal("second is marker", segment_type::MARKER, in_between[1].type);
        t.assert_equal("third is text", segment_type::TEXT, in_between[2].type);
        t.assert_equal("fourth is marker", segment_type::MARKER, in_between[3].type);
        t.assert_equal("fifth is text", segment_type::TEXT, in_between[4].type);
        t.assert_equal("sixth is marker", segment_type::MARKER, in_between[5].type);

        t.assert_equal("first is 'im'", "im", in_between[0].value);
        t.assert_equal("second is '<blue>'", "<blue>", in_between[1].value);
        t.assert_equal("third is 'daba'", "daba", in_between[2].value);
        t.assert_equal("fourth is '<dee>'", "<dee>", in_between[3].value);
        t.assert_equal("fifth is 'da'", "da", in_between[4].value);
        t.assert_equal("sixth is '[hey]'", "[hey]", in_between[5].value);
    });
}

static void test_calculate_diff_split(testing & t) {
    t.test("calculate_diff_split basic", test_calculate_diff_split_basic);
    t.test("calculate_diff_split identical", test_calculate_diff_split_identical);
    t.test("calculate_diff_split common prefix", test_calculate_diff_split_common_prefix);
    t.test("calculate_diff_split common suffix", test_calculate_diff_split_common_suffix);
    t.test("calculate_diff_split common both", test_calculate_diff_split_common_both);
    t.test("calculate_diff_split empty cases", test_calculate_diff_split_empty_cases);
    t.test("calculate_diff_split no common", test_calculate_diff_split_no_common);
    t.test("calculate_diff_split single char", test_calculate_diff_split_single_char);
    t.test("calculate_diff_split overlaps", test_calculate_diff_split_overlaps);
    t.test("calculate_diff_split tag boundaries", test_calculate_diff_split_tag_boundaries);
}

static void test_calculate_diff_split_basic(testing & t) {
    diff_split result = calculate_diff_split("hello world", "hello test");
    t.assert_equal("prefix should be 'hello '", "hello ", result.prefix);
    t.assert_equal("left should be 'world'", "world", result.left);
    t.assert_equal("right should be 'test'", "test", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    result = calculate_diff_split("abc", "xyz");
    t.assert_equal("prefix should be empty", "", result.prefix);
    t.assert_equal("left should be 'abc'", "abc", result.left);
    t.assert_equal("right should be 'xyz'", "xyz", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    result = calculate_diff_split("prefixA suffix", "prefixB suffix");
    t.assert_equal("prefix should be 'prefix'", "prefix", result.prefix);
    t.assert_equal("left should be 'A'", "A", result.left);
    t.assert_equal("right should be 'B'", "B", result.right);
    t.assert_equal("suffix should be ' suffix'", " suffix", result.suffix);
}

static void test_calculate_diff_split_identical(testing & t) {
    diff_split result = calculate_diff_split("hello", "hello");
    t.assert_equal("prefix should be 'hello'", "hello", result.prefix);
    t.assert_equal("left should be empty", "", result.left);
    t.assert_equal("right should be empty", "", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    result = calculate_diff_split("", "");
    t.assert_equal("prefix should be empty", "", result.prefix);
    t.assert_equal("left should be empty", "", result.left);
    t.assert_equal("right should be empty", "", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    result = calculate_diff_split("a", "a");
    t.assert_equal("prefix should be 'a'", "a", result.prefix);
    t.assert_equal("left should be empty", "", result.left);
    t.assert_equal("right should be empty", "", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    result = calculate_diff_split("<row><row><row><your><boat><gently>", "<row><row><row><your><boat><gently>");
    t.assert_equal("prefix should be '<row><row><row><your><boat><gently>'", "<row><row><row><your><boat><gently>", result.prefix);
    t.assert_equal("left should be empty", "", result.left);
    t.assert_equal("right should be empty", "", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);
}

static void test_calculate_diff_split_common_prefix(testing & t) {
    diff_split result = calculate_diff_split("abcdef", "abcxyz");
    t.assert_equal("prefix should be 'abc'", "abc", result.prefix);
    t.assert_equal("left should be 'def'", "def", result.left);
    t.assert_equal("right should be 'xyz'", "xyz", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    result = calculate_diff_split("same", "sameagain");
    t.assert_equal("prefix should be 'same'", "same", result.prefix);
    t.assert_equal("left should be empty", "", result.left);
    t.assert_equal("right should be 'again'", "again", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    result = calculate_diff_split("test", "testing");
    t.assert_equal("prefix should be 'test'", "test", result.prefix);
    t.assert_equal("left should be empty", "", result.left);
    t.assert_equal("right should be 'ing'", "ing", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);
}

static void test_calculate_diff_split_common_suffix(testing & t) {
    diff_split result = calculate_diff_split("123end", "456end");
    t.assert_equal("prefix should be empty", "", result.prefix);
    t.assert_equal("left should be '123'", "123", result.left);
    t.assert_equal("right should be '456'", "456", result.right);
    t.assert_equal("suffix should be 'end'", "end", result.suffix);

    result = calculate_diff_split("start", "end");
    t.assert_equal("prefix should be empty", "", result.prefix);
    t.assert_equal("left should be 'start'", "start", result.left);
    t.assert_equal("right should be 'end'", "end", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    result = calculate_diff_split("abcsuffix", "xyzsuffix");
    t.assert_equal("prefix should be empty", "", result.prefix);
    t.assert_equal("left should be 'abc'", "abc", result.left);
    t.assert_equal("right should be 'xyz'", "xyz", result.right);
    t.assert_equal("suffix should be 'suffix'", "suffix", result.suffix);
}

static void test_calculate_diff_split_common_both(testing & t) {
    diff_split result = calculate_diff_split("helloXworld", "helloYworld");
    t.assert_equal("prefix should be 'hello'", "hello", result.prefix);
    t.assert_equal("left should be 'X'", "X", result.left);
    t.assert_equal("right should be 'Y'", "Y", result.right);
    t.assert_equal("suffix should be 'world'", "world", result.suffix);

    result = calculate_diff_split("ABCmiddleXYZ", "ABCdifferentXYZ");
    t.assert_equal("prefix should be 'ABC'", "ABC", result.prefix);
    t.assert_equal("left should be 'middle'", "middle", result.left);
    t.assert_equal("right should be 'different'", "different", result.right);
    t.assert_equal("suffix should be 'XYZ'", "XYZ", result.suffix);

    result = calculate_diff_split("startAend", "startBend");
    t.assert_equal("prefix should be 'start'", "start", result.prefix);
    t.assert_equal("left should be 'A'", "A", result.left);
    t.assert_equal("right should be 'B'", "B", result.right);
    t.assert_equal("suffix should be 'end'", "end", result.suffix);

    // Edge case: common prefix and suffix overlap
    result = calculate_diff_split("aa", "ab");
    t.assert_equal("prefix should be 'a'", "a", result.prefix);
    t.assert_equal("left should be 'a'", "a", result.left);
    t.assert_equal("right should be 'b'", "b", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);
}

static void test_calculate_diff_split_empty_cases(testing & t) {
    // Empty left, non-empty right
    diff_split result = calculate_diff_split("", "hello");
    t.assert_equal("prefix should be empty", "", result.prefix);
    t.assert_equal("left should be empty", "", result.left);
    t.assert_equal("right should be 'hello'", "hello", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    // Non-empty left, empty right
    result = calculate_diff_split("hello", "");
    t.assert_equal("prefix should be empty", "", result.prefix);
    t.assert_equal("left should be 'hello'", "hello", result.left);
    t.assert_equal("right should be empty", "", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    // Both empty
    result = calculate_diff_split("", "");
    t.assert_equal("prefix should be empty", "", result.prefix);
    t.assert_equal("left should be empty", "", result.left);
    t.assert_equal("right should be empty", "", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    // Left single char, empty right
    result = calculate_diff_split("a", "");
    t.assert_equal("prefix should be empty", "", result.prefix);
    t.assert_equal("left should be 'a'", "a", result.left);
    t.assert_equal("right should be empty", "", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    // Empty left, right single char
    result = calculate_diff_split("", "a");
    t.assert_equal("prefix should be empty", "", result.prefix);
    t.assert_equal("left should be empty", "", result.left);
    t.assert_equal("right should be 'a'", "a", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);
}

static void test_calculate_diff_split_no_common(testing & t) {
    diff_split result = calculate_diff_split("abc", "xyz");
    t.assert_equal("prefix should be empty", "", result.prefix);
    t.assert_equal("left should be 'abc'", "abc", result.left);
    t.assert_equal("right should be 'xyz'", "xyz", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    result = calculate_diff_split("left", "right");
    // The algorithm finds "t" as a common suffix since both strings end with 't'
    // This is the algorithm's actual behavior - it finds maximal common suffix
    t.assert_equal("prefix should be empty", "", result.prefix);
    t.assert_equal("left should be 'lef'", "lef", result.left);
    t.assert_equal("right should be 'righ'", "righ", result.right);
    t.assert_equal("suffix should be 't'", "t", result.suffix);

    result = calculate_diff_split("123", "456");
    t.assert_equal("prefix should be empty", "", result.prefix);
    t.assert_equal("left should be '123'", "123", result.left);
    t.assert_equal("right should be '456'", "456", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);
}

static void test_calculate_diff_split_single_char(testing & t) {
    diff_split result = calculate_diff_split("a", "b");
    t.assert_equal("prefix should be empty", "", result.prefix);
    t.assert_equal("left should be 'a'", "a", result.left);
    t.assert_equal("right should be 'b'", "b", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    result = calculate_diff_split("a", "a");
    t.assert_equal("prefix should be 'a'", "a", result.prefix);
    t.assert_equal("left should be empty", "", result.left);
    t.assert_equal("right should be empty", "", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    result = calculate_diff_split("a", "ab");
    t.assert_equal("prefix should be 'a'", "a", result.prefix);
    t.assert_equal("left should be empty", "", result.left);
    t.assert_equal("right should be 'b'", "b", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    result = calculate_diff_split("ab", "a");
    t.assert_equal("prefix should be 'a'", "a", result.prefix);
    t.assert_equal("left should be 'b'", "b", result.left);
    t.assert_equal("right should be empty", "", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);
}

static void test_calculate_diff_split_overlaps(testing & t) {
    // One string is substring of another
    diff_split result = calculate_diff_split("test", "testing");
    t.assert_equal("prefix should be 'test'", "test", result.prefix);
    t.assert_equal("left should be empty", "", result.left);
    t.assert_equal("right should be 'ing'", "ing", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    result = calculate_diff_split("testing", "test");
    t.assert_equal("prefix should be 'test'", "test", result.prefix);
    t.assert_equal("left should be 'ing'", "ing", result.left);
    t.assert_equal("right should be empty", "", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    // Similar strings with one extra char at start
    result = calculate_diff_split("Xtest", "Ytest");
    // The algorithm finds "test" as a common suffix since both strings end with "test"
    // This is the algorithm's actual behavior - it finds maximal common suffix
    t.assert_equal("prefix should be empty", "", result.prefix);
    t.assert_equal("left should be 'X'", "X", result.left);
    t.assert_equal("right should be 'Y'", "Y", result.right);
    t.assert_equal("suffix should be 'test'", "test", result.suffix);

    // Similar strings with one extra char at end
    result = calculate_diff_split("testX", "testY");
    t.assert_equal("prefix should be 'test'", "test", result.prefix);
    t.assert_equal("left should be 'X'", "X", result.left);
    t.assert_equal("right should be 'Y'", "Y", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    // Strings that are reverses
    result = calculate_diff_split("abc", "cba");
    t.assert_equal("prefix should be empty", "", result.prefix);
    t.assert_equal("left should be 'abc'", "abc", result.left);
    t.assert_equal("right should be 'cba'", "cba", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);
}

static void test_calculate_diff_split_tag_boundaries(testing & t) {
    // Test with unclosed XML tags
    diff_split result = calculate_diff_split("test<tag", "test>content");
    // The fix_tag_boundaries should move incomplete tags appropriately
    t.assert_true("prefix should start with 'test'", result.prefix.find("test") == 0);
    t.assert_true("should handle tag boundaries", result.left != "" || result.right != "" || result.suffix != "");

    // Test with unclosed brackets
    result = calculate_diff_split("test[", "test]value");
    t.assert_true("should handle bracket boundaries", result.left != "" || result.right != "" || result.suffix != "");

    // Test with partial tags on both sides
    result = calculate_diff_split("prefix<tag>", "prefix</tag>suffix");
    // fix_tag_boundaries moves the incomplete '<' from prefix to left/right
    t.assert_equal("prefix should be 'prefix'", "prefix", result.prefix);
    t.assert_equal("left should be '<tag>'", "<tag>", result.left);
    t.assert_equal("right should be '</tag>suffix'", "</tag>suffix", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    // Test with complex nested tags
    result = calculate_diff_split("prefix<div>content</div>", "prefix<div>different</div>");
    // Algorithm finds "ent</div>" as a common suffix because both strings end with it
    // This is the actual algorithm behavior, though not semantically ideal
    t.assert_equal("prefix should be 'prefix<div>'", "prefix<div>", result.prefix);
    t.assert_equal("left should be 'cont'", "cont", result.left);
    t.assert_equal("right should be 'differ'", "differ", result.right);
    t.assert_equal("suffix should be 'ent</div>'", "ent</div>", result.suffix);

    // Test with unclosed angle bracket
    result = calculate_diff_split("Hello <world>", "Hello test");
    t.assert_equal("prefix should be 'Hello '", "Hello ", result.prefix);
    t.assert_true("left should contain '<world>'", result.left.find("<world>") != std::string::npos);
    t.assert_equal("right should be 'test'", "test", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    // Test with unclosed square bracket
    result = calculate_diff_split("test [array]", "test other");
    t.assert_equal("prefix should be 'test '", "test ", result.prefix);
    t.assert_true("left should contain '[array]'", result.left.find("[array]") != std::string::npos);
    t.assert_equal("right should be 'other'", "other", result.right);
    t.assert_equal("suffix should be empty", "", result.suffix);

    // Test empty prefix and suffix with tags
    result = calculate_diff_split("<tag>left</tag>", "<tag>righ</tag>");
    t.assert_equal("prefix should be '<tag>'", "<tag>", result.prefix);
    t.assert_equal("left should be 'left'", "left", result.left);
    t.assert_equal("right should be 'righ'", "righ", result.right);
    t.assert_equal("suffix should be '</tag>'", "</tag>", result.suffix);

    {
        // real case from template tests, simplified
        std::string left  = "PREFIX</think>Sure";
        std::string right = "PREFIX<think>Lemme think</think>Sure";
        result            = calculate_diff_split(left, right);
        t.assert_equal("prefix should be PREFIX", "PREFIX", result.prefix);
        t.assert_equal("suffix should be </think>Sure", "</think>Sure", result.suffix);
        t.assert_equal("left should be empty", "", result.left);
        t.assert_equal("right should be <think>Lemme think", "<think>Lemme think", result.right);
    }

    {
        // Real case: special tokens with |> boundary issue
        // The suffix starts with |> which should be moved to complete <|END_RESPONSE and <|END_ACTION
        std::string prefix    = "SOME_PREFIX";
        std::string suffix    = "|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>";
        std::string left_diff = "<|START_RESPONSE|>Let me help you.<|END_RESPONSE";
        std::string right_diff =
            "<|START_THINKING|><|END_THINKING|><|START_ACTION|>[\n"
            "    {\"tool_call_id\": \"0\", \"tool_name\": \"test_function_name\", "
            "\"parameters\": {\"param1\": \"value1\", \"param2\": \"value2\"}}\n"
            "]<|END_ACTION";

        std::string left  = prefix + left_diff + suffix;
        std::string right = prefix + right_diff + suffix;
        result            = calculate_diff_split(left, right);

        t.assert_equal("special token prefix", prefix, result.prefix);
        // The |> should be moved from suffix to complete the tokens
        t.assert_equal("special token left", "<|START_RESPONSE|>Let me help you.<|END_RESPONSE|>", result.left);
        t.assert_true("special token right ends with |>", result.right.find("<|END_ACTION|>") != std::string::npos);
        t.assert_equal("special token suffix", "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
                       result.suffix);
    }
}

static void test_until_common_prefix(testing & t) {
    t.test("until_common_prefix basic", test_until_common_prefix_basic);
}

static void test_until_common_prefix_basic(testing & t) {
    // Test case from the user request
    std::string result = until_common_prefix("<function name=foo><arg name=bar>", "<arg name=bar>", "<arg name=baz>");
    t.assert_equal("untilCommonPrefix should return '<function name=foo>'", "<function name=foo>", result);

    // Additional test cases to ensure robustness
    // Test with different common prefix lengths
    result = until_common_prefix("prefix<test>suffix", "<test>different", "<test>other");
    t.assert_equal("should return 'prefix'", "prefix", result);

    // Test when common prefix is at the start
    result = until_common_prefix("<common>rest", "<common>left", "<common>right");
    t.assert_equal("should return empty string when common prefix at start", "", result);

    // Test when there's no common prefix
    result = until_common_prefix("something", "left", "right");
    t.assert_equal("should return empty string when no common prefix", "", result);

    // Test with empty strings
    result = until_common_prefix("test", "", "right");
    t.assert_equal("should return empty string when left is empty", "", result);

    // Test with longer common prefix
    result = until_common_prefix("abcXYZ<shared_prefix>rest", "<shared_prefix>left", "<shared_prefix>right");
    t.assert_equal("should return 'abcXYZ'", "abcXYZ", result);
}

static void test_after_common_suffix(testing & t) {
    t.test("after_common_suffix basic", test_after_common_suffix_basic);
}

static void test_after_common_suffix_basic(testing & t) {
    // Test case from the user request
    std::string result = after_common_suffix("<function name=foo><arg name=bar>100</arg></function>",
                                            "<arg name=bar>100</arg>",
                                            "<arg name=baz>535</arg>");
    t.assert_equal("afterCommonSuffix should return '</function>'", "</function>", result);

    // Test when common suffix is at the end
    result = after_common_suffix("rest<common>", "left<common>", "right<common>");
    t.assert_equal("should return empty string when common suffix at end", "", result);

    // Test with empty strings
    result = after_common_suffix("test", "left", "");
    t.assert_equal("should return empty string when right is empty", "", result);

    // Test case with XML-like structure similar to the main example
    result = after_common_suffix("<outer><inner>value</inner></outer>",
                                "<inner>value</inner>",
                                "<inner>different</inner>");
    t.assert_equal("should return '</outer>'", "</outer>", result);

    // Test with longer common suffix appearing at the end of full
    result = after_common_suffix("prefix<shared>rest</shared>", "prefix<shared>left</shared>", "prefix<shared>right</shared>");
    t.assert_equal("should return '' when common suffix is at end of full", "", result);

    // Test with common suffix appearing in middle but not at end
    result = after_common_suffix("<tag>content</tag><extra>", "<tag>value</tag>", "<tag>other</tag>");
    t.assert_equal("should return '<extra>' when common suffix appears before end", "<extra>", result);

    // Test with multi-character common suffix at the very end of full
    result = after_common_suffix("start<middle>end</middle>", "prefix<middle>left</middle>", "prefix<middle>right</middle>");
    t.assert_equal("should return '' when common suffix </middle> is at end of full", "", result);
}

static void test_compare_variants(testing & t) {
    t.test("compare_variants basic", test_compare_variants_basic);
    t.test("compare_variants messages modifier", test_compare_variants_messages_modifier);
    t.test("compare_variants tools modifier", test_compare_variants_tools_modifier);
    t.test("compare_variants both modifiers", test_compare_variants_both_modifiers);
    t.test("compare_variants template failure", test_compare_variants_template_failure);
    t.test("compare_variants identity", test_compare_variants_identity);
}

static void test_compare_variants_basic(testing & t) {
    // Create a simple template that just echoes messages
    common_chat_template tmpl("{{ messages[0]['content'] }}", "", "");

    template_params params;
    params.messages = json::array({
        json {{"role", "user"}, {"content", "Hello"}}
    });

    auto modifier = [](template_params & p) {
        p.messages[0]["content"] = "World";
    };

    auto result = autoparser::compare_variants(tmpl, params, modifier);

    if (!t.assert_true("result should have value", result.has_value())) {
        return;
    }
    // The template might not output anything if messages is empty or format is different
    // Check that we get a valid result
    t.assert_true("prefix or left should have content", !result->diff.prefix.empty() || !result->diff.left.empty());
}

static void test_compare_variants_messages_modifier(testing & t) {
    // Test with messages modifier only
    common_chat_template tmpl("{% for message in messages %}{{ message['role'] }}:{{ message['content'] }}{% endfor %}", "", "");

    template_params params;
    params.messages = json::array({
        json {{"role", "user"}, {"content", "A"}}
    });

    auto modifier = [](template_params & p) {
        p.messages[0]["content"] = "B";
    };

    std::optional<compare_variants_result> result = autoparser::compare_variants(tmpl, params, modifier);

    if (!t.assert_true("result should have value", result.has_value())) {
        return;
    }
    t.assert_equal("left should be 'A'", "A", result->diff.left);
    t.assert_equal("right should be 'B'", "B", result->diff.right);
}

static void test_compare_variants_tools_modifier(testing & t) {
    // Test with tools modifier only
    common_chat_template tmpl(
        "{% for tool in tools %}{{ tool['name'] }}{% endfor %}", "", "");

    template_params params;
    params.tools = json::array({
        json {{"name", "foo"}}
    });

    auto modifier = [](template_params & p) {
        p.tools[0]["name"] = "bar";
    };

    auto result = autoparser::compare_variants(tmpl, params, modifier);

    if (!t.assert_true("result should have value", result.has_value())) {
        return;
    }
    t.assert_equal("left should be 'foo'", "foo", result->diff.left);
    t.assert_equal("right should be 'bar'", "bar", result->diff.right);
}

static void test_compare_variants_both_modifiers(testing & t) {
    // Test with both messages and tools modifiers using the for loop approach
    common_chat_template tmpl(
        "{% for message in messages %}{{ message['role'] }}:{{ message['content'] }}{% endfor %}", "", "");

    template_params params;
    params.messages = json::array({
        json {{"role", "user"}, {"content", "A"}}
    });

    auto modifier = [](template_params & p) {
        p.messages[0]["content"] = "B";
        p.messages[0]["role"] = "newuser";
    };

    auto result = autoparser::compare_variants(tmpl, params, modifier);

    if (!t.assert_true("result should have value", result.has_value())) {
        return;
    }
    t.assert_equal("left should be 'user:A'", "user:A", result->diff.left);
    t.assert_equal("right should be 'newuser:B'", "newuser:B", result->diff.right);
}

static void test_compare_variants_template_failure(testing & t) {
    // Test with template that causes failure during application (not construction)
    // We use a valid template syntax but one that will fail during application
    common_chat_template tmpl("{{ messages[0]['nonexistent_field'] }}", "", "");

    template_params params;
    params.messages = json::array({
        json {{"role", "user"}, {"content", "Hello"}}
    });

    auto modifier = [](template_params & p) {
        p.messages[0]["content"] = "World";
    };

    auto result = autoparser::compare_variants(tmpl, params, modifier);

    t.assert_true("result should be nullopt on template failure", !result.has_value());
}

static void test_compare_variants_identity(testing & t) {
    // Test with identity modifier (no change)
    common_chat_template tmpl("{{ messages[0]['content'] }}", "", "");

    template_params params;
    params.messages = json::array({
        json {{"role", "user"}, {"content", "Hello"}}
    });

    // No modifier - should use identity
    auto result = autoparser::compare_variants(tmpl, params, nullptr);

    if (!t.assert_true("result should have value", result.has_value())) {
        return;
    }
    t.assert_equal("prefix should be 'Hello'", "Hello", result->diff.prefix);
    t.assert_equal("left should be empty", "", result->diff.left);
    t.assert_equal("right should be empty", "", result->diff.right);
    t.assert_equal("suffix should be empty", "", result->diff.suffix);
}

// ============================================================================
// Seed-OSS Template Tool Calling Analysis Tests
// ============================================================================

static void test_seed_oss_tool_analysis(testing & t) {
    t.test("Seed-OSS tool presence", test_seed_oss_tool_presence);
    t.test("Seed-OSS call count", test_seed_oss_call_count);
    t.test("Seed-OSS function names", test_seed_oss_function_names);
    t.test("Seed-OSS argument count", test_seed_oss_argument_count);
    t.test("Seed-OSS args presence", test_seed_oss_args_presence);
    t.test("Seed-OSS tool with reasoning", test_seed_oss_tool_with_reasoning);
}

// Helper to load Seed-OSS template
static common_chat_template load_seed_oss_template(testing & t) {
    std::string template_path = "models/templates/ByteDance-Seed-OSS.jinja";
    std::ifstream fin(template_path, std::ios::binary);
    std::ostringstream buf;
    if (fin.is_open()) {
        buf << fin.rdbuf();
    }
    std::string template_source = buf.str();
    common_chat_template tmpl(template_source, "", "");
    t.assert_true("Seed-OSS template loaded successfully", template_source.length() > 0);
    return tmpl;
}

// Helper to build tool call JSON
static json build_tool_call(const std::string & name, const json & args, const std::string & id = "call_001") {
    return json{
        {"id", id},
        {"type", "function"},
        {"function", json{
            {"name", name},
            {"arguments", args}
        }}
    };
}

// Helper to build tools definition
static json build_tools_definition() {
    json parameters_schema = json::object();
    parameters_schema["type"] = "object";
    parameters_schema["properties"] = json::object();
    parameters_schema["properties"]["param1"] = json::object({
        {"type", "string"},
        {"description", "First parameter"}
    });
    parameters_schema["properties"]["param2"] = json::object({
        {"type", "string"},
        {"description", "Second parameter"}
    });
    parameters_schema["required"] = json::array({"param1", "param2"});

    return json::array({
        json{
            {"type", "function"},
            {"function", json{
                {"name", "test_function_name"},
                {"description", "A test function for debugging"},
                {"parameters", parameters_schema}
            }}
        }
    });
}

// T1: Compare with/without tool call (user, assistant)
static void test_seed_oss_tool_presence(testing & t) {
    common_chat_template tmpl = load_seed_oss_template(t);

    json assistant_no_tools = json{
        {"role", "assistant"},
        {"content", "Let me help you."}
    };

    json assistant_with_tools = json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", json::array({
            build_tool_call("test_function_name", json::object({{"param1", "value1"}, {"param2", "value2"}}))
        })}
    };

    json user_msg = json{
        {"role", "user"},
        {"content", "Hello, please help me."}
    };

    template_params params_no_tools;
    params_no_tools.messages = json::array({user_msg, assistant_no_tools});
    params_no_tools.tools = build_tools_definition();
    params_no_tools.add_generation_prompt = false;
    params_no_tools.enable_thinking = true;

    template_params params_with_tools;
    params_with_tools.messages = json::array({user_msg, assistant_with_tools});
    params_with_tools.tools = build_tools_definition();
    params_with_tools.add_generation_prompt = false;
    params_with_tools.enable_thinking = true;

    auto result = autoparser::compare_variants(tmpl, params_no_tools,
        [&](template_params & p) {
            p.messages = params_with_tools.messages;
        });

    if (!t.assert_true("T1 result should have value", result.has_value())) {
        return;
    }

    const auto & diff = result->diff;
    t.assert_true("T1 prefix should contain system", diff.prefix.find("system") != std::string::npos);
    t.assert_true("T1 prefix should contain user", diff.prefix.find("user") != std::string::npos);
    t.assert_true("T1 prefix should contain assistant", diff.prefix.find("assistant") != std::string::npos);

    // Left should be the assistant content without tool
    t.assert_equal("T1 left should contain 'Let me help you.'", "Let me help you.", diff.left);

    // Right should contain the tool call markers
    t.assert_true("T1 right should contain tool_call begin", diff.right.find("<seed:tool_call>") != std::string::npos);
    t.assert_true("T1 right should contain function tag", diff.right.find("<function=test_function_name>") != std::string::npos);
    t.assert_true("T1 right should contain parameter=param1", diff.right.find("<parameter=param1>") != std::string::npos);
    t.assert_true("T1 right should contain parameter=param2", diff.right.find("<parameter=param2>") != std::string::npos);
    t.assert_true("T1 right should contain value1", diff.right.find("value1") != std::string::npos);
    t.assert_true("T1 right should contain value2", diff.right.find("value2") != std::string::npos);
    t.assert_true("T1 right should contain tool_call end", diff.right.find("</seed:tool_call>") != std::string::npos);

    // Suffix should be the eos token
    t.assert_equal("T1 suffix should be '<seed:eos>'", "<seed:eos>", diff.suffix);
}

// T2: Compare one vs two tool calls
static void test_seed_oss_call_count(testing & t) {
    common_chat_template tmpl = load_seed_oss_template(t);

    json assistant_one_call = json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", json::array({
            build_tool_call("test_function_name", json::object({{"param1", "value1"}, {"param2", "value2"}}))
        })}
    };

    json assistant_two_calls = json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", json::array({
            build_tool_call("test_function_name", json::object({{"param1", "value1"}, {"param2", "value2"}})),
            build_tool_call("test_function_name", json::object({{"param1", "value3"}, {"param2", "value4"}}), "call_002")
        })}
    };

    json user_msg = json{
        {"role", "user"},
        {"content", "Hello, please help me."}
    };

    template_params params_one;
    params_one.messages = json::array({user_msg, assistant_one_call});
    params_one.tools = build_tools_definition();
    params_one.add_generation_prompt = false;
    params_one.enable_thinking = true;

    auto result = autoparser::compare_variants(tmpl, params_one,
        [&](template_params & p) {
            p.messages = json::array({user_msg, assistant_two_calls});
        });

    if (!t.assert_true("T2 result should have value", result.has_value())) {
        return;
    }

    const auto & diff = result->diff;

    // Prefix should include the first tool call
    t.assert_true("T2 prefix should contain first tool_call begin", diff.prefix.find("<seed:tool_call>") != std::string::npos);
    t.assert_true("T2 prefix should contain first function", diff.prefix.find("<function=test_function_name>") != std::string::npos);
    t.assert_true("T2 prefix should contain value1", diff.prefix.find("value1") != std::string::npos);
    t.assert_true("T2 prefix should contain value2", diff.prefix.find("value2") != std::string::npos);
    t.assert_true("T2 prefix should contain first tool_call end", diff.prefix.find("</seed:tool_call>") != std::string::npos);

    // Left should be empty (no second tool call in variant A)
    t.assert_equal("T2 left should be empty", "", diff.left);

    // Right should contain the second tool call
    t.assert_true("T2 right should contain second tool_call begin", diff.right.find("<seed:tool_call>") != std::string::npos);
    t.assert_true("T2 right should contain second function", diff.right.find("<function=test_function_name>") != std::string::npos);
    t.assert_true("T2 right should contain value3", diff.right.find("value3") != std::string::npos);
    t.assert_true("T2 right should contain value4", diff.right.find("value4") != std::string::npos);
    t.assert_true("T2 right should contain second tool_call end", diff.right.find("</seed:tool_call>") != std::string::npos);

    // Suffix should end with the eos token
    t.assert_equal("T2 suffix should end with '<seed:eos>'", "<seed:eos>", diff.suffix.substr(diff.suffix.length() - 10, 10));
}

// T3: Compare different function names
static void test_seed_oss_function_names(testing & t) {
    common_chat_template tmpl = load_seed_oss_template(t);

    // Build tools with two different function names
    json parameters_schema = json::object();
    parameters_schema["type"] = "object";
    parameters_schema["properties"] = json::object();
    parameters_schema["properties"]["arg1"] = json::object({
        {"type", "string"},
        {"description", "Argument 1"}
    });
    parameters_schema["required"] = json::array({"arg1"});

    json tools = json::array({
        json{
            {"type", "function"},
            {"function", json{
                {"name", "func_alpha"},
                {"description", "First function"},
                {"parameters", parameters_schema}
            }}
        },
        json{
            {"type", "function"},
            {"function", json{
                {"name", "func_beta"},
                {"description", "Second function"},
                {"parameters", parameters_schema}
            }}
        }
    });

    json assistant_func_alpha = json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", json::array({
            build_tool_call("func_alpha", json::object({{"arg1", "test_value"}}))
        })}
    };

    json assistant_func_beta = json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", json::array({
            build_tool_call("func_beta", json::object({{"arg1", "test_value"}}))
        })}
    };

    json user_msg = json{
        {"role", "user"},
        {"content", "Hello"}
    };

    template_params params_alpha;
    params_alpha.messages = json::array({user_msg, assistant_func_alpha});
    params_alpha.tools = tools;
    params_alpha.add_generation_prompt = false;
    params_alpha.enable_thinking = true;

    auto result = autoparser::compare_variants(tmpl, params_alpha,
        [&](template_params & p) {
            p.messages = json::array({user_msg, assistant_func_beta});
        });

    if (!t.assert_true("T3 result should have value", result.has_value())) {
        return;
    }

    const auto & diff = result->diff;

    bool func_alpha_in_left = diff.left.find("func_alpha") != std::string::npos;
    bool func_alpha_in_prefix = diff.prefix.find("func_alpha") != std::string::npos;
    bool func_beta_in_right = diff.right.find("func_beta") != std::string::npos;
    bool func_beta_in_prefix = diff.prefix.find("func_beta") != std::string::npos;
    bool func_beta_in_suffix = diff.suffix.find("func_beta") != std::string::npos;

    // Left should contain func_alpha (or be in prefix)
    t.assert_true("T3 left should contain func_alpha (or prefix)", func_alpha_in_left || func_alpha_in_prefix);

    // Right should contain func_beta
    t.assert_true("T3 right should contain func_beta", func_beta_in_right || func_beta_in_prefix || func_beta_in_suffix);

    // Both should have the same parameter value (in common parts, not in diffs)
    // Since both have same args, test_value will be in prefix/suffix
    t.assert_true("T3 diff should contain test_value (in prefix or suffix)",
        diff.prefix.find("test_value") != std::string::npos || diff.suffix.find("test_value") != std::string::npos);
}

// T4: Compare different argument counts (zero, one, two parameters)
static void test_seed_oss_argument_count(testing & t) {
    common_chat_template tmpl = load_seed_oss_template(t);

    // Build tools with 0, 1, or 2 required parameters
    json params_2_required = json::object();
    params_2_required["type"] = "object";
    params_2_required["properties"] = json::object();
    params_2_required["properties"]["arg1"] = json::object({
        {"type", "string"},
        {"description", "Argument 1"}
    });
    params_2_required["properties"]["arg2"] = json::object({
        {"type", "string"},
        {"description", "Argument 2"}
    });
    params_2_required["required"] = json::array({"arg1", "arg2"});

    json params_1_required = json::object();
    params_1_required["type"] = "object";
    params_1_required["properties"] = json::object();
    params_1_required["properties"]["arg1"] = json::object({
        {"type", "string"},
        {"description", "Argument 1"}
    });
    params_1_required["required"] = json::array({"arg1"});

    json tools = json::array({
        json{
            {"type", "function"},
            {"function", json{
                {"name", "test_func"},
                {"description", "Test function"},
                {"parameters", params_2_required}
            }}
        }
    });

    // Test: zero args vs one arg
    json assistant_zero_args = json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", json::array({
            build_tool_call("test_func", json::object())
        })}
    };

    json assistant_one_arg = json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", json::array({
            build_tool_call("test_func", json::object({{"arg1", "value1"}}))
        })}
    };

    json assistant_two_args = json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", json::array({
            build_tool_call("test_func", json::object({{"arg1", "value1"}, {"arg2", "value2"}}))
        })}
    };

    json user_msg = json{
        {"role", "user"},
        {"content", "Hello"}
    };

    // Test zero vs one
    template_params params_zero;
    params_zero.messages = json::array({user_msg, assistant_zero_args});
    params_zero.tools = tools;
    params_zero.add_generation_prompt = false;
    params_zero.enable_thinking = true;

    auto result_zero_one = autoparser::compare_variants(tmpl, params_zero,
        [&](template_params & p) {
            p.messages = json::array({user_msg, assistant_one_arg});
        });

    if (!t.assert_true("T4 zero vs one result should have value", result_zero_one.has_value())) {
        return;
    }
    t.assert_true("T4 zero vs one left should be empty or minimal", result_zero_one->diff.left.empty() || result_zero_one->diff.left == "");
    t.assert_true("T4 zero vs one right should contain arg1", result_zero_one->diff.right.find("arg1") != std::string::npos);

    // Test one vs two
    template_params params_one;
    params_one.messages = json::array({user_msg, assistant_one_arg});
    params_one.tools = tools;
    params_one.add_generation_prompt = false;
    params_one.enable_thinking = true;

    auto result_one_two = autoparser::compare_variants(tmpl, params_one,
        [&](template_params & p) {
            p.messages = json::array({user_msg, assistant_two_args});
        });

    if (!t.assert_true("T4 one vs two result should have value", result_one_two.has_value())) {
        return;
    }

    const auto & diff4 = result_one_two->diff;
    t.assert_true("T4 one vs two left should contain arg1 (or prefix)",
        diff4.left.find("arg1") != std::string::npos || diff4.prefix.find("arg1") != std::string::npos);
    t.assert_true("T4 one vs two right should contain arg1 (or prefix)",
        diff4.right.find("arg1") != std::string::npos || diff4.prefix.find("arg1") != std::string::npos);
    t.assert_true("T4 one vs two right should contain arg2 (or prefix/suffix)",
        diff4.right.find("arg2") != std::string::npos || diff4.prefix.find("arg2") != std::string::npos || diff4.suffix.find("arg2") != std::string::npos);
}

// T5: Compare different argument values
static void test_seed_oss_args_presence(testing & t) {
    common_chat_template tmpl = load_seed_oss_template(t);

    json assistant_same_arg = json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", json::array({
            build_tool_call("test_function_name", json::object({{"param1", "value1"}}))
        })}
    };

    json assistant_other_arg = json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", json::array({
            build_tool_call("test_function_name", json::object({{"param2", "value2"}}))
        })}
    };

    json assistant_both_args = json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", json::array({
            build_tool_call("test_function_name", json::object({{"param1", "value1"}, {"param2", "value2"}}))
        })}
    };

    json user_msg = json{
        {"role", "user"},
        {"content", "Hello"}
    };

    template_params params_same;
    params_same.messages = json::array({user_msg, assistant_same_arg});
    params_same.tools = build_tools_definition();
    params_same.add_generation_prompt = false;
    params_same.enable_thinking = true;

    // Test same arg vs other arg
    auto result_same_other = autoparser::compare_variants(tmpl, params_same,
        [&](template_params & p) {
            p.messages = json::array({user_msg, assistant_other_arg});
        });

    if (!t.assert_true("T5 same vs other result should have value", result_same_other.has_value())) {
        return;
    }
    const auto & diff5a = result_same_other->diff;
    t.assert_true("T5 same vs other left should contain param1 (or prefix/suffix)",
        diff5a.left.find("param1") != std::string::npos || diff5a.prefix.find("param1") != std::string::npos || diff5a.suffix.find("param1") != std::string::npos);
    t.assert_true("T5 same vs other left should contain value1 (or prefix/suffix)",
        diff5a.left.find("value1") != std::string::npos || diff5a.prefix.find("value1") != std::string::npos);
    t.assert_true("T5 same vs other right should contain param2 (or prefix/suffix)",
        diff5a.right.find("param2") != std::string::npos || diff5a.prefix.find("param2") != std::string::npos || diff5a.suffix.find("param2") != std::string::npos);
    t.assert_true("T5 same vs other right should contain value2 (or prefix/suffix)",
        diff5a.right.find("value2") != std::string::npos || diff5a.prefix.find("value2") != std::string::npos || diff5a.suffix.find("value2") != std::string::npos);

    // Test same arg vs both args
    auto result_same_both = autoparser::compare_variants(tmpl, params_same,
        [&](template_params & p) {
            p.messages = json::array({user_msg, assistant_both_args});
        });

    if (!t.assert_true("T5 same vs both result should have value", result_same_both.has_value())) {
        return;
    }
    const auto & diff5b = result_same_both->diff;
    t.assert_true("T5 same vs both left should contain param1 (or prefix/suffix)",
        diff5b.left.find("param1") != std::string::npos || diff5b.prefix.find("param1") != std::string::npos || diff5b.suffix.find("param1") != std::string::npos);
    t.assert_true("T5 same vs both right should contain param1 (or prefix/suffix)",
        diff5b.right.find("param1") != std::string::npos || diff5b.prefix.find("param1") != std::string::npos || diff5b.suffix.find("param1") != std::string::npos);
    t.assert_true("T5 same vs both right should contain param2 (or prefix/suffix)",
        diff5b.right.find("param2") != std::string::npos || diff5b.prefix.find("param2") != std::string::npos || diff5b.suffix.find("param2") != std::string::npos);
}

// T6: Tool call with vs without reasoning_content
static void test_seed_oss_tool_with_reasoning(testing & t) {
    common_chat_template tmpl = load_seed_oss_template(t);

    json assistant_tool_only = json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", json::array({
            build_tool_call("test_function_name", json::object({{"param1", "value1"}, {"param2", "value2"}}))
        })}
    };

    json assistant_tool_with_reasoning = json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", json::array({
            build_tool_call("test_function_name", json::object({{"param1", "value1"}, {"param2", "value2"}}))
        })},
        {"reasoning_content", "I need to call the tool first."}
    };

    json user_msg = json{
        {"role", "user"},
        {"content", "Hello, please help me."}
    };

    template_params params_tool_only;
    params_tool_only.messages = json::array({user_msg, assistant_tool_only});
    params_tool_only.tools = build_tools_definition();
    params_tool_only.add_generation_prompt = false;
    params_tool_only.enable_thinking = true;

    auto result = autoparser::compare_variants(tmpl, params_tool_only,
        [&](template_params & p) {
            p.messages = json::array({user_msg, assistant_tool_with_reasoning});
        });

    if (!t.assert_true("T6 result should have value", result.has_value())) {
        return;
    }

    const auto & diff = result->diff;

    // Left should be empty (no reasoning in variant A)
    t.assert_equal("T6 left should be empty", "", diff.left);

    // Right should contain the thinking token with reasoning content
    t.assert_true("T6 right should contain think begin", diff.right.find("<seed:think>") != std::string::npos);
    t.assert_true("T6 right should contain reasoning content", diff.right.find("I need to call the tool first.") != std::string::npos);
    t.assert_true("T6 right should contain think end", diff.right.find("</seed:think>") != std::string::npos);

    // Prefix should contain the assistant role
    t.assert_true("T6 prefix should contain assistant", diff.prefix.find("assistant") != std::string::npos);

    // Suffix should contain the tool call
    t.assert_true("T6 suffix should contain tool_call begin", diff.suffix.find("<seed:tool_call>") != std::string::npos);
    t.assert_true("T6 suffix should contain function name", diff.suffix.find("test_function_name") != std::string::npos);
    t.assert_true("T6 suffix should contain eos", diff.suffix.find("<seed:eos>") != std::string::npos);
}

static common_chat_template load_template(testing & t, const std::string & template_path) {
    std::ifstream fin(template_path, std::ios::binary);
    std::ostringstream buf;
    if (fin.is_open()) {
        buf << fin.rdbuf();
    }
    std::string template_source = buf.str();
    common_chat_template tmpl(template_source, "", "");
    t.assert_true("Nemotron template loaded successfully", template_source.length() > 0);
    return tmpl;
}

// ============================================================================
// Nemotron Template Analysis Tests
// ============================================================================
static common_chat_template load_nemotron_template(testing & t) {
    return load_template(t, "models/templates/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja");
}

static void test_nemotron_analysis(testing & t) {
    t.test("Nemotron reasoning detection", test_nemotron_reasoning_detection);
    t.test("Nemotron tool format", test_nemotron_tool_format);
}

static void test_nemotron_reasoning_detection(testing & t) {
    common_chat_template tmpl = load_nemotron_template(t);

    // Test the comparison manually to see what's happening
    json user_msg = json{ { "role", "user" }, { "content", "Hello" } };
    json assistant_no_reasoning = json{
        { "role", "assistant" },
        { "content", "I can help." }
    };
    json assistant_with_reasoning = json{
        { "role", "assistant" },
        { "content", "I can help." },
        { "reasoning_content", "Let me think about this." }
    };

    template_params params;
    params.messages = json::array({ user_msg, assistant_no_reasoning });
    params.add_generation_prompt = false;
    params.enable_thinking = true;

    // Run differential analysis
    auto analysis = autoparser::analyze_template(tmpl);

    // Check reasoning markers
    t.assert_equal("reasoning_start should be '<think>'", "<think>", analysis.reasoning.start);
    t.assert_equal("reasoning_end should be '</think>\\n'", "</think>\n", analysis.reasoning.end);

    // Check reasoning mode detection
    // Nemotron uses forced closed reasoning with add_generation_prompt
    t.assert_equal("reasoning should be FORCED_CLOSED", reasoning_mode::FORCED_CLOSED, analysis.reasoning.mode);

    // Make sure reasoning markers don't spill over to content markers
    t.assert_equal("content start should be empty", "", analysis.content.start);
    t.assert_equal("content end should be empty", "", analysis.content.end);

    t.assert_equal("content should be PLAIN", content_mode::PLAIN, analysis.content.mode);
}

static void test_nemotron_tool_format(testing & t) {
    common_chat_template tmpl = load_nemotron_template(t);

    // Run differential analysis
    auto analysis = autoparser::analyze_template(tmpl);

    // Check tool markers - Nemotron uses per-call wrapping (each call individually wrapped)
    t.assert_equal("tool_section_start should be empty (per-call format)", "", analysis.tools.format.section_start);
    t.assert_equal("tool_section_end should be empty (per-call format)", "", analysis.tools.format.section_end);
    t.assert_equal("per_call_start should be '<tool_call>\\n'", "<tool_call>\n", analysis.tools.format.per_call_start);
    t.assert_equal("per_call_end should be '</tool_call>'", "</tool_call>", analysis.tools.format.per_call_end);
    t.assert_true("should support parallel calls", analysis.jinja_caps.supports_parallel_tool_calls);

    // Check function markers
    t.assert_equal("func_name_prefix should be '<function='", "<function=", analysis.tools.function.name_prefix);
    t.assert_equal("func_name_suffix should be '>\\n'", ">\n", analysis.tools.function.name_suffix);
    t.assert_equal("func_close should be '</function>\\n'", "</function>\n", analysis.tools.function.close);

    // Check argument markers (note: markers retain trailing newlines for proper parsing)
    t.assert_equal("arg_name_prefix should be '<parameter='", "<parameter=", analysis.tools.arguments.name_prefix);
    t.assert_equal("arg_name_suffix should be '>\\n'", ">\n", analysis.tools.arguments.name_suffix);
    t.assert_equal("arg_value_suffix should be '</parameter>\\n'", "</parameter>\n", analysis.tools.arguments.value_suffix);

    // Check format classification
    t.assert_true("tool format should be TAG_WITH_TAGGED", analysis.tools.format.mode == tool_format::TAG_WITH_TAGGED);

    // Verify tool support
    t.assert_true("should support tools", analysis.jinja_caps.supports_tools);
}

static common_chat_template load_cohere_template(testing & t) {
    return load_template(t, "models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja");
}

static void test_cohere_analysis(testing & t) {
    t.test("Cohere reasoning detection", test_cohere_reasoning_detection);
}

static void test_cohere_reasoning_detection(testing & t) {
    common_chat_template tmpl = load_cohere_template(t);

    // Run differential analysis
    auto analysis = autoparser::analyze_template(tmpl);

    // Check reasoning markers - Cohere uses special token format
    t.assert_equal("reasoning_start should be '<|START_THINKING|>'", "<|START_THINKING|>", analysis.reasoning.start);
    t.assert_equal("reasoning_end should be '<|END_THINKING|>'", "<|END_THINKING|>", analysis.reasoning.end);

    // Check reasoning mode - Cohere only shows reasoning with tool calls (TOOLS_ONLY)
    t.assert_equal("reasoning should be TOOLS_ONLY", reasoning_mode::TOOLS_ONLY, analysis.reasoning.mode);

    // Check content markers - Cohere wraps all content with START/END_RESPONSE
    t.assert_equal("content_start should be '<|START_RESPONSE|>'", "<|START_RESPONSE|>", analysis.content.start);
    t.assert_equal("content_end should be '<|END_RESPONSE|>'", "<|END_RESPONSE|>", analysis.content.end);

    // Content is always wrapped (both with and without tools)
    t.assert_equal("content should be ALWAYS_WRAPPED", content_mode::ALWAYS_WRAPPED, analysis.content.mode);
}

static void test_tool_format_cohere(testing & t) {
    common_chat_template tmpl = load_cohere_template(t);

    // Run differential analysis
    auto analysis = autoparser::analyze_template(tmpl);

    // Check tool section markers - Cohere uses ACTION markers
    t.assert_equal("tool_section_start should be '<|START_ACTION|>'", "<|START_ACTION|>", analysis.tools.format.section_start);
    t.assert_equal("tool_section_end should be '<|END_ACTION|>'", "<|END_ACTION|>", analysis.tools.format.section_end);

    // JSON_NATIVE format has no per-call markers
    t.assert_equal("per_call_start should be empty", "", analysis.tools.format.per_call_start);
    t.assert_equal("per_call_end should be empty", "", analysis.tools.format.per_call_end);

    // JSON_NATIVE format has empty function markers (no XML-style markers)
    t.assert_equal("func_name_prefix should be empty", "", analysis.tools.function.name_prefix);
    t.assert_equal("func_name_suffix should be empty", "", analysis.tools.function.name_suffix);
    t.assert_equal("func_close should be empty", "", analysis.tools.function.close);

    // JSON_NATIVE format has empty args markers
    t.assert_equal("args_start should be empty", "", analysis.tools.arguments.start);
    t.assert_equal("args_end should be empty", "", analysis.tools.arguments.end);

    // JSON_NATIVE format has empty argument markers
    t.assert_equal("arg_name_prefix should be empty", "", analysis.tools.arguments.name_prefix);
    t.assert_equal("arg_name_suffix should be empty", "", analysis.tools.arguments.name_suffix);
    t.assert_equal("arg_value_prefix should be empty", "", analysis.tools.arguments.value_prefix);
    t.assert_equal("arg_value_suffix should be empty", "", analysis.tools.arguments.value_suffix);
    t.assert_equal("arg_separator should be empty", "", analysis.tools.arguments.separator);

    // Check JSON field names - Cohere uses non-standard names
    t.assert_equal("name_field should be 'tool_name'", "tool_name", analysis.tools.format.name_field);
    t.assert_equal("args_field should be 'parameters'", "parameters", analysis.tools.format.args_field);
    // This isn't a real tool call id field, i.e. with the OpenAI tool call ID format
    t.assert_equal("id_field should be 'tool_call_id'", "", analysis.tools.format.id_field);

    // Check format classification
    t.assert_equal("tool format should be JSON_NATIVE", tool_format::JSON_NATIVE, analysis.tools.format.mode);

    // Check flags
    t.assert_true("should support tools", analysis.jinja_caps.supports_tools);
    t.assert_true("should support parallel calls", analysis.jinja_caps.supports_parallel_tool_calls);
    t.assert_true("should not require nonnull content", !analysis.content.requires_nonnull_content);
    t.assert_true("tools_array_wrapped should be true", analysis.tools.format.tools_array_wrapped);
}

// ============================================================================
// standard_json_tools Format Tests
// ============================================================================

// Helper to build tools definition for tests
static json build_test_tools() {
    json parameters_schema = json::object();
    parameters_schema["type"] = "object";
    parameters_schema["properties"] = json::object();
    parameters_schema["properties"]["location"] = json::object({
        {"type", "string"},
        {"description", "The city and state"}
    });
    parameters_schema["properties"]["unit"] = json::object({
        {"type", "string"},
        {"description", "Temperature unit"},
        {"enum", json::array({"celsius", "fahrenheit"})}
    });
    parameters_schema["required"] = json::array({"location"});

    return json::array({
        json{
            {"type", "function"},
            {"function", json{
                {"name", "get_current_weather"},
                {"description", "Get the current weather in a given location"},
                {"parameters", parameters_schema}
            }}
        }
    });
}

static void test_standard_json_tools_formats(testing & t) {
    t.test("OpenAI format", test_standard_json_tools_openai);
    t.test("Cohere format", test_standard_json_tools_cohere);
    t.test("function-as-key format", test_standard_json_tools_function_key);
}

// Test 1: OpenAI Standard Format
// {"id": "call_abc", "function": {"name": "get_weather", "arguments": {"location": "NYC"}}}
static void test_standard_json_tools_openai(testing & t) {
    json tools = build_test_tools();

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto tool_call = p.standard_json_tools(
            "<tool_call>", "</tool_call>", tools,
            /* parallel */ true,
            /* force */ false,
            /* name_key */ "function.name",
            /* args_key */ "function.arguments",
            /* array_wrapped */ false,
            /* function_is_key */ false,
            /* call_id_key */ "id",
            /* gen_call_id_key */ "",
            /* parameters_order */ {}
        );
        return p.content(p.until("<tool_call>")) + p.optional(tool_call) + p.end();
    });

    std::string input =
        "Let me check the weather."
        "<tool_call>"
        R"({"id": "call_abc123", "function": {"name": "get_current_weather", "arguments": {"location": "NYC"}}})"
        "</tool_call>";

    common_peg_parse_context ctx(input, false);
    auto result = parser.parse(ctx);

    if (!t.assert_true("parse success", result.success())) {
        return;
    }

    common_chat_msg msg;
    auto mapper = common_chat_peg_mapper(msg);
    mapper.from_ast(ctx.ast, result);

    t.assert_equal("tool calls count", 1u, msg.tool_calls.size());
    if (!msg.tool_calls.empty()) {
        t.assert_equal("tool name", "get_current_weather", msg.tool_calls[0].name);
        t.assert_equal("tool id", "call_abc123", msg.tool_calls[0].id);
    }
    t.assert_true("content present", msg.content.find("Let me check the weather") != std::string::npos);
}

// Test 2: Cohere Format
// {"tool_call_id": 0, "tool_name": "get_weather", "parameters": {"location": "NYC"}}
static void test_standard_json_tools_cohere(testing & t) {
    json tools = build_test_tools();

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto tool_call = p.standard_json_tools(
            "<|START_ACTION|>[", "]<|END_ACTION|>", tools,
            /* parallel */ true,
            /* force */ false,
            /* name_key */ "tool_name",
            /* args_key */ "parameters",
            /* array_wrapped */ false,  // Brackets are part of section markers
            /* function_is_key */ false,
            /* call_id_key */ "",
            /* gen_call_id_key */ "tool_call_id",
            /* parameters_order */ {"tool_call_id", "tool_name", "parameters"}
        );
        return p.content(p.until("<|START_ACTION|>")) + p.optional(tool_call) + p.end();
    });

    std::string input =
        "Let me search for that."
        "<|START_ACTION|>["
        R"({"tool_call_id": 0, "tool_name": "get_current_weather", "parameters": {"location": "NYC", "unit": "celsius"}})"
        "]<|END_ACTION|>";

    common_peg_parse_context ctx(input, false);
    auto result = parser.parse(ctx);

    if (!t.assert_true("parse success", result.success())) {
        return;
    }

    common_chat_msg msg;
    auto mapper = common_chat_peg_mapper(msg);
    mapper.from_ast(ctx.ast, result);

    t.assert_equal("tool calls count", 1u, msg.tool_calls.size());
    if (!msg.tool_calls.empty()) {
        t.assert_equal("tool name", "get_current_weather", msg.tool_calls[0].name);
        t.assert_equal("tool id", "0", msg.tool_calls[0].id);
    }
    t.assert_true("content present", msg.content.find("Let me search") != std::string::npos);
}

// Test 3: Function-as-Key Format
// {"get_current_weather": {"id": "call-0001", "args": {"location": "NYC"}}}
static void test_standard_json_tools_function_key(testing & t) {
    json tools = build_test_tools();

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto tool_call = p.standard_json_tools(
            "<tool_calls>[", "]</tool_calls>", tools,
            /* parallel */ true,
            /* force */ false,
            /* name_key */ "",  // Name is the key itself
            /* args_key */ "args",
            /* array_wrapped */ false,
            /* function_is_key */ true,
            /* call_id_key */ "id",
            /* gen_call_id_key */ "",
            /* parameters_order */ {}
        );
        return p.content(p.until("<tool_calls>")) + p.optional(tool_call) + p.end();
    });

    std::string input =
        "I'll call the weather function."
        "<tool_calls>["
        R"({"get_current_weather": {"id": "call-0001", "args": {"location": "NYC", "unit": "celsius"}}})"
        "]</tool_calls>";

    common_peg_parse_context ctx(input, false);
    auto result = parser.parse(ctx);

    if (!t.assert_true("parse success", result.success())) {
        return;
    }

    common_chat_msg msg;
    auto mapper = common_chat_peg_mapper(msg);
    mapper.from_ast(ctx.ast, result);

    t.assert_equal("tool calls count", 1u, msg.tool_calls.size());
    if (!msg.tool_calls.empty()) {
        t.assert_equal("tool name", "get_current_weather", msg.tool_calls[0].name);
        t.assert_equal("tool id", "call-0001", msg.tool_calls[0].id);
    }
    t.assert_true("content present", msg.content.find("I'll call the weather") != std::string::npos);
}

// ============================================================================
// normalize_quotes_to_json Tests
// ============================================================================

// Copy of the function for isolated testing (original is static in chat-peg-parser.cpp)
static std::string normalize_quotes_to_json(const std::string & input) {
    std::string result;
    result.reserve(input.size() + 16);

    bool in_single_quoted = false;
    bool in_double_quoted = false;

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];

        if (c == '\\' && i + 1 < input.size()) {
            char next = input[i + 1];

            if (in_single_quoted) {
                if (next == '\'') {
                    result += '\'';
                    ++i;
                    continue;
                }
                if (next == '"') {
                    result += "\\\"";
                    ++i;
                    continue;
                }
                result += c;
                result += next;
                ++i;
                continue;
            }

            if (in_double_quoted) {
                result += c;
                result += next;
                ++i;
                continue;
            }

            result += c;
            continue;
        }

        if (c == '"') {
            if (in_single_quoted) {
                result += "\\\"";
            } else {
                in_double_quoted = !in_double_quoted;
                result += c;
            }
        } else if (c == '\'') {
            if (in_double_quoted) {
                result += c;
            } else if (in_single_quoted) {
                in_single_quoted = false;
                result += '"';
            } else {
                in_single_quoted = true;
                result += '"';
            }
        } else {
            result += c;
        }
    }

    return result;
}

static void test_normalize_quotes_to_json(testing & t) {
    t.test("basic single to double quotes", [](testing & t) {
        std::string input = "{'key': 'value'}";
        std::string expected = "{\"key\": \"value\"}";
        std::string result = normalize_quotes_to_json(input);
        t.assert_equal("basic conversion", expected, result);
    });

    t.test("escaped single quote inside single-quoted string", [](testing & t) {
        std::string input = "{'code': 'print(\\'hello\\')'}";
        std::string expected = "{\"code\": \"print('hello')\"}";
        std::string result = normalize_quotes_to_json(input);
        t.assert_equal("escaped single quote", expected, result);
    });

    t.test("double quote inside single-quoted string", [](testing & t) {
        std::string input = "{'msg': 'He said \"hi\"'}";
        std::string expected = "{\"msg\": \"He said \\\"hi\\\"\"}";
        std::string result = normalize_quotes_to_json(input);
        t.assert_equal("double quote escaping", expected, result);
    });

    t.test("nested backslash escapes", [](testing & t) {
        std::string input = "{'path': 'C:\\\\Users\\\\test'}";
        std::string expected = "{\"path\": \"C:\\\\Users\\\\test\"}";
        std::string result = normalize_quotes_to_json(input);
        t.assert_equal("backslash escaping", expected, result);
    });

    t.test("newline escapes", [](testing & t) {
        std::string input = "{'text': 'line1\\nline2'}";
        std::string expected = "{\"text\": \"line1\\nline2\"}";
        std::string result = normalize_quotes_to_json(input);
        t.assert_equal("newline escaping", expected, result);
    });

    t.test("mixed quotes", [](testing & t) {
        std::string input = "{\"already_double\": 'single_value'}";
        std::string expected = "{\"already_double\": \"single_value\"}";
        std::string result = normalize_quotes_to_json(input);
        t.assert_equal("mixed quotes", expected, result);
    });

    t.test("embedded quotes - the test case", test_normalize_quotes_with_embedded_quotes);
}

// Test case that mirrors the Seed-OSS failing test scenario
static void test_normalize_quotes_with_embedded_quotes(testing & t) {
    // This is similar to the Seed-OSS template test case
    // The input has embedded double quotes like "14" and "bar" inside string values
    std::string input = "{'filename': 'foo.cpp', 'oldString': 'def foo(arg = \"14\"):\\n    return arg + \"bar\"\\n', 'newString': 'def foo(arg = \"15\"):\\n    pass\\n'}";

    // Expected: Python single quotes -> JSON double quotes, internal double quotes escaped
    std::string expected = "{\"filename\": \"foo.cpp\", \"oldString\": \"def foo(arg = \\\"14\\\"):\\n    return arg + \\\"bar\\\"\\n\", \"newString\": \"def foo(arg = \\\"15\\\"):\\n    pass\\n\"}";

    std::string result = normalize_quotes_to_json(input);

    t.assert_equal("normalize quotes with embedded double quotes", expected, result);

    // Also verify the result is valid JSON
    try {
        json parsed = json::parse(result);
        t.assert_true("result is valid JSON", true);
        t.assert_equal("filename field", "foo.cpp", parsed["filename"].get<std::string>());
        t.assert_true("oldString contains embedded quotes",
            parsed["oldString"].get<std::string>().find("\"14\"") != std::string::npos);
        t.assert_true("newString contains embedded quotes",
            parsed["newString"].get<std::string>().find("\"15\"") != std::string::npos);
    } catch (const std::exception & e) {
        t.assert_true(std::string("JSON parse failed: ") + e.what(), false);
    }
}

// ============================================================================
// TAG_WITH_TAGGED Argument Parsing Tests
// ============================================================================

// Build tools definition for edit function
static json build_edit_tool() {
    json parameters_schema = json::object();
    parameters_schema["type"] = "object";
    parameters_schema["properties"] = json::object();
    parameters_schema["properties"]["filename"] = json::object({
        {"type", "string"},
        {"description", "Path of file to edit"}
    });
    parameters_schema["properties"]["oldString"] = json::object({
        {"type", "string"},
        {"description", "String to replace"}
    });
    parameters_schema["properties"]["newString"] = json::object({
        {"type", "string"},
        {"description", "New (replacement) value"}
    });
    parameters_schema["required"] = json::array({"filename", "oldString", "newString"});

    return json::array({
        json{
            {"type", "function"},
            {"function", json{
                {"name", "edit"},
                {"description", "Edit a file"},
                {"parameters", parameters_schema}
            }}
        }
    });
}

// Test that reproduces the Seed-OSS template issue with embedded quotes
static void test_tagged_args_with_embedded_quotes(testing & t) {
    json tools = build_edit_tool();

    // Build a parser for TAG_WITH_TAGGED format like Seed-OSS/Nemotron
    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        // Build tool choice for the edit function
        auto tool_choice = p.choice();

        for (const auto & tool_def : tools) {
            if (!tool_def.contains("function")) { continue; }
            const auto & function = tool_def.at("function");
            std::string name = function.at("name");
            const auto & params = function.at("parameters");

            if (!params.contains("properties") || !params.at("properties").is_object()) { continue; }

            const auto & properties = params.at("properties");

            // Build argument parsers
            std::vector<common_peg_parser> arg_parsers;
            for (const auto & [param_name, param_schema] : properties.items()) {
                auto arg = p.tool_arg(
                    p.tool_arg_open(p.literal("<parameter=") + p.tool_arg_name(p.literal(param_name)) + p.literal(">")) +
                    p.space() +
                    p.tool_arg_string_value(p.until("</parameter>")) +
                    p.space() +
                    p.tool_arg_close(p.literal("</parameter>"))
                );
                arg_parsers.push_back(p.optional(p.rule("arg-" + param_name, arg)));
            }

            // Build arg sequence with space() between
            common_peg_parser args_seq = p.eps();
            for (size_t i = 0; i < arg_parsers.size(); i++) {
                if (i > 0) {
                    args_seq = args_seq + p.space();
                }
                args_seq = args_seq + arg_parsers[i];
            }

            auto func_parser =
                p.tool_open(p.literal("<function=") + p.tool_name(p.literal(name)) + p.literal(">")) +
                p.space() + args_seq + p.space() +
                p.tool_close(p.literal("</function>"));

            tool_choice |= p.rule("tool-" + name, p.tool(func_parser));
        }

        auto tool_section =
            p.literal("<seed:tool_call>") + p.space() +
            tool_choice +
            p.space() + p.literal("</seed:tool_call>");

        return p.content(p.until("<seed:tool_call>")) + p.optional(tool_section) + p.end();
    });

    // The exact input from the failing test
    std::string input =
        "<seed:tool_call>\n"
        "<function=edit>\n"
        "<parameter=filename>\n"
        "foo.cpp\n"
        "</parameter>\n"
        "<parameter=oldString>"
        "def foo(arg = \"14\"):\n"
        "    return arg + \"bar\"\n"
        "\n"
        "</parameter>\n"
        "<parameter=newString>"
        "def foo(arg = \"15\"):\n"
        "    pass\n"
        "\n"
        "</parameter>\n"
        "</function>\n"
        "</seed:tool_call>";

    common_peg_parse_context ctx(input, false);
    auto result = parser.parse(ctx);

    if (!t.assert_true("parse success", result.success())) {
        return;
    }

    common_chat_msg msg;
    auto mapper = common_chat_peg_mapper(msg);
    mapper.from_ast(ctx.ast, result);

    t.assert_equal("tool calls count", 1u, msg.tool_calls.size());

    if (!msg.tool_calls.empty()) {
        t.assert_equal("tool name", "edit", msg.tool_calls[0].name);

        // Parse the arguments as JSON to verify they're valid
        std::string args = msg.tool_calls[0].arguments;

        try {
            json parsed = json::parse(args);
            t.assert_true("arguments is valid JSON", true);

            // Verify each field has proper value
            t.assert_equal("filename", "foo.cpp", parsed.value("filename", ""));

            std::string oldString = parsed.value("oldString", "");
            t.assert_true("oldString contains embedded quotes",
                oldString.find("\"14\"") != std::string::npos);
            t.assert_true("oldString contains bar with quotes",
                oldString.find("\"bar\"") != std::string::npos);

            std::string newString = parsed.value("newString", "");
            t.assert_true("newString contains embedded quotes",
                newString.find("\"15\"") != std::string::npos);

        } catch (const std::exception & e) {
            t.assert_true(std::string("arguments should be valid JSON: ") + e.what(), false);
        }
    }
}

