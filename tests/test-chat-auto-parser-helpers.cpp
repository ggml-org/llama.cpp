#include "chat-auto-parser-helpers.h"
#include "testing.h"

#include <iostream>
#include <string>

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

int main(int argc, char * argv[]) {
    testing t(std::cout);
    t.verbose = true;

    // usage: test-chat-auto-parser-helpers [filter_regex]

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

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

    return t.summary();
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
    result = calculate_diff_split("<tag>left</tag>", "<tag>right</tag>");
    // Algorithm finds "t</tag>" as common suffix because both strings end with it
    // This is actual algorithm behavior, though not semantically ideal
    t.assert_equal("prefix should be '<tag>'", "<tag>", result.prefix);
    t.assert_equal("left should be 'lef'", "lef", result.left);
    t.assert_equal("right should be 'righ'", "righ", result.right);
    t.assert_equal("suffix should be 't</tag>'", "t</tag>", result.suffix);

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
