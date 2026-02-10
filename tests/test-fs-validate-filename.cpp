#include "common.h"

#include <cstdio>
#include <string>

#undef NDEBUG
#include <cassert>

static int n_tests  = 0;
static int n_failed = 0;

static void test(const char * desc, bool expected, const std::string & filename, bool allow_subdirs = false) {
    bool result = fs_validate_filename(filename, allow_subdirs);
    n_tests++;
    if (result != expected) {
        n_failed++;
        printf("  FAIL: %s (got %s, expected %s)\n", desc,
               result ? "true" : "false", expected ? "true" : "false");
    }
}

int main(void) {
    // --- Basic valid filenames ---
    test("simple ascii",            true,  "hello.txt");
    test("no extension",            true,  "readme");
    test("multiple dots",           true,  "archive.tar.gz");
    test("leading dot (hidden)",    true,  ".gitignore");
    test("unicode filename",        true,  "\xc3\xa9\xc3\xa0\xc3\xbc.txt"); // éàü.txt
    test("max length (255 bytes)",  true,  std::string(255, 'a'));

    // --- Basic invalid filenames ---
    test("empty string",            false, "");
    test("over 255 bytes",          false, std::string(256, 'a'));
    test("just a dot",              false, ".");
    test("double dot",              false, "..");
    test("leading space",           false, " foo");
    test("trailing space",          false, "foo ");
    test("trailing dot",            false, "foo.");

    // --- Double dots ---
    test("contains double dot",     true,  "foo..bar");
    test("leading double dot",      true,  "..foo");
    test("trailing double dot",     false, "foo.."); // trailing dot

    // --- Control characters ---
    test("null byte",               false, std::string("foo\x00bar", 7));
    test("newline",                 false, "foo\nbar");
    test("tab",                     false, "foo\tbar");
    test("C0 control (0x01)",       false, "foo\x01""bar");
    test("DEL (0x7F)",              false, std::string("foo\x7f""bar"));
    test("C1 control (0x80)",       false, "foo\xc2\x80""bar"); // U+0080
    test("C1 control (0x9F)",       false, "foo\xc2\x9f""bar"); // U+009F

    // --- Illegal characters ---
    test("colon",                   false, "foo:bar");
    test("asterisk",                false, "foo*bar");
    test("question mark",           false, "foo?bar");
    test("double quote",            false, "foo\"bar");
    test("less than",               false, "foo<bar");
    test("greater than",            false, "foo>bar");
    test("pipe",                    false, "foo|bar");
    test("forward slash",           false, "foo/bar");
    test("backslash",               false, "foo\\bar");

    // --- Unicode special codepoints ---
    test("fullwidth period U+FF0E", false, "foo\xef\xbc\x8e""bar");
    test("division slash U+2215",   false, "foo\xe2\x88\x95""bar");
    test("set minus U+2216",        false, "foo\xe2\x88\x96""bar");
    test("replacement char U+FFFD", false, "foo\xef\xbf\xbd""bar");
    test("BOM U+FEFF",              false, "foo\xef\xbb\xbf""bar");

    // --- Invalid UTF-8 ---
    test("invalid continuation",    false, std::string("foo\x80""bar"));
    test("truncated sequence",      false, std::string("foo\xc3"));
    test("overlong slash (2-byte)", false, std::string("foo\xc0\xaf""bar", 6)); // U+002F as 2-byte
    test("overlong dot (2-byte)",   false, std::string("foo\xc0\xae""bar", 6)); // U+002E as 2-byte
    test("overlong 'a' (2-byte)",   false, std::string("foo\xc1\xa1""bar", 6)); // U+0061 as 2-byte
    test("overlong 'A' (2-byte)",   false, std::string("foo\xc1\x81""bar", 6)); // U+0041 as 2-byte
    test("overlong null (2-byte)",  false, std::string("foo\xc0\x80""bar", 6)); // U+0000 as 2-byte

    // --- Paths without allow_subdirs ---
    test("forward slash blocked",   false, "foo/bar");
    test("backslash blocked",       false, "foo\\bar");

    // --- Paths with allow_subdirs=true ---
    test("simple subdir",           true,  "foo/bar",           true);
    test("backslash subdir",        true,  "foo\\bar",          true);
    test("deep path",               true,  "a/b/c/d.txt",       true);
    test("trailing slash",          true,  "foo/bar/",          true);
    test("colon in path",           false, "foo/b:r/baz",       true);
    test("control char in path",    false, "foo/b\nar/baz",     true);

    // --- Leading separators ---
    test("leading slash",           false, "/foo/bar",          true);
    test("leading backslash",       false, "\\foo\\bar",        true);

    // --- Dotdot in paths ---
    test("leading dotdot in path",  false, "../bar",            true);
    test("dotdot in path",          false, "foo/../bar",        true);
    test("dotdot component leading",  true,  "foo/..bar/baz",   true);
    test("dotdot component middle",   true,  "foo/ba..r/baz",   true);
    test("dotdot component trailing", false, "foo/bar../baz",   true); // trailing dot

    // --- Per-component checks ---
    test("leading space in component",      false, "foo/ bar/baz",     true);
    test("trailing space in component",     false, "foo/bar /baz",     true);
    test("trailing dot in component",       false, "foo/bar./baz",     true);
    test("dot component in path",           false, "foo/./bar",        true);
    test("leading space after slash",       false, "foo/ bar",         true);
    test("trailing space before slash",     false, "bar /baz",         true);
    test("trailing dot before slash",       false, "bar./baz",         true);

    if (n_failed) {
        printf("\n%d/%d tests failed\n", n_failed, n_tests);
        fflush(stdout);
        assert(false);
    }

    printf("OK\n");

    return 0;
}
