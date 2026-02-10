#include "common.h"

#include <cstdio>
#include <string>

#undef NDEBUG
#include <cassert>

static int n_tests  = 0;
static int n_failed = 0;

static const char SEP = DIRECTORY_SEPARATOR;

static void test_normalize(const char * desc, const std::string & expected, const std::string & input) {
    std::string result = fs_normalize_filepath(input);
    n_tests++;
    if (result != expected) {
        n_failed++;
        printf("  FAIL: %s (got \"%s\", expected \"%s\")\n", desc, result.c_str(), expected.c_str());
    }
}

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
    test("precomposed accent",      true,  "caf\xc3\xa9");                   // café (U+00E9)
    test("combining accent",        true,  "cafe\xcc\x81");                  // café (e + U+0301)
    test("japanese hiragana",       true,  "\xe3\x81\x82\xe3\x81\x84\xe3\x81\x86.txt"); // あいう.txt
    test("korean hangul",           true,  "\xed\x95\x9c\xea\xb8\x80.txt"); // 한글.txt
    test("max length (255 bytes)",  true,  std::string(255, 'a'));

    // --- Basic invalid filenames ---
    test("empty string",            false, "");
    test("over 255 bytes",          false, std::string(256, 'a'));
    test("just a dot",              false, ".");
    test("double dot",              false, "..");
    test("leading space",           false, " foo");
    test("trailing space",          false, "foo ");
    test("trailing dot",            false, "foo.");
    test("dot path",                false, "./././");

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
    test("replacement char U+FFFD", false, "foo\xef\xbf\xbd""bar");
    test("BOM U+FEFF",              false, "foo\xef\xbb\xbf""bar");

    // --- Windows bestfit characters (map to path traversal chars under WideCharToMultiByte) ---
    test("fullwidth solidus U+FF0F",    false, "foo\xef\xbc\x8f""bar"); // / on CP 874, 1250-1258
    test("fullwidth rev solidus U+FF3C",false, "foo\xef\xbc\xbc""bar"); // \ on CP 874, 1250-1258
    test("fullwidth colon U+FF1A",      false, "foo\xef\xbc\x9a""bar"); // : on CP 874, 1250-1258
    test("division slash U+2215",       false, "foo\xe2\x88\x95""bar"); // / on CP 1250, 1252, 1254
    test("set minus U+2216",            false, "foo\xe2\x88\x96""bar"); // \ on CP 1250, 1252, 1254
    test("fraction slash U+2044",       false, "foo\xe2\x81\x84""bar"); // / on CP 1250, 1252, 1254
    test("ratio U+2236",               false, "foo\xe2\x88\xb6""bar"); // : on CP 1250, 1252, 1254
    test("armenian full stop U+0589",   false, "foo\xd6\x89""bar");     // : on CP 1250, 1252, 1254
    test("yen sign U+00A5",             false, "foo\xc2\xa5""bar");     // \ on CP 932 (Japanese)
    test("won sign U+20A9",             false, "foo\xe2\x82\xa9""bar"); // \ on CP 949 (Korean)
    test("acute accent U+00B4",         false, "foo\xc2\xb4""bar");     // / on CP 1253 (Greek)

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
    test("dot path",                false, "./././",            true);

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

    // --- fs_normalize_filepath ---
    test_normalize("passthrough simple",        "foo.txt",                              "foo.txt");
    test_normalize("passthrough subdir",        std::string("foo") + SEP + "bar.txt",   "foo/bar.txt");
    test_normalize("backslash to sep",          std::string("foo") + SEP + "bar.txt",   "foo\\bar.txt");
    test_normalize("mixed separators",          std::string("a") + SEP + "b" + SEP + "c", "a/b\\c");
    test_normalize("duplicate slashes",         std::string("foo") + SEP + "bar",       "foo//bar");
    test_normalize("duplicate backslashes",     std::string("foo") + SEP + "bar",       "foo\\\\bar");
    test_normalize("triple slashes",            std::string("foo") + SEP + "bar",       "foo///bar");
    test_normalize("leading slash stripped",     "foo",                                  "/foo");
    test_normalize("leading backslash stripped", "foo",                                  "\\foo");
    test_normalize("multiple leading slashes",   "foo",                                 "///foo");
    test_normalize("leading dot-slash stripped",  "foo",                                "./foo");
    test_normalize("leading dot-backslash stripped", "foo",                             ".\\foo");
    test_normalize("deep path normalized",
        std::string("a") + SEP + "b" + SEP + "c" + SEP + "d.txt",
        "/a//b\\c/d.txt");

    // --- normalize doesn't validate and doesn't trim dot segments in the middle of the path ---
    test_normalize("dotdot retained",           std::string("foo") + SEP + ".." + SEP + "bar", "foo/../bar");
    test_normalize("dotdot at start retained",  std::string("..") + SEP + "bar",               "../bar");
    test_normalize("dotdot at end retained",    std::string("foo") + SEP + "..",               "foo/..");
    test_normalize("dot component retained mid", std::string("foo") + SEP + "." + SEP + "bar", "foo/./bar");

    if (n_failed) {
        printf("\n%d/%d tests failed\n", n_failed, n_tests);
        fflush(stdout);
        assert(false);
    }

    printf("OK\n");

    return 0;
}
