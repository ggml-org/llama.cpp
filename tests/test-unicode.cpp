#include "../src/unicode.h"

#include <cstdio>
#include <string>
#include <vector>

static int check(const char * name,
                 const std::string & text,
                 const std::vector<std::string> & regex_exprs,
                 const std::vector<std::string> & expected) {
    const auto actual = unicode_regex_split(text, regex_exprs, false);

    if (actual != expected) {
        fprintf(stderr, "%s: unexpected split:", name);
        for (const auto & piece : actual) {
            fprintf(stderr, " [%s]", piece.c_str());
        }
        fprintf(stderr, "\n");
        return 1;
    }

    return 0;
}

int main() {
    int nfail = 0;

    nfail += check("symbols", " ~foo", { "[~][A-Za-z]+| ?[\\p{S}]+|\\s+" }, { " ~", "foo" });

    // GPT4O uses a hand written splitter, std::regex overflows the stack on long runs
    // keep in sync with LLAMA_VOCAB_PRE_TYPE_GPT4O in src/llama-vocab.cpp
    const std::vector<std::string> gpt4o = {
        "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };

    nfail += check("gpt4o words", "Hello world!", gpt4o, { "Hello", " world", "!" });
    nfail += check("gpt4o case",  "getUserByID",  gpt4o, { "get", "User", "By", "ID" });

    // a run long enough to exhaust the stack of the std::regex based fallback
    const std::string long_run(1 << 17, 'x');
    nfail += check("gpt4o long run", long_run, gpt4o, { long_run });

    return nfail;
}
