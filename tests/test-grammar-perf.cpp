#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../src/unicode.h"
#include "../src/llama-grammar.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
static long get_peak_rss_kb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
    return usage.ru_maxrss / 1024; // macOS reports bytes
#else
    return usage.ru_maxrss;        // Linux reports KB
#endif
}
#else
static long get_peak_rss_kb() { return 0; }
#endif

// ---------------------------------------------------------------------------
// Token/piece helpers (same as test-grammar-integration.cpp)
// ---------------------------------------------------------------------------

struct token_and_piece {
    llama_token token;
    std::string piece;
};

static std::vector<token_and_piece> parse_tokens(const std::string & input) {
    std::vector<token_and_piece> result;
    result.reserve(input.size());
    size_t offset = 0;
    while (offset < input.size()) {
        try {
            if (static_cast<unsigned char>(input[offset]) == 0xff) {
                if (offset + 5 > input.size()) {
                    throw std::runtime_error("not enough bytes for token id");
                }
                uint32_t val =
                    (static_cast<unsigned char>(input[offset + 1]) << 24) |
                    (static_cast<unsigned char>(input[offset + 2]) << 16) |
                    (static_cast<unsigned char>(input[offset + 3]) << 8)  |
                    (static_cast<unsigned char>(input[offset + 4]));
                auto piece = "<[" + std::to_string(val) + "]>";
                result.push_back({static_cast<llama_token>(val), piece});
                offset += 5;
            } else {
                uint32_t cpt = unicode_cpt_from_utf8(input, offset);
                result.push_back({0, unicode_cpt_to_utf8(cpt)});
            }
        } catch (const std::invalid_argument &) {
            ++offset;
            result.push_back({0, unicode_cpt_to_utf8(0xFFFD)});
        }
    }
    return result;
}

static bool match_string(const std::string & input, llama_grammar * grammar) {
    const auto parsed = parse_tokens(input);
    auto & stacks_cur = llama_grammar_get_stacks(grammar);

    for (const auto & in : parsed) {
        try {
            llama_grammar_accept_token(*grammar, in.token, in.piece);
        } catch (const std::runtime_error &) {
            return false;
        }
        if (stacks_cur.empty()) {
            return false;
        }
    }

    for (const auto & stack : stacks_cur) {
        if (stack.empty()) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Grammar generation — 50 tool schemas with parallel tool calls
// ---------------------------------------------------------------------------

// Each tool has: name, 2-4 typed parameters (string/integer/boolean/number).
// The grammar enforces JSON structure for each tool's arguments.

struct tool_param {
    std::string name;
    std::string type; // "string", "integer", "boolean", "number"
};

struct tool_def {
    std::string name;
    std::vector<tool_param> params;
};

// GBNF identifiers can't contain underscores; replace with hyphens
static std::string gbnf_name(const std::string & name) {
    std::string result = name;
    for (auto & c : result) {
        if (c == '_') c = '-';
    }
    return result;
}

static std::string json_rule_for_type(const std::string & type) {
    if (type == "string")  return "string";
    if (type == "integer") return "integer";
    if (type == "boolean") return "boolean";
    if (type == "number")  return "number";
    return "string";
}

static std::string generate_grammar(const std::vector<tool_def> & tools) {
    std::string g;

    // Primitives
    g += "# Primitive types\n";
    g += "ws ::= [ \\t\\n]*\n";
    g += "string ::= \"\\\"\" [^\"\\\\]* \"\\\"\" \n";
    g += "integer ::= \"-\"? [0-9]+\n";
    g += "number ::= \"-\"? [0-9]+ (\".\" [0-9]+)?\n";
    g += "boolean ::= \"true\" | \"false\"\n";
    g += "\n";

    // Root: parallel tool calls
    g += "root ::= \"<tool_calls>\" ws tool-call+ ws \"</tool_calls>\"\n";
    g += "tool-call ::= ws \"<tool>\" ws tool-body ws \"</tool>\" ws\n";
    g += "tool-body ::= ";

    for (size_t i = 0; i < tools.size(); ++i) {
        if (i > 0) g += " | ";
        g += "tool-" + gbnf_name(tools[i].name);
    }
    g += "\n\n";

    // Per-tool rules
    for (const auto & tool : tools) {
        g += "tool-" + gbnf_name(tool.name) + " ::= \"{\" ws ";
        g += "\"\\\"name\\\": \\\"" + tool.name + "\\\",\" ws ";
        g += "\"\\\"arguments\\\": {\" ws ";

        for (size_t j = 0; j < tool.params.size(); ++j) {
            if (j > 0) g += "\",\" ws ";
            g += "\"\\\"" + tool.params[j].name + "\\\": \" ";
            g += json_rule_for_type(tool.params[j].type) + " ws ";
        }

        g += "\"}\" ws \"}\" \n";
    }

    return g;
}

// Generate a grammar with high ambiguity: many alternatives share a long common
// prefix so the grammar engine must maintain many parallel stacks simultaneously.
// This stresses advance_stack deduplication heavily.
static std::string generate_ambiguous_grammar(int n_tools) {
    std::string g;

    g += "ws ::= [ \\t\\n]*\n";
    g += "string ::= \"\\\"\" [^\"\\\\]* \"\\\"\" \n";
    g += "integer ::= \"-\"? [0-9]+\n";
    g += "boolean ::= \"true\" | \"false\"\n";
    g += "\n";

    // Each tool shares the same outer structure; disambiguation happens only
    // at the very end of the arguments object where a distinguishing key appears.
    // This forces the grammar to track all n_tools stacks in parallel through
    // the entire shared prefix.
    g += "root ::= \"{\" ws \"\\\"tool\\\":\" ws tool-choice ws \"}\"\n";
    g += "tool-choice ::= ";
    for (int i = 0; i < n_tools; ++i) {
        if (i > 0) g += " | ";
        g += "tc" + std::to_string(i);
    }
    g += "\n\n";

    // All tools share: {"name": "...", "args": {"shared_a": <string>, "shared_b": <int>, ..., "unique_X": <val>}}
    // The shared prefix is long (several key-value pairs), the unique key differs per tool.
    for (int i = 0; i < n_tools; ++i) {
        g += "tc" + std::to_string(i) + " ::= ";
        g += "\"{\" ws ";
        g += "\"\\\"name\\\": \\\"tool\\\"\" ws \",\" ws ";           // same name for all!
        g += "\"\\\"args\\\": {\" ws ";
        // 5 shared key-value pairs
        g += "\"\\\"alpha\\\": \" string ws \",\" ws ";
        g += "\"\\\"beta\\\": \" integer ws \",\" ws ";
        g += "\"\\\"gamma\\\": \" boolean ws \",\" ws ";
        g += "\"\\\"delta\\\": \" string ws \",\" ws ";
        g += "\"\\\"epsilon\\\": \" integer ws \",\" ws ";
        // Unique key that disambiguates (only the key name differs)
        g += "\"\\\"unique-" + std::to_string(i) + "\\\": \" string ws ";
        g += "\"}\" ws \"}\" \n";
    }

    return g;
}

static std::string make_ambiguous_tool_call(int tool_idx) {
    std::string s;
    s += "{\"tool\": {\"name\": \"tool\", \"args\": {";
    s += "\"alpha\": \"hello\", ";
    s += "\"beta\": 123, ";
    s += "\"gamma\": true, ";
    s += "\"delta\": \"world\", ";
    s += "\"epsilon\": 456, ";
    s += "\"unique-" + std::to_string(tool_idx) + "\": \"done\"";
    s += "}}}";
    return s;
}

static std::vector<tool_def> make_tools(int n) {
    // Realistic-ish tool names and parameter patterns
    static const char * prefixes[] = {
        "search", "get", "create", "update", "delete",
        "list", "fetch", "send", "check", "run",
        "query", "remove", "modify", "inspect",
    };
    static const int n_prefixes = sizeof(prefixes) / sizeof(prefixes[0]);
    static const char * domains[] = {
        "users", "files", "messages", "tasks", "events",
        "logs", "configs", "sessions", "tokens", "roles",
        "alerts", "metrics", "reports", "schemas", "hooks",
    };
    static const int n_domains = sizeof(domains) / sizeof(domains[0]);

    // Param templates per tool type
    static const std::vector<std::vector<tool_param>> param_templates = {
        {{"query", "string"}, {"limit", "integer"}, {"offset", "integer"}},
        {{"id", "string"}, {"verbose", "boolean"}},
        {{"name", "string"}, {"description", "string"}, {"priority", "integer"}, {"active", "boolean"}},
        {{"id", "string"}, {"field", "string"}, {"value", "string"}},
        {{"id", "string"}},
    };

    std::vector<tool_def> tools;
    tools.reserve(n);

    for (int i = 0; i < n; ++i) {
        const char * prefix = prefixes[i % n_prefixes];
        const char * domain = domains[(i / n_prefixes) % n_domains];
        std::string name = std::string(prefix) + "_" + domain + "_" + std::to_string(i);

        const auto & params = param_templates[i % param_templates.size()];
        // Slightly vary param names per tool to increase grammar complexity
        std::vector<tool_param> tool_params;
        for (const auto & p : params) {
            tool_params.push_back({p.name + "_" + std::to_string(i), p.type});
        }
        tools.push_back({name, tool_params});
    }
    return tools;
}

static std::string make_valid_tool_call(const tool_def & tool) {
    std::string s;
    s += "{\"name\": \"" + tool.name + "\", \"arguments\": {";
    for (size_t j = 0; j < tool.params.size(); ++j) {
        if (j > 0) s += ", ";
        s += "\"" + tool.params[j].name + "\": ";
        if (tool.params[j].type == "string") {
            s += "\"example_value\"";
        } else if (tool.params[j].type == "integer") {
            s += "42";
        } else if (tool.params[j].type == "boolean") {
            s += "true";
        } else if (tool.params[j].type == "number") {
            s += "3.14";
        }
    }
    s += "}}";
    return s;
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

using hrclock = std::chrono::high_resolution_clock;

static double ms_since(hrclock::time_point t0) {
    return std::chrono::duration<double, std::milli>(hrclock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

static void bench_init(const std::string & grammar_str, int iters) {
    fprintf(stderr, "\n--- Grammar init benchmark (%d iterations) ---\n", iters);

    auto t0 = hrclock::now();
    for (int i = 0; i < iters; ++i) {
        auto * g = llama_grammar_init_impl(nullptr, grammar_str.c_str(), "root",
                                           false, nullptr, 0, nullptr, 0);
        assert(g != nullptr);
        llama_grammar_free_impl(g);
    }
    double elapsed = ms_since(t0);
    fprintf(stderr, "  Total: %.1f ms  |  Per-iter: %.3f ms\n", elapsed, elapsed / iters);
}

static void bench_match(const std::string & grammar_str,
                         const std::vector<std::string> & test_strings,
                         int iters) {
    fprintf(stderr, "\n--- String match benchmark (%d iterations x %zu strings) ---\n",
            iters, test_strings.size());

    auto * grammar = llama_grammar_init_impl(nullptr, grammar_str.c_str(), "root",
                                             false, nullptr, 0, nullptr, 0);
    assert(grammar != nullptr);

    const llama_grammar_stacks stacks_orig = llama_grammar_get_stacks(grammar);
    llama_grammar_stacks & stacks_cur = llama_grammar_get_stacks(grammar);

    // Warm-up: verify strings actually match
    for (const auto & s : test_strings) {
        bool ok = match_string(s, grammar);
        if (!ok) {
            fprintf(stderr, "  ERROR: string failed to match during warm-up!\n");
            fprintf(stderr, "  String (first 200 chars): %.200s\n", s.c_str());
            assert(false && "warm-up match failed");
        }
        stacks_cur = stacks_orig;
    }

    auto t0 = hrclock::now();
    for (int i = 0; i < iters; ++i) {
        for (const auto & s : test_strings) {
            match_string(s, grammar);
            stacks_cur = stacks_orig;
        }
    }
    double elapsed = ms_since(t0);
    double per_string = elapsed / (iters * test_strings.size());
    fprintf(stderr, "  Total: %.1f ms  |  Per-string: %.3f ms\n", elapsed, per_string);

    llama_grammar_free_impl(grammar);
}

static void bench_clone(const std::string & grammar_str, int iters) {
    fprintf(stderr, "\n--- Grammar clone benchmark (%d iterations) ---\n", iters);

    auto * grammar = llama_grammar_init_impl(nullptr, grammar_str.c_str(), "root",
                                             false, nullptr, 0, nullptr, 0);
    assert(grammar != nullptr);

    auto t0 = hrclock::now();
    for (int i = 0; i < iters; ++i) {
        auto * cloned = llama_grammar_clone_impl(*grammar);
        llama_grammar_free_impl(cloned);
    }
    double elapsed = ms_since(t0);
    fprintf(stderr, "  Total: %.1f ms  |  Per-iter: %.3f ms\n", elapsed, elapsed / iters);

    llama_grammar_free_impl(grammar);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    fprintf(stderr, "=== Grammar Performance Benchmark ===\n\n");

    long rss_start = get_peak_rss_kb();

    // ---------------------------------------------------------------
    // Benchmark 1: 50-tool grammar (low ambiguity, realistic)
    // ---------------------------------------------------------------
    {
        fprintf(stderr, "========================================\n");
        fprintf(stderr, "Benchmark 1: 50-tool grammar (low ambiguity)\n");
        fprintf(stderr, "========================================\n");

        auto tools = make_tools(50);
        std::string grammar_str = generate_grammar(tools);
        fprintf(stderr, "Grammar size: %zu bytes, %zu tools\n", grammar_str.size(), tools.size());

        std::vector<std::string> test_strings;
        for (int i = 0; i < 10; ++i) {
            std::string s = "<tool_calls> <tool> " + make_valid_tool_call(tools[i * 5]) + " </tool> </tool_calls>";
            test_strings.push_back(s);
        }
        for (int i = 0; i < 5; ++i) {
            std::string s = "<tool_calls>";
            int n_tools = 2 + (i % 3);
            for (int j = 0; j < n_tools; ++j) {
                s += " <tool> " + make_valid_tool_call(tools[(i * 7 + j * 3) % 50]) + " </tool>";
            }
            s += " </tool_calls>";
            test_strings.push_back(s);
        }

        size_t total_len = 0;
        for (const auto & s : test_strings) {
            total_len += s.size();
        }
        fprintf(stderr, "Test strings: %zu (avg length: %zu chars)\n",
                test_strings.size(), total_len / test_strings.size());

        bench_init(grammar_str, 200);
        bench_match(grammar_str, test_strings, 50);
        bench_clone(grammar_str, 5000);
    }

    // ---------------------------------------------------------------
    // Benchmark 2: High-ambiguity grammar (50 tools, shared prefix)
    // All alternatives share the same structure until the final key.
    // This forces the grammar to maintain 50 parallel stacks.
    // ---------------------------------------------------------------
    {
        fprintf(stderr, "\n========================================\n");
        fprintf(stderr, "Benchmark 2: 50-tool grammar (HIGH ambiguity)\n");
        fprintf(stderr, "========================================\n");

        std::string grammar_str = generate_ambiguous_grammar(50);
        fprintf(stderr, "Grammar size: %zu bytes\n", grammar_str.size());

        std::vector<std::string> test_strings;
        for (int i = 0; i < 15; ++i) {
            test_strings.push_back(make_ambiguous_tool_call(i * 3 % 50));
        }

        size_t total_len = 0;
        for (const auto & s : test_strings) {
            total_len += s.size();
        }
        fprintf(stderr, "Test strings: %zu (avg length: %zu chars)\n",
                test_strings.size(), total_len / test_strings.size());

        bench_init(grammar_str, 200);
        bench_match(grammar_str, test_strings, 50);
        bench_clone(grammar_str, 5000);
    }

    // ---------------------------------------------------------------
    // Benchmark 3: 120-tool high-ambiguity grammar
    // Same shared-prefix structure, but 120 parallel stacks.
    // ---------------------------------------------------------------
    {
        fprintf(stderr, "\n========================================\n");
        fprintf(stderr, "Benchmark 3: 120-tool grammar (HIGH ambiguity)\n");
        fprintf(stderr, "========================================\n");

        std::string grammar_str = generate_ambiguous_grammar(120);
        fprintf(stderr, "Grammar size: %zu bytes\n", grammar_str.size());

        std::vector<std::string> test_strings;
        test_strings.reserve(15);
        for (int i = 0; i < 15; ++i) {
            test_strings.push_back(make_ambiguous_tool_call(i * 7 % 120));
        }

        size_t total_len = 0;
        for (const auto & s : test_strings) {
            total_len += s.size();
        }
        fprintf(stderr, "Test strings: %zu (avg length: %zu chars)\n",
                test_strings.size(), total_len / test_strings.size());

        bench_init(grammar_str, 200);
        bench_match(grammar_str, test_strings, 50);
        bench_clone(grammar_str, 5000);
    }

    long rss_end = get_peak_rss_kb();
    fprintf(stderr, "\n--- Memory ---\n");
    fprintf(stderr, "  Peak RSS: %ld KB (delta from start: %ld KB)\n", rss_end, rss_end - rss_start);

    fprintf(stderr, "\n=== Benchmark complete ===\n");
    return 0;
}
