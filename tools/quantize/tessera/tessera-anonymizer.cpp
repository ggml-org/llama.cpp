//
// tessera-anonymizer.cpp
//
// Tier-2 escalation anonymizer: a pure text transform that scrubs a source
// payload while keeping its reasoning value. See tessera-anonymizer.h and
// docs/self-improving-loop-design.md sections 4.1 / 4.5 / Phase 5 / R9.
//
// Pseudonyms are deterministic (keyed by order of first appearance) and
// type-preserving: the case-shape of the original is kept. Each pseudonym
// carries a reserved "ts" prefix so it is unmistakably synthetic and can
// never collide with a whitelisted public identifier, which makes local
// de-anonymization (whole-identifier replacement of map keys) unambiguous.
// The shape render of the base token "ts<letters>" is:
//   ALL_CAPS original -> "TS<LETTERS>"   (e.g. MAX_BUF -> TSA)
//   Title    original -> "Ts<letters>"   (e.g. MyStruct -> Tsb)
//   lower/mixed       -> "ts<letters>"   (e.g. compute  -> tsc)
// Length is kept short and deterministic rather than matched exactly; the
// case-shape (the part that carries naming-convention semantics) is exact.
//

#include "tessera-anonymizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>

using json = nlohmann::json;

static const char * TS_ANON_MAP_SCHEMA = "llama.tessera.anonymizer.v1";

void ts_anon_default_params(struct ts_anon_params * p) {
    if (!p) return;
    p->level    = TS_ANON_BALANCED;  // learning.anonymizerAggressiveness default
    p->emit_map = false;
}

int ts_anon_level_from_string(const char * s, enum ts_anon_level * out) {
    if (!s || !out) return -1;
    const std::string v = s;
    if (v == "light")      { *out = TS_ANON_LIGHT;      return 0; }
    if (v == "balanced")   { *out = TS_ANON_BALANCED;   return 0; }
    if (v == "aggressive") { *out = TS_ANON_AGGRESSIVE; return 0; }
    return -1;
}

const char * ts_anon_level_to_string(enum ts_anon_level level) {
    switch (level) {
        case TS_ANON_LIGHT:      return "light";
        case TS_ANON_BALANCED:   return "balanced";
        case TS_ANON_AGGRESSIVE: return "aggressive";
        default:                 return "balanced";
    }
}

// Portable strdup-equivalent (strdup is POSIX; _strdup on Windows).
static char * ts_anon_dup(const std::string & s) {
    char * p = (char *) malloc(s.size() + 1);
    if (!p) return NULL;
    memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

// Bijective base-26: 0->a, 25->z, 26->aa, 27->ab, ... Collision-free per
// index, so distinct symbols (distinct appearance indices) never collide.
static std::string ts_anon_index_letters(int n) {
    std::string s;
    n += 1;
    while (n > 0) {
        s.push_back((char) ('a' + (n - 1) % 26));
        n = (n - 1) / 26;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

enum ts_anon_shape { TS_SHAPE_LOWER, TS_SHAPE_UPPER, TS_SHAPE_TITLE };

// Classify an identifier's case-shape. Digits/underscores are ignored for
// the classification; camelCase and other mixed forms collapse to LOWER.
static enum ts_anon_shape ts_anon_detect_shape(const std::string & id) {
    int  first_alpha = -1;
    bool any_lower   = false;
    for (size_t k = 0; k < id.size(); k++) {
        const char c = id[k];
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
            if (first_alpha < 0) first_alpha = (int) k;
            if (c >= 'a' && c <= 'z') any_lower = true;
        }
    }
    if (first_alpha < 0) return TS_SHAPE_LOWER;  // digits/underscore only
    const bool first_upper = (id[first_alpha] >= 'A' && id[first_alpha] <= 'Z');
    if (first_upper && !any_lower) return TS_SHAPE_UPPER;
    if (first_upper) {
        bool upper_later = false;
        for (size_t k = first_alpha + 1; k < id.size(); k++) {
            if (id[k] >= 'A' && id[k] <= 'Z') { upper_later = true; break; }
        }
        if (!upper_later) return TS_SHAPE_TITLE;
    }
    return TS_SHAPE_LOWER;
}

static std::string ts_anon_render(int index, enum ts_anon_shape shape) {
    std::string core = "ts" + ts_anon_index_letters(index);
    if (shape == TS_SHAPE_UPPER) {
        for (char & c : core) {
            if (c >= 'a' && c <= 'z') c = (char) (c - 'a' + 'A');
        }
    } else if (shape == TS_SHAPE_TITLE) {
        if (!core.empty() && core[0] >= 'a' && core[0] <= 'z') {
            core[0] = (char) (core[0] - 'a' + 'A');
        }
    }
    return core;
}

// Public-aware whitelist: language keywords, preprocessor directive names,
// and common stdlib/libc identifiers that carry no project identity, so
// renaming them would only damage reasoning fidelity. Intentionally
// conservative: project/library-specific public APIs (e.g. ggml_*/llama_*)
// are NOT listed and therefore ARE renamed - when in doubt, scrub.
static bool ts_anon_is_public(const std::string & id) {
    static const std::unordered_set<std::string> kPublic = {
        // C keywords
        "auto", "break", "case", "char", "const", "continue", "default", "do",
        "double", "else", "enum", "extern", "float", "for", "goto", "if",
        "inline", "int", "long", "register", "restrict", "return", "short",
        "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
        "unsigned", "void", "volatile", "while",
        // C11
        "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Generic", "_Noreturn",
        "_Static_assert", "_Thread_local",
        // C++ keywords
        "bool", "catch", "class", "constexpr", "const_cast", "delete",
        "dynamic_cast", "explicit", "false", "final", "friend", "mutable",
        "namespace", "new", "noexcept", "nullptr", "operator", "override",
        "private", "protected", "public", "reinterpret_cast", "static_assert",
        "static_cast", "template", "this", "throw", "true", "try", "typename",
        "using", "virtual",
        // common fixed-width / libc types
        "size_t", "ssize_t", "ptrdiff_t", "intptr_t", "uintptr_t",
        "int8_t", "int16_t", "int32_t", "int64_t",
        "uint8_t", "uint16_t", "uint32_t", "uint64_t",
        "FILE", "wchar_t", "char16_t", "char32_t",
        // preprocessor directive names (keep #include etc. readable)
        "define", "elif", "endif", "error", "ifdef", "ifndef",
        "include", "line", "pragma", "undef", "once", "defined",
        // ubiquitous libc / std staples
        "printf", "fprintf", "sprintf", "snprintf", "vsnprintf",
        "malloc", "calloc", "realloc", "free",
        "memcpy", "memmove", "memset", "memcmp",
        "strlen", "strcmp", "strncmp", "strcpy", "strncpy", "strcat",
        "strchr", "strstr", "strdup",
        "atoi", "atol", "atof", "abs", "exit", "abort",
        "sqrt", "pow", "sin", "cos", "tan", "exp", "log", "floor", "ceil", "fabs",
        "std", "string", "vector", "map", "set", "array", "pair", "tuple",
        "unordered_map", "unordered_set", "shared_ptr", "unique_ptr",
        "make_shared", "make_unique", "cout", "cerr", "endl",
        // ubiquitous entry-point / constants
        "main", "argc", "argv", "NULL",
    };
    return kPublic.find(id) != kPublic.end();
}

static bool ts_anon_ident_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool ts_anon_ident_char(char c) {
    return ts_anon_ident_start(c) || (c >= '0' && c <= '9');
}

// Superset of identifier chars used to sniff path-like runs (foo/bar.c).
static bool ts_anon_path_char(char c) {
    return ts_anon_ident_char(c) || c == '.' || c == '/' || c == '~' || c == '-';
}

static bool ts_anon_digit(char c) { return c >= '0' && c <= '9'; }

static bool ts_anon_hexdig(char c) {
    return ts_anon_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

// Scan one numeric literal starting at in[i]; sets *is_float. Handles hex
// (0x), binary (0b), decimal, fractional, exponent, and integer/float
// suffixes. Best-effort on exotic forms.
static size_t ts_anon_scan_number(const std::string & in, size_t i, bool * is_float) {
    const size_t n = in.size();
    size_t j = i;
    *is_float = false;

    if (in[j] == '0' && j + 1 < n && (in[j + 1] == 'x' || in[j + 1] == 'X')) {
        j += 2;
        while (j < n && (ts_anon_hexdig(in[j]) || in[j] == '\'')) j++;
    } else if (in[j] == '0' && j + 1 < n && (in[j + 1] == 'b' || in[j + 1] == 'B')) {
        j += 2;
        while (j < n && (in[j] == '0' || in[j] == '1' || in[j] == '\'')) j++;
    } else {
        while (j < n && (ts_anon_digit(in[j]) || in[j] == '\'')) j++;
        if (j < n && in[j] == '.') {
            *is_float = true;
            j++;
            while (j < n && (ts_anon_digit(in[j]) || in[j] == '\'')) j++;
        }
        if (j < n && (in[j] == 'e' || in[j] == 'E')) {
            *is_float = true;
            j++;
            if (j < n && (in[j] == '+' || in[j] == '-')) j++;
            while (j < n && ts_anon_digit(in[j])) j++;
        }
    }
    if (j < n && (in[j] == 'p' || in[j] == 'P')) {  // hex-float exponent
        *is_float = true;
        j++;
        if (j < n && (in[j] == '+' || in[j] == '-')) j++;
        while (j < n && ts_anon_digit(in[j])) j++;
    }
    while (j < n && (in[j] == 'u' || in[j] == 'U' || in[j] == 'l' ||
                     in[j] == 'L' || in[j] == 'f' || in[j] == 'F')) j++;
    return j;
}

static void ts_anon_scan(const std::string & in,
                         enum ts_anon_level level,
                         std::string & out,
                         std::unordered_map<std::string, std::string> & sym2pseudo,
                         int & counter) {
    const size_t n = in.size();
    const bool scrub_text   = (level >= TS_ANON_BALANCED);    // strings + comments
    const bool scrub_heavy  = (level >= TS_ANON_AGGRESSIVE);  // numbers + paths

    auto rename_ident = [&](const std::string & id) -> std::string {
        if (ts_anon_is_public(id)) return id;
        auto it = sym2pseudo.find(id);
        if (it != sym2pseudo.end()) return it->second;
        const std::string pseudo = ts_anon_render(counter++, ts_anon_detect_shape(id));
        sym2pseudo.emplace(id, pseudo);
        return pseudo;
    };

    size_t i = 0;
    while (i < n) {
        const char c = in[i];

        // line comment
        if (c == '/' && i + 1 < n && in[i + 1] == '/') {
            size_t j = i;
            while (j < n && in[j] != '\n') j++;
            out += scrub_text ? "// <comment>" : in.substr(i, j - i);
            i = j;
            continue;
        }

        // block comment
        if (c == '/' && i + 1 < n && in[i + 1] == '*') {
            size_t j = i + 2;
            while (j + 1 < n && !(in[j] == '*' && in[j + 1] == '/')) j++;
            const size_t end = (j + 1 < n) ? j + 2 : n;
            out += scrub_text ? "/* <comment> */" : in.substr(i, end - i);
            i = end;
            continue;
        }

        // string literal
        if (c == '"') {
            size_t j = i + 1;
            while (j < n && in[j] != '"') {
                if (in[j] == '\\' && j + 1 < n) j++;
                j++;
            }
            const size_t end = (j < n) ? j + 1 : n;
            out += scrub_text ? "\"<str>\"" : in.substr(i, end - i);
            i = end;
            continue;
        }

        // char literal
        if (c == '\'') {
            size_t j = i + 1;
            while (j < n && in[j] != '\'') {
                if (in[j] == '\\' && j + 1 < n) j++;
                j++;
            }
            const size_t end = (j < n) ? j + 1 : n;
            out += scrub_text ? "'<c>'" : in.substr(i, end - i);
            i = end;
            continue;
        }

        // identifier, or path-like run at aggressive level
        if (ts_anon_ident_start(c)) {
            size_t j = i;
            while (j < n && ts_anon_path_char(in[j])) j++;
            const std::string run = in.substr(i, j - i);
            if (scrub_heavy && run.find('/') != std::string::npos) {
                out += "<path>";  // e.g. src/foo/bar.c, include/foo.h
                i = j;
                continue;
            }
            // not a path: re-read a strict identifier from i
            size_t k = i;
            while (k < n && ts_anon_ident_char(in[k])) k++;
            out += rename_ident(in.substr(i, k - i));
            i = k;
            continue;
        }

        // numeric literal
        if (ts_anon_digit(c) || (c == '.' && i + 1 < n && ts_anon_digit(in[i + 1]))) {
            bool is_float = false;
            const size_t j = ts_anon_scan_number(in, i, &is_float);
            if (scrub_heavy) {
                out += is_float ? "0.0" : "0";
            } else {
                out += in.substr(i, j - i);
            }
            i = j;
            continue;
        }

        // path-like run starting with '/' or '~' at aggressive level (a '/'
        // here is not a comment start). A lone '/' (division) falls through.
        if (scrub_heavy && (c == '/' || c == '~')) {
            size_t j = i;
            while (j < n && ts_anon_path_char(in[j])) j++;
            if (j > i + 1) {
                out += "<path>";
                i = j;
                continue;
            }
        }

        // anything else: punctuation, whitespace, operators - verbatim
        out.push_back(c);
        i++;
    }
}

int ts_anonymize_run(const struct ts_anon_params * p,
                     const char * input_text,
                     char ** output_text,
                     char ** map_json) {
    if (map_json) *map_json = NULL;
    if (!p || !input_text || !output_text) return -1;
    *output_text = NULL;

    std::string out;
    std::unordered_map<std::string, std::string> sym2pseudo;
    int counter = 0;
    ts_anon_scan(std::string(input_text), p->level, out, sym2pseudo, counter);

    *output_text = ts_anon_dup(out);
    if (!*output_text) return -1;

    if (p->emit_map && map_json) {
        json j;
        j["schema"] = TS_ANON_MAP_SCHEMA;
        j["level"]  = ts_anon_level_to_string(p->level);
        json symbols = json::object();
        for (const auto & kv : sym2pseudo) {
            symbols[kv.second] = kv.first;  // pseudonym -> original
        }
        j["symbols"] = symbols;
        *map_json = ts_anon_dup(j.dump(2));
        if (!*map_json) {
            free(*output_text);
            *output_text = NULL;
            return -1;
        }
    }
    return 0;
}
