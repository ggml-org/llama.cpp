//
// test_anonymizer.cpp
//
// Smoke test for the tier-2 anonymizer (ts_anonymize_run). Covers the five
// contract guarantees: consistency (same symbol -> same pseudonym),
// public-awareness (whitelisted identifiers unchanged), type/shape
// preservation (case-shape kept), level escalation (aggressive scrubs more
// than light), and exact round-trip de-anonymization via the emitted map.
//

#include "tessera-anonymizer.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

using json = nlohmann::json;

static int g_fail = 0;

static void check(const char * name, bool ok) {
    if (!ok) {
        std::printf("FAIL %s\n", name);
        g_fail++;
    } else {
        std::printf("ok   %s\n", name);
    }
}

static bool is_id_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool is_id_char(char c) {
    return is_id_start(c) || (c >= '0' && c <= '9');
}

// Count whole-identifier occurrences of tok in text (mirrors the engine's
// identifier rule so substring accidents like "tsa" in "tsaa" don't count).
static int count_whole_tokens(const std::string & text, const std::string & tok) {
    int count = 0;
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        if (is_id_start(text[i])) {
            size_t j = i;
            while (j < n && is_id_char(text[j])) j++;
            if (text.substr(i, j - i) == tok) count++;
            i = j;
        } else {
            i++;
        }
    }
    return count;
}

// Local de-anonymization: whole-identifier replacement of each map key with
// its original value. This is the exact contract the Swift service uses.
static std::string deanonymize(const std::string & text, const json & symbols) {
    std::string out;
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        if (is_id_start(text[i])) {
            size_t j = i;
            while (j < n && is_id_char(text[j])) j++;
            const std::string tok = text.substr(i, j - i);
            if (symbols.contains(tok)) {
                out += symbols.at(tok).get<std::string>();
            } else {
                out += tok;
            }
            i = j;
        } else {
            out.push_back(text[i]);
            i++;
        }
    }
    return out;
}

// Find the pseudonym key that maps to a given original ("" if absent).
static std::string key_for_value(const json & symbols, const std::string & orig) {
    for (auto it = symbols.begin(); it != symbols.end(); ++it) {
        if (it.value().get<std::string>() == orig) return it.key();
    }
    return "";
}

static bool all_upper(const std::string & s) {
    bool any = false;
    for (char c : s) {
        if (c >= 'a' && c <= 'z') return false;
        if (c >= 'A' && c <= 'Z') any = true;
    }
    return any;
}

static bool all_lower(const std::string & s) {
    bool any = false;
    for (char c : s) {
        if (c >= 'A' && c <= 'Z') return false;
        if (c >= 'a' && c <= 'z') any = true;
    }
    return any;
}

static bool is_title(const std::string & s) {
    if (s.empty() || !(s[0] >= 'A' && s[0] <= 'Z')) return false;
    for (size_t k = 1; k < s.size(); k++) {
        if (s[k] >= 'A' && s[k] <= 'Z') return false;
    }
    return true;
}

static int run_anon(enum ts_anon_level level, bool emit_map,
                    const std::string & input, std::string & out, std::string & map_str) {
    ts_anon_params p;
    ts_anon_default_params(&p);
    p.level    = level;
    p.emit_map = emit_map;
    char * o = NULL;
    char * m = NULL;
    const int rc = ts_anonymize_run(&p, input.c_str(), &o, &m);
    if (rc != 0) {
        free(o);
        free(m);
        return rc;
    }
    out     = o ? o : "";
    map_str = m ? m : "";
    free(o);
    free(m);
    return 0;
}

// A small source snippet with user symbols of each case-shape, a whitelisted
// keyword set, a string literal, a comment, and numeric constants.
static const char * kSrc =
    "// compute the sum for Acme Corp\n"
    "#include <stdio.h>\n"
    "#define MAX_BUF 1024\n"
    "struct Widget {\n"
    "    int count;\n"
    "};\n"
    "int addValues(int a, int b) {\n"
    "    const char * tag = \"secret-project-x\";\n"
    "    int total = a + b + 42;\n"
    "    printf(\"%s: %d\\n\", tag, total);\n"
    "    return total;\n"
    "}\n";

int main() {
    const std::string src = kSrc;

    // ------------------------------------------------------------------
    // Case 1: consistency - same symbol -> same pseudonym everywhere
    // ------------------------------------------------------------------
    {
        std::string out, map_str;
        check("case1: light rc == 0", run_anon(TS_ANON_LIGHT, true, src, out, map_str) == 0);
        const json symbols = json::parse(map_str).at("symbols");

        // "total" appears 3x in the source; it is renamed to ONE pseudonym
        // reused 3x, and the original no longer appears as a whole token.
        const std::string pseudo = key_for_value(symbols, "total");
        check("case1: total has a pseudonym", !pseudo.empty());
        check("case1: original total gone", count_whole_tokens(out, "total") == 0);
        check("case1: pseudonym reused 3x", count_whole_tokens(out, pseudo) == 3);
    }

    // ------------------------------------------------------------------
    // Case 2: public-awareness - whitelisted identifiers unchanged
    // ------------------------------------------------------------------
    {
        std::string out, map_str;
        run_anon(TS_ANON_AGGRESSIVE, true, src, out, map_str);
        const json symbols = json::parse(map_str).at("symbols");

        check("case2: int kept",     count_whole_tokens(out, "int")    == count_whole_tokens(src, "int"));
        check("case2: return kept",  count_whole_tokens(out, "return") == 1);
        check("case2: printf kept",  count_whole_tokens(out, "printf") == 1);
        check("case2: struct kept",  count_whole_tokens(out, "struct") == 1);
        check("case2: include kept", count_whole_tokens(out, "include") == 1);
        // whitelisted names are never keys in the map
        check("case2: int not mapped",     key_for_value(symbols, "int").empty());
        check("case2: printf not mapped",  key_for_value(symbols, "printf").empty());
    }

    // ------------------------------------------------------------------
    // Case 3: type/shape preservation - case-shape kept per symbol
    // ------------------------------------------------------------------
    {
        std::string out, map_str;
        run_anon(TS_ANON_LIGHT, true, src, out, map_str);
        const json symbols = json::parse(map_str).at("symbols");

        const std::string k_caps  = key_for_value(symbols, "MAX_BUF");    // ALL_CAPS
        const std::string k_title = key_for_value(symbols, "Widget");     // Title
        const std::string k_camel = key_for_value(symbols, "addValues");  // camel -> lower
        check("case3: ALL_CAPS -> upper", !k_caps.empty()  && all_upper(k_caps));
        check("case3: Title -> title",    !k_title.empty() && is_title(k_title));
        check("case3: camel -> lower",    !k_camel.empty() && all_lower(k_camel));

        // reserved-prefix invariant: every pseudonym starts with "ts"
        // (case-insensitive), which is what makes round-trip unambiguous.
        bool prefixed = true;
        for (auto it = symbols.begin(); it != symbols.end(); ++it) {
            const std::string k = it.key();
            if (k.size() < 2) { prefixed = false; break; }
            const char a = (k[0] >= 'A' && k[0] <= 'Z') ? (char) (k[0] - 'A' + 'a') : k[0];
            const char b = (k[1] >= 'A' && k[1] <= 'Z') ? (char) (k[1] - 'A' + 'a') : k[1];
            if (a != 't' || b != 's') { prefixed = false; break; }
        }
        check("case3: all pseudonyms ts-prefixed", prefixed);
    }

    // ------------------------------------------------------------------
    // Case 4: level escalation - aggressive scrubs more than light
    // ------------------------------------------------------------------
    {
        std::string light_out, light_map;
        std::string agg_out,   agg_map;
        run_anon(TS_ANON_LIGHT,      true, src, light_out, light_map);
        run_anon(TS_ANON_AGGRESSIVE, true, src, agg_out,   agg_map);

        // light keeps strings, comments, and numbers verbatim
        check("case4: light keeps string",  light_out.find("secret-project-x") != std::string::npos);
        check("case4: light keeps comment", light_out.find("Acme")             != std::string::npos);
        check("case4: light keeps number",  light_out.find("42")               != std::string::npos);

        // aggressive scrubs all three
        check("case4: agg scrubs string",  agg_out.find("secret-project-x") == std::string::npos);
        check("case4: agg scrubs comment", agg_out.find("Acme")             == std::string::npos);
        check("case4: agg scrubs number",  agg_out.find("42")               == std::string::npos);
        check("case4: agg emits <str>",     agg_out.find("<str>")     != std::string::npos);
        check("case4: agg emits <comment>", agg_out.find("<comment>") != std::string::npos);
    }

    // ------------------------------------------------------------------
    // Case 5: round-trip de-anonymization via the emitted map (exact at light)
    // ------------------------------------------------------------------
    {
        std::string out, map_str;
        run_anon(TS_ANON_LIGHT, true, src, out, map_str);
        const json j = json::parse(map_str);
        check("case5: map schema", j.at("schema").get<std::string>() == "llama.tessera.anonymizer.v1");
        check("case5: map level",  j.at("level").get<std::string>()  == "light");
        const std::string restored = deanonymize(out, j.at("symbols"));
        check("case5: round-trip exact", restored == src);
    }

    // ------------------------------------------------------------------
    // Case 6: API contract - error handling + emit_map gating + level parse
    // ------------------------------------------------------------------
    {
        char * o = NULL;
        char * m = NULL;
        ts_anon_params p;
        ts_anon_default_params(&p);
        check("case6: null input rc == -1", ts_anonymize_run(&p, NULL, &o, &m) == -1);

        // emit_map false -> map_json stays NULL
        std::string out, map_str;
        run_anon(TS_ANON_BALANCED, false, src, out, map_str);
        check("case6: no map when emit_map false", map_str.empty());

        enum ts_anon_level lvl;
        check("case6: parse balanced",   ts_anon_level_from_string("balanced", &lvl) == 0 && lvl == TS_ANON_BALANCED);
        check("case6: parse aggressive", ts_anon_level_from_string("aggressive", &lvl) == 0 && lvl == TS_ANON_AGGRESSIVE);
        check("case6: reject unknown",   ts_anon_level_from_string("extreme", &lvl) != 0);
    }

    std::printf("\n%s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
