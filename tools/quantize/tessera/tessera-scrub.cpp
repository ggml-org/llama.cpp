//
// tessera-scrub.cpp
//
// Heuristic secret redactor for text payloads (tier-2 egress privacy, R4).
// See tessera-scrub.h for the contract, guarantees, and known limits.
//
// Each rule is a (regex, placeholder) pair applied in order; matches are
// replaced with a stable type-labelled placeholder. Redaction is one-way and
// deterministic, so no map is produced. Rules are ordered structured-first
// (PEM blocks) then specific high-confidence tokens, then email/ip/path/env.
//

#include "tessera-scrub.h"

#include <cstdlib>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

// One redaction rule. `flags` lets a rule opt into case-insensitivity
// (used for Bearer headers and secret env names, which vary in case).
struct ts_scrub_spec {
    const char *          pattern;
    const char *          replacement;
    std::regex_constants::syntax_option_type flags;
};

static const std::regex_constants::syntax_option_type kECMA = std::regex::ECMAScript;
static const std::regex_constants::syntax_option_type kICASE =
    (std::regex_constants::syntax_option_type) (std::regex::ECMAScript | std::regex::icase);

// The rule table. Order matters: multi-line/structured first, then the
// specific token families, then the shape-based detectors. See the header
// for why each discriminator is shaped the way it is.
static const ts_scrub_spec kSpecs[] = {
    // PEM private-key blocks (multi-line), e.g. -----BEGIN RSA PRIVATE KEY-----
    { R"(-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----)",
      "<secret:pem>", kECMA },

    // AWS access key ids (AKIA permanent, ASIA temporary) + 16 uppercase alnum
    { R"(\b(?:AKIA|ASIA)[0-9A-Z]{16}\b)", "<secret:aws_key>", kECMA },

    // GitHub tokens: classic gh<pousr>_ (36+) and fine-grained github_pat_ (20+)
    { R"(\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{20,})\b)",
      "<secret:github_token>", kECMA },

    // Slack tokens: xox[baprs]-...
    { R"(\bxox[baprs]-[A-Za-z0-9-]{10,}\b)", "<secret:slack_token>", kECMA },

    // HTTP bearer tokens (case-insensitive header name)
    { R"(\bBearer [A-Za-z0-9._~+/=-]{20,})", "<secret:bearer>", kICASE },

    // Generic api keys. A HYPHEN is required after the prefix: C/C++
    // identifiers cannot contain a hyphen, so sk-.../key-.../token-... with a
    // long alnum tail is almost certainly a key string, not a symbol.
    { R"(\b(?:sk|pk|rk|key|token|apikey|api-key)-[A-Za-z0-9-]{20,}\b)",
      "<secret:api_key>", kECMA },

    // Email addresses
    { R"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", "<secret:email>", kECMA },

    // IPv6, full 8-group form only. Compressed forms (fe80::1) are skipped on
    // purpose: a '::' matcher would also hit the C++ scope operator std::foo.
    { R"(\b[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4}){7}\b)", "<secret:ipv6>", kECMA },

    // IPv4 with validated 0-255 octets; \b keeps a letter-prefixed tail
    // (v1.2.3.4) from matching.
    { R"(\b(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\b)",
      "<secret:ipv4>", kECMA },

    // Absolute Unix paths under a known top-level dir (avoids URL paths and
    // division), plus home-relative ~/...
    { R"(/(?:Users|home|etc|var|tmp|opt|srv|root|private|Library|Applications)\b[^\s"'<>|]*)",
      "<secret:path>", kECMA },
    { R"(~/[^\s"'<>|]+)", "<secret:path>", kECMA },

    // Windows drive paths, e.g. C:\Users\foo
    { R"([A-Za-z]:\\[^\s"'<>|]*)", "<secret:path>", kECMA },

    // Secret-looking env assignments: keep the (informative) name, mask the
    // value. The name must END with a secret suffix at a word boundary, so
    // API_KEY=v is caught but MAX_BUF=4096 and token_count=3 are not.
    { R"ENV(\b([A-Za-z_][A-Za-z0-9_]*(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|APIKEY)\b\s*[=:]\s*)([^\s"'<>|]+))ENV",
      "$1<secret:env>", kICASE },
};

struct ts_scrub_rule {
    std::regex  re;
    std::string replacement;
};

struct ts_scrub_ruleset {
    bool                       ok = false;
    std::vector<ts_scrub_rule> rules;
};

// Compile the table once. Any regex_error fails the whole set closed.
static ts_scrub_ruleset ts_scrub_build() {
    ts_scrub_ruleset rs;
    try {
        rs.rules.reserve(sizeof(kSpecs) / sizeof(kSpecs[0]));
        for (const ts_scrub_spec & spec : kSpecs) {
            rs.rules.push_back(ts_scrub_rule{ std::regex(spec.pattern, spec.flags), spec.replacement });
        }
        rs.ok = true;
    } catch (const std::regex_error &) {
        rs.ok = false;
        rs.rules.clear();
    }
    return rs;
}

static const ts_scrub_ruleset & ts_scrub_rules() {
    static const ts_scrub_ruleset rs = ts_scrub_build();  // thread-safe one-time init
    return rs;
}

// Portable strdup-equivalent (strdup is POSIX; _strdup on Windows).
static char * ts_scrub_dup(const std::string & s) {
    char * p = (char *) malloc(s.size() + 1);
    if (!p) return NULL;
    memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

int ts_scrub_run(const char * input_text, char ** output_text, int * n_redactions) {
    if (n_redactions) *n_redactions = 0;
    if (!input_text || !output_text) return -1;
    *output_text = NULL;

    const ts_scrub_ruleset & rs = ts_scrub_rules();
    if (!rs.ok) return -1;  // fail closed: never emit un-scrubbed text

    std::string work(input_text);
    int count = 0;
    try {
        for (const ts_scrub_rule & rule : rs.rules) {
            auto it  = std::sregex_iterator(work.begin(), work.end(), rule.re);
            auto end = std::sregex_iterator();
            count += (int) std::distance(it, end);
            work = std::regex_replace(work, rule.re, rule.replacement);
        }
    } catch (const std::regex_error &) {
        return -1;  // fail closed
    }

    *output_text = ts_scrub_dup(work);
    if (!*output_text) return -1;
    if (n_redactions) *n_redactions = count;
    return 0;
}
