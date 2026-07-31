//
// test_scrub.cpp
//
// Tests for tessera-scrub.h: each secret detector, determinism, the
// redaction count, name-preserving env masking, the boundary guards that
// keep version strings / ordinary code intact, and null handling.
// Returns non-zero on any failure.
//

#include "tessera-scrub.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static int g_fail = 0;

static void check(const char * name, bool ok) {
    if (ok) {
        std::printf("ok   %s\n", name);
    } else {
        std::printf("FAIL %s\n", name);
        g_fail++;
    }
}

static bool scrub(const std::string & in, std::string & out, int * n = nullptr) {
    char * o = nullptr;
    int rc = ts_scrub_run(in.c_str(), &o, n);
    if (rc != 0) {
        free(o);
        return false;
    }
    out = o ? o : "";
    free(o);
    return true;
}

static bool has(const std::string & s, const std::string & sub) {
    return s.find(sub) != std::string::npos;
}

int main() {
    std::string out;

    // --- each detector fires with the right placeholder ---
    check("api key",      scrub("k = \"sk-abcdefghijklmnopqrstuvwxyz123456\";", out) && has(out, "<secret:api_key>"));
    check("aws key",      scrub("id AKIAIOSFODNN7EXAMPLE end", out) && has(out, "<secret:aws_key>"));
    check("github token", scrub("t ghp_abcdefghijklmnopqrstuvwxyz0123456789 end", out) && has(out, "<secret:github_token>"));
    check("slack token",  scrub("xoxb-1234567890-abcdef", out) && has(out, "<secret:slack_token>"));
    check("bearer",       scrub("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9abcdef", out) && has(out, "<secret:bearer>"));
    check("email",        scrub("mail alice.zhang@acme-corp.com ok", out) && has(out, "<secret:email>"));
    check("ipv6 full",    scrub("addr 2001:0db8:0000:0000:0000:ff00:0042:8329 ok", out) && has(out, "<secret:ipv6>"));
    check("ipv4",         scrub("host 192.168.1.100 ok", out) && has(out, "<secret:ipv4>"));
    check("unix path",    scrub("open /Users/alice/acme/src/widget.c now", out) && has(out, "<secret:path>"));
    check("home path",    scrub("open ~/acme/secret.txt now", out) && has(out, "<secret:path>"));
    check("windows path", scrub("open C:\\Users\\alice\\secret.txt now", out) && has(out, "<secret:path>"));

    // PEM block (multi-line) collapses to one placeholder
    {
        std::string pem =
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1Pt8Qu\n"
            "-----END RSA PRIVATE KEY-----\n";
        check("pem block", scrub(pem, out) && has(out, "<secret:pem>") && !has(out, "MIIBOg"));
    }

    // --- env: keep the informative name, mask the value ---
    check("env masks value", scrub("API_KEY=supersecretvalue123", out) &&
          has(out, "API_KEY=<secret:env>") && !has(out, "supersecretvalue123"));

    // --- determinism: same input -> identical output ---
    {
        std::string in = "key sk-abcdefghijklmnopqrstuvwxyz123456 mail a@b.co ip 10.0.0.1";
        std::string o1, o2;
        check("deterministic", scrub(in, o1) && scrub(in, o2) && o1 == o2);
    }

    // --- redaction count ---
    {
        int n = -1;
        std::string in = "a@b.co and 10.0.0.1 and sk-abcdefghijklmnopqrstuvwxyz123456";
        check("count == 3", scrub(in, out, &n) && n == 3);
    }

    // --- boundary guards: things that must NOT be scrubbed ---
    check("3-part version kept", scrub("version 1.2.3 release", out) && !has(out, "<secret:"));
    check("v-prefixed ver kept", scrub("tag v1.2.3.4 here", out) && !has(out, "<secret:ipv4>"));
    check("c++ scope kept",      scrub("auto x = std::vector<int>();", out) && !has(out, "<secret:"));
    check("non-secret env kept", scrub("#define MAX_BUF 4096", out) && !has(out, "<secret:"));
    check("token_count kept",    scrub("int token_count = 3;", out) && !has(out, "<secret:"));

    // --- ordinary code passes through with zero secret tags ---
    {
        std::string code =
            "#include <stdio.h>\n"
            "struct Widget { int count; double ratio; };\n"
            "static int compute(struct Widget * w) {\n"
            "    int total = w->count * 3 + 7;\n"
            "    double scale = 3.14;\n"
            "    const char * name = \"widget\";\n"
            "    return total / 2;\n"
            "}\n"
            "int main(int argc, char ** argv) { return compute(0); }\n";
        check("ordinary code clean", scrub(code, out) && !has(out, "<secret:"));
    }

    // --- null handling ---
    {
        char * o = nullptr;
        check("null input rc == -1", ts_scrub_run(nullptr, &o, nullptr) == -1);
    }

    if (g_fail == 0) {
        std::printf("\nall tests passed\n");
        return 0;
    }
    std::printf("\n%d check(s) failed\n", g_fail);
    return 1;
}
