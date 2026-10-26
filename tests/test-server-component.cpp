#include "testing.h"

#include "../tools/server/server-common.h"
#include "../tools/server/server-decision.h"
#include "../tools/server/server-schema.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <regex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// unit tests for server internal components, no model required
// NOTE: end-to-end tests live in tools/server/tests
// see skills/code-review/SKILL.md for more

//
// server_tokens (text only, has_mtmd = false)
//

static void test_server_tokens_basic(testing & t) {
    server_tokens st(llama_tokens{1, 2, 3}, false);
    t.assert_equal("size", (size_t) 3, st.size());
    t.assert_true("not empty", !st.empty());
    t.assert_equal("operator[]", 2, st[1]);

    st.push_back(4);
    t.assert_equal("push_back", (size_t) 4, st.size());
    t.assert_true("get_tokens", st.get_tokens() == llama_tokens({1, 2, 3, 4}));

    st.insert({5, 6});
    t.assert_true("insert appends", st.get_tokens() == llama_tokens({1, 2, 3, 4, 5, 6}));

    st.set_token(0, 9);
    t.assert_equal("set_token", 9, st[0]);

    bool thrown = false;
    try {
        st.push_back(LLAMA_TOKEN_NULL);
    } catch (const std::exception &) {
        thrown = true;
    }
    t.assert_true("push_back(LLAMA_TOKEN_NULL) throws", thrown);
    t.assert_equal("size unchanged after throw", (size_t) 6, st.size());

    st.clear();
    t.assert_true("clear", st.empty());
}

static void test_server_tokens_pos(testing & t) {
    server_tokens st(llama_tokens{1, 2, 3, 4, 5}, false);
    t.assert_equal("pos_next()", (llama_pos) 5, st.pos_next());
    t.assert_equal("pos_next(0)", (llama_pos) 0, st.pos_next(0));
    t.assert_equal("pos_next(3)", (llama_pos) 3, st.pos_next(3));

    t.assert_equal("size_up_to_pos(0)", (size_t) 0, st.size_up_to_pos(0));
    t.assert_equal("size_up_to_pos(3)", (size_t) 3, st.size_up_to_pos(3));
    t.assert_equal("size_up_to_pos clamps", (size_t) 5, st.size_up_to_pos(100));
}

static void test_server_tokens_keep_first(testing & t) {
    server_tokens st(llama_tokens{1, 2, 3, 4, 5}, false);
    st.keep_first(5);
    t.assert_equal("keep all", (size_t) 5, st.size());
    st.keep_first(2);
    t.assert_true("keep 2", st.get_tokens() == llama_tokens({1, 2}));
    st.keep_first(0);
    t.assert_true("keep 0", st.empty());
}

static void test_server_tokens_common_prefix(testing & t) {
    server_tokens a(llama_tokens{1, 2, 3, 4}, false);
    server_tokens empty(llama_tokens{}, false);

    t.assert_equal("both empty",  (size_t) 0, empty.get_common_prefix(empty));
    t.assert_equal("one empty",   (size_t) 0, a.get_common_prefix(empty));
    t.assert_equal("identical",   (size_t) 4, a.get_common_prefix(a));
    t.assert_equal("diverge at 2", (size_t) 2, a.get_common_prefix(server_tokens(llama_tokens{1, 2, 9, 4}, false)));
    t.assert_equal("diverge at 0", (size_t) 0, a.get_common_prefix(server_tokens(llama_tokens{9, 2, 3, 4}, false)));
    t.assert_equal("b is prefix",  (size_t) 2, a.get_common_prefix(server_tokens(llama_tokens{1, 2}, false)));
    t.assert_equal("a is prefix",  (size_t) 4, a.get_common_prefix(server_tokens(llama_tokens{1, 2, 3, 4, 5, 6}, false)));
}

static void test_server_tokens_serialize(testing & t) {
    // same packing as slot save/restore in server-context.cpp
    auto pack = [](const std::vector<char> & data) {
        llama_tokens packed(data.size() / sizeof(llama_token));
        memcpy(packed.data(), data.data(), data.size());
        return packed;
    };

    t.test("roundtrip", [&](testing & t) {
        server_tokens st(llama_tokens{10, 20, 30}, false);
        std::vector<char> data = st.serialize();
        t.assert_equal("aligned to llama_token", (size_t) 0, data.size() % sizeof(llama_token));

        server_tokens restored = server_tokens::deserialize(pack(data), false);
        t.assert_true("tokens", restored.get_tokens() == st.get_tokens());
    });

    t.test("roundtrip empty", [&](testing & t) {
        server_tokens st(llama_tokens{}, false);
        server_tokens restored = server_tokens::deserialize(pack(st.serialize()), false);
        t.assert_true("empty", restored.empty());
    });

    t.test("legacy plain token list", [&](testing & t) {
        server_tokens restored = server_tokens::deserialize(llama_tokens{7, 8, 9}, false);
        t.assert_true("tokens", restored.get_tokens() == llama_tokens({7, 8, 9}));
    });

    auto expect_throw = [](testing & t, const std::string & msg, const llama_tokens & packed) {
        bool thrown = false;
        try {
            server_tokens::deserialize(packed, false);
        } catch (const std::exception &) {
            thrown = true;
        }
        t.assert_true(msg, thrown);
    };

    t.test("corrupted", [&](testing & t) {
        llama_tokens good = pack(server_tokens(llama_tokens{10, 20, 30}, false).serialize());

        llama_tokens bad_version = good;
        bad_version[1] = 999;
        expect_throw(t, "bad version", bad_version);

        llama_tokens truncated = good;
        truncated.pop_back();
        expect_throw(t, "truncated", truncated);

        llama_tokens trailing = good;
        trailing.push_back(42);
        expect_throw(t, "trailing data", trailing);

        llama_tokens huge_count = good;
        huge_count[2] = 0x7fffffff;
        expect_throw(t, "huge token count", huge_count);

        expect_throw(t, "marker only", llama_tokens{LLAMA_TOKEN_NULL});
    });
}

static void test_server_tokens_clone(testing & t) {
    server_tokens st(llama_tokens{1, 2, 3}, false);
    server_tokens cl = st.clone();
    cl.push_back(4);
    t.assert_equal("original untouched", (size_t) 3, st.size());
    t.assert_equal("clone grew", (size_t) 4, cl.size());
    t.assert_true("get_text_tokens", cl.get_text_tokens() == llama_tokens({1, 2, 3, 4}));
}

static void test_server_tokens(testing & t) {
    t.test("basic",         test_server_tokens_basic);
    t.test("pos",           test_server_tokens_pos);
    t.test("keep_first",    test_server_tokens_keep_first);
    t.test("common_prefix", test_server_tokens_common_prefix);
    t.test("serialize",     test_server_tokens_serialize);
    t.test("clone",         test_server_tokens_clone);
}

//
// string / json helpers
//

static void test_utf8(testing & t) {
    t.test("validate_utf8", [](testing & t) {
        t.assert_equal("empty",            (size_t) 0, validate_utf8(""));
        t.assert_equal("ascii",            (size_t) 3, validate_utf8("abc"));
        t.assert_equal("full 2-byte",      (size_t) 3, validate_utf8("a\xC3\xA9"));
        t.assert_equal("full 3-byte",      (size_t) 4, validate_utf8("a\xE2\x82\xAC"));
        t.assert_equal("full 4-byte",      (size_t) 5, validate_utf8("a\xF0\x9F\x98\x80"));
        t.assert_equal("cut 2-byte (1/2)", (size_t) 1, validate_utf8("a\xC3"));
        t.assert_equal("cut 3-byte (1/3)", (size_t) 1, validate_utf8("a\xE2"));
        t.assert_equal("cut 3-byte (2/3)", (size_t) 1, validate_utf8("a\xE2\x82"));
        t.assert_equal("cut 4-byte (1/4)", (size_t) 1, validate_utf8("a\xF0"));
        t.assert_equal("cut 4-byte (2/4)", (size_t) 1, validate_utf8("a\xF0\x9F"));
        t.assert_equal("cut 4-byte (3/4)", (size_t) 1, validate_utf8("a\xF0\x9F\x98"));
        t.assert_equal("cut only",         (size_t) 0, validate_utf8("\xF0\x9F\x98"));
        t.assert_equal("full then cut",    (size_t) 4, validate_utf8("\xF0\x9F\x98\x80\xC3"));
    });

    t.test("is_valid_utf8", [](testing & t) {
        t.assert_true("empty",          is_valid_utf8(""));
        t.assert_true("ascii",          is_valid_utf8("hello"));
        t.assert_true("multi-byte",     is_valid_utf8("\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80"));
        t.assert_true("truncated",      !is_valid_utf8("\xE2\x82"));
        t.assert_true("bad cont. byte", !is_valid_utf8("\xC3\x41"));
        t.assert_true("lone cont. byte", !is_valid_utf8("\x80"));
        t.assert_true("invalid lead",   !is_valid_utf8("\xFF"));
    });
}

static void test_sse(testing & t) {
    t.test("oai", [](testing & t) {
        t.assert_equal("single", std::string("data: {\"a\":1}\n\n"), format_oai_sse(json{{"a", 1}}));
        t.assert_equal("array",  std::string("data: {\"a\":1}\n\ndata: {\"b\":2}\n\n"),
                format_oai_sse(json::array({json{{"a", 1}}, json{{"b", 2}}})));
        t.assert_equal("invalid utf8 replaced", std::string("data: \"x\xEF\xBF\xBD\"\n\n"), format_oai_sse(json("x\xC3")));
    });

    t.test("oai_resp", [](testing & t) {
        json ev = {{"event", "response.created"}, {"data", {{"id", "r1"}}}};
        t.assert_equal("single", std::string("event: response.created\ndata: {\"id\":\"r1\"}\n\n"), format_oai_resp_sse(ev));
        t.assert_equal("array",  std::string("event: response.created\ndata: {\"id\":\"r1\"}\n\nevent: response.created\ndata: {\"id\":\"r1\"}\n\n"),
                format_oai_resp_sse(json::array({ev, ev})));
    });

    t.test("anthropic", [](testing & t) {
        json ev = {{"event", "message_start"}, {"data", {{"type", "message_start"}}}};
        t.assert_equal("with event", std::string("event: message_start\ndata: {\"type\":\"message_start\"}\n\n"), format_anthropic_sse(ev));
        t.assert_equal("data only",  std::string("data: {\"type\":\"ping\"}\n\n"), format_anthropic_sse(json{{"type", "ping"}}));
    });
}

static void test_json_helpers(testing & t) {
    t.test("json_is_array_of_numbers", [](testing & t) {
        t.assert_true("ints",         json_is_array_of_numbers(json::array({1, 2, 3})));
        t.assert_true("empty array",  json_is_array_of_numbers(json::array()));
        t.assert_true("float",        !json_is_array_of_numbers(json::array({1, 2.5})));
        t.assert_true("string",       !json_is_array_of_numbers(json::array({1, "a"})));
        t.assert_true("not an array", !json_is_array_of_numbers(json(1)));
    });

    t.test("json_is_array_of_mixed_numbers_strings", [](testing & t) {
        t.assert_true("mixed",        json_is_array_of_mixed_numbers_strings(json::array({"a", 1})));
        t.assert_true("numbers only", !json_is_array_of_mixed_numbers_strings(json::array({1, 2})));
        t.assert_true("strings only", !json_is_array_of_mixed_numbers_strings(json::array({"a", "b"})));
        t.assert_true("empty",        !json_is_array_of_mixed_numbers_strings(json::array()));
    });

    t.test("json_is_array_and_contains_numbers", [](testing & t) {
        t.assert_true("contains",     json_is_array_and_contains_numbers(json::array({"a", 1})));
        t.assert_true("strings only", !json_is_array_and_contains_numbers(json::array({"a"})));
        t.assert_true("not an array", !json_is_array_and_contains_numbers(json(1)));
    });

    t.test("json_get_nested_values", [](testing & t) {
        json js = {{"a", {{"b", {{"c", 1}}}}}, {"x", 2}};
        json res = json_get_nested_values({"a/b/c", "x", "a/b", "a/missing", "x/y"}, js);
        t.assert_equal("nested",  1, res.at("a/b/c").get<int>());
        t.assert_equal("top",     2, res.at("x").get<int>());
        t.assert_true("object",   res.at("a/b") == json{{"c", 1}});
        t.assert_true("missing",  !res.contains("a/missing"));
        t.assert_true("not obj",  !res.contains("x/y"));
        t.assert_equal("count", (size_t) 3, res.size());
    });
}

static void test_lora_helpers(testing & t) {
    // only pointer identity is used, never dereferenced
    auto * p0 = reinterpret_cast<llama_adapter_lora *>(0x10);
    auto * p1 = reinterpret_cast<llama_adapter_lora *>(0x20);

    auto make = [](llama_adapter_lora * ptr, float scale) {
        common_adapter_lora_info info;
        info.scale = scale;
        info.ptr   = ptr;
        return info;
    };

    t.test("parse_lora_request", [](testing & t) {
        auto res = parse_lora_request(json::array({
            json{{"id", 0}, {"scale", 0.5}},
            json{{"id", 2}, {"scale", 1.0}},
            json{{"id", 0}, {"scale", 0.25}}, // last one wins
        }));
        t.assert_equal("count", (size_t) 2, res.size());
        t.assert_equal("id 0", 0.25f, res.at(0));
        t.assert_equal("id 2", 1.0f,  res.at(2));

        auto defaults = parse_lora_request(json::array({json::object()}));
        t.assert_equal("default id/scale", 0.0f, defaults.at(-1));
    });

    t.test("are_lora_equal", [&](testing & t) {
        t.assert_true("both empty",   are_lora_equal({}, {}));
        t.assert_true("same",         are_lora_equal({make(p0, 1.0f)}, {make(p0, 1.0f)}));
        t.assert_true("diff scale",   !are_lora_equal({make(p0, 1.0f)}, {make(p0, 0.5f)}));
        t.assert_true("diff ptr",     !are_lora_equal({make(p0, 1.0f)}, {make(p1, 1.0f)}));
        t.assert_true("diff size",    !are_lora_equal({make(p0, 1.0f)}, {}));
    });

    t.test("lora_get_enabled_ids", [&](testing & t) {
        auto ids = lora_get_enabled_ids({make(p0, 0.0f), make(p1, 0.5f), make(p0, -1.0f), make(p1, 2.0f)});
        t.assert_true("ids", ids == std::vector<size_t>({1, 3}));
    });
}

//
// server_pipe
//

static void test_server_pipe(testing & t) {
    auto stop_now = []() { return true; };

    t.test("drain then eof", [](testing & t) {
        server_pipe<int> p;
        t.assert_true("write 1", p.write(1));
        t.assert_true("write 2", p.write(2));
        p.close_write();

        int v = 0;
        t.assert_true("read 1", p.read(v, nullptr) && v == 1);
        t.assert_true("read 2", p.read(v, nullptr) && v == 2);
        t.assert_true("eof", !p.read(v, nullptr));
    });

    t.test("broken pipe", [](testing & t) {
        server_pipe<int> p;
        p.close_read();
        t.assert_true("write fails", !p.write(1));
    });

    t.test("max_size drops oldest", [](testing & t) {
        server_pipe<int> p;
        p.max_size = 2;
        p.write(1);
        p.write(2);
        p.write(3);
        p.close_write();

        int v = 0;
        t.assert_true("read 2", p.read(v, nullptr) && v == 2);
        t.assert_true("read 3", p.read(v, nullptr) && v == 3);
        t.assert_true("eof", !p.read(v, nullptr));
    });

    t.test("queued data wins over should_stop", [&](testing & t) {
        server_pipe<int> p;
        p.write(1);
        int v = 0;
        t.assert_true("read", p.read(v, stop_now) && v == 1);
    });

    t.test("should_stop closes the pipe", [&](testing & t) {
        server_pipe<int> p;
        int v = 0;
        t.assert_true("read stopped", !p.read(v, stop_now));
        t.assert_true("write fails", !p.write(1));
    });

    t.test("should_stop as deadline keeps the pipe open", [&](testing & t) {
        server_pipe<int> p;
        int v = 0;
        t.assert_true("read stopped", !p.read(v, stop_now, /* close_on_stop */ false));
        t.assert_true("write ok", p.write(1));
        t.assert_true("read again", p.read(v, nullptr) && v == 1);
    });

    t.test("cross thread", [](testing & t) {
        constexpr int n = 1000;
        server_pipe<int> p;
        std::thread producer([&]() {
            for (int i = 0; i < n; i++) {
                p.write(int(i));
            }
            p.close_write();
        });

        int v = 0;
        int count = 0;
        bool in_order = true;
        while (p.read(v, nullptr)) {
            in_order &= v == count;
            count++;
        }
        producer.join();

        t.assert_equal("count", n, count);
        t.assert_true("in order", in_order);
    });
}

//
// server_decision_escape_special_tokens
//

// "<|name|>" with U+00A6 in place of "|"
static std::string escaped(const std::string & name) {
    return "<\xC2\xA6" + name + "\xC2\xA6>";
}

static void test_decision_escape(testing & t) {
    t.test("cases", [](testing & t) {
        const std::vector<std::pair<std::string, std::string>> cases = {
            { "",                 ""                               },
            { "no tokens",        "no tokens"                      },
            { "<|im_start|>user", escaped("im_start") + "user"     },
            { "<|a_1|><|B|>",     escaped("a_1") + escaped("B")    },
            { "<|<|a|>",          "<|" + escaped("a")              },
            { "<|a|>|>",          escaped("a") + "|>"              },
            { "<||>",             "<||>"                           },
            { "<|a b|>",          "<|a b|>"                        },
            { "<|a|b|>",          "<|a|b|>"                        },
            { "<|caf\xC3\xA9|>",  "<|caf\xC3\xA9|>"                },
            { "<|abc",            "<|abc"                          },
        };
        for (const auto & [text, expected] : cases) {
            t.assert_equal(text, expected, server_decision_escape_special_tokens(text));
        }
    });

    // every string of up to 6 chars from "<|>a-" gives the text of the regex
    t.test("same as std::regex", [](testing & t) {
        static const std::regex re_special("<\\|([A-Za-z0-9_]+)\\|>");
        const std::string chars = "<|>a-";
        size_t n_strings = 0;
        size_t n_diff    = 0;
        for (size_t len = 0; len <= 6; len++) {
            size_t n = 1;
            for (size_t k = 0; k < len; k++) {
                n *= chars.size();
            }
            for (size_t idx = 0; idx < n; idx++) {
                std::string text;
                for (size_t k = 0, v = idx; k < len; k++, v /= chars.size()) {
                    text += chars[v % chars.size()];
                }
                if (server_decision_escape_special_tokens(text) != std::regex_replace(text, re_special, "<\xC2\xA6$1\xC2\xA6>")) {
                    n_diff++;
                }
                n_strings++;
            }
        }
        t.assert_equal("strings that differ, of " + std::to_string(n_strings), (size_t) 0, n_diff);
    });

    // std::regex overflowed the stack on these
    t.test("long input", [](testing & t) {
        const std::string name(1000000, 'a');
        t.assert_true("not closed", server_decision_escape_special_tokens("<|" + name) == "<|" + name);
        t.assert_true("closed", server_decision_escape_special_tokens("<|" + name + "|>") == escaped(name));
    });
}

static void test_rerank_template(testing & t) {
    struct test_case {
        const char * name;
        std::string tmpl;
        std::string query;
        std::string doc;
        std::string expected;
    };
    const test_case cases[] = {
        {"plain inputs", "Q: {query}\nD: {document}", "query", "doc", "Q: query\nD: doc"},
        {"document placeholder in query", "Q: {query}\nD: {document}", "Explain {document}.", "doc", "Q: Explain {document}.\nD: doc"},
        {"query placeholder in query", "{query}|{document}", "{query}", "doc", "{query}|doc"},
        {"placeholders in both inputs", "{query}|{document}", "{query} {document}", "{query} {document}", "{query} {document}|{query} {document}"},
        {"repeated placeholders", "{query}|{document}|{query}|{document}", "{document}", "doc", "{document}|doc|{document}|doc"},
        {"adjacent reversed placeholders", "{document}{query}", "query", "doc", "docquery"},
        {"query only", "{query}", "{document}", "doc", "{document}"},
        {"document only", "{document}", "query", "{query}", "{query}"},
        {"no placeholders", "fixed", "{document}", "{query}", "fixed"},
        {"unknown and incomplete placeholders", "{unknown}|{query}|{document}|{query", "query", "doc", "{unknown}|query|doc|{query"},
        {"nested braces", "{{query}}|{{document}}", "query", "doc", "{query}|{doc}"},
        {"placeholder formed at boundary", "{query}ument}", "{doc", "doc", "{document}"},
        {"empty inputs", "Q: {query}\nD: {document}", "", "", "Q: \nD: "},
        {"empty template", "", "query", "doc", ""},
    };
    for (const auto & test : cases) {
        t.assert_equal(test.name, test.expected, format_rerank_template(test.tmpl, test.query, test.doc));
    }
}

//
// server_decision_context::format_answer, 2 class scores (minicpm-ocr-jev)
//

static void test_decision_class_scores(testing & t) {
    server_decision_context ctx;
    ctx.type = COMMON_DECISION_TYPE_MINICPM_OCR_JEV;

    const server_decision_question choice = {"q", SERVER_DECISION_QUESTION_CHOICE, json(), {{"0", json()}, {"1", json()}}};
    const server_decision_question noul   = {"q", SERVER_DECISION_QUESTION_NOUL,   json(), {{"false", json()}, {"true", json()}}};

    const std::vector<std::pair<float, float>> cases = {
        {0.0f, 0.0f}, {-1.5f, 2.25f}, {3.0f, -0.5f}, {80.0f, -80.0f}, {1000.0f, 1001.0f},
    };
    for (const auto & [z0, z1] : cases) {
        const double p1  = 1.0 / (1.0 + std::exp((double) z0 - (double) z1));
        const json   a   = ctx.format_answer(choice, {{z0, z1}});
        const double ap0 = a.at("probabilities").at("0").get<double>();
        const double ap1 = a.at("probabilities").at("1").get<double>();
        t.assert_true("p1", std::fabs(ap1 - p1) <= 1e-6);
        t.assert_true("sum", std::fabs(ap0 + ap1 - 1.0) <= 1e-9);
        // a tie picks the first class
        t.assert_equal("choice", std::string(z1 > z0 ? "1" : "0"), a.at("choice").get<std::string>());
        t.assert_true("confidence", std::fabs(a.at("confidence").get<double>() - (2.0 * std::max(ap0, ap1) - 1.0)) <= 1e-9);

        const json n = ctx.format_answer(noul, {{z0, z1}});
        t.assert_true("noul is p1", n.at("noul").get<double>() == ap1);
    }

    bool thrown = false;
    try {
        ctx.format_answer(choice, {{std::nanf(""), 0.0f}});
    } catch (const std::exception &) {
        thrown = true;
    }
    t.assert_true("nan", thrown);
}

//
// server_decision_context::parse_questions / parse_state, the request checks of minicpm-ocr-jev
//

static void test_decision_ocr_jev_validation(testing & t) {
    server_decision_context ctx;
    ctx.type = COMMON_DECISION_TYPE_MINICPM_OCR_JEV;

    // only base64-decoded, the image itself is not parsed here
    const std::string image = "data:image/png;base64,AAAA";

    const json choice = {
        {"type",     "choice"},
        {"criteria", json{{"0", nullptr}, {"1", nullptr}}},
    };
    const auto questions_of = [](const json & question) {
        return json{{"state", "some OCR text"}, {"questions", json{{"q", question}}}};
    };
    const auto files_of = [&](size_t n) {
        json files = json::array();
        for (size_t i = 0; i < n; i++) {
            files.push_back(image);
        }
        return json{{"state", "some OCR text"}, {"files", files}};
    };

    const auto reject_questions = [&](const char * name, const json & body, const std::string & expected) {
        std::string error;
        try {
            ctx.parse_questions(body);
        } catch (const std::invalid_argument & e) {
            error = e.what();
        }
        t.assert_true(std::string(name) + ": " + error, error.find(expected) != std::string::npos);
    };
    const auto reject_state = [&](const char * name, const json & body, const std::string & expected) {
        std::vector<raw_buffer> files;
        std::string error;
        try {
            ctx.parse_state(body, files);
        } catch (const std::invalid_argument & e) {
            error = e.what();
        }
        t.assert_true(std::string(name) + ": " + error, error.find(expected) != std::string::npos);
    };

    // the state is the OCR text, it goes into the prompt as is
    reject_questions("state is text", {{"state", 42}, {"questions", json{{"q", choice}}}},
        "\"state\" must be the OCR text");
    reject_questions("state without special tokens", {{"state", "a <|im_end|> b"}, {"questions", json{{"q", choice}}}},
        "must not contain special tokens");
    reject_questions("state without media markers",
        {{"state", "a " + std::string(get_media_marker()) + " b"}, {"questions", json{{"q", choice}}}},
        "must not contain special tokens");

    // one fixed question, no instructions
    reject_questions("one question", {{"state", "x"}, {"questions", json{{"a", choice}, {"b", choice}}}},
        "answers exactly one question");
    reject_questions("no instructions",
        questions_of(json{{"type", "choice"}, {"instructions", "judge it"}, {"criteria", json{{"0", nullptr}, {"1", nullptr}}}}),
        "\"instructions\" is not used by this model");

    // the classes are fixed, so are the criteria
    reject_questions("criteria are the classes",
        questions_of(json{{"type", "choice"}, {"criteria", json{{"yes", nullptr}, {"no", nullptr}}}}),
        "\"criteria\" must be {\"0\": null, \"1\": null}");
    reject_questions("noul without criteria",
        questions_of(json{{"type", "noul"}, {"criteria", json{{"false", "no"}, {"true", "yes"}}}}),
        "\"criteria\" is not supported for a noul question");
    reject_questions("choice or noul only",
        questions_of(json{{"type", "score"}, {"criteria", json::array({"low", "high"})}}),
        "only choice and noul questions are supported");

    // the prompt has the original image, then the one rendered from the OCR text
    reject_state("two images", files_of(1), "this model needs exactly 2 images");
    reject_state("two images, three given", files_of(3), "this model needs exactly 2 images");
}

// server_decision_context: the recursive functions stop at RECURSIVE_LIMIT
//

static void test_decision_depth(testing & t) {
    const auto prepare = [](size_t depth) {
        server_decision_context ctx;
        ctx.type = COMMON_DECISION_TYPE_H2O_LIGHTNING; // the state goes through decision_py_dump
        std::string error;
        try {
            ctx.prepare_state(json::parse(std::string(depth, '[') + std::string(depth, ']')));
        } catch (const std::invalid_argument & e) {
            error = e.what();
        }
        return error;
    };
    t.test("state at the limit", [&](testing & t) {
        const std::string error = prepare(128);
        t.assert_true("accepted: " + error, error.empty());
    });
    t.test("deep state", [&](testing & t) {
        const std::string error = prepare(1000);
        t.assert_true("rejected as nested: " + error, error.find("nested more than 128 levels deep") != std::string::npos);
    });
}

// DRY sequence breakers
//

static void test_dry_sequence_breakers(testing & t) {
    common_params params_base;

    const auto eval = [&](const json & data) {
        return server_schema::eval_llama_cmpl_schema(nullptr, params_base, {}, data);
    };

    {
        json data;
        data["dry_sequence_breakers"] = json::array({"ab", "ab", "cd", "ab"});
        const auto params = eval(data);
        const std::vector<std::string> expected = {"ab", "cd"};
        t.assert_true("duplicates removed, order kept", params.sampling.dry_sequence_breakers == expected);
    }

    {
        std::vector<std::string> many;
        for (int i = 0; i <= LLAMA_DRY_MAX_SEQ_BREAKERS; ++i) {
            many.push_back("b" + std::to_string(i));
        }
        json data;
        data["dry_sequence_breakers"] = many;

        bool thrown = false;
        try {
            eval(data);
        } catch (const std::exception &) {
            thrown = true;
        }
        t.assert_true("more than the cap is rejected", thrown);
    }
}

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("server_tokens", test_server_tokens);
    t.test("utf8",          test_utf8);
    t.test("sse",           test_sse);
    t.test("json_helpers",  test_json_helpers);
    t.test("lora_helpers",  test_lora_helpers);
    t.test("server_pipe",   test_server_pipe);
    t.test("decision_escape", test_decision_escape);
    t.test("rerank_template", test_rerank_template);
    t.test("decision_class_scores", test_decision_class_scores);
    t.test("decision_ocr_jev_validation", test_decision_ocr_jev_validation);
    t.test("decision_depth",  test_decision_depth);
    t.test("dry_sequence_breakers", test_dry_sequence_breakers);

    return t.summary();
}
