#include "testing.h"

#include "../tools/server/server-common.h"
#include "../tools/server/server-decision.h"
#include "../tools/server/server-perf.h"

#include <cmath>
#include <cstdlib>
#include <deque>
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


//
// server_perf_compute (synthetic samples)
//

static constexpr int64_t S = 1000000; // us per second

static server_perf_tick perf_tick(int64_t t_s) {
    server_perf_tick t;
    t.t_us  = t_s*S;
    t.ts_ms = 1700000000000 + t_s*1000;
    return t;
}

static void assert_near(testing & t, const std::string & msg, double expected, const json & v) {
    if (!v.is_number()) {
        t.assert_true(msg + " (not a number: " + v.dump() + ")", false);
        return;
    }
    const double a = v.get<double>();
    t.assert_true(msg + " (expected " + std::to_string(expected) + ", got " + std::to_string(a) + ")",
            std::fabs(a - expected) <= 1e-9*std::max(1.0, std::fabs(expected)));
}

static void test_perf_compute(testing & t) {
    server_perf_info info;
    info.n_cpu     = 4;
    info.clk_tck   = 100;
    info.gpu_ok    = true;
    info.gpu_name  = "GPU";
    info.disk_ok   = true;
    info.disk_name = "md0";
    server_perf_baseline base;
    base.has_io        = true;
    base.io_read_bytes = 1000;

    t.test("partial window, cpu, mem, gpu, nvme", [&](testing & t) {
        std::deque<server_perf_tick> ring;
        for (int i = 0; i < 3; ++i) {
            auto k = perf_tick(i);
            k.has_cpu    = true;
            k.cpu_total  = 1000 + 100*i;
            k.cpu_idle   = 900 + 50*i;  // 50% busy
            k.proc_ticks = 20*i;        // 0.2 s of CPU per second on 4 CPUs: 5%
            k.has_mem    = true;
            k.mem_total  = 256ull << 30;
            k.mem_free   = 10ull << 30;
            k.mem_available = 100ull << 30;
            k.mem_cached = 200ull << 30;
            k.has_gpu    = true;
            k.gpu_busy_pct  = 3;
            k.gpu_mem_used  = 11ull << 30;
            k.gpu_mem_total = 12ull << 30;
            k.has_disk     = true;
            k.disk_sectors = 8000000ull*i; // 4.096 GB/s
            k.n_gen_total  = i == 0 ? 0 : 1;
            ring.push_back(k);
        }
        const json r = server_perf_compute(ring, base, info, 2*S);
        t.assert_equal("ts_ms", (int64_t) 1700000002000, r["ts_ms"].get<int64_t>());
        t.assert_equal("window_s", 60, r["window_s"].get<int>());
        assert_near(t, "cpu busy", 50.0, r["cpu"]["busy_pct"]);
        assert_near(t, "cpu proc", 5.0,  r["cpu"]["proc_pct"]);
        t.assert_equal("n_cpu", 4, r["cpu"]["n_cpu"].get<int>());
        t.assert_equal("occupied = total - free", (uint64_t) (246ull << 30), r["mem"]["occupied_bytes"].get<uint64_t>());
        t.assert_equal("used = total - available", (uint64_t) (156ull << 30), r["mem"]["used_bytes"].get<uint64_t>());
        assert_near(t, "gpu busy", 3.0, r["gpu"]["busy_pct"]);
        t.assert_equal("gpu name", std::string("GPU"), r["gpu"]["name"].get<std::string>());
        assert_near(t, "nvme 1 s", 4.096e9, r["nvme"]["read_bps"]);
        assert_near(t, "nvme 10 s (partial)", 4.096e9, r["nvme"]["read_bps_10s"]);
        // 1 token in the only second that produced one
        assert_near(t, "gen window over active seconds", 1.0, r["gen"]["tps_window"]);
        t.assert_true("no request yet: headline null", r["gen"]["tps_headline"].is_null());
        t.assert_true("no request yet: prompt null", r["prompt"]["tps_headline"].is_null());
        t.assert_true("dense model: predict null", r["predict"].is_null());
        t.assert_true("dense model: preload null", r["preload"].is_null());
    });

    t.test("one sample, empty ring", [&](testing & t) {
        std::deque<server_perf_tick> ring = { perf_tick(0) };
        ring[0].has_cpu = true;
        const json r = server_perf_compute(ring, base, info, 0);
        t.assert_true("no rate from one sample", r["cpu"]["busy_pct"].is_null());
        t.assert_true("ts_ms set", r["ts_ms"].is_number());
        const json e = server_perf_compute({}, base, info, 0);
        t.assert_true("empty ring", e["ts_ms"].is_null());
    });

    t.test("counter reset", [&](testing & t) {
        std::deque<server_perf_tick> ring;
        const uint64_t gen[] = { 100, 102, 5, 6 }; // reload between the 2nd and 3rd sample
        for (int i = 0; i < 4; ++i) {
            auto k = perf_tick(i);
            k.n_gen_total = gen[i];
            ring.push_back(k);
        }
        const json r = server_perf_compute(ring, base, info, 3*S);
        assert_near(t, "the reset interval counts as no tokens", 3.0/2.0, r["gen"]["tps_window"]);
    });

    t.test("gen headline: running request, then the last one", [&](testing & t) {
        std::deque<server_perf_tick> ring = { perf_tick(0), perf_tick(1) };
        auto & k = ring.back();
        k.generating = true;
        k.cur_gen_n  = 11;
        k.cur_gen_t_first_us = 1*S;
        k.cur_gen_t_last_us  = 21*S;
        k.last_gen_steps = 63;
        k.last_gen_us    = 120*S;
        k.last_gen_n     = 64;
        json r = server_perf_compute(ring, base, info, 21*S);
        assert_near(t, "live: 10 tokens after the first in 20 s", 0.5, r["gen"]["tps_headline"]);
        assert_near(t, "last request", 63.0/120.0, r["gen"]["tps_last"]);
        t.assert_equal("n_tokens of the running request", (uint64_t) 11, r["gen"]["n_tokens"].get<uint64_t>());

        // the first token alone has no rate: the previous request's average until the second one
        k.cur_gen_n = 1;
        k.cur_gen_t_last_us = k.cur_gen_t_first_us;
        r = server_perf_compute(ring, base, info, 21*S);
        t.assert_true("one token: null (the panel prints ...), not the last request's rate", r["gen"]["tps_headline"].is_null());
        assert_near(t, "one token: tps_last still the last request", 63.0/120.0, r["gen"]["tps_last"]);
        k.last_gen_steps = 0;
        k.last_gen_us    = 0;
        r = server_perf_compute(ring, base, info, 21*S);
        t.assert_true("one token of the first request: null (the panel prints ...)", r["gen"]["tps_headline"].is_null());
        k.last_gen_steps = 63;
        k.last_gen_us    = 120*S;

        k.generating = false;
        r = server_perf_compute(ring, base, info, 30*S);
        assert_near(t, "idle: the last request's average", 63.0/120.0, r["gen"]["tps_headline"]);
        t.assert_equal("idle n_tokens", (uint64_t) 64, r["gen"]["n_tokens"].get<uint64_t>());
        t.assert_true("idle, no tokens in the window", r["gen"]["tps_window"].is_null());
        t.assert_true("not 0 when idle", r["gen"]["tps_headline"].get<double>() > 0.0);
    });

    t.test("running prompt batch: layer progress", [&](testing & t) {
        std::deque<server_perf_tick> ring = { perf_tick(29), perf_tick(30) };
        auto & k = ring.back();
        k.processing            = true;
        k.cur_prompt_n          = 0;
        k.cur_prompt_t_start_us = 1*S;
        k.progress = { 41, 93, 1600, 0, 1*S };
        k.last_prompt_n  = 100;
        k.last_prompt_us = 10*S;
        json r = server_perf_compute(ring, base, info, 31*S);
        assert_near(t, "pp live = 1600 x 42/93 / 30 s", 1600.0*42/93/30.0, r["prompt"]["tps_headline"]);
        t.assert_equal("layer", 42, r["prompt"]["layer"].get<int>());
        t.assert_equal("n_layer", 93, r["prompt"]["n_layer"].get<int>());
        assert_near(t, "last", 10.0, r["prompt"]["tps_last"]);

        // a second batch of the same request: the first one's tokens count fully
        k.cur_prompt_n = 1600;
        k.progress     = { 9, 93, 800, 0, 50*S };
        r = server_perf_compute(ring, base, info, 61*S);
        assert_near(t, "pp live, second batch", (1600.0 + 800.0*10/93)/60.0, r["prompt"]["tps_headline"]);

        // between batches (no batch runs): the layer is unknown
        k.progress = { 92, 93, 0, 0, 50*S };
        r = server_perf_compute(ring, base, info, 61*S);
        t.assert_true("no layer between batches", r["prompt"]["layer"].is_null());
        assert_near(t, "pp live without a batch", 1600.0/60.0, r["prompt"]["tps_headline"]);

        // idle: the last request's prompt speed
        k.processing = false;
        r = server_perf_compute(ring, base, info, 100*S);
        assert_near(t, "idle prompt headline", 10.0, r["prompt"]["tps_headline"]);
    });

    t.test("precision and recall windows", [&](testing & t) {
        std::deque<server_perf_tick> ring;
        uint64_t hits = 0, guesses = 0, actual = 0;
        for (int i = 0; i <= 70; ++i) {
            auto k = perf_tick(i);
            k.predict.enabled = true;
            k.predict.mode    = 2;
            if (i > 0) {
                // 50% precision in the first 10 s, 25% after
                hits    += i <= 10 ? 8 : 4;
                guesses += 16;
                actual  += 8;
            }
            k.predict.hits    = hits;
            k.predict.guesses = guesses;
            k.predict.actual  = actual;
            k.predict.probes    = 3*(uint64_t) i;
            k.predict.step_seen = (uint64_t) i;
            ring.push_back(k);
            if (ring.size() > 61) {
                ring.pop_front();
            }
        }
        const json r = server_perf_compute(ring, base, info, 70*S);
        assert_near(t, "precision window (t = 10..70)", 240.0/960.0, r["predict"]["precision_window"]);
        assert_near(t, "recall window", 240.0/480.0, r["predict"]["recall_window"]);
        assert_near(t, "precision total", 320.0/1120.0, r["predict"]["precision_total"]);
        assert_near(t, "recall total", 320.0/560.0, r["predict"]["recall_total"]);
        t.assert_equal("mode", std::string("prefetch"), r["predict"]["mode"].get<std::string>());
        t.assert_equal("guesses", (uint64_t) 1120, r["predict"]["guesses"].get<uint64_t>());
        t.assert_equal("probes since the baseline", (uint64_t) 210, r["predict"]["probes"].get<uint64_t>());
        t.assert_equal("steps since the baseline", (uint64_t) 70, r["predict"]["steps"].get<uint64_t>());
    });

    t.test("preload wasted, lifetime", [&](testing & t) {
        const uint64_t MB = 1000000;
        auto k1 = perf_tick(1);
        k1.has_io = true;
        k1.io_read_bytes = base.io_read_bytes + 10*MB; // all bytes the process read since the baseline
        k1.predict.enabled = true;
        k1.predict.mode    = 2;
        k1.predict.bytes_read_predictor = 4*MB;
        k1.predict.bytes_hit     = 1*MB;
        k1.predict.bytes_pending = 3*MB;
        std::deque<server_perf_tick> ring = { perf_tick(0), k1 };
        json r = server_perf_compute(ring, base, info, 1*S);
        assert_near(t, "nothing wasted yet", 0.0, r["preload"]["wasted_pct"]);
        t.assert_equal("read total", 10*MB, r["preload"]["bytes_read_total"].get<uint64_t>());

        // read at tick 1, expired at tick 100: counts against the lifetime total
        auto k100 = k1;
        k100.t_us = 100*S;
        k100.predict.bytes_pending = 0;
        k100.predict.bytes_wasted  = 3*MB;
        ring = { k1, k100 };
        r = server_perf_compute(ring, base, info, 100*S);
        assert_near(t, "3 of 10 MB wasted", 30.0, r["preload"]["wasted_pct"]);

        // credited at tick 70 instead: pending -> hit and credited, wasted unchanged
        auto k70 = k1;
        k70.t_us = 70*S;
        k70.predict.bytes_pending  = 2*MB;
        k70.predict.bytes_hit      = 2*MB;
        k70.predict.bytes_credited = 1*MB;
        ring = { k1, k70 };
        r = server_perf_compute(ring, base, info, 70*S);
        assert_near(t, "credit leaves wasted unchanged", 0.0, r["preload"]["wasted_pct"]);
        t.assert_equal("pending -= B", 2*MB, r["preload"]["bytes_pending"].get<uint64_t>());
        t.assert_equal("hit += B", 2*MB, r["preload"]["bytes_hit"].get<uint64_t>());
        t.assert_equal("credited += B", 1*MB, r["preload"]["bytes_credited"].get<uint64_t>());

        // the same entry expired instead: wasted rises by exactly B / read total
        auto k70e = k1;
        k70e.t_us = 70*S;
        k70e.predict.bytes_pending = 2*MB;
        k70e.predict.bytes_wasted  = 1*MB;
        ring = { k1, k70e };
        r = server_perf_compute(ring, base, info, 70*S);
        assert_near(t, "expiry adds B / total", 100.0*1/10, r["preload"]["wasted_pct"]);

        // late loads count as wasted
        k70e.predict.bytes_late = 1*MB;
        ring = { k1, k70e };
        r = server_perf_compute(ring, base, info, 70*S);
        assert_near(t, "late counts", 100.0*2/10, r["preload"]["wasted_pct"]);

        // an idle stretch with expiries never goes above 100%
        auto kx = k1;
        kx.predict.bytes_wasted = 30*MB;
        ring = { k1, kx };
        r = server_perf_compute(ring, base, info, 200*S);
        assert_near(t, "clamped", 100.0, r["preload"]["wasted_pct"]);

        // nothing read since the baseline: no ratio
        auto k0 = k1;
        k0.io_read_bytes = base.io_read_bytes;
        ring = { k1, k0 };
        r = server_perf_compute(ring, base, info, 1*S);
        t.assert_true("denominator 0 -> null", r["preload"]["wasted_pct"].is_null());

        // counters since the baseline
        server_perf_baseline b2 = base;
        b2.predict.bytes_wasted = 1*MB;
        ring = { k1, k100 };
        r = server_perf_compute(ring, b2, info, 100*S);
        assert_near(t, "since the baseline", 20.0, r["preload"]["wasted_pct"]);
    });

    t.test("null groups", [&](testing & t) {
        server_perf_info none;
        auto k = perf_tick(1);
        k.predict.enabled = true;
        k.predict.mode    = 1;
        std::deque<server_perf_tick> ring = { perf_tick(0), k };
        const json r = server_perf_compute(ring, base, none, 1*S);
        t.assert_true("cpu null off Linux", r["cpu"].is_null());
        t.assert_true("mem null", r["mem"].is_null());
        t.assert_true("gpu null without NVML", r["gpu"].is_null());
        t.assert_true("nvme null without a device", r["nvme"].is_null());
        t.assert_equal("observe mode", std::string("observe"), r["predict"]["mode"].get<std::string>());
        t.assert_true("observe mode: preload null", r["preload"].is_null());
        t.assert_true("gen group always there", r["gen"].is_object() && r["prompt"].is_object());
    });
}


//
// projected prompt speed, replayed from a Kimi-K3 run (1 Hz samples of GET /perf and the server's batch timings)
// batches: "tokens:seconds" in order; ticks: "seconds since the request started:layers done of the running batch"
//

struct perf_replay {
    const char * name;
    double       final_tps;
    const char * batches;
    const char * ticks;
};

static const perf_replay PERF_REPLAYS[] = {
    // A request 1: final 12.540057569748647
    { "A1", 12.540058, "67:40.28 1565:76.15 4:14.03", "1.0:2 2.0:4 3.0:5 4.0:7 5.0:8 6.0:10 7.0:11 8.0:13 9.0:15 10.0:17 11.0:20 12.0:22 13.0:24 14.0:26 15.0:28 16.0:30 17.0:32 18.0:34 19.0:36 20.0:38 21.0:40 22.0:43 23.0:45 24.0:48 25.0:51 26.0:54 27.0:58 28.0:61 29.0:64 30.0:67 31.0:69 32.0:72 33.0:74 34.0:77 35.0:79 36.0:82 37.0:84 38.0:87 39.0:90 40.0:93 41.0:2 42.0:3 43.0:4 44.0:5 45.0:6 46.0:7 47.0:8 48.0:9 49.0:10 50.0:11 51.0:12 52.0:14 53.0:15 54.0:16 55.0:17 56.0:18 57.0:19 58.0:20 59.0:21 60.0:23 61.1:24 62.1:25 63.1:26 64.1:27 65.1:28 66.1:29 67.1:31 68.1:32 69.1:33 70.1:34 71.1:36 72.1:37 73.1:38 74.1:39 75.1:40 76.1:41 77.1:43 78.1:44 79.1:45 80.1:47 81.1:48 82.1:49 83.1:51 84.1:52 85.1:54 86.1:55 87.1:57 88.1:58 89.1:60 90.1:61 91.1:62 92.1:63 93.1:65 94.1:66 95.1:67 96.1:68 97.1:70 98.1:71 99.1:72 100.1:73 101.1:74 102.1:76 103.1:77 104.1:78 105.1:79 106.1:81 107.1:82 108.1:83 109.1:84 110.1:85 111.1:87 112.1:88 113.1:89 114.1:90 115.1:92 116.1:93 117.1:-1 118.1:-1 119.1:-1 120.1:-1 121.1:-1 122.1:-1 123.1:-1 124.1:-1 125.1:-1 126.1:-1 127.1:-1 128.1:-1 129.1:-1 130.1:-1" },
    // A request 2: final 15.445084543747761
    { "A2", 15.445085, "67:31.34 1564:69.93 4:4.59", "1.0:2 2.0:3 3.0:5 4.0:7 5.0:9 6.0:11 7.0:14 8.0:16 9.0:19 10.0:22 11.0:24 12.0:27 13.0:30 14.0:32 15.0:35 16.0:38 17.0:40 18.0:43 19.0:46 20.0:50 21.0:53 22.0:58 23.0:61 24.0:65 25.0:68 26.0:72 27.0:75 28.0:78 29.0:82 30.0:85 31.0:89 32.0:93 33.0:2 34.0:3 35.0:4 36.0:5 37.0:7 38.0:8 39.0:9 40.0:10 41.0:11 42.0:12 43.0:14 44.0:15 45.0:17 46.0:18 47.0:19 48.0:20 49.0:21 50.0:23 51.0:24 52.0:25 53.0:26 54.0:28 55.0:29 56.0:30 57.0:32 58.0:33 59.1:34 60.1:36 61.1:37 62.1:38 63.1:39 64.1:41 65.1:42 66.1:43 67.1:44 68.1:46 69.1:47 70.1:49 71.1:50 72.1:51 73.1:53 74.1:54 75.1:56 76.1:57 77.1:59 78.1:60 79.1:62 80.1:63 81.1:65 82.1:66 83.1:67 84.1:69 85.1:70 86.1:71 87.1:73 88.1:74 89.1:76 90.1:77 91.1:78 92.1:80 93.1:81 94.1:83 95.1:84 96.1:85 97.1:87 98.1:88 99.1:89 100.1:90 101.1:92 102.1:93 103.1:-1 104.1:-1 105.1:-1 106.1:-1" },
    // A request 3: final 15.094927626373009
    { "A3", 15.094928, "67:32.83 1565:69.28 4:6.27", "1.3:2 2.3:4 3.3:6 4.3:7 5.3:9 6.3:11 7.3:14 8.3:17 9.3:20 10.3:22 11.3:25 12.3:27 13.3:30 14.3:32 15.3:35 16.3:37 17.3:40 18.3:43 19.3:46 20.3:49 21.3:52 22.3:56 23.3:59 24.3:63 25.3:66 26.3:69 27.3:72 28.3:75 29.3:78 30.3:82 31.3:85 32.3:88 33.3:92 34.3:2 35.3:3 36.3:4 37.3:5 38.3:6 39.3:7 40.3:9 41.3:10 42.3:11 43.3:12 44.3:14 45.3:15 46.3:17 47.3:18 48.3:19 49.3:20 50.3:21 51.3:23 52.4:24 53.4:25 54.4:27 55.4:28 56.4:29 57.4:30 58.4:32 59.4:33 60.4:34 61.4:36 62.4:37 63.4:38 64.4:39 65.4:41 66.4:42 67.4:43 68.4:45 69.4:46 70.4:48 71.4:49 72.4:50 73.4:52 74.4:53 75.4:55 76.4:56 77.4:58 78.4:59 79.4:61 80.4:62 81.4:64 82.4:65 83.4:66 84.4:68 85.4:69 86.4:70 87.4:71 88.4:73 89.4:74 90.4:76 91.4:77 92.4:78 93.4:80 94.4:81 95.4:83 96.4:84 97.4:85 98.4:87 99.4:88 100.4:90 101.4:91 102.4:93 103.4:-1 104.4:-1 105.4:-1 106.4:-1 107.4:-1 108.4:-1" },
    // B request 1: final 2.4451813397970894
    { "B1", 2.445181, "67:91.49 1565:568.23 4:9.35", "1.0:2 2.0:3 3.0:3 4.0:4 5.0:5 6.0:6 7.0:7 8.0:7 9.0:8 10.0:9 11.0:10 12.0:10 13.0:11 14.0:12 15.0:14 16.0:15 17.0:16 18.0:17 19.0:18 20.0:19 21.0:20 22.0:21 23.0:22 24.0:23 25.0:24 26.0:25 27.0:25 28.0:26 29.0:27 30.0:28 31.0:29 32.0:30 33.0:31 34.0:32 35.0:33 36.0:34 37.0:34 38.0:35 39.0:36 40.0:37 41.0:38 42.0:39 43.0:40 44.0:41 45.0:42 46.0:43 47.0:44 48.0:45 49.0:46 50.0:47 51.0:48 52.0:49 53.0:50 54.0:51 55.0:52 56.0:54 57.0:55 58.0:57 59.0:58 60.1:59 61.1:61 62.1:62 63.1:63 64.1:64 65.1:65 66.1:66 67.1:67 68.1:68 69.1:69 70.1:70 71.1:71 72.1:72 73.1:73 74.1:74 75.1:75 76.1:76 77.1:77 78.1:78 79.1:80 80.1:81 81.1:82 82.1:83 83.1:84 84.1:85 85.1:86 86.1:87 87.1:88 88.1:89 89.1:91 90.1:92 91.1:93 92.1:2 93.1:2 94.1:2 95.1:2 96.1:2 97.1:3 98.1:3 99.1:3 100.1:3 101.1:4 102.1:4 103.1:4 104.1:5 105.1:5 106.1:5 107.1:5 108.1:6 109.1:6 110.1:6 111.1:6 112.1:7 113.1:7 114.1:7 115.1:7 116.1:8 117.1:8 118.1:8 119.1:8 120.1:9 121.1:9 122.1:9 123.1:9 124.1:10 125.1:10 126.1:10 127.1:10 128.1:11 129.1:11 130.1:11 131.1:11 132.1:12 133.1:12 134.1:12 135.1:12 136.1:13 137.1:13 138.1:13 139.1:14 140.1:14 141.1:14 142.1:14 143.1:15 144.1:15 145.1:15 146.1:15 147.1:16 148.1:16 149.1:16 150.1:16 151.1:17 152.1:17 153.1:17 154.1:17 155.1:18 156.1:18 157.1:18 158.1:18 159.1:19 160.1:19 161.1:19 162.1:19 163.1:19 164.1:20 165.1:20 166.1:20 167.1:20 168.1:20 169.1:20 170.1:20 171.1:21 172.1:21 173.1:21 174.1:21 175.2:21 176.2:21 177.2:21 178.2:22 179.2:22 180.2:22 181.2:22 182.2:22 183.2:22 184.2:22 185.2:23 186.2:23 187.2:23 188.2:23 189.2:23 190.2:23 191.2:24 192.2:24 193.2:24 194.2:24 195.2:24 196.2:24 197.2:24 198.2:25 199.2:25 200.2:25 201.2:25 202.2:25 203.2:25 204.2:26 205.2:26 206.2:26 207.2:26 208.2:26 209.2:26 210.2:26 211.2:27 212.2:27 213.2:27 214.2:27 215.2:27 216.2:27 217.2:27 218.2:28 219.2:28 220.2:28 221.2:28 222.2:28 223.2:28 224.2:29 225.2:29 226.2:29 227.2:29 228.2:29 229.2:29 230.2:29 231.2:30 232.2:30 233.2:30 234.2:30 235.2:30 236.2:30 237.2:30 238.2:31 239.2:31 240.2:31 241.2:31 242.2:31 243.2:31 244.2:32 245.2:32 246.2:32 247.2:32 248.2:32 249.2:32 250.2:32 251.2:33 252.2:33 253.2:33 254.2:33 255.2:33 256.2:33 257.2:33 258.2:34 259.2:34 260.2:34 261.2:34 262.2:34 263.2:34 264.2:34 265.2:35 266.2:35 267.2:35 268.2:35 269.2:35 270.2:35 271.2:36 272.2:36 273.2:36 274.2:36 275.2:36 276.2:36 277.2:36 278.2:37 279.2:37 280.2:37 281.2:37 282.2:37 283.2:37 284.2:37 285.2:38 286.2:38 287.2:38 288.2:38 289.2:38 290.2:38 291.2:38 292.2:39 293.2:39 294.3:39 295.3:39 296.3:39 297.3:39 298.3:40 299.3:40 300.3:40 301.3:40 302.3:40 303.3:40 304.3:41 305.3:41 306.3:41 307.3:41 308.3:41 309.3:41 310.3:41 311.3:42 312.3:42 313.3:42 314.3:42 315.3:42 316.3:42 317.3:42 318.3:43 319.3:43 320.3:43 321.3:43 322.3:43 323.3:43 324.3:43 325.3:44 326.3:44 327.3:44 328.3:44 329.3:44 330.3:44 331.3:44 332.3:45 333.3:45 334.3:45 335.3:45 336.3:45 337.3:45 338.3:45 339.3:46 340.3:46 341.3:46 342.3:46 343.3:46 344.3:46 345.3:46 346.3:47 347.3:47 348.3:47 349.3:47 350.3:47 351.3:47 352.3:48 353.3:48 354.3:48 355.3:48 356.3:48 357.3:48 358.3:48 359.3:49 360.3:49 361.3:49 362.3:49 363.3:49 364.3:49 365.3:49 366.3:50 367.3:50 368.3:50 369.3:50 370.3:50 371.3:50 372.3:50 373.3:51 374.3:51 375.3:51 376.3:51 377.3:51 378.3:51 379.3:52 380.3:52 381.3:52 382.3:52 383.3:52 384.3:52 385.3:52 386.3:53 387.3:53 388.3:53 389.3:53 390.3:53 391.3:53 392.3:54 393.3:54 394.3:54 395.3:54 396.3:54 397.3:54 398.3:54 399.3:55 400.3:55 401.3:55 402.3:55 403.3:55 404.3:55 405.3:56 406.3:56 407.3:56 408.3:56 409.3:56 410.3:56 411.3:56 412.3:57 413.4:57 414.4:57 415.4:57 416.4:57 417.4:57 418.4:57 419.4:58 420.4:58 421.4:58 422.4:58 423.4:58 424.4:58 425.4:59 426.4:59 427.4:59 428.4:59 429.4:59 430.4:59 431.4:59 432.4:60 433.4:60 434.4:60 435.4:60 436.4:60 437.4:60 438.4:61 439.4:61 440.4:61 441.4:61 442.4:61 443.4:61 444.4:61 445.4:62 446.4:62 447.4:62 448.4:62 449.4:62 450.4:62 451.4:62 452.4:63 453.4:63 454.4:63 455.4:63 456.4:63 457.4:63 458.4:63 459.4:64 460.4:64 461.4:64 462.4:64 463.4:64 464.4:64 465.4:64 466.4:65 467.4:65 468.4:65 469.4:65 470.4:65 471.4:65 472.4:66 473.4:66 474.4:66 475.4:66 476.4:66 477.4:66 478.4:66 479.4:67 480.4:67 481.4:67 482.4:67 483.4:67 484.4:67 485.4:67 486.4:68 487.4:68 488.4:68 489.4:68 490.4:68 491.4:68 492.4:68 493.4:69 494.4:69 495.4:69 496.4:69 497.4:69 498.4:69 499.4:69 500.4:70 501.4:70 502.4:70 503.4:70 504.4:70 505.4:70 506.4:70 507.4:71 508.4:71 509.4:71 510.4:71 511.4:71 512.4:71 513.4:72 514.4:72 515.4:72 516.4:72 517.4:72 518.4:72 519.4:72 520.4:72 521.4:73 522.4:73 523.4:73 524.4:73 525.4:73 526.4:73 527.4:74 528.4:74 529.4:74 530.4:74 531.4:74 532.4:74 533.4:74 534.4:75 535.4:75 536.4:75 537.5:75 538.5:75 539.5:75 540.5:76 541.5:76 542.5:76 543.5:76 544.5:76 545.5:76 546.5:76 547.5:77 548.5:77 549.5:77 550.5:77 551.5:77 552.5:77 553.5:77 554.5:78 555.5:78 556.5:78 557.5:78 558.5:78 559.5:78 560.5:78 561.5:79 562.5:79 563.5:79 564.5:79 565.5:79 566.5:79 567.5:80 568.5:80 569.5:80 570.5:80 571.5:80 572.5:80 573.5:80 574.5:81 575.5:81 576.5:81 577.5:81 578.5:81 579.5:81 580.5:81 581.5:82 582.5:82 583.5:82 584.5:82 585.5:82 586.5:82 587.5:82 588.5:83 589.5:83 590.5:83 591.5:83 592.5:83 593.5:83 594.5:84 595.5:84 596.5:84 597.5:84 598.5:84 599.5:84 600.5:84 601.5:85 602.5:85 603.5:85 604.5:85 605.5:85 606.5:85 607.5:85 608.5:86 609.5:86 610.5:86 611.5:86 612.5:86 613.5:86 614.5:86 615.5:87 616.5:87 617.5:87 618.5:87 619.5:87 620.5:87 621.5:88 622.5:88 623.5:88 624.5:88 625.5:88 626.5:88 627.5:88 628.5:89 629.5:89 630.5:89 631.5:89 632.5:89 633.5:89 634.5:89 635.5:90 636.5:90 637.5:90 638.5:90 639.5:90 640.5:90 641.5:90 642.5:91 643.5:91 644.5:91 645.5:91 646.5:91 647.5:91 648.5:92 649.5:92 650.5:92 651.5:92 652.5:92 653.5:92 654.5:93 655.5:93 656.5:93 657.5:93 658.5:93 659.5:93 660.5:-1 661.6:-1 662.6:-1 663.6:-1 664.6:-1 665.6:-1 666.6:-1 667.6:-1 668.6:-1" },
    // A0 request 1: final 21.501002885103397
    { "A01", 21.501003, "1636:76.09", "1.0:2 2.0:3 3.0:4 4.0:5 5.0:6 6.0:7 7.0:8 8.0:9 9.0:10 10.0:11 11.0:12 12.0:14 13.0:15 14.0:16 15.0:18 16.0:19 17.0:20 18.0:21 19.0:22 20.0:23 21.0:24 22.0:25 23.0:27 24.0:28 25.0:29 26.0:30 27.0:32 28.0:33 29.0:34 30.0:35 31.0:36 32.0:38 33.0:39 34.0:40 35.0:41 36.0:42 37.0:44 38.0:45 39.0:46 40.0:48 41.0:49 42.0:50 43.0:52 44.0:53 45.0:54 46.0:56 47.0:57 48.0:58 49.0:60 50.0:61 51.0:62 52.0:64 53.0:65 54.0:66 55.0:68 56.1:69 57.1:70 58.1:71 59.1:72 60.1:73 61.1:75 62.1:76 63.1:77 64.1:79 65.1:80 66.1:81 67.1:82 68.1:84 69.1:85 70.1:86 71.1:87 72.1:88 73.1:90 74.1:91 75.1:92 76.1:93" },
};

static std::vector<std::pair<double, double>> perf_pairs(const char * s) {
    std::vector<std::pair<double, double>> out;
    std::string str(s);
    size_t p = 0;
    while (p < str.size()) {
        size_t e = str.find(' ', p);
        if (e == std::string::npos) {
            e = str.size();
        }
        const std::string tok = str.substr(p, e - p);
        const size_t c = tok.find(':');
        out.push_back({ std::stod(tok.substr(0, c)), std::stod(tok.substr(c + 1)) });
        p = e + 1;
    }
    return out;
}

// headlines of a replayed request; hist: the batches of the earlier requests of the same server
// the server's split of the replayed prompt: the system part (67 tokens) ends where the user turn starts
static server_prompt_plan_params perf_replay_prompt(const std::vector<std::pair<double, double>> & batches) {
    server_prompt_plan_params pp;
    for (const auto & b : batches) {
        pp.n_total += (int32_t) b.first;
    }
    pp.n_batch = pp.n_ubatch = 2048;
    pp.checkpoints = batches.size() > 1; // the CHECKPOINTS=0 run has a single batch
    pp.user_starts = { 67 };
    pp.last_user_pos = 67;
    pp.min_step = 8192;
    pp.mem_empty = true;
    return pp;
}

static std::vector<std::pair<double, double>> perf_replay_run(testing & t, const perf_replay & r, std::vector<server_perf_batch> & hist) {
    const auto batches = perf_pairs(r.batches);

    // the plan comes from the prompt, as in the server; it must match the batches the server logged
    const server_prompt_plan_params prompt = perf_replay_prompt(batches);
    const std::vector<int32_t> plan = server_prompt_plan(prompt);
    std::vector<int32_t> logged;
    for (const auto & b : batches) {
        logged.push_back((int32_t) b.first);
    }
    t.assert_true(std::string(r.name) + ": plan from the prompt == logged batches", plan == logged);
    const int64_t t0 = 1000*S;
    std::vector<std::pair<double, double>> out;
    server_perf_info info;
    server_perf_baseline base;
    // the running batch follows the samples: a layer count that drops starts the next batch (the samples and the
    // log are timed by different clocks), a sample without a layer after the full batches is the small last one
    size_t i = 0;
    double prev_layer = 0.0;
    for (const auto & [t, layer] : perf_pairs(r.ticks)) {
        if (layer >= 0 && layer < prev_layer && i + 1 < batches.size()) {
            i++;
        }
        if (layer < 0 && prev_layer > 0 && i + 1 < batches.size() && batches[i + 1].first <= 4) {
            i++;
        }
        prev_layer = layer;
        server_perf_tick k = perf_tick(0);
        k.t_us = t0 + (int64_t) (t*S);
        k.processing = true;
        k.cur_prompt_t_start_us = t0;
        k.plan.hist = hist;
        double t_end = 0.0;
        for (size_t j = 0; j < i; ++j) {
            k.plan.done.push_back({ (int32_t) batches[j].first, (int64_t) (batches[j].second*S) });
            k.cur_prompt_n += (uint64_t) batches[j].first;
            t_end += batches[j].second;
        }
        t_end = std::min(t_end, t); // the batch started before this sample
        k.plan.cur_n = plan[i];
        // what the server publishes once this batch is built: the split of the rest, a checkpoint at the user turn
        server_prompt_plan_params rest = prompt;
        rest.pos = (int32_t) k.cur_prompt_n + plan[i];
        rest.mem_empty = false;
        rest.last_ckpt = prompt.checkpoints && i >= 1 ? 67 : -1;
        k.plan.remaining = server_prompt_plan(rest);
        k.progress = { (int32_t) layer - 1, 93, k.plan.cur_n, 0, t0 + (int64_t) (t_end*S) };
        std::deque<server_perf_tick> ring = { k };
        const json res = server_perf_compute(ring, base, info, k.t_us);
        out.push_back({ t, res["prompt"]["tps_headline"].is_number() ? res["prompt"]["tps_headline"].get<double>() : 0.0 });
    }
    for (const auto & [n, sec] : batches) {
        hist.push_back({ (int32_t) n, (int64_t) (sec*S) });
    }
    return out;
}


// the prompt row once generation starts, and the projection's fallbacks
static void test_perf_prompt_edges(testing & t) {
    static constexpr int64_t T0 = 1000*S; // request start
    t.test("prompt rate final from the first token", [](testing & t) {
        server_perf_counters c;
        server_slot_stats st;
        st.t_start = 1000*S;
        c.on_prompt_start(st.t_start);
        st.n_prompt_processed = 1636;
        st.set_prompt_last(1000*S + 130461920);
        c.on_prompt_batch(st, 1636, 130461920);
        st.n_gen = 1;
        st.t_gen_last = st.t_prompt_last;
        c.on_gen(st);

        server_perf_tick k = perf_tick(1131);
        k.processing     = c.processing.load();
        k.generating     = c.generating.load();
        k.last_prompt_n  = c.last_prompt_n.load();
        k.last_prompt_us = c.last_prompt_us.load();
        const json r = server_perf_compute({ k }, server_perf_baseline(), server_perf_info(), k.t_us);
        t.assert_true("generating, not processing", k.generating && !k.processing);
        assert_near(t, "this request's prompt rate (timings.prompt_per_second)", 1636/130.46192, r["prompt"]["tps_headline"]);
    });

    t.test("no layer reports and no fit: the done-based rate", [](testing & t) {
        server_perf_tick k = perf_tick(1);
        k.processing = true;
        k.cur_prompt_t_start_us = T0;
        k.plan.cur_n = 1636;
        k.progress = { -1, 93, 1636, 0, T0 };
        json r = server_perf_compute({ k }, server_perf_baseline(), server_perf_info(), T0 + 1*S);
        assert_near(t, "nothing done after 1 s: 0, not 1636 t/s", 0.0, r["prompt"]["tps_headline"]);

        // a later batch: the tokens of the finished ones over the elapsed time
        k.plan.done = { { 67, 40*S } };
        k.cur_prompt_n = 67;
        k.plan.cur_n = 1565;
        k.plan.remaining = { 4 };
        k.progress = { -1, 93, 1565, 0, T0 + 40*S };
        r = server_perf_compute({ k }, server_perf_baseline(), server_perf_info(), T0 + 41*S);
        assert_near(t, "67 tokens in 41 s", 67.0/41.0, r["prompt"]["tps_headline"]);
        t.assert_true("not an estimate", !r["prompt"]["estimate"].get<bool>());

        // an unknown split (media, several slots): the same
        k.progress = { 46, 93, 1565, 0, T0 + 40*S };
        k.plan.valid = false;
        r = server_perf_compute({ k }, server_perf_baseline(), server_perf_info(), T0 + 41*S);
        assert_near(t, "invalid plan: tokens done over elapsed", (67.0 + 1565.0*47/93)/41.0, r["prompt"]["tps_headline"]);
    });

    t.test("the small last batch costs a tenth of the running batch, or a decode step, when none was seen", [](testing & t) {
        server_perf_tick k = perf_tick(0);
        k.processing = true;
        k.cur_prompt_t_start_us = T0;
        k.plan.done = { { 67, 40*S } };
        k.cur_prompt_n = 67;
        k.plan.cur_n = 1565;
        k.plan.remaining = { 4 };
        k.progress = { 45, 93, 1565, 0, T0 + 40*S }; // 46 of 93 layers in 30 s
        const json r = server_perf_compute({ k }, server_perf_baseline(), server_perf_info(), T0 + 70*S);
        const double cur_tot = 30.0/(46.0/93.0);
        assert_near(t, "elapsed + rest of the batch + a tenth of it", 1636.0/(70.0 + (cur_tot - 30.0) + 0.1*cur_tot), r["prompt"]["tps_headline"]);
        // a measured decode step larger than that wins (a decode step costs about one pass over the used experts)
        server_perf_tick k2 = k;
        k2.tot_gen_steps = 10;
        k2.tot_gen_us    = 200*S;
        const json r2 = server_perf_compute({ k2 }, server_perf_baseline(), server_perf_info(), T0 + 70*S);
        assert_near(t, "elapsed + rest of the batch + one 20 s decode step", 1636.0/(70.0 + (cur_tot - 30.0) + 20.0), r2["prompt"]["tps_headline"]);
        // batch k/K is shown only for a valid plan
        server_perf_tick k3 = k;
        k3.plan.valid = false;
        const json r3 = server_perf_compute({ k3 }, server_perf_baseline(), server_perf_info(), T0 + 70*S);
        t.assert_true("no batch k/K for an invalid plan", r3["prompt"]["batch"].is_null() && r3["prompt"]["n_batches"].is_null());
        t.assert_true("batch k/K for a valid plan", !r["prompt"]["batch"].is_null());
    });

    t.test("a fit with a negative intercept costs no less than 0", [](testing & t) {
        server_perf_tick k = perf_tick(0);
        k.processing = true;
        k.cur_prompt_t_start_us = T0;
        k.plan.hist = { { 100, 1*S }, { 1000, 100*S } }; // a = -10 s, b = 0.11 s per token
        k.plan.cur_n = 10;
        k.plan.remaining = { 10 };
        k.progress = { -1, 93, 10, 0, T0 };
        const json r = server_perf_compute({ k }, server_perf_baseline(), server_perf_info(), T0 + 5*S);
        assert_near(t, "elapsed only: 20 tokens in 5 s", 4.0, r["prompt"]["tps_headline"]);
    });

    t.test("a batch decoded in parts keeps the rest as the running batch", [](testing & t) {
        server_perf_counters c;
        c.on_prompt_plan(2048, { 4 });
        c.on_prompt_batch(server_slot_stats(), 1024, 10*S);
        t.assert_equal("half left", 1024, c.plan.cur_n);
        c.on_prompt_batch(server_slot_stats(), 1024, 10*S);
        t.assert_equal("done", 0, c.plan.cur_n);
        c.on_prompt_plan(4, {}, false);
        c.on_prompt_plan(4, {}, true);
        t.assert_true("unknown stays unknown for the request", !c.plan.valid);
        c.on_prompt_start(0);
        t.assert_true("a new request starts known", c.plan.valid);
    });
}

static void test_perf_prompt_projection(testing & t) {
    t.test("plan of the server's prompt split", [](testing & t) {
        server_prompt_plan_params p;
        p.n_total = 1636; p.n_batch = 2048; p.n_ubatch = 2048; p.checkpoints = true;
        p.user_starts = { 67 }; p.last_user_pos = 67; p.min_step = 8192; p.mem_empty = true;
        t.assert_true("user turn, main batch, last 4 tokens", server_prompt_plan(p) == std::vector<int32_t>({ 67, 1565, 4 }));
        p.checkpoints = false;
        t.assert_true("without checkpoints: one batch", server_prompt_plan(p) == std::vector<int32_t>({ 1636 }));
        p.checkpoints = true;
        p.n_total = 5000; p.n_batch = p.n_ubatch = 2048;
        // full batches, cut 4 + n_ubatch and 4 tokens before the end
        t.assert_true("long prompt", server_prompt_plan(p) == std::vector<int32_t>({ 67, 2048, 837, 2044, 4 }));
        p.pos = 1632; p.n_total = 1636; p.mem_empty = false; p.last_ckpt = 67;
        t.assert_true("only the last 4 tokens left", server_prompt_plan(p) == std::vector<int32_t>({ 4 }));
    });

    // A1..A3 ran on one server in this order, B1 and A01 each on a fresh one; B1 is replayed a second time with its
    // own batches as the history (self-trained: an optimistic stand-in for the second request of an OFFLOAD=0 server)
    std::vector<server_perf_batch> hist_a;
    std::vector<server_perf_batch> hist_b;
    std::vector<perf_replay> runs(std::begin(PERF_REPLAYS), std::end(PERF_REPLAYS));
    for (const auto & r : PERF_REPLAYS) {
        if (std::string(r.name) == "B1") {
            runs.push_back({ "B1-self-trained", r.final_tps, r.batches, r.ticks });
        }
    }
    for (const auto & r : runs) {
        const std::string name = r.name;
        std::vector<server_perf_batch> fresh;
        auto & hist = name == "A1" || name == "A2" || name == "A3" ? hist_a : name.rfind("B1", 0) == 0 ? hist_b : fresh;
        const bool warm = !hist.empty();
        const auto batches = perf_pairs(r.batches);
        const auto out = perf_replay_run(t, r, hist);
        // from when on the headline stays within 20% of the final rate
        double t_ok = 0.0;
        for (const auto & [tt, h] : out) {
            const double err = std::fabs(h - r.final_tps)/r.final_tps;
            if (err > 0.2) {
                t_ok = tt + 1.0;
            }
        }
        // a server that has run a request before projects from its batch costs; a cold one only knows the running batch
        double limit = 30.0;
        if (!warm) {
            // one batch: its own layer rate; several: the first batch, then a third of the main batch (OFFLOAD=0 runs
            // the first layers on the GPU, so the layer rate settles late)
            limit = batches.size() == 1 ? 10.0 : batches[0].second + batches[1].second/3.0;
        }
        t.assert_true(name + ": within 20% of " + std::to_string(r.final_tps) + " from t = " + std::to_string(t_ok) +
                " s (limit " + std::to_string(limit) + " s)", t_ok <= limit);
        if (t.verbose) {
            printf("    %s: within 20%% from %.0f s, %zu ticks\n", name.c_str(), t_ok, out.size());
        }
    }

    t.test("exactly the final rate once the prompt is done", [](testing & t) {
        for (const auto & r : PERF_REPLAYS) {
            uint64_t n = 0;
            double sec = 0.0;
            for (const auto & [bn, bs] : perf_pairs(r.batches)) {
                n += (uint64_t) bn;
                sec += bs;
            }
            server_perf_tick k = perf_tick(500);
            k.last_prompt_n  = n;
            k.last_prompt_us = (uint64_t) std::llround(n/r.final_tps*1e6);
            const json res = server_perf_compute({ k }, server_perf_baseline(), server_perf_info(), 500*S);
            assert_near(t, std::string(r.name) + " final", (double) n/((double) k.last_prompt_us/1e6), res["prompt"]["tps_headline"]);
            t.assert_true(std::string(r.name) + " final vs timings", std::fabs(res["prompt"]["tps_headline"].get<double>() - r.final_tps) < 1e-3*r.final_tps);
            GGML_UNUSED(sec);
        }
    });
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
    t.test("perf_compute",    test_perf_compute);
    t.test("perf_prompt_projection", test_perf_prompt_projection);
    t.test("perf_prompt_edges", test_perf_prompt_edges);

    return t.summary();
}
