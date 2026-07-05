#include "testing.h"

#include "../tools/server/server-chat.h"
#include "../tools/server/server-common.h"
#include "../tools/server/server-decision.h"
#include "../tools/server/server-schema.h"
#include "../tools/server/server-slot-sched.h"
#include "../tools/server/server-task.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
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

static void test_server_tokens_prefix_hashes(testing & t) {
    const size_t n_blk = server_tokens::hash_block_size;

    auto make_tokens = [](size_t n) {
        llama_tokens res(n);
        for (size_t i = 0; i < n; i++) {
            res[i] = 100 + (llama_token) i;
        }
        return res;
    };

    {
        server_tokens a(make_tokens(3 * n_blk + 10), false);
        t.assert_equal("full blocks only", (size_t) 3, a.get_prefix_hashes().size());
        t.assert_equal("same tokens", 3 * n_blk, a.get_common_prefix_lower(a.clone().get_prefix_hashes()));
        t.assert_equal("no hashes",   (size_t) 0, a.get_common_prefix_lower({}));

        // the exact value is 2 * n_blk + 5
        llama_tokens toks = make_tokens(3 * n_blk + 10);
        toks[2 * n_blk + 5] = 7;
        server_tokens b(toks, false);
        t.assert_equal("diverge in block 2", 2 * n_blk, a.get_common_prefix_lower(b.get_prefix_hashes()));
        t.assert_equal("exact, diverge in block 2", 2 * n_blk + 5, a.get_common_prefix(b));
        t.assert_equal("exact, same tokens",        3 * n_blk + 10, a.get_common_prefix(a.clone()));
        t.assert_equal("b is shorter",       1 * n_blk, a.get_common_prefix_lower(server_tokens(make_tokens(n_blk + 3), false).get_prefix_hashes()));

        toks[0] = 7;
        t.assert_equal("diverge in block 0", (size_t) 0, a.get_common_prefix_lower(server_tokens(toks, false).get_prefix_hashes()));
    }

    {
        server_tokens a(make_tokens(3 * n_blk), false);
        const auto hashes = a.get_prefix_hashes();

        server_tokens b = a.clone();
        b.set_token(n_blk + 1, 7);
        t.assert_equal("set_token", n_blk, b.get_common_prefix_lower(hashes));

        b = a.clone();
        b.keep_first(n_blk + 1);
        for (size_t i = n_blk + 1; i < 3 * n_blk; i++) {
            b.push_back(7);
        }
        t.assert_equal("keep_first + push_back", n_blk, b.get_common_prefix_lower(hashes));

        b.clear();
        t.assert_equal("clear", (size_t) 0, b.get_prefix_hashes().size());
    }

    {
        // the image chunk starts in block 0 and ends in a later block
        mtmd::input_chunks chunks(mtmd_test_create_input_chunks());
        const mtmd_input_chunk * image = chunks[1];
        const size_t n_image = mtmd_input_chunk_get_n_tokens(image);
        const size_t i_image = n_blk - 4;

        auto make_mtmd = [&](bool with_image, llama_token last) {
            server_tokens res(make_tokens(i_image), true);
            if (with_image) {
                res.push_back(image);
            }
            while (res.size() < i_image + n_image) {
                res.push_back(5);
            }
            res.push_back(last);
            return res;
        };

        server_tokens a = make_mtmd(true, 1);
        t.assert_true("image is longer than a block", n_image > n_blk);
        t.assert_equal("exact value", i_image + n_image, a.get_common_prefix(make_mtmd(true, 2)));
        t.assert_equal("block ends inside the image", i_image, a.get_common_prefix_lower(make_mtmd(true, 2).get_prefix_hashes()));
        t.assert_equal("text in place of the image",  (size_t) 0, a.get_common_prefix_lower(make_mtmd(false, 1).get_prefix_hashes()));
    }
}

static void test_server_tokens(testing & t) {
    t.test("basic",         test_server_tokens_basic);
    t.test("pos",           test_server_tokens_pos);
    t.test("keep_first",    test_server_tokens_keep_first);
    t.test("common_prefix", test_server_tokens_common_prefix);
    t.test("serialize",     test_server_tokens_serialize);
    t.test("clone",         test_server_tokens_clone);
    t.test("prefix_hashes", test_server_tokens_prefix_hashes);
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

//
// server_prompt
//

static void test_server_prompt_reuse(testing & t) {
    // 110 text tokens, the position of a token is its index
    auto make_prompt = [](std::vector<std::array<llama_pos, 2>> checkpoints) {
        llama_tokens toks(110);
        for (size_t i = 0; i < toks.size(); i++) {
            toks[i] = 100 + (llama_token) i;
        }

        server_prompt res;
        res.tokens = server_tokens(toks, false);
        for (const auto & range : checkpoints) {
            auto ckpt = std::make_shared<common_prompt_checkpoint>();
            ckpt->pos_min  = range[0];
            ckpt->pos_max  = range[1];
            ckpt->n_tokens = range[1] + 1;
            res.checkpoints.push_back(std::move(ckpt));
        }
        return res;
    };

    t.test("recurrent", [&](testing & t) {
        // a recurrent state holds one position: the live state is at 109, n_swa is 0
        const auto prompt = make_prompt({{95, 95}, {99, 99}, {105, 105}});
        const llama_pos pos_min = 109;

        auto res = prompt.get_reuse(110, true, 0, pos_min);
        t.assert_true("new tokens after the prompt, no checkpoint", res.ckpt == nullptr);
        t.assert_equal("new tokens after the prompt, n_past", 110, res.n_past);

        res = prompt.get_reuse(110, false, 0, pos_min);
        t.assert_true("same prompt, last checkpoint", res.ckpt && res.ckpt->pos_max == 105);
        t.assert_equal("same prompt, n_past", 106, res.n_past);
        t.assert_equal("same prompt, pos_next", 106, res.pos_next);

        // the closest checkpoint before the new tokens is used, the one after them is not
        res = prompt.get_reuse(104, true, 0, pos_min);
        t.assert_true("cut at 104, checkpoint", res.ckpt && res.ckpt->pos_max == 99);
        t.assert_equal("cut at 104, n_past", 100, res.n_past);
        t.assert_equal("cut at 104, pos_next", 100, res.pos_next);

        // the state at 99 already holds the token at 99, it cannot be used to process it again
        res = prompt.get_reuse(99, true, 0, pos_min);
        t.assert_true("cut at 99, checkpoint", res.ckpt && res.ckpt->pos_max == 95);
        t.assert_equal("cut at 99, n_past", 96, res.n_past);

        res = prompt.get_reuse(50, true, 0, pos_min);
        t.assert_true("cut at 50, no checkpoint", res.ckpt == nullptr);
        t.assert_equal("cut at 50, n_past", 0, res.n_past);
        t.assert_equal("cut at 50, pos_next", 0, res.pos_next);
    });

    t.test("swa", [&](testing & t) {
        // n_swa is 8: the live state starts at 102, the checkpoint holds the positions 92 to 99
        const auto prompt = make_prompt({{92, 99}});
        const llama_pos pos_min = 102;

        auto res = prompt.get_reuse(110, true, 8, pos_min);
        t.assert_true("new tokens after the prompt, no checkpoint", res.ckpt == nullptr);
        t.assert_equal("new tokens after the prompt, n_past", 110, res.n_past);

        // the token at 104 needs the positions 97 to 103, the live state starts too late for a cut at 104
        res = prompt.get_reuse(104, true, 8, pos_min);
        t.assert_true("cut at 104, checkpoint", res.ckpt != nullptr);
        t.assert_equal("cut at 104, n_past", 99, res.n_past);
        t.assert_equal("cut at 104, pos_next", 99, res.pos_next);

        res = prompt.get_reuse(95, true, 8, pos_min);
        t.assert_true("cut at 95, no checkpoint", res.ckpt == nullptr);
        t.assert_equal("cut at 95, n_past", 0, res.n_past);
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

    t.test("lora_should_clear_cache", [&](testing & t) {
        // the adapter pointer is only read when the configs differ
        t.assert_true("both empty", !lora_should_clear_cache({}, {}));
        t.assert_true("same",       !lora_should_clear_cache({make(p0, 1.0f)}, {make(p0, 1.0f)}));
        t.assert_true("to no lora", lora_should_clear_cache({make(p0, 0.0f)}, {}));
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
                int x = i;
                p.write(std::move(x));
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

    for (const float bad : {std::nanf(""), INFINITY}) {
        bool thrown = false;
        try {
            ctx.format_answer(choice, {{bad, 0.0f}});
        } catch (const std::exception &) {
            thrown = true;
        }
        t.assert_true("nan or +inf", thrown);
    }
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
            ctx.parse_questions(body, false);
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

    reject_state("state is text", {{"state", 42}}, "\"state\" must be the OCR text");

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

static void test_stop_strings(testing & t) {
    common_params params_base;

    const auto eval = [&](const json & data) {
        return server_schema::eval_llama_cmpl_schema(nullptr, params_base, {}, data);
    };

    {
        json data;
        data["stop"] = "";
        t.assert_true("empty string is ignored", eval(data).antiprompt.empty());
    }

    {
        json data;
        data["stop"] = json::array({""});
        t.assert_true("empty string in array is ignored", eval(data).antiprompt.empty());
    }

    {
        json data;
        data["stop"] = json::array({"", "zzz"});
        const std::vector<std::string> expected = {"zzz"};
        t.assert_true("only non-empty strings are kept", eval(data).antiprompt == expected);
    }

    {
        params_base.antiprompt = {"User:"};
        json data;
        data["stop"] = json::array({""});
        const std::vector<std::string> expected = {"User:"};
        t.assert_true("falls back to CLI defaults", eval(data).antiprompt == expected);
    }
}

//
// glob_match
//

static void test_glob_match(testing & t) {
    t.test("patterns", [](testing & t) {
        t.assert_true("literal", glob_match("src/main.cpp", "src/main.cpp"));
        t.assert_true("star", glob_match("src/*.cpp", "src/main.cpp"));
        t.assert_true("star stops at /", !glob_match("*.cpp", "src/main.cpp"));
        t.assert_true("double star", glob_match("**/*.cpp", "a/b/c/main.cpp"));
        t.assert_true("double star at end", glob_match("src/**", "src/a/b"));
        t.assert_true("question mark", glob_match("ma?n.cpp", "main.cpp"));
        t.assert_true("question mark not /", !glob_match("a?b", "a/b"));
        t.assert_true("class", glob_match("[a-c]x", "bx"));
        t.assert_true("negated class", !glob_match("[!a-c]x", "bx"));
        t.assert_true("unclosed class", glob_match("[ab", "[ab"));
        t.assert_true("mismatch", !glob_match("main.cpp", "main.c"));
    });
    t.test("long input", [](testing & t) {
        const std::string path(100000, 'a');
        t.assert_true("long literal", glob_match(path, path));
        t.assert_true("long path", glob_match("**", path));
        t.assert_true("too many wildcards", !glob_match(std::string(1000, '*'), "a"));
    });
}

//
// OpenAI speech API
//

static void test_speech_convert(testing & t) {
    const json base = {{"model", "tts"}, {"input", "Hello"}, {"voice", "alba"}};

    auto with = [&](const char * key, const json & value) {
        json body = base;
        body[key] = value;
        return body;
    };
    auto rejected = [&](const char * key, const json & value) {
        try {
            convert_speech_to_tts(with(key, value));
        } catch (const std::invalid_argument &) {
            return true;
        }
        return false;
    };

    // defaults
    json out = convert_speech_to_tts(base);
    t.assert_equal("input kept",        std::string("Hello"), out.at("input").get<std::string>());
    t.assert_equal("voice kept",        std::string("alba"),  out.at("voice").get<std::string>());
    t.assert_equal("default format",    std::string("wav"),   out.at("response_format").get<std::string>());
    t.assert_true ("no stream",         !out.contains("stream"));

    // response_format
    out = convert_speech_to_tts(with("response_format", "wav"));
    t.assert_equal("wav",               std::string("wav"),   out.at("response_format").get<std::string>());
    out = convert_speech_to_tts(with("response_format", "pcm"));
    t.assert_equal("pcm is 16-bit",     std::string("pcm16"), out.at("response_format").get<std::string>());
    for (const char * format : {"mp3", "opus", "aac", "flac", "pcm16"}) {
        t.assert_true(std::string("rejects ") + format, rejected("response_format", format));
    }

    // custom voice
    out = convert_speech_to_tts(with("voice", json{{"id", "voice_1234"}}));
    t.assert_equal("voice object",      std::string("voice_1234"), out.at("voice").get<std::string>());

    // stream_format
    out = convert_speech_to_tts(with("stream_format", "audio"));
    t.assert_true ("audio streams",     out.at("stream").get<bool>());
    t.assert_true ("rejects sse",       rejected("stream_format", "sse"));

    // speed is forwarded, the model decides whether it is supported
    out = convert_speech_to_tts(with("speed", 1.5));
    t.assert_equal("speed forwarded",   1.5, out.at("speed").get<double>());

    // instructions are ignored
    t.assert_true ("instructions",      !rejected("instructions", "whisper"));
}

static void test_pcm_f32_to_s16(testing & t) {
    auto sample = [](const std::string & bytes, size_t i) {
        int16_t v;
        memcpy(&v, bytes.data() + i * sizeof(v), sizeof(v));
        return (int) v;
    };

    const std::vector<float> samples = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, -2.0f};
    const std::string bytes = pcm_f32_to_s16(samples.data(), samples.size());
    t.assert_equal("size",           (size_t) 12, bytes.size());
    t.assert_equal("zero",           0,      sample(bytes, 0));
    t.assert_equal("max",            32767,  sample(bytes, 1));
    t.assert_equal("min",            -32767, sample(bytes, 2));
    t.assert_equal("rounds",         16384,  sample(bytes, 3));
    t.assert_equal("clamps above",   32767,  sample(bytes, 4));
    t.assert_equal("clamps below",   -32767, sample(bytes, 5));
    t.assert_true ("empty",          pcm_f32_to_s16(nullptr, 0).empty());
}

static void test_chat_reasoning_presence(testing & t) {
    common_chat_msg previous;
    previous.role = "assistant";
    auto current = previous;
    current.reasoning_content_present = true;
    const auto diffs = common_chat_msg_diff::compute_diffs(previous, current);
    t.assert_equal("one empty reasoning delta", (size_t) 1, diffs.size());
    if (diffs.size() != 1) {
        return;
    }
    const auto delta = server_chat_msg_diff_to_json_oaicompat(diffs[0]);
    t.assert_true("empty reasoning field is present", delta == json({ { "reasoning_content", "" } }));
    t.assert_true("unchanged presence has no delta", common_chat_msg_diff::compute_diffs(current, current).empty());

    previous = current;
    current.content = "<think>Visible content</think>";
    auto streamed = delta;
    streamed["role"] = "assistant";
    for (const auto & diff : common_chat_msg_diff::compute_diffs(previous, current)) {
        const auto chunk = server_chat_msg_diff_to_json_oaicompat(diff);
        for (const auto & [key, value] : chunk.items()) {
            streamed[key] = value;
        }
    }
    const auto history = common_chat_msgs_parse_oaicompat(json::array({ streamed }));
    t.assert_true("streamed and complete messages agree", history[0] == current);

    previous = current;
    current.reasoning_content = "Thought";
    const auto reasoning_diffs = common_chat_msg_diff::compute_diffs(previous, current);
    t.assert_equal("one reasoning text delta", (size_t) 1, reasoning_diffs.size());
    if (reasoning_diffs.size() == 1) {
        t.assert_true("reasoning deltas compare text", !(diffs[0] == reasoning_diffs[0]));
        t.assert_true("reasoning text is serialized", server_chat_msg_diff_to_json_oaicompat(reasoning_diffs[0]) ==
                      json({ { "reasoning_content", "Thought" } }));
    }
}

static void test_chat_empty_reasoning_formats(testing & t) {
    const std::string source =
        "{# GCML tool_calls #}{% for m in messages %}"
        "{{ m.role + '<|role_sep|>\n' + m.content + '<|message_sep|>\n\n' }}{% endfor %}"
        "{% if add_generation_prompt %}assistant<|role_sep|>\n<think>{% endif %}";
    common_chat_templates_ptr templates(common_chat_templates_init(nullptr, source));
    common_chat_templates_inputs inputs;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    const std::string text = "<think>Visible content</think>\n\nAnswer";
    for (bool with_content : { false, true }) {
        task_result_state state(common_chat_session(templates.get(), nullptr, inputs));
        server_task_result_cmpl_partial partial{};
        partial.res_type = TASK_RESPONSE_TYPE_OAI_RESP;
        partial.n_decoded = 1;
        partial.content = common_chat_input("</think>\n\n");
        partial.update(state);
        t.assert_true("empty reasoning updates state", state.chat_msg.has_reasoning_content());
        t.assert_true("thinking block started", state.thinking_block_started);

        auto anthropic_blocks = json::array();
        for (int step = 0; step < (with_content ? 2 : 1); ++step) {
            if (step == 1) {
                partial.content = common_chat_input(text);
                partial.n_decoded = 2;
                partial.update(state);
            }
            const auto anthropic_events = partial.to_json_anthropic();
            for (const auto & event : anthropic_events) {
                const auto & data = event.at("data");
                if (event.at("event") == "content_block_start") {
                    t.assert_equal("Anthropic block index", anthropic_blocks.size(), data.at("index").get<size_t>());
                    anthropic_blocks.push_back(data.at("content_block"));
                } else if (event.at("event") == "content_block_delta") {
                    const auto & delta = data.at("delta");
                    const char * key = delta.at("type") == "thinking_delta" ? "thinking" : "text";
                    anthropic_blocks[data.at("index").get<size_t>()][key] = delta.at(key);
                }
            }
            const auto response_events = partial.to_json_oaicompat_resp();
            int added_items = 0;
            for (const auto & event : response_events) {
                if (event.at("event") == "response.output_item.added") {
                    ++added_items;
                    t.assert_equal("Responses item index", step, event.at("data").at("output_index").get<int>());
                    t.assert_true("Responses item type", event.at("data").at("item").at("type") ==
                                  (step == 0 ? "reasoning" : "message"));
                }
            }
            t.assert_equal("Responses starts each item", 1, added_items);
        }

        server_task_result_cmpl_final result{};
        result.update(state);
        const auto final_events = result.to_json_anthropic_stream();
        for (const auto & event : final_events) {
            if (event.at("event") == "content_block_delta") {
                const auto & data = event.at("data");
                if (data.at("delta").at("type") == "signature_delta") {
                    anthropic_blocks[data.at("index").get<size_t>()]["signature"] = data.at("delta").at("signature");
                }
            }
        }
        const auto anthropic = result.to_json_anthropic();
        t.assert_true("Anthropic stream preserves empty block", anthropic_blocks == anthropic.at("content"));
        const auto messages = server_chat_convert_anthropic_to_oai({ { "messages", json::array({ anthropic }) } });
        const auto history = common_chat_msgs_parse_oaicompat(messages.at("messages"));
        t.assert_equal("Anthropic history size", (size_t) 1, history.size());
        t.assert_true("Anthropic history keeps reasoning field", history[0].has_reasoning_content());
        t.assert_equal("Anthropic history keeps text", with_content ? text : "", history[0].render_content());

        const auto complete_events = result.to_json_oaicompat_resp_stream();
        const auto streamed = complete_events.back().at("data").at("response");
        for (const auto & response : { result.to_json_oaicompat_resp(), streamed }) {
            const auto converted = server_chat_convert_responses_to_chatcmpl({ { "input", response.at("output") } });
            const auto response_history = common_chat_msgs_parse_oaicompat(converted.at("messages"));
            t.assert_equal("Responses history size", (size_t) 1, response_history.size());
            t.assert_true("Responses history keeps reasoning field", response_history[0].has_reasoning_content());
            t.assert_equal("Responses history keeps text", with_content ? text : "", response_history[0].render_content());
        }
    }

    common_chat_msg prefill;
    prefill.role = "assistant";
    prefill.content = "Hello";
    prefill.reasoning_content_present = true;
    inputs.messages = { prefill };
    inputs.continue_final_message = COMMON_CHAT_CONTINUATION_CONTENT;
    task_result_state state(common_chat_session(templates.get(), nullptr, inputs));
    server_task_result_cmpl_partial progress{};
    progress.res_type = TASK_RESPONSE_TYPE_OAI_RESP;
    progress.is_progress = true;
    progress.update(state);
    t.assert_true("prompt progress has no chat deltas", progress.oaicompat_msg_diffs.empty());
    t.assert_true("prompt progress leaves thinking unopened", !state.thinking_block_started);
    t.assert_true("Anthropic prompt progress has no content events", progress.to_json_anthropic().empty());
    const auto progress_events = progress.to_json_oaicompat_resp();
    t.assert_true("Responses prompt progress creates response", progress_events[0].at("event") == "response.created");
    t.assert_true("Responses prompt progress is retained", progress_events.back().at("data").contains("prompt_progress"));
    server_task_result_cmpl_partial partial{};
    partial.res_type = TASK_RESPONSE_TYPE_OAI_RESP;
    partial.n_decoded = 1;
    partial.content = common_chat_input(" world");
    partial.update(state);
    const auto anthropic_events = partial.to_json_anthropic();
    t.assert_true("Anthropic message starts before content blocks", anthropic_events[0].at("event") == "message_start");
    const auto response_events = partial.to_json_oaicompat_resp();
    size_t added_items = 0;
    for (const auto & event : response_events) {
        if (event.at("event") == "response.output_item.added") {
            t.assert_equal("continued Responses item index", added_items++, event.at("data").at("output_index").get<size_t>());
        }
    }
    t.assert_equal("continued Responses starts reasoning and text", (size_t) 2, added_items);
    size_t started_blocks = 0;
    for (const auto & event : anthropic_events) {
        if (event.at("event") == "content_block_start") {
            t.assert_equal("continued Anthropic block index", started_blocks++, event.at("data").at("index").get<size_t>());
        }
    }
    t.assert_equal("continued Anthropic starts reasoning and text", (size_t) 2, started_blocks);

    task_result_state realtime(common_chat_session(templates.get(), nullptr, inputs));
    std::vector<common_chat_msg_diff> realtime_diffs;
    realtime.update_chat_msg(common_chat_input(" world"), true, realtime_diffs, true);
    t.assert_equal("Realtime has only the content delta", (size_t) 1, realtime_diffs.size());
    if (realtime_diffs.size() == 1) {
        t.assert_equal("Realtime continuation text", std::string(" world"), realtime_diffs[0].content_delta);
        t.assert_true("Realtime has no synthetic reasoning header", !realtime_diffs[0].has_reasoning_content());
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
    t.test("server_prompt_reuse", test_server_prompt_reuse);
    t.test("server_pipe",   test_server_pipe);
    t.test("decision_escape", test_decision_escape);
    t.test("rerank_template", test_rerank_template);
    t.test("decision_class_scores", test_decision_class_scores);
    t.test("decision_ocr_jev_validation", test_decision_ocr_jev_validation);
    t.test("decision_depth",  test_decision_depth);
    t.test("glob_match",      test_glob_match);
    t.test("dry_sequence_breakers", test_dry_sequence_breakers);
    t.test("stop_strings",    test_stop_strings);
    t.test("speech_convert",  test_speech_convert);
    t.test("pcm_f32_to_s16",  test_pcm_f32_to_s16);
    t.test("chat_reasoning_presence", test_chat_reasoning_presence);
    t.test("chat_empty_reasoning_formats", test_chat_empty_reasoning_formats);

    return t.summary();
}
