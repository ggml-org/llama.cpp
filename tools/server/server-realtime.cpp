#include "server-realtime.h"
#include "server-common.h"
#include "server-context.h"
#include "server-task.h"

#include "base64.hpp"
#include "chat.h"
#include "common.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstring>
#include <mutex>
#include <thread>

//
// a realtime conversation, modeled after the OpenAI realtime API (GA)
//
// one slot holds the conversation for the whole session: text and audio go into it as they arrive, so a response
// starts right after the last audio. session.type "realtime" holds a conversation, "transcription" transcribes each
// committed audio buffer on its own.
//

namespace {

// output_modalities of a response: unset, the model switches with its own tokens; audio and interleaved responses
// open with the token of the mode, text-only responses cannot switch to audio
enum realtime_out_mode {
    REALTIME_OUT_AUTO,
    REALTIME_OUT_TEXT,
    REALTIME_OUT_AUDIO,
    REALTIME_OUT_MIXED,
};

bool parse_output_modalities(const json & j, realtime_out_mode & mode, std::string & err) {
    if (j.is_null()) {
        mode = REALTIME_OUT_AUTO;
        return true;
    }
    if (!j.is_array() || j.empty()) {
        err = "output_modalities must be a non-empty array";
        return false;
    }
    bool text  = false;
    bool audio = false;
    for (const auto & m : j) {
        if (m == "text") {
            text = true;
        } else if (m == "audio") {
            audio = true;
        } else {
            err = "output_modalities: unknown modality " + m.dump();
            return false;
        }
    }
    if (text && audio) {
        mode = REALTIME_OUT_MIXED;
    } else {
        mode = audio ? REALTIME_OUT_AUDIO : REALTIME_OUT_TEXT;
    }
    return true;
}

bool parse_max_output_tokens(const json & j, int32_t & n) {
    if (j == "inf") {
        n = -1;
        return true;
    }
    if (!j.is_number_integer() || j.get<int64_t>() < 1) {
        return false;
    }
    n = (int32_t) std::min<int64_t>(j.get<int64_t>(), INT32_MAX);
    return true;
}

json output_modalities_json(realtime_out_mode mode) {
    switch (mode) {
        case REALTIME_OUT_TEXT:  return json::array({"text"});
        case REALTIME_OUT_AUDIO: return json::array({"audio"});
        case REALTIME_OUT_MIXED: return json::array({"text", "audio"});
        default:                 return nullptr;
    }
}

struct realtime_conversation {
    const server_realtime_deps &  deps;
    server_http_context::ws_conn & conn;

    std::string id = "sess_" + random_string();

    // session config
    std::string       type = "realtime";
    std::string       instructions;
    std::string       language; // transcription
    realtime_out_mode out_mode          = REALTIME_OUT_AUTO;
    int32_t           max_output_tokens = -1; // "inf"
    int               input_rate        = 16000;
    int               output_rate       = 24000;

    // the text around a turn, from the chat template
    std::string first_header; // system prompt + first user header
    std::string user_header;
    std::string user_footer;
    std::string assistant_header;
    std::string assistant_footer;

    llama_token tok_audio_start = LLAMA_TOKEN_NULL;
    llama_token tok_mixed_start = LLAMA_TOKEN_NULL;

    // conversation state, owned by the reader thread
    bool        started   = false; // something went into the slot, the session config is fixed
    bool        turn_open = false; // a user turn is being filled
    bool        has_audio = false; // the open turn has audio not committed yet
    int         n_turns   = 0;
    std::string next_header;       // text that goes in before the next user turn

    // the item or response the results belong to, shared with the result loop
    std::mutex  mutex;
    std::string last_item_id;
    std::string transcribe_item_id; // transcription: the item being transcribed
    std::string transcript;
    std::string response_id;
    std::string response_item_id;
    std::string response_text;
    int         idx_text  = -1;
    int         idx_audio = -1;
    json        parts     = json::array();

    std::atomic<bool> closed{false};

    realtime_conversation(const server_realtime_deps & deps, server_http_context::ws_conn & conn) : deps(deps), conn(conn) {}

    //
    // events
    //

    void send_event(json ev) {
        ev["event_id"] = "event_" + random_string();
        conn.send(safe_json_to_str(ev));
    }

    void send_error(const std::string & message, const std::string & type_err = "invalid_request_error") {
        send_event({
            {"type",  "error"},
            {"error", {{"type", type_err}, {"code", nullptr}, {"message", message}, {"param", nullptr}}},
        });
    }

    json session_json() const {
        json audio_in = {{"format", {{"type", "audio/pcm"}, {"rate", input_rate}}}, {"turn_detection", nullptr}};
        if (type == "transcription") {
            audio_in["transcription"] = {{"model", deps.meta.model_name}, {"language", language.empty() ? json(nullptr) : json(language)}};
            return {
                {"id",     id},
                {"object", "realtime.transcription_session"},
                {"type",   "transcription"},
                {"audio",  {{"input", audio_in}}},
            };
        }
        json audio = {{"input", audio_in}};
        if (deps.cap.audio_gen) {
            audio["output"] = {{"format", {{"type", "audio/pcm"}, {"rate", output_rate}}}};
        }
        return {
            {"id",                id},
            {"object",            "realtime.session"},
            {"type",              "realtime"},
            {"model",             deps.meta.model_name},
            {"instructions",      instructions},
            {"output_modalities", output_modalities_json(out_mode)},
            {"max_output_tokens", max_output_tokens < 0 ? json("inf") : json(max_output_tokens)},
            {"audio",             audio},
        };
    }

    //
    // slot controls, run on the slot in the order they are posted
    //

    bool control(server_control_action action, std::string text = {}, std::vector<float> audio = {},
                 server_control_response response = {}) {
        server_response_reader ctl(deps.queue_tasks, deps.queue_results, 1);
        server_task task(SERVER_TASK_TYPE_CONTROL);
        task.id                       = ctl.get_new_id();
        task.params.control_action    = action;
        task.params.control_cmpl_id   = id;
        task.params.control_text      = std::move(text);
        task.params.control_audio     = std::move(audio);
        task.params.control_response  = response;
        ctl.post_task(std::move(task));
        auto result = ctl.next([this]() { return closed.load() || !conn.is_open(); });
        auto * res = result ? dynamic_cast<server_task_result_control *>(result.get()) : nullptr;
        return res && res->success;
    }

    //
    // chat template
    //

    std::string render(const std::vector<common_chat_msg> & messages, bool add_generation_prompt = false) const {
        const auto & chat = deps.meta.chat_params;
        common_chat_templates_inputs inputs;
        inputs.messages              = messages;
        inputs.add_generation_prompt = add_generation_prompt;
        inputs.use_jinja             = chat.use_jinja;
        inputs.enable_thinking       = false;
        inputs.chat_template_kwargs  = chat.chat_template_kwargs;
        return common_chat_templates_apply(chat.tmpls.get(), inputs).prompt.str();
    }

    // the text around a user and an assistant turn, from placeholder messages
    bool derive_turn_text() {
        const std::string user  = "@@USER@@";
        const std::string reply = "@@REPLY@@";
        auto msg = [](const char * role, const std::string & content) {
            common_chat_msg m;
            m.role    = role;
            m.content = content;
            return m;
        };
        std::vector<common_chat_msg> sys;
        if (!instructions.empty()) {
            sys.push_back(msg("system", instructions));
        }
        auto with = [&](std::vector<common_chat_msg> rest, bool add_generation_prompt = false) {
            std::vector<common_chat_msg> all = sys;
            all.insert(all.end(), rest.begin(), rest.end());
            return render(all, add_generation_prompt);
        };
        const std::string a = with({msg("user", user)});
        const std::string g = with({msg("user", user)}, true);
        const std::string c = with({msg("user", user), msg("assistant", reply)});
        const std::string d = with({msg("user", user), msg("assistant", reply), msg("user", user)});

        const size_t pa = a.find(user);
        const size_t pc = c.find(reply);
        const size_t pd = d.find(user, c.size());
        if (pa == std::string::npos || pc == std::string::npos || pd == std::string::npos ||
            c.compare(0, a.size(), a) != 0 || d.compare(0, c.size(), c) != 0 || g.compare(0, a.size(), a) != 0 || pc < a.size()) {
            return false;
        }
        // the response opens with the generation prompt, a past turn may render differently (e.g. its think block)
        first_header     = a.substr(0, pa);
        user_footer      = a.substr(pa + user.size());
        assistant_header = g.substr(a.size());
        assistant_footer = c.substr(pc + reply.size());
        user_header      = d.substr(c.size(), pd - c.size());
        return true;
    }

    //
    // conversation
    //

    // the header of a user turn goes in before its first audio or text
    bool ensure_turn() {
        if (turn_open) {
            return true;
        }
        const std::string header = n_turns == 0 ? first_header : next_header + user_header;
        if (!control(SERVER_CONTROL_ACTION_TEXT_APPEND, header)) {
            return false;
        }
        next_header.clear();
        started   = true;
        turn_open = true;
        return true;
    }

    void on_append(const json & ev) {
        std::string pcm;
        try {
            pcm = base64::decode(json_value(ev, "audio", std::string()));
        } catch (const std::exception &) {
            send_error("invalid base64 audio");
            return;
        }
        std::vector<float> samples(pcm.size() / 2);
        for (size_t i = 0; i < samples.size(); i++) {
            int16_t v;
            memcpy(&v, pcm.data() + 2 * i, sizeof(v));
            samples[i] = v / 32768.0f;
        }
        if (samples.empty() || !ensure_turn()) {
            return;
        }
        if (control(SERVER_CONTROL_ACTION_AUDIO_APPEND, {}, std::move(samples))) {
            has_audio = true;
        }
    }

    // ends the audio of the open turn, it becomes a user item
    bool commit() {
        if (!has_audio) {
            send_error("input_audio_buffer.commit: the buffer is empty", "input_audio_buffer_commit_empty");
            return false;
        }
        if (!control(SERVER_CONTROL_ACTION_AUDIO_END)) {
            return false;
        }
        has_audio = false;

        std::lock_guard<std::mutex> lock(mutex);
        const json prev = last_item_id.empty() ? json(nullptr) : json(last_item_id);
        last_item_id = "item_" + random_string();
        send_event({{"type", "input_audio_buffer.committed"}, {"previous_item_id", prev}, {"item_id", last_item_id}});
        const json item = {{"id", last_item_id}, {"object", "realtime.item"}, {"type", "message"}, {"role", "user"},
                           {"status", "completed"}, {"content", json::array({{{"type", "input_audio"}}})}};
        send_event({{"type", "conversation.item.added"}, {"previous_item_id", prev}, {"item", item}});
        send_event({{"type", "conversation.item.done"},  {"previous_item_id", prev}, {"item", item}});
        return true;
    }

    void on_commit() {
        if (!commit() || type != "transcription") {
            return;
        }
        // transcription: the committed audio is transcribed on its own, then the slot is cleared for the next one
        {
            std::lock_guard<std::mutex> lock(mutex);
            transcribe_item_id = last_item_id;
            transcript.clear();
        }
        server_control_response response;
        response.n_predict = max_output_tokens;
        for (llama_token t : {tok_audio_start, tok_mixed_start}) {
            if (t != LLAMA_TOKEN_NULL) {
                response.banned.push_back(t);
            }
        }
        if (!control(SERVER_CONTROL_ACTION_RESPONSE_CREATE, user_footer + assistant_header, {}, response)) {
            return;
        }
        control(SERVER_CONTROL_ACTION_CLEAR);
        turn_open = false;
        n_turns   = 0;
    }

    void on_item_create(const json & ev) {
        if (type == "transcription") {
            send_error("conversation.item.create: not supported in a transcription session");
            return;
        }
        const json item = ev.value("item", json::object());
        if (json_value(item, "type", std::string()) != "message" || json_value(item, "role", std::string()) != "user") {
            send_error("conversation.item.create: only user messages are supported");
            return;
        }
        std::string text;
        for (const auto & part : item.value("content", json::array())) {
            const std::string t = json_value(part, "type", std::string());
            if (t != "input_text") {
                send_error("conversation.item.create: unsupported content type: " + t);
                return;
            }
            text += json_value(part, "text", std::string());
        }
        if (!ensure_turn() || !control(SERVER_CONTROL_ACTION_TEXT_APPEND, text)) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        const json prev = last_item_id.empty() ? json(nullptr) : json(last_item_id);
        last_item_id = item.contains("id") && item["id"].is_string() ? item["id"].get<std::string>() : "item_" + random_string();
        const json out = {{"id", last_item_id}, {"object", "realtime.item"}, {"type", "message"}, {"role", "user"},
                          {"status", "completed"}, {"content", json::array({{{"type", "input_text"}, {"text", text}}})}};
        send_event({{"type", "conversation.item.added"}, {"previous_item_id", prev}, {"item", out}});
        send_event({{"type", "conversation.item.done"},  {"previous_item_id", prev}, {"item", out}});
    }

    void on_response_create(const json & ev) {
        if (type == "transcription") {
            send_error("response.create: not supported in a transcription session");
            return;
        }
        if (!turn_open) {
            send_error("response.create: commit audio or create a message first", "conversation_empty");
            return;
        }
        realtime_out_mode mode = out_mode;
        const json resp = ev.value("response", json::object());
        if (resp.contains("output_modalities")) {
            std::string err;
            if (!parse_output_modalities(resp["output_modalities"], mode, err)) {
                send_error("response.create: " + err);
                return;
            }
        }
        if (!deps.cap.audio_gen && mode != REALTIME_OUT_TEXT && mode != REALTIME_OUT_AUTO) {
            send_error("response.create: the model cannot generate audio");
            return;
        }
        if (!deps.cap.text_gen && mode != REALTIME_OUT_AUDIO && mode != REALTIME_OUT_AUTO) {
            send_error("response.create: the model cannot generate text");
            return;
        }
        // audio not committed yet belongs to this turn
        if (has_audio && !commit()) {
            return;
        }

        // audio and interleaved responses open with the token of the mode, a text-only response cannot switch
        std::string prompt = user_footer + assistant_header;
        server_control_response response;
        response.n_predict = max_output_tokens;
        if (resp.contains("max_output_tokens") && !parse_max_output_tokens(resp["max_output_tokens"], response.n_predict)) {
            send_error("response.create: max_output_tokens must be an integer or \"inf\"");
            return;
        }
        auto piece = [&](llama_token t) { return common_token_to_piece(deps.vocab, t, true); };
        switch (mode) {
            case REALTIME_OUT_TEXT:
                for (llama_token t : {tok_audio_start, tok_mixed_start}) {
                    if (t != LLAMA_TOKEN_NULL) {
                        response.banned.push_back(t);
                    }
                }
                break;
            case REALTIME_OUT_AUDIO: prompt += piece(tok_audio_start); break;
            case REALTIME_OUT_MIXED: prompt += piece(tok_mixed_start); break;
            default: break;
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            response_id      = "resp_" + random_string();
            response_item_id = "item_" + random_string();
            response_text.clear();
            parts     = json::array();
            idx_text  = -1;
            idx_audio = -1;
            send_event({{"type", "response.created"}, {"response", {{"id", response_id}, {"object", "realtime.response"},
                        {"status", "in_progress"}, {"output_modalities", output_modalities_json(mode)}, {"output", json::array()}}}});
            send_event({{"type", "response.output_item.added"}, {"response_id", response_id}, {"output_index", 0},
                        {"item", {{"id", response_item_id}, {"object", "realtime.item"}, {"type", "message"}, {"role", "assistant"},
                                  {"status", "in_progress"}, {"content", json::array()}}}});
        }
        if (!control(SERVER_CONTROL_ACTION_RESPONSE_CREATE, prompt, {}, response)) {
            return;
        }
        // the response ends the turn, its tokens are in the slot; the footer goes in before the next turn
        next_header = assistant_footer;
        turn_open   = false;
        n_turns++;
    }

    void on_session_update(const json & ev) {
        const json session = ev.value("session", json::object());
        if (session.contains("type") && session["type"].is_string() && session["type"] != type) {
            const std::string t = session["type"];
            if (t != "realtime" && t != "transcription") {
                send_error("session.update: unknown session type " + t);
                return;
            }
            if (started) {
                send_error("session.update: the session type cannot change after the first audio or message");
                return;
            }
            type = t;
        }
        if (session.contains("instructions") && session["instructions"].is_string()) {
            if (started) {
                send_error("session.update: instructions cannot change after the first audio or message");
            } else {
                instructions = session["instructions"];
            }
        }
        if (session.contains("output_modalities")) {
            std::string err;
            if (!parse_output_modalities(session["output_modalities"], out_mode, err)) {
                send_error("session.update: " + err);
                return;
            }
        }
        if (session.contains("max_output_tokens") && !parse_max_output_tokens(session["max_output_tokens"], max_output_tokens)) {
            send_error("session.update: max_output_tokens must be an integer or \"inf\"");
            return;
        }
        const json audio = session.value("audio", json::object());
        const json in    = audio.value("input", json::object());
        if (in.contains("format") && in["format"].is_object() && json_value(in["format"], "rate", input_rate) != input_rate) {
            send_error("session.update: the input audio must be PCM16 at " + std::to_string(input_rate) + " Hz");
        }
        if (in.contains("transcription") && in["transcription"].is_object()) {
            language = json_value(in["transcription"], "language", language);
        }
        if (!started && !configure()) {
            send_error("the chat template does not fit the realtime turn format", "server_error");
            return;
        }
        send_event({{"type", "session.updated"}, {"session", session_json()}});
    }

    // turn text of the session type, before anything went into the slot
    bool configure() {
        const std::string saved = instructions;
        if (type == "transcription") {
            const auto preset = common_chat_get_asr_prompt(deps.meta.chat_params.tmpls.get(), language);
            instructions = preset.system;
        }
        const bool ok = derive_turn_text();
        instructions = saved;
        return ok;
    }

    void handle(const std::string & msg) {
        json ev;
        try {
            ev = json::parse(msg);
        } catch (const std::exception &) {
            send_error("invalid JSON");
            return;
        }
        const std::string t = json_value(ev, "type", std::string());
        if (t == "session.update" || t == "transcription_session.update") {
            on_session_update(ev);
        } else if (t == "input_audio_buffer.append") {
            on_append(ev);
        } else if (t == "input_audio_buffer.commit") {
            on_commit();
        } else if (t == "input_audio_buffer.clear") {
            // the audio already went into the conversation
            send_event({{"type", "input_audio_buffer.cleared"}});
        } else if (t == "conversation.item.create") {
            on_item_create(ev);
        } else if (t == "response.create") {
            on_response_create(ev);
        } else if (t == "response.cancel") {
            control(SERVER_CONTROL_ACTION_RESPONSE_CANCEL);
        } else {
            send_error("unknown event type: " + t);
        }
    }

    //
    // results of the slot
    //

    static std::string encode_pcm16(const std::vector<float> & samples) {
        std::string bytes(samples.size() * sizeof(int16_t), '\0');
        for (size_t i = 0; i < samples.size(); i++) {
            const int16_t v = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, samples[i])) * 32767.0f);
            memcpy(&bytes[2 * i], &v, sizeof(v));
        }
        return base64::encode(bytes);
    }

    // a part of the response is opened on its first delta, in order of appearance
    int open_part(const char * part_type) {
        json part = std::string(part_type) == "output_text" ? json{{"type", part_type}, {"text", ""}} : json{{"type", part_type}, {"transcript", ""}};
        parts.push_back(part);
        send_event({{"type", "response.content_part.added"}, {"response_id", response_id}, {"item_id", response_item_id},
                    {"output_index", 0}, {"content_index", parts.size() - 1}, {"part", part}});
        return (int) parts.size() - 1;
    }

    void on_result(const server_task_result_realtime & res) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!transcribe_item_id.empty()) {
            if (!res.text.empty()) {
                transcript += res.text;
                send_event({{"type", "conversation.item.input_audio_transcription.delta"}, {"item_id", transcribe_item_id},
                            {"content_index", 0}, {"delta", res.text}});
            }
            if (res.response_done) {
                send_event({{"type", "conversation.item.input_audio_transcription.completed"}, {"item_id", transcribe_item_id},
                            {"content_index", 0}, {"transcript", transcript}});
                transcribe_item_id.clear();
            }
            return;
        }

        const json base = {{"response_id", response_id}, {"item_id", response_item_id}, {"output_index", 0}};
        if (!res.text.empty()) {
            if (idx_text < 0) {
                idx_text = open_part("output_text");
            }
            response_text += res.text;
            json ev = base;
            ev["type"]          = "response.output_text.delta";
            ev["content_index"] = idx_text;
            ev["delta"]         = res.text;
            send_event(ev);
        }
        if (!res.audio.empty()) {
            if (idx_audio < 0) {
                idx_audio = open_part("output_audio");
            }
            json ev = base;
            ev["type"]          = "response.output_audio.delta";
            ev["content_index"] = idx_audio;
            ev["delta"]         = encode_pcm16(res.audio);
            send_event(ev);
        }
        if (!res.response_done) {
            return;
        }

        const std::string status = res.cancelled ? "cancelled" : "completed";
        for (size_t i = 0; i < parts.size(); i++) {
            json ev = base;
            ev["content_index"] = i;
            if ((int) i == idx_text) {
                parts[i]["text"] = response_text;
                ev["type"]       = "response.output_text.done";
                ev["text"]       = response_text;
            } else {
                ev["type"] = "response.output_audio.done";
            }
            send_event(ev);
            ev.erase("text");
            ev["type"] = "response.content_part.done";
            ev["part"] = parts[i];
            send_event(ev);
        }
        const json item = {{"id", response_item_id}, {"object", "realtime.item"}, {"type", "message"}, {"role", "assistant"},
                           {"status", status}, {"content", parts}};
        send_event({{"type", "response.output_item.done"}, {"response_id", response_id}, {"output_index", 0}, {"item", item}});
        send_event({{"type", "response.done"}, {"response", {{"id", response_id}, {"object", "realtime.response"}, {"status", status},
                    {"output", json::array({item})}, {"usage", {{"output_tokens", res.n_decoded}}}}}});
        last_item_id = response_item_id;
    }

    //
    // session
    //

    void run() {
        input_rate = mtmd_get_audio_sample_rate(deps.mctx);
        if (deps.cap.audio_gen) {
            output_rate = mtmd_gen_audio_get_info(deps.mctx).sample_rate;
        }
        auto single = [&](const char * text) {
            const llama_tokens t = common_tokenize(deps.vocab, text, false, true);
            return t.size() == 1 ? t[0] : LLAMA_TOKEN_NULL;
        };
        tok_audio_start = single("<|audio_start|>");
        tok_mixed_start = single("<|mixed_start|>");
        if (!configure()) {
            send_error("the chat template does not fit the realtime turn format", "server_error");
            conn.close();
            return;
        }

        server_response_reader rd(deps.queue_tasks, deps.queue_results, 1);
        {
            server_task task(SERVER_TASK_TYPE_REALTIME);
            task.id                       = rd.get_new_id();
            task.params.sampling          = deps.sampling;
            task.params.stream            = true;
            task.params.oaicompat_cmpl_id = id;
            task.params.oaicompat_model   = deps.meta.model_name;
            rd.post_task(std::move(task));
        }
        // the slot answers once it holds the session
        {
            auto result = rd.next([this]() { return !conn.is_open(); });
            if (!result || result->is_error()) {
                if (result) {
                    send_error(json_value(result->to_json(), "message", std::string("server error")), "server_error");
                }
                conn.close();
                return;
            }
        }
        send_event({{"type", "session.created"}, {"session", session_json()}});

        // client events are handled on this thread, results are sent from the loop below
        std::thread reader([this]() {
            std::string msg;
            while (conn.read(msg) != 0) {
                handle(msg);
            }
            closed = true;
            conn.close();
        });

        while (true) {
            auto result = rd.next([this]() { return !conn.is_open(); });
            if (!result) {
                break;
            }
            if (result->is_error()) {
                send_error(json_value(result->to_json(), "message", std::string("server error")), "server_error");
                break;
            }
            if (auto * res = dynamic_cast<server_task_result_realtime *>(result.get())) {
                on_result(*res);
            }
        }

        closed = true;
        conn.close();
        reader.join();
    }
};

} // namespace

void server_realtime_conversation(const server_realtime_deps & deps, server_http_context::ws_conn & conn) {
    realtime_conversation session(deps, conn);
    session.run();
}
