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

// output_modalities of a response, unset: the model switches between text and audio itself
bool parse_output_modalities(const json & j, mtmd_helper_gen_audio_out_mode & mode, std::string & err) {
    if (j.is_null()) {
        mode = MTMD_HELPER_GEN_AUDIO_OUT_AUTO;
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
        mode = MTMD_HELPER_GEN_AUDIO_OUT_MIXED;
    } else {
        mode = audio ? MTMD_HELPER_GEN_AUDIO_OUT_AUDIO : MTMD_HELPER_GEN_AUDIO_OUT_TEXT;
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

json output_modalities_json(mtmd_helper_gen_audio_out_mode mode) {
    switch (mode) {
        case MTMD_HELPER_GEN_AUDIO_OUT_TEXT:  return json::array({"text"});
        case MTMD_HELPER_GEN_AUDIO_OUT_AUDIO: return json::array({"audio"});
        case MTMD_HELPER_GEN_AUDIO_OUT_MIXED: return json::array({"text", "audio"});
        default:                 return nullptr;
    }
}

struct realtime_conversation {
    const server_realtime_deps &  deps;
    server_http_context::ws_conn & conn;

    std::string id = "sess_" + random_string();

    // session config
    std::string                    type = "realtime";
    std::string                    instructions;
    std::string                    language; // transcription
    mtmd_helper_gen_audio_out_mode out_mode          = MTMD_HELPER_GEN_AUDIO_OUT_AUTO;
    int32_t                        max_output_tokens = -1; // "inf"
    int                            input_rate        = 16000;
    int                            output_rate       = 24000;

    // the text around a turn, from the chat template
    std::string first_header; // system prompt + first user header
    std::string user_header;
    std::string user_footer;
    std::string assistant_header;
    std::string assistant_footer;


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
        response.out_mode  = MTMD_HELPER_GEN_AUDIO_OUT_TEXT;
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
        mtmd_helper_gen_audio_out_mode mode = out_mode;
        const json resp = ev.value("response", json::object());
        if (resp.contains("output_modalities")) {
            std::string err;
            if (!parse_output_modalities(resp["output_modalities"], mode, err)) {
                send_error("response.create: " + err);
                return;
            }
        }
        if (!deps.cap.audio_gen && mode != MTMD_HELPER_GEN_AUDIO_OUT_TEXT && mode != MTMD_HELPER_GEN_AUDIO_OUT_AUTO) {
            send_error("response.create: the model cannot generate audio");
            return;
        }
        if (!deps.cap.text_gen && mode != MTMD_HELPER_GEN_AUDIO_OUT_AUDIO && mode != MTMD_HELPER_GEN_AUDIO_OUT_AUTO) {
            send_error("response.create: the model cannot generate text");
            return;
        }
        // audio not committed yet belongs to this turn
        if (has_audio && !commit()) {
            return;
        }

        // the gen-audio pipeline opens the response in the asked mode
        const std::string prompt = user_footer + assistant_header;
        server_control_response response;
        response.n_predict = max_output_tokens;
        if (resp.contains("max_output_tokens") && !parse_max_output_tokens(resp["max_output_tokens"], response.n_predict)) {
            send_error("response.create: max_output_tokens must be an integer or \"inf\"");
            return;
        }
        response.out_mode = mode;

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

//
// server_realtime_slot
//

server_realtime_slot::server_realtime_slot(llama_context * ctx, mtmd_context * mctx, llama_seq_id seq_id, int32_t n_ctx,
                                           const common_params_sampling & sampling, server_response & queue_results, int id_task, int index)
    : ctx(ctx), model(llama_get_model(ctx)), vocab(llama_model_get_vocab(llama_get_model(ctx))), mctx(mctx), seq_id(seq_id),
      n_ctx(n_ctx), sampling(sampling), queue_results(queue_results), id_task(id_task), index(index) {
    if (mtmd_gen_audio_get_info(mctx).type != MTMD_GEN_AUDIO_TYPE_NONE) {
        gen.init(ctx, mctx);
    }
}

void server_realtime_slot::push(server_realtime_op && op) {
    ops.push_back(std::move(op));
}

void server_realtime_slot::cancel() {
    cancelled = responding;
}

bool server_realtime_slot::is_idle() const {
    return ops.empty() && !responding;
}

bool server_realtime_slot::fail(const std::string & msg, enum error_type type) {
    error      = msg;
    error_type = type;
    return false;
}

void server_realtime_slot::send_result(std::string text, std::vector<float> audio, bool response_done) {
    auto res = std::make_unique<server_task_result_realtime>();
    res->id            = id_task;
    res->index         = index;
    res->text          = std::move(text);
    res->audio         = std::move(audio);
    res->response_done = response_done;
    res->cancelled     = response_done && cancelled;
    res->n_decoded     = n_decoded;
    queue_results.send(std::move(res));
}

// appends tokens or embedding rows to the conversation, logits of the last one if asked
bool server_realtime_slot::decode(const llama_token * tokens, const float * rows, int32_t n, bool logits_last) {
    const int32_t n_embd  = llama_model_n_embd_inp(model);
    const int32_t n_batch = llama_n_batch(ctx);
    if (n_past + n >= n_ctx) {
        return fail("the conversation does not fit in the context", ERROR_TYPE_EXCEED_CONTEXT_SIZE);
    }
    for (int32_t i0 = 0; i0 < n; i0 += n_batch) {
        const int32_t n_cur = std::min(n_batch, n - i0);
        llama_batch b = llama_batch_init(n_cur, rows ? n_embd : 0, 1);
        for (int32_t i = 0; i < n_cur; i++) {
            if (rows) {
                memcpy(b.embd + (size_t) i * n_embd, rows + (size_t) (i0 + i) * n_embd, n_embd * sizeof(float));
            } else {
                b.token[i] = tokens[i0 + i];
            }
            b.pos[i]       = n_past + i0 + i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = seq_id;
            b.logits[i]    = logits_last && i0 + i == n - 1;
        }
        b.n_tokens = n_cur;
        const int ret = llama_decode(ctx, b);
        llama_batch_free(b);
        if (ret != 0) {
            return fail("failed to decode the conversation");
        }
    }
    n_past += n;
    return true;
}

// decodes the rows the audio stream has ready
bool server_realtime_slot::audio_rows() {
    while (true) {
        const int32_t n = mtmd_audio_stream_step(stream.get(), LLAMA_TOKEN_NULL);
        if (n == -2) {
            return fail("audio stream error");
        }
        if (n <= 0) {
            return true;
        }
        if (!decode(nullptr, mtmd_audio_stream_get_output_embd(stream.get()), n, false)) {
            return false;
        }
    }
}

// samples the next text token of a response from the logits of the slot's own last decode; several slots
// respond in turns on the same context, so this runs before any other slot decodes
void server_realtime_slot::sample_next() {
    sampled = LLAMA_TOKEN_NULL;
    if (gen.valid() && !gen.next_is_text()) {
        return;
    }
    sampled = common_sampler_sample(smpl.get(), ctx, -1);
    common_sampler_accept(smpl.get(), sampled, true);
}

bool server_realtime_slot::response_start(const server_realtime_op & op) {
    auto sparams = sampling;
    n_predict = op.response.n_predict;
    n_decoded = 0;
    cancelled = false;

    if (gen.valid()) {
        // the gen-audio pipeline continues the conversation and follows the model's text/audio switching
        mtmd_helper::gen_audio::inp inp;
        inp.set_prompt(op.text);
        inp.data.seq_id   = seq_id;
        inp.data.n_past   = n_past;
        inp.data.seed     = sampling.seed;
        inp.data.stream   = true;
        inp.data.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
        inp.data.out_mode = op.response.out_mode;
        if (gen.set_input(inp.get()) != 0) {
            return fail("failed to start the response");
        }
        // the pipeline may bias some text tokens, e.g. the switch to audio of a text-only response
        for (const auto & b : gen.logit_bias()) {
            sparams.logit_bias.push_back(b);
        }
        int32_t ret;
        while ((ret = gen.step_prompt(llama_n_batch(ctx))) > 0) {
        }
        if (ret < 0) {
            return fail("failed to start the response");
        }
        const float * h = llama_get_embeddings_ith(ctx, -1);
        h_state.assign(h, h + llama_model_n_embd(model));
    } else {
        const llama_tokens toks = common_tokenize(vocab, op.text, false, true);
        if (toks.empty()) {
            return fail("failed to start the response");
        }
        if (!decode(toks.data(), nullptr, (int32_t) toks.size(), true)) {
            return false;
        }
    }
    smpl.reset(common_sampler_init(model, sparams));
    // the logits of the last decode are this slot's only until another slot decodes
    sample_next();
    responding = true;
    return true;
}

// one text token or audio frame of the response being generated
bool server_realtime_slot::response_step() {
    bool        stop = false;
    std::string text;
    std::vector<float> audio;

    if (gen.valid()) {
        const float * h_next = nullptr;
        if (gen.step_gen(sampled, h_state.data(), &h_next, &stop) != 0) {
            return fail("response generation failed");
        }
        if (sampled != LLAMA_TOKEN_NULL && !stop && !llama_vocab_is_control(vocab, sampled)) {
            text = common_token_to_piece(vocab, sampled, false);
        }
        if (h_next) {
            h_state.assign(h_next, h_next + llama_model_n_embd(model));
        }
    } else {
        stop = llama_vocab_is_eog(vocab, sampled);
        if (!stop) {
            if (!llama_vocab_is_control(vocab, sampled)) {
                text = common_token_to_piece(vocab, sampled, false);
            }
            if (!decode(&sampled, nullptr, 1, true)) {
                return false;
            }
        }
    }
    n_decoded++;

    // the pipeline decodes on its own, the response ends before the context is full
    const bool ctx_full = gen.valid() ?
        llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id) + 2 >= n_ctx :
        n_past + 1 >= n_ctx;
    const bool done = stop || cancelled || ctx_full || (n_predict > 0 && n_decoded >= n_predict);
    if (!done) {
        sample_next();
    }
    if (gen.valid()) {
        if (done) {
            gen.flush();
        }
        int32_t sample_rate = 0;
        const char * data   = nullptr;
        size_t data_len     = 0;
        if (gen.get_output(&sample_rate, &data, &data_len) == 0 && data_len > 0) {
            audio.assign((const float *) data, (const float *) data + data_len / sizeof(float));
        }
    }
    if (!text.empty() || !audio.empty()) {
        send_result(std::move(text), std::move(audio));
    }
    if (done) {
        if (gen.valid()) {
            // the pipeline decoded the response, the sequence ends where it stopped
            n_past = llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id) + 1;
        }
        send_result({}, {}, true);
        responding = false;
        cancelled  = false;
    }
    return true;
}

bool server_realtime_slot::update() {
    llama_set_embeddings(ctx, true);

    if (responding) {
        return response_step();
    }

    while (!ops.empty() && !responding) {
        server_realtime_op op = std::move(ops.front());
        ops.pop_front();
        switch (op.action) {
            case SERVER_CONTROL_ACTION_TEXT_APPEND:
                {
                    const llama_tokens toks = common_tokenize(vocab, op.text, false, true);
                    if (!toks.empty() && !decode(toks.data(), nullptr, (int32_t) toks.size(), false)) {
                        return false;
                    }
                } break;
            case SERVER_CONTROL_ACTION_AUDIO_APPEND:
                {
                    if (!stream) {
                        stream.reset(mtmd_audio_stream_init(mctx, mtmd_audio_stream_params_default()));
                        if (!stream) {
                            return fail("failed to initialize the audio stream");
                        }
                    }
                    mtmd_audio_stream_push(stream.get(), op.audio.data(), op.audio.size());
                    if (!audio_rows()) {
                        return false;
                    }
                } break;
            case SERVER_CONTROL_ACTION_AUDIO_COMMIT:
            case SERVER_CONTROL_ACTION_AUDIO_END:
                {
                    // the end of the audio of a turn
                    if (stream) {
                        mtmd_audio_stream_flush(stream.get(), true);
                        if (!audio_rows()) {
                            return false;
                        }
                        stream.reset();
                    }
                } break;
            case SERVER_CONTROL_ACTION_RESPONSE_CREATE:
                {
                    if (!response_start(op)) {
                        return false;
                    }
                } break;
            case SERVER_CONTROL_ACTION_CLEAR:
                {
                    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
                    stream.reset();
                    n_past = 0;
                } break;
            default:
                break;
        }
    }
    return true;
}
