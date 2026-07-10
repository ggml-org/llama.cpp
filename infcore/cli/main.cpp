// infcore CLI — корпоративная лицензия.
// Терминальный клиент к gateway: список моделей, переключение (admin), чат.
// Адрес/ключ: флаги --url/--key либо env INFCORE_URL/INFCORE_KEY.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "httplib.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace {

struct Opts {
    std::string url = "http://127.0.0.1:8080";
    std::string key;
};

void usage() {
    std::printf(
        "infcore-cli [--url URL] [--key KEY] <команда>\n"
        "  models                 доступные модели (/v1/models)\n"
        "  admin-models           полный список со статусом (/admin/models)\n"
        "  enable  <model>        включить модель\n"
        "  disable <model>        выключить модель\n"
        "  chat -m <model> [текст] чат (текст из аргумента или stdin)\n"
        "\nenv: INFCORE_URL, INFCORE_KEY (флаги имеют приоритет)\n");
}

httplib::Headers auth(const Opts& o) {
    if (o.key.empty()) return {};
    return {{"Authorization", "Bearer " + o.key}};
}

// Выполняет запрос, печатает тело при ошибке. Возвращает распарсенный JSON или null.
json request(const Opts& o, const char* method, const std::string& path, const std::string& body) {
    httplib::Client cli(o.url);
    cli.set_read_timeout(300, 0);
    httplib::Result r =
        (std::string(method) == "POST")
            ? cli.Post(path, auth(o), body, "application/json")
            : cli.Get(path, auth(o));
    if (!r) { std::fprintf(stderr, "infcore-cli: нет связи с %s\n", o.url.c_str()); std::exit(2); }
    if (r->status < 200 || r->status >= 300) {
        std::fprintf(stderr, "infcore-cli: ошибка %d: %s\n", r->status, r->body.c_str());
        std::exit(1);
    }
    json j = json::parse(r->body, nullptr, false);
    if (j.is_discarded()) {
        std::fprintf(stderr, "infcore-cli: неожиданный ответ (не JSON): %s\n", r->body.c_str());
        std::exit(1);
    }
    return j;
}

int cmd_models(const Opts& o, bool admin) {
    json j = request(o, "GET", admin ? "/admin/models" : "/v1/models", "");
    if (!j.contains("data")) { std::printf("(пусто)\n"); return 0; }
    for (const auto& m : j.at("data")) {
        const std::string id = m.value("id", "?");
        const std::string modality = m.value("modality", "");
        if (admin) {
            std::printf("%-24s %-10s %-9s %s\n", id.c_str(), modality.c_str(),
                        m.value("enabled", true) ? "enabled" : "disabled",
                        m.value("managed", false) ? "managed" : m.value("backend_url", "").c_str());
        } else {
            std::printf("%-24s %s\n", id.c_str(), modality.c_str());
        }
    }
    return 0;
}

int cmd_toggle(const Opts& o, const std::string& name, bool enable) {
    json j = request(o, "POST", "/admin/models/" + name + (enable ? "/enable" : "/disable"), "");
    std::printf("%s: %s\n", name.c_str(), j.value("enabled", enable) ? "enabled" : "disabled");
    return 0;
}

int cmd_chat(const Opts& o, const std::string& model, const std::string& text) {
    json body = {{"model", model},
                 {"messages", json::array({{{"role", "user"}, {"content", text}}})}};
    json j = request(o, "POST", "/v1/chat/completions", body.dump());
    if (j.contains("choices") && !j.at("choices").empty()) {
        const auto& msg = j.at("choices")[0].value("message", json::object());
        std::printf("%s\n", msg.value("content", "").c_str());
    } else {
        std::printf("%s\n", j.dump(2).c_str());
    }
    return 0;
}

std::string read_stdin() {
    std::string s, line;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), stdin)) > 0) s.append(buf, n);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    Opts o;
    if (const char* e = std::getenv("INFCORE_URL")) o.url = e;
    if (const char* e = std::getenv("INFCORE_KEY")) o.key = e;

    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--url" && i + 1 < argc)      o.url = argv[++i];
        else if (a == "--key" && i + 1 < argc) o.key = argv[++i];
        else args.push_back(a);
    }
    if (args.empty()) { usage(); return 1; }

    const std::string& cmd = args[0];
    if (cmd == "models")        return cmd_models(o, false);
    if (cmd == "admin-models")  return cmd_models(o, true);
    if (cmd == "enable"  && args.size() >= 2) return cmd_toggle(o, args[1], true);
    if (cmd == "disable" && args.size() >= 2) return cmd_toggle(o, args[1], false);
    if (cmd == "chat") {
        std::string model, text;
        for (size_t i = 1; i < args.size(); ++i) {
            if (args[i] == "-m" && i + 1 < args.size()) model = args[++i];
            else { if (!text.empty()) text += " "; text += args[i]; }
        }
        if (model.empty()) { std::fprintf(stderr, "infcore-cli: укажите модель: chat -m <model>\n"); return 1; }
        if (text.empty()) text = read_stdin();
        return cmd_chat(o, model, text);
    }

    usage();
    return 1;
}
