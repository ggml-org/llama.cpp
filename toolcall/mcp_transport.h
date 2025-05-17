#pragma once

#include "mcp_messages.h"
#include <string>
#include <tuple>
#include <map>

namespace toolcall
{
    template <typename T>
    using callback = std::function<void(const T&)>;

    template <typename... MessageTypes>
    class mcp_message_observer {
    public:
        template <typename T>
        void subscribe(std::string key, callback<T> callback) {
            auto& map =
                std::get<std::map<std::string, toolcall::callback<T>>>(
                    subscribers_);

            map.insert({key, callback});
        }

        template <typename T>
        void subscribe(callback<T> callback) {
            auto& map =
                std::get<std::map<std::string, toolcall::callback<T>>>(
                    subscribers_);

            map.insert({T::Method, callback});
        }

        template <typename T>
        void unsubscribe(std::string key) {
            auto& map =
                std::get<std::map<std::string, toolcall::callback<T>>>(
                    subscribers_);

            map.erase(key);
        }

        void notify(const nlohmann::json & message) {
            std::string key;
            if (message.contains("id")) {
                key = message["id"].dump();

            } else if (message.contains("method")) {
                key = message["method"].dump();

            } else {
                return;
            }
            std::apply([&key, &message, this](auto&... maps) {
                (..., [&] {
                    auto it = maps.find(key);
                    if (it != maps.end()) {
                        using callback_type = decltype(it->second);
                        using T = typename std::decay<typename callback_type::argument_type>::type;

                        it->second(T::fromJson(message));
                        maps.erase(it);
                    }
                }());
            }, subscribers_);
        }

    private:
        std::tuple<std::map<std::string, toolcall::callback<MessageTypes>>...> subscribers_;
    };

    class mcp_transport : public mcp_message_observer<mcp::initialize_response,
                                                      mcp::tools_list_response,
                                                      mcp::tools_list_changed_notification,
                                                      mcp::tools_call_response> {
    public:
        virtual ~mcp_transport() = default;

        template <typename Req, typename Resp>
        bool send(const Req & message, callback<Resp> on_response) {
            if (message.id().has_value()) {
                std::string id = message.id().value().dump();
                subscribe(id, on_response);
            }
            return send(message);
        }

        template <typename Req>
        bool send(const Req & message) {
            nlohmann::json json = message.toJson();
            return send(json.dump(-1));
        }

        virtual void start() = 0;
        virtual void stop() = 0;
        virtual bool send(const std::string & request_json) = 0;
    };
}
