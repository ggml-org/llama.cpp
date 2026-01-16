#include "value.h"
#include "runtime.h"
#include "caps.h"

// note: the json dependency is only for defining input in a convenient way
// we can remove it in the future when we figure out a better way to define inputs using jinja::value
#include <nlohmann/json.hpp>

#include <functional>
#include <sstream>

#define FILENAME "jinja-caps"

using json = nlohmann::ordered_json;

namespace jinja {

using caps_json_fn = std::function<json()>;
using caps_analyze_fn = std::function<void(bool, value &, value &)>;

static void caps_try_execute(jinja::program & prog,
                             const caps_json_fn & messages_fn,
                             const caps_json_fn & tools_fn,
                             const caps_analyze_fn & analyze_fn) {
    context ctx;
    ctx.is_get_stats = true;
    jinja::global_from_json(ctx, json{
        {"messages", messages_fn()},
        {"tools", tools_fn()},
        {"bos_token", ""},
        {"eos_token", ""},
        {"add_generation_prompt", true}
    }, true);

    auto messages = ctx.get_val("messages");
    auto tools = ctx.get_val("tools");

    bool success = false;
    try {
        jinja::runtime runtime(ctx);
        runtime.execute(prog);
        success = true;
    } catch (const std::exception & e) {
        JJ_DEBUG("Exception during execution: %s", e.what());
        // ignore exceptions during capability analysis
    }

    analyze_fn(success, messages, tools);
}

// for debugging only
static void caps_print_stats(value & v, const std::string & path) {
    std::string ops;
    for (const auto & name : v->stats.ops) {
        ops += name + " ";
    }
    JJ_DEBUG("Value %s, type: %s %s, ops: %s",
                path.c_str(),
                v->type().c_str(),
                v->stats.used ? "(used)" : "",
                ops.c_str());
}

std::string caps::to_string() const {
    std::ostringstream ss;
    ss << "Caps(\n";
    ss << "  requires_typed_content=" << requires_typed_content << "\n";
    ss << "  supports_tools=" << supports_tools << "\n";
    ss << "  supports_tool_calls=" << supports_tool_calls << "\n";
    ss << "  supports_parallel_tool_calls=" << supports_parallel_tool_calls << "\n";
    ss << "  supports_system_role=" << supports_system_role << "\n";
    ss << ")";
    return ss.str();
}

caps caps_get(jinja::program & prog) {
    caps result;

    static const auto has_op = [](value & v, const std::string & op_name) {
        return v->stats.ops.find(op_name) != v->stats.ops.end();
    };

    // case: typed content requirement
    caps_try_execute(
        prog,
        [&]() {
            // messages
            return json::array({
                {
                    {"role", "user"},
                    {"content", "content"}
                }
            });
        },
        [&]() {
            // tools
            return json{nullptr};
        },
        [&](bool, value & messages, value &) {
            auto & content = messages->at(0)->at("content");
            caps_print_stats(content, "messages[0].content");
            if (has_op(content, "selectattr") || has_op(content, "array_access")) {
                // accessed as an array
                result.requires_typed_content = true;
            }
        }
    );


    // case: system prompt support
    caps_try_execute(
        prog,
        [&]() {
            // messages
            return json::array({
                {
                    {"role", "system"},
                    {"content", "System message"}
                },
                {
                    {"role", "user"},
                    {"content", "User message"}
                },
            });
        },
        [&]() {
            // tools
            return json::array();
        },
        [&](bool, value & messages, value &) {
            auto & content = messages->at(0)->at("content");
            caps_print_stats(content, "messages[0].content");
            if (!content->stats.used) {
                result.supports_system_role = false;
            }
        }
    );

    // case: tools support
    caps_try_execute(
        prog,
        [&]() {
            // messages
            return json::array({
                {
                    {"role", "user"},
                    {"content", "User message"},
                },
                {
                    {"role", "assistant"},
                    {"content", "Assistant message"},
                    {"tool_calls", json::array({
                        {
                            {"id", "call1"},
                            {"type", "function"},
                            {"function", {
                                {"name", "tool1"},
                                {"arguments", {
                                    {"arg", "value"}
                                }}
                            }}
                        },
                        {
                            {"id", "call2"},
                            {"type", "function"},
                            {"function", {
                                {"name", "tool2"},
                                {"arguments", {
                                    {"arg", "value"}
                                }}
                            }}
                        }
                    })}
                },
                {
                    {"role", "user"},
                    {"content", "User message"},
                },
            });
        },
        [&]() {
            // tools
            return json::array({
                {
                    {"name", "tool"},
                    {"type", "function"},
                    {"function", {
                        {"name", "tool"},
                        {"description", "Tool description"},
                        {"parameters", {
                            {"type", "object"},
                            {"properties", {
                                {"arg", {
                                    {"type", "string"},
                                    {"description", "Arg description"},
                                }},
                            }},
                            {"required", json::array({ "arg" })},
                        }},
                    }},
                },
            });
        },
        [&](bool success, value & messages, value & tools) {
            if (!success) {
                result.supports_tool_calls = false;
                result.supports_tools = false;
                return;
            }

            auto & tool_name = tools->at(0)->at("function")->at("name");
            caps_print_stats(tool_name, "tools[0].function.name");
            if (!tool_name->stats.used) {
                result.supports_tools = false;
            }

            auto & tool_calls = messages->at(1)->at("tool_calls");;
            caps_print_stats(tool_calls, "messages[1].tool_calls");
            if (!tool_calls->stats.used) {
                result.supports_tool_calls = false;
            }

            // check for second tool call usage
            auto & tool_call_1 = tool_calls->at(1)->at("function");
            caps_print_stats(tool_call_1, "messages[1].tool_calls[1].function");
            if (!tool_call_1->stats.used) {
                result.supports_parallel_tool_calls = false;
            }
        }
    );

    JJ_DEBUG("%s\n", result.to_string().c_str());

    return result;
}

} // namespace jinja
