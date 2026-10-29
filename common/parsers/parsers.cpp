#include "parsers.h"

#include "log.h"

void foreach_function(const json & tools, const std::function<void(size_t, const json &)> & fn) {
    for (size_t i = 0; i < tools.size(); i++) {
        const auto & tool = tools[i];
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            LOG_INF("Skipping tool without function: %s", tool.dump(2).c_str());
            continue;
        }
        fn(i, tool);
    }
}

void foreach_parameter(const json & function, const std::function<void(size_t, const common_chat_schema_property &, const common_chat_schema_document_ptr &)> & fn) {
    auto         params = common_chat_tool_parameters(function);
    auto         doc    = std::make_shared<const common_chat_schema_document>(common_chat_schema_from_json(params));
    const auto * object = dynamic_cast<const common_chat_schema_object *>(doc->root.get());
    if (!object) {
        return;
    }
    for (size_t i = 0; i < object->properties.size(); i++) {
        fn(i, object->properties[i], doc);
    }
}

common_peg_parser build_chat_peg_tool_calls(
        common_chat_peg_builder & p, const json & tools, const std::string & marker, const std::string & block,
        bool parallel_tool_calls, bool allow_json_for_string) {
    const std::string call_start   = "<" + marker + block + ">";
    const std::string call_end     = "</" + marker + block + ">";
    const std::string invoke_start = "<" + marker + "invoke";
    const std::string invoke_end   = "</" + marker + "invoke>";
    const std::string param_start  = "<" + marker + "parameter";
    const std::string param_end    = "</" + marker + "parameter>";

    auto tool_choice = p.choice();
    foreach_function(tools, [&](size_t tool_index, const json & tool) {
        const auto & function = tool.at("function");
        const std::string name = function.at("name");
        const std::string tool_rule = "tool-" + std::to_string(tool_index);

        std::vector<common_peg_parser> required_parsers;
        std::vector<common_peg_parser> optional_parsers;
        foreach_parameter(function, [&](size_t param_index, const common_chat_schema_property & param, const common_chat_schema_document_ptr & doc) {
            const std::string arg_rule = tool_rule + "-arg-" + std::to_string(param_index);
            bool is_string;
            bool is_json;
            if (allow_json_for_string) {
                const auto types = param.schema->value_types();
                is_string = types.has(common_chat_schema::TYPE_STRING);
                is_json   = !types.is_only(common_chat_schema::TYPE_STRING);
            } else {
                is_string = param.schema->may_be_string();
                is_json   = !is_string;
            }

            auto arg_open = p.literal(param_start + " name=\"") +
                p.tool_arg_name(p.literal(param.name)) + p.literal("\" string=\"");
            auto arg_value = p.choice();
            if (is_string) {
                arg_value |= p.tool_arg_open(arg_open + p.literal("true\">")) +
                    p.tool_arg_string_value(p.until(param_end));
            }
            if (is_json) {
                arg_value |= p.tool_arg_open(arg_open + p.literal("false\">")) +
                    p.tool_arg_json_value(p.schema(p.json(), arg_rule + "-schema", doc, *param.schema));
            }
            auto arg = p.rule(arg_rule, p.tool_arg(arg_value + p.tool_arg_close(p.literal(param_end))));
            if (param.required) {
                required_parsers.push_back(arg);
            } else {
                optional_parsers.push_back(arg);
            }
        });

        common_peg_parser args_seq = p.eps();
        for (size_t i = 0; i < required_parsers.size(); i++) {
            if (i > 0) {
                args_seq = args_seq + p.space();
            }
            args_seq = args_seq + required_parsers[i];
        }

        if (!optional_parsers.empty()) {
            common_peg_parser any_opt = p.choice();
            for (const auto & opt : optional_parsers) {
                any_opt |= opt;
            }
            args_seq = args_seq + p.repeat(p.space() + any_opt, 0, -1);
        }

        auto func_parser = p.tool(
            p.tool_open(p.literal(invoke_start + " name=\"") + p.tool_name(p.literal(name)) + p.literal("\">\n")) +
            args_seq + p.space() + p.tool_close(p.literal(invoke_end)));
        tool_choice |= p.rule(tool_rule, func_parser);
    });

    auto calls = p.literal(call_start) + p.space() + tool_choice;
    if (parallel_tool_calls) {
        calls = calls + p.zero_or_more(p.space() + tool_choice);
    }
    return p.trigger_rule("tool-call", calls + p.space() + p.literal(call_end));
}
