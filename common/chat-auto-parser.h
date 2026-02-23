#pragma once

#include "chat-diff-analyzer.h"
#include "chat.h"
#include "chat-peg-parser.h"
#include "common.h"

#include <chrono>
#include <string>

using json = nlohmann::ordered_json;

namespace autoparser {

struct templates_params {
    json                                  messages;
    json                                  tools;
    common_chat_tool_choice               tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    json                                  json_schema;
    bool                                  parallel_tool_calls = true;
    common_reasoning_format               reasoning_format    = COMMON_REASONING_FORMAT_AUTO;
    bool                                  stream              = true;
    std::string                           grammar;
    bool                                  add_generation_prompt = false;
    bool                                  enable_thinking       = true;
    std::chrono::system_clock::time_point now                   = std::chrono::system_clock::now();
    json                                  extra_context;
    bool                                  add_bos       = false;
    bool                                  add_eos       = false;
    bool                                  is_inference  = true;
    bool                                  add_inference = false;
    bool                                  mark_input    = true;  // whether to mark input strings in the jinja context
};

class peg_generator {
  public:
    static common_chat_params generate_parser(const common_chat_template &    tmpl,
                                              const struct templates_params & inputs);

    static common_chat_params generate_parser(const common_chat_template &    tmpl,
                                              const struct templates_params & inputs,
                                              const autoparser &        analysis);
};

}  // namespace autoparser
