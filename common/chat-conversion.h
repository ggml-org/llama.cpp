// Chat conversion functions for OpenAI API compatibility

#pragma once

#include "chat.h"
#include "nlohmann/json.hpp"

#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// Convert OpenAI Responses API format to OpenAI Chat Completions API format
json common_chat_convert_responses_to_chatcmpl(const json & body);

// Convert Anthropic Messages API format to OpenAI Chat Completions API format
json common_chat_convert_anthropic_to_oai(const json & body);

// DEPRECATED: only used in tests
json common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text = false);

json common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools);

json common_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff);
