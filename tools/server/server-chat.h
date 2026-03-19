// Chat conversion functions for server (Responses API, Anthropic API, OAI streaming diffs)

#pragma once

#include "chat.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

// Convert OpenAI Responses API format to OpenAI Chat Completions API format
json common_chat_convert_responses_to_chatcmpl(const json & body);

// Convert Anthropic Messages API format to OpenAI Chat Completions API format
json common_chat_convert_anthropic_to_oai(const json & body);

json common_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff);
