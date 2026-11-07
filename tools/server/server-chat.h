// Chat conversion functions for server (Responses API, Anthropic API, OAI streaming diffs)

#pragma once

#include "chat.h"
#include "server-common.h"
#include "server-http.h"

#include "json.h"

// Convert OpenAI Responses API format to OpenAI Chat Completions API format
json server_chat_convert_responses_to_chatcmpl(const json & body);

// Convert OpenAI Responses API tools to OpenAI Chat Completions API tools, non-function tools are skipped
json server_chat_convert_responses_tools(const json & tools);

// Flatten a Responses-API namespaced tool name. Returns name unchanged when ns is empty.
std::string server_chat_responses_tool_name(const std::string & ns, const std::string & name);

// Split flattened tool name into (namespace, tool_name)
std::pair<std::string, std::string> server_chat_responses_split_tool_name(const std::string & name);

// Convert Anthropic Messages API format to OpenAI Chat Completions API format
json server_chat_convert_anthropic_to_oai(const json & body);

// convert OpenAI transcriptions API format to OpenAI Chat Completions API format
json convert_transcriptions_to_chatcmpl(
    const json & body,
    const common_chat_templates * tmpls,
    const std::map<std::string, uploaded_file> & in_files,
    std::vector<raw_buffer> & out_files);

// prompt of the "asr" chat template, live ASR models (e.g. LFM3-Audio live ASR) transcribe with it
// empty if the model has no such template, throws on a template error
llama_tokens server_chat_get_asr_prompt(const llama_model * model);

json server_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff);
