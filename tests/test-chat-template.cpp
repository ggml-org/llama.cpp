#include <string>
#include <vector>
#include <sstream>
#include <regex>
#include <iostream>
#include <fstream>
#include <filesystem>

#include <nlohmann/json.hpp>

#undef NDEBUG
#include <cassert>

#include "llama.h"
#include "common.h"
#include "chat.h"
#include "jinja/runtime.h"
#include "jinja/parser.h"
#include "jinja/lexer.h"
#include "jinja/caps.h"

using json = nlohmann::ordered_json;

int main_automated_tests(void);

void run_multiple(std::string dir_path, bool stop_on_first_failure, json input, bool use_common = false);
void run_single(std::string contents, json input, bool use_common = false, const std::string & output_path = "");



std::string HELP = R"(
Usage: test-chat-template [OPTIONS] PATH_TO_TEMPLATE
Options:
  -h, --help               Show this help message and exit.
  --json <path>            Path to the JSON input file.
  --stop-on-first-fail     Stop testing on the first failure (default: false).
  --no-common              Use direct Jinja engine instead of common chat templates (default: use common).
  --output <path>          Path to output results (only for single template runs).
If PATH_TO_TEMPLATE is a file, runs that single template.
If PATH_TO_TEMPLATE is a directory, runs all .jinja files in that directory.
If PATH_TO_TEMPLATE is omitted, runs automated tests (default CI mode).
)";

std::string DEFAULT_JSON = R"({
    "messages": [
        {
            "role": "user",
            "content": "Hello, how are you?"
        },
        {
            "role": "assistant",
            "content": "I am fine, thank you!"
        }
    ],
    "bos_token": "<s>",
    "eos_token": "</s>",
    "add_generation_prompt": true
})";

int main(int argc, char ** argv) {
    std::vector<std::string> args(argv, argv + argc);

    std::string tmpl_path;
    std::string json_path;
    std::string output_path;
    bool stop_on_first_fail = false;
    bool use_common = true;

    for (size_t i = 1; i < args.size(); i++) {
        if (args[i] == "--help" || args[i] == "-h") {
            std::cout << HELP << "\n";
            return 0;
        } else if (args[i] == "--json" && i + 1 < args.size()) {
            json_path = args[i + 1];
            i++;
        } else if (args[i] == "--stop-on-first-fail") {
            stop_on_first_fail = true;
        } else if (args[i] == "--output" && i + 1 < args.size()) {
            output_path = args[i + 1];
            i++;
        } else if (args[i] == "--no-common") {
            use_common = true;
        } else if (tmpl_path.empty()) {
            tmpl_path = args[i];
        } else {
            std::cerr << "Unknown argument: " << args[i] << "\n";
            std::cout << HELP << "\n";
            return 1;
        }
    }

    if (tmpl_path.empty()) {
        return main_automated_tests();
    }

    json input_json;
    if (!json_path.empty()) {
        std::ifstream json_file(json_path);
        if (!json_file) {
            std::cerr << "Error: Could not open JSON file: " << json_path << "\n";
            return 1;
        }
        std::string content = std::string(
            std::istreambuf_iterator<char>(json_file),
            std::istreambuf_iterator<char>());
        input_json = json::parse(content);
    } else {
        input_json = json::parse(DEFAULT_JSON);
    }

    std::filesystem::path p(tmpl_path);
    if (std::filesystem::is_directory(p)) {
        run_multiple(tmpl_path, stop_on_first_fail, input_json, use_common);
    } else if (std::filesystem::is_regular_file(p)) {
        std::ifstream infile(tmpl_path);
        std::string contents = std::string(
            std::istreambuf_iterator<char>(infile),
            std::istreambuf_iterator<char>());
        run_single(contents, input_json, use_common, output_path);
    } else {
        std::cerr << "Error: PATH_TO_TEMPLATE is not a valid file or directory: " << tmpl_path << "\n";
        return 1;
    }

    return 0;
}

void run_multiple(std::string dir_path, bool stop_on_first_fail, json input, bool use_common) {
    std::vector<std::string> failed_tests;

    // list all files in models/templates/ and run each
    size_t test_count = 0;

    for (const auto & entry : std::filesystem::directory_iterator(dir_path)) {
        // only process .jinja files
        if (entry.path().extension() == ".jinja" && entry.is_regular_file()) {
            test_count++;
            std::cout << "\n\n=== RUNNING TEMPLATE FILE: " << entry.path().string() << " ===\n";
            std::ifstream infile(entry.path());
            std::string contents((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
            try {
                run_single(contents, input, use_common);
            } catch (const std::exception & e) {
                std::cout << "Exception: " << e.what() << "\n";
                std::cout << "=== ERROR WITH TEMPLATE FILE: " << entry.path().string() << " ===\n";
                failed_tests.push_back(entry.path().string());
                if (stop_on_first_fail) {
                    break;
                }
            }
        }
    }

    std::cout << "\n\n=== TEST SUMMARY ===\n";
    std::cout << "Total tests run: " << test_count << "\n";
    std::cout << "Total failed tests: " << failed_tests.size() << "\n";
    for (const auto & test : failed_tests) {
        std::cout << "FAILED TEST: " << test << "\n";
    }
}


static std::string normalize_newlines(const std::string & s) {
#ifdef _WIN32
  static const std::regex nl_regex("\r\n");
  return std::regex_replace(s, nl_regex, "\n");
#else
  return s;
#endif
}


static std::string format_using_common(
            const std::string & template_str,
            const std::string & bos_token,
            const std::string & eos_token,
            std::vector<common_chat_msg> & messages,
            std::vector<common_chat_tool> tools = {}) {
    auto tmpls = common_chat_templates_init(/* model= */ nullptr, template_str, bos_token, eos_token);
    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.messages = messages;
    inputs.tools = tools;
    inputs.add_generation_prompt = true;
    auto output = common_chat_templates_apply(tmpls.get(), inputs).prompt;
    output = normalize_newlines(output);
    return output;
}


// skip libcommon, use direct jinja engine
static jinja::value_string format_using_direct_engine(
            const std::string & template_str,
            json & input) {
    // lexing
    jinja::lexer lexer;
    auto lexer_res = lexer.tokenize(template_str);

    // compile to AST
    jinja::program ast = jinja::parse_from_tokens(lexer_res);

    // check caps for workarounds
    jinja::caps_get(ast);

    std::cout << "\n=== RUN ===\n";
    jinja::context ctx(template_str);

    jinja::global_from_json(ctx, input, true);

    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(ast);
    auto parts = runtime.gather_string_parts(results);

    std::cout << "\n=== RESULTS ===\n";
    for (const auto & part : parts->as_string().parts) {
        std::cout << (part.is_input ? "DATA" : "TMPL") << ": " << part.val << "\n";
    }

    return parts;
}


void run_single(std::string contents, json input, bool use_common, const std::string & output_path) {
    jinja::enable_debug(true);

    jinja::value_string output_parts;

    if (use_common) {
        std::string bos_token = "<s>";
        std::string eos_token = "</s>";
        if (input.contains("bos_token")) {
            bos_token = input["bos_token"].get<std::string>();
        }
        if (input.contains("eos_token")) {
            eos_token = input["eos_token"].get<std::string>();
        }
        nlohmann::ordered_json msgs_json = input["messages"];
        nlohmann::ordered_json tools_json = input["tools"];
        auto messages = common_chat_msgs_parse_oaicompat(msgs_json);
        auto tools = common_chat_tools_parse_oaicompat(tools_json);
        auto output = format_using_common(contents, bos_token, eos_token, messages, tools);
        std::cout << "\n=== OUTPUT ===\n";
        std::cout << output << "\n";
        output_parts = jinja::mk_val<jinja::value_string>(output);

    } else {
        output_parts = format_using_direct_engine(contents, input);
        std::cout << "\n=== OUTPUT ===\n";
        std::cout << output_parts->as_string().str() << "\n";
    }

    if (!output_path.empty()) {
        std::ofstream outfile(output_path);
        if (!outfile) {
            throw std::runtime_error("Could not open output file: " + output_path);
        }
        outfile << output_parts->as_string().str();
        outfile.close();
        std::cout << "\n=== OUTPUT WRITTEN TO " << output_path << " ===\n";
    }
}





//
// Automated tests for chat templates
//

#define U8C(x) (const char*)(u8##x)

static common_chat_msg simple_msg(const std::string & role, const std::string & content) {
    common_chat_msg msg;
    msg.role = role;
    msg.content = content;
    return msg;
}

int main_automated_tests(void) {
    // jinja::enable_debug(true);

    std::vector<llama_chat_message> conversation {
        {"system", "You are a helpful assistant"},
        {"user", "Hello"},
        {"assistant", "Hi there"},
        {"user", "Who are you"},
        {"assistant", "   I am an assistant   "},
        {"user", "Another question"},
    };

    // std::string wrong = /* .template_str= */ u8"[gMASK]<sop>{% for item in messages %}{% if item['tools'] is defined %}<|system|>\n你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n# 可用工具{% set tools = item['tools'] %}{% for tool in tools %}{% if tool['type'] == 'function' %}\n\n## {{ tool['function']['name'] }}\n\n{{ tool['function'] | tojson(indent=4) }}\n......{% endif %}{% endfor %}{% endif %}{% if item['content'] %}<|{{ item['role'] }}|>{{ item['metadata'] }}\n{{ item['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}";
    struct TestCase {
        std::string name;
        std::string template_str;
        std::string expected_output;
        std::string expected_output_jinja;
        std::string bos_token = "";
        std::string eos_token = "";
        bool supported_with_jinja = true;
    };
    std::vector<TestCase> test_cases {
        {
            /* .name= */ "teknium/OpenHermes-2.5-Mistral-7B",
            /* .template_str= */ "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
            /* .expected_output= */ "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nWho are you<|im_end|>\n<|im_start|>assistant\n   I am an assistant   <|im_end|>\n<|im_start|>user\nAnother question<|im_end|>\n<|im_start|>assistant\n",
            /* .expected_output_jinja= */ "",
            /* .bos_token= */ "",
            /* .eos_token= */ "",
        },
        {
            /* .name= */ "mistralai/Mistral-7B-Instruct-v0.2 (NOTE: Old pre-v1 without a system prompt)",
            /* .template_str= */ "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
            /* .expected_output= */ "[INST] You are a helpful assistant\nHello [/INST]Hi there</s>[INST] Who are you [/INST]   I am an assistant   </s>[INST] Another question [/INST]",
            /* .expected_output_jinja= */ "",
            /* .bos_token= */ "",
            /* .eos_token= */ "</s>",
        },
        {
            /* .name= */ "TheBloke/FusionNet_34Bx2_MoE-AWQ",
            /* .template_str= */ "{%- for idx in range(0, messages|length) -%}\n{%- if messages[idx]['role'] == 'user' -%}\n{%- if idx > 1 -%}\n{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}\n{%- else -%}\n{{- messages[idx]['content'] + ' [/INST]' -}}\n{%- endif -%}\n{% elif messages[idx]['role'] == 'system' %}\n{{- '[INST] <<SYS>>\\n' + messages[idx]['content'] + '\\n<</SYS>>\\n\\n' -}}\n{%- elif messages[idx]['role'] == 'assistant' -%}\n{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}\n{% endif %}\n{% endfor %}",
            /* .expected_output= */       "[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nHello [/INST]Hi there</s><s>[INST] Who are you [/INST]   I am an assistant   </s><s>[INST] Another question [/INST]",
            /* .expected_output_jinja= */ "[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nHello [/INST] Hi there </s><s>[INST] Who are you [/INST]    I am an assistant    </s><s>[INST] Another question [/INST]",
            /* .bos_token= */ "<s>",
            /* .eos_token= */ "</s>",
        },
        {
            /* .name= */ "bofenghuang/vigogne-2-70b-chat",
            /* .template_str= */ "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif true == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'Vous êtes Vigogne, un assistant IA créé par Zaion Lab. Vous suivez extrêmement bien les instructions. Aidez autant que vous le pouvez.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}",
            /* .expected_output= */       "[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nHello [/INST]Hi there</s>[INST] Who are you [/INST]I am an assistant</s>[INST] Another question [/INST]",
            /* .expected_output_jinja= */ "[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nHello [/INST] Hi there </s>[INST] Who are you [/INST] I am an assistant </s>[INST] Another question [/INST]",
            /* .bos_token= */ "",
            /* .eos_token= */ "</s>",
        },
        {
            /* .name= */ "mlabonne/AlphaMonarch-7B",
            /* .template_str= */ "{% for message in messages %}{{bos_token + message['role'] + '\\n' + message['content'] + eos_token + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ bos_token + 'assistant\\n' }}{% endif %}",
            /* .expected_output= */ "system\nYou are a helpful assistant</s>\n<s>user\nHello</s>\n<s>assistant\nHi there</s>\n<s>user\nWho are you</s>\n<s>assistant\n   I am an assistant   </s>\n<s>user\nAnother question</s>\n<s>assistant\n",
            /* .expected_output_jinja= */ "<s>system\nYou are a helpful assistant</s>\n<s>user\nHello</s>\n<s>assistant\nHi there</s>\n<s>user\nWho are you</s>\n<s>assistant\n   I am an assistant   </s>\n<s>user\nAnother question</s>\n<s>assistant\n",
            /* .bos_token= */ "<s>",
            /* .eos_token= */ "</s>",
        },
        {
            /* .name= */ "google/gemma-7b-it",
            /* .template_str= */ "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}",
            /* .expected_output= */       "<start_of_turn>user\nYou are a helpful assistant\n\nHello<end_of_turn>\n<start_of_turn>model\nHi there<end_of_turn>\n<start_of_turn>user\nWho are you<end_of_turn>\n<start_of_turn>model\nI am an assistant<end_of_turn>\n<start_of_turn>user\nAnother question<end_of_turn>\n<start_of_turn>model\n",
            /* .expected_output_jinja= */ "<start_of_turn>user\nYou are a helpful assistant\nHello<end_of_turn>\n<start_of_turn>model\nHi there<end_of_turn>\n<start_of_turn>user\nWho are you<end_of_turn>\n<start_of_turn>model\nI am an assistant<end_of_turn>\n<start_of_turn>user\nAnother question<end_of_turn>\n<start_of_turn>model\n",
        },
        {
            /* .name= */ "OrionStarAI/Orion-14B-Chat",
            /* .template_str= */ "{% for message in messages %}{% if loop.first %}{{ bos_token }}{% endif %}{% if message['role'] == 'user' %}{{ 'Human: ' + message['content'] + '\\n\\nAssistant: ' + eos_token }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}",
            /* .expected_output= */       "Human: You are a helpful assistant\n\nHello\n\nAssistant: </s>Hi there</s>Human: Who are you\n\nAssistant: </s>   I am an assistant   </s>Human: Another question\n\nAssistant: </s>",
            /* .expected_output_jinja= */ "Human: You are a helpful assistant\nHello\n\nAssistant: </s>Hi there</s>Human: Who are you\n\nAssistant: </s>   I am an assistant   </s>Human: Another question\n\nAssistant: </s>",
            /* .bos_token= */ "",
            /* .eos_token= */ "</s>",
        },
        {
            /* .name= */ "openchat/openchat-3.5-0106",
            // The included chat_template differs from the author's suggestions here: https://huggingface.co/openchat/openchat_3.5/discussions/5#65448109b4a3f3a2f486fd9d
            // So we match against the included template but implement the suggested version.
            /* .template_str= */ "{{ bos_token }}{% for message in messages %}{{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>'}}{% endfor %}{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}",
            /* .expected_output= */                            "You are a helpful assistant<|end_of_turn|>GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant: Hi there<|end_of_turn|>GPT4 Correct User: Who are you<|end_of_turn|>GPT4 Correct Assistant:    I am an assistant   <|end_of_turn|>GPT4 Correct User: Another question<|end_of_turn|>GPT4 Correct Assistant:",
            /* .expected_output_jinja= */ "GPT4 Correct System: You are a helpful assistant<|end_of_turn|>GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant: Hi there<|end_of_turn|>GPT4 Correct User: Who are you<|end_of_turn|>GPT4 Correct Assistant:    I am an assistant   <|end_of_turn|>GPT4 Correct User: Another question<|end_of_turn|>GPT4 Correct Assistant:",
        },
        {
            /* .name= */ "deepseek-ai/deepseek-coder-33b-instruct",
            /* .template_str= */ "{% if not add_generation_prompt is defined %}\n{% set add_generation_prompt = false %}\n{% endif %}\n{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{{bos_token}}{%- if not ns.found -%}\n{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{% if add_generation_prompt %}\n{{'### Response:'}}\n{% endif %}",
            /* .expected_output= */ "You are a helpful assistant### Instruction:\nHello\n### Response:\nHi there\n<|EOT|>\n### Instruction:\nWho are you\n### Response:\n   I am an assistant   \n<|EOT|>\n### Instruction:\nAnother question\n### Response:\n",
            /* .expected_output_jinja= */ "",
        },
        {
            /* .name= */ "eachadea/vicuna-13b-1.1",
            // No template included in tokenizer_config.json, so this template likely needs to be manually set.
            /* .template_str= */ "{%- for message in messages %}{%- if message['role'] == 'system' -%}{{- '' + message['content'] + '\n\n' -}}{%- else -%}{%- if message['role'] == 'user' -%}{{-'USER: ' + message['content'] + '\n'-}}{%- else -%}{{-'ASSISTANT: ' + message['content'] + '</s>\n' -}}{%- endif -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{-'ASSISTANT:'-}}{%- endif -%}",
            /* .expected_output= */ "You are a helpful assistant\n\nUSER: Hello\nASSISTANT: Hi there</s>\nUSER: Who are you\nASSISTANT:    I am an assistant   </s>\nUSER: Another question\nASSISTANT:",
            /* .expected_output_jinja= */ "",
            /* .bos_token= */ "",
            /* .eos_token= */ "",
        },
        {
            /* .name= */ "Orca-Vicuna",
            // No template included in tokenizer_config.json, so this template likely needs to be manually set.
            /* .template_str= */ "{%- for message in messages %}{%- if message['role'] == 'system' -%}{{-'SYSTEM: ' + message['content'] + '\n' -}}{%- else -%}{%- if message['role'] == 'user' -%}{{-'USER: ' + message['content'] + '\n'-}}{%- else -%}{{-'ASSISTANT: ' + message['content'] + '</s>\n' -}}{%- endif -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{-'ASSISTANT:'-}}{%- endif -%}",
            /* .expected_output= */ "SYSTEM: You are a helpful assistant\nUSER: Hello\nASSISTANT: Hi there</s>\nUSER: Who are you\nASSISTANT:    I am an assistant   </s>\nUSER: Another question\nASSISTANT:",
            /* .expected_output_jinja= */ "",
            /* .bos_token= */ "",
            /* .eos_token= */ "",
        },
        {
            /* .name= */ "CohereForAI/c4ai-command-r-plus",
            /* .template_str= */ "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true %}{% set loop_messages = messages %}{% set system_message = 'You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message != false %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}",
            /* .expected_output= */ "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are a helpful assistant<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Hi there<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Who are you<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>I am an assistant<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Another question<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
            /* .expected_output_jinja= */ "",
        },
        {
            /* .name= */ "Llama-3",
            /* .template_str= */ "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}",
            /* .expected_output= */ "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHi there<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWho are you<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI am an assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nAnother question<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            /* .expected_output_jinja= */ "",
        },
        {
            /* .name= */ "Phi-3-mini",
            /* .template_str= */ "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}",
            /* .expected_output= */     "<|system|>\nYou are a helpful assistant<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\nHi there<|end|>\n<|user|>\nWho are you<|end|>\n<|assistant|>\n   I am an assistant   <|end|>\n<|user|>\nAnother question<|end|>\n<|assistant|>\n",
            /* .expected_output_jinja= */ "<|user|>\nYou are a helpful assistant\nHello<|end|>\n<|assistant|>\nHi there<|end|>\n<|user|>\nWho are you<|end|>\n<|assistant|>\n   I am an assistant   <|end|>\n<|user|>\nAnother question<|end|>\n<|assistant|>\n",
        },
        {
            /* .name= */ "Phi-3-small",
            /* .template_str= */ "{{ bos_token }}{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}",
            /* .expected_output= */ "<|system|>\nYou are a helpful assistant<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\nHi there<|end|>\n<|user|>\nWho are you<|end|>\n<|assistant|>\n   I am an assistant   <|end|>\n<|user|>\nAnother question<|end|>\n<|assistant|>\n",
            /* .expected_output_jinja= */ "",
        },
        {
            /* .name= */ "Phi-3-medium",
            /* .template_str= */ "{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}",
            /* .expected_output= */     "<|system|>\nYou are a helpful assistant<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\nHi there<|end|>\n<|user|>\nWho are you<|end|>\n<|assistant|>\n   I am an assistant   <|end|>\n<|user|>\nAnother question<|end|>\n<|assistant|>\n",
            /* .expected_output_jinja= */ "<|user|>\nYou are a helpful assistant\nHello<|end|>\n<|assistant|>\nHi there<|end|>\n<|user|>\nWho are you<|end|>\n<|assistant|>\n   I am an assistant   <|end|>\n<|user|>\nAnother question<|end|>\n<|assistant|>\n",
        },
        {
            /* .name= */ "Phi-3-vision",
            /* .template_str= */ "{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>\n' }}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{- '<|assistant|>\n' -}}{% endif %}",
            /* .expected_output= */ "<|system|>\nYou are a helpful assistant<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\nHi there<|end|>\n<|user|>\nWho are you<|end|>\n<|assistant|>\n   I am an assistant   <|end|>\n<|user|>\nAnother question<|end|>\n<|assistant|>\n",
            /* .expected_output_jinja= */ "",
            /* .bos_token= */ "",
            /* .eos_token= */ "",
        },
        {
            /* .name= */ "ChatGLM3",
            /* .template_str= */ "{% for message in messages %}{% if loop.first %}[gMASK]sop<|{{ message['role'] }}|>\n {{ message['content'] }}{% else %}<|{{ message['role'] }}|>\n {{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}",
            /* .expected_output= */       "[gMASK]sop<|system|>\n You are a helpful assistant<|user|>\n Hello<|assistant|>\n Hi there<|user|>\n Who are you<|assistant|>\n    I am an assistant   <|user|>\n Another question<|assistant|>",
            /* .expected_output_jinja= */ "[gMASK]sop<|system|>\n You are a helpful assistant<|user|>\n Hello<|assistant|>\n Hi there<|user|>\n Who are you<|assistant|>\n    I am an assistant   <|user|>\n Another question<|assistant|>",
        },
        {
            /* .name= */ "ChatGLM4",
            /* .template_str= */ U8C("[gMASK]<sop>{% for item in messages %}{% if item['tools'] is defined %}<|system|>\n你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n# 可用工具{% set tools = item['tools'] %}{% for tool in tools %}{% if tool['type'] == 'function' %}\n\n## {{ tool['function']['name'] }}\n\n{{ tool['function'] | tojson(indent=4) }}\n......{% endif %}{% endfor %}{% endif %}{% if item['content'] %}<|{{ item['role'] }}|>{{ item['metadata'] }}\n{{ item['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}"),
            /* .expected_output= */ "[gMASK]<sop><|system|>\nYou are a helpful assistant<|user|>\nHello<|assistant|>\nHi there<|user|>\nWho are you<|assistant|>\n   I am an assistant   <|user|>\nAnother question<|assistant|>\n",
            /* .expected_output_jinja= */ "",
            /* .bos_token= */ "",
            /* .eos_token= */ "",
        },
        {
            /* .name= */ "GLMEdge",
            /* .template_str= */ "{% for item in messages %}{% if item['role'] == 'system' %}<|system|>\n{{ item['content'] }}{% elif item['role'] == 'user' %}<|user|>\n{{ item['content'] }}{% elif item['role'] == 'assistant' %}<|assistant|>\n{{ item['content'] }}{% endif %}{% endfor %}<|assistant|>",
            /* .expected_output= */ "<|system|>\nYou are a helpful assistant<|user|>\nHello<|assistant|>\nHi there<|user|>\nWho are you<|assistant|>\n   I am an assistant   <|user|>\nAnother question<|assistant|>",
            /* .expected_output_jinja= */ "<|system|>\nYou are a helpful assistant<|user|>\nHello<|assistant|>\nHi there<|user|>\nWho are you<|assistant|>\n   I am an assistant   <|user|>\nAnother question<|assistant|>",
            /* .bos_token= */ "",
            /* .eos_token= */ "",
        },
        {
            /* .name= */ "MiniCPM-3B-OpenHermes-2.5-v2-GGUF",
            /* .template_str= */ U8C("{% for message in messages %}{% if message['role'] == 'user' %}{{'<用户>' + message['content'].strip() + '<AI>'}}{% else %}{{message['content'].strip()}}{% endif %}{% endfor %}"),
            /* .expected_output= */ U8C("You are a helpful assistant<用户>Hello<AI>Hi there<用户>Who are you<AI>I am an assistant<用户>Another question<AI>"),
            /* .expected_output_jinja= */ "",
            /* .bos_token= */ "",
            /* .eos_token= */ "",
        },
        {
            /* .name= */ "DeepSeek-V2",
            /* .template_str= */ "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}",
            /* .expected_output= */ U8C("You are a helpful assistant\n\nUser: Hello\n\nAssistant: Hi there<｜end▁of▁sentence｜>User: Who are you\n\nAssistant:    I am an assistant   <｜end▁of▁sentence｜>User: Another question\n\nAssistant:"),
            /* .expected_output_jinja= */ "",
            /* .bos_token= */ "",
            /* .eos_token= */ "<｜end▁of▁sentence｜>",
        },
        {
            /* .name= */ "ibm-granite/granite-3.0-8b-instruct",
            /* .template_str= */ "{%- if tools %}\n    {{- '<|start_of_role|>available_tools<|end_of_role|>\n' }}\n    {%- for tool in tools %}\n    {{- tool | tojson(indent=4) }}\n    {%- if not loop.last %}\n        {{- '\n\n' }}\n    {%- endif %}\n    {%- endfor %}\n    {{- '<|end_of_text|>\n' }}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n    {{- '<|start_of_role|>system<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}\n    {%- elif message['role'] == 'user' %}\n    {{- '<|start_of_role|>user<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}\n    {%- elif message['role'] == 'assistant' %}\n    {{- '<|start_of_role|>assistant<|end_of_role|>'  + message['content'] + '<|end_of_text|>\n' }}\n    {%- elif message['role'] == 'assistant_tool_call' %}\n    {{- '<|start_of_role|>assistant<|end_of_role|><|tool_call|>' + message['content'] + '<|end_of_text|>\n' }}\n    {%- elif message['role'] == 'tool_response' %}\n    {{- '<|start_of_role|>tool_response<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}\n    {%- endif %}\n    {%- if loop.last and add_generation_prompt %}\n    {{- '<|start_of_role|>assistant<|end_of_role|>' }}\n    {%- endif %}\n{%- endfor %}",
            /* .expected_output= */       "<|start_of_role|>system<|end_of_role|>You are a helpful assistant<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>Hi there<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Who are you<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>   I am an assistant   <|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Another question<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
            /* .expected_output_jinja= */ "<|start_of_role|>system<|end_of_role|>You are a helpful assistant<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>Hi there<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Who are you<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>   I am an assistant   <|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Another question<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
        },
        {
            /* .name= */ "mistralai/Mistral-7B-Instruct-v0.2 (mistralai 'v1' template with a system prompt)",
            /* .template_str= */ "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.first and system_message is defined %}\n            {{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}\n        {%- else %}\n            {{- ' [INST] ' + message['content'] + ' [/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n",
            /* .expected_output= */ " [INST] You are a helpful assistant\n\nHello [/INST] Hi there</s> [INST] Who are you [/INST]    I am an assistant   </s> [INST] Another question [/INST]",
            /* .expected_output_jinja= */ " [INST] You are a helpful assistant\n\nHello [/INST] Hi there</s> [INST] Who are you [/INST]    I am an assistant   </s> [INST] Another question [/INST]",
            /* .bos_token= */ "",
            /* .eos_token= */ "</s>",
        },
        {
            /* .name= */ "Mistral-Large-Instruct-2407 (mistralai 'v3' template; modified to have system prompt at start)",
            /* .template_str= */ "{%- if messages[0][\"role\"] == \"system\" %}\n    {%- set system_message = messages[0][\"content\"] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n{%- set user_messages = loop_messages | selectattr(\"role\", \"equalto\", \"user\") | list %}\n\n{#- This block checks for alternating user/assistant messages, skipping tool calling messages #}\n{%- set ns = namespace() %}\n{%- set ns.index = 0 %}\n{%- for message in loop_messages %}\n    {%- if not (message.role == \"tool\" or message.role == \"tool_results\" or (message.tool_calls is defined and message.tool_calls is not none)) %}\n        {%- if (message[\"role\"] == \"user\") != (ns.index % 2 == 0) %}\n            {{- raise_exception(\"After the optional system message, conversation roles must alternate user/assistant/user/assistant/...\") }}\n        {%- endif %}\n        {%- set ns.index = ns.index + 1 %}\n    {%- endif %}\n{%- endfor %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if message[\"role\"] == \"user\" %}\n        {%- if tools is not none and (message == user_messages[-1]) %}\n            {{- \"[AVAILABLE_TOOLS] [\" }}\n            {%- for tool in tools %}\n                {%- set tool = tool.function %}\n                {{- '{\"type\": \"function\", \"function\": {' }}\n                {%- for key, val in tool.items() if key != \"return\" %}\n                    {%- if val is string %}\n                        {{- '\"' + key + '\": \"' + val + '\"' }}\n                    {%- else %}\n                        {{- '\"' + key + '\": ' + val|tojson }}\n                    {%- endif %}\n                    {%- if not loop.last %}\n                        {{- \", \" }}\n                    {%- endif %}\n                {%- endfor %}\n                {{- \"}}\" }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- else %}\n                    {{- \"]\" }}\n                {%- endif %}\n            {%- endfor %}\n            {{- \"[/AVAILABLE_TOOLS]\" }}\n            {%- endif %}\n        {%- if loop.last and system_message is defined %}\n            {{- \"[INST] \" + system_message + \"\\n\\n\" + message[\"content\"] + \"[/INST]\" }}\n        {%- else %}\n            {{- \"[INST] \" + message[\"content\"] + \"[/INST]\" }}\n        {%- endif %}\n    {%- elif message.tool_calls is defined and message.tool_calls is not none %}\n        {{- \"[TOOL_CALLS] [\" }}\n        {%- for tool_call in message.tool_calls %}\n            {%- set out = tool_call.function|tojson %}\n            {{- out[:-1] }}\n            {%- if not tool_call.id is defined or tool_call.id|length != 9 %}\n                {{- raise_exception(\"Tool call IDs should be alphanumeric strings with length 9!\") }}\n            {%- endif %}\n            {{- ', \"id\": \"' + tool_call.id + '\"}' }}\n            {%- if not loop.last %}\n                {{- \", \" }}\n            {%- else %}\n                {{- \"]\" + eos_token }}\n            {%- endif %}\n        {%- endfor %}\n    {%- elif message[\"role\"] == \"assistant\" %}\n        {{- \" \" + message[\"content\"]|trim + eos_token}}\n    {%- elif message[\"role\"] == \"tool_results\" or message[\"role\"] == \"tool\" %}\n        {%- if message.content is defined and message.content.content is defined %}\n            {%- set content = message.content.content %}\n        {%- else %}\n            {%- set content = message.content %}\n        {%- endif %}\n        {{- '[TOOL_RESULTS] {\"content\": ' + content|string + \", \" }}\n        {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}\n            {{- raise_exception(\"Tool call IDs should be alphanumeric strings with length 9!\") }}\n        {%- endif %}\n        {{- '\"call_id\": \"' + message.tool_call_id + '\"}[/TOOL_RESULTS]' }}\n    {%- else %}\n        {{- raise_exception(\"Only user and assistant roles are supported, with the exception of an initial optional system message!\") }}\n    {%- endif %}\n{%- endfor %}\n",
            /* .expected_output= */       "[INST] You are a helpful assistant\n\nHello[/INST] Hi there</s>[INST] Who are you[/INST] I am an assistant</s>[INST] Another question[/INST]",
            /* .expected_output_jinja= */ "[INST] Hello[/INST] Hi there</s>[INST] Who are you[/INST] I am an assistant</s>[INST] You are a helpful assistant\n\nAnother question[/INST]",
            /* .bos_token= */ "",
            /* .eos_token= */ "</s>",
        },
        {
            /* .name= */ "Mistral-Nemo-Instruct-2407 (mistralai 'v3-tekken' template; modified to have system prompt at start)",
            /* .template_str= */ "{%- if messages[0][\"role\"] == \"system\" %}\n    {%- set system_message = messages[0][\"content\"] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n{%- set user_messages = loop_messages | selectattr(\"role\", \"equalto\", \"user\") | list %}\n\n{#- This block checks for alternating user/assistant messages, skipping tool calling messages #}\n{%- set ns = namespace() %}\n{%- set ns.index = 0 %}\n{%- for message in loop_messages %}\n    {%- if not (message.role == \"tool\" or message.role == \"tool_results\" or (message.tool_calls is defined and message.tool_calls is not none)) %}\n        {%- if (message[\"role\"] == \"user\") != (ns.index % 2 == 0) %}\n            {{- raise_exception(\"After the optional system message, conversation roles must alternate user/assistant/user/assistant/...\") }}\n        {%- endif %}\n        {%- set ns.index = ns.index + 1 %}\n    {%- endif %}\n{%- endfor %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if message[\"role\"] == \"user\" %}\n        {%- if tools is not none and (message == user_messages[-1]) %}\n            {{- \"[AVAILABLE_TOOLS][\" }}\n            {%- for tool in tools %}\n                {%- set tool = tool.function %}\n                {{- '{\"type\": \"function\", \"function\": {' }}\n                {%- for key, val in tool.items() if key != \"return\" %}\n                    {%- if val is string %}\n                        {{- '\"' + key + '\": \"' + val + '\"' }}\n                    {%- else %}\n                        {{- '\"' + key + '\": ' + val|tojson }}\n                    {%- endif %}\n                    {%- if not loop.last %}\n                        {{- \", \" }}\n                    {%- endif %}\n                {%- endfor %}\n                {{- \"}}\" }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- else %}\n                    {{- \"]\" }}\n                {%- endif %}\n            {%- endfor %}\n            {{- \"[/AVAILABLE_TOOLS]\" }}\n            {%- endif %}\n        {%- if loop.last and system_message is defined %}\n            {{- \"[INST]\" + system_message + \"\\n\\n\" + message[\"content\"] + \"[/INST]\" }}\n        {%- else %}\n            {{- \"[INST]\" + message[\"content\"] + \"[/INST]\" }}\n        {%- endif %}\n    {%- elif (message.tool_calls is defined and message.tool_calls is not none) %}\n        {{- \"[TOOL_CALLS][\" }}\n        {%- for tool_call in message.tool_calls %}\n            {%- set out = tool_call.function|tojson %}\n            {{- out[:-1] }}\n            {%- if not tool_call.id is defined or tool_call.id|length != 9 %}\n                {{- raise_exception(\"Tool call IDs should be alphanumeric strings with length 9!\") }}\n            {%- endif %}\n            {{- ', \"id\": \"' + tool_call.id + '\"}' }}\n            {%- if not loop.last %}\n                {{- \", \" }}\n            {%- else %}\n                {{- \"]\" + eos_token }}\n            {%- endif %}\n        {%- endfor %}\n    {%- elif message[\"role\"] == \"assistant\" %}\n        {{- message[\"content\"] + eos_token}}\n    {%- elif message[\"role\"] == \"tool_results\" or message[\"role\"] == \"tool\" %}\n        {%- if message.content is defined and message.content.content is defined %}\n            {%- set content = message.content.content %}\n        {%- else %}\n            {%- set content = message.content %}\n        {%- endif %}\n        {{- '[TOOL_RESULTS]{\"content\": ' + content|string + \", \" }}\n        {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}\n            {{- raise_exception(\"Tool call IDs should be alphanumeric strings with length 9!\") }}\n        {%- endif %}\n        {{- '\"call_id\": \"' + message.tool_call_id + '\"}[/TOOL_RESULTS]' }}\n    {%- else %}\n        {{- raise_exception(\"Only user and assistant roles are supported, with the exception of an initial optional system message!\") }}\n    {%- endif %}\n{%- endfor %}\n",
            /* .expected_output= */       "[INST]You are a helpful assistant\n\nHello[/INST]Hi there</s>[INST]Who are you[/INST]   I am an assistant   </s>[INST]Another question[/INST]",
            /* .expected_output_jinja= */ "[INST]Hello[/INST]Hi there</s>[INST]Who are you[/INST]   I am an assistant   </s>[INST]You are a helpful assistant\n\nAnother question[/INST]",
            /* .bos_token= */ "",
            /* .eos_token= */ "</s>",
        },
        {
            /* .name= */ "mistralai/Mistral-Large-Instruct-2411 (mistralai 'v7' template)",
            /* .template_str= */ "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + '[/INST]' }}{% elif message['role'] == 'system' %}{{ '[SYSTEM_PROMPT] ' + message['content'] + '[/SYSTEM_PROMPT]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token }}{% else %}{{ raise_exception('Only user, system and assistant roles are supported!') }}{% endif %}{% endfor %}",
            /* .expected_output= */ "[SYSTEM_PROMPT] You are a helpful assistant[/SYSTEM_PROMPT][INST] Hello[/INST] Hi there</s>[INST] Who are you[/INST]    I am an assistant   </s>[INST] Another question[/INST]",
            /* .expected_output_jinja= */ "",
            /* .bos_token= */ "",
            /* .eos_token= */ "</s>",
        },
        {
            /* .name= */ "ai-sage/GigaChat-20B-A3B-instruct",
            /* .template_str= */ "{% if messages[0]['role'] == 'system' -%}\n    {%- set loop_messages = messages[1:] -%}\n    {%- set system_message = bos_token + messages[0]['content'] + additional_special_tokens[1] -%}\n{%- else -%}\n    {%- set loop_messages = messages -%}\n    {%- set system_message = bos_token + '' -%}\n{%- endif -%}\n{%- for message in loop_messages %}\n    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}\n    {% endif %}\n    \n    {%- if loop.index0 == 0 -%}\n        {{ system_message -}}\n    {%- endif -%}\n    {%- if message['role'] == 'user' -%}\n        {{ message['role'] + additional_special_tokens[0] + message['content'] + additional_special_tokens[1] -}}\n        {{ 'available functions' + additional_special_tokens[0] + additional_special_tokens[2] + additional_special_tokens[3]  + additional_special_tokens[1] -}}\n    {%- endif -%}\n    {%- if message['role'] == 'assistant' -%}\n        {{ message['role'] + additional_special_tokens[0] + message['content'] + additional_special_tokens[1] -}}\n    {%- endif -%}\n    {%- if loop.last and add_generation_prompt -%}\n        {{ 'assistant' + additional_special_tokens[0] -}}\n    {%- endif -%}\n{%- endfor %}",
            /* .expected_output= */ "<s>You are a helpful assistant<|message_sep|>user<|role_sep|>Hello<|message_sep|>available functions<|role_sep|>[]<|message_sep|>assistant<|role_sep|>Hi there<|message_sep|>user<|role_sep|>Who are you<|message_sep|>available functions<|role_sep|>[]<|message_sep|>assistant<|role_sep|>   I am an assistant   <|message_sep|>user<|role_sep|>Another question<|message_sep|>available functions<|role_sep|>[]<|message_sep|>assistant<|role_sep|>",
            /* .expected_output_jinja= */ "",
            /* .bos_token= */ "",
            /* .eos_token= */ "",
            /* .supported_with_jinja= */ false, // Requires additional_special_tokens as extra context
        },
        {
            /* .name= */ "Infinigence/Megrez-3B-Instruct",
            /* .template_str= */ U8C("{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|role_start|>system<|role_end|>你是Megrez-3B-Instruct，将针对用户的问题给出详细的、积极的回答。<|turn_end|>' }}{% endif %}{{ '<|role_start|>' + message['role'] + '<|role_end|>' + message['content'] + '<|turn_end|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|role_start|>assistant<|role_end|>' }}{% endif %}"),
            /* .expected_output= */ "<|role_start|>system<|role_end|>You are a helpful assistant<|turn_end|><|role_start|>user<|role_end|>Hello<|turn_end|><|role_start|>assistant<|role_end|>Hi there<|turn_end|><|role_start|>user<|role_end|>Who are you<|turn_end|><|role_start|>assistant<|role_end|>   I am an assistant   <|turn_end|><|role_start|>user<|role_end|>Another question<|turn_end|><|role_start|>assistant<|role_end|>",
            /* .expected_output_jinja= */ "",
            /* .bos_token= */ "",
            /* .eos_token= */ "",
        },
        {
            /* .name= */ "phi-4",
            /* .template_str= */ "{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|><|im_start|>assistant<|im_sep|>'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}",
            /* .expected_output= */ "<|im_start|>system<|im_sep|>You are a helpful assistant<|im_end|><|im_start|>user<|im_sep|>Hello<|im_end|><|im_start|>assistant<|im_sep|>Hi there<|im_end|><|im_start|>user<|im_sep|>Who are you<|im_end|><|im_start|>assistant<|im_sep|>   I am an assistant   <|im_end|><|im_start|>user<|im_sep|>Another question<|im_end|><|im_start|>assistant<|im_sep|>",
            /* .expected_output_jinja= */ "",
            /* .bos_token= */ "",
            /* .eos_token= */ "",
        },
        {
            /* .name= */ "yandex/YandexGPT-5-Lite-8B-instruct",
            /* .template_str= */ "<s>{%- set names = {'assistant': ' Ассистент:', 'user': ' Пользователь:'} %}\n{%- set tools_prefix = 'Тебе доступны следующие функции:' %}\n{%- macro __render_tool(tool) %}\n    {%- set name = tool.function.name %}\n    {%- set description = tool.function.description|default('') %}\n    {%- set parameters = tool.function.parameters|tojson %}\n    {{- '\\n' }}function {{ '{' }}'name':'{{ name }}',\n    {%- if tool.function.description %}'description':'{{ description }}',{% endif %}\n'parameters':{{ parameters }}\n    {{- '}' }}\n{%- endmacro %}\n{%- macro __render_tools(tools) %}\n    {{- tools_prefix }}\n    {%- for tool in tools %}\n        {{- __render_tool(tool) }}\n    {%- endfor %}\n    {{- '\\n\\n' }}\n{%- endmacro %}\n{%- macro __render_tool_message(message) %}\n    {{- '\\n\\nРезультат вызова' }} {{ message.name }}: {{ message.content }} {{ '\\n\\n' }}\n{%- endmacro %}\n{%- if tools -%}\n    {{- __render_tools(tools) }}\n{%- endif -%}\n{%- macro __render_user_message(message) %}\n{{ names.user }} {{ message.content + '\\n\\n' }}\n{%- endmacro %}\n{%- macro __render_assistant_message(message) %}\n    {{- names.assistant }}\n    {%- set call = message['function_call'] %}\n    {%- if call %}\n        {{- '\\n[TOOL_CALL_START]' }}{{ call.name }}{{ '\\n' }}{{ call.arguments|tojson }}\n    {%- else %}\n        {{- ' ' + message.content + '\\n\\n' }}\n    {%- endif %}\n{%- endmacro %}\n{%- if not add_generation_prompt is defined %}\n{%- set add_generation_prompt = false %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'user' %}\n        {{- __render_user_message(message) }}\n    {%- endif %}\n    {%- if message.role == 'assistant' and not loop.last %}\n        {{- __render_assistant_message(message) }}\n    {%- endif %}\n    {%- if message.role == 'tool' %}\n        {{- __render_tool_message(message) }}\n    {%- endif %}\n    {%- if loop.last %}\n        {{- ' Ассистент:[SEP]' }}\n    {%- endif %}\n{%- endfor %}\n",
            /* .expected_output= */ " Пользователь: Hello\n\n Ассистент: Hi there\n\n Пользователь: Who are you\n\n Ассистент:    I am an assistant   \n\n Пользователь: Another question\n\n Ассистент:[SEP]",
            /* .expected_output_jinja= */ "<s> Пользователь: You are a helpful assistant\nHello\n\n Ассистент: Hi there\n\n Пользователь: Who are you\n\n Ассистент:    I am an assistant   \n\n Пользователь: Another question\n\n Ассистент:[SEP]",
            /* .bos_token= */ "<s>",
            /* .eos_token= */ "",
        },
        {
            /* .name= */ "inclusionAI/Ling-lite",
            /* .template_str */ "{% for message in messages %}{% set role = message['role'] | lower %}{% if role == 'user' %}{% set role = 'HUMAN' %}{% endif %}{% set role = role | upper %}{{ '<role>' + role + '</role>' + message['content'] }}{% endfor %}{% if add_generation_prompt %}{{ '<role>ASSISTANT</role>' }}{% endif %}",
            /* .expected_output= */ "<role>SYSTEM</role>You are a helpful assistant<role>HUMAN</role>Hello<role>ASSISTANT</role>Hi there<role>HUMAN</role>Who are you<role>ASSISTANT</role>   I am an assistant   <role>HUMAN</role>Another question<role>ASSISTANT</role>",
            /* .expected_output_jinja= */ "",
            /* .bos_token= */ "",
            /* .eos_token= */ "",
        },
        {
            /* .name= */ "ByteDance-Seed/Seed-OSS-36B-Instruct",
            /* .template_str */ "{# <seed:bos> #}{%- for message in messages %}{%- if message.role in [\"user\", \"system\"] %}{{ bos_token + message.role + \"\\n\" + message.content + eos_token }}{%- elif message.role == \"assistant\" %}{{ bos_token + message.role }}{%- if message.content is defined and message.content is string and message.content|trim|length > 0 %}{{ \"\\n\" + message.content|trim + eos_token }}{%- endif %}{%- else %}{{ bos_token + message.role + \"\\n\" + message.content + eos_token }}{%- endif %}{%- endfor %}{%- if add_generation_prompt %}{{ bos_token + \"assistant\\n\" }}{%- endif %}",
            /* .expected_output= */ "<seed:bos>system\nYou are a helpful assistant<seed:eos><seed:bos>user\nHello<seed:eos><seed:bos>assistant\nHi there<seed:eos><seed:bos>user\nWho are you<seed:eos><seed:bos>assistant\nI am an assistant<seed:eos><seed:bos>user\nAnother question<seed:eos><seed:bos>assistant\n",
            /* .expected_output_jinja= */ "<seed:bos>system\nYou are a helpful assistant<seed:eos><seed:bos>user\nHello<seed:eos><seed:bos>assistant\nHi there<seed:eos><seed:bos>user\nWho are you<seed:eos><seed:bos>assistant\nI am an assistant<seed:eos><seed:bos>user\nAnother question<seed:eos><seed:bos>assistant\n",
            /* .bos_token= */ "<seed:bos>",
            /* .eos_token= */ "<seed:eos>",
        }
    };
    std::vector<char> formatted_chat(1024);
    int32_t res;

    // list all supported templates
    std::vector<const char *> supported_tmpl;
    res = llama_chat_builtin_templates(nullptr, 0);
    assert(res > 0);
    supported_tmpl.resize(res);
    res = llama_chat_builtin_templates(supported_tmpl.data(), supported_tmpl.size());
    std::cout << "Built-in chat templates:\n";
    for (auto tmpl : supported_tmpl) {
        std::cout << "  " << tmpl << "\n";
    }

    // test invalid chat template
    res = llama_chat_apply_template("INVALID TEMPLATE", conversation.data(), conversation.size(), true, formatted_chat.data(), formatted_chat.size());
    assert(res < 0);
    const auto add_generation_prompt = true;

    for (const auto & test_case : test_cases) {
        std::cout << "\n\n=== " << test_case.name << " ===\n\n";
        formatted_chat.resize(1024);
        res = llama_chat_apply_template(
            test_case.template_str.c_str(),
            conversation.data(),
            conversation.size(),
            add_generation_prompt,
            formatted_chat.data(),
            formatted_chat.size()
        );
        formatted_chat.resize(res);
        std::string output(formatted_chat.data(), formatted_chat.size());
        if (output != test_case.expected_output) {
            std::cout << "Expected:\n" << test_case.expected_output << "\n";
            std::cout << "-------------------------\n";
            std::cout << "Actual:\n" << output << "\n";
            std::cout.flush();
            assert(output == test_case.expected_output);
        }
    }

    std::vector<common_chat_msg> messages;
    for (const auto & msg : conversation) {
        messages.push_back(simple_msg(msg.role, msg.content));
    }
    for (const auto & test_case : test_cases) {
        if (!test_case.supported_with_jinja) {
            continue;
        }
        std::cout << "\n\n=== " << test_case.name << " (jinja) ===\n\n";
        try {
            auto output = format_using_common(
                                test_case.template_str,
                                test_case.bos_token,
                                test_case.eos_token,
                                messages);
            auto expected_output = normalize_newlines(test_case.expected_output_jinja.empty() ? test_case.expected_output : test_case.expected_output_jinja);
            if (output != expected_output) {
                std::cout << "Template:```\n" << test_case.template_str << "\n```";
                std::cout << "-------------------------\n";
                std::cout << "Expected:```\n" << expected_output << "\n```";
                std::cout << "-------------------------\n";
                std::cout << "Actual:```\n" << output << "\n```";
                std::cout.flush();
                assert(output == expected_output);
            }
        } catch (const std::exception & e) {
            std::cerr << "ERROR: " << e.what() << "\n";
            assert(false);
        }
    }

    // TODO: llama_chat_format_single will be deprecated, remove these tests later

    // test llama_chat_format_single for system message
    std::cout << "\n\n=== llama_chat_format_single (system message) ===\n\n";
    std::vector<common_chat_msg> chat2;
    auto sys_msg = simple_msg("system", "You are a helpful assistant");

    auto fmt_sys = [&](std::string tmpl_str) {
        auto tmpls = common_chat_templates_init(/* model= */ nullptr, tmpl_str);
        auto output = common_chat_format_single(tmpls.get(), chat2, sys_msg, false, /* use_jinja= */ false);
        std::cout << "fmt_sys(" << tmpl_str << ") : " << output << "\n";
        std::cout << "-------------------------\n";
        return output;
    };
    assert(fmt_sys("chatml") == "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n");
    assert(fmt_sys("mistral-v1") == " [INST] You are a helpful assistant\n\n");
    assert(fmt_sys("mistral-v3") == "[INST] You are a helpful assistant\n\n");
    assert(fmt_sys("mistral-v3-tekken") == "[INST]You are a helpful assistant\n\n");
    assert(fmt_sys("mistral-v7") == "[SYSTEM_PROMPT] You are a helpful assistant[/SYSTEM_PROMPT]");
    assert(fmt_sys("llama2") == "[INST] You are a helpful assistant\n");
    assert(fmt_sys("llama2-sys") == "[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\n");
    assert(fmt_sys("mistral") == "[INST] You are a helpful assistant\n"); // for old pre-v1 templates
    assert(fmt_sys("gemma")  == ""); // for gemma, system message is merged with user message
    assert(fmt_sys("llama3") == "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|>");
    assert(fmt_sys("gigachat") == "<s>You are a helpful assistant<|message_sep|>");


    // test llama_chat_format_single for user message
    std::cout << "\n\n=== llama_chat_format_single (user message) ===\n\n";
    chat2.push_back(simple_msg("system", "You are a helpful assistant"));
    chat2.push_back(simple_msg("user", "Hello"));
    chat2.push_back(simple_msg("assistant", "I am assistant"));
    auto new_msg = simple_msg("user", "How are you");

    auto fmt_single = [&](const std::string & tmpl_str) {
        auto tmpls = common_chat_templates_init(/* model= */ nullptr, tmpl_str.c_str());
        auto output = common_chat_format_single(tmpls.get(), chat2, new_msg, true, /* use_jinja= */ false);
        std::cout << "fmt_single(" << tmpl_str << ") : " << output << "\n";
        std::cout << "-------------------------\n";
        return output;
    };
    assert(fmt_single("chatml") == "\n<|im_start|>user\nHow are you<|im_end|>\n<|im_start|>assistant\n");
    assert(fmt_single("mistral-v1") == " [INST] How are you [/INST]");
    assert(fmt_single("mistral-v3") == "[INST] How are you[/INST]");
    assert(fmt_single("mistral-v3-tekken") == "[INST]How are you[/INST]");
    assert(fmt_single("mistral-v7") == "[INST] How are you[/INST]");
    assert(fmt_single("llama2") == "[INST] How are you [/INST]");
    assert(fmt_single("mistral") == "[INST] How are you [/INST]"); // for old pre-v1 templates
    assert(fmt_single("gemma")  == "\n<start_of_turn>user\nHow are you<end_of_turn>\n<start_of_turn>model\n");
    assert(fmt_single("llama3") == "<|start_header_id|>user<|end_header_id|>\n\nHow are you<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
    // assert(fmt_single("gigachat") == "user<|role_sep|>How are you<|message_sep|>available functions<|role_sep|>[]<|message_sep|>assistant<|role_sep|>");

    std::cout << "\nOK: All tests passed successfully.\n";

    return 0;
}
