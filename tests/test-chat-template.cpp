#include <string>
#include <vector>
#include <sstream>

#undef NDEBUG
#include <cassert>

#include "llama.h"
#include "common.h"

int main(void) {
    llama_chat_message conversation[] = {
        {"system", "You are a helpful assistant"},
        {"user", "Hello"},
        {"assistant", "Hi there"},
        {"user", "Who are you"},
        {"assistant", "   I am an assistant   "},
        {"user", "Another question"},
    };
    size_t message_count = 6;
    std::vector<std::string> templates = {
        // teknium/OpenHermes-2.5-Mistral-7B
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
        // mistralai/Mistral-7B-Instruct-v0.2
        "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
        // TheBloke/FusionNet_34Bx2_MoE-AWQ
        "{%- for idx in range(0, messages|length) -%}\\n{%- if messages[idx]['role'] == 'user' -%}\\n{%- if idx > 1 -%}\\n{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}\\n{%- else -%}\\n{{- messages[idx]['content'] + ' [/INST]' -}}\\n{%- endif -%}\\n{% elif messages[idx]['role'] == 'system' %}\\n{{- '[INST] <<SYS>>\\\\n' + messages[idx]['content'] + '\\\\n<</SYS>>\\\\n\\\\n' -}}\\n{%- elif messages[idx]['role'] == 'assistant' -%}\\n{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}\\n{% endif %}\\n{% endfor %}",
        // bofenghuang/vigogne-2-70b-chat
        "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif true == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'Vous êtes Vigogne, un assistant IA créé par Zaion Lab. Vous suivez extrêmement bien les instructions. Aidez autant que vous le pouvez.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\\\n' + system_message + '\\\\n<</SYS>>\\\\n\\\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\\\n' + content.strip() + '\\\\n<</SYS>>\\\\n\\\\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}",
        // mlabonne/AlphaMonarch-7B
        "{% for message in messages %}{{bos_token + message['role'] + '\\n' + message['content'] + eos_token + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ bos_token + 'assistant\\n' }}{% endif %}",
        // google/gemma-7b-it
        "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}",
        // OrionStarAI/Orion-14B-Chat
        "{% for message in messages %}{% if loop.first %}{{ bos_token }}{% endif %}{% if message['role'] == 'user' %}{{ 'Human: ' + message['content'] + '\\n\\nAssistant: ' + eos_token }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}",
        // openchat/openchat-3.5-0106
        // The included chat_template differs from the author's suggestions here: https://huggingface.co/openchat/openchat_3.5/discussions/5#65448109b4a3f3a2f486fd9d
        // So we match against the included template but implement the suggested version.
        "{{ bos_token }}{% for message in messages %}{{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>'}}{% endfor %}{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}",
        // deepseek-ai/deepseek-coder-33b-instruct
        "{% if not add_generation_prompt is defined %}\n{% set add_generation_prompt = false %}\n{% endif %}\n{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{{bos_token}}{%- if not ns.found -%}\n{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{% if add_generation_prompt %}\n{{'### Response:'}}\n{% endif %}",
        // eachadea/vicuna-13b-1.1
        // No template included in tokenizer_config.json, so this template likely needs to be manually set.
        "{%- for message in messages %}{%- if message['role'] == 'system' -%}{{- '' + message['content'] + '\n\n' -}}{%- else -%}{%- if message['role'] == 'user' -%}{{-'USER: ' + message['content'] + '\n'-}}{%- else -%}{{-'ASSISTANT: ' + message['content'] + '</s>\n' -}}{%- endif -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{-'ASSISTANT:'-}}{%- endif -%}",
        // Orca-Vicuna
        // No template included in tokenizer_config.json, so this template likely needs to be manually set.
        "{%- for message in messages %}{%- if message['role'] == 'system' -%}{{-'SYSTEM: ' + message['content'] + '\n' -}}{%- else -%}{%- if message['role'] == 'user' -%}{{-'USER: ' + message['content'] + '\n'-}}{%- else -%}{{-'ASSISTANT: ' + message['content'] + '</s>\n' -}}{%- endif -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{-'ASSISTANT:'-}}{%- endif -%}",
        // CohereForAI/c4ai-command-r-plus
        "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true %}{% set loop_messages = messages %}{% set system_message = 'You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message != false %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}",
        // Llama-3
        "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}",
        //Phi-3-mini
        "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}",
        //Phi-3-small
        "{{ bos_token }}{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}",
        //Phi-3-medium
        "{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}",
        //Phi-3-vision
        "{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>\n' }}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{- '<|assistant|>\n' -}}{% endif %}",
        // ChatGLM3
        "{% for message in messages %}{% if loop.first %}[gMASK]sop<|{{ message['role'] }}|>\n {{ message['content'] }}{% else %}<|{{ message['role'] }}|>\n {{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}",
        // ChatGLM4
        u8"[gMASK]<sop>{% for item in messages %}{% if item['tools'] is defined %}<|system|>\n你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n# 可用工具{% set tools = item['tools'] %}{% for tool in tools %}{% if tool['type'] == 'function' %}\n\n## {{ tool['function']['name'] }}\n\n{{ tool['function'] | tojson(indent=4) }}\n......{% endif %}{% endfor %}{% endif %}{% if item['content'] %}<|{{ item['role'] }}|>{{ item['metadata'] }}\n{{ item['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}",
        // MiniCPM-3B-OpenHermes-2.5-v2-GGUF
        u8"{% for message in messages %}{% if message['role'] == 'user' %}{{'<用户>' + message['content'].strip() + '<AI>'}}{% else %}{{message['content'].strip()}}{% endif %}{% endfor %}",
        // DeepSeek-V2
        "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}",
    };
    std::vector<std::string> expected_output = {
        // teknium/OpenHermes-2.5-Mistral-7B
        "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nWho are you<|im_end|>\n<|im_start|>assistant\n   I am an assistant   <|im_end|>\n<|im_start|>user\nAnother question<|im_end|>\n<|im_start|>assistant\n",
        // mistralai/Mistral-7B-Instruct-v0.2
        "[INST] You are a helpful assistant\nHello [/INST]Hi there</s>[INST] Who are you [/INST]   I am an assistant   </s>[INST] Another question [/INST]",
        // TheBloke/FusionNet_34Bx2_MoE-AWQ
        "[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nHello [/INST] Hi there </s><s>[INST] Who are you [/INST]    I am an assistant    </s><s>[INST] Another question [/INST]",
        // bofenghuang/vigogne-2-70b-chat
        "[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nHello [/INST] Hi there </s>[INST] Who are you [/INST] I am an assistant </s>[INST] Another question [/INST]",
        // mlabonne/AlphaMonarch-7B
        "system\nYou are a helpful assistant</s>\n<s>user\nHello</s>\n<s>assistant\nHi there</s>\n<s>user\nWho are you</s>\n<s>assistant\n   I am an assistant   </s>\n<s>user\nAnother question</s>\n<s>assistant\n",
        // google/gemma-7b-it
        "<start_of_turn>user\nYou are a helpful assistant\n\nHello<end_of_turn>\n<start_of_turn>model\nHi there<end_of_turn>\n<start_of_turn>user\nWho are you<end_of_turn>\n<start_of_turn>model\nI am an assistant<end_of_turn>\n<start_of_turn>user\nAnother question<end_of_turn>\n<start_of_turn>model\n",
        // OrionStarAI/Orion-14B-Chat
        "Human: You are a helpful assistant\n\nHello\n\nAssistant: </s>Hi there</s>Human: Who are you\n\nAssistant: </s>   I am an assistant   </s>Human: Another question\n\nAssistant: </s>",
        // openchat/openchat-3.5-0106
        "You are a helpful assistant<|end_of_turn|>GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant: Hi there<|end_of_turn|>GPT4 Correct User: Who are you<|end_of_turn|>GPT4 Correct Assistant:    I am an assistant   <|end_of_turn|>GPT4 Correct User: Another question<|end_of_turn|>GPT4 Correct Assistant:",
        // deepseek-ai/deepseek-coder-33b-instruct
        "You are a helpful assistant### Instruction:\nHello\n### Response:\nHi there\n<|EOT|>\n### Instruction:\nWho are you\n### Response:\n   I am an assistant   \n<|EOT|>\n### Instruction:\nAnother question\n### Response:\n",
        // eachadea/vicuna-13b-1.1
        "You are a helpful assistant\n\nUSER: Hello\nASSISTANT: Hi there</s>\nUSER: Who are you\nASSISTANT:    I am an assistant   </s>\nUSER: Another question\nASSISTANT:",
        // Orca-Vicuna
        "SYSTEM: You are a helpful assistant\nUSER: Hello\nASSISTANT: Hi there</s>\nUSER: Who are you\nASSISTANT:    I am an assistant   </s>\nUSER: Another question\nASSISTANT:",
        // CohereForAI/c4ai-command-r-plus
        "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are a helpful assistant<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Hi there<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Who are you<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>I am an assistant<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Another question<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        // Llama 3
        "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHi there<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWho are you<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI am an assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nAnother question<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        //Phi-3-mini
        "<|system|>\nYou are a helpful assistant<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\nHi there<|end|>\n<|user|>\nWho are you<|end|>\n<|assistant|>\n   I am an assistant   <|end|>\n<|user|>\nAnother question<|end|>\n<|assistant|>\n",
        //Phi-3-small
        "<|system|>\nYou are a helpful assistant<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\nHi there<|end|>\n<|user|>\nWho are you<|end|>\n<|assistant|>\n   I am an assistant   <|end|>\n<|user|>\nAnother question<|end|>\n<|assistant|>\n",
        //Phi-3-medium
        "<|system|>\nYou are a helpful assistant<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\nHi there<|end|>\n<|user|>\nWho are you<|end|>\n<|assistant|>\n   I am an assistant   <|end|>\n<|user|>\nAnother question<|end|>\n<|assistant|>\n",
        //Phi-3-vision
        "<|system|>\nYou are a helpful assistant<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\nHi there<|end|>\n<|user|>\nWho are you<|end|>\n<|assistant|>\n   I am an assistant   <|end|>\n<|user|>\nAnother question<|end|>\n<|assistant|>\n",
        // ChatGLM3
        "[gMASK]sop<|system|>\n You are a helpful assistant<|user|>\n Hello<|assistant|>\n Hi there<|user|>\n Who are you<|assistant|>\n    I am an assistant   <|user|>\n Another question<|assistant|>",
        // ChatGLM4
        "[gMASK]<sop><|system|>\nYou are a helpful assistant<|user|>\nHello<|assistant|>\nHi there<|user|>\nWho are you<|assistant|>\n   I am an assistant   <|user|>\nAnother question<|assistant|>",
        // MiniCPM-3B-OpenHermes-2.5-v2-GGUF
        u8"You are a helpful assistant<用户>Hello<AI>Hi there<用户>Who are you<AI>I am an assistant<用户>Another question<AI>",
        // DeepSeek-V2
        u8"You are a helpful assistant\n\nUser: Hello\n\nAssistant: Hi there<｜end▁of▁sentence｜>User: Who are you\n\nAssistant:    I am an assistant   <｜end▁of▁sentence｜>User: Another question\n\nAssistant:",
    };
    std::vector<char> formatted_chat(1024);
    int32_t res;

    // test invalid chat template
    res = llama_chat_apply_template(nullptr, "INVALID TEMPLATE", conversation, message_count, true, formatted_chat.data(), formatted_chat.size());
    assert(res < 0);

    for (size_t i = 0; i < templates.size(); i++) {
        std::string custom_template = templates[i];
        std::string expected = expected_output[i];
        formatted_chat.resize(1024);
        res = llama_chat_apply_template(
            nullptr,
            custom_template.c_str(),
            conversation,
            message_count,
            true,
            formatted_chat.data(),
            formatted_chat.size()
        );
        formatted_chat.resize(res);
        std::string output(formatted_chat.data(), formatted_chat.size());
        printf("%s\n", output.c_str());
        printf("-------------------------\n");
        assert(output == expected);
    }


    // test llama_chat_format_single for system message
    printf("\n\n=== llama_chat_format_single (system message) ===\n\n");
    std::vector<common_chat_msg> chat2;
    common_chat_msg sys_msg{"system", "You are a helpful assistant"};

    auto fmt_sys = [&](std::string tmpl) {
        auto output = common_chat_format_single(nullptr, tmpl, chat2, sys_msg, false);
        printf("fmt_sys(%s) : %s\n", tmpl.c_str(), output.c_str());
        printf("-------------------------\n");
        return output;
    };
    assert(fmt_sys("chatml") == "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n");
    assert(fmt_sys("llama2") == "[INST] You are a helpful assistant\n");
    assert(fmt_sys("gemma")  == ""); // for gemma, system message is merged with user message
    assert(fmt_sys("llama3") == "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|>");


    // test llama_chat_format_single for user message
    printf("\n\n=== llama_chat_format_single (user message) ===\n\n");
    chat2.push_back({"system", "You are a helpful assistant"});
    chat2.push_back({"user", "Hello"});
    chat2.push_back({"assistant", "I am assistant"});
    common_chat_msg new_msg{"user", "How are you"};

    auto fmt_single = [&](std::string tmpl) {
        auto output = common_chat_format_single(nullptr, tmpl, chat2, new_msg, true);
        printf("fmt_single(%s) : %s\n", tmpl.c_str(), output.c_str());
        printf("-------------------------\n");
        return output;
    };
    assert(fmt_single("chatml") == "\n<|im_start|>user\nHow are you<|im_end|>\n<|im_start|>assistant\n");
    assert(fmt_single("llama2") == "[INST] How are you [/INST]");
    assert(fmt_single("gemma")  == "\n<start_of_turn>user\nHow are you<end_of_turn>\n<start_of_turn>model\n");
    assert(fmt_single("llama3") == "<|start_header_id|>user<|end_header_id|>\n\nHow are you<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");

    return 0;
}
