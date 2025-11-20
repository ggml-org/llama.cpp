#pragma once

#include "chat.h"

struct common_chat_templates;

common_chat_params common_chat_templates_apply_jinja(
    const struct common_chat_templates * tmpls,
    const struct common_chat_templates_inputs & inputs);

void common_chat_parse(class common_chat_msg_parser & builder);

void common_chat_parse_content_only(class common_chat_msg_parser & builder);
