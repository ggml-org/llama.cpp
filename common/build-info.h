#pragma once

int llama_build_number(void);

const char * llama_commit(void);
const char * llama_compiler(void);

const char * llama_build_target(void);
const char * llama_build_info(void);

void llama_print_build_info(void);
