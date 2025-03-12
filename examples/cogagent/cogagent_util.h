#ifndef COGAGENT_UTIL_H
#define COGAGENT_UTIL_H

#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include <string>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <stdexcept>
#include <sstream>
#include <cinttypes>
#include <limits>

extern void set_processing_text(llama_context * ctx, bool value);

extern void set_cross_input(llama_context * ctx, std::vector<float> &value);

void print_dims(struct ggml_tensor * input_tensor, const char * name);

struct ggml_tensor * get_tensor(struct ggml_context * dst_ctx, struct ggml_context * src_ctx, std::string tensor_name, int &count_failed);

void save_tensor_filename(struct ggml_tensor * input_tensor, std::string filename);

void save_tensor_from_data(std::vector<float> tensor_data, int* dims, std::string filename);

bool load_from_gguf(const char * fname, struct ggml_context * ctx_ggml, struct gguf_context * ctx_gguf);

int get_input(
    std::vector<float> &input_data, const char * filename
);

#endif