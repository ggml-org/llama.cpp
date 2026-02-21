#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

#include <string>

static bool test_arch(const std::string & arch_name) {

}

int main(int argc, char ** argv) {
    gguf_context_ptr metadata(gguf_init_empty());
    llama_model_params params = llama_model_default_params();
    llama_model_ptr(llama_model_empty(metadata.get(), params));
}
