#include "llama.h"

int main(void) {
    bool (* volatile supports_mixed_batch)(const struct llama_model *) = llama_model_supports_mixed_batch;
    return supports_mixed_batch == NULL;
}
