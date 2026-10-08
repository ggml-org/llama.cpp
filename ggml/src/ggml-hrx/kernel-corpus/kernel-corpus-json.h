#pragma once

#include "kernel-corpus.h"

#include <string>

namespace ggml::hrx {

std::string serialize_kernel_corpus_json(const KernelCorpus & corpus);

}  // namespace ggml::hrx
