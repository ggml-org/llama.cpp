#pragma once

#include "command-program.h"
#include "kernel-corpus/kernel-corpus.h"
#include "status.h"

#include <string>

namespace ggml::hrx {

Status dump_command_program_kernels_if_requested(const CommandProgram & program,
                                                 const KernelCorpus &   corpus,
                                                 const std::string &    target,
                                                 const std::string &    command_shape);

}  // namespace ggml::hrx
