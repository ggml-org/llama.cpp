define newline


endef

$(error Build system changed:$(newline)\
The Makefile build has been replaced by CMake.$(newline)$(newline)\
For build instructions see:$(newline)\
https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md$(newline)${newline})
