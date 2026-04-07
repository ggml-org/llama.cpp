// Compile stb_image implementation for llama-agent (terminal image preview).
// This avoids depending on mtmd's shared library for these symbols on Windows.
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
