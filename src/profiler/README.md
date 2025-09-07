Profiler integration for llama.cpp
----------------------------------

- `llama_perfetto.cpp`: C/C++ glue exposing simple C-callable helpers used across the codebase.
- `perfetto.cc` / `perfetto.h`: Perfetto SDK amalgamation (vendored).

Notes
- The integration is backend-agnostic and works with CPU-only builds.
- For Vulkan, optional counters/timeline hooks are resolved dynamically at runtime. This covers
  platforms like macOS, Linux (including Mali GPUs), and Android.
