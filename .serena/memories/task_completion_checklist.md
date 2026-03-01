# Task Completion Checklist

When a coding task is completed, perform these steps:

1. **Format code**: `clang-format-19 -i <modified files>`
2. **Build**: `ninja -C build` (ensure no compilation errors)
3. **Test**: `ctest --test-dir build --output-on-failure`
4. **For ggml changes**: Run `./build/bin/test-backend-ops` on multiple backends
5. **For PR submission**: Add `ggml-ci` to commit message for extended CI
