# Contributing to Open Source: llama.cpp Guide

## Introduction

Contributing to open-source projects like llama.cpp is an excellent way to give back to the community, improve your skills, and build your professional portfolio. This lesson covers how to effectively contribute to llama.cpp and other ML open-source projects.

## Why Contribute?

### Personal Benefits
- **Skill Development**: Learn from experienced developers
- **Portfolio Building**: Showcase your work to potential employers
- **Networking**: Connect with the ML/AI community
- **Problem Solving**: Work on real-world challenges
- **Recognition**: Build your reputation in the field

### Community Benefits
- **Bug Fixes**: Help improve stability
- **Features**: Add capabilities others need
- **Documentation**: Make project more accessible
- **Testing**: Validate changes across platforms
- **Support**: Help other users

## Understanding llama.cpp Project

### Project Structure

```
llama.cpp/
├── .github/             # GitHub workflows, issue templates
├── CMakeLists.txt       # Build configuration
├── Makefile             # Alternative build system
├── examples/            # Example applications
├── ggml/                # GGML library (tensor operations)
├── include/             # Public headers
│   └── llama.h         # Main API header
├── src/                 # Implementation
│   ├── llama.cpp       # Core inference
│   ├── llama-vocab.cpp # Tokenization
│   └── llama-*.cpp     # Various modules
├── tests/               # Test suite
├── scripts/             # Utility scripts
├── models/              # Model conversion scripts
└── docs/                # Documentation
```

### Key Components

1. **Core Library** (`src/llama.cpp`)
   - Model loading
   - Inference engine
   - Memory management

2. **GGML** (`ggml/`)
   - Tensor operations
   - Backend abstraction (CPU, CUDA, Metal)
   - Computation graphs

3. **Applications** (`examples/`)
   - llama-cli: Command-line interface
   - llama-server: HTTP server
   - llama-bench: Benchmarking tool

4. **Model Tools** (`scripts/`, `examples/`)
   - Model conversion
   - Quantization utilities

## Getting Started

### 1. Set Up Development Environment

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/llama.cpp.git
cd llama.cpp

# Add upstream remote
git remote add upstream https://github.com/ggerganov/llama.cpp.git

# Build the project
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug -j$(nproc)

# Run tests
ctest --output-on-failure
```

### 2. Keep Your Fork Updated

```bash
# Fetch latest from upstream
git fetch upstream

# Update your main branch
git checkout main
git merge upstream/main

# Push to your fork
git push origin main
```

## Types of Contributions

### 1. Bug Reports

**Good Bug Report Example:**

```markdown
## Description
Model inference fails with segmentation fault when using context size > 4096

## Steps to Reproduce
1. Build llama.cpp with `cmake .. -DLLAMA_CUDA=ON`
2. Run: `./llama-cli -m model.gguf -p "Test" -n 10 -c 8192`
3. Observe crash

## Expected Behavior
Should generate text with 8192 context size

## Actual Behavior
Segmentation fault after processing ~4100 tokens

## Environment
- OS: Ubuntu 22.04
- GPU: NVIDIA RTX 4090 (24GB)
- CUDA: 12.2
- llama.cpp version: commit abc123

## Logs
```
[debug output...]
```

## Additional Context
Works fine with -c 4096, fails with any value > 4096
```

### 2. Bug Fixes

```bash
# Create a branch for your fix
git checkout -b fix/context-size-overflow

# Make your changes
# (edit files, add tests)

# Test your changes
cmake --build build
cd build && ctest

# Commit with descriptive message
git add src/llama.cpp
git commit -m "Fix context size overflow for sizes > 4096

- Allocate sufficient memory for large contexts
- Add bounds checking in context allocation
- Add test case for context sizes up to 32768

Fixes #1234"

# Push to your fork
git push origin fix/context-size-overflow
```

### 3. New Features

**Feature Proposal Process:**

1. **Open Discussion Issue First**
   - Explain the feature and use case
   - Get feedback from maintainers
   - Discuss implementation approach

2. **Implementation**
   - Follow project coding style
   - Add comprehensive tests
   - Update documentation
   - Consider backward compatibility

**Example Feature: Mirostat v3 Sampling**

```cpp
// src/llama-sampling.cpp

// Add to llama_sampler_chain_params
struct llama_sampler_chain_params {
    // ... existing fields ...
    bool use_mirostat_v3;
    float mirostat_tau;
    float mirostat_eta;
};

// Implement Mirostat v3
llama_token llama_sample_token_mirostat_v3(
    struct llama_context * ctx,
    llama_token_data_array * candidates,
    float tau,
    float eta,
    float * mu
) {
    // Implementation
    // ...
}

// Add to sampling chain
if (params.use_mirostat_v3) {
    llama_sample_token_mirostat_v3(
        ctx,
        &candidates_p,
        params.mirostat_tau,
        params.mirostat_eta,
        &mu
    );
}
```

**Add Tests:**

```cpp
// tests/test-sampling.cpp
TEST_CASE("mirostat_v3_sampling") {
    llama_model * model = load_test_model();
    llama_context * ctx = llama_new_context_with_model(model, params);

    // Test basic functionality
    std::vector<float> logits = {1.0, 2.0, 3.0, 4.0, 5.0};
    float mu = 5.0;

    llama_token token = llama_sample_token_mirostat_v3(
        ctx, logits.data(), logits.size(),
        3.0,  // tau
        0.1,  // eta
        &mu
    );

    REQUIRE(token >= 0);
    REQUIRE(token < logits.size());

    // Test convergence
    // ...
}
```

**Update Documentation:**

```markdown
# docs/sampling.md

## Mirostat v3 Sampling

Mirostat v3 is an improved sampling method that dynamically adjusts the
threshold based on the observed entropy.

### Parameters

- `mirostat_tau` (float): Target entropy (default: 5.0)
- `mirostat_eta` (float): Learning rate (default: 0.1)

### Usage

```bash
./llama-cli -m model.gguf \
  --sampling-method mirostat-v3 \
  --mirostat-tau 5.0 \
  --mirostat-eta 0.1
```

### Example

```cpp
llama_sampling_params params = llama_sampling_default_params();
params.use_mirostat_v3 = true;
params.mirostat_tau = 5.0;
params.mirostat_eta = 0.1;
```
```

### 4. Documentation

**Documentation improvements are always welcome:**

- Fix typos and grammar
- Add examples
- Clarify confusing sections
- Translate to other languages
- Add diagrams and visualizations

**Example Documentation PR:**

```markdown
# docs/quantization.md

## GGUF Quantization Formats

### Overview

GGUF supports multiple quantization formats, each offering different
trade-offs between model size, speed, and quality.

### Format Comparison

| Format | Size | Speed | Quality | Use Case |
|--------|------|-------|---------|----------|
| Q4_0   | 4.3GB | Fast | Good | CPU inference, memory constrained |
| Q4_K_M | 4.5GB | Fast | Better | Balanced quality/size |
| Q5_K_M | 5.0GB | Medium | Excellent | Best quality for size |
| Q8_0   | 7.2GB | Slower | Near-perfect | Maximum quality |

### Choosing a Format

**For CPU-only inference:**
- Use Q4_K_M for best balance
- Use Q4_0 if memory is tight
- Use Q5_K_M if you have extra RAM

**For GPU inference:**
- Use Q4_K_M or Q5_K_M
- Q8_0 for minimal quality loss
- Avoid Q2/Q3 formats

### Example Usage

```bash
# Convert to Q4_K_M
./llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M

# Convert to Q5_K_M
./llama-quantize model.gguf model-q5_k_m.gguf Q5_K_M
```
```

### 5. Code Review

Contributing through code review helps:
- Catch bugs early
- Improve code quality
- Learn from others
- Help maintainers

**Good Code Review Comments:**

```markdown
# PR #1234: Add support for Llama 3.1

## Overall
Great work! The implementation looks solid. A few suggestions:

## Specific Comments

**File: `src/llama.cpp` Line 123**
```cpp
if (n_vocab > 0) {
    vocab.resize(n_vocab);
}
```
Should we check if `n_vocab` is within reasonable bounds?
Maybe add: `if (n_vocab > MAX_VOCAB_SIZE) return false;`

**File: `src/llama-vocab.cpp` Line 456**
Consider adding a comment explaining the Llama 3.1 tokenizer differences
from Llama 2.

**File: `tests/test-llama.cpp`**
Great test coverage! Could we add a test for the edge case where
`n_vocab == 0`?

## Performance
Have you benchmarked this against Llama 2? Would be good to verify
performance is comparable.

## Documentation
Please add usage example to `docs/models.md` showing how to load
Llama 3.1 models.
```

## Coding Standards

### C++ Style Guide

```cpp
// Use descriptive names
int n_tokens;          // Good
int n;                // Avoid

// Function naming: snake_case
void llama_sample_token(/*...*/);

// Struct naming: llama_prefix
struct llama_context {
    // ...
};

// Constants: UPPER_CASE
#define LLAMA_MAX_CONTEXT 32768

// Use const where appropriate
const std::vector<float> & logits;

// Avoid auto when type isn't obvious
auto tokens = model.tokenize(text);  // OK (obvious)
auto result = process();             // Avoid (unclear)

// Use nullptr instead of NULL
llama_model * model = nullptr;

// Use range-based for loops
for (const auto & token : tokens) {
    // ...
}

// Use early returns
if (!model) {
    return nullptr;
}
// ... rest of function

// Add comments for complex logic
// Calculate perplexity using cross-entropy
// H(p, q) = -Σ p(x) log q(x)
float perplexity = calculate_perplexity(logits);
```

### Commit Messages

**Good commit message format:**

```
Short (50 chars or less) summary

More detailed explanatory text, if necessary. Wrap it to about 72
characters. The blank line separating the summary from the body is
critical.

Further paragraphs come after blank lines.

- Bullet points are okay
- Use a hyphen or asterisk

If the commit fixes an issue, reference it:
Fixes #123
Closes #456

Co-authored-by: Name <email@example.com>
```

**Examples:**

```bash
# Good
git commit -m "Fix memory leak in KV cache allocation

The KV cache was not being freed when context was destroyed,
leading to GPU memory leaks over time.

- Add explicit cleanup in llama_free()
- Add test to detect memory leaks
- Update documentation

Fixes #1234"

# Bad
git commit -m "fix bug"
git commit -m "WIP"
git commit -m "changes"
```

## Pull Request Process

### 1. Creating a PR

**PR Template:**

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed? What problem does it solve?

## Changes
- List of specific changes
- Another change
- And another

## Testing
How was this tested?
- [ ] Existing tests pass
- [ ] New tests added
- [ ] Manually tested on X, Y, Z

## Screenshots (if applicable)
[Add screenshots for UI changes]

## Checklist
- [ ] Code follows project style
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Commit messages are descriptive
- [ ] No merge conflicts

## Related Issues
Fixes #123
Relates to #456
```

### 2. Responding to Review

```markdown
> Should we check if n_vocab is within reasonable bounds?

Good catch! Added bounds checking:

```cpp
if (n_vocab > LLAMA_MAX_VOCAB || n_vocab < 0) {
    fprintf(stderr, "Invalid vocab size: %d\n", n_vocab);
    return false;
}
```

> Could we add a test for the edge case where n_vocab == 0?

Added test case in commit abc1234:
```cpp
TEST_CASE("zero_vocab_size") {
    REQUIRE(llama_load_model(null_vocab_model) == nullptr);
}
```

> Have you benchmarked this?

Yes! Results:
- Llama 2: 45.3 tokens/sec
- Llama 3.1: 44.8 tokens/sec (1.1% slower, within margin of error)
```

### 3. Getting Merged

After approval:
- Maintainer will merge your PR
- Delete your feature branch
- Update your local repo
- Celebrate! 🎉

## Building Your Open Source Profile

### 1. Start Small

Good first contributions:
- Fix typos in documentation
- Add code comments
- Improve error messages
- Add examples
- Fix compiler warnings

**Finding good first issues:**

```bash
# GitHub CLI
gh issue list --label "good first issue"

# Or check:
# https://github.com/ggerganov/llama.cpp/labels/good%20first%20issue
```

### 2. Be Active in Community

- Answer questions in Issues
- Help review PRs
- Participate in Discussions
- Share your use cases
- Write blog posts about your contributions

### 3. Document Your Contributions

Keep track for your portfolio:

```markdown
# My Contributions to llama.cpp

## Merged Pull Requests
1. **[#1234]** Add Mirostat v3 sampling (March 2025)
   - Implemented new sampling method
   - 500+ LOC, comprehensive tests
   - Featured in v1.5.0 release

2. **[#1235]** Fix GPU memory leak (March 2025)
   - Critical bug fix
   - Reduced OOM errors by 95%

3. **[#1236]** Improve quantization docs (April 2025)
   - Added comparison table
   - 50+ upvotes, helped hundreds of users

## Code Reviews
- Reviewed 15+ PRs
- Helped identify 3 critical bugs
- Mentored 2 first-time contributors

## Community Impact
- Answered 50+ questions in Issues
- Blog post: "Getting Started with llama.cpp" (2000+ views)
- Conference talk: "Contributing to Open Source ML" (150 attendees)
```

## Common Pitfalls to Avoid

### 1. Not Following Guidelines
❌ Submitting PR without reading CONTRIBUTING.md
✅ Read project guidelines first

### 2. Large, Unfocused PRs
❌ PR with 50 files changed, multiple unrelated features
✅ Small, focused PRs that do one thing well

### 3. Poor Communication
❌ "Fixed stuff"
✅ Clear description of what, why, and how

### 4. Ignoring Feedback
❌ Arguing with every code review comment
✅ Be receptive to feedback, ask questions

### 5. Breaking Changes Without Discussion
❌ Changing public API without warning
✅ Discuss breaking changes first

## Resources

### llama.cpp Resources
- [GitHub Repository](https://github.com/ggerganov/llama.cpp)
- [Discord Server](https://discord.gg/llama-cpp)
- [Wiki](https://github.com/ggerganov/llama.cpp/wiki)

### General Open Source
- [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)
- [First Contributions](https://firstcontributions.github.io/)
- [Open Source Friday](https://opensourcefriday.com/)

### Development Tools
- [GitHub CLI](https://cli.github.com/)
- [Git](https://git-scm.com/book/en/v2)
- [CMake](https://cmake.org/documentation/)

## Summary

Contributing to llama.cpp:

1. **Start Small**: Begin with documentation or small fixes
2. **Follow Guidelines**: Read CONTRIBUTING.md and coding standards
3. **Communicate**: Open discussions before major changes
4. **Test Thoroughly**: Add tests for all changes
5. **Document**: Update docs with your changes
6. **Be Patient**: Reviews take time, be respectful
7. **Learn**: Every contribution is a learning opportunity
8. **Have Fun**: Enjoy being part of the community!

Your contributions, no matter how small, make a difference. Welcome to the llama.cpp community! 🦙

---

**Authors**: Agent 5 (Documentation Specialist)
**Last Updated**: 2025-11-18
**Estimated Reading Time**: 35 minutes
