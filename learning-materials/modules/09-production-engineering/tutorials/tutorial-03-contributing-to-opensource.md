# Tutorial: Contributing to Open Source (llama.cpp)

## Introduction

This tutorial guides you through making your first contribution to llama.cpp, from finding an issue to getting your PR merged.

**Time Required**: 2-4 hours
**Result**: A merged contribution to llama.cpp!

## Part 1: Getting Started

### Step 1: Set Up Your Environment

```bash
# 1. Fork llama.cpp on GitHub
# Visit: https://github.com/ggerganov/llama.cpp
# Click "Fork" button

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/llama.cpp.git
cd llama.cpp

# 3. Add upstream remote
git remote add upstream https://github.com/ggerganov/llama.cpp.git

# 4. Verify setup
git remote -v
# You should see:
# origin    https://github.com/YOUR_USERNAME/llama.cpp.git
# upstream  https://github.com/ggerganov/llama.cpp.git

# 5. Build the project
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j$(nproc)

# 6. Run tests
ctest --output-on-failure
```

### Step 2: Find Something to Contribute

**Option A: Good First Issues**

```bash
# Using GitHub CLI
gh issue list --label "good first issue" --repo ggerganov/llama.cpp

# Or visit:
# https://github.com/ggerganov/llama.cpp/labels/good%20first%20issue
```

**Option B: Fix Something You've Encountered**

- Typos in documentation
- Missing examples
- Unclear error messages
- Small bugs you've found

**Option C: Improve Documentation**

Documentation improvements are always welcome:
- Add missing examples
- Clarify confusing sections
- Fix typos
- Add diagrams

## Part 2: Making Your First Contribution

### Example 1: Documentation Improvement

Let's improve the quantization documentation.

**Step 1: Create a branch**

```bash
# Update main first
git checkout main
git pull upstream main

# Create feature branch
git checkout -b docs/improve-quantization-guide
```

**Step 2: Make changes**

Edit `docs/quantization.md`:

```markdown
<!-- BEFORE -->
## Quantization

Quantization reduces model size.

<!-- AFTER -->
## Quantization

Quantization reduces model size by using lower precision numbers, trading some quality for significant size and speed improvements.

### Quick Start

```bash
# Download and convert a model
./llama-quantize original.gguf quantized.gguf Q4_K_M
```

### Choosing a Quantization Level

| Format | Size Reduction | Quality | Best For |
|--------|----------------|---------|----------|
| Q4_K_M | ~4x smaller | Good | General use, CPU |
| Q5_K_M | ~3x smaller | Better | Higher quality needs |
| Q8_0   | ~2x smaller | Excellent | Maximum quality |

### Example Comparison

For a 7B parameter model:
- FP16: ~14 GB
- Q4_K_M: ~4 GB (4x smaller, 95% quality)
- Q5_K_M: ~5 GB (3x smaller, 98% quality)
- Q8_0: ~7 GB (2x smaller, 99% quality)

### Recommended Settings

**For CPU inference:**
```bash
./llama-quantize model.gguf model-q4.gguf Q4_K_M
```

**For GPU with limited VRAM:**
```bash
./llama-quantize model.gguf model-q5.gguf Q5_K_M
```

**For maximum quality:**
```bash
./llama-quantize model.gguf model-q8.gguf Q8_0
```
```

**Step 3: Commit**

```bash
git add docs/quantization.md
git commit -m "Improve quantization documentation

- Add quick start guide with example
- Add comparison table for different formats
- Include file size examples for 7B model
- Add recommended settings for different use cases
- Improve formatting and clarity

Makes it easier for new users to choose the right
quantization format for their needs."
```

**Step 4: Push**

```bash
git push origin docs/improve-quantization-guide
```

**Step 5: Create Pull Request**

Visit your fork on GitHub and click "Create Pull Request"

**PR Template:**

```markdown
## Description

Improves quantization documentation with practical examples and clear guidance.

## Changes

- Added quick start guide
- Added comparison table showing size/quality trade-offs
- Included concrete file size examples
- Added recommended settings for common scenarios
- Improved formatting for better readability

## Motivation

Current documentation is brief and doesn't help users choose between quantization formats. This makes it easier for new users to make informed decisions.

## Screenshots (if applicable)

N/A - documentation only

## Checklist

- [x] Changes follow project style
- [x] Documentation is clear and accurate
- [x] Examples are tested and working
- [x] Commit message is descriptive
- [x] No breaking changes

## Related Issues

Helps with #[issue-number] (if applicable)
```

### Example 2: Code Contribution

Let's add bounds checking to prevent crashes.

**Step 1: Identify the problem**

```cpp
// Current code in src/llama.cpp
void process_batch(int n_tokens) {
    for (int i = 0; i < n_tokens; i++) {
        // Process token
    }
}
```

**Issue**: No validation of `n_tokens`, could cause crashes.

**Step 2: Create branch**

```bash
git checkout main
git pull upstream main
git checkout -b fix/add-bounds-checking
```

**Step 3: Implement fix**

```cpp
// Improved code
void process_batch(int n_tokens) {
    // Validate input
    if (n_tokens < 0 || n_tokens > MAX_BATCH_SIZE) {
        fprintf(stderr, "Error: Invalid batch size: %d (must be 0-%d)\n",
                n_tokens, MAX_BATCH_SIZE);
        return;
    }

    for (int i = 0; i < n_tokens; i++) {
        // Process token
    }
}
```

**Step 4: Add test**

Create or update `tests/test-llama.cpp`:

```cpp
TEST_CASE("batch processing validation") {
    // Test negative batch size
    REQUIRE_NOTHROW(process_batch(-1));  // Should handle gracefully

    // Test zero batch size
    REQUIRE_NOTHROW(process_batch(0));

    // Test excessive batch size
    REQUIRE_NOTHROW(process_batch(999999));  // Should reject

    // Test valid batch size
    REQUIRE_NOTHROW(process_batch(32));
}
```

**Step 5: Test your changes**

```bash
# Build
cmake --build build

# Run tests
cd build && ctest

# Manual testing
./bin/llama-cli -m ../models/test-model.gguf -p "Test"
```

**Step 6: Commit**

```bash
git add src/llama.cpp tests/test-llama.cpp
git commit -m "Add bounds checking for batch size

- Validate n_tokens is within valid range
- Return early with error message for invalid values
- Prevents potential crashes from invalid batch sizes
- Add test cases to prevent regression

Fixes #1234"
```

**Step 7: Push and create PR**

```bash
git push origin fix/add-bounds-checking
```

## Part 3: Code Review Process

### Responding to Feedback

**Example Review Comment:**

> Could you add a comment explaining the MAX_BATCH_SIZE limit?

**Your Response:**

```markdown
Good point! I've added a comment explaining the limit:

```cpp
// Maximum batch size to prevent memory overflow
// Calculated as: (available_memory / token_size) / safety_factor
#define MAX_BATCH_SIZE 2048

void process_batch(int n_tokens) {
    // Validate input to ensure we don't exceed memory limits
    if (n_tokens < 0 || n_tokens > MAX_BATCH_SIZE) {
        // ...
    }
}
```

Updated in commit abc1234.
```

### Handling Merge Conflicts

If upstream changed while you were working:

```bash
# Fetch latest changes
git fetch upstream

# Rebase your branch
git rebase upstream/main

# Resolve conflicts if any
# Edit conflicted files
git add resolved-file.cpp
git rebase --continue

# Force push (rebase rewrites history)
git push --force-with-lease origin fix/add-bounds-checking
```

### Making Requested Changes

```bash
# Make the changes
vim src/llama.cpp

# Commit
git add src/llama.cpp
git commit -m "Address review feedback

- Add explanatory comments as suggested
- Rename variable for clarity
- Update error message format"

# Push
git push origin fix/add-bounds-checking
```

## Part 4: After Your PR is Merged

### Celebrate! 🎉

You're now a llama.cpp contributor!

### Update Your Fork

```bash
# Switch to main
git checkout main

# Pull latest changes (includes your PR!)
git pull upstream main

# Push to your fork
git push origin main

# Delete feature branch
git branch -d fix/add-bounds-checking
git push origin --delete fix/add-bounds-checking
```

### Update Your Portfolio

Add to your README or portfolio:

```markdown
## Open Source Contributions

### llama.cpp

**PR #1234: Add bounds checking for batch size**
- Added input validation to prevent crashes
- Improved error messages
- Added comprehensive test coverage
- [View PR](https://github.com/ggerganov/llama.cpp/pull/1234)

Impact: Prevents potential crashes from invalid batch sizes, improving stability for all users.
```

## Part 5: Continuing to Contribute

### Find More Issues

```bash
# Look for issues you can help with
gh issue list --repo ggerganov/llama.cpp --label "good first issue"

# Filter by topic
gh issue list --repo ggerganov/llama.cpp --search "documentation"
```

### Help Others

- Answer questions in issues
- Review other PRs
- Help newcomers
- Share your experience

### Build Expertise

- Focus on specific area (e.g., quantization, GPU support)
- Tackle progressively harder issues
- Propose new features (after discussion)

## Best Practices

### DO ✅

- **Read CONTRIBUTING.md** before starting
- **Discuss large changes** in issues first
- **Write clear commit messages**
- **Add tests** for code changes
- **Update documentation**
- **Be patient and respectful**
- **Respond to feedback** promptly

### DON'T ❌

- **Submit huge PRs** (keep them focused)
- **Change unrelated code**
- **Ignore review comments**
- **Break existing functionality**
- **Force push** after review started
- **Take criticism personally**

## Troubleshooting

### PR Not Getting Reviewed?

- Wait patiently (maintainers are busy)
- Ping politely after 1 week
- Ensure CI is passing
- Check if you addressed all feedback

### Tests Failing in CI?

```bash
# Run tests locally first
cmake --build build
cd build && ctest

# Check specific test
ctest -R test_name -V
```

### Merge Conflicts?

```bash
# Update and rebase
git fetch upstream
git rebase upstream/main

# Resolve conflicts
# ... edit files ...
git add resolved-files
git rebase --continue
```

## Resources

- [llama.cpp Contributing Guide](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- [First Contributions](https://firstcontributions.github.io/)
- [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/)
- [GitHub Pull Request Docs](https://docs.github.com/en/pull-requests)

## Quick Reference

### Git Commands

```bash
# Update from upstream
git fetch upstream && git rebase upstream/main

# Create branch
git checkout -b feature/my-feature

# Commit
git add files && git commit -m "descriptive message"

# Push
git push origin feature/my-feature

# Amend last commit
git commit --amend

# Interactive rebase (clean up commits)
git rebase -i HEAD~3
```

### GitHub CLI

```bash
# Create PR
gh pr create --title "Title" --body "Description"

# Check PR status
gh pr status

# View PR
gh pr view 1234

# Check CI status
gh pr checks 1234
```

## Summary

Contributing to open source:

1. **Start small** - Documentation, tests, small fixes
2. **Communicate** - Ask questions, discuss changes
3. **Be thorough** - Test, document, follow guidelines
4. **Stay patient** - Reviews take time
5. **Learn continuously** - Each contribution teaches you
6. **Help others** - Share your knowledge
7. **Have fun** - Enjoy being part of the community!

Your contributions make llama.cpp better for everyone! 🦙

---

**Congratulations!** You've completed Module 9: Production Engineering.

You're now equipped to build, deploy, monitor, and contribute to production LLM inference systems!
