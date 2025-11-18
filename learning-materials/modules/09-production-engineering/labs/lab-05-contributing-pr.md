# Lab 5: Contributing to llama.cpp

## Objectives

- ✅ Set up development environment
- ✅ Find and fix a real issue
- ✅ Write tests for your changes
- ✅ Submit a pull request
- ✅ Respond to code review

**Estimated Time**: 3-5 hours
**Note**: This involves contributing to the real llama.cpp repository!

## Part 1: Environment Setup

### Task 1.1: Fork and Clone

```bash
# 1. Fork on GitHub: https://github.com/ggerganov/llama.cpp

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/llama.cpp.git
cd llama.cpp

# 3. Add upstream remote
git remote add upstream https://github.com/ggerganov/llama.cpp.git

# 4. Verify remotes
git remote -v
# origin    https://github.com/YOUR_USERNAME/llama.cpp.git (fetch)
# upstream  https://github.com/ggerganov/llama.cpp.git (fetch)
```

### Task 1.2: Build and Test

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure

# Test manually
./bin/llama-cli -m ../models/your-model.gguf -p "Test"
```

## Part 2: Finding an Issue

### Task 2.1: Browse Good First Issues

Visit: https://github.com/ggerganov/llama.cpp/labels/good%20first%20issue

Or use GitHub CLI:
```bash
gh issue list --label "good first issue" --repo ggerganov/llama.cpp
```

### Task 2.2: Choose an Issue

Good candidates:
- Documentation improvements
- Test coverage gaps
- Minor bug fixes
- Code cleanup

**✏️ Task**: Choose an issue and comment your intention to work on it.

## Part 3: Making Changes

### Task 3.1: Create Feature Branch

```bash
# Update main
git checkout main
git pull upstream main

# Create branch
git checkout -b fix/issue-1234-description

# Work on your fix
# ... edit files ...

# Test your changes
cmake --build build
cd build && ctest
```

### Task 3.2: Example Fix - Documentation

Let's improve documentation:

```markdown
<!-- Before: docs/quantization.md -->
## Quantization

Use `llama-quantize` to convert models.

<!-- After: docs/quantization.md -->
## Quantization

Quantization reduces model size and speeds up inference by using lower precision numbers.

### Available Formats

| Format | Size Reduction | Quality | Use Case |
|--------|----------------|---------|----------|
| Q4_K_M | ~4x smaller | Good | Recommended default |
| Q5_K_M | ~3x smaller | Better | Higher quality needed |
| Q8_0   | ~2x smaller | Excellent | Maximum quality |

### Usage

```bash
# Convert to Q4_K_M (recommended)
./llama-quantize model.gguf model-q4.gguf Q4_K_M

# Convert to Q5_K_M (higher quality)
./llama-quantize model.gguf model-q5.gguf Q5_K_M
```

### Choosing a Format

- **CPU-only**: Use Q4_K_M
- **GPU available**: Use Q4_K_M or Q5_K_M
- **Maximum quality**: Use Q8_0
```

### Task 3.3: Example Fix - Code

Add bounds checking:

```cpp
// Before: src/llama.cpp
void process_tokens(int n_tokens) {
    for (int i = 0; i < n_tokens; i++) {
        // Process token
    }
}

// After: src/llama.cpp
void process_tokens(int n_tokens) {
    // Add validation
    if (n_tokens < 0 || n_tokens > MAX_TOKENS) {
        fprintf(stderr, "Error: Invalid token count: %d\n", n_tokens);
        return;
    }

    for (int i = 0; i < n_tokens; i++) {
        // Process token
    }
}
```

### Task 3.4: Add Tests

```cpp
// tests/test-llama.cpp

TEST_CASE("process_tokens validation") {
    // Test negative tokens
    process_tokens(-1);  // Should handle gracefully

    // Test zero tokens
    process_tokens(0);   // Should succeed

    // Test excessive tokens
    process_tokens(999999);  // Should fail gracefully
}
```

## Part 4: Submitting Pull Request

### Task 4.1: Commit Changes

```bash
# Stage changes
git add docs/quantization.md src/llama.cpp tests/test-llama.cpp

# Commit with descriptive message
git commit -m "Improve quantization documentation and add validation

- Add comparison table for quantization formats
- Include usage examples for each format
- Add guidance for choosing formats
- Add bounds checking in process_tokens()
- Add test cases for token validation

Fixes #1234"
```

### Task 4.2: Push to Fork

```bash
git push origin fix/issue-1234-description
```

### Task 4.3: Create Pull Request

Visit your fork on GitHub and click "Create Pull Request"

Use this template:

```markdown
## Description

Improves quantization documentation and adds input validation.

## Changes

- Added comprehensive quantization format comparison table
- Included practical usage examples
- Added bounds checking for token count
- Added test cases to prevent regression

## Testing

- [x] All existing tests pass
- [x] New tests added and passing
- [x] Manual testing completed
- [x] Documentation builds without errors

## Related Issues

Fixes #1234

## Checklist

- [x] Code follows project style guidelines
- [x] Comments added for complex logic
- [x] Tests cover new functionality
- [x] Documentation updated
- [x] No breaking changes
- [x] Commit messages are clear
```

## Part 5: Code Review

### Task 5.1: Respond to Feedback

Example review comment:
> Could you add a comment explaining why we need this bounds check?

Your response:
```markdown
Good suggestion! I've added a comment:

```cpp
// Validate token count to prevent buffer overflow and ensure
// we don't attempt to process an invalid number of tokens
if (n_tokens < 0 || n_tokens > MAX_TOKENS) {
    fprintf(stderr, "Error: Invalid token count: %d\n", n_tokens);
    return;
}
```

Updated in commit abc1234.
```

### Task 5.2: Make Requested Changes

```bash
# Make changes based on review
vim src/llama.cpp

# Commit updates
git add src/llama.cpp
git commit -m "Add explanatory comment for bounds check

As suggested in review, added comment explaining the validation
logic and its purpose."

# Push updates
git push origin fix/issue-1234-description
```

### Task 5.3: Merge Conflicts

If upstream changed:

```bash
# Update from upstream
git fetch upstream
git rebase upstream/main

# Resolve conflicts if any
# Edit conflicted files
git add resolved-file.cpp
git rebase --continue

# Force push (rebase rewrites history)
git push --force-with-lease origin fix/issue-1234-description
```

## Part 6: After Merge

### Task 6.1: Celebrate! 🎉

Your contribution is now part of llama.cpp!

### Task 6.2: Clean Up

```bash
# Update local main
git checkout main
git pull upstream main
git push origin main

# Delete feature branch
git branch -d fix/issue-1234-description
git push origin --delete fix/issue-1234-description
```

### Task 6.3: Update Your Portfolio

Document your contribution:

```markdown
## My Contributions to llama.cpp

### PR #1234: Improve Quantization Documentation

**Description**: Enhanced documentation with comparison tables and examples, added input validation.

**Impact**: Helped users choose appropriate quantization, improved code safety.

**Stats**:
- Lines changed: +125 / -15
- Files modified: 3
- Tests added: 5

**Link**: https://github.com/ggerganov/llama.cpp/pull/1234
```

## Deliverables

- ✅ Merged pull request
- ✅ Tests passing in CI
- ✅ Documentation updated
- ✅ Code review addressed
- ✅ Portfolio entry created

## Best Practices

### Good Commit Messages

```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what and why, not how.

- Bullet points for multiple changes
- Reference issues: Fixes #123

Co-authored-by: Name <email@example.com>
```

### PR Etiquette

- ✅ Small, focused changes
- ✅ Clear description
- ✅ Tests included
- ✅ Documentation updated
- ✅ Respectful communication
- ❌ Don't: Large, unfocused PRs
- ❌ Don't: Ignore review feedback
- ❌ Don't: Break existing functionality

## Challenge Tasks

1. Fix a real bug from the issues list
2. Add a new feature (with maintainer approval)
3. Improve test coverage
4. Help review other PRs
5. Answer questions in issues

## Verification

Track your contribution status:

```bash
# Check PR status
gh pr status --repo ggerganov/llama.cpp

# View CI results
gh pr checks YOUR_PR_NUMBER --repo ggerganov/llama.cpp

# See discussion
gh pr view YOUR_PR_NUMBER --repo ggerganov/llama.cpp
```

## Resources

- [Contributing Guide](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- [Code of Conduct](https://github.com/ggerganov/llama.cpp/blob/master/CODE_OF_CONDUCT.md)
- [GitHub Docs](https://docs.github.com/en/pull-requests)
- [First Contributions](https://firstcontributions.github.io/)

---

**Congratulations!** You've completed Module 9: Production Engineering.

You now have the skills to:
- Build comprehensive test suites
- Set up CI/CD pipelines
- Conduct performance testing
- Implement security best practices
- Contribute to open source projects

Your llama.cpp production engineering journey continues in the real world!
