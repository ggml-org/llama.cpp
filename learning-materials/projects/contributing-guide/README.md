# Capstone Project: Contributing to llama.cpp

**Difficulty**: Intermediate-Advanced
**Estimated Time**: 20-40 hours (ongoing)
**Modules Required**: All modules
**Prerequisites**: C++17, Git, CMake, Testing

---

## Project Overview

Learn to contribute to the llama.cpp open-source project through practical contributions.

**Contribution Types**:
1. Bug fixes
2. Performance optimizations
3. New quantization formats
4. Backend improvements (CUDA, Metal, ROCm)
5. Documentation
6. Testing and CI/CD

---

## Getting Started

### Step 1: Development Environment Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/llama.cpp
cd llama.cpp

# Build with all backends
mkdir build && cd build
cmake .. -DLLAMA_CUDA=ON -DLLAMA_METAL=ON
make -j8

# Run tests
make test
```

### Step 2: Find an Issue

**Good First Issues**:
- Documentation improvements
- Example scripts
- Testing edge cases
- Performance benchmarks

**Intermediate**:
- Quantization format variants
- Model architecture support
- API improvements

**Advanced**:
- Kernel optimizations (CUDA/Metal)
- New sampling methods
- Multi-GPU strategies

### Step 3: Development Workflow

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes with tests
3. Run CI locally: `./scripts/run-ci.sh`
4. Commit with descriptive messages
5. Push and create Pull Request
6. Address review feedback

---

## Contribution Ideas

### Idea 1: Add New Quantization Format

**Task**: Implement IQ5_XXS (5-bit importance quantization)

**Steps**:
1. Study existing IQ formats in `ggml-quants.c`
2. Implement quantization function
3. Implement dequantization function
4. Add CUDA/Metal kernels
5. Benchmark perplexity
6. Update documentation

**Expected PR**: ~500 lines, 2-3 weeks

### Idea 2: Optimize CUDA Kernel

**Task**: Improve `dequantize_mul_mat_vec_q4_k` performance by 10%

**Steps**:
1. Profile current kernel with Nsight Compute
2. Identify bottlenecks (memory vs compute)
3. Implement optimization (tiling, shared memory)
4. Benchmark on multiple GPUs
5. Ensure correctness with tests

**Expected PR**: ~200 lines, 1-2 weeks

### Idea 3: Add Model Architecture Support

**Task**: Support new model (e.g., Gemma, Phi-3)

**Steps**:
1. Understand model architecture differences
2. Update `llama.cpp` model loading
3. Handle architecture-specific layers
4. Test conversion from HuggingFace
5. Add example

**Expected PR**: ~300 lines, 2-3 weeks

---

## Code Style & Guidelines

**C++ Style**:
```cpp
// Good: Clear variable names, const correctness
static void ggml_compute_forward_mul_mat_q4_k(
    const struct ggml_compute_params * params,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst) {
    
    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    
    // Implementation...
}

// Bad: Unclear names, missing const
void compute(void* p, void* s0, void* s1, void* d) {
    int n = ((int*)s0)[0];  // Magic index
}
```

**Testing**:
```bash
# Always test before PR
./build/bin/test-quantize
./build/bin/test-sampling
./build/bin/test-backend-ops

# Perplexity validation
./build/bin/perplexity -m model.gguf -f wikitext-test.txt
```

---

## Submitting Your Contribution

### Pull Request Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass  
- [ ] Manual testing completed
- [ ] Perplexity checked (if quantization)
- [ ] Benchmarks run (if performance)

## Screenshots (if UI changes)

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. Automated CI runs (GitHub Actions)
2. Maintainer review (1-7 days)
3. Address feedback
4. Approval and merge
5. Your name in CONTRIBUTORS! 🎉

---

## Learning Path

**Month 1**: Small fixes, documentation
**Month 2-3**: Feature additions, optimizations
**Month 4+**: Complex features, architectural changes

**Success Metrics**:
- 3+ merged PRs
- Understanding of codebase
- Active community participation

---

**Resources**:
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Contributing Guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- [Discord Community](https://discord.gg/llama-cpp)

---

**Maintained by**: Agent 8 (Integration Coordinator)
**Last Updated**: 2025-11-18
