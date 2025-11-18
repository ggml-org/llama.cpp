# Module 1: Foundations

Welcome to Module 1 of the LLaMA.cpp Learning System! This module will build your foundational understanding of llama.cpp, its architecture, and core concepts needed for LLM inference.

## Overview

This module introduces you to llama.cpp from the ground up. You'll learn what llama.cpp is, why it exists, how to build and run it, and how to perform basic inference. By the end of this module, you'll be comfortable navigating the codebase and understanding the fundamentals of efficient LLM inference on CPUs.

**Estimated Time**: 15-20 hours
**Difficulty Level**: Beginner to Intermediate
**Prerequisites**:
- Intermediate Python programming skills
- Basic understanding of neural networks and language models
- Familiarity with command line and basic build tools
- C/C++ reading ability (helpful but not required)

## Learning Objectives

By completing this module, you will:

- ✅ **Understand** what llama.cpp is and its role in the LLM ecosystem
- ✅ **Master** the GGUF file format structure and metadata system
- ✅ **Build** llama.cpp from source with various configuration options
- ✅ **Perform** basic text generation and inference tasks
- ✅ **Calculate** memory requirements for different models and configurations
- ✅ **Navigate** the llama.cpp codebase confidently
- ✅ **Explain** key architectural decisions and design trade-offs

## Module Structure

### Lesson 1.1: Introduction to LLaMA-CPP
**Time**: 2 hours | **Difficulty**: Beginner

Get introduced to llama.cpp and understand its place in the LLM inference landscape.

**What you'll learn**:
- What llama.cpp is and why it exists
- Comparison with PyTorch, TensorFlow, and other inference engines
- Use cases: edge deployment, CPU inference, resource-constrained environments
- Overview of quantization and why it matters

**Materials**:
- 📄 Documentation: "What is LLaMA-CPP?"
- 📄 Documentation: "History and Evolution of LLM Inference"
- 💻 Code Example: Inference speed comparison
- 📝 Tutorial: "Your First 10 Minutes with LLaMA-CPP"
- 🎯 Interview Questions: 5 conceptual questions

**Hands-On**:
- Lab 1.1: Install and run pre-built binaries
- Exercise: Generate text with different model sizes

---

### Lesson 1.2: GGUF File Format
**Time**: 3 hours | **Difficulty**: Beginner to Intermediate

Deep dive into the GGUF file format, the native format used by llama.cpp.

**What you'll learn**:
- GGUF structure: headers, metadata, tensors
- Metadata key-value pairs and their purposes
- Tensor layout, alignment, and memory mapping
- Why GGUF replaced GGML format
- Converting models from HuggingFace to GGUF

**Materials**:
- 📄 Documentation: "GGUF Format Deep Dive"
- 📄 Paper Summary: "GGUF Specification"
- 💻 Python Example: GGUF metadata reader
- 💻 Python Example: GGUF converter utility
- 🔬 Lab 1.2: Exploring GGUF Files
- 🎯 Interview Questions: 5 format-related questions

**Hands-On**:
- Lab 1.2: Inspect model metadata programmatically
- Exercise: Convert a HuggingFace model to GGUF format
- Challenge: Modify GGUF metadata fields

---

### Lesson 1.3: Build System & Toolchain
**Time**: 2 hours | **Difficulty**: Intermediate

Learn to build llama.cpp from source and understand the build configuration.

**What you'll learn**:
- CMake build system basics
- Build configuration options and flags
- Enabling different backends (CPU, CUDA, Metal, OpenCL)
- Compiler optimization flags and their impact
- Cross-compilation for different platforms
- Troubleshooting common build issues

**Materials**:
- 📄 Documentation: "Building LLaMA-CPP from Source"
- 📄 Documentation: "CMake Options Reference"
- 💻 Shell Scripts: Build automation examples
- 🔬 Lab 1.3: Build Configuration
- 🎯 Interview Questions: 3 build system questions

**Hands-On**:
- Lab 1.3: Build llama.cpp with different backend configurations
- Exercise: Enable CUDA support and verify
- Challenge: Cross-compile for ARM architecture

---

### Lesson 1.4: Basic Inference
**Time**: 4 hours | **Difficulty**: Intermediate

Perform your first inference operations and understand the generation process.

**What you'll learn**:
- Loading models into memory
- Context windows and token limits
- The token generation loop
- Sampling methods: greedy, top-k, top-p, temperature
- Generation parameters and their effects
- Using llama-cli and understanding command-line options

**Materials**:
- 📄 Documentation: "Inference Fundamentals"
- 💻 Python Examples: 5 basic inference scripts
- 💻 C++ Example: Using the llama.h API
- 🔬 Lab 1.4: First Inference
- 📝 Tutorial: "Text Generation Walkthrough"
- 🎯 Interview Questions: 7 inference questions

**Hands-On**:
- Lab 1.4: Load a model and generate text
- Exercise: Experiment with different sampling parameters
- Project: Build a simple chatbot script

---

### Lesson 1.5: Memory Management Basics
**Time**: 2 hours | **Difficulty**: Intermediate

Understand how llama.cpp manages memory during inference.

**What you'll learn**:
- Calculating memory requirements from parameter count
- Memory breakdown: weights, KV cache, activations
- Memory-mapped file loading (mmap)
- Context size vs memory usage relationship
- Quantization's effect on memory
- Troubleshooting out-of-memory errors

**Materials**:
- 📄 Documentation: "Memory Management in LLaMA-CPP"
- 💻 Python Example: Memory requirement calculator
- 🔬 Lab 1.5: Memory Profiling
- 🎯 Interview Questions: 5 memory questions

**Hands-On**:
- Lab 1.5: Profile memory usage during inference
- Exercise: Calculate memory for different quantization levels
- Challenge: Optimize inference for 8GB RAM constraint

---

### Lesson 1.6: Codebase Navigation
**Time**: 2 hours | **Difficulty**: Intermediate

Learn to navigate and understand the llama.cpp source code.

**What you'll learn**:
- Source code organization (`/src/`, `/include/`, `/common/`)
- Key data structures: `llama_model`, `llama_context`, `llama_batch`
- Important functions and their purposes
- Model initialization and loading flow
- Inference pipeline code path
- Backend abstraction layer

**Materials**:
- 📄 Documentation: "Codebase Architecture Guide"
- 📄 Documentation: "Important Functions Reference"
- 💻 Annotated Code: Key file walkthroughs
- 📝 Tutorial: "Code Reading Guide"

**Hands-On**:
- Exercise: Find the implementation of specific features
- Exercise: Trace a token through the generation pipeline
- Challenge: Identify where sampling happens in the code

---

## Module Assessment

### Capstone Lab
**Lab 1.6**: Build a complete inference pipeline
- Build llama.cpp with optimizations
- Load a quantized model
- Implement custom sampling logic
- Add performance metrics
- Create a simple CLI interface

### Quiz
**20 Interview-Style Questions** covering:
- Conceptual understanding (5 questions)
- GGUF format knowledge (4 questions)
- Build and configuration (3 questions)
- Inference mechanics (5 questions)
- Memory management (3 questions)

### Mini-Project
Build a command-line inference tool that:
- Accepts model path and prompt as arguments
- Supports configurable generation parameters
- Shows token-by-token generation
- Reports performance metrics (tokens/sec, memory usage)
- Handles errors gracefully

---

## Success Criteria

You should be able to:

- [ ] Build llama.cpp from source successfully
- [ ] Explain what GGUF is and how it differs from other formats
- [ ] Load a model and generate coherent text
- [ ] Calculate memory requirements for a given model
- [ ] Navigate the codebase to find specific functionality
- [ ] Explain the inference pipeline from prompt to output
- [ ] Troubleshoot common build and runtime issues
- [ ] Configure sampling parameters for different use cases

---

## Learning Paths

### Python Developer Track
If you're focusing on Python applications:
- Complete lessons 1.1, 1.2, 1.4, 1.5
- Focus on Python examples
- Skip deep C++ code navigation (1.6)
- Proceed to Module 2 (selected lessons) or Module 8

### CUDA Engineer Track
If you're targeting GPU optimization:
- Complete all lessons thoroughly
- Pay special attention to 1.3 (build with CUDA)
- Deep dive into 1.6 (codebase navigation)
- Proceed to Module 2, then Module 4 (GPU Acceleration)

### Full-Stack Track
Complete all lessons in order before proceeding to Module 2.

---

## Prerequisites Check

Before starting, ensure you have:

**Knowledge Prerequisites**:
- ✓ Comfortable with Python (functions, classes, file I/O)
- ✓ Understanding of what language models are
- ✓ Basic neural network concepts (layers, parameters, inference)
- ✓ Command-line comfort (navigating directories, running commands)

**System Prerequisites**:
- ✓ Linux, macOS, or Windows (WSL recommended)
- ✓ At least 8GB RAM (16GB+ recommended)
- ✓ 20GB free disk space
- ✓ C++ compiler (GCC, Clang, or MSVC)
- ✓ CMake 3.14+
- ✓ Python 3.8+

**Optional (for specific lessons)**:
- ✓ NVIDIA GPU + CUDA Toolkit (for GPU exercises)
- ✓ Git (for cloning the repository)

---

## How to Use This Module

### Recommended Approach

1. **Read First**: Start with documentation and paper summaries
2. **Watch/Follow**: Go through tutorials step-by-step
3. **Code Along**: Run and modify code examples
4. **Hands-On**: Complete labs (most important!)
5. **Practice**: Do exercises and challenges
6. **Test**: Answer interview questions to check understanding
7. **Build**: Complete the mini-project

### Time Management

- **Focused Learning**: 2-3 hours per session
- **Completion Timeline**: 1-2 weeks at 10 hours/week
- **Weekend Intensive**: 2 weekends (8-10 hours each)

### Getting Help

- **Stuck on a concept?** Re-read the documentation, try the code examples
- **Build issues?** Check the troubleshooting guide in Lesson 1.3
- **Code not working?** Verify your environment matches prerequisites
- **Still stuck?** Check the llama.cpp GitHub issues or community forums

---

## Module Resources

### Code Examples Location
All code examples are in: `/learning-materials/modules/01-foundations/code/`

### Lab Materials Location
Lab instructions and starter code: `/learning-materials/modules/01-foundations/labs/`

### Documentation Location
Detailed guides and references: `/learning-materials/modules/01-foundations/docs/`

### Tutorial Location
Step-by-step tutorials: `/learning-materials/modules/01-foundations/tutorials/`

---

## What's Next?

After completing Module 1, you'll be ready for:

**Module 2: Core Implementation** (for deep technical understanding)
- Model architecture deep dive
- Tokenization internals
- KV cache implementation
- Advanced inference techniques

**Module 3: Quantization & Optimization** (for performance focus)
- Quantization theory and practice
- GGUF quantization formats
- Performance profiling and optimization

**Module 8: Integration & Applications** (for Python developers)
- Python bindings and llama-cpp-python
- Building RAG systems
- Chat applications
- Function calling

Choose based on your learning track and goals!

---

## Feedback and Improvements

This is a living curriculum. As you progress through the module:
- Note what works well and what doesn't
- Identify gaps or unclear explanations
- Suggest additional examples or exercises
- Share your projects and learnings

---

**Module Owner**: Agent 8 (Integration Coordinator)
**Content Contributors**: Agents 1, 3, 4, 5, 6
**Last Updated**: 2025-11-18
**Version**: 1.0

Ready to start? Begin with [Lesson 1.1: Introduction to LLaMA-CPP](docs/lesson-1.1-introduction.md)!
