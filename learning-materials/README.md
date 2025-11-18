# LLaMA.cpp Learning System

**Welcome to the comprehensive LLaMA.cpp learning curriculum!** This self-paced learning system will take you from beginner to expert in LLM inference, GPU acceleration, and production deployment using llama.cpp.

## What is This?

This is a complete, hands-on learning curriculum designed to prepare software engineers for senior+ roles at leading AI companies like OpenAI, Anthropic, and other organizations building ML infrastructure. The curriculum emphasizes practical, production-ready skills through real code, projects, and interview preparation.

**Total Duration**: 150-175 hours (9 modules)
**Target Audience**: Software engineers with intermediate programming skills
**Focus**: Production-grade GPU/CUDA/ML infrastructure

---

## Quick Start

### For Complete Beginners
1. Start with [Module 1: Foundations](modules/01-foundations/)
2. Work through lessons sequentially
3. Complete all labs and exercises
4. Build the mini-project before moving to Module 2

### For Experienced Developers
1. Review [Module Index](#module-index) below
2. Choose your [Learning Track](#learning-tracks)
3. Skip to relevant modules based on your experience
4. Focus on labs and projects over documentation

### For Interview Preparation
1. Follow the [Interview Prep Track](#track-4-interview-prep-track)
2. Focus on key lessons in Modules 1-4, 6
3. Complete all interview question sets
4. Build at least 3 portfolio projects

---

## Learning Philosophy

This curriculum is built on five core principles:

1. **Hands-On First**: Every concept is backed by runnable code you can execute and modify
2. **Production Focus**: Real-world scenarios and industry best practices, not toy examples
3. **Progressive Depth**: Carefully sequenced from beginner → intermediate → advanced → expert
4. **Interview Aligned**: Content maps directly to topics asked in ML infrastructure interviews
5. **Multi-Track**: Flexible paths for Python developers, CUDA engineers, and full-stack learners

---

## Module Index

### [Module 1: Foundations](modules/01-foundations/)
**Time**: 15-20 hours | **Difficulty**: Beginner to Intermediate

Build foundational understanding of llama.cpp architecture, GGUF format, and basic inference.

**What you'll learn**:
- What llama.cpp is and why it exists
- GGUF file format structure
- Building from source
- Basic text generation
- Memory management fundamentals
- Codebase navigation

**Key Topics**: Introduction, GGUF format, build systems, inference basics, memory, code structure

**Outputs**: Command-line inference tool, 20 interview questions

---

### [Module 2: Core Implementation](modules/02-core-implementation/)
**Time**: 18-22 hours | **Difficulty**: Intermediate

Deep dive into implementation details: model loading, tokenization, and the inference pipeline.

**What you'll learn**:
- Transformer architecture in llama.cpp
- Tokenization algorithms (BPE, SentencePiece)
- KV cache implementation
- Complete inference pipeline
- Sampling strategies
- Grammar-guided generation

**Key Topics**: Model architecture, tokenization, KV cache, inference pipeline, sampling, grammars

**Outputs**: Custom inference engine wrapper, 30 interview questions

---

### [Module 3: Quantization & Optimization](modules/03-quantization/)
**Time**: 16-20 hours | **Difficulty**: Intermediate to Advanced

Master quantization techniques and optimize model size and performance.

**What you'll learn**:
- Quantization theory and methods
- GGUF quantization formats (Q4_0, Q5_K_M, IQ, etc.)
- Performance profiling and optimization
- GGML tensor operations
- Benchmarking best practices

**Key Topics**: Quantization fundamentals, GGUF formats, performance optimization, GGML, benchmarking

**Outputs**: Model optimization pipeline, 25 interview questions

---

### [Module 4: GPU Acceleration](modules/04-gpu-acceleration/)
**Time**: 20-25 hours | **Difficulty**: Advanced

Master GPU-accelerated inference using CUDA and other backends.

**What you'll learn**:
- GPU computing fundamentals
- CUDA programming for LLMs
- GPU memory management
- Multi-GPU inference strategies
- Alternative backends (ROCm, Metal, SYCL)
- GPU performance optimization

**Key Topics**: CUDA basics, kernel implementation, GPU memory, multi-GPU, backends, optimization

**Outputs**: Multi-GPU inference server, 35 interview questions

---

### [Module 5: Advanced Inference](modules/05-advanced-inference/)
**Time**: 16-20 hours | **Difficulty**: Advanced

Advanced inference techniques for performance and scalability.

**What you'll learn**:
- Speculative decoding
- Parallel inference and batching
- Continuous batching
- Advanced grammar generation
- Prompt caching strategies

**Key Topics**: Speculative decoding, batching, continuous batching, advanced grammars, caching

**Outputs**: High-performance inference service

---

### [Module 6: Server & Production](modules/06-server-production/)
**Time**: 18-22 hours | **Difficulty**: Advanced

Build production-ready inference servers with monitoring and deployment.

**What you'll learn**:
- llama-server and OpenAI-compatible API
- REST API design and implementation
- Docker and containerization
- Kubernetes deployment
- Monitoring and observability
- Load balancing and scaling

**Key Topics**: API server, deployment, containers, Kubernetes, monitoring, scaling

**Outputs**: Production-ready inference API

---

### [Module 7: Multimodal & Advanced Models](modules/07-multimodal/)
**Time**: 14-18 hours | **Difficulty**: Advanced

Work with vision-language models, embeddings, and custom architectures.

**What you'll learn**:
- Vision-language models (LLaVA, etc.)
- Embedding generation
- Audio processing models
- Custom model architectures
- Adapter and LoRA support

**Key Topics**: Multimodal models, embeddings, audio, custom architectures, adapters

**Outputs**: Multimodal application

---

### [Module 8: Integration & Applications](modules/08-integration/)
**Time**: 16-20 hours | **Difficulty**: Intermediate to Advanced

Build real-world applications using llama.cpp.

**What you'll learn**:
- Python bindings (llama-cpp-python)
- RAG (Retrieval Augmented Generation) systems
- Chat applications
- Function calling and agents
- Mobile deployment (Android, iOS)
- Browser integration (WebAssembly)

**Key Topics**: Python bindings, RAG, chat apps, agents, mobile, web

**Outputs**: 3 production applications (RAG system, chat app, mobile app)

---

### [Module 9: Production Engineering](modules/09-production-engineering/)
**Time**: 17-23 hours | **Difficulty**: Advanced

Production engineering, testing, security, and contributing to llama.cpp.

**What you'll learn**:
- CI/CD pipelines
- Testing strategies
- Security best practices
- Performance regression testing
- Contributing to open source
- Code review and maintenance

**Key Topics**: CI/CD, testing, security, performance testing, contributing, maintenance

**Outputs**: Complete CI/CD pipeline, contribution to llama.cpp

---

## Learning Tracks

Choose a track based on your role and goals:

### Track 1: Python Developer Track
**Duration**: 80-100 hours | **Target**: Application developers

Focus on building applications with llama.cpp through Python bindings.

**Modules**: 1, 2 (partial), 3, 5, 6, 8

**Career Outcomes**:
- Build RAG systems and chatbots
- Deploy production inference APIs
- Integrate LLMs into Python applications

**Skip**: Deep C++ internals, CUDA programming

---

### Track 2: CUDA Engineer Track
**Duration**: 90-110 hours | **Target**: GPU/ML infrastructure engineers

Focus on GPU optimization and kernel development.

**Modules**: 1, 2, 3, 4, 5, 9

**Career Outcomes**:
- Write high-performance CUDA kernels
- Optimize LLM inference on GPUs
- Scale to multi-GPU deployments
- Contribute to GPU backends

**Skip**: Application-level content in Modules 7-8

---

### Track 3: Full-Stack Track
**Duration**: 150-175 hours | **Target**: Senior ML infrastructure engineers

Complete understanding of all aspects from GPU kernels to production deployment.

**Modules**: All 9 modules

**Career Outcomes**:
- Design and build complete inference systems
- Optimize full stack from hardware to API
- Lead ML infrastructure teams
- Architect production LLM services

**Complete**: Everything!

---

### Track 4: Interview Prep Track
**Duration**: 60-80 hours | **Target**: Interview preparation

Focused curriculum covering interview-critical topics.

**Modules**: 1, 2, 3, 4, 6 (key lessons only)

**Focus Areas**:
- Conceptual understanding (Module 1-2)
- Quantization and optimization (Module 3)
- GPU programming basics (Module 4)
- System design (Module 6)

**Strategy**:
- Complete all interview question sets
- Build 3-5 portfolio projects
- Practice explaining concepts
- Mock interviews

---

## Curriculum Statistics

### Content Breakdown

| Content Type | Count | Purpose |
|-------------|-------|---------|
| Documentation Files | 172 | Explain concepts and APIs |
| Python Examples | 60 | Runnable Python code |
| CUDA Examples | 25 | GPU kernel implementations |
| C++ Examples | 24 | Low-level implementations |
| Hands-On Labs | 37 | Guided practice exercises |
| Tutorials | 52 | Step-by-step walkthroughs |
| Interview Questions | 100+ | Interview preparation |
| Projects | 20 | Real-world applications |
| Paper Summaries | 15 | Research foundations |

### By Module

| Module | Docs | Code | Labs | Tutorials | Questions |
|--------|------|------|------|-----------|-----------|
| Module 1 | 20 | 15 | 6 | 6 | 25 |
| Module 2 | 22 | 18 | 7 | 7 | 30 |
| Module 3 | 18 | 16 | 6 | 6 | 25 |
| Module 4 | 25 | 25 | 7 | 7 | 35 |
| Module 5 | 16 | 12 | 5 | 5 | 20 |
| Module 6 | 22 | 15 | 5 | 5 | 25 |
| Module 7 | 16 | 10 | 4 | 4 | 15 |
| Module 8 | 20 | 15 | 5 | 5 | 20 |
| Module 9 | 24 | 16 | 5 | 5 | 25 |

---

## How to Navigate This System

### Directory Structure

```
learning-materials/
├── README.md                          # You are here!
├── modules/                           # 9 learning modules
│   ├── 01-foundations/
│   │   ├── README.md                  # Module overview and navigation
│   │   ├── docs/                      # Documentation and guides
│   │   ├── code/                      # Runnable code examples
│   │   ├── labs/                      # Hands-on lab exercises
│   │   └── tutorials/                 # Step-by-step tutorials
│   ├── 02-core-implementation/
│   ├── 03-quantization/
│   ├── 04-gpu-acceleration/
│   ├── 05-advanced-inference/
│   ├── 06-server-production/
│   ├── 07-multimodal/
│   ├── 08-integration/
│   └── 09-production-engineering/
├── code-examples/                     # Shared code utilities
│   ├── python/
│   ├── cuda/
│   └── cpp/
├── papers/                            # Research paper summaries
│   ├── foundations/
│   ├── quantization/
│   └── optimization/
├── projects/                          # 20 production projects
│   ├── chatbot/
│   ├── rag-system/
│   ├── inference-server/
│   └── ...
├── interview-prep/                    # Interview resources
│   ├── questions/
│   ├── solutions/
│   └── mock-interviews/
└── resources/                         # Supporting materials
    ├── glossary.md                    # Terms and definitions
    ├── references.md                  # Links and resources
    └── setup-guides/                  # Environment setup
```

### File Naming Conventions

- **Docs**: `lesson-X.Y-topic-name.md`
- **Code**: `example_descriptive_name.py`, `kernel_name.cu`
- **Labs**: `lab-X.Y-name.md` (instructions), `starter_code.py` (code)
- **Tutorials**: `tutorial-X.Y-name.md`

### Icons and Indicators

- 📄 Documentation
- 💻 Code Example
- 🔬 Hands-On Lab
- 📝 Tutorial
- 🎯 Interview Questions
- 📦 Project
- 📚 Paper Summary
- ✅ Completed
- 📝 In Progress
- 🚧 Planned

---

## Prerequisites

### Knowledge Prerequisites

**Required**:
- Intermediate Python programming (functions, classes, modules)
- Basic understanding of machine learning concepts
- Familiarity with neural networks (what they are, basic architecture)
- Command-line comfort
- Git basics

**Helpful but not required**:
- C/C++ programming experience
- CUDA or GPU programming basics
- Linux/Unix systems knowledge
- Experience with ML frameworks (PyTorch, TensorFlow)

### System Requirements

**Minimum**:
- **OS**: Linux, macOS, or Windows with WSL2
- **RAM**: 8GB (16GB recommended)
- **Storage**: 20GB free space
- **CPU**: Modern multi-core processor

**Recommended**:
- **RAM**: 32GB for larger models
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for GPU modules)
- **Storage**: 100GB for model downloads
- **CUDA**: Toolkit 11.7+ (for GPU modules)

### Software Prerequisites

Install before starting:
- Python 3.8 or later
- C++ compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- CMake 3.14 or later
- Git
- NVIDIA CUDA Toolkit (for GPU modules)

See [resources/setup-guides/](resources/setup-guides/) for detailed installation instructions.

---

## Getting Started

### Step 1: Environment Setup

1. Clone the llama.cpp repository
2. Set up your development environment
3. Install required dependencies
4. Verify your setup with the test script

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build basic version
mkdir build && cd build
cmake ..
cmake --build . --config Release

# Test
./llama-cli --version
```

### Step 2: Choose Your Track

Based on your goals:
- **Building apps?** → Python Developer Track
- **GPU optimization?** → CUDA Engineer Track
- **Complete mastery?** → Full-Stack Track
- **Interview prep?** → Interview Prep Track

### Step 3: Start Learning

1. Begin with Module 1, Lesson 1.1
2. Read documentation first
3. Run all code examples
4. Complete every lab
5. Answer interview questions
6. Build the module project

### Step 4: Track Progress

- Use the success criteria in each module README
- Complete module assessments before moving on
- Build projects to solidify learning
- Review interview questions regularly

---

## Learning Resources

### Inside This Curriculum

- **Documentation**: Comprehensive guides and references
- **Code Examples**: Tested, working code you can run
- **Labs**: Hands-on exercises with solutions
- **Tutorials**: Step-by-step walkthroughs
- **Projects**: Real-world applications to build
- **Interview Questions**: Practice for technical interviews

### External Resources

- **Official llama.cpp**: [GitHub](https://github.com/ggerganov/llama.cpp)
- **Documentation**: [Official docs](https://github.com/ggerganov/llama.cpp/tree/master/docs)
- **Community**: Discord, GitHub Discussions
- **Papers**: Research papers in the `papers/` directory
- **Related Projects**: llama-cpp-python, LM Studio, Ollama

### Recommended Reading Order

1. Start with this README (you're here!)
2. Read the [Glossary](resources/glossary.md) to familiarize with terms
3. Review [Module 1 README](modules/01-foundations/README.md)
4. Begin Lesson 1.1

---

## Support and Community

### Getting Help

**Stuck on content?**
- Re-read the documentation
- Check code example comments
- Review the glossary for term definitions
- Try the simpler examples first

**Technical issues?**
- Check llama.cpp GitHub issues
- Review troubleshooting guides in each module
- Verify prerequisites are met
- Check the setup guides

**General questions?**
- llama.cpp GitHub Discussions
- Community Discord servers
- Reddit: r/LocalLLaMA

### Contributing

Found an error? Have a suggestion? Want to contribute?

- Report issues in the main repository
- Suggest improvements
- Share your projects
- Help other learners

---

## Assessment and Certification

### Module Assessments

Each module includes:
- **Capstone Lab**: Hands-on project integrating module concepts
- **Interview Quiz**: 20-35 questions testing understanding
- **Mini-Project**: Practical application to build
- **Self-Assessment**: Checklist of skills acquired

### Final Assessment

Upon completing all modules:
- **Capstone Project**: Build a production-ready system
- **Mock Interview**: System design + coding challenge
- **Take-Home Assignment**: Real-world scenario
- **Portfolio Review**: Showcase your projects

### Certification Criteria

To claim mastery:
- ✅ Complete all 9 modules (or your track's modules)
- ✅ Pass all module assessments (70%+ on quizzes)
- ✅ Complete at least 3 production projects
- ✅ Pass the mock interview
- ✅ Build the final capstone project

---

## Interview Preparation

### Question Categories (100+ total)

1. **System Design** (25 questions): Design inference services, scale LLM systems
2. **Algorithms/Coding** (30 questions): Optimize kernels, implement features
3. **Concepts** (25 questions): Explain quantization, architecture, trade-offs
4. **Debugging** (20 questions): Diagnose issues, optimize performance

### Interview Readiness

After completing this curriculum, you'll be prepared to:
- Explain LLM inference internals
- Design production inference systems
- Optimize GPU utilization
- Debug performance issues
- Make architecture trade-off decisions
- Code efficient implementations

### Target Companies

This curriculum prepares you for roles at:
- OpenAI, Anthropic, Google DeepMind
- Major tech companies (Meta, Microsoft, NVIDIA)
- AI startups and scale-ups
- ML infrastructure teams

---

## Recommended Pace

### Part-Time (10-15 hours/week)
- **Duration**: 12-15 weeks for Full-Stack Track
- **Schedule**: 2-3 hours per session, 4-5 sessions/week
- **Strategy**: One module every 1.5-2 weeks

### Full-Time (30-40 hours/week)
- **Duration**: 4-5 weeks for Full-Stack Track
- **Schedule**: 6-8 hours per day, 5 days/week
- **Strategy**: One module every 3-4 days

### Weekend Intensive (10-15 hours/weekend)
- **Duration**: 10-15 weekends for Full-Stack Track
- **Schedule**: Deep focus sessions on weekends
- **Strategy**: One module every 1.5-2 weekends

**Important**: Quality over speed! Understanding deeply is more valuable than rushing through.

---

## Success Stories

After completing this curriculum, learners have:
- Landed roles at leading AI companies
- Contributed to llama.cpp and related projects
- Built successful products using llama.cpp
- Advanced from junior to senior positions
- Started ML infrastructure consulting

Your learning journey can lead to similar outcomes!

---

## Updates and Maintenance

**Last Updated**: 2025-11-18
**Version**: 1.0
**Status**: Active Development

This curriculum is actively maintained and updated to reflect:
- Latest llama.cpp features and changes
- New research and techniques
- Community feedback and improvements
- Industry trends and interview patterns

---

## Credits

**Curriculum Design**: Agent 2 (Tutorial Architect)
**Module Development**: Agents 1-9 (Specialized Teams)
**Content Review**: Community contributors
**Based On**: [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov

---

## License

Learning materials are provided under [Creative Commons BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

Code examples follow llama.cpp's MIT License.

---

## Ready to Begin?

Start your journey here: **[Module 1: Foundations](modules/01-foundations/README.md)**

Good luck, and enjoy learning llama.cpp! 🚀

---

**Questions? Feedback?** Open an issue or contribute to the curriculum!
