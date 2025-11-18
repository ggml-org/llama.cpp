# Module 5: Advanced Inference - Interview Questions

**Purpose**: Interview preparation for advanced inference techniques
**Target Level**: Senior to Staff Engineers
**Module Coverage**: Module 5 - Speculative Decoding, Continuous Batching, Advanced Sampling
**Question Count**: 15 (distributed across 4 categories)
**Last Updated**: 2025-11-18
**Created By**: Agent 8 (Integration Coordinator)

---

## Table of Contents

1. [Conceptual Questions](#conceptual-questions) (4 questions)
2. [Technical Questions](#technical-questions) (4 questions)
3. [System Design Questions](#system-design-questions) (4 questions)
4. [Debugging Questions](#debugging-questions) (3 questions)

---

## Conceptual Questions

### Question 1: Speculative Decoding Fundamentals

**Category**: Conceptual | **Difficulty**: Senior (L5/L6) | **Time**: 20 minutes

**Question**: Explain speculative decoding. How does it achieve speedup? What are the quality trade-offs?

**Key Points**:
- Draft model generates k tokens quickly
- Target model verifies in parallel
- Acceptance rate determines speedup
- No quality degradation (mathematically equivalent)
- Speedup formula: k × acceptance_rate / (1 + draft_overhead)

**Model Answer Requirements**:
- Algorithm explanation with example
- Acceptance sampling mechanism
- Tree-based vs sequential speculation
- Draft model selection criteria
- Performance analysis (2-3x speedup typical)

**Rubric**: 14/28 (Senior), 20/28 (Staff)

---

### Question 2: Continuous Batching

**Category**: Conceptual | **Difficulty**: Senior (L5/L6) | **Time**: 20 minutes

**Question**: What is continuous batching (iteration-level batching)? How does it differ from static batching? Implement a simple continuous batcher.

**Key Points**:
- Add/remove requests mid-batch
- Maximizes GPU utilization
- Reduces average latency
- Complexity in KV cache management
- Used in vLLM, TensorRT-LLM

**Code Required**: Python implementation showing request lifecycle

---

### Question 3: Advanced Sampling Strategies

**Category**: Conceptual | **Difficulty**: Mid-Senior (L4/L5) | **Time**: 15 minutes

**Question**: Compare Mirostat, locally typical sampling, and classifier-free guidance. When would you use each?

**Key Points**:
- Mirostat: Perplexity targeting for coherence
- Locally typical: Entropy-based selection
- CFG: Guidance for specific attributes
- Use cases and parameter tuning

---

### Question 4: Constrained Decoding with Grammars

**Category**: Conceptual | **Difficulty**: Senior (L5/L6) | **Time**: 20 minutes

**Question**: Explain constrained decoding using grammars (GBNF). How do you ensure outputs follow JSON schema? What's the performance impact?

**Key Points**:
- Grammar compilation to state machine
- Token filtering at each step
- Performance overhead (10-30%)
- Alternative: Guided generation, JSON mode

---

## Technical Questions

### Question 5: Implementing Speculative Decoding

**Category**: Technical | **Difficulty**: Staff (L6/L7) | **Time**: 45 minutes

**Question**: Implement speculative decoding with a draft model. Handle acceptance/rejection and KV cache management.

**Implementation**: Complete Python/C++ code showing:
- Draft generation (k=4 tokens)
- Parallel verification
- Acceptance sampling
- Cache rollback on rejection

---

### Question 6: Prefix Caching Implementation

**Category**: Technical | **Difficulty**: Senior (L5/L6) | **Time**: 30 minutes

**Question**: Implement prefix caching to share KV cache across requests with common prefixes (e.g., system prompts).

**Code Requirements**:
- Prefix matching (trie/hash)
- Cache sharing and COW
- Memory management
- Hit rate tracking

---

### Question 7: Batched Beam Search

**Category**: Technical | **Difficulty**: Senior (L5/L6) | **Time**: 35 minutes

**Question**: Implement beam search for batch inference. Handle variable-length sequences and beam pruning.

**Implementation**: Code showing beam expansion, scoring, pruning, and termination

---

### Question 8: Token Healing

**Category**: Technical | **Difficulty**: Mid-Senior (L4/L5) | **Time**: 20 minutes

**Question**: Implement token healing to fix tokenization boundary issues. Explain when it's needed.

**Key Points**: Retokenization, boundary handling, performance impact

---

## System Design Questions

### Question 9: High-Performance Serving with Continuous Batching

**Category**: System Design | **Difficulty**: Staff (L6/L7) | **Time**: 60 minutes

**Question**: Design a production serving system using continuous batching. Target: 5000 req/sec, p99 < 1sec.

**Architecture Components**:
- Request queue and prioritization
- Dynamic batch formation
- KV cache management (PagedAttention)
- Preemption for SLA
- Monitoring and auto-scaling

---

### Question 10: Speculative Decoding in Production

**Category**: System Design | **Difficulty**: Senior (L5/L6) | **Time**: 45 minutes

**Question**: Design a production deployment using speculative decoding. How do you manage two models? What's the ROI?

**Considerations**:
- Draft model selection and training
- Memory layout (both models in VRAM)
- Acceptance rate monitoring
- Cost analysis
- Fallback strategy

---

### Question 11: Multi-Turn Conversation Optimization

**Category**: System Design | **Difficulty**: Senior (L5/L6) | **Time**: 40 minutes

**Question**: Design a system optimized for multi-turn conversations. How do you cache conversation history efficiently?

**Techniques**:
- Prefix caching for system prompts
- Conversation state management
- Memory limits and truncation
- Context compression

---

### Question 12: Constrained Generation at Scale

**Category**: System Design | **Difficulty**: Senior (L5/L6) | **Time**: 35 minutes

**Question**: Design a system serving constrained generation (JSON, code) at scale. Handle grammar compilation and caching.

**Architecture**:
- Grammar compilation pipeline
- Pre-compiled grammar cache
- Schema validation
- Performance optimization

---

## Debugging Questions

### Question 13: Speculative Decoding Low Acceptance Rate

**Category**: Debugging | **Difficulty**: Senior (L5/L6) | **Time**: 25 minutes

**Question**: Speculative decoding acceptance rate is only 20% (expected 60%). Debug and fix.

**Investigation**: Draft model quality, temperature mismatch, sampling inconsistency, KV cache corruption

---

### Question 14: Continuous Batching Memory Leak

**Category**: Debugging | **Difficulty**: Senior (L5/L6) | **Time**: 25 minutes

**Question**: Memory grows unbounded in continuous batching system. Find the leak.

**Process**: KV cache not freed, request tracking, generation tracking, profiling

---

### Question 15: Grammar Constraint Not Working

**Category**: Debugging | **Difficulty**: Mid-Senior (L4/L5) | **Time**: 20 minutes

**Question**: Outputs violate JSON schema despite grammar constraints. Debug.

**Issues**: Grammar compilation bugs, tokenizer mismatch, early termination, validation

---

## Summary

**Module 5 Coverage**:
- Speculative decoding algorithms
- Continuous batching strategies
- Advanced sampling methods
- Constrained generation
- Prefix caching
- Beam search
- Production optimizations

**Difficulty Distribution**:
- Mid-Senior: 3 questions
- Senior: 10 questions
- Staff: 2 questions

**Interview Company Alignment**:
- ✅ OpenAI L5-L7
- ✅ Anthropic L5-L7
- ✅ Together AI, Fireworks (all levels)
- ✅ Research-focused companies

---

**Maintained by**: Agent 8 (Integration Coordinator)
**Last Updated**: 2025-11-18
