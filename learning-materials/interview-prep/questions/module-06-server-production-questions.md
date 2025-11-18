# Module 6: Server & Production - Interview Questions

**Purpose**: Interview preparation for production deployment and server implementation
**Target Level**: Senior to Staff Engineers
**Module Coverage**: Module 6 - HTTP Servers, APIs, Monitoring, Deployment
**Question Count**: 20 (5 per category)
**Last Updated**: 2025-11-18
**Created By**: Agent 8 (Integration Coordinator)

---

## Conceptual Questions (5)

### Question 1: REST API Design for LLM Serving
**Difficulty**: Mid-Senior (L4/L5) | **Time**: 20 minutes

**Question**: Design a REST API for llama.cpp inference. What endpoints would you expose? How do you handle streaming?

**Key Endpoints**:
- POST /v1/completions
- POST /v1/chat/completions
- GET /v1/models
- POST /v1/embeddings
- Server-Sent Events for streaming

### Question 2: Load Balancing Strategies
**Difficulty**: Senior (L5/L6) | **Time**: 20 minutes

**Question**: Compare different load balancing strategies for LLM serving: round-robin, least-connections, queue-depth-based.

### Question 3: Monitoring and Observability
**Difficulty**: Senior (L5/L6) | **Time**: 20 minutes

**Question**: What metrics would you track for a production LLM server? How would you set up alerting?

**Metrics**: Latency (p50/p90/p99), throughput, GPU utilization, queue depth, error rate, token/sec

### Question 4: Rate Limiting and Abuse Prevention
**Difficulty**: Mid-Senior (L4/L5) | **Time**: 15 minutes

**Question**: Design a rate limiting system for an LLM API. How do you prevent abuse while maintaining good UX?

### Question 5: Cost Attribution and Tracking
**Difficulty**: Senior (L5/L6) | **Time**: 20 minutes

**Question**: Design a system to track and attribute costs per user/request. How do you measure cost accurately?

---

## Technical Questions (5)

### Question 6: Implementing OpenAI-Compatible Server
**Difficulty**: Senior (L5/L6) | **Time**: 45 minutes

**Question**: Implement an OpenAI-compatible API server using llama.cpp. Support streaming and function calling.

**Code Required**: FastAPI/Flask server with proper error handling

### Question 7: Graceful Shutdown and Request Draining
**Difficulty**: Mid-Senior (L4/L5) | **Time**: 25 minutes

**Question**: Implement graceful shutdown that drains in-flight requests. Handle SIGTERM properly.

### Question 8: Health Checks and Readiness Probes
**Difficulty**: Mid (L4) | **Time**: 20 minutes

**Question**: Implement health check endpoints for Kubernetes. What checks would you include?

### Question 9: Concurrent Request Handling
**Difficulty**: Senior (L5/L6) | **Time**: 30 minutes

**Question**: Implement thread-safe request handling with connection pooling and timeout management.

### Question 10: Metrics Instrumentation
**Difficulty**: Mid-Senior (L4/L5) | **Time**: 25 minutes

**Question**: Instrument llama.cpp server with Prometheus metrics. Implement custom collectors.

---

## System Design Questions (5)

### Question 11: Global Multi-Region Deployment
**Difficulty**: Staff (L6/L7) | **Time**: 60 minutes

**Question**: Design a global LLM serving infrastructure across 3 regions. Handle routing, replication, and failover.

**Components**: CDN, regional clusters, model replication, geo-routing, disaster recovery

### Question 12: Blue-Green Deployment for Model Updates
**Difficulty**: Senior (L5/L6) | **Time**: 45 minutes

**Question**: Design a zero-downtime deployment strategy for model updates. How do you verify new model quality?

**Strategy**: Blue-green, canary deployment, traffic splitting, automated rollback

### Question 13: Auto-Scaling Architecture
**Difficulty**: Senior (L5/L6) | **Time**: 45 minutes

**Question**: Design an auto-scaling system that handles traffic spikes (10x) within 2 minutes.

**Considerations**: Cold start time, warmup, predictive scaling, cost optimization

### Question 14: Multi-Tenancy and Isolation
**Difficulty**: Senior (L5/L6) | **Time**: 40 minutes

**Question**: Design a multi-tenant serving platform with resource isolation and fair scheduling.

### Question 15: Disaster Recovery Plan
**Difficulty**: Staff (L6/L7) | **Time**: 45 minutes

**Question**: Design disaster recovery for LLM serving. Target: RPO < 1 hour, RTO < 15 minutes.

---

## Debugging Questions (5)

### Question 16: Intermittent 504 Timeouts
**Difficulty**: Senior (L5/L6) | **Time**: 25 minutes

**Question**: Users report occasional 504 timeouts (1% of requests). Debug and fix.

**Investigation**: Queue buildup, GC pauses, network issues, resource exhaustion

### Question 17: Memory Leak in Long-Running Server
**Difficulty**: Senior (L5/L6) | **Time**: 25 minutes

**Question**: Server memory grows from 8GB to 32GB over 24 hours. Find the leak.

### Question 18: Uneven Load Distribution
**Difficulty**: Mid-Senior (L4/L5) | **Time**: 20 minutes

**Question**: Load balancer sends 80% traffic to one server. Why? Fix it.

### Question 19: Prometheus Metrics Not Updating
**Difficulty**: Mid (L3/L4) | **Time**: 15 minutes

**Question**: Metrics endpoint returns stale data. Debug.

### Question 20: WebSocket Connection Drops
**Difficulty**: Mid-Senior (L4/L5) | **Time**: 20 minutes

**Question**: Streaming responses disconnect after 30 seconds. Fix persistent connection handling.

---

## Summary

**Module 6 Coverage**:
- REST API design and implementation
- Production server architecture
- Monitoring and observability
- Deployment strategies
- Rate limiting and security
- Multi-region deployment
- Auto-scaling
- Debugging production issues

**Difficulty Distribution**:
- Mid: 2 questions
- Mid-Senior: 6 questions
- Senior: 10 questions
- Staff: 2 questions

**Interview Company Alignment**:
- ✅ OpenAI L4-L7
- ✅ Anthropic L4-L7
- ✅ Cloud providers (AWS, GCP, Azure)
- ✅ Startups (all levels)

---

**Maintained by**: Agent 8 (Integration Coordinator)
**Last Updated**: 2025-11-18
