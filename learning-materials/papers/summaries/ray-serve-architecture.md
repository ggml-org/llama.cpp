# Ray Serve Architecture for LLM Deployment

**Module**: 6 - Production Serving | **Impact**: ⭐⭐⭐⭐

---

## Summary

Ray Serve provides distributed model serving with autoscaling, load balancing, and multi-model orchestration. See serving-systems-comparison.md for positioning vs other systems.

**Key Features**: Python-native, distributed, autoscaling, composable deployments

---

## Core Concepts

### 1. Deployments

```python
from ray import serve

@serve.deployment(num_replicas=4, ray_actor_options={"num_gpus": 0.25})
class LLMModel:
    def __init__(self, model_path):
        from transformers import pipeline
        self.model = pipeline("text-generation", model=model_path)

    def __call__(self, request):
        prompt = request.query_params["prompt"]
        return self.model(prompt, max_length=100)

serve.run(LLMModel.bind("gpt2"))
```

### 2. Autoscaling

```python
@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 5,
    }
)
class AutoScalingLLM:
    # Automatically scales based on load
    pass
```

### 3. Model Composition (RAG Example)

```python
@serve.deployment
class Retriever:
    def retrieve(self, query):
        # Vector DB search
        return relevant_docs

@serve.deployment
class Generator:
    def __init__(self):
        self.llm = LLM("llama-2-7b")

    def generate(self, context, query):
        prompt = f"Context: {context}\nQuery: {query}"
        return self.llm.generate(prompt)

@serve.deployment
class RAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    async def __call__(self, query):
        docs = await self.retriever.retrieve.remote(query)
        response = await self.generator.generate.remote(docs, query)
        return response

# Compose pipeline
retriever = Retriever.bind()
generator = Generator.bind()
app = RAGPipeline.bind(retriever, generator)
serve.run(app)
```

---

## Use Cases

✅ **Multi-model systems**: Serve multiple models behind single API
✅ **Complex pipelines**: RAG, ensemble, cascade
✅ **Autoscaling**: Handle variable load
✅ **Python ecosystem**: Easy integration with existing code

---

**Status**: Complete | Module 6 Complete (2/2) papers
