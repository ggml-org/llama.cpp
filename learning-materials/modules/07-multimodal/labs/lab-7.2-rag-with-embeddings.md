# Lab 7.2: Building RAG with Embeddings

**Estimated Time**: 90-120 minutes
**Difficulty**: Advanced
**Prerequisites**: Module 7.1 complete, understanding of embeddings

## Learning Objectives

- Generate embeddings for semantic search
- Build a complete RAG (Retrieval Augmented Generation) system
- Optimize retrieval quality and performance
- Deploy a production-ready knowledge base system

## Setup

```bash
pip install llama-cpp-python numpy faiss-cpu

# Optional: GPU FAISS
# pip install faiss-gpu
```

**Models Needed**:
- Embedding model: `nomic-embed-text-v1.5.Q4_K_M.gguf` (~110MB)
- LLM: `llama-2-7b-chat.Q4_K_M.gguf` (~4GB)

---

## Part 1: Embedding Generation (30 minutes)

### Task 1: Generate Embeddings

Create a sample knowledge base (`knowledge.txt`):
```
Paris is the capital and most populous city of France.
The Eiffel Tower is a wrought-iron lattice tower in Paris, built in 1889.
Machine learning is a subset of artificial intelligence.
Python is a high-level programming language created by Guido van Rossum.
The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.
```

Generate embeddings:
```bash
python embedding_generator.py \
    --model nomic-embed-text-v1.5.Q4_K_M.gguf \
    --input knowledge.txt \
    --output knowledge_embeddings.npy \
    --cache-dir ./cache \
    --similarity-test
```

**Questions**:
1. What is the embedding dimension?
2. Which documents are most similar to each other?
3. How long does it take to generate embeddings for 100 documents?

### Task 2: Semantic Search

Create `semantic_search.py`:
```python
import numpy as np
from llama_cpp import Llama

# Load embedding model
model = Llama(
    model_path="nomic-embed-text-v1.5.Q4_K_M.gguf",
    embedding=True,
    verbose=False
)

# Load embeddings and documents
embeddings = np.load("knowledge_embeddings.npy")
with open("knowledge.txt") as f:
    documents = [line.strip() for line in f if line.strip()]

def search(query, top_k=3):
    # Generate query embedding
    query_emb = model.create_embedding(query)
    query_vec = np.array(query_emb['data'][0]['embedding'])
    query_vec = query_vec / np.linalg.norm(query_vec)

    # Compute similarities
    similarities = np.dot(embeddings, query_vec)

    # Get top results
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    for i, idx in enumerate(top_indices, 1):
        print(f"\n{i}. Score: {similarities[idx]:.4f}")
        print(f"   {documents[idx]}")

# Test queries
queries = [
    "Tell me about Paris",
    "What is AI?",
    "Programming languages"
]

for query in queries:
    print(f"\nQuery: {query}")
    search(query)
```

**Deliverable**: Test 10 queries and record:
- Top-3 results
- Relevance scores
- Precision@3 (manual evaluation)

---

## Part 2: RAG System Development (40 minutes)

### Task 3: Build Complete RAG

Use provided `simple_rag_system.py`:

```bash
python simple_rag_system.py \
    --embed-model nomic-embed-text-v1.5.Q4_K_M.gguf \
    --llm-model llama-2-7b-chat.Q4_K_M.gguf \
    --documents knowledge.txt \
    --save-index knowledge.index.json \
    --interactive
```

**Test Questions**:
```
Question: What is the capital of France?
Question: When was the Eiffel Tower built?
Question: What is machine learning?
Question: Who created Python?
Question: What is the largest ocean?
```

**Evaluation**:
For each answer, rate:
- Correctness (1-5)
- Completeness (1-5)
- Coherence (1-5)
- Relevance of retrieved docs

### Task 4: Improve Retrieval

Experiment with retrieval parameters:

```python
# Modify simple_rag_system.py

# Test different top_k values
for k in [1, 3, 5, 10]:
    result = rag.query(question, top_k=k)
    evaluate_answer(result)

# Test with context truncation
def truncate_context(docs, max_chars=500):
    return [doc['text'][:max_chars] for doc in docs]

# Test with reranking
def rerank_by_length(docs):
    # Prefer shorter, more focused documents
    return sorted(docs, key=lambda d: len(d['text']))
```

**Findings**: Which configuration works best?

---

## Part 3: Advanced RAG Techniques (30 minutes)

### Challenge 1: Hybrid Search

Implement BM25 + semantic search:

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRAG(SimpleRAG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bm25 = None

    def add_documents(self, documents, **kwargs):
        super().add_documents(documents, **kwargs)

        # Build BM25 index
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def hybrid_retrieve(self, query, top_k=3, alpha=0.5):
        # Semantic search
        semantic_results = self.retrieve(query, top_k=top_k*2)
        semantic_scores = {doc['id']: doc['score'] for doc in semantic_results}

        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize and combine
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        combined_scores = {}

        for doc_id in range(len(self.documents)):
            sem_score = semantic_scores.get(doc_id, 0)
            bm25_score = bm25_scores[doc_id] / max_bm25

            combined_scores[doc_id] = alpha * sem_score + (1-alpha) * bm25_score

        # Get top-k
        top_ids = sorted(combined_scores.keys(),
                        key=lambda x: combined_scores[x],
                        reverse=True)[:top_k]

        return [{'text': self.documents[i]['text'],
                'score': combined_scores[i],
                'id': i} for i in top_ids]
```

**Test**: Does hybrid search improve accuracy?

### Challenge 2: Query Expansion

Expand queries before retrieval:

```python
def expand_query(original_query, llm):
    """Generate related queries"""
    prompt = f"""Generate 3 related search queries for: "{original_query}"

Related queries:
1."""

    response = llm.create_completion(prompt, max_tokens=100)
    # Parse and use multiple queries
    # Aggregate results
```

---

## Part 4: Scalability and Performance (30 minutes)

### Task 5: Large Knowledge Base

Create a larger dataset:
```bash
# Download Wikipedia abstracts or use your own data
# 1,000+ documents

python embedding_generator.py \
    --model nomic-embed-text-v1.5.Q4_K_M.gguf \
    --input large_corpus.txt \
    --output large_embeddings.npz \
    --format npz \
    --cache-dir ./cache
```

**Benchmark**:
- Indexing time
- Index size
- Query latency
- Memory usage

### Task 6: FAISS Optimization

Compare search methods:

```python
import faiss
import numpy as np
import time

# Load embeddings
embeddings = np.load("large_embeddings.npy")
n, d = embeddings.shape

# 1. Flat (exact) search
index_flat = faiss.IndexFlatIP(d)
index_flat.add(embeddings)

# 2. IVF (approximate) search
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatIP(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
index_ivf.train(embeddings)
index_ivf.add(embeddings)

# 3. HNSW (graph-based)
index_hnsw = faiss.IndexHNSWFlat(d, 32)
index_hnsw.add(embeddings)

# Benchmark
query = embeddings[0:1]  # Use first embedding as query

for name, index in [("Flat", index_flat),
                     ("IVF", index_ivf),
                     ("HNSW", index_hnsw)]:
    times = []
    for _ in range(100):
        start = time.time()
        D, I = index.search(query, k=10)
        times.append(time.time() - start)

    print(f"{name:10} - Avg: {np.mean(times)*1000:.2f}ms, "
          f"P99: {np.percentile(times, 99)*1000:.2f}ms")
```

**Results Table**:
| Method | Avg Latency | P99 Latency | Memory | Accuracy |
|--------|-------------|-------------|--------|----------|
| Flat   |             |             |        | 100%     |
| IVF    |             |             |        |          |
| HNSW   |             |             |        |          |

---

## Part 5: Production Deployment (20 minutes)

### Task 7: REST API

Create `rag_api.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from simple_rag_system import SimpleRAG

app = FastAPI()

# Initialize RAG (load once at startup)
rag = SimpleRAG(
    embedding_model_path="nomic-embed.gguf",
    llm_model_path="llama-2-7b-chat.gguf"
)
rag.load_index("knowledge.index.json")

class Query(BaseModel):
    question: str
    top_k: int = 3
    max_tokens: int = 512

@app.post("/query")
async def query_rag(query: Query):
    try:
        result = rag.query(
            query.question,
            top_k=query.top_k,
            max_tokens=query.max_tokens,
            show_context=True
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "num_documents": len(rag.documents)}

# Run with: uvicorn rag_api:app --reload
```

Test:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Python?", "top_k": 3}'
```

---

## Deliverables

1. **Embedding Analysis** (Part 1):
   - Similarity matrix visualization
   - Query performance report

2. **RAG Evaluation** (Part 2):
   - Test results for 10+ queries
   - Configuration comparison

3. **Advanced Techniques** (Part 3):
   - Hybrid search implementation
   - Performance comparison

4. **Scalability Report** (Part 4):
   - Benchmarks for different index sizes
   - FAISS optimization results

5. **API Demo** (Part 5):
   - Working REST API
   - Client examples
   - Documentation

---

## Bonus Challenges

### Challenge 1: Multi-Query RAG
Retrieve with multiple queries and aggregate:
```python
def multi_query_rag(questions, aggregation='union'):
    all_docs = []
    for q in questions:
        docs = rag.retrieve(q, top_k=5)
        all_docs.extend(docs)

    # Remove duplicates and re-rank
    # ...
```

### Challenge 2: Contextual Compression
Compress retrieved documents to fit more context:
```python
def compress_context(docs, query, compression_llm):
    """Extract only query-relevant parts of documents"""
    compressed = []
    for doc in docs:
        summary = compression_llm.summarize(doc, focus=query)
        compressed.append(summary)
    return compressed
```

### Challenge 3: Conversational RAG
Maintain conversation history:
```python
class ConversationalRAG:
    def __init__(self, rag):
        self.rag = rag
        self.history = []

    def chat(self, message):
        # Include history in retrieval
        # Update history
        # Generate response
```

---

## Reflection

1. How does embedding quality affect RAG performance?
2. What's the right balance between retrieval quantity and quality?
3. When should you use hybrid search vs. pure semantic search?
4. What are the main challenges in production RAG deployment?

---

**Lab Complete!** You've built a complete RAG system with llama.cpp.
