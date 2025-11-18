# Embedding Models in LLaMA.cpp

## Table of Contents
1. [Introduction to Embeddings](#introduction-to-embeddings)
2. [Embedding Model Architectures](#embedding-model-architectures)
3. [Supported Models](#supported-models)
4. [Generating Embeddings](#generating-embeddings)
5. [Pooling Strategies](#pooling-strategies)
6. [RAG System Design](#rag-system-design)
7. [Vector Databases](#vector-databases)
8. [Performance Optimization](#performance-optimization)
9. [Evaluation and Quality](#evaluation-and-quality)
10. [Production Best Practices](#production-best-practices)

---

## Introduction to Embeddings

### What are Embeddings?

Embeddings are dense vector representations of text that capture semantic meaning in a continuous space. Unlike sparse representations (like bag-of-words), embeddings:

- **Capture Semantics**: Similar meanings → similar vectors
- **Dense Representation**: Typically 384-1024 dimensions (vs. vocab size)
- **Continuous Space**: Enables mathematical operations (similarity, arithmetic)
- **Transfer Learning**: Trained on large corpora, work across domains

### Why Use Embeddings?

```python
# Traditional keyword matching
query = "AI safety research"
doc1 = "Research on artificial intelligence alignment"  # ❌ Miss (no exact match)
doc2 = "AI safety best practices guide"                 # ✅ Match ("AI safety")

# Embedding-based semantic search
# Both doc1 and doc2 would match with high similarity! ✅
```

**Key Applications**:
1. **Semantic Search**: Find relevant documents by meaning
2. **RAG (Retrieval Augmented Generation)**: Provide context to LLMs
3. **Clustering**: Group similar texts
4. **Recommendation**: Find similar items
5. **Duplicate Detection**: Identify near-duplicate content
6. **Classification**: Embeddings as features

### Embedding Space Visualization

```
High-dimensional space (e.g., 768D) - Conceptual 2D projection:

    "dog" •
          ↘
    "puppy" •    • "cat"
              ↘ ↙
            • "pet"
               |
    "computer" • ← Far from animal cluster
```

---

## Embedding Model Architectures

### Sentence-BERT (SBERT) Architecture

SBERT modifies BERT for efficient sentence embeddings:

```
Input: "Artificial intelligence is transforming society"
         ↓
    [Tokenization]
         ↓
    [BERT Encoder] (12 or 24 layers)
         ↓
    [CLS] [Token 1] [Token 2] ... [Token N] [SEP]
     ↓        ↓          ↓            ↓
    [Pooling Strategy - Mean/CLS/Max]
         ↓
    Final Embedding (768D)
```

**Key Differences from BERT**:
- **Training**: Siamese/triplet networks for similarity
- **Pooling**: Mean pooling instead of just [CLS] token
- **Efficiency**: Single forward pass (vs. pairwise in BERT)

### Contrastive Learning

Modern embedding models use contrastive learning:

```python
# Training Objective
anchor = "The cat sat on the mat"
positive = "A cat is sitting on a mat"      # Similar meaning
negative = "Python programming tutorial"     # Different meaning

# Goal: Make anchor and positive close, anchor and negative far
loss = contrastive_loss(
    similarity(embed(anchor), embed(positive)),  # Maximize
    similarity(embed(anchor), embed(negative))   # Minimize
)
```

**Training Data**:
- **Supervised**: Labeled pairs (question-answer, paraphrase)
- **Weakly Supervised**: Web data (title-body, adjacent sentences)
- **Unsupervised**: Contrastive learning with augmentation

---

## Supported Models

llama.cpp supports various embedding models through its unified API:

### 1. Nomic-Embed-Text

**Overview**: High-performance open embedding model

```yaml
Model: nomic-embed-text-v1.5
Dimensions: 768
Context Length: 8192
License: Apache 2.0
Size: ~270M parameters
```

**Strengths**:
- Long context support (8K tokens)
- Strong performance on benchmarks
- Fully open-source and reproducible
- Good multilingual support

**Use Cases**:
- Document retrieval
- Long-form content
- Research and academic applications

**Download**:
```bash
# From Nomic
wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf
```

### 2. BGE (BAAI General Embedding)

**Overview**: State-of-the-art Chinese and English embeddings

```yaml
Models:
  - bge-small-en-v1.5: 384D, 33M params
  - bge-base-en-v1.5: 768D, 109M params
  - bge-large-en-v1.5: 1024D, 335M params
  - bge-m3: Multilingual, 1024D
```

**Strengths**:
- Top MTEB benchmark scores
- Excellent for production
- Multilingual variants
- Different size options

**Special Features**:
```python
# BGE uses instruction prefix for queries
query_prefix = "Represent this sentence for searching relevant passages: "
query_embedding = model.embed(query_prefix + query)

# No prefix for documents
doc_embedding = model.embed(document)
```

**Use Cases**:
- Production semantic search
- Multilingual applications
- When you need SOTA performance

### 3. E5 (Text Embeddings by Weakly-Supervised Contrastive Pre-training)

**Overview**: Strong baseline embeddings

```yaml
Models:
  - e5-small-v2: 384D
  - e5-base-v2: 768D
  - e5-large-v2: 1024D
  - multilingual-e5: Multilingual
```

**Strengths**:
- Well-balanced performance/efficiency
- Strong transfer learning
- Good for general purpose

**Use Cases**:
- General-purpose embeddings
- When you need reliable baseline
- Resource-constrained scenarios

### 4. all-MiniLM

**Overview**: Compact and fast embedding model

```yaml
Model: all-MiniLM-L6-v2
Dimensions: 384
Parameters: 22M
Speed: Very fast
```

**Strengths**:
- Very small and fast
- Good quality for size
- Low memory footprint

**Use Cases**:
- Edge deployment
- Real-time applications
- Prototyping

### Model Comparison

| Model | Dim | Params | MTEB Score | Speed | Best For |
|-------|-----|--------|------------|-------|----------|
| nomic-embed-text | 768 | 270M | 62.4 | Medium | Long context |
| bge-large-en | 1024 | 335M | 63.9 | Slow | Quality |
| bge-base-en | 768 | 109M | 63.0 | Medium | Balanced |
| bge-small-en | 384 | 33M | 62.2 | Fast | Efficiency |
| e5-large | 1024 | 335M | 62.3 | Slow | General |
| all-MiniLM-L6 | 384 | 22M | 58.8 | Very Fast | Speed |

*MTEB = Massive Text Embedding Benchmark*

---

## Generating Embeddings

### Using llama.cpp C++ API

```cpp
#include "llama.h"

// Initialize model for embeddings
llama_model_params model_params = llama_model_default_params();
model_params.use_mmap = true;
llama_model *model = llama_load_model_from_file("nomic-embed.gguf", model_params);

llama_context_params ctx_params = llama_context_default_params();
ctx_params.embeddings = true;  // Enable embedding mode
ctx_params.n_ctx = 512;
llama_context *ctx = llama_new_context_with_model(model, ctx_params);

// Tokenize input
std::string text = "This is a test document";
std::vector<llama_token> tokens = llama_tokenize(model, text, true);

// Create batch
llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

// Decode to get embeddings
llama_decode(ctx, batch);

// Get embeddings
int n_embd = llama_n_embd(model);
float *embeddings = llama_get_embeddings(ctx);

// Normalize (important for cosine similarity)
float norm = 0.0f;
for (int i = 0; i < n_embd; i++) {
    norm += embeddings[i] * embeddings[i];
}
norm = sqrtf(norm);
for (int i = 0; i < n_embd; i++) {
    embeddings[i] /= norm;
}
```

### Using Python API

```python
from llama_cpp import Llama

# Load model with embedding support
model = Llama(
    model_path="nomic-embed-text-v1.5.Q4_K_M.gguf",
    embedding=True,  # Enable embedding mode
    n_ctx=512,
    n_batch=512,
    verbose=False
)

def get_embedding(text: str) -> list[float]:
    """
    Generate normalized embedding for text
    """
    # Create embedding
    embedding = model.create_embedding(text)

    # Extract vector
    vector = embedding['data'][0]['embedding']

    # Normalize (if not already normalized)
    import numpy as np
    vector = np.array(vector)
    vector = vector / np.linalg.norm(vector)

    return vector.tolist()

# Example usage
query_emb = get_embedding("What is machine learning?")
doc1_emb = get_embedding("Machine learning is a subset of AI")
doc2_emb = get_embedding("The weather is sunny today")

# Compute similarity
similarity_doc1 = cosine_similarity(query_emb, doc1_emb)  # High
similarity_doc2 = cosine_similarity(query_emb, doc2_emb)  # Low

print(f"Query-Doc1 similarity: {similarity_doc1:.4f}")  # ~0.85
print(f"Query-Doc2 similarity: {similarity_doc2:.4f}")  # ~0.15
```

### Cosine Similarity

```python
import numpy as np

def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Calculate cosine similarity between two vectors
    Returns value in [-1, 1], typically [0, 1] for embeddings
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # If already normalized, just dot product
    # Otherwise:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return dot_product / (norm1 * norm2)

# For batch computation
def batch_cosine_similarity(query_vec, doc_vecs):
    """
    Compute similarity between query and multiple documents
    """
    query_vec = np.array(query_vec)
    doc_vecs = np.array(doc_vecs)

    # Assuming normalized vectors
    similarities = np.dot(doc_vecs, query_vec)

    return similarities
```

---

## Pooling Strategies

Different ways to aggregate token embeddings into a single vector:

### 1. Mean Pooling (Most Common)

```python
def mean_pooling(token_embeddings, attention_mask):
    """
    Average all token embeddings, weighted by attention mask
    """
    # token_embeddings: [seq_len, hidden_dim]
    # attention_mask: [seq_len] (1 for real tokens, 0 for padding)

    # Expand mask
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())

    # Sum embeddings
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=0)

    # Sum mask
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=0), min=1e-9)

    # Average
    return sum_embeddings / sum_mask
```

**Pros**: Captures full sequence information
**Cons**: May dilute important information
**Best for**: General-purpose embeddings

### 2. CLS Token Pooling

```python
def cls_pooling(token_embeddings):
    """
    Use only the [CLS] token embedding
    """
    # token_embeddings: [seq_len, hidden_dim]
    return token_embeddings[0]  # First token is [CLS]
```

**Pros**: Simple, designed for this purpose in BERT
**Cons**: May miss information from rest of sequence
**Best for**: BERT-family models trained with this

### 3. Max Pooling

```python
def max_pooling(token_embeddings, attention_mask):
    """
    Take maximum value across tokens for each dimension
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())

    # Set padding tokens to large negative value
    token_embeddings[input_mask_expanded == 0] = -1e9

    # Max across sequence dimension
    return torch.max(token_embeddings, dim=0)[0]
```

**Pros**: Captures salient features
**Cons**: Can be noisy, loses context
**Best for**: Specific feature extraction tasks

### 4. Weighted Mean (Attention-Based)

```python
def attention_pooling(token_embeddings, attention_scores):
    """
    Weight tokens by attention scores before averaging
    """
    # attention_scores: [seq_len] from attention mechanism
    weights = F.softmax(attention_scores, dim=0).unsqueeze(-1)
    weighted_embeddings = token_embeddings * weights
    return torch.sum(weighted_embeddings, dim=0)
```

**Pros**: Focuses on important tokens
**Cons**: More complex, requires attention weights
**Best for**: When attention information is available

### Pooling in llama.cpp

```cpp
// llama.cpp uses mean pooling by default for embeddings
// Set in context params:
ctx_params.embeddings = true;
ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;  // Default

// Other options:
// LLAMA_POOLING_TYPE_NONE  - No pooling, get all token embeddings
// LLAMA_POOLING_TYPE_CLS   - Use first token
// LLAMA_POOLING_TYPE_LAST  - Use last token
// LLAMA_POOLING_TYPE_MEAN  - Mean pooling (recommended)
```

---

## RAG System Design

### What is RAG?

**Retrieval Augmented Generation** enhances LLM responses with external knowledge:

```
User Query
    ↓
[Generate Embedding]
    ↓
[Search Vector Database] → Find top-k relevant documents
    ↓
[Retrieve Documents]
    ↓
[Construct Prompt] ← User Query + Retrieved Context
    ↓
[LLM Generation]
    ↓
Grounded Response
```

### Complete RAG Pipeline

```python
class RAGSystem:
    def __init__(self, embedding_model_path, llm_model_path):
        # Initialize embedding model
        self.embed_model = Llama(
            model_path=embedding_model_path,
            embedding=True,
            n_ctx=512
        )

        # Initialize LLM
        self.llm = Llama(
            model_path=llm_model_path,
            n_ctx=4096,
            n_gpu_layers=35
        )

        # Vector store (in-memory for simplicity)
        self.documents = []
        self.embeddings = []

    def add_documents(self, documents: list[str]):
        """
        Index documents by generating embeddings
        """
        for doc in documents:
            # Generate embedding
            emb = self.embed_model.create_embedding(doc)
            vector = emb['data'][0]['embedding']

            # Store
            self.documents.append(doc)
            self.embeddings.append(vector)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """
        Retrieve most relevant documents
        """
        # Generate query embedding
        query_emb = self.embed_model.create_embedding(query)
        query_vector = np.array(query_emb['data'][0]['embedding'])

        # Compute similarities
        doc_vectors = np.array(self.embeddings)
        similarities = np.dot(doc_vectors, query_vector)

        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

    def generate(self, query: str, top_k: int = 3) -> str:
        """
        RAG: Retrieve + Generate
        """
        # Retrieve relevant documents
        relevant_docs = self.retrieve(query, top_k)

        # Construct prompt
        context = "\n\n".join([f"Document {i+1}: {doc}"
                               for i, doc in enumerate(relevant_docs)])

        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""

        # Generate response
        response = self.llm.create_completion(
            prompt,
            max_tokens=512,
            temperature=0.7,
            stop=["Question:", "\n\n\n"]
        )

        return response['choices'][0]['text'].strip()

# Usage
rag = RAGSystem(
    embedding_model_path="nomic-embed.gguf",
    llm_model_path="llama-2-7b.gguf"
)

# Index knowledge base
documents = [
    "Paris is the capital of France and its largest city.",
    "The Eiffel Tower is located in Paris and was built in 1889.",
    "French cuisine is renowned worldwide for its quality.",
]
rag.add_documents(documents)

# Query
answer = rag.generate("What is the Eiffel Tower?")
print(answer)
# Output: "The Eiffel Tower is a landmark located in Paris, France.
#          It was constructed in 1889..."
```

### Advanced RAG Techniques

#### 1. Hybrid Search

```python
def hybrid_search(query, documents, alpha=0.5):
    """
    Combine semantic search with keyword search
    """
    # Semantic scores
    semantic_scores = semantic_search(query, documents)

    # Keyword scores (BM25)
    keyword_scores = bm25_search(query, documents)

    # Combine
    final_scores = alpha * semantic_scores + (1 - alpha) * keyword_scores

    return final_scores
```

#### 2. Reranking

```python
def rerank_results(query, initial_results, reranker_model, top_k=3):
    """
    Rerank initial results with a more powerful model
    """
    # Get detailed scores for top candidates
    scores = []
    for doc in initial_results[:10]:  # Rerank top 10
        score = reranker_model.score(query, doc)
        scores.append(score)

    # Select final top-k
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [initial_results[i] for i in top_indices]
```

#### 3. Chunking Strategies

```python
def chunk_document(text, chunk_size=512, overlap=50):
    """
    Split document into overlapping chunks
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

# Usage
large_doc = "..." # 10,000 word document
chunks = chunk_document(large_doc, chunk_size=256, overlap=32)
rag.add_documents(chunks)  # Index chunks instead of full doc
```

---

## Vector Databases

For production RAG, use dedicated vector databases:

### ChromaDB

```python
import chromadb
from chromadb.config import Settings

# Initialize
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

collection = client.create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"}
)

# Add documents
collection.add(
    embeddings=embeddings_list,
    documents=documents_list,
    ids=[f"doc_{i}" for i in range(len(documents_list))]
)

# Query
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)
```

### FAISS

```python
import faiss
import numpy as np

# Create index
dimension = 768
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine for normalized)

# Add vectors
vectors = np.array(embeddings_list).astype('float32')
faiss.normalize_L2(vectors)  # Normalize for cosine similarity
index.add(vectors)

# Search
query_vector = np.array([query_embedding]).astype('float32')
faiss.normalize_L2(query_vector)
distances, indices = index.search(query_vector, k=5)

# Get top documents
top_docs = [documents_list[i] for i in indices[0]]
```

### Comparison

| Database | Best For | Pros | Cons |
|----------|----------|------|------|
| ChromaDB | Small-medium scale | Easy to use, persistent | Not for huge scale |
| FAISS | Large scale | Very fast, scalable | In-memory, complex |
| Qdrant | Production | Full-featured, filtering | Requires server |
| Weaviate | Enterprise | Comprehensive, cloud | Heavy, complex |
| Pinecone | Cloud | Managed, scalable | Costly, vendor lock-in |

---

## Performance Optimization

### Batch Embedding Generation

```python
def batch_embed(texts: list[str], model, batch_size=32):
    """
    Generate embeddings in batches for efficiency
    """
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Generate embeddings for batch
        batch_embeddings = [
            model.create_embedding(text)['data'][0]['embedding']
            for text in batch
        ]

        embeddings.extend(batch_embeddings)

    return embeddings
```

### Caching

```python
from functools import lru_cache
import hashlib

class CachedEmbedder:
    def __init__(self, model):
        self.model = model
        self.cache = {}

    def embed(self, text: str):
        # Hash text for cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash in self.cache:
            return self.cache[text_hash]

        # Generate embedding
        emb = self.model.create_embedding(text)
        vector = emb['data'][0]['embedding']

        # Cache
        self.cache[text_hash] = vector
        return vector
```

### Quantization for Embeddings

```python
def quantize_embeddings(embeddings, bits=8):
    """
    Quantize embeddings to reduce storage (768 floats → 768 bytes)
    """
    embeddings = np.array(embeddings)

    # Find min/max
    min_val = embeddings.min()
    max_val = embeddings.max()

    # Quantize to uint8
    scale = (max_val - min_val) / (2**bits - 1)
    quantized = ((embeddings - min_val) / scale).astype(np.uint8)

    return quantized, min_val, scale

def dequantize_embeddings(quantized, min_val, scale):
    """
    Restore quantized embeddings
    """
    return quantized.astype(np.float32) * scale + min_val

# Usage - 75% storage reduction!
q_embs, min_val, scale = quantize_embeddings(embeddings)
# ... store and retrieve ...
original_embs = dequantize_embeddings(q_embs, min_val, scale)
```

### Approximate Nearest Neighbors

```python
import faiss

# Use IVF index for faster search on large datasets
dimension = 768
nlist = 100  # Number of clusters

# Create IVF index
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# Train index
index.train(training_vectors)
index.add(all_vectors)

# Search with speed/accuracy tradeoff
index.nprobe = 10  # Search 10 clusters (vs all 100)
distances, indices = index.search(query_vector, k=5)
```

---

## Evaluation and Quality

### Embedding Quality Metrics

#### 1. Retrieval Accuracy

```python
def evaluate_retrieval(queries, relevant_docs, rag_system, k=5):
    """
    Measure retrieval accuracy
    """
    metrics = {
        'precision_at_k': [],
        'recall_at_k': [],
        'mrr': []  # Mean Reciprocal Rank
    }

    for query, relevant in zip(queries, relevant_docs):
        # Retrieve
        retrieved = rag_system.retrieve(query, top_k=k)

        # Precision@K
        relevant_retrieved = set(retrieved) & set(relevant)
        precision = len(relevant_retrieved) / k
        metrics['precision_at_k'].append(precision)

        # Recall@K
        recall = len(relevant_retrieved) / len(relevant)
        metrics['recall_at_k'].append(recall)

        # MRR
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                metrics['mrr'].append(1.0 / (i + 1))
                break
        else:
            metrics['mrr'].append(0.0)

    return {k: np.mean(v) for k, v in metrics.items()}
```

#### 2. Embedding Similarity Distribution

```python
def analyze_embedding_quality(model, test_pairs):
    """
    Analyze embedding quality with positive/negative pairs
    """
    positive_sims = []
    negative_sims = []

    for text1, text2, is_similar in test_pairs:
        emb1 = model.embed(text1)
        emb2 = model.embed(text2)
        sim = cosine_similarity(emb1, emb2)

        if is_similar:
            positive_sims.append(sim)
        else:
            negative_sims.append(sim)

    print(f"Positive pairs: {np.mean(positive_sims):.3f} ± {np.std(positive_sims):.3f}")
    print(f"Negative pairs: {np.mean(negative_sims):.3f} ± {np.std(negative_sims):.3f}")
    print(f"Separation: {np.mean(positive_sims) - np.mean(negative_sims):.3f}")
```

---

## Production Best Practices

### 1. Model Selection

```python
# Choose based on requirements
def select_embedding_model(requirements):
    if requirements.latency == "ultra_low" and requirements.scale == "edge":
        return "all-MiniLM-L6-v2"  # 384D, 22M params

    elif requirements.quality == "sota" and requirements.resources == "high":
        return "bge-large-en-v1.5"  # 1024D, best quality

    elif requirements.context_length > 2048:
        return "nomic-embed-text-v1.5"  # 8K context

    else:
        return "bge-base-en-v1.5"  # Balanced default
```

### 2. Error Handling

```python
def robust_embed(text, model, max_retries=3):
    """
    Robust embedding with retries and fallbacks
    """
    # Truncate if too long
    max_length = 512  # Model dependent
    if len(text.split()) > max_length:
        text = ' '.join(text.split()[:max_length])

    for attempt in range(max_retries):
        try:
            embedding = model.create_embedding(text)
            return embedding['data'][0]['embedding']
        except Exception as e:
            if attempt == max_retries - 1:
                # Return zero vector as fallback
                return [0.0] * model.n_embd()
            time.sleep(0.1 * (attempt + 1))
```

### 3. Monitoring

```python
import time
from collections import deque

class EmbeddingMonitor:
    def __init__(self, window_size=1000):
        self.latencies = deque(maxlen=window_size)
        self.errors = 0
        self.total_requests = 0

    def record_request(self, latency, success=True):
        self.total_requests += 1
        self.latencies.append(latency)
        if not success:
            self.errors += 1

    def get_stats(self):
        return {
            'avg_latency_ms': np.mean(self.latencies) * 1000,
            'p95_latency_ms': np.percentile(self.latencies, 95) * 1000,
            'p99_latency_ms': np.percentile(self.latencies, 99) * 1000,
            'error_rate': self.errors / max(self.total_requests, 1),
            'throughput': len(self.latencies) / sum(self.latencies)
        }
```

---

## Summary

Embedding models in llama.cpp enable powerful semantic search and RAG applications:

✅ **Models**: Nomic-Embed, BGE, E5, all-MiniLM
✅ **Generation**: Efficient C++ and Python APIs
✅ **Pooling**: Mean, CLS, max pooling strategies
✅ **RAG**: Complete retrieval-augmented generation pipeline
✅ **Optimization**: Batching, caching, quantization, ANN search
✅ **Production**: Vector databases, monitoring, error handling

**Next Steps**:
- Complete Lab 7.2 to build a RAG system
- Experiment with different embedding models
- Integrate with vector databases
- Explore Lesson 7.3 on audio integration

---

**References**:
- Reimers & Gurevych (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- Xiao et al. (2023). "C-Pack: Packaged Resources To Advance General Chinese Embedding" (BGE)
- Nussbaum et al. (2024). "Nomic Embed: Training a Reproducible Long Context Text Embedder"
- Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
