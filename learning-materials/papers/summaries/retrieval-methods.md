# Advanced Retrieval Methods for RAG Systems

**Module**: 8 - RAG and Knowledge Systems | **Impact**: ⭐⭐⭐⭐

---

## Summary

Deep dive into retrieval techniques beyond basic similarity search. Covers dense retrieval, sparse retrieval, hybrid approaches, and reranking.

---

## 1. Dense Retrieval (Embedding-based)

### Bi-Encoder Architecture
```python
# Separate encoders for query and documents
query_embedding = query_encoder(query)  # [768]
doc_embeddings = doc_encoder(documents)  # [N, 768]

# Cosine similarity
similarities = cosine_similarity(query_embedding, doc_embeddings)
top_k = argsort(similarities)[:k]
```

**Pros**: Fast at scale (pre-compute doc embeddings)
**Cons**: May miss keyword matches

---

## 2. Sparse Retrieval (BM25)

```python
from rank_bm25 import BM25Okapi

# Traditional keyword search
corpus = [doc.split() for doc in documents]
bm25 = BM25Okapi(corpus)

query_tokens = query.split()
scores = bm25.get_scores(query_tokens)
top_k = argsort(scores)[:k]
```

**Pros**: Exact keyword matching, no ML required
**Cons**: Doesn't understand semantics

---

## 3. Hybrid Retrieval

```python
def hybrid_search(query, dense_retriever, sparse_retriever, alpha=0.5):
    # Dense retrieval
    dense_scores = dense_retriever.search(query)

    # Sparse retrieval
    sparse_scores = sparse_retriever.search(query)

    # Combine (normalized)
    dense_normalized = normalize(dense_scores)
    sparse_normalized = normalize(sparse_scores)

    final_scores = alpha * dense_normalized + (1 - alpha) * sparse_normalized
    return final_scores
```

**Best of both worlds**: Semantic + keyword matching

---

## 4. Cross-Encoder Reranking

```python
from sentence_transformers import CrossEncoder

# After initial retrieval, rerank with cross-encoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

initial_docs = retrieve_top_100(query)  # Fast bi-encoder

# Rerank (slower but more accurate)
pairs = [[query, doc] for doc in initial_docs]
scores = cross_encoder.predict(pairs)
reranked = [doc for _, doc in sorted(zip(scores, initial_docs), reverse=True)]

return reranked[:10]  # Return top 10 after reranking
```

**Pattern**: Bi-encoder (fast, retrieve 100) → Cross-encoder (accurate, rerank to 10)

---

## 5. ColBERT (Late Interaction)

```python
# Token-level matching (between bi-encoder and cross-encoder)

# Query tokens: [Q1, Q2, Q3] → embeddings [e_Q1, e_Q2, e_Q3]
# Doc tokens: [D1, D2, ..., D10] → embeddings [e_D1, ..., e_D10]

# Max similarity for each query token
score = sum([max(cosine(e_Qi, e_Dj) for j in docs) for i in query])

# Better than bi-encoder, faster than cross-encoder
```

---

## 6. Query Expansion

```python
def query_expansion_llm(query, llm):
    expanded = llm.generate(
        f"Generate 3 alternative phrasings of this question:\n{query}"
    )

    queries = [query] + parse_alternatives(expanded)

    # Retrieve for all queries
    all_docs = []
    for q in queries:
        docs = retrieve(q, k=10)
        all_docs.extend(docs)

    # Deduplicate and return top docs
    return deduplicate(all_docs)[:10]
```

---

## 7. Key Takeaways

✅ **Dense (embeddings)**: Semantic understanding
✅ **Sparse (BM25)**: Exact keyword matching
✅ **Hybrid**: Combine for best results
✅ **Reranking**: Cross-encoder improves top-k quality
✅ **Query expansion**: Multiple phrasings capture more

**Recommendation**: Hybrid retrieval (100 docs) → Cross-encoder rerank (10 docs) → LLM generation

---

**Status**: Complete | Module 8 Complete (2/2) papers
