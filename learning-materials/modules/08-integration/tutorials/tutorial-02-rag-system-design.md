# Tutorial: RAG System Design and Implementation

**Estimated Time**: 90 minutes
**Level**: Advanced

## Overview

Learn to design and implement production-ready RAG (Retrieval-Augmented Generation) systems with best practices for chunking, retrieval, and generation.

## 1. RAG Architecture Patterns

### Basic RAG

```
Query → Embed → Retrieve → Augment → Generate → Answer
```

### Advanced RAG

```
Query → Rewrite → Multi-Retrieve → Rerank → Augment → Generate → Verify → Answer
```

## 2. Document Processing

### Smart Chunking Strategy

```python
from typing import List
from dataclasses import dataclass

@dataclass
class Chunk:
    content: str
    start_char: int
    end_char: int
    metadata: dict

class SemanticChunker:
    """Chunk by semantic boundaries."""

    def __init__(self, max_chunk_size: int = 512, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str) -> List[Chunk]:
        """Chunk text at paragraph/section boundaries."""
        chunks = []

        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')

        current_chunk = []
        current_size = 0
        start_char = 0

        for para in paragraphs:
            para_size = len(para)

            # If adding this paragraph exceeds max, save current chunk
            if current_size + para_size > self.max_chunk_size and current_size > self.min_chunk_size:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(Chunk(
                    content=chunk_text,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata={'num_paragraphs': len(current_chunk)}
                ))

                start_char += len(chunk_text)
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(Chunk(
                content=chunk_text,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                metadata={'num_paragraphs': len(current_chunk)}
            ))

        return chunks
```

### Overlap Strategy

```python
class OverlappingChunker:
    """Create overlapping chunks to preserve context."""

    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[Chunk]:
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Find sentence boundary
            if end < len(text):
                # Look for period, question mark, or exclamation
                for i in range(end, start, -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break

            chunk_text = text[start:end]
            chunks.append(Chunk(
                content=chunk_text,
                start_char=start,
                end_char=end,
                metadata={}
            ))

            start = end - self.overlap

        return chunks
```

## 3. Embedding Strategies

### Batch Embedding

```python
from typing import List
import numpy as np

class EmbeddingService:
    def __init__(self, model_path: str, batch_size: int = 32):
        self.llm = Llama(model_path=model_path, embedding=True)
        self.batch_size = batch_size

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Efficient batch embedding."""
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            for text in batch:
                result = self.llm.create_embedding(text)
                emb = np.array(result['data'][0]['embedding'])
                embeddings.append(emb)

        return np.array(embeddings)

    def embed_with_cache(self, text: str, cache: dict) -> np.ndarray:
        """Cache embeddings."""
        if text in cache:
            return cache[text]

        embedding = self.embed(text)
        cache[text] = embedding
        return embedding
```

## 4. Retrieval Optimization

### Hybrid Search

```python
class HybridRetriever:
    """Combine vector and keyword search."""

    def __init__(self, vector_store, bm25_index):
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def retrieve(
        self,
        query: str,
        k: int = 10,
        vector_weight: float = 0.7
    ) -> List[Document]:
        """Hybrid retrieval with reranking."""
        # Vector search
        vector_results = self.vector_store.search(query, k=k)

        # BM25 search
        keyword_results = self.bm25_index.search(query, k=k)

        # Combine and rerank
        combined = self._combine_results(
            vector_results,
            keyword_results,
            vector_weight
        )

        return combined[:k]

    def _combine_results(
        self,
        vector_results,
        keyword_results,
        weight
    ) -> List[Document]:
        """Combine results with weighted scores."""
        scores = {}

        # Add vector scores
        for doc, score in vector_results:
            scores[doc.id] = weight * score

        # Add keyword scores
        for doc, score in keyword_results:
            if doc.id in scores:
                scores[doc.id] += (1 - weight) * score
            else:
                scores[doc.id] = (1 - weight) * score

        # Sort by combined score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.get_document(doc_id) for doc_id, _ in ranked]
```

### Reranking

```python
from sentence_transformers import CrossEncoder

class Reranker:
    """Rerank results using cross-encoder."""

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """Rerank documents."""
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score pairs
        scores = self.model.predict(pairs)

        # Rank by score
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]
```

## 5. Context Building

### Smart Context Assembly

```python
class ContextBuilder:
    """Build context from retrieved documents."""

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens

    def build_context(
        self,
        query: str,
        documents: List[tuple],
        include_sources: bool = True
    ) -> str:
        """Build context that fits in token limit."""
        context_parts = []
        total_tokens = 0

        for i, (doc, score) in enumerate(documents, 1):
            # Estimate tokens (4 chars ≈ 1 token)
            doc_tokens = len(doc) // 4

            if total_tokens + doc_tokens > self.max_tokens:
                # Truncate document to fit
                remaining_tokens = self.max_tokens - total_tokens
                truncated = doc[:remaining_tokens * 4]
                doc = truncated + "..."

            if include_sources:
                context_parts.append(f"[Source {i}] {doc}")
            else:
                context_parts.append(doc)

            total_tokens += doc_tokens

            if total_tokens >= self.max_tokens:
                break

        return "\n\n".join(context_parts)

    def deduplicate_context(self, documents: List[str]) -> List[str]:
        """Remove redundant information."""
        unique_docs = []
        seen_content = set()

        for doc in documents:
            # Simple deduplication based on first sentence
            first_sentence = doc.split('.')[0]
            if first_sentence not in seen_content:
                unique_docs.append(doc)
                seen_content.add(first_sentence)

        return unique_docs
```

## 6. Prompt Engineering for RAG

### Effective Prompts

```python
class RAGPrompts:
    @staticmethod
    def qa_prompt(context: str, question: str) -> str:
        return f"""Answer the question based on the context below. If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    @staticmethod
    def summarize_prompt(context: str) -> str:
        return f"""Summarize the key points from the following information:

{context}

Summary:"""

    @staticmethod
    def cite_sources_prompt(context: str, question: str) -> str:
        return f"""Answer the question using the provided sources. Cite sources using [1], [2], etc.

Sources:
{context}

Question: {question}

Answer with citations:"""
```

## 7. Quality Evaluation

### RAG Metrics

```python
class RAGEvaluator:
    """Evaluate RAG system quality."""

    def evaluate_retrieval(
        self,
        queries: List[str],
        retrieved: List[List[str]],
        relevant: List[List[str]]
    ) -> dict:
        """Compute retrieval metrics."""
        precisions = []
        recalls = []

        for ret, rel in zip(retrieved, relevant):
            ret_set = set(ret)
            rel_set = set(rel)

            if len(ret_set) > 0:
                precision = len(ret_set & rel_set) / len(ret_set)
                precisions.append(precision)

            if len(rel_set) > 0:
                recall = len(ret_set & rel_set) / len(rel_set)
                recalls.append(recall)

        return {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1': 2 * np.mean(precisions) * np.mean(recalls) / (np.mean(precisions) + np.mean(recalls))
        }

    def evaluate_generation(
        self,
        generated: List[str],
        references: List[str]
    ) -> dict:
        """Compute generation metrics."""
        from rouge import Rouge
        rouge = Rouge()

        scores = rouge.get_scores(generated, references, avg=True)
        return scores
```

## 8. Production Patterns

### Async RAG Pipeline

```python
import asyncio

class AsyncRAG:
    async def query(self, question: str) -> dict:
        # Parallel retrieval from multiple sources
        results = await asyncio.gather(
            self.retrieve_from_vector_db(question),
            self.retrieve_from_keyword_search(question),
            self.retrieve_from_graph_db(question)
        )

        # Combine and rerank
        combined = self.combine_results(*results)

        # Generate answer
        answer = await self.generate_async(question, combined)

        return {
            'answer': answer,
            'sources': combined
        }
```

### Caching Strategy

```python
from functools import lru_cache
import hashlib

class RAGWithCache:
    def __init__(self):
        self.query_cache = {}
        self.embedding_cache = {}

    @lru_cache(maxsize=1000)
    def embed_cached(self, text: str):
        """Cache embeddings."""
        return self.embedder.embed(text)

    def query_with_cache(self, question: str):
        """Cache query results."""
        cache_key = hashlib.md5(question.encode()).hexdigest()

        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        result = self.query(question)
        self.query_cache[cache_key] = result
        return result
```

## Summary

Key patterns for production RAG:
- Smart chunking strategies
- Hybrid search with reranking
- Efficient embedding with caching
- Context optimization
- Quality evaluation
- Async processing

---

**Tutorial**: 02 - RAG System Design
**Module**: 08 - Integration & Applications
**Version**: 1.0
