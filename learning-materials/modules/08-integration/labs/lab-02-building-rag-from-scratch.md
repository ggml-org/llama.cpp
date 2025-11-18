# Lab 8.2: Building RAG From Scratch

**Estimated Time**: 3-4 hours
**Difficulty**: Advanced
**Prerequisites**: Lessons 8.1, 8.2

## Objective

Build a complete RAG (Retrieval-Augmented Generation) system from scratch with document processing, vector search, and context-augmented generation.

## Learning Outcomes

- Implement document chunking strategies
- Build vector embeddings and similarity search
- Integrate retrieval with generation
- Optimize RAG pipeline performance
- Evaluate RAG system quality

---

## Part 1: Document Processing Pipeline (45 min)

### Task 1.1: Document Loader

Implement loaders for multiple formats:

```python
# document_loader.py
from typing import List, Dict
from pathlib import Path
import PyPDF2
import docx

class Document:
    def __init__(self, content: str, metadata: Dict = None):
        self.content = content
        self.metadata = metadata or {}

class DocumentLoader:
    @staticmethod
    def load_txt(path: str) -> Document:
        with open(path, 'r') as f:
            return Document(f.read(), {'source': path})

    @staticmethod
    def load_pdf(path: str) -> Document:
        # TODO: Implement PDF loading
        pass

    @staticmethod
    def load_directory(directory: str) -> List[Document]:
        # TODO: Load all supported files
        pass
```

**Exercise**: Complete the PDF loader and directory loader.

### Task 1.2: Text Chunking

Implement multiple chunking strategies:

```python
# chunker.py
from typing import List
from abc import ABC, abstractmethod

class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass

class FixedSizeChunker(Chunker):
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        # TODO: Implement
        pass

class SemanticChunker(Chunker):
    # TODO: Implement semantic chunking
    pass
```

**Exercise**: Implement both chunking strategies and compare results.

---

## Part 2: Vector Store Implementation (60 min)

### Task 2.1: Embedding Generation

```python
# embedder.py
from llama_cpp import Llama
import numpy as np

class Embedder:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=512
        )

    def embed(self, text: str) -> np.ndarray:
        result = self.llm.create_embedding(text)
        return np.array(result['data'][0]['embedding'])

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        # TODO: Implement batch embedding
        pass
```

### Task 2.2: Vector Store with FAISS

```python
# vector_store.py
import faiss
import numpy as np
from typing import List, Tuple

class FAISSVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.metadata = []

    def add(self, vectors: np.ndarray, documents: List[str], metadata: List[Dict]):
        # TODO: Implement
        pass

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple]:
        # TODO: Implement
        pass

    def save(self, path: str):
        # TODO: Implement persistence
        pass

    def load(self, path: str):
        # TODO: Implement loading
        pass
```

**Exercise**: Complete the vector store implementation with FAISS.

---

## Part 3: RAG Pipeline (60 min)

### Task 3.1: Complete RAG System

```python
# rag_system.py
from typing import List, Dict, Tuple

class RAGSystem:
    def __init__(
        self,
        llm_path: str,
        embedding_path: str,
        chunk_size: int = 512
    ):
        self.llm = Llama(model_path=llm_path)
        self.embedder = Embedder(embedding_path)
        self.chunker = FixedSizeChunker(chunk_size)
        self.vector_store = FAISSVectorStore(dimension=4096)

    def index_documents(self, documents: List[Document]):
        """Index documents for retrieval."""
        # TODO: Implement indexing pipeline
        pass

    def retrieve(self, query: str, k: int = 3) -> List[Tuple]:
        """Retrieve relevant documents."""
        # TODO: Implement retrieval
        pass

    def generate(self, query: str, k: int = 3) -> str:
        """Generate answer with RAG."""
        # TODO: Implement generation with context
        pass
```

### Task 3.2: Context Building

Implement smart context building:

```python
def build_context(
    self,
    query: str,
    retrieved_docs: List[Tuple],
    max_tokens: int = 2000
) -> str:
    """
    Build context from retrieved documents.

    Should:
    - Rank by relevance
    - Fit within token limit
    - Include source citations
    - Handle redundancy
    """
    # TODO: Implement
    pass
```

---

## Part 4: Evaluation (45 min)

### Task 4.1: Quality Metrics

Implement RAG evaluation:

```python
# evaluator.py
class RAGEvaluator:
    def evaluate_retrieval(
        self,
        queries: List[str],
        expected_docs: List[List[str]],
        retrieved_docs: List[List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality.

        Metrics:
        - Precision@K
        - Recall@K
        - MRR (Mean Reciprocal Rank)
        """
        # TODO: Implement metrics
        pass

    def evaluate_generation(
        self,
        queries: List[str],
        generated: List[str],
        expected: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate generation quality.

        Metrics:
        - ROUGE scores
        - Factual accuracy
        - Relevance
        """
        # TODO: Implement
        pass
```

---

## Part 5: Optimization (30 min)

### Task 5.1: Performance Tuning

Optimize your RAG system:

1. **Chunking Optimization**
   - Test different chunk sizes
   - Compare fixed vs semantic chunking

2. **Retrieval Optimization**
   - Implement re-ranking
   - Add query expansion
   - Try different similarity metrics

3. **Generation Optimization**
   - Optimize prompt templates
   - Tune context window usage

**Exercise**: Create performance benchmark comparing different configurations.

---

## Challenges

### Challenge 1: Multi-Modal RAG
Add support for images and tables in documents.

### Challenge 2: Hybrid Search
Combine vector search with keyword search (BM25).

### Challenge 3: Query Rewriting
Implement automatic query expansion and reformulation.

### Challenge 4: Streaming RAG
Stream both retrieval and generation results.

---

## Testing Your RAG System

### Test Dataset

Create test data:

```python
# test_rag.py
import pytest

test_documents = [
    """Python is a high-level programming language...""",
    """Machine learning is a subset of AI...""",
    """LLaMA is a large language model..."""
]

test_queries = [
    ("What is Python?", "Python is a programming language"),
    ("Explain ML", "Machine learning"),
    ("Tell me about LLaMA", "LLaMA")
]

def test_indexing():
    rag = RAGSystem(...)
    rag.index_documents(test_documents)
    assert len(rag.vector_store.documents) > 0

def test_retrieval():
    rag = RAGSystem(...)
    results = rag.retrieve("What is Python?", k=3)
    assert len(results) == 3
    assert "Python" in results[0][0]

def test_generation():
    rag = RAGSystem(...)
    answer = rag.generate("What is Python?")
    assert "Python" in answer
```

---

## Success Criteria

- [X] Document loading for multiple formats
- [X] Chunking strategies implemented
- [X] Vector store with FAISS working
- [X] RAG pipeline complete
- [X] Evaluation metrics implemented
- [X] Performance optimizations applied
- [X] Tests passing

## Submission

Submit:
1. Complete RAG system code
2. Performance benchmark results
3. Example queries and responses
4. Evaluation metrics report

---

**Lab**: 8.2 - Building RAG From Scratch
**Module**: 08 - Integration & Applications
**Version**: 1.0
