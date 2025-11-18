# RAG Systems - Retrieval-Augmented Generation

**Module 8, Lesson 2**
**Estimated Time**: 4 hours
**Difficulty**: Intermediate to Advanced

## Overview

Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge retrieval to provide accurate, up-to-date, and contextual responses. This lesson covers the complete architecture and implementation of RAG systems using llama.cpp.

## Learning Objectives

By the end of this lesson, you will be able to:
- Understand RAG architecture and components
- Implement document processing and chunking strategies
- Build and query vector databases
- Integrate retrieval with llama.cpp inference
- Optimize RAG pipeline performance
- Handle multi-modal and complex document types

## Prerequisites

- Module 1: Foundations
- Lesson 8.1: Python Bindings
- Understanding of vector embeddings
- Basic knowledge of databases (helpful)

---

## 1. RAG Architecture Overview

### What is RAG?

RAG enhances LLM responses by:
1. **Retrieving** relevant information from a knowledge base
2. **Augmenting** the prompt with retrieved context
3. **Generating** responses using the enriched prompt

### RAG Pipeline

```
┌─────────────────────────────────────────────────────┐
│                     RAG SYSTEM                      │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │         1. Document Processing               │  │
│  │  - Load documents                            │  │
│  │  - Chunk into segments                       │  │
│  │  - Generate embeddings                       │  │
│  └────────────┬─────────────────────────────────┘  │
│               │                                     │
│               ▼                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │         2. Vector Database                   │  │
│  │  - Store embeddings                          │  │
│  │  - Index for fast retrieval                  │  │
│  │  - Metadata management                       │  │
│  └────────────┬─────────────────────────────────┘  │
│               │                                     │
│  USER QUERY   │                                     │
│      │        │                                     │
│      ▼        ▼                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │         3. Retrieval                         │  │
│  │  - Query embedding                           │  │
│  │  - Similarity search                         │  │
│  │  - Ranking and filtering                     │  │
│  └────────────┬─────────────────────────────────┘  │
│               │                                     │
│               ▼                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │         4. Context Augmentation              │  │
│  │  - Format retrieved documents                │  │
│  │  - Construct prompt                          │  │
│  │  - Handle context window limits              │  │
│  └────────────┬─────────────────────────────────┘  │
│               │                                     │
│               ▼                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │         5. LLM Generation                    │  │
│  │  - llama.cpp inference                       │  │
│  │  - Stream response                           │  │
│  │  - Post-processing                           │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Key Components

1. **Document Loader**: Ingests various document formats
2. **Text Splitter**: Chunks documents for embedding
3. **Embedding Model**: Converts text to vectors
4. **Vector Store**: Stores and indexes embeddings
5. **Retriever**: Finds relevant documents
6. **LLM**: Generates responses with context

---

## 2. Document Processing

### Document Loading

```python
from typing import List, Dict
from pathlib import Path
import PyPDF2
import docx

class DocumentLoader:
    """Load documents from various sources."""

    @staticmethod
    def load_txt(file_path: str) -> str:
        """Load plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Load PDF file."""
        text = []
        with open(file_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            for page in pdf.pages:
                text.append(page.extract_text())
        return '\n'.join(text)

    @staticmethod
    def load_docx(file_path: str) -> str:
        """Load Word document."""
        doc = docx.Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

    @staticmethod
    def load_directory(directory: str) -> List[Dict[str, str]]:
        """Load all supported documents from directory."""
        documents = []
        path = Path(directory)

        loaders = {
            '.txt': DocumentLoader.load_txt,
            '.pdf': DocumentLoader.load_pdf,
            '.docx': DocumentLoader.load_docx,
        }

        for file_path in path.rglob('*'):
            if file_path.suffix in loaders:
                content = loaders[file_path.suffix](str(file_path))
                documents.append({
                    'content': content,
                    'source': str(file_path),
                    'type': file_path.suffix
                })

        return documents
```

### Text Chunking Strategies

#### Fixed-Size Chunking

```python
class FixedSizeChunker:
    """Chunk text into fixed-size segments with overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """Split text into chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            if chunk:
                chunks.append(chunk)

            start = end - self.overlap

        return chunks
```

#### Semantic Chunking

```python
class SemanticChunker:
    """Chunk text based on semantic boundaries."""

    def __init__(self, max_chunk_size: int = 512):
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> List[str]:
        """Split text on paragraph boundaries."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > self.max_chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks
```

#### Sentence-Based Chunking

```python
import re

class SentenceChunker:
    """Chunk text by sentences."""

    def __init__(self, sentences_per_chunk: int = 5, overlap_sentences: int = 1):
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can use spaCy/nltk for better results)
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str) -> List[str]:
        """Chunk by sentences."""
        sentences = self.split_sentences(text)
        chunks = []

        for i in range(0, len(sentences), self.sentences_per_chunk - self.overlap_sentences):
            chunk_sentences = sentences[i:i + self.sentences_per_chunk]
            chunk = ' '.join(chunk_sentences)
            if chunk:
                chunks.append(chunk)

        return chunks
```

### Document Metadata

```python
from typing import Any
from datetime import datetime

class Document:
    """Document with content and metadata."""

    def __init__(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        doc_id: str = None
    ):
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id or self._generate_id()

        # Add timestamps
        if 'created_at' not in self.metadata:
            self.metadata['created_at'] = datetime.now().isoformat()

    def _generate_id(self) -> str:
        """Generate unique document ID."""
        import hashlib
        return hashlib.md5(self.content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'doc_id': self.doc_id,
            'content': self.content,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create from dictionary."""
        return cls(
            content=data['content'],
            metadata=data['metadata'],
            doc_id=data['doc_id']
        )
```

---

## 3. Embeddings

### Using llama.cpp for Embeddings

```python
from llama_cpp import Llama
import numpy as np
from typing import List

class LlamaCppEmbedder:
    """Generate embeddings using llama.cpp."""

    def __init__(self, model_path: str, n_ctx: int = 512):
        self.llm = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=n_ctx,
            verbose=False
        )

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        result = self.llm.create_embedding(text)
        return np.array(result['data'][0]['embedding'])

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]

    def embed_documents(self, documents: List[Document]) -> List[np.ndarray]:
        """Generate embeddings for documents."""
        return [self.embed(doc.content) for doc in documents]
```

### Alternative: Sentence Transformers

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SentenceTransformerEmbedder:
    """Generate embeddings using Sentence Transformers."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embedding generation."""
        return self.model.encode(texts, convert_to_numpy=True)
```

---

## 4. Vector Databases

### In-Memory Vector Store

```python
import numpy as np
from typing import List, Tuple

class InMemoryVectorStore:
    """Simple in-memory vector store."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = []
        self.documents = []
        self.metadata = []

    def add(
        self,
        vector: np.ndarray,
        document: str,
        metadata: Dict[str, Any] = None
    ):
        """Add vector and document."""
        assert len(vector) == self.dimension
        self.vectors.append(vector)
        self.documents.append(document)
        self.metadata.append(metadata or {})

    def add_batch(
        self,
        vectors: List[np.ndarray],
        documents: List[str],
        metadata: List[Dict[str, Any]] = None
    ):
        """Add multiple vectors."""
        metadata = metadata or [{} for _ in vectors]
        for vec, doc, meta in zip(vectors, documents, metadata):
            self.add(vec, doc, meta)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """Search for k most similar vectors."""
        if not self.vectors:
            return []

        # Compute cosine similarities
        similarities = []
        for vec in self.vectors:
            similarity = np.dot(query_vector, vec) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vec)
            )
            similarities.append(similarity)

        # Get top k
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append((
                self.documents[idx],
                similarities[idx],
                self.metadata[idx]
            ))

        return results

    def save(self, file_path: str):
        """Save to disk."""
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump({
                'dimension': self.dimension,
                'vectors': self.vectors,
                'documents': self.documents,
                'metadata': self.metadata
            }, f)

    @classmethod
    def load(cls, file_path: str) -> 'InMemoryVectorStore':
        """Load from disk."""
        import pickle
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        store = cls(data['dimension'])
        store.vectors = data['vectors']
        store.documents = data['documents']
        store.metadata = data['metadata']
        return store
```

### FAISS Integration

```python
import faiss
import numpy as np

class FAISSVectorStore:
    """Vector store using FAISS for efficient similarity search."""

    def __init__(self, dimension: int, index_type: str = 'flat'):
        self.dimension = dimension
        self.documents = []
        self.metadata = []

        # Create index
        if index_type == 'flat':
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == 'ivf':
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def add(self, vector: np.ndarray, document: str, metadata: Dict = None):
        """Add single vector."""
        self.index.add(vector.reshape(1, -1).astype('float32'))
        self.documents.append(document)
        self.metadata.append(metadata or {})

    def add_batch(
        self,
        vectors: np.ndarray,
        documents: List[str],
        metadata: List[Dict] = None
    ):
        """Add multiple vectors efficiently."""
        self.index.add(vectors.astype('float32'))
        self.documents.extend(documents)
        self.metadata.extend(metadata or [{} for _ in documents])

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """Search for k nearest neighbors."""
        query = query_vector.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx],
                    float(dist),
                    self.metadata[idx]
                ))

        return results
```

### ChromaDB Integration

```python
import chromadb
from chromadb.config import Settings

class ChromaVectorStore:
    """Vector store using ChromaDB."""

    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        document: str,
        embedding: np.ndarray,
        metadata: Dict = None,
        doc_id: str = None
    ):
        """Add document with embedding."""
        self.collection.add(
            documents=[document],
            embeddings=[embedding.tolist()],
            metadatas=[metadata or {}],
            ids=[doc_id or str(hash(document))]
        )

    def add_batch(
        self,
        documents: List[str],
        embeddings: List[np.ndarray],
        metadata: List[Dict] = None,
        ids: List[str] = None
    ):
        """Add multiple documents."""
        self.collection.add(
            documents=documents,
            embeddings=[emb.tolist() for emb in embeddings],
            metadatas=metadata or [{} for _ in documents],
            ids=ids or [str(hash(doc)) for doc in documents]
        )

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_dict: Dict = None
    ) -> List[Tuple[str, float, Dict]]:
        """Search for similar documents."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=filter_dict
        )

        output = []
        for doc, dist, meta in zip(
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        ):
            output.append((doc, dist, meta))

        return output
```

---

## 5. RAG Pipeline Implementation

### Basic RAG System

```python
from llama_cpp import Llama
from typing import List, Dict, Any

class SimpleRAG:
    """Simple RAG implementation."""

    def __init__(
        self,
        llm_model_path: str,
        embedding_model_path: str,
        n_ctx: int = 4096
    ):
        # Initialize LLM
        self.llm = Llama(
            model_path=llm_model_path,
            n_ctx=n_ctx,
            n_gpu_layers=35,
            verbose=False
        )

        # Initialize embedder
        self.embedder = LlamaCppEmbedder(embedding_model_path)

        # Initialize vector store
        self.vector_store = InMemoryVectorStore(
            dimension=4096  # Adjust based on embedding model
        )

    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Index documents for retrieval."""
        # Generate embeddings
        embeddings = self.embedder.embed_batch(documents)

        # Add to vector store
        self.vector_store.add_batch(
            vectors=embeddings,
            documents=documents,
            metadata=metadata
        )

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve relevant documents."""
        # Embed query
        query_embedding = self.embedder.embed(query)

        # Search
        results = self.vector_store.search(query_embedding, k=k)

        return [(doc, score) for doc, score, _ in results]

    def generate(
        self,
        query: str,
        k: int = 3,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """Generate response with RAG."""
        # Retrieve relevant documents
        results = self.retrieve(query, k=k)

        # Build context
        context = "\n\n".join([
            f"Document {i+1}:\n{doc}"
            for i, (doc, _) in enumerate(results)
        ])

        # Build prompt
        prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:"""

        # Generate response
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["Question:", "\n\n"]
        )

        return {
            'answer': response['choices'][0]['text'].strip(),
            'sources': [doc for doc, _ in results],
            'prompt': prompt
        }
```

### Advanced RAG with Reranking

```python
from typing import List, Tuple
import numpy as np

class AdvancedRAG(SimpleRAG):
    """RAG with advanced retrieval and reranking."""

    def __init__(self, *args, rerank_model_path: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rerank_model = None
        if rerank_model_path:
            # Load reranking model (e.g., cross-encoder)
            from sentence_transformers import CrossEncoder
            self.rerank_model = CrossEncoder(rerank_model_path)

    def retrieve_and_rerank(
        self,
        query: str,
        k: int = 3,
        initial_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Retrieve and rerank documents."""
        # Initial retrieval (get more candidates)
        initial_results = self.retrieve(query, k=initial_k)

        if not self.rerank_model:
            return initial_results[:k]

        # Rerank using cross-encoder
        pairs = [[query, doc] for doc, _ in initial_results]
        scores = self.rerank_model.predict(pairs)

        # Sort by reranking scores
        reranked = sorted(
            zip([doc for doc, _ in initial_results], scores),
            key=lambda x: x[1],
            reverse=True
        )

        return reranked[:k]

    def generate_with_citations(
        self,
        query: str,
        k: int = 3,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """Generate response with citations."""
        # Retrieve and rerank
        results = self.retrieve_and_rerank(query, k=k)

        # Build context with citations
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(f"[{i}] {doc}")

        context = "\n\n".join(context_parts)

        # Build prompt with citation instructions
        prompt = f"""Use the following context to answer the question. Cite your sources using [1], [2], etc.

Context:
{context}

Question: {query}

Answer (with citations):"""

        # Generate
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7
        )

        return {
            'answer': response['choices'][0]['text'].strip(),
            'sources': [
                {'text': doc, 'score': float(score), 'index': i}
                for i, (doc, score) in enumerate(results, 1)
            ],
            'prompt': prompt
        }
```

### Conversational RAG

```python
class ConversationalRAG(SimpleRAG):
    """RAG system with conversation history."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history = []

    def add_to_history(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({
            'role': role,
            'content': content
        })

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def generate_conversational(
        self,
        query: str,
        k: int = 3,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """Generate response considering conversation history."""
        # Retrieve relevant documents
        results = self.retrieve(query, k=k)

        # Build context
        context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, (doc, _) in enumerate(results)])

        # Build conversation history
        history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.conversation_history[-4:]  # Last 2 turns
        ])

        # Build prompt
        prompt = f"""Context:
{context}

Conversation History:
{history}

User: {query}