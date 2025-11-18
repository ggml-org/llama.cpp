# Tutorial: Building Production RAG Systems with LLaMA.cpp

**Duration**: 2-3 hours
**Level**: Advanced
**Goal**: Build a complete, production-ready RAG system using llama.cpp

---

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Optimization Techniques](#optimization-techniques)
5. [Deployment](#deployment)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)

---

## Introduction

Retrieval Augmented Generation (RAG) combines the power of semantic search with large language models to create systems that can answer questions using external knowledge bases. This tutorial will guide you through building a production-ready RAG system from scratch.

### What You'll Build

A complete RAG system with:
- Document ingestion and indexing
- Semantic search with embeddings
- Context-aware generation
- REST API interface
- Monitoring and logging
- Scalable architecture

### Prerequisites

- Completed Module 7.2 Lab
- Python 3.8+
- 16GB+ RAM
- GPU recommended (but optional)

---

## System Architecture

### High-Level Design

```
┌─────────────────┐
│   Documents     │
│   (.txt, .pdf)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Chunking     │  ← Split into manageable pieces
│   & Cleaning    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embedding     │  ← nomic-embed or BGE
│   Generation    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Database │  ← FAISS, ChromaDB, etc.
│   (Indexing)    │
└────────┬────────┘
         │
    [Storage]
         │
┌────────┴────────┐
│                 │
│   Query Time    │
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Embedding │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Search   │  ← Find top-k similar docs
│  (Retrieval)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Rerank & Filter │  ← Optional: improve results
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Prompt Builder  │  ← Construct context
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Generation  │  ← llama.cpp model
│  (llama-cpp)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Response      │
└─────────────────┘
```

### Component Breakdown

1. **Document Processor**: Ingests and prepares documents
2. **Embedding Service**: Generates vector embeddings
3. **Vector Store**: Stores and indexes embeddings
4. **Retrieval Engine**: Finds relevant documents
5. **Generation Service**: Produces answers with LLM
6. **API Layer**: Exposes functionality via REST API

---

## Step-by-Step Implementation

### Step 1: Document Processing

**Create `document_processor.py`**:

```python
import re
from pathlib import Path
from typing import List, Dict
import PyPDF2


class DocumentProcessor:
    """Process and chunk documents for RAG"""

    def __init__(self, chunk_size=512, chunk_overlap=50):
        """
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_document(self, file_path: str) -> str:
        """Load document from various formats"""
        path = Path(file_path)

        if path.suffix == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()

        elif path.suffix == '.pdf':
            text = ""
            with open(path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                for page in pdf.pages:
                    text += page.extract_text()
            return text

        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,;:!?-]', '', text)

        return text.strip()

    def chunk_text(self, text: str) -> List[Dict]:
        """Split text into overlapping chunks"""
        # Simple word-based chunking
        words = text.split()
        chunks = []

        start = 0
        chunk_id = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)

            chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'start_idx': start,
                'end_idx': min(end, len(words)),
                'word_count': len(chunk_words)
            })

            chunk_id += 1
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def process_document(self, file_path: str, metadata: Dict = None) -> List[Dict]:
        """Complete processing pipeline"""
        # Load
        text = self.load_document(file_path)

        # Clean
        text = self.clean_text(text)

        # Chunk
        chunks = self.chunk_text(text)

        # Add metadata
        if metadata is None:
            metadata = {}

        metadata['source'] = str(file_path)

        for chunk in chunks:
            chunk['metadata'] = metadata

        return chunks

    def process_directory(self, dir_path: str) -> List[Dict]:
        """Process all documents in directory"""
        all_chunks = []

        for file_path in Path(dir_path).glob('**/*'):
            if file_path.is_file() and file_path.suffix in ['.txt', '.pdf']:
                print(f"Processing: {file_path}")

                try:
                    chunks = self.process_document(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        return all_chunks


# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor(chunk_size=256, chunk_overlap=32)

    # Process single document
    chunks = processor.process_document("knowledge.txt")
    print(f"Created {len(chunks)} chunks from document")

    # Process directory
    all_chunks = processor.process_directory("./documents")
    print(f"Total chunks: {len(all_chunks)}")
```

### Step 2: Embedding Service

**Create `embedding_service.py`**:

```python
import numpy as np
from llama_cpp import Llama
from typing import List, Union
import hashlib
import json
from pathlib import Path


class EmbeddingService:
    """Manage embedding generation with caching"""

    def __init__(self, model_path: str, cache_dir: str = "./cache"):
        """
        Args:
            model_path: Path to embedding model
            cache_dir: Directory for caching embeddings
        """
        print(f"Loading embedding model: {model_path}")

        self.model = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=512,
            n_batch=512,
            verbose=False
        )

        self.n_embd = self.model.n_embd()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        print(f"✓ Embedding service ready (dim={self.n_embd})")

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()

    def _load_from_cache(self, text: str) -> Union[np.ndarray, None]:
        """Load embedding from cache"""
        cache_file = self.cache_dir / f"{self._cache_key(text)}.npy"

        if cache_file.exists():
            return np.load(cache_file)

        return None

    def _save_to_cache(self, text: str, embedding: np.ndarray):
        """Save embedding to cache"""
        cache_file = self.cache_dir / f"{self._cache_key(text)}.npy"
        np.save(cache_file, embedding)

    def embed(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Generate embedding for text"""
        # Check cache
        if use_cache:
            cached = self._load_from_cache(text)
            if cached is not None:
                return cached

        # Generate
        result = self.model.create_embedding(text)
        embedding = np.array(result['data'][0]['embedding'], dtype=np.float32)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Cache
        if use_cache:
            self._save_to_cache(text, embedding)

        return embedding

    def embed_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        embeddings = []

        for i, text in enumerate(texts):
            if show_progress and i % 10 == 0:
                print(f"Embedding: {i+1}/{len(texts)}", end='\r')

            emb = self.embed(text)
            embeddings.append(emb)

        if show_progress:
            print()

        return np.array(embeddings)


# Example usage
if __name__ == "__main__":
    service = EmbeddingService("nomic-embed-text-v1.5.Q4_K_M.gguf")

    # Single embedding
    text = "Machine learning is a subset of AI"
    emb = service.embed(text)
    print(f"Embedding shape: {emb.shape}")

    # Batch
    texts = [
        "Machine learning is a subset of AI",
        "Python is a programming language",
        "The sky is blue"
    ]
    embs = service.embed_batch(texts)
    print(f"Batch embeddings shape: {embs.shape}")

    # Similarity
    sim = np.dot(embs[0], embs[1])
    print(f"Similarity: {sim:.4f}")
```

### Step 3: Vector Store with FAISS

**Create `vector_store.py`**:

```python
import numpy as np
import faiss
import json
from typing import List, Dict, Tuple
from pathlib import Path


class VectorStore:
    """FAISS-based vector store"""

    def __init__(self, dimension: int):
        """
        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine for normalized)
        self.documents = []

    def add(self, embeddings: np.ndarray, documents: List[Dict]):
        """
        Add embeddings and documents to store

        Args:
            embeddings: Array of shape (N, dimension)
            documents: List of document metadata dicts
        """
        if len(embeddings) != len(documents):
            raise ValueError("Embeddings and documents must have same length")

        # Normalize embeddings
        faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings.astype('float32'))

        # Store documents
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents

        Args:
            query_embedding: Query vector
            top_k: Number of results

        Returns:
            List of documents with scores
        """
        # Normalize query
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))

        # Gather results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result['score'] = float(score)
                results.append(result)

        return results

    def save(self, directory: str):
        """Save index and documents"""
        directory = Path(directory)
        directory.mkdir(exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(directory / "faiss.index"))

        # Save documents
        with open(directory / "documents.json", 'w') as f:
            json.dump({
                'dimension': self.dimension,
                'num_documents': len(self.documents),
                'documents': self.documents
            }, f)

        print(f"✓ Saved index to {directory}")

    def load(self, directory: str):
        """Load index and documents"""
        directory = Path(directory)

        # Load FAISS index
        self.index = faiss.read_index(str(directory / "faiss.index"))

        # Load documents
        with open(directory / "documents.json", 'r') as f:
            data = json.load(f)
            self.dimension = data['dimension']
            self.documents = data['documents']

        print(f"✓ Loaded index with {len(self.documents)} documents")

    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            'dimension': self.dimension,
            'num_documents': len(self.documents),
            'index_size_mb': self.index.ntotal * self.dimension * 4 / 1024 / 1024
        }


# Example usage
if __name__ == "__main__":
    # Create store
    store = VectorStore(dimension=768)

    # Add documents
    embeddings = np.random.randn(100, 768).astype('float32')
    documents = [{'id': i, 'text': f"Document {i}"} for i in range(100)]

    store.add(embeddings, documents)

    # Search
    query = np.random.randn(768).astype('float32')
    results = store.search(query, top_k=5)

    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.4f} - {result['text']}")

    # Save/load
    store.save("./index")
    store2 = VectorStore(dimension=768)
    store2.load("./index")
```

### Step 4: RAG Pipeline

**Create `rag_pipeline.py`**:

```python
from document_processor import DocumentProcessor
from embedding_service import EmbeddingService
from vector_store import VectorStore
from llama_cpp import Llama
from typing import List, Dict


class RAGPipeline:
    """Complete RAG pipeline"""

    def __init__(self, embedding_model_path: str, llm_model_path: str,
                 n_gpu_layers: int = 0):
        """
        Initialize RAG pipeline

        Args:
            embedding_model_path: Path to embedding model
            llm_model_path: Path to LLM
            n_gpu_layers: GPU layers for LLM
        """
        print("Initializing RAG pipeline...")

        # Components
        self.processor = DocumentProcessor(chunk_size=256, chunk_overlap=32)
        self.embedder = EmbeddingService(embedding_model_path)

        # Initialize vector store
        self.store = VectorStore(dimension=self.embedder.n_embd)

        # Load LLM
        print(f"Loading LLM: {llm_model_path}")
        self.llm = Llama(
            model_path=llm_model_path,
            n_ctx=4096,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )

        print("✓ RAG pipeline ready")

    def ingest_directory(self, directory: str):
        """Ingest all documents from directory"""
        print(f"\nIngesting documents from {directory}...")

        # Process documents
        chunks = self.processor.process_directory(directory)
        print(f"✓ Processed {len(chunks)} chunks")

        # Generate embeddings
        print("Generating embeddings...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.embed_batch(texts)
        print(f"✓ Generated {len(embeddings)} embeddings")

        # Add to vector store
        self.store.add(embeddings, chunks)
        print(f"✓ Indexed {len(chunks)} chunks")

        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant documents"""
        # Generate query embedding
        query_emb = self.embedder.embed(query, use_cache=False)

        # Search
        results = self.store.search(query_emb, top_k=top_k)

        return results

    def generate(self, query: str, context_docs: List[Dict],
                max_tokens: int = 512) -> str:
        """Generate answer from context"""
        # Build context
        context = "\n\n".join([
            f"[{i+1}] {doc['text']}"
            for i, doc in enumerate(context_docs)
        ])

        # Build prompt
        prompt = f"""Answer the question based on the provided context. If you cannot answer based on the context, say so.

Context:
{context}

Question: {query}

Answer:"""

        # Generate
        response = self.llm.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["Question:", "\n\n\n"]
        )

        return response['choices'][0]['text'].strip()

    def query(self, question: str, top_k: int = 3,
             max_tokens: int = 512) -> Dict:
        """Complete RAG query"""
        # Retrieve
        docs = self.retrieve(question, top_k=top_k)

        if not docs:
            return {
                'answer': "No relevant documents found.",
                'sources': []
            }

        # Generate
        answer = self.generate(question, docs, max_tokens=max_tokens)

        return {
            'answer': answer,
            'sources': [
                {
                    'text': doc['text'][:200],
                    'score': doc['score'],
                    'source': doc.get('metadata', {}).get('source', 'unknown')
                }
                for doc in docs
            ]
        }

    def save_index(self, directory: str):
        """Save indexed knowledge base"""
        self.store.save(directory)

    def load_index(self, directory: str):
        """Load indexed knowledge base"""
        self.store.load(directory)


# Example usage
if __name__ == "__main__":
    # Initialize
    rag = RAGPipeline(
        embedding_model_path="nomic-embed-text-v1.5.Q4_K_M.gguf",
        llm_model_path="llama-2-7b-chat.Q4_K_M.gguf",
        n_gpu_layers=35
    )

    # Ingest documents
    rag.ingest_directory("./documents")

    # Save index
    rag.save_index("./rag_index")

    # Query
    result = rag.query("What is machine learning?")
    print("\nAnswer:", result['answer'])
    print("\nSources:")
    for src in result['sources']:
        print(f"- {src['text'][:100]}... (score: {src['score']:.3f})")
```

---

## Part 2: Production API

**Create `rag_api.py`**:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API")

# Initialize RAG (global)
rag = None

@app.on_event("startup")
async def startup():
    global rag
    logger.info("Initializing RAG pipeline...")

    rag = RAGPipeline(
        embedding_model_path="nomic-embed-text-v1.5.Q4_K_M.gguf",
        llm_model_path="llama-2-7b-chat.Q4_K_M.gguf",
        n_gpu_layers=35
    )

    # Load pre-built index
    try:
        rag.load_index("./rag_index")
        logger.info("✓ Loaded existing index")
    except:
        logger.warning("No existing index found")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    max_tokens: int = 512

@app.post("/query")
async def query(req: QueryRequest):
    """Query the RAG system"""
    try:
        result = rag.query(
            req.question,
            top_k=req.top_k,
            max_tokens=req.max_tokens
        )
        return result
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    stats = rag.store.get_stats()
    return {
        "status": "healthy",
        "stats": stats
    }

@app.post("/ingest")
async def ingest(directory: str):
    """Ingest new documents"""
    try:
        num_chunks = rag.ingest_directory(directory)
        rag.save_index("./rag_index")
        return {"status": "success", "chunks_indexed": num_chunks}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(500, detail=str(e))

# Run with: uvicorn rag_api:app --host 0.0.0.0 --port 8000
```

---

## Conclusion

You now have a complete, production-ready RAG system! This system includes:

✅ Document processing and chunking
✅ Embedding generation with caching
✅ Vector search with FAISS
✅ LLM-powered generation
✅ REST API interface
✅ Persistent storage

### Next Steps

1. Add monitoring and metrics
2. Implement user feedback loop
3. Add authentication
4. Scale with distributed indexing
5. Deploy with Docker/Kubernetes

**Congratulations!** You've built a production RAG system from scratch!
