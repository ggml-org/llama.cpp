# Project 1: Full-Stack RAG Application

**Estimated Time**: 15-20 hours
**Difficulty**: Advanced
**Team Size**: 1-2 people

## Project Overview

Build a production-ready RAG (Retrieval-Augmented Generation) application with document management, vector search, and web interface. This project combines multiple Module 8 concepts into a complete system.

## Features

### Core Features
- [X] Document upload and processing (PDF, DOCX, TXT, MD)
- [X] Automatic chunking and embedding
- [X] Vector similarity search
- [X] Context-augmented generation
- [X] Source citation and tracking
- [X] Conversation history

### Advanced Features
- [ ] Multi-user support with authentication
- [ ] Document collections/namespaces
- [ ] Hybrid search (vector + keyword)
- [ ] Query rewriting and expansion
- [ ] Answer quality scoring
- [ ] Export conversations

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Frontend (React/Vue)               │
│  - Document upload                              │
│  - Chat interface                               │
│  - Document browser                             │
└─────────────┬───────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────┐
│          Backend API (FastAPI/Flask)            │
│  - Document processing                          │
│  - Vector search                                │
│  - LLM integration                              │
└─────────────┬───────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────┐
│              Storage Layer                      │
│  - PostgreSQL (metadata)                        │
│  - FAISS/ChromaDB (vectors)                     │
│  - S3/MinIO (documents)                         │
└─────────────────────────────────────────────────┘
```

## Technology Stack

- **Backend**: FastAPI or Flask
- **Frontend**: React or Vue.js
- **Vector DB**: FAISS or ChromaDB
- **Database**: PostgreSQL
- **Storage**: MinIO or S3
- **LLM**: llama.cpp with llama-cpp-python
- **Deployment**: Docker Compose

## Implementation Guide

### Phase 1: Backend API (Week 1)

#### 1.1 Project Setup

```bash
mkdir rag-app && cd rag-app
mkdir backend frontend docs

# Backend structure
cd backend
mkdir -p app/{api,models,services,utils}
```

#### 1.2 Document Processing Service

```python
# app/services/document_processor.py
from typing import List, BinaryIO
from app.models.document import Document, Chunk

class DocumentProcessor:
    def process_file(self, file: BinaryIO, filename: str) -> Document:
        """Process uploaded file."""
        # Extract text based on file type
        text = self._extract_text(file, filename)

        # Create document
        doc = Document(
            filename=filename,
            content=text,
            metadata=self._extract_metadata(file)
        )

        # Chunk document
        doc.chunks = self._chunk_document(doc)

        return doc

    def _chunk_document(self, doc: Document) -> List[Chunk]:
        """Chunk document for embedding."""
        # Implement smart chunking
        pass
```

#### 1.3 Vector Store Service

```python
# app/services/vector_store.py
import chromadb
from typing import List, Tuple

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict]
    ):
        """Add documents to vector store."""
        collection = self.client.get_or_create_collection("documents")
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[str(i) for i in range(len(documents))]
        )

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5
    ) -> List[Tuple[str, float]]:
        """Search for similar documents."""
        collection = self.client.get_collection("documents")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
```

#### 1.4 RAG Service

```python
# app/services/rag_service.py
from llama_cpp import Llama

class RAGService:
    def __init__(self):
        self.llm = Llama(model_path="./models/model.gguf")
        self.vector_store = VectorStore()
        self.embedder = Embedder()

    async def query(
        self,
        question: str,
        collection_id: str = None
    ) -> dict:
        """Answer question using RAG."""
        # 1. Embed query
        query_embedding = self.embedder.embed(question)

        # 2. Retrieve relevant documents
        results = self.vector_store.search(query_embedding, n_results=5)

        # 3. Build context
        context = self._build_context(results)

        # 4. Generate answer
        answer = self._generate_answer(question, context)

        return {
            "answer": answer,
            "sources": [r["metadata"] for r in results],
            "context": context
        }
```

### Phase 2: Frontend (Week 2)

#### 2.1 React Components

```jsx
// src/components/DocumentUpload.jsx
import React, { useState } from 'react';
import { uploadDocument } from '../api';

function DocumentUpload() {
    const [file, setFile] = useState(null);
    const [uploading, setUploading] = useState(false);

    const handleUpload = async () => {
        setUploading(true);
        try {
            await uploadDocument(file);
            alert('Document uploaded successfully!');
        } catch (error) {
            alert('Upload failed: ' + error.message);
        }
        setUploading(false);
    };

    return (
        <div className="upload-container">
            <input
                type="file"
                onChange={(e) => setFile(e.target.files[0])}
                accept=".pdf,.docx,.txt,.md"
            />
            <button onClick={handleUpload} disabled={!file || uploading}>
                {uploading ? 'Uploading...' : 'Upload Document'}
            </button>
        </div>
    );
}
```

```jsx
// src/components/ChatInterface.jsx
import React, { useState } from 'react';
import { queryRAG } from '../api';

function ChatInterface() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async () => {
        const userMessage = { role: 'user', content: input };
        setMessages([...messages, userMessage]);
        setInput('');
        setLoading(true);

        try {
            const response = await queryRAG(input);
            setMessages([
                ...messages,
                userMessage,
                {
                    role: 'assistant',
                    content: response.answer,
                    sources: response.sources
                }
            ]);
        } catch (error) {
            alert('Query failed: ' + error.message);
        }

        setLoading(false);
    };

    return (
        <div className="chat-interface">
            <div className="messages">
                {messages.map((msg, i) => (
                    <Message key={i} message={msg} />
                ))}
            </div>
            <div className="input-area">
                <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
                    disabled={loading}
                />
                <button onClick={handleSubmit} disabled={loading}>
                    Send
                </button>
            </div>
        </div>
    );
}
```

### Phase 3: Deployment (Week 3)

#### 3.1 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/rag
      - MODEL_PATH=/models/model.gguf
    volumes:
      - ./models:/models
      - ./data:/data
    depends_on:
      - db

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=rag
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## Evaluation Criteria

| Criteria | Weight | Description |
|----------|--------|-------------|
| Functionality | 30% | All core features working |
| Code Quality | 20% | Clean, documented, tested |
| UI/UX | 20% | Intuitive, responsive interface |
| Performance | 15% | Fast search and generation |
| Documentation | 15% | Complete setup and usage docs |

## Bonus Features (+20%)

- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Custom embedding models
- [ ] GraphQL API
- [ ] Mobile app

## Deliverables

1. **Source Code**: Complete application with documentation
2. **Deployment**: Docker Compose setup
3. **Demo Video**: 5-minute walkthrough
4. **Documentation**:
   - Architecture diagram
   - API documentation
   - User guide
   - Development setup guide

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [React Documentation](https://react.dev/)
- [llama-cpp-python Examples](https://github.com/abetlen/llama-cpp-python)

---

**Project**: 01 - Full-Stack RAG Application
**Module**: 08 - Integration & Applications
**Version**: 1.0
