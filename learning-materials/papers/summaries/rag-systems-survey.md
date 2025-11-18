# Retrieval-Augmented Generation (RAG) Systems Survey

**Module**: 8 - RAG and Knowledge Systems | **Impact**: ⭐⭐⭐⭐⭐

---

## Executive Summary

RAG combines retrieval (search) with generation (LLMs) to ground responses in external knowledge. Essential for production LLM applications requiring factual accuracy, up-to-date information, or domain-specific knowledge.

**Key Pattern**: Query → Retrieve relevant docs → Generate with context

---

## 1. Basic RAG Architecture

```python
class BasicRAG:
    def __init__(self, vector_db, llm):
        self.vector_db = vector_db  # ChromaDB, Pinecone, etc.
        self.llm = llm              # LLaMA, GPT, etc.

    def query(self, question):
        # Step 1: Retrieve relevant documents
        docs = self.vector_db.similarity_search(question, k=5)

        # Step 2: Format context
        context = "\n\n".join([doc.content for doc in docs])

        # Step 3: Generate answer with context
        prompt = f"""Context:
{context}

Question: {question}

Answer based on the context above:"""

        answer = self.llm.generate(prompt)
        return answer
```

---

## 2. RAG Pipeline Components

### 2.1 Document Processing

```python
# Chunking strategies
def chunk_text(text, chunk_size=512, overlap=128):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

# Semantic chunking (better)
def semantic_chunk(text, model):
    sentences = split_sentences(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(" ".join(current_chunk)) > 512:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]  # Keep last sentence for overlap

    return chunks
```

### 2.2 Embedding and Indexing

```python
from sentence_transformers import SentenceTransformer
import chromadb

# Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# Store in vector DB
client = chromadb.Client()
collection = client.create_collection("docs")

for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    collection.add(
        ids=[f"doc_{i}"],
        embeddings=[embedding],
        documents=[chunk]
    )
```

### 2.3 Retrieval

```python
def retrieve(query, collection, k=5):
    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results['documents'][0]
```

---

## 3. Advanced RAG Patterns

### 3.1 Hypothetical Document Embeddings (HyDE)

```python
def hyde_retrieval(query, llm, collection, k=5):
    # Generate hypothetical answer
    hypothetical_answer = llm.generate(
        f"Question: {query}\nAnswer:"
    )

    # Retrieve using hypothetical answer (often better than query)
    docs = collection.query(
        query_texts=[hypothetical_answer],
        n_results=k
    )

    return docs
```

### 3.2 Self-RAG (Retrieval on Demand)

```python
def self_rag(query, llm, collection):
    # LLM decides if retrieval is needed
    decision = llm.generate(
        f"Does this question require external knowledge? {query}\nYes/No:"
    )

    if "yes" in decision.lower():
        docs = retrieve(query, collection)
        context = "\n".join(docs)
        return llm.generate(f"Context: {context}\n\nQ: {query}\nA:")
    else:
        return llm.generate(f"Q: {query}\nA:")
```

### 3.3 Iterative RAG

```python
def iterative_rag(query, llm, collection, max_iterations=3):
    context = []

    for i in range(max_iterations):
        # Generate next search query
        search_query = llm.generate(
            f"Original question: {query}\n"
            f"Context so far: {context}\n"
            f"What should I search for next?"
        )

        # Retrieve
        docs = retrieve(search_query, collection, k=3)
        context.extend(docs)

        # Check if sufficient
        sufficient = llm.generate(
            f"Q: {query}\nContext: {context}\n"
            f"Is this enough to answer? Yes/No:"
        )

        if "yes" in sufficient.lower():
            break

    # Final answer
    return llm.generate(
        f"Context: {'\n'.join(context)}\nQ: {query}\nA:"
    )
```

---

## 4. RAG with llama.cpp

### 4.1 Python Integration

```python
from llama_cpp import Llama
import chromadb

# Initialize
llm = Llama(model_path="llama-2-7b-q4_K_M.gguf", n_ctx=4096)
collection = chromadb.Client().create_collection("docs")

def rag_query(question):
    # Retrieve
    results = collection.query(query_texts=[question], n_results=5)
    context = "\n\n".join(results['documents'][0])

    # Generate
    prompt = f"""[INST] Answer the question based on this context:

Context:
{context}

Question: {question} [/INST]"""

    response = llm(prompt, max_tokens=512)
    return response['choices'][0]['text']
```

### 4.2 Server Mode RAG

```bash
# Start llama.cpp server
./llama-server -m model.gguf --port 8080

# Python RAG client
import requests

def rag_with_server(question, docs):
    prompt = f"Context: {docs}\n\nQ: {question}\nA:"

    response = requests.post(
        "http://localhost:8080/completion",
        json={"prompt": prompt, "n_predict": 256}
    )

    return response.json()['content']
```

---

## 5. Evaluation Metrics

```python
# Retrieval quality
def evaluate_retrieval(retrieved_docs, ground_truth_docs):
    recall_at_k = len(set(retrieved_docs) & set(ground_truth_docs)) / len(ground_truth_docs)
    return recall_at_k

# Generation quality
def evaluate_generation(generated, reference):
    from rouge import Rouge
    rouge = Rouge()
    scores = rouge.get_scores(generated, reference)
    return scores

# End-to-end
def evaluate_rag(questions, answers, rag_system):
    correct = 0
    for q, expected_a in zip(questions, answers):
        generated_a = rag_system.query(q)
        if compare_answers(generated_a, expected_a):
            correct += 1
    return correct / len(questions)
```

---

## 6. Production Considerations

**Chunking**: 256-512 tokens optimal, overlap 20-50%
**Embedding Model**: all-MiniLM-L6-v2 (fast), BGE-large (quality)
**Vector DB**: ChromaDB (local), Pinecone/Weaviate (production)
**Reranking**: Use cross-encoder after retrieval for better ranking
**Caching**: Cache embeddings and frequent queries
**Monitoring**: Track retrieval quality, latency, relevance

---

## 7. Key Takeaways

✅ **RAG is essential**: Grounds LLMs in facts, reduces hallucination
✅ **Simple RAG**: Retrieve → Context → Generate (good baseline)
✅ **Advanced patterns**: HyDE, Self-RAG, Iterative RAG (better quality)
✅ **llama.cpp friendly**: Easy to integrate with Python or server mode
✅ **Production**: Focus on chunking, embedding quality, retrieval metrics

---

## Further Reading

- **RAG Survey**: https://arxiv.org/abs/2312.10997
- **LlamaIndex**: Popular RAG framework
- **LangChain**: RAG + agent framework

---

**Status**: Complete | Module 8 (1/2) papers
