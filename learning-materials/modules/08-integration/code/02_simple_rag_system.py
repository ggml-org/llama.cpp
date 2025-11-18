#!/usr/bin/env python3
"""
Simple RAG (Retrieval-Augmented Generation) System

This example demonstrates:
- Document loading and chunking
- Embedding generation
- Vector similarity search
- Context-augmented generation
"""

from llama_cpp import Llama
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
import pickle


class SimpleRAG:
    """Simple RAG implementation with in-memory vector store."""

    def __init__(
        self,
        llm_model_path: str,
        embedding_model_path: str = None,
        n_ctx: int = 4096
    ):
        """
        Initialize RAG system.

        Args:
            llm_model_path: Path to LLM model for generation
            embedding_model_path: Path to embedding model (if different)
            n_ctx: Context window size
        """
        print("Initializing RAG system...")

        # Initialize LLM for generation
        self.llm = Llama(
            model_path=llm_model_path,
            n_ctx=n_ctx,
            n_gpu_layers=35,
            verbose=False
        )

        # Initialize embedder
        if embedding_model_path:
            self.embedder = Llama(
                model_path=embedding_model_path,
                embedding=True,
                n_ctx=512,
                verbose=False
            )
        else:
            # Use same model for embeddings
            self.embedder = self.llm

        # Vector store
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict] = []

        print("RAG system initialized!")

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    def add_document(self, text: str, metadata: Dict = None, chunk: bool = True):
        """
        Add document to RAG system.

        Args:
            text: Document text
            metadata: Document metadata
            chunk: Whether to chunk the document
        """
        if chunk:
            chunks = self.chunk_text(text)
        else:
            chunks = [text]

        for i, chunk_text in enumerate(chunks):
            # Generate embedding
            embedding_result = self.embedder.create_embedding(chunk_text)
            embedding = np.array(embedding_result['data'][0]['embedding'])

            # Store
            self.documents.append(chunk_text)
            self.embeddings.append(embedding)

            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['chunk_index'] = i
            chunk_metadata['total_chunks'] = len(chunks)
            self.metadata.append(chunk_metadata)

    def add_documents_from_directory(self, directory: str):
        """
        Add all text files from a directory.

        Args:
            directory: Directory path
        """
        directory = Path(directory)

        for file_path in directory.glob('*.txt'):
            print(f"Adding {file_path.name}...")

            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            self.add_document(
                text,
                metadata={'source': str(file_path), 'filename': file_path.name}
            )

        print(f"Added {len(self.documents)} document chunks")

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float, Dict]]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (document, similarity_score, metadata) tuples
        """
        if not self.documents:
            return []

        # Embed query
        query_result = self.embedder.create_embedding(query)
        query_embedding = np.array(query_result['data'][0]['embedding'])

        # Compute similarities
        similarities = []
        for doc_embedding in self.embeddings:
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
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

    def generate(
        self,
        query: str,
        k: int = 3,
        max_tokens: int = 512,
        stream: bool = False
    ) -> str:
        """
        Generate answer using RAG.

        Args:
            query: User query
            k: Number of documents to retrieve
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response

        Returns:
            Generated answer
        """
        # Retrieve relevant documents
        results = self.search(query, k=k)

        if not results:
            # No documents, answer without context
            prompt = f"Question: {query}\n\nAnswer:"
        else:
            # Build context from retrieved documents
            context_parts = []
            for i, (doc, score, meta) in enumerate(results, 1):
                source = meta.get('filename', 'Unknown')
                context_parts.append(f"[Document {i}] (Source: {source})\n{doc}")

            context = "\n\n".join(context_parts)

            # Build prompt
            prompt = f"""Use the following context to answer the question. If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""

        # Generate response
        if stream:
            print("\nAssistant: ", end="", flush=True)
            full_response = ""

            response_stream = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                stream=True,
                stop=["Question:", "\n\n\n"]
            )

            for output in response_stream:
                token = output['choices'][0]['text']
                print(token, end="", flush=True)
                full_response += token

            print()
            return full_response.strip()
        else:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                stop=["Question:", "\n\n\n"]
            )

            return response['choices'][0]['text'].strip()

    def save_index(self, file_path: str):
        """Save vector index to disk."""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }

        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Index saved to {file_path}")

    def load_index(self, file_path: str):
        """Load vector index from disk."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self.metadata = data['metadata']

        print(f"Index loaded from {file_path}")
        print(f"Loaded {len(self.documents)} documents")


def main():
    """Example usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python 02_simple_rag_system.py <model_path> [documents_directory]")
        print("\nExample:")
        print("  python 02_simple_rag_system.py ./models/model.gguf ./documents")
        sys.exit(1)

    model_path = sys.argv[1]
    docs_dir = sys.argv[2] if len(sys.argv) > 2 else None

    # Initialize RAG
    rag = SimpleRAG(
        llm_model_path=model_path,
        n_ctx=4096
    )

    # Load documents
    if docs_dir:
        rag.add_documents_from_directory(docs_dir)
    else:
        # Add sample documents
        print("No documents directory provided. Using sample documents...")

        sample_docs = [
            {
                "text": "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
                "metadata": {"source": "python_intro.txt"}
            },
            {
                "text": "Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data. Common machine learning algorithms include linear regression, decision trees, neural networks, and support vector machines.",
                "metadata": {"source": "ml_basics.txt"}
            },
            {
                "text": "LLaMA (Large Language Model Meta AI) is a family of large language models developed by Meta AI. The models are designed to be efficient and accessible for research purposes. LLaMA models come in various sizes ranging from 7B to 65B parameters.",
                "metadata": {"source": "llama_info.txt"}
            }
        ]

        for doc in sample_docs:
            rag.add_document(doc["text"], doc["metadata"])

    # Interactive query loop
    print("\n" + "=" * 60)
    print("RAG System - Ask questions about your documents")
    print("Type 'quit' to exit")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("\nQuestion: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                break

            if not query:
                continue

            # Generate answer
            answer = rag.generate(query, k=3, stream=True)

            # Show sources
            print("\nSources:")
            results = rag.search(query, k=3)
            for i, (doc, score, meta) in enumerate(results, 1):
                source = meta.get('source', 'Unknown')
                print(f"  [{i}] {source} (relevance: {score:.3f})")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
