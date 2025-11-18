#!/usr/bin/env python3
"""
Simple RAG (Retrieval Augmented Generation) System
===================================================

A complete RAG implementation using llama.cpp for both embeddings and generation.

Requirements:
    pip install llama-cpp-python numpy faiss-cpu

Usage:
    python simple_rag_system.py --embed-model nomic-embed.gguf \\
                                 --llm-model llama-2-7b-chat.Q4_K_M.gguf \\
                                 --documents knowledge_base.txt
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from llama_cpp import Llama

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: FAISS not available. Using simple numpy search.")


class SimpleRAG:
    """Simple Retrieval Augmented Generation system"""

    def __init__(self, embedding_model_path, llm_model_path,
                 n_ctx=4096, n_gpu_layers=0):
        """
        Initialize RAG system

        Args:
            embedding_model_path: Path to embedding model GGUF
            llm_model_path: Path to language model GGUF
            n_ctx: Context length for LLM
            n_gpu_layers: GPU layers for LLM
        """
        print("Initializing RAG system...")

        # Load embedding model
        print(f"Loading embedding model: {embedding_model_path}")
        self.embed_model = Llama(
            model_path=embedding_model_path,
            embedding=True,
            n_ctx=512,
            verbose=False
        )

        # Load language model
        print(f"Loading language model: {llm_model_path}")
        self.llm = Llama(
            model_path=llm_model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )

        # Storage
        self.documents = []
        self.embeddings = None
        self.index = None
        self.n_embd = self.embed_model.n_embd()

        print("✓ RAG system initialized")

    def embed_text(self, text):
        """Generate embedding for text"""
        result = self.embed_model.create_embedding(text)
        embedding = np.array(result['data'][0]['embedding'], dtype=np.float32)

        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def add_documents(self, documents: List[str], show_progress=True):
        """
        Add documents to the knowledge base

        Args:
            documents: List of document texts
            show_progress: Show progress during indexing
        """
        print(f"\nIndexing {len(documents)} documents...")

        embeddings = []

        for i, doc in enumerate(documents):
            if show_progress and (i % 10 == 0 or i == len(documents) - 1):
                print(f"Progress: {i+1}/{len(documents)}", end='\r')

            embedding = self.embed_text(doc)
            embeddings.append(embedding)

            self.documents.append({
                'text': doc,
                'id': len(self.documents)
            })

        if show_progress:
            print()

        # Convert to numpy array
        self.embeddings = np.array(embeddings, dtype=np.float32)

        # Build FAISS index if available
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.n_embd)  # Inner product (cosine for normalized)
            self.index.add(self.embeddings)
            print(f"✓ Built FAISS index with {len(documents)} documents")
        else:
            print(f"✓ Indexed {len(documents)} documents (numpy search)")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve most relevant documents for query

        Args:
            query: Query text
            top_k: Number of documents to retrieve

        Returns:
            List of documents with relevance scores
        """
        if len(self.documents) == 0:
            return []

        # Generate query embedding
        query_emb = self.embed_text(query)

        # Search
        if HAS_FAISS and self.index is not None:
            # FAISS search
            scores, indices = self.index.search(
                query_emb.reshape(1, -1),
                min(top_k, len(self.documents))
            )
            scores = scores[0]
            indices = indices[0]
        else:
            # Numpy search
            scores = np.dot(self.embeddings, query_emb)
            indices = np.argsort(scores)[-top_k:][::-1]
            scores = scores[indices]

        # Gather results
        results = []
        for idx, score in zip(indices, scores):
            results.append({
                'text': self.documents[idx]['text'],
                'score': float(score),
                'id': self.documents[idx]['id']
            })

        return results

    def generate_answer(self, query: str, context_docs: List[Dict],
                       max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Generate answer using retrieved context

        Args:
            query: User query
            context_docs: Retrieved documents
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated answer text
        """
        # Build context
        context_texts = [doc['text'] for doc in context_docs]
        context = "\n\n".join([f"[{i+1}] {text}"
                              for i, text in enumerate(context_texts)])

        # Build prompt
        prompt = f"""Use the following context to answer the question. If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""

        # Generate
        response = self.llm.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["Question:", "\n\n\n", "Context:"]
        )

        return response['choices'][0]['text'].strip()

    def query(self, question: str, top_k: int = 3, max_tokens: int = 512,
             temperature: float = 0.7, show_context: bool = False) -> Dict:
        """
        Complete RAG query: retrieve + generate

        Args:
            question: User question
            top_k: Number of documents to retrieve
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            show_context: Include retrieved context in response

        Returns:
            dict with answer and metadata
        """
        # Retrieve relevant documents
        docs = self.retrieve(question, top_k)

        if len(docs) == 0:
            return {
                'answer': "No documents in knowledge base.",
                'context': [],
                'error': 'empty_kb'
            }

        # Generate answer
        answer = self.generate_answer(question, docs, max_tokens, temperature)

        result = {
            'answer': answer,
            'context': docs if show_context else None,
            'num_docs_retrieved': len(docs)
        }

        return result

    def save_index(self, path: str):
        """Save indexed knowledge base to disk"""
        save_data = {
            'documents': self.documents,
            'embeddings': self.embeddings.tolist(),
            'n_embd': self.n_embd
        }

        with open(path, 'w') as f:
            json.dump(save_data, f)

        print(f"✓ Saved index to {path}")

    def load_index(self, path: str):
        """Load indexed knowledge base from disk"""
        with open(path, 'r') as f:
            save_data = json.load(f)

        self.documents = save_data['documents']
        self.embeddings = np.array(save_data['embeddings'], dtype=np.float32)
        self.n_embd = save_data['n_embd']

        # Rebuild FAISS index if available
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.n_embd)
            self.index.add(self.embeddings)

        print(f"✓ Loaded index with {len(self.documents)} documents")


def load_documents_from_file(file_path: str) -> List[str]:
    """Load documents from text file (one per line)"""
    documents = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(line)

    return documents


def main():
    parser = argparse.ArgumentParser(
        description="Simple RAG system using llama.cpp"
    )
    parser.add_argument(
        "--embed-model",
        required=True,
        help="Path to embedding model GGUF"
    )
    parser.add_argument(
        "--llm-model",
        required=True,
        help="Path to language model GGUF"
    )
    parser.add_argument(
        "--documents",
        help="Path to documents file (one per line)"
    )
    parser.add_argument(
        "--load-index",
        help="Load pre-built index from file"
    )
    parser.add_argument(
        "--save-index",
        help="Save index to file after building"
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="Number of GPU layers for LLM"
    )
    parser.add_argument(
        "--interactive",
        action='store_true',
        help="Interactive query mode"
    )

    args = parser.parse_args()

    # Initialize RAG
    rag = SimpleRAG(
        embedding_model_path=args.embed_model,
        llm_model_path=args.llm_model,
        n_gpu_layers=args.n_gpu_layers
    )

    # Load or build index
    if args.load_index:
        rag.load_index(args.load_index)
    elif args.documents:
        documents = load_documents_from_file(args.documents)
        rag.add_documents(documents)

        if args.save_index:
            rag.save_index(args.save_index)
    else:
        print("Error: Must provide --documents or --load-index")
        return

    # Interactive mode
    if args.interactive:
        print("\n" + "="*80)
        print("RAG INTERACTIVE MODE")
        print("="*80)
        print("Enter questions (or 'quit' to exit)\n")

        while True:
            try:
                question = input("Question: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    break

                if not question:
                    continue

                # Query RAG
                print("\nSearching knowledge base...")
                result = rag.query(question, top_k=3, show_context=True)

                # Display answer
                print("\n" + "-"*80)
                print("ANSWER:")
                print("-"*80)
                print(result['answer'])

                # Display context
                if result['context']:
                    print("\n" + "-"*80)
                    print("RETRIEVED CONTEXT:")
                    print("-"*80)
                    for i, doc in enumerate(result['context'], 1):
                        print(f"\n[{i}] Score: {doc['score']:.4f}")
                        print(f"    {doc['text'][:200]}...")

                print("\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}\n")

    else:
        # Single query demo
        test_questions = [
            "What is the capital of France?",
            "Explain quantum computing",
            "What is machine learning?"
        ]

        print("\n" + "="*80)
        print("DEMO QUERIES")
        print("="*80)

        for question in test_questions:
            print(f"\nQuestion: {question}")
            result = rag.query(question, top_k=2)
            print(f"Answer: {result['answer']}")
            print("-"*80)


if __name__ == "__main__":
    main()


# Example usage:
"""
# Build index from documents
python simple_rag_system.py \\
    --embed-model nomic-embed-text-v1.5.Q4_K_M.gguf \\
    --llm-model llama-2-7b-chat.Q4_K_M.gguf \\
    --documents knowledge_base.txt \\
    --save-index kb.index.json \\
    --interactive

# Load existing index
python simple_rag_system.py \\
    --embed-model nomic-embed-text-v1.5.Q4_K_M.gguf \\
    --llm-model llama-2-7b-chat.Q4_K_M.gguf \\
    --load-index kb.index.json \\
    --n-gpu-layers 35 \\
    --interactive

# Sample knowledge_base.txt:
# Paris is the capital and largest city of France.
# Machine learning is a subset of artificial intelligence.
# Quantum computing uses quantum-mechanical phenomena for computation.
# Python is a high-level programming language.
# The Eiffel Tower is located in Paris, France.
"""
