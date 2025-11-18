#!/usr/bin/env python3
"""
Embedding Generator
===================

Generate embeddings for text documents using llama.cpp embedding models.
Supports batch processing, caching, and various output formats.

Requirements:
    pip install llama-cpp-python numpy

Usage:
    python embedding_generator.py --model nomic-embed.gguf --input documents.txt
"""

import argparse
import json
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from llama_cpp import Llama


class EmbeddingGenerator:
    """Generate and manage text embeddings"""

    def __init__(self, model_path, n_ctx=512, n_batch=512, cache_dir=None):
        """
        Initialize embedding model

        Args:
            model_path: Path to embedding model GGUF file
            n_ctx: Context length
            n_batch: Batch size for processing
            cache_dir: Directory for caching embeddings (None = no caching)
        """
        print(f"Loading embedding model: {model_path}")

        self.model = Llama(
            model_path=model_path,
            embedding=True,  # Enable embedding mode
            n_ctx=n_ctx,
            n_batch=n_batch,
            verbose=False
        )

        self.n_embd = self.model.n_embd()
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"✓ Model loaded (embedding dimension: {self.n_embd})")

    def _get_cache_path(self, text):
        """Get cache file path for text"""
        if not self.cache_dir:
            return None

        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache_dir / f"{text_hash}.emb"

    def _load_from_cache(self, text):
        """Load embedding from cache if available"""
        cache_path = self._get_cache_path(text)

        if cache_path and cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        return None

    def _save_to_cache(self, text, embedding):
        """Save embedding to cache"""
        cache_path = self._get_cache_path(text)

        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)

    def generate_embedding(self, text, normalize=True, use_cache=True):
        """
        Generate embedding for a single text

        Args:
            text: Input text
            normalize: Whether to L2-normalize the embedding
            use_cache: Whether to use cached embeddings

        Returns:
            numpy array of shape (n_embd,)
        """
        # Check cache
        if use_cache:
            cached = self._load_from_cache(text)
            if cached is not None:
                return cached

        # Generate embedding
        result = self.model.create_embedding(text)
        embedding = np.array(result['data'][0]['embedding'], dtype=np.float32)

        # Normalize
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        # Cache
        if use_cache:
            self._save_to_cache(text, embedding)

        return embedding

    def generate_batch_embeddings(self, texts, normalize=True, use_cache=True,
                                  show_progress=True):
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of input texts
            normalize: Whether to L2-normalize embeddings
            use_cache: Whether to use cached embeddings
            show_progress: Show progress bar

        Returns:
            numpy array of shape (len(texts), n_embd)
        """
        embeddings = []

        for i, text in enumerate(texts):
            if show_progress and (i % 10 == 0 or i == len(texts) - 1):
                print(f"Processing: {i+1}/{len(texts)}", end='\r')

            embedding = self.generate_embedding(text, normalize, use_cache)
            embeddings.append(embedding)

        if show_progress:
            print()  # New line after progress

        return np.array(embeddings)

    def save_embeddings(self, embeddings, output_path, format='npy'):
        """
        Save embeddings to file

        Args:
            embeddings: numpy array of embeddings
            output_path: Output file path
            format: Format ('npy', 'npz', 'json', 'txt')
        """
        output_path = Path(output_path)

        if format == 'npy':
            np.save(output_path, embeddings)
        elif format == 'npz':
            np.savez_compressed(output_path, embeddings=embeddings)
        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump(embeddings.tolist(), f)
        elif format == 'txt':
            np.savetxt(output_path, embeddings, fmt='%.6f')
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"✓ Saved embeddings to {output_path}")

    def cosine_similarity(self, emb1, emb2):
        """
        Compute cosine similarity between two embeddings

        Args:
            emb1, emb2: Embedding vectors (assumed normalized)

        Returns:
            Similarity score in [0, 1] (or [-1, 1] if not normalized)
        """
        return np.dot(emb1, emb2)

    def find_similar(self, query_embedding, corpus_embeddings, top_k=5):
        """
        Find most similar embeddings to query

        Args:
            query_embedding: Query embedding vector
            corpus_embeddings: Array of corpus embeddings (N, n_embd)
            top_k: Number of top results to return

        Returns:
            List of (index, similarity) tuples
        """
        # Compute similarities (assuming normalized embeddings)
        similarities = np.dot(corpus_embeddings, query_embedding)

        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]

        return list(zip(top_indices, top_scores))


def load_documents(input_path):
    """
    Load documents from file

    Supports:
        - Text file (one document per line)
        - JSONL (one JSON object per line with 'text' field)
    """
    documents = []
    input_path = Path(input_path)

    with open(input_path, 'r', encoding='utf-8') as f:
        if input_path.suffix == '.jsonl':
            for line in f:
                obj = json.loads(line)
                documents.append(obj.get('text', ''))
        else:
            for line in f:
                line = line.strip()
                if line:
                    documents.append(line)

    return documents


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for text documents"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to embedding model GGUF file"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input file (text or JSONL)"
    )
    parser.add_argument(
        "--output",
        help="Output file for embeddings (default: input.npy)"
    )
    parser.add_argument(
        "--format",
        choices=['npy', 'npz', 'json', 'txt'],
        default='npy',
        help="Output format"
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory for caching embeddings"
    )
    parser.add_argument(
        "--no-normalize",
        action='store_true',
        help="Don't L2-normalize embeddings"
    )
    parser.add_argument(
        "--similarity-test",
        action='store_true',
        help="Run similarity test after generation"
    )

    args = parser.parse_args()

    # Load documents
    print(f"Loading documents from {args.input}...")
    documents = load_documents(args.input)
    print(f"✓ Loaded {len(documents)} documents")

    # Initialize generator
    generator = EmbeddingGenerator(
        model_path=args.model,
        cache_dir=args.cache_dir
    )

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generator.generate_batch_embeddings(
        documents,
        normalize=not args.no_normalize,
        use_cache=args.cache_dir is not None
    )

    print(f"✓ Generated embeddings: {embeddings.shape}")

    # Save embeddings
    output_path = args.output or f"{Path(args.input).stem}.{args.format}"
    generator.save_embeddings(embeddings, output_path, format=args.format)

    # Save document index
    index_path = Path(output_path).with_suffix('.index.json')
    with open(index_path, 'w') as f:
        json.dump({
            'documents': documents,
            'embedding_dim': generator.n_embd,
            'normalized': not args.no_normalize,
            'model': args.model
        }, f, indent=2)
    print(f"✓ Saved document index to {index_path}")

    # Similarity test
    if args.similarity_test and len(documents) >= 2:
        print("\n" + "="*80)
        print("SIMILARITY TEST")
        print("="*80)

        # Compare first document to all others
        query_emb = embeddings[0]
        query_doc = documents[0]

        print(f"\nQuery document: {query_doc[:100]}...")
        print("\nMost similar documents:")

        similar = generator.find_similar(query_emb, embeddings[1:], top_k=min(5, len(documents)-1))

        for rank, (idx, score) in enumerate(similar, 1):
            doc_idx = idx + 1  # Offset by 1 (we excluded first doc)
            doc = documents[doc_idx]
            print(f"\n{rank}. Similarity: {score:.4f}")
            print(f"   {doc[:100]}...")


if __name__ == "__main__":
    main()


# Example usage:
"""
# Generate embeddings from text file
python embedding_generator.py \\
    --model nomic-embed-text-v1.5.Q4_K_M.gguf \\
    --input documents.txt \\
    --output embeddings.npy \\
    --cache-dir ./cache

# With similarity test
python embedding_generator.py \\
    --model bge-base-en-v1.5.Q4_K_M.gguf \\
    --input corpus.jsonl \\
    --format npz \\
    --similarity-test

# Generate without normalization
python embedding_generator.py \\
    --model e5-large-v2.Q4_K_M.gguf \\
    --input articles.txt \\
    --no-normalize \\
    --format json
"""
