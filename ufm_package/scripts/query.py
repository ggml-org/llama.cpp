#!/usr/bin/env python3
"""
query.py — Query the coding RAG with Gemma 4.

Usage:
    python query.py "How do I use subgroupShuffleXor in ROCm HIP?"
    python query.py "What is the Q4_K block format?" --top-k 8
    python query.py --interactive   # REPL mode

The system:
  1. Embeds your question with nomic-embed-text
  2. Retrieves top-K relevant chunks from ChromaDB (weighted by source priority)
  3. Passes chunks + question to Gemma 4 (26B MoE) via Ollama
  4. Streams the answer

Dependencies:
    pip install chromadb ollama
    Ollama must have: gemma4:26b  (or gemma4:31b)
                      nomic-embed-text
"""

import argparse
import sys
from pathlib import Path

try:
    import chromadb
    import ollama
except ImportError:
    sys.exit("Run: pip install chromadb ollama")

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH      = Path(__file__).parent / "db"
EMBED_MODEL  = "nomic-embed-text"
GEN_MODEL    = "gemma4:26b"          # swap to gemma4:31b if you have headroom
TOP_K        = 6
MAX_CTX_CHARS = 8000                 # keep context under ~2K tokens

# ── System prompt — tuned for technical accuracy ──────────────────────────────
SYSTEM_PROMPT = """You are a GPU/systems programming assistant with deep knowledge of:
- AMD ROCm, HIP, and RDNA4 architecture (gfx1201 / RX 9070 XT)
- Vulkan compute shaders and GLSL, including cooperative matrix extensions
- llama.cpp internals, GGUF quantisation formats (Q4_K, Q8_0, etc.)
- Low-level GPU memory management and kernel optimisation

When answering:
- Be precise about API names, types, and byte layouts.
- If you cite something from the provided context, say so.
- If the context doesn't contain enough information, say so clearly — do not guess.
- Prefer code examples when they add clarity.
- Note version-specific behaviour when it matters (e.g. ROCm 7.2.1 vs 7.0)."""

# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Embed the question, query ChromaDB, return ranked results."""
    client     = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection("coding_docs")

    # Embed question
    q_embed = ollama.embed(model=EMBED_MODEL, input=question).embeddings[0]

    # Query — get more than top_k so we can re-weight
    results = collection.query(
        query_embeddings = [q_embed],
        n_results        = min(top_k * 3, 30),
        include          = ["documents", "metadatas", "distances"],
    )

    # Re-rank: combine cosine similarity with source weight
    # distance is cosine distance (0=identical), so similarity = 1 - distance
    items = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1.0 - dist
        weight     = meta.get("weight", 1.0)
        score      = similarity * weight
        items.append({
            "text":   doc,
            "source": meta.get("source", "unknown"),
            "url":    meta.get("source_url", ""),
            "score":  score,
        })

    # Sort by combined score, take top_k
    items.sort(key=lambda x: x["score"], reverse=True)
    return items[:top_k]

# ── Context building ──────────────────────────────────────────────────────────

def build_context(chunks: list[dict], max_chars: int = MAX_CTX_CHARS) -> str:
    parts = []
    total = 0
    for c in chunks:
        header = f"[{c['source']}]\n"
        block  = header + c["text"] + "\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n---\n".join(parts)

# ── Generation ────────────────────────────────────────────────────────────────

def answer(question: str, top_k: int = TOP_K, show_sources: bool = True):
    """Retrieve relevant docs and generate a streamed answer."""
    try:
        chunks = retrieve(question, top_k)
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        print("Did you run ingest.py first?")
        sys.exit(1)

    if show_sources:
        print("\n── Sources ──────────────────────────────────────────────────")
        for i, c in enumerate(chunks, 1):
            print(f"  {i}. {c['source']}  (score {c['score']:.3f})")
        print("────────────────────────────────────────────────────────────\n")

    context = build_context(chunks)
    user_msg = f"""Here is relevant technical documentation:

{context}

---
Question: {question}"""

    # Stream the response
    print("── Answer ───────────────────────────────────────────────────")
    stream = ollama.chat(
        model    = GEN_MODEL,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        stream   = True,
        options  = {
            "temperature": 0.1,   # low temp for technical accuracy
            "num_ctx":     16384, # context window for generation
        }
    )
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print("\n────────────────────────────────────────────────────────────\n")

# ── Interactive REPL ──────────────────────────────────────────────────────────

def repl(top_k: int = TOP_K):
    print(f"Coding RAG — {GEN_MODEL} + nomic-embed-text")
    print("Type your question. 'quit' to exit. 'sources off' to hide source list.\n")
    show_sources = True
    while True:
        try:
            q = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break
        if q.lower() == "sources off":
            show_sources = False
            print("Sources hidden.")
            continue
        if q.lower() == "sources on":
            show_sources = True
            print("Sources visible.")
            continue
        answer(q, top_k=top_k, show_sources=show_sources)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Query the coding RAG")
    p.add_argument("question", nargs="?", help="Question to ask")
    p.add_argument("--top-k", type=int, default=TOP_K, help=f"Chunks to retrieve (default {TOP_K})")
    p.add_argument("--interactive", "-i", action="store_true", help="REPL mode")
    p.add_argument("--no-sources", action="store_true", help="Don't print source list")
    p.add_argument("--model", default=GEN_MODEL, help=f"Ollama model to use (default {GEN_MODEL})")
    args = p.parse_args()

    GEN_MODEL = args.model

    if args.interactive:
        repl(top_k=args.top_k)
    elif args.question:
        answer(args.question, top_k=args.top_k, show_sources=not args.no_sources)
    else:
        p.print_help()
