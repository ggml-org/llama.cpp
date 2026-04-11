#!/usr/bin/env python3
"""
ingest.py — Crawl technical docs, embed, store in ChromaDB.

Usage:
    python ingest.py                    # index all sources
    python ingest.py --source rocm      # index only sources tagged 'rocm'
    python ingest.py --refresh          # re-fetch and overwrite existing

Dependencies:
    pip install chromadb requests beautifulsoup4 ollama pyyaml
    Ollama must be running with: ollama pull nomic-embed-text
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
import yaml
from bs4 import BeautifulSoup

# ── Try importing chromadb and ollama; give clear error if missing ────────────
try:
    import chromadb
except ImportError:
    sys.exit("Run: pip install chromadb")

try:
    import ollama
except ImportError:
    sys.exit("Run: pip install ollama")

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH        = Path(__file__).parent / "db"
SOURCES_YAML   = Path(__file__).parent / "config" / "sources.yaml"
EMBED_MODEL    = "nomic-embed-text"
CHUNK_SIZE     = 600    # characters
CHUNK_OVERLAP  = 100
BATCH_SIZE     = 32     # embeddings per API call

# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks. Tries to break on newlines."""
    chunks = []
    start  = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            # Try to break on a newline near the chunk boundary
            nl = text.rfind("\n", start, end)
            if nl > start + size // 2:
                end = nl
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

# ── Fetching ──────────────────────────────────────────────────────────────────

def fetch_text(url: str) -> Optional[str]:
    """Fetch URL and extract plain text. Handles HTML and raw text."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; CodingRAG/1.0)"}
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "html" in ct:
            soup = BeautifulSoup(r.text, "html.parser")
            # Remove nav, footer, script, style
            for tag in soup(["nav","footer","script","style","header"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        else:
            return r.text
    except Exception as e:
        print(f"  [WARN] Failed to fetch {url}: {e}")
        return None

# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via Ollama."""
    resp = ollama.embed(model=EMBED_MODEL, input=texts)
    return resp.embeddings

# ── Main ingest ───────────────────────────────────────────────────────────────

def ingest(filter_tag: Optional[str] = None, refresh: bool = False):
    # Load sources
    with open(SOURCES_YAML) as f:
        config = yaml.safe_load(f)
    sources = config["sources"]

    if filter_tag:
        sources = [s for s in sources if filter_tag in s.get("tags", [])]
        print(f"Filtered to {len(sources)} sources with tag '{filter_tag}'")

    # Connect to ChromaDB
    client     = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name="coding_docs",
        metadata={"hnsw:space": "cosine"}
    )

    total_chunks = 0

    for source in sources:
        name   = source["name"]
        url    = source["url"]
        weight = source.get("weight", 1.0)
        tags   = source.get("tags", [])

        print(f"\n→ {name}")
        print(f"  {url}")

        # Check if already indexed (unless --refresh)
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        existing = collection.get(where={"source_hash": url_hash}, limit=1)
        if existing["ids"] and not refresh:
            print(f"  [SKIP] Already indexed. Use --refresh to update.")
            continue

        # Delete old chunks for this source
        if refresh:
            old = collection.get(where={"source_hash": url_hash})
            if old["ids"]:
                collection.delete(ids=old["ids"])
                print(f"  [DEL] Removed {len(old['ids'])} old chunks")

        # Fetch and chunk
        text = fetch_text(url)
        if not text:
            continue

        chunks = chunk_text(text)
        print(f"  {len(text):,} chars → {len(chunks)} chunks")

        # Embed and store in batches
        for i in range(0, len(chunks), BATCH_SIZE):
            batch  = chunks[i : i + BATCH_SIZE]
            embeds = embed_batch(batch)

            ids  = [f"{url_hash}_{i+j}" for j in range(len(batch))]
            metas = [{
                "source":      name,
                "source_url":  url,
                "source_hash": url_hash,
                "weight":      weight,
                "tags":        ",".join(tags),
                "chunk_idx":   i + j,
            } for j in range(len(batch))]

            collection.add(
                ids        = ids,
                embeddings = embeds,
                documents  = batch,
                metadatas  = metas,
            )
            print(f"  [{i+len(batch):>4}/{len(chunks)}] embedded", end="\r")
            time.sleep(0.05)  # be gentle on Ollama

        total_chunks += len(chunks)
        print(f"  done — {len(chunks)} chunks indexed")

    print(f"\n✓ Total: {total_chunks} new chunks across {len(sources)} sources")
    print(f"  DB at: {DB_PATH}")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Index technical docs for coding RAG")
    p.add_argument("--source", help="Only index sources with this tag")
    p.add_argument("--refresh", action="store_true", help="Re-fetch and overwrite")
    args = p.parse_args()
    ingest(filter_tag=args.source, refresh=args.refresh)
