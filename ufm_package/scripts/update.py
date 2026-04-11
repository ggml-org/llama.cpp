#!/usr/bin/env python3
"""
update.py — Refresh stale sources.

Checks each source for changes (via ETag / Last-Modified headers for web,
or latest commit hash for GitHub raw files) and only re-ingests what changed.

Usage:
    python update.py                    # check and update all
    python update.py --source rocm      # check only rocm-tagged sources
    python update.py --force            # re-ingest everything regardless
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import requests
import yaml

try:
    import chromadb
    import ollama
except ImportError:
    sys.exit("Run: pip install chromadb ollama")

CACHE_FILE  = Path(__file__).parent / "db" / ".source_cache.json"
SOURCES_YAML = Path(__file__).parent / "config" / "sources.yaml"

def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}

def save_cache(cache: dict):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))

def get_remote_fingerprint(url: str) -> str:
    """Return ETag or Last-Modified or content hash — whatever the server gives."""
    try:
        r = requests.head(url, timeout=10,
                          headers={"User-Agent": "CodingRAG/1.0"})
        etag = r.headers.get("ETag", "")
        lmod = r.headers.get("Last-Modified", "")
        if etag:
            return f"etag:{etag}"
        if lmod:
            return f"lmod:{lmod}"
        # Fallback: fetch and hash content (slower)
        r2 = requests.get(url, timeout=20,
                          headers={"User-Agent": "CodingRAG/1.0"})
        return "hash:" + hashlib.md5(r2.content).hexdigest()
    except Exception as e:
        return f"err:{e}"

def update(filter_tag=None, force=False):
    with open(SOURCES_YAML) as f:
        config = yaml.safe_load(f)
    sources = config["sources"]

    if filter_tag:
        sources = [s for s in sources if filter_tag in s.get("tags", [])]

    cache = load_cache()
    stale = []

    print("Checking sources for changes...")
    for source in sources:
        url  = source["url"]
        name = source["name"]
        fp   = get_remote_fingerprint(url)
        cached_fp = cache.get(url, "")

        if force or fp != cached_fp:
            status = "FORCE" if force else ("NEW" if not cached_fp else "CHANGED")
            print(f"  [{status}] {name}")
            stale.append(source)
            cache[url] = fp
        else:
            print(f"  [OK]    {name}")
        time.sleep(0.2)

    if not stale:
        print("\nAll sources up to date.")
        return

    print(f"\n{len(stale)} source(s) to update. Running ingest...")

    # Import and call ingest directly
    sys.path.insert(0, str(Path(__file__).parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ingest", Path(__file__).parent / "ingest.py")
    ingest_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ingest_mod)

    for source in stale:
        print(f"\nRe-indexing: {source['name']}")
        ingest_mod.ingest_one(source, refresh=True)

    save_cache(cache)
    print("\nDone. Cache updated.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", help="Only check sources with this tag")
    p.add_argument("--force", action="store_true", help="Re-ingest all")
    args = p.parse_args()
    update(filter_tag=args.source, force=args.force)
