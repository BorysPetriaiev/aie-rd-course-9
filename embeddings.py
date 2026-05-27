"""
Embedding module: OpenAI text-embedding-3-small with two-level disk caching.

Caching strategy:
  Level 1 — per-batch: cache/batches/<hash>.npy  (survives partial API failures)
  Level 2 — aggregated: passed as cache_file arg   (fast reload on re-runs)

Rate-limit handling:
  Automatic retry with backoff on HTTP 429. Extracts the exact wait time from
  the error message when available, otherwise uses exponential backoff (1→60s).

Usage:
  export OPENAI_API_KEY=sk-...    (or put in .env)
  from embeddings import embed_texts
"""
import os
import re
import json
import time
import hashlib
import numpy as np
from pathlib import Path

BATCH_DIR = Path("cache/batches")
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSIONS = 512   # Matryoshka: good quality, ~614 MB for 300K docs
BATCH_SIZE = 500            # Safe limit for OpenAI API (max 2048)
MAX_RETRIES = 8             # Retry attempts on rate-limit errors


def _client():
    """Build OpenAI client, raise early with a helpful message if key is missing."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set.\n"
            "  export OPENAI_API_KEY=sk-...   or add it to .env"
        )
    return openai.OpenAI(api_key=api_key)


def _embed_batch_with_retry(
    client,
    batch: list[str],
    model: str,
    dimensions: int,
) -> np.ndarray:
    """
    Call OpenAI embeddings API for one batch.
    Retries up to MAX_RETRIES times on HTTP 429 (rate limit), using the
    wait time from the error message when available, else exponential backoff.
    """
    import openai  # already installed; imported here to avoid top-level dep

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.embeddings.create(input=batch, model=model, dimensions=dimensions)
            return np.array([e.embedding for e in resp.data], dtype=np.float32)

        except openai.RateLimitError as exc:
            if attempt == MAX_RETRIES - 1:
                raise  # exhausted retries

            # OpenAI often tells us exactly how long to wait, e.g. "try again in 1.5s"
            match = re.search(r"try again in (\d+(?:\.\d+)?)s", str(exc))
            if match:
                wait = float(match.group(1)) + 0.2   # add small buffer
            else:
                wait = min(2 ** attempt, 60)          # exponential backoff up to 60s

            print(f"\n  ⏳ Rate limit (attempt {attempt + 1}/{MAX_RETRIES}), "
                  f"waiting {wait:.1f}s…", flush=True)
            time.sleep(wait)

        except openai.APIError as exc:
            # Transient server errors — back off and retry
            if attempt == MAX_RETRIES - 1:
                raise
            wait = min(2 ** attempt, 30)
            print(f"\n  ⚠️  API error ({exc}), retrying in {wait}s…", flush=True)
            time.sleep(wait)


def embed_texts(
    texts: list[str],
    cache_file: Path | None = None,
    model: str = DEFAULT_MODEL,
    dimensions: int = DEFAULT_DIMENSIONS,
    batch_size: int = BATCH_SIZE,
    label: str = "",
) -> np.ndarray:
    """
    Embed a list of texts via OpenAI API.

    Args:
        texts:      List of strings to embed.
        cache_file: If given and exists, load from there (fast path).
                    If given and doesn't exist, save result there after embedding.
        model:      OpenAI embedding model name.
        dimensions: Reduced-dimension size (Matryoshka). Default 512.
        batch_size: API calls per batch.
        label:      Label for progress output (e.g. "corpus" or "queries").

    Returns:
        float32 ndarray of shape (len(texts), dimensions).
    """
    # ── Fast path: aggregated cache exists ──────────────────────────────────
    if cache_file and cache_file.exists():
        arr = np.load(cache_file)
        print(f"  ✅ Loaded {label or 'embeddings'}: {arr.shape} from {cache_file}")
        return arr

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    client = _client()
    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size
    all_parts: list[np.ndarray] = []

    # ── Per-batch embedding with individual batch cache ──────────────────────
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]

        # Cache key encodes content + model + dims so changing any invalidates it
        key_src = json.dumps({"texts": batch, "model": model, "dims": dimensions})
        key = hashlib.sha256(key_src.encode()).hexdigest()[:20]
        batch_cache = BATCH_DIR / f"{key}.npy"

        if batch_cache.exists():
            part = np.load(batch_cache)
        else:
            part = _embed_batch_with_retry(client, batch, model, dimensions)
            np.save(batch_cache, part)

        all_parts.append(part)
        batch_num = i // batch_size + 1
        done = min(i + batch_size, total)
        print(f"  [{batch_num}/{total_batches}] {label + ' ' if label else ''}Embedded {done:,}/{total:,}", end="\r")

    print()
    result = np.vstack(all_parts)

    # ── Save aggregated cache ────────────────────────────────────────────────
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, result)
        print(f"  💾 Saved {result.shape} to {cache_file} ({result.nbytes / 1e6:.0f} MB)")

    return result
