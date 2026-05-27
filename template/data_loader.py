"""
Data loader: stream MS MARCO from BeIR, build subsets for scaling experiments.

What it does for you:
  - Loads qrels + queries + corpus from HuggingFace
  - Picks N eval queries with ground truth
  - Streams the 8.84M corpus, keeps relevant docs + N distractors
  - Returns reproducible subsets via fixed seed

What you need to do:
  - Decide N_EVAL_QUERIES, DISTRACTOR_TARGET, SUBSET_SIZES
  - Call build_subset(size) inside your scaling loop
  - Cache the corpus to disk so you don't re-stream every run
"""
import json
import random
from pathlib import Path
from typing import Optional
from datasets import load_dataset


SEED = 42


def load_qrels_and_queries(split: str = "validation") -> tuple[dict, dict]:
    """Load qrels (query_id -> set of relevant doc_ids) and queries (id -> text)."""
    qrels_ds = load_dataset("BeIR/msmarco-qrels", split=split)
    qrels: dict[str, set[str]] = {}
    for row in qrels_ds:
        qrels.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))

    queries_ds = load_dataset("BeIR/msmarco", "queries", split="queries")
    queries = {row["_id"]: row["text"] for row in queries_ds}
    return qrels, queries


def pick_eval_queries(qrels: dict, queries: dict, n: int) -> tuple[list[dict], set[str]]:
    """Pick N queries that have at least one relevant doc. Returns (eval_set, all_relevant_ids)."""
    eval_set = []
    relevant_ids: set[str] = set()
    for qid, rel in qrels.items():
        if qid in queries and rel:
            eval_set.append({"qid": qid, "query": queries[qid], "relevant_ids": list(rel)})
            relevant_ids.update(rel)
        if len(eval_set) >= n:
            break
    return eval_set, relevant_ids


def build_corpus_pool(relevant_ids: set[str], n_distractors: int) -> list[dict]:
    """
    Download MS MARCO corpus (non-streaming), keep all relevant docs +
    first n_distractors others in dataset order. Cached locally by HuggingFace
    (~1.5 GB one-time download), then reads from disk — much faster and more
    reliable than streaming over HTTP.

    FIX vs template: switched from streaming=True to streaming=False to avoid
    HuggingFace HTTP timeout errors on large parquet files. Trade-off: requires
    ~1.5 GB disk space for the HF cache, but iteration is then O(local disk) and
    doesn't time out.

    Note: distractors are deterministic by HF dataset order (not seeded).
    Reproducible random sampling happens later in build_subset() with SEED.
    """
    import os
    # Increase HF download timeout (default 10s is too short for large files)
    os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "300")

    print("  Downloading corpus parquet to HF cache (~1.5 GB, one-time)...")
    corpus_ds = load_dataset("BeIR/msmarco", "corpus", split="corpus")  # non-streaming
    relevant_docs: dict[str, str] = {}
    distractors: list[dict] = []
    for row in corpus_ds:
        did = str(row["_id"])
        text = row.get("text") or row.get("title", "")
        if did in relevant_ids:
            relevant_docs[did] = text
        elif len(distractors) < n_distractors:
            distractors.append({"id": did, "text": text})
        if len(relevant_docs) == len(relevant_ids) and len(distractors) >= n_distractors:
            break

    if len(relevant_docs) < len(relevant_ids):
        missing = len(relevant_ids) - len(relevant_docs)
        print(f"  ⚠️  {missing} relevant docs not found in corpus (qrel mismatch — OK, continuing)")

    pool = [{"id": did, "text": txt} for did, txt in relevant_docs.items()]
    pool.extend(distractors)
    return pool


def save_cache(pool: list[dict], eval_set: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"corpus": pool, "eval_set": eval_set}, open(path, "w"))


def load_cache(path: Path) -> tuple[list[dict], list[dict]]:
    data = json.load(open(path))
    return data["corpus"], data["eval_set"]


def build_subset(pool: list[dict], eval_set: list[dict], size: int, seed: int = SEED) -> list[dict]:
    """
    Build a reproducible subset of `size` docs that includes ALL relevant docs
    from eval_set + random distractors. Same seed -> same subset.
    """
    relevant_ids = {rid for e in eval_set for rid in e["relevant_ids"]}
    relevant = [d for d in pool if d["id"] in relevant_ids]
    distractors = [d for d in pool if d["id"] not in relevant_ids]

    rng = random.Random(seed)
    rng.shuffle(distractors)

    n_dist = max(size - len(relevant), 0)
    if n_dist > len(distractors):
        raise ValueError(
            f"Need {n_dist} distractors but pool has only {len(distractors)}. "
            f"Increase distractor_target in build_corpus_pool()."
        )
    return relevant + distractors[:n_dist]


if __name__ == "__main__":
    # Example wiring — edit the constants and run once to cache the corpus
    N_EVAL_QUERIES = 100
    DISTRACTOR_TARGET = 300_000
    CACHE_PATH = Path(__file__).parent / "cache" / "corpus.json"

    if CACHE_PATH.exists():
        print(f"Cache exists at {CACHE_PATH}, skipping rebuild.")
    else:
        print("Loading qrels + queries...")
        qrels, queries = load_qrels_and_queries()
        eval_set, relevant_ids = pick_eval_queries(qrels, queries, N_EVAL_QUERIES)
        print(f"Picked {len(eval_set)} eval queries with {len(relevant_ids)} relevant docs.")

        print(f"Streaming corpus (~5-10 min)...")
        pool = build_corpus_pool(relevant_ids, DISTRACTOR_TARGET)
        print(f"Pool: {len(pool)} docs")

        save_cache(pool, eval_set, CACHE_PATH)
        print(f"Cached to {CACHE_PATH}")
