"""
Main scaling experiment: "Find the Breaking Point" in RAG retrieval.

Pipeline:
  1. Load MS MARCO data via template/data_loader.py  (cached to cache/corpus.json)
  2. Embed the full pool once via OpenAI API         (cached to cache/pool_embeddings.npy)
  3. Embed all eval queries                           (cached to cache/query_embeddings.npy)
  4. For each corpus size [1K, 10K, 100K, 300K]:
       a. Build subset  (template/data_loader.build_subset)
       b. Baseline  — DenseRetriever (numpy brute-force)
       c. Fix       — HybridRetriever (BM25 + dense + RRF)
       d. Measure   — Recall@1, Recall@10, MRR@10, latency p50/p95/p99, RAM
  5. Save results to results/results.json
  6. Generate plots via visualize.py

Prerequisites:
  pip install -r requirements.txt
  export OPENAI_API_KEY=sk-...
"""
import os
# ── Load .env before anything else ──────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional; use: export OPENAI_API_KEY=... as fallback

# ── Set HuggingFace timeouts before any datasets import ─────────────────────
# Default (10s) is too short for large parquet files on slow connections.
os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "300")    # 5 min per request
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

import sys
import json
import time
import psutil
import numpy as np
from pathlib import Path

# ── Import template modules ──────────────────────────────────────────────────
TEMPLATE_DIR = Path(__file__).parent / "template"
sys.path.insert(0, str(TEMPLATE_DIR))

from data_loader import (
    load_qrels_and_queries,
    pick_eval_queries,
    build_corpus_pool,
    save_cache,
    load_cache,
    build_subset,
)
from metrics import evaluate

from embeddings import embed_texts
from retriever import DenseRetriever, HybridRetriever


# ─── Configuration ───────────────────────────────────────────────────────────

CORPUS_SIZES      = [1_000, 10_000, 100_000, 300_000]
N_EVAL_QUERIES    = 100          # Number of eval queries (keep ≥50 for stable metrics)
DISTRACTOR_TARGET = 300_000      # Pool size: relevant docs + this many distractors
TOP_K             = 10           # k for Recall@K and MRR@K

CACHE_DIR         = Path("cache")
RESULTS_DIR       = Path("results")

CORPUS_CACHE      = CACHE_DIR / "corpus.json"
POOL_EMB_CACHE    = CACHE_DIR / "pool_embeddings.npy"
POOL_IDS_CACHE    = CACHE_DIR / "pool_ids.json"
QUERY_EMB_CACHE   = CACHE_DIR / "query_embeddings.npy"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def ram_mb() -> float:
    """Current process RSS in MB."""
    return psutil.Process().memory_info().rss / 1024 ** 2


def latency_stats(latencies: list[float]) -> dict[str, float]:
    """Compute p50 / p95 / p99 from a list of per-query latencies (ms)."""
    a = np.array(latencies)
    return {
        "p50": round(float(np.percentile(a, 50)), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
        "p99": round(float(np.percentile(a, 99)), 2),
        "mean": round(float(np.mean(a)), 2),
    }


def run_dense(
    retriever: DenseRetriever,
    eval_set: list[dict],
    query_embeddings: np.ndarray,
    k: int,
) -> tuple[dict, list[float]]:
    """Run dense retrieval over all eval queries. Returns (metrics, latencies_ms)."""
    retrieved, latencies = [], []
    for i, entry in enumerate(eval_set):
        doc_ids, lat = retriever.search(query_embeddings[i], k=k)
        retrieved.append(doc_ids)
        latencies.append(lat)
    metrics = evaluate(eval_set, retrieved, ks=(1, 5, 10))
    return metrics, latencies


def run_hybrid(
    retriever: HybridRetriever,
    eval_set: list[dict],
    query_embeddings: np.ndarray,
    k: int,
) -> tuple[dict, list[float]]:
    """Run hybrid retrieval over all eval queries. Returns (metrics, latencies_ms)."""
    retrieved, latencies = [], []
    for i, entry in enumerate(eval_set):
        doc_ids, lat = retriever.search(entry["query"], query_embeddings[i], k=k)
        retrieved.append(doc_ids)
        latencies.append(lat)
    metrics = evaluate(eval_set, retrieved, ks=(1, 5, 10))
    return metrics, latencies


def print_result_row(label: str, metrics: dict, lat: dict, ram: float) -> None:
    print(
        f"    [{label}] "
        f"R@1={metrics['recall@1']:.4f}  R@10={metrics['recall@10']:.4f}  "
        f"MRR@10={metrics['mrr@10']:.4f}  "
        f"p50={lat['p50']:.1f}ms  p95={lat['p95']:.1f}ms  "
        f"RAM Δ={ram:+.0f}MB"
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Step 1: Load corpus pool ─────────────────────────────────────────────
    if CORPUS_CACHE.exists():
        print("📂  Loading corpus from cache...")
        pool, eval_set = load_cache(CORPUS_CACHE)
    else:
        print("🌐  Downloading qrels + queries from HuggingFace...")
        qrels, queries = load_qrels_and_queries()
        eval_set, relevant_ids = pick_eval_queries(qrels, queries, N_EVAL_QUERIES)
        print(f"    Picked {len(eval_set)} eval queries with {len(relevant_ids)} relevant docs")

        print(f"🌊  Streaming MS MARCO corpus (~5–10 min, target {DISTRACTOR_TARGET:,} distractors)...")
        pool = build_corpus_pool(relevant_ids, DISTRACTOR_TARGET)
        save_cache(pool, eval_set, CORPUS_CACHE)
        print(f"    Cached to {CORPUS_CACHE}")

    print(f"✅  Pool: {len(pool):,} docs  |  Eval: {len(eval_set)} queries")

    # ── Step 2: Embed all pool docs ──────────────────────────────────────────
    if POOL_IDS_CACHE.exists():
        pool_ids = json.load(open(POOL_IDS_CACHE))
    else:
        pool_ids = [d["id"] for d in pool]
        json.dump(pool_ids, open(POOL_IDS_CACHE, "w"))

    print(f"\n🔢  Embedding {len(pool):,} corpus docs (cached per-batch)...")
    pool_embeddings = embed_texts(
        [d["text"] for d in pool],
        cache_file=POOL_EMB_CACHE,
        label="corpus",
    )
    print(f"✅  Pool embeddings: {pool_embeddings.shape}  ({pool_embeddings.nbytes / 1e6:.0f} MB)")

    # Build fast id → row-index mapping
    id_to_idx: dict[str, int] = {did: i for i, did in enumerate(pool_ids)}

    # ── Step 3: Embed queries ────────────────────────────────────────────────
    print(f"\n🔢  Embedding {len(eval_set)} queries...")
    query_embeddings = embed_texts(
        [e["query"] for e in eval_set],
        cache_file=QUERY_EMB_CACHE,
        label="queries",
    )
    print(f"✅  Query embeddings: {query_embeddings.shape}")

    # ── Step 4: Scaling loop ─────────────────────────────────────────────────
    all_results: list[dict] = []

    for size in CORPUS_SIZES:
        print(f"\n{'═'*65}")
        print(f"  📊  Corpus size: {size:,}")
        print(f"{'═'*65}")

        # Build subset and extract its embeddings from the pre-computed pool
        subset = build_subset(pool, eval_set, size)
        indices = [id_to_idx[d["id"]] for d in subset]
        sub_emb = pool_embeddings[indices]
        sub_ids = [d["id"] for d in subset]

        print(f"  Subset: {len(subset):,} docs  "
              f"({sub_emb.nbytes / 1e6:.0f} MB embeddings)")

        row: dict = {"size": size}

        # ── Baseline: Dense brute-force ──────────────────────────────────────
        print("\n  ▶  Baseline — Dense brute-force (numpy):")
        ram_before = ram_mb()
        dense = DenseRetriever()
        dense.build_index(sub_ids, sub_emb)
        ram_delta_dense = ram_mb() - ram_before

        t_start = time.perf_counter()
        metrics_d, latencies_d = run_dense(dense, eval_set, query_embeddings, k=TOP_K)
        total_time_d = time.perf_counter() - t_start
        lat_d = latency_stats(latencies_d)

        print_result_row("Dense", metrics_d, lat_d, ram_delta_dense)
        print(f"    Total eval time: {total_time_d:.1f}s  "
              f"({total_time_d/len(eval_set)*1000:.1f}ms/query)")

        row["baseline"] = {
            "metrics": metrics_d,
            "latency": lat_d,
            "ram_mb": round(ram_delta_dense, 1),
        }
        del dense   # free RAM before building hybrid index

        # ── Fix: Hybrid BM25 + Dense + RRF ───────────────────────────────────
        print("\n  ▶  Fix — Hybrid BM25 + Dense + RRF:")
        ram_before = ram_mb()
        hybrid = HybridRetriever()
        hybrid.build_index(subset, sub_emb)
        ram_delta_hybrid = ram_mb() - ram_before

        t_start = time.perf_counter()
        metrics_h, latencies_h = run_hybrid(hybrid, eval_set, query_embeddings, k=TOP_K)
        total_time_h = time.perf_counter() - t_start
        lat_h = latency_stats(latencies_h)

        print_result_row("Hybrid", metrics_h, lat_h, ram_delta_hybrid)
        print(f"    Total eval time: {total_time_h:.1f}s  "
              f"({total_time_h/len(eval_set)*1000:.1f}ms/query)")

        row["hybrid"] = {
            "metrics": metrics_h,
            "latency": lat_h,
            "ram_mb": round(ram_delta_hybrid, 1),
        }
        del hybrid  # free RAM

        all_results.append(row)

    # ── Step 5: Save results ─────────────────────────────────────────────────
    results_path = RESULTS_DIR / "results.json"
    json.dump(all_results, open(results_path, "w"), indent=2)
    print(f"\n\n✅  Results saved → {results_path}")

    # ── Step 6: Visualize ────────────────────────────────────────────────────
    try:
        import visualize
        visualize.plot_all(all_results, RESULTS_DIR)
        visualize.print_summary_table(all_results)
    except Exception as e:
        print(f"⚠️  Visualization failed: {e} — run visualize.py manually.")

    print("\n🏁  Done!")


if __name__ == "__main__":
    main()
