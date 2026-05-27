"""
Retriever implementations:

  DenseRetriever  — brute-force cosine similarity with numpy (O(N) baseline)
  HybridRetriever — BM25 + dense + Reciprocal Rank Fusion (the "fix")
"""
import time
from typing import Optional
import numpy as np


# ─── Utilities ───────────────────────────────────────────────────────────────

def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization. Safe against zero-norm vectors."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-9)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """
    Merge multiple ranked lists using RRF:
      score(doc) = Σ  1 / (k + rank_in_list_i)
    Higher score = better merged rank.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


# ─── Dense Retriever ─────────────────────────────────────────────────────────

class DenseRetriever:
    """
    Brute-force cosine similarity search using numpy matrix multiplication.
    Complexity: O(N × D) per query — latency grows linearly with corpus size.

    This is the baseline that we expect to break under scaling.
    """

    def __init__(self) -> None:
        self._embeddings: Optional[np.ndarray] = None   # shape (N, D), normalized
        self._doc_ids: list[str] = []

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def n_docs(self) -> int:
        """Number of documents currently indexed."""
        return len(self._doc_ids)

    # ── build ────────────────────────────────────────────────────────────────

    def build_index(self, doc_ids: list[str], embeddings: np.ndarray) -> None:
        """
        Normalize embeddings and store them.
        embeddings: float32 ndarray of shape (N, D).
        """
        self._doc_ids = list(doc_ids)
        self._embeddings = l2_normalize(embeddings.astype(np.float32))

    # ── search ───────────────────────────────────────────────────────────────

    def search(self, query_embedding: np.ndarray, k: int = 10) -> tuple[list[str], float]:
        """
        Return top-k doc IDs ranked by cosine similarity.

        Returns:
            (ranked_doc_ids, latency_ms)
        """
        q = l2_normalize(query_embedding.reshape(1, -1)).flatten()

        t0 = time.perf_counter()
        scores = self._embeddings @ q          # (N,) cosine similarities
        top_k = np.argpartition(scores, -k)[-k:]  # partial sort for speed
        top_k = top_k[np.argsort(scores[top_k])[::-1]]
        latency_ms = (time.perf_counter() - t0) * 1000

        return [self._doc_ids[i] for i in top_k], latency_ms


# ─── Hybrid Retriever ─────────────────────────────────────────────────────────

class HybridRetriever:
    """
    BM25 (sparse) + Dense (dense) combined via Reciprocal Rank Fusion.

    Why this should fix recall@1 degradation:
    - BM25 is strong on rare / exact-match terms that dense embeddings dilute
      in large corpora (many near-duplicate passages push the true answer down).
    - RRF boosts docs that appear in BOTH rankings → more robust relevance signal.

    Trade-off: higher latency (BM25 scan + dense scan + RRF merge).
    """

    def __init__(self, rrf_k: int = 60, candidate_k: int = 100) -> None:
        """
        Args:
            rrf_k:       RRF constant (60 is standard; higher = smoother blending).
            candidate_k: Number of candidates fetched from each retriever before merging.
        """
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self._dense = DenseRetriever()
        self._bm25 = None
        self._doc_ids: list[str] = []

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def n_docs(self) -> int:
        """Number of documents currently indexed."""
        return len(self._doc_ids)

    # ── build ────────────────────────────────────────────────────────────────

    def build_index(self, docs: list[dict], embeddings: np.ndarray) -> None:
        """
        Build BM25 and dense indexes.

        Args:
            docs:       List of {"id": str, "text": str} dicts.
            embeddings: float32 ndarray of shape (N, D) aligned with docs.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError("rank_bm25 not installed. Run: pip install rank-bm25") from exc

        self._doc_ids = [d["id"] for d in docs]

        print("    Building BM25 index...", end=" ")
        t0 = time.perf_counter()
        tokenized = [d["text"].lower().split() for d in docs]
        self._bm25 = BM25Okapi(tokenized)
        print(f"done in {(time.perf_counter()-t0):.1f}s")

        print("    Building dense index...", end=" ")
        t0 = time.perf_counter()
        self._dense.build_index(self._doc_ids, embeddings)
        print(f"done in {(time.perf_counter()-t0):.1f}s")

    # ── search ───────────────────────────────────────────────────────────────

    def search(
        self, query: str, query_embedding: np.ndarray, k: int = 10
    ) -> tuple[list[str], float]:
        """
        Hybrid search: BM25 top-candidate_k + Dense top-candidate_k → RRF → top-k.

        Returns:
            (ranked_doc_ids, latency_ms)
        """
        t0 = time.perf_counter()

        # Clamp candidate_k to actual corpus size (avoids argpartition out-of-bounds)
        ck = min(self.candidate_k, len(self._doc_ids))

        # BM25 ranking
        tokens = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokens)
        bm25_top_idx = np.argpartition(bm25_scores, -ck)[-ck:]
        bm25_top_idx = bm25_top_idx[np.argsort(bm25_scores[bm25_top_idx])[::-1]]
        bm25_ranking = [self._doc_ids[i] for i in bm25_top_idx]

        # Dense ranking
        dense_ranking, _ = self._dense.search(query_embedding, k=ck)

        # Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=self.rrf_k)

        latency_ms = (time.perf_counter() - t0) * 1000
        return fused[:k], latency_ms
