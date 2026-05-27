"""
Evaluation metrics — fixed implementation so everyone reports comparable numbers.

Conventions:
  - Rank 1 = first position (NOT zero-indexed)
  - retrieved is an ordered list of doc_ids returned by the retriever
  - relevant is a set of ground-truth relevant doc_ids
  - In eval_set from data_loader, relevant_ids comes as list[str] — evaluate()
    converts to set internally, so you can pass eval_set as-is.
"""
from typing import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant doc appears in retrieved[:k], else 0.0 (per-query)."""
    return float(any(d in relevant for d in retrieved[:k]))


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1/rank of first relevant doc in retrieved[:k] (rank starts at 1). 0 if not found."""
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(
    eval_set: list[dict],
    retrieved_per_query: list[list[str]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """
    Run all metrics over the eval set. Returns flat dict like:
      {"recall@1": 0.7, "recall@5": 0.93, "recall@10": 0.93, "mrr@10": 0.80}
    """
    assert len(eval_set) == len(retrieved_per_query), \
        f"eval_set ({len(eval_set)}) and retrieved_per_query ({len(retrieved_per_query)}) length mismatch"

    n = len(eval_set)
    out: dict[str, float] = {}
    for k in ks:
        recall = sum(
            recall_at_k(ret, set(e["relevant_ids"]), k)
            for e, ret in zip(eval_set, retrieved_per_query)
        ) / n
        out[f"recall@{k}"] = round(recall, 4)

    max_k = max(ks)
    mrr = sum(
        reciprocal_rank(ret, set(e["relevant_ids"]), max_k)
        for e, ret in zip(eval_set, retrieved_per_query)
    ) / n
    out[f"mrr@{max_k}"] = round(mrr, 4)
    return out
