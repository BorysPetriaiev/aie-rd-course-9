"""
Visualization for the RAG scaling experiment.

Generates:
  results/scaling_results.png — 4-panel plot (Recall@1, Recall@10, MRR@10, Latency)
  results/ram_usage.png       — RAM delta per corpus size

Usage:
  python visualize.py                          # reads results/results.json
  python visualize.py path/to/results.json
"""
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


# ── Formatter for log-scale x-axis ───────────────────────────────────────────
_fmt = mticker.FuncFormatter(lambda x, _: f"{int(x):,}")


def _setup_ax(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Corpus size (log scale)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(_fmt)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=8)


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_all(results: list[dict], out_dir: Path) -> None:
    """Generate and save all scaling plots."""
    out_dir.mkdir(exist_ok=True)
    sizes = [r["size"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "RAG Scaling Experiment — MS MARCO (OpenAI text-embedding-3-small)",
        fontsize=13, fontweight="bold", y=1.01
    )

    # ── Panel 1: Recall@1 ────────────────────────────────────────────────────
    ax = axes[0][0]
    ax.plot(sizes, [r["baseline"]["metrics"]["recall@1"] for r in results],
            "o-", color="#2196F3", linewidth=2, markersize=7, label="Dense baseline")
    ax.plot(sizes, [r["hybrid"]["metrics"]["recall@1"] for r in results],
            "s--", color="#4CAF50", linewidth=2, markersize=7, label="Hybrid BM25+dense+RRF")
    # Mark 20% drop threshold from the 1K baseline
    base_r1 = results[0]["baseline"]["metrics"]["recall@1"]
    ax.axhline(base_r1 * 0.8, color="red", linestyle=":", alpha=0.6, label="−20% threshold")
    _setup_ax(ax, "Recall@1 vs Corpus Size", "Recall@1")

    # ── Panel 2: Recall@10 ───────────────────────────────────────────────────
    ax = axes[0][1]
    ax.plot(sizes, [r["baseline"]["metrics"]["recall@10"] for r in results],
            "o-", color="#2196F3", linewidth=2, markersize=7, label="Dense baseline")
    ax.plot(sizes, [r["hybrid"]["metrics"]["recall@10"] for r in results],
            "s--", color="#4CAF50", linewidth=2, markersize=7, label="Hybrid BM25+dense+RRF")
    base_r10 = results[0]["baseline"]["metrics"]["recall@10"]
    ax.axhline(base_r10 * 0.8, color="red", linestyle=":", alpha=0.6, label="−20% threshold")
    _setup_ax(ax, "Recall@10 vs Corpus Size", "Recall@10")

    # ── Panel 3: MRR@10 ──────────────────────────────────────────────────────
    ax = axes[1][0]
    ax.plot(sizes, [r["baseline"]["metrics"]["mrr@10"] for r in results],
            "o-", color="#2196F3", linewidth=2, markersize=7, label="Dense baseline")
    ax.plot(sizes, [r["hybrid"]["metrics"]["mrr@10"] for r in results],
            "s--", color="#4CAF50", linewidth=2, markersize=7, label="Hybrid BM25+dense+RRF")
    base_mrr = results[0]["baseline"]["metrics"]["mrr@10"]
    ax.axhline(base_mrr * 0.8, color="red", linestyle=":", alpha=0.6, label="−20% threshold")
    _setup_ax(ax, "MRR@10 vs Corpus Size", "MRR@10")

    # ── Panel 4: Latency ─────────────────────────────────────────────────────
    ax = axes[1][1]
    ax.plot(sizes, [r["baseline"]["latency"]["p50"] for r in results],
            "o-", color="#2196F3", linewidth=2, markersize=7, label="Dense p50")
    ax.plot(sizes, [r["baseline"]["latency"]["p95"] for r in results],
            "o--", color="#2196F3", linewidth=1.5, markersize=6, alpha=0.6, label="Dense p95")
    ax.plot(sizes, [r["hybrid"]["latency"]["p50"] for r in results],
            "s-", color="#4CAF50", linewidth=2, markersize=7, label="Hybrid p50")
    ax.plot(sizes, [r["hybrid"]["latency"]["p95"] for r in results],
            "s--", color="#4CAF50", linewidth=1.5, markersize=6, alpha=0.6, label="Hybrid p95")
    ax.axhline(100, color="orange", linestyle=":", alpha=0.7, label="100 ms (UX threshold)")
    _setup_ax(ax, "Search Latency vs Corpus Size", "Latency (ms)")

    plt.tight_layout()
    out_path = out_dir / "scaling_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈  Saved: {out_path}")

    # ── RAM usage bar chart ───────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    x = np.arange(len(sizes))
    width = 0.35
    ax2.bar(x - width/2, [r["baseline"]["ram_mb"] for r in results],
            width, label="Dense baseline", color="#2196F3", alpha=0.8)
    ax2.bar(x + width/2, [r["hybrid"]["ram_mb"] for r in results],
            width, label="Hybrid BM25+dense+RRF", color="#4CAF50", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{s:,}" for s in sizes])
    ax2.set_xlabel("Corpus size")
    ax2.set_ylabel("RAM delta (MB)")
    ax2.set_title("Index RAM Usage vs Corpus Size", fontweight="bold")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_path2 = out_dir / "ram_usage.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈  Saved: {out_path2}")


def print_summary_table(results: list[dict]) -> None:
    """Print a compact comparison table to stdout."""
    header = (
        f"\n{'─'*110}\n"
        f"{'Size':>10} │ "
        f"{'R@1 base':>8} {'R@10 base':>9} {'MRR base':>8} │ "
        f"{'R@1 hyb':>8} {'R@10 hyb':>9} {'MRR hyb':>8} │ "
        f"{'p50 base':>8} {'p50 hyb':>8} │ "
        f"{'R@1 Δ':>7}\n"
        f"{'─'*110}"
    )
    print(header)
    for r in results:
        s   = r["size"]
        bm  = r["baseline"]["metrics"]
        hm  = r["hybrid"]["metrics"]
        bl  = r["baseline"]["latency"]
        hl  = r["hybrid"]["latency"]
        delta_r1 = hm["recall@1"] - bm["recall@1"]
        print(
            f"{s:>10,} │ "
            f"{bm['recall@1']:>8.4f} {bm['recall@10']:>9.4f} {bm['mrr@10']:>8.4f} │ "
            f"{hm['recall@1']:>8.4f} {hm['recall@10']:>9.4f} {hm['mrr@10']:>8.4f} │ "
            f"{bl['p50']:>8.1f} {hl['p50']:>8.1f} │ "
            f"{delta_r1:>+7.4f}"
        )
    print("─" * 110)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/results.json"
    results = json.load(open(path))
    plot_all(results, Path("results"))
    print_summary_table(results)
