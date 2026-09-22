"""
Automated Publication Figure Generator (Module 2.34)
Reads empirical benchmark and ablation results directly and renders
high-resolution publication figures into paper/figures/.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "experiments" / "results"
FIGURES_DIR = ROOT_DIR / "paper" / "figures"


def generate_figure1_retrieval(baselines: List[Dict[str, Any]]):
    """Figure 1: Retrieval Recall@K (K=1, 3, 5) and NDCG@5 across Progressive Architectures."""
    models = [b["model_name"].replace(" (Direct LLM)", "").replace(" (Dense RAG)", "").replace(" (Hybrid RAG)", "").replace(" (Hybrid + Rerank)", "").replace(" (Proposed PetroRAG)", "") for b in baselines[1:]]
    # Models: Dense RAG, Hybrid RAG, Hybrid + Rerank, PetroRAG
    short_labels = ["Dense RAG", "Hybrid RAG", "Hybrid+Rerank", "PetroRAG (Full)"]

    r1 = [b["recall_at_1"] for b in baselines[1:]]
    r3 = [b["recall_at_3"] for b in baselines[1:]]
    r5 = [b["recall_at_5"] for b in baselines[1:]]
    ndcg = [b["ndcg_at_5"] for b in baselines[1:]]

    x = np.arange(len(short_labels))
    width = 0.20

    plt.figure(figsize=(9, 5), dpi=300)
    plt.rcParams.update({"font.size": 10, "font.family": "serif"})

    plt.bar(x - 1.5 * width, r1, width, label="Recall@1", color="#4A90E2", alpha=0.9, edgecolor="black", linewidth=0.8)
    plt.bar(x - 0.5 * width, r3, width, label="Recall@3", color="#50E3C2", alpha=0.9, edgecolor="black", linewidth=0.8)
    plt.bar(x + 0.5 * width, r5, width, label="Recall@5", color="#F5A623", alpha=0.9, edgecolor="black", linewidth=0.8)
    plt.bar(x + 1.5 * width, ndcg, width, label="NDCG@5", color="#D0021B", alpha=0.9, edgecolor="black", linewidth=0.8)

    plt.ylabel("Score [0.0 - 1.0]", fontweight="bold")
    plt.title("Information Retrieval Performance Across Progressive RAG Architectures", fontweight="bold", pad=12)
    plt.xticks(x, short_labels, fontweight="bold")
    plt.ylim(0.0, 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    out_path = FIGURES_DIR / "fig1_retrieval_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Generated {out_path}")


def generate_figure2_faithfulness_hallucination(baselines: List[Dict[str, Any]]):
    """Figure 2: Faithfulness vs. Hallucination Rate across progressive architectures."""
    short_labels = ["Direct LLM", "Dense RAG", "Hybrid RAG", "Hybrid+Rerank", "PetroRAG (Full)"]
    faithfulness = [b["faithfulness"] for b in baselines]
    hallucination = [b["hallucination_rate"] for b in baselines]
    abstention = [b["abstention_f1"] for b in baselines]

    plt.figure(figsize=(8, 5), dpi=300)
    plt.rcParams.update({"font.size": 10, "font.family": "serif"})

    colors = ["#9013FE", "#4A90E2", "#50E3C2", "#F5A623", "#417505"]
    sizes = [180, 220, 260, 300, 380]

    for i, (label, f, h, a) in enumerate(zip(short_labels, faithfulness, hallucination, abstention)):
        plt.scatter(h, f, s=sizes[i], color=colors[i], edgecolors="black", linewidths=1.2, label=f"{label} (Abst $F_1$: {a:.2f})", zorder=4)
        offset_x = 0.02
        offset_y = 0.02 if i != 4 else -0.04
        plt.text(h + offset_x, f + offset_y, label, fontsize=9, fontweight="bold")

    plt.xlabel("Hallucination Rate (Lower is Better) $\\rightarrow$", fontweight="bold")
    plt.ylabel("Claim Faithfulness (Higher is Better) $\\rightarrow$", fontweight="bold")
    plt.title("Empirical Safety Frontier: Faithfulness vs. Hallucination Suppression", fontweight="bold", pad=12)
    plt.xlim(-0.05, 1.1)
    plt.ylim(-0.05, 1.1)
    plt.axvline(0.0, color="gray", linestyle=":", alpha=0.5)
    plt.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="center right", frameon=True, fontsize=8)
    plt.tight_layout()

    out_path = FIGURES_DIR / "fig2_hallucination_vs_faithfulness.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Generated {out_path}")


def generate_figure3_token_compression(ablations: List[Dict[str, Any]]):
    """Figure 3: Prompt Token Economy and Savings with Extractive Context Compression."""
    labels = ["PetroRAG (Full)", r"$\setminus$ Hybrid", r"$\setminus$ MetaFilter", r"$\setminus$ Reranker", r"$\setminus$ Compression", r"$\setminus$ Verifier", r"$\setminus$ Abstention"]
    prompt_tokens = [a["mean_prompt_tokens"] for a in ablations]
    savings = [a["token_savings_percent"] for a in ablations]

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    plt.rcParams.update({"font.size": 10, "font.family": "serif"})

    x = np.arange(len(labels))
    width = 0.45

    color1 = "#2C3E50"
    rects1 = ax1.bar(x, prompt_tokens, width, color=color1, alpha=0.85, edgecolor="black", label="Mean Prompt Tokens")
    ax1.set_ylabel("Mean Prompt Tokens (Lower is Better)", color=color1, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontweight="bold")
    ax1.set_ylim(0, max(prompt_tokens) * 1.25)

    ax2 = ax1.twinx()
    color2 = "#E67E22"
    ax2.plot(x, savings, color=color2, marker="o", linewidth=2.2, markersize=7, label="Token Savings (%)")
    ax2.set_ylabel("Token Savings (%) $\\rightarrow$", color=color2, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(-5, 30)

    plt.title("Ablation Study: Impact of Subsystems on Context Length & Token Economy", fontweight="bold", pad=12)
    fig.tight_layout()

    out_path = FIGURES_DIR / "fig3_token_compression.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Generated {out_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    baselines_path = RESULTS_DIR / "baseline_comparison_results.json"
    ablations_path = RESULTS_DIR / "ablation_study_results.json"

    with open(baselines_path, "r", encoding="utf-8") as f:
        baselines = json.load(f)

    with open(ablations_path, "r", encoding="utf-8") as f:
        ablations = json.load(f)

    generate_figure1_retrieval(baselines)
    generate_figure2_faithfulness_hallucination(baselines)
    generate_figure3_token_compression(ablations)


if __name__ == "__main__":
    main()
