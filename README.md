# PetroRAG

> **Intelligent Retrieval-Augmented Decision Support System for Oil & Gas Fields**  
> *Final Year Project (Software Engineering) & High-Impact Empirical Research Study*

---

## 🎯 Dual Purpose

1. **Enterprise Decision Support Software**: A scalable, production-ready system for upstream/midstream Oil & Gas operations, enabling engineers to query equipment manuals, maintenance procedures, incident reports, and safety protocols with high confidence and verified citations.
2. **Empirical Research Paper**: A rigorous scientific investigation into domain-adapted hybrid retrieval, cross-encoder reranking, claim-level grounding verification, and unanswerable query abstention in safety-critical industrial environments.

For full architectural blueprints, formal hypotheses ($H_1, H_0$), evaluation metrics, and ablation matrices, refer to [PROJECT_CHARTER.md](PROJECT_CHARTER.md).

---

## 🔬 Experimental Progression

| Configuration | Retrieval Layer | Ranking / Fusion | Verification | Target Efficacy |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline 0** | None (Direct LLM) | N/A | None | Parametric memory benchmark |
| **Baseline 1** | Dense Vector Search | Cosine Similarity | None | Standard naive RAG baseline |
| **Model 2** | Dense + BM25 Sparse | Reciprocal Rank Fusion (RRF) | None | Hybrid lexical-semantic synergy |
| **Model 3** | Dense + BM25 Sparse | Cross-Encoder Reranker | None | Deep contextual relevance |
| **Proposed PetroRAG** | Query Understanding + Metadata Filter + Hybrid | RRF + Cross-Encoder + Context Compactor | Grounding & Citation Verifier + Abstention | High faithfulness, low hallucination, verified citations |

---

## 📁 Repository Structure

```
├── data/
│   ├── raw/                  # Source PDFs, manuals, reports (Public / Synthetic)
│   ├── processed/            # Extracted markdown, OCR text, metadata JSONs
│   └── ground_truth/         # Gold-standard QA benchmark (JSON/Parquet)
├── src/
│   ├── core/                 # Config, logging, telemetry, security
│   ├── ingestion/            # Parsers, chunking strategies, metadata extractors
│   ├── retrieval/            # Vector store, BM25, RRF fusion, Cross-Encoder reranker
│   ├── generation/           # Prompt templates, LLM client adapters, context optimizers
│   ├── verification/         # Claim extractor, NLI grounding checker, citation validator, abstainer
│   ├── api/                  # FastAPI endpoints for chat, search, document management
│   └── ui/                   # Frontend interface (decision support dashboard)
├── experiments/
│   ├── baselines/            # Runners for Baseline 0, 1, 2, 3
│   ├── petrorag/             # Runner for Proposed PetroRAG pipeline
│   ├── ablation/             # Component ablation runner scripts
│   ├── eval/                 # Evaluation harness (Retrieval + Generation + Hallucination)
│   └── results/              # Raw experiment logs, CSV/JSON outputs, statistical test notebooks
├── paper/
│   ├── sections/             # LaTeX/Markdown drafts (Intro, Lit Review, Methodology, etc.)
│   ├── figures/              # Auto-generated plots, architectural diagrams
│   └── tables/               # Automated LaTeX table generation from experimental outputs
└── tests/                    # Unit and integration test suites
```

---

## 🚀 Getting Started

Instructions for environment setup, data ingestion, baseline evaluations, and local deployment will be continuously maintained here.
