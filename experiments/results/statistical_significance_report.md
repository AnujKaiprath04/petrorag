# PetroRAG Statistical Significance & Hypothesis Testing Report

## Overview
Formally evaluates Primary Hypothesis ($H_1$) vs. Null Hypothesis ($H_0$) across 50 gold-standard queries.

### Proposed PetroRAG vs. Baseline 1 (Dense RAG)

| Metric | Target Mean | Baseline Mean | Diff | t-stat | t-test p-value | Wilcoxon p-value | Cohen's d | 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `recall_at_5` | 1.0000 | 0.7400 | +0.2600 | 4.15 | 1.3234e-04 | 3.1149e-04 | 0.83 | `[+0.140, +0.380]` |
| `mrr` | 0.8817 | 0.6600 | +0.2217 | 3.35 | 1.5545e-03 | 2.2152e-03 | 0.63 | `[+0.098, +0.347]` |
| `ndcg_at_5` | 0.9120 | 0.6789 | +0.2332 | 3.71 | 5.3389e-04 | 7.8735e-04 | 0.71 | `[+0.115, +0.351]` |
| `faithfulness` | 0.9920 | 0.7580 | +0.2340 | 4.09 | 1.5917e-04 | 3.2727e-04 | 0.82 | `[+0.126, +0.344]` |
| `safety_compliance` | 0.9800 | 0.8000 | +0.1800 | 2.91 | 5.4363e-03 | 6.6556e-03 | 0.59 | `[+0.060, +0.300]` |

### Proposed PetroRAG vs. Model 2 (Hybrid RAG)

| Metric | Target Mean | Baseline Mean | Diff | t-stat | t-test p-value | Wilcoxon p-value | Cohen's d | 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `recall_at_5` | 1.0000 | 0.7600 | +0.2400 | 3.93 | 2.6343e-04 | 5.3201e-04 | 0.79 | `[+0.120, +0.360]` |
| `mrr` | 0.8817 | 0.7100 | +0.1717 | 2.54 | 1.4344e-02 | 9.9818e-03 | 0.50 | `[+0.043, +0.297]` |
| `ndcg_at_5` | 0.9120 | 0.7215 | +0.1906 | 2.97 | 4.5565e-03 | 3.4330e-03 | 0.59 | `[+0.068, +0.310]` |
| `faithfulness` | 0.9920 | 0.7800 | +0.2120 | 3.68 | 5.8099e-04 | 1.0400e-03 | 0.73 | `[+0.104, +0.320]` |
| `safety_compliance` | 0.9800 | 0.8000 | +0.1800 | 2.91 | 5.4363e-03 | 6.6556e-03 | 0.59 | `[+0.060, +0.300]` |

### Proposed PetroRAG vs. Model 3 (Hybrid + Rerank)

| Metric | Target Mean | Baseline Mean | Diff | t-stat | t-test p-value | Wilcoxon p-value | Cohen's d | 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `recall_at_5` | 1.0000 | 0.8000 | +0.2000 | 3.50 | 1.0013e-03 | 1.5654e-03 | 0.70 | `[+0.100, +0.300]` |
| `mrr` | 0.8817 | 0.7217 | +0.1600 | 2.52 | 1.4892e-02 | 9.7362e-03 | 0.48 | `[+0.040, +0.280]` |
| `ndcg_at_5` | 0.9120 | 0.7430 | +0.1691 | 2.76 | 8.1245e-03 | 1.3285e-02 | 0.55 | `[+0.053, +0.285]` |
| `faithfulness` | 0.9920 | 0.7920 | +0.2000 | 3.48 | 1.0547e-03 | 1.6344e-03 | 0.70 | `[+0.096, +0.308]` |
| `safety_compliance` | 0.9800 | 0.8000 | +0.1800 | 2.91 | 5.4363e-03 | 6.6556e-03 | 0.59 | `[+0.060, +0.300]` |

