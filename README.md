# 🏛️ Dual-Stage Vietnamese Legal Information Retrieval System

> **Top 7 Private Leaderboard Solution** — UIT Data Science Challenge 2026 (DSC 2026)  
> *A cleanroom, production-grade hybrid retrieval and reranking engine for large-scale statutory law.*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Private LB](https://img.shields.io/badge/Private%20Leaderboard-Top%207-gold.svg)](#-empirical-results--ablation-study)
[![OOF Recall@5](https://img.shields.io/badge/OOF%20Recall%405-0.9505-brightgreen.svg)](#-empirical-results--ablation-study)
[![Candidate Coverage](https://img.shields.io/badge/Top--30%20Coverage-99.32%25-success.svg)](#stage-1--multi-encoder-hybrid-retrieval)

---

## 📌 Executive Summary

This repository hosts the official source code for our **Top 7** winning solution in the **UIT Data Science Challenge 2026 (Legal Information Retrieval track)**. The task mandates identifying the most relevant legal articles/clauses from an authoritative corpus of **8,512 Vietnamese statutory laws** in response to natural language legal queries.

The architecture is built upon a **Two-Stage Hybrid Pipeline** designed with strict cleanroom engineering principles:
1. **Stage 1 (Retrieval & Multi-Encoder Shortlisting):** Multi-view candidate generation combining tuned lexical search (`BM25Okapi`) with complementary Bi-Encoders (`AITeamVN/Vietnamese_Bi-Encoder` and `vnlegal-lal`) over **179,737 moderate legal regions**, achieving **99.32% gold candidate coverage** on the Top-30 shortlist.
2. **Stage 2 (Deep Reranking & Bayesian Prior Fusion):** High-capacity Cross-Encoder (CE-LR 45D) coupled with an instruction-tuned **Qwen3-0.6B LLM Reranker** (trained via Multi-task Listwise + Pairwise + BCE loss). Query-level calibration shifts are eliminated using **Per-Query Logit Centering**, and predictions are enriched with **Gold Document Propensity** and **Gold Centroid Head** priors, advancing Out-of-Fold (OOF) Recall@5 to **0.9505** and locking **Top 7 on the Private Leaderboard**.

```text
                                  [User Legal Query]
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
   [BM25Okapi Lexical]                                     [Multi-Bi-Encoder Dense]
 (Word segmentation & tuning)                             (AITeamVN + VNLegal-LAL)
            │                                                           │
            └─────────────────────────────┬─────────────────────────────┘
                                          ▼
                   [Hybrid Fusion & 179.7K Region Screening]
                   Candidate Pool Coverage: 99.32% (Top-30)
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
  [AIT Cross-Encoder]                                        [Qwen3-0.6B Reranker]
    (CE-LR 45D Model)                                      (Multi-task Loss Fine-tuned)
            │                                                           │
            │                                            [Per-Query Logit Centering]
            │                                                           │
            └─────────────────────────────┬─────────────────────────────┘
                                          ▼
                       [Statistical Bayesian Prior Injection]
                       ├── Gold Document Propensity (w = 0.75)
                       └── Gold Centroid Head (Semantic Alignment)
                                          │
                                          ▼
                   [Final Clean Fusion: OOF Recall@5 = 0.9505]
                        Top 7 Private Leaderboard Solution
```

---

## 🚀 The Engineering Story: Two-Phase Evolution

Our competitive advantage stemmed from strict **engineering hygiene and rigorous forensic auditing**, splitting the project into two distinct phases:

```text
[Phase 1: Exploratory R&D (LegalIR)]
├── Parent-Scope Chunking & GBDT Learning-to-Rank (72D -> 145D Handcrafted Features)
├── Kinship Graphs, Law Authority & Citation Network Modeling
└── Forensic Discovery: Uncovered 53 contaminated rules with article parity mismatch
              │
              ▼ (Disciplined Decision: Freeze Phase 1 & Rebuild from scratch)
[Phase 2: SOTA Cleanroom Architecture (Endgame)]
├── 8,512 Clean Corpus Universe & 179,737 Moderate Legal Regions
├── Multi-Bi-Encoder Ensembling (Candidate Recall: 99.32%)
├── Deep Cross-Encoder + Centered Qwen3-0.6B LLM Reranker
└── Bayesian Priors: Gold Document Propensity (OOF Recall@5: 0.9505 -> Top 7 Private LB)
```

- **Phase 1 Archive (`research_archive/legalir_public_phase/`):** Contains exploratory work with 145-dimensional GBDT features and the detailed forensic audit logs identifying deployment parity failure.
- **Phase 2 Production (`src/`):** Cleanroom codebase with zero data leakage, group-isolated 5-fold cross-validation, and production reranking modules.

---

## 📊 Empirical Results & Ablation Study

All metrics below are evaluated under the frozen **6,991 primary evaluable queries** across 5 grouped folds (Seed 276) matching the official competition evaluation protocol.

| Pipeline Stage | Model / Strategy | OOF Recall@5 | Δ vs Prev | Description / Key Insight |
|---|---|:---:|:---:|---|
| **Raw Baseline** | BM25Okapi (Passage-level) | ~0.7600 | — | Lexical search baseline over raw parent articles |
| **Stage 02B Retriever** | VietLegal-E5 Bi-Encoder | 0.8520 | +9.20% | Dense representations on moderate regions |
| **Stage 02B Retriever** | AITeamVN Bi-Encoder | 0.8710 | +1.90% | State-of-the-art Vietnamese sentence embeddings |
| **Stage 02B Multi-Retrieval** | AIT + LAL + BM25 (Top-30) | **0.9932\*** | — | **\*Candidate Coverage**: 99.32% gold recall in top-30 pool |
| **Stage 03B1 Reranker** | AIT Cross-Encoder (CE-LR 45D) | 0.9426 | +7.16% | Full cross-attention interaction over top-30 shortlist |
| **Stage 07B LLM Rerank** | Qwen3-0.6B Reranker (Raw) | 0.9452 | +0.26% | Instruction-tuned with Listwise + Pairwise + BCE loss |
| **Stage 07N Calibration** | Qwen3-0.6B + Logit Centering | 0.9465 | +0.13% | Subtracts query-mean logit to eliminate query scale bias |
| **Stage 07O Semantic Prior** | Gold Centroid Head | 0.9474 | +0.09% | Distance projection to empirical gold embedding centroid |
| **Stage 07R Statistical Prior**| Gold Document Propensity | 0.9489 | +0.15% | Bayesian prior addressing long-tail statutory frequency |
| **Stage 07Y Final Fusion** | **AIT + Qwen + Propensity ($w=0.75$)** | **0.9505** | **+0.16%** | **Final Winning Submission (`GOLD_PROP_AIT_W75_K5.zip`)** |

---

## 🔬 Core Architectural Innovations

### 1. Structure-Aware Moderate Region Segmentation (Stage 00C & 02B)
Vietnamese statutory texts exhibit extreme structural variance—ranging from concise definitions to sprawling 40-clause articles.
- Raw documents were cleaned from 8,532 to **8,512 retrievable documents** (discarding 20 empty-passage artifacts).
- Structured segmentation produced **179,737 moderate regions** (Điều, Khoản, Phụ lục) preserving hierarchical context without exceeding transformer context windows.

### 2. Multi-Bi-Encoder Complementary Screening (Stage 02B)
To guarantee that the reranker receives the correct article in its candidate window:
- Integrated `AITeamVN/Vietnamese_Bi-Encoder` (568M) and `vnlegal-lal` (596M).
- Constructed atomic split (2048) and coarse-pack (1024) representations.
- Generated a compact **Top-30 shortlist** with an unprecedented **99.32% oracle gold coverage**.

### 3. Per-Query Logit-Centered Qwen3-0.6B Reranking (Stage 07B & 07N)
Fine-tuned `Qwen3-Reranker-0.6B` using hard negatives mined from Stage 02 retrieval pools:
- **Multi-task Loss:** Combined ListNet listwise ranking loss, margin pairwise loss, and binary cross-entropy.
- **Per-Query Logit Centering:** 
  $$\hat{s}(q, d_i) = s(q, d_i) - \frac{1}{|K|}\sum_{j \in K} s(q, d_j)$$
  Zero-centers the reranker output per candidate slate, eliminating inter-query distribution shifts and ensuring cross-query comparability.

### 4. Gold Document Propensity & Gold Centroid Head (Stage 07O, 07R & 07Y)
- **Gold Document Propensity:** Formulated a regularized empirical prior representing statutory citation frequency across the legal corpus, penalizing spurious one-off lexical matches.
- **Gold Centroid Head:** Projected query-candidate pairs against an aggregated centroid of confirmed gold document vectors.
- **Robust Fusion ($w_{prop}=0.75$):** Linearly combined cross-encoder logits with the propensity prior, hitting **0.9505 OOF Recall@5**.

---

## 📁 Repository Structure

```text
├── src/                                   # SOTA Cleanroom Production Pipeline
│   ├── common/                            # Metric definitions & evaluation protocols
│   ├── stage00_audit/                     # Corpus sanitization (8,512 docs universe)
│   ├── stage01_query_analysis/            # Query length, intent & duplicate analysis
│   ├── stage02_candidate_generation/      # 179.7K region indexer & multi-bi-encoders
│   ├── stage03_rerank/                    # Cross-encoder inference & shortlist builder
│   ├── stage04_private/                   # Private leaderboard inference runners
│   ├── stage05_analysis/                  # Per-fold diagnostics & error breakdown
│   ├── stage06_evidence/                  # Sub-clause evidence routing & verification
│   ├── stage07_breakthrough/              # Qwen3-0.6B fine-tuning & logit centering
│   └── stage07_local/                     # Centroid generation & local CV replay
│
├── reports/                               # Immutable evaluation artifacts & audits
│   ├── stage02a_parent_anchor/            # Baseline retriever reports & token audits
│   ├── stage07r_gold_doc_propensity/      # Propensity audit logs & validation splits
│   └── stage07y_final_clean_fusion/       # FUSION_AUDIT.json (Frozen 0.9505 evidence)
│
├── configs/                               # Stage configurations and hyperparameters
├── reproduce_gold_local_cv.py             # Preflight check & local-CV replay script
├── requirements-gold-local-lock.txt       # Locked dependencies for exact reproduction
│
└── research_archive/                      # PHASE 1 RESEARCH ARCHIVE (Historical R&D)
    └── legalir_public_phase/
        ├── README.md                      # Phase 1 technical summary & audit retrospective
        ├── docs/gemini/                   # 53-Rule forensic audit & discovery logs
        ├── src/gemini/                    # Hand-crafted 145D GBDT feature extraction
        └── src/exp_final/                 # Exploratory set rankers and early fusion
```

---

## ⚡ Quickstart & Local Reproduction

### 1. Environment Setup

```bash
# Clone the unified repository
git clone https://github.com/ngmhoang62/endgame.git
cd endgame

# Create isolated Python 3.11 environment
py -3.11 -m venv .venv
source .venv/Scripts/activate  # On Windows: .venv\Scripts\activate

# Install locked dependencies
pip install -r requirements-gold-local-lock.txt
```

### 2. Data Preparation
Place the official competition dataset inside `data/official_v1/`:
```text
data/official_v1/
  train.json
  public-official.json
  private-official.json
  DATA_SNAPSHOT_MANIFEST.json
  selected-contexts/           # 8,532 context_*.json files
```

Verify data integrity against the locked manifest:
```bash
python reproduce_gold_local_cv.py check-data
```

### 3. Replay Gold Local CV
Execute the end-to-end evaluation stages:
```bash
python reproduce_gold_local_cv.py run
```
To verify the final fusion metric against the immutable historical benchmark:
```bash
python reproduce_gold_local_cv.py verify-cv
```

---

## 📜 Citation & Acknowledgments

This solution was engineered for the **UIT Data Science Challenge 2026**. Special thanks to the organizers at University of Information Technology (VNU-HCM) for providing the benchmark datasets and computational testbed.

```bibtex
@misc{dsc2026_legalir_top7,
  author = {Nguyen Minh Hoang},
  title  = {Dual-Stage Vietnamese Legal Information Retrieval with Multi-Encoder Screening and Logit-Centered LLM Reranking},
  year   = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/ngmhoang62/endgame}}
}
```
