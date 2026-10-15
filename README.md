# 🏛️ Dual-Stage Vietnamese Legal Information Retrieval System

> **Top 7 Private Leaderboard Solution** — UIT Data Science Challenge 2026 (DSC 2026)  
> *A cleanroom, production-grade hybrid retrieval and reranking engine for large-scale statutory law.*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Private LB](https://img.shields.io/badge/Private%20Leaderboard-Top%207-gold.svg)](#-empirical-results--ablation-study)
[![OOF Recall@5](https://img.shields.io/badge/OOF%20Recall%405-0.9505-brightgreen.svg)](#-empirical-results--ablation-study)
[![Candidate Acquisition](https://img.shields.io/badge/Recall%40100-99.32%25-success.svg)](#stage-1--multi-encoder-hybrid-retrieval)
[![Parameter Budget](https://img.shields.io/badge/Active%20Params-1.73B%20%2F%204.0B-purple.svg)](#5-strict-parameter-budget-governance-173b--40b)

---

## 📌 Executive Summary

This repository hosts the official source code for our **Top 7** winning solution in the **UIT Data Science Challenge 2026 (Legal Information Retrieval track)**. The task mandates identifying the most relevant legal articles/clauses from an authoritative corpus of **8,512 Vietnamese statutory laws** in response to natural language legal queries.

The architecture is built upon a **Two-Stage Hybrid Pipeline** designed with strict cleanroom engineering principles:
1. **Stage 1 (Retrieval & Multi-Encoder Shortlisting):** Multi-view candidate generation combining tuned lexical search (`BM25Okapi`) with 3 complementary Bi-Encoders (`AITeamVN/Vietnamese_Bi-Encoder`, `vnlegal-lal`, and `vietlegal-e5`/`harrier`) via GPU matrix search over **179,737 moderate legal regions**, securing **99.32% Recall@100** oracle candidate acquisition and an initial **93.58% pre-rerank OOF Recall@5** on the Top-30 shortlist.
2. **Stage 2 (Deep Reranking & Bayesian Prior Fusion):** High-capacity XLM-R Cross-Encoders (`AITeamVN/Vietnamese_Reranker` & `bge-reranker-v2-m3` fine-tuned with LoRA) coupled with an instruction-tuned **Qwen3-Reranker-0.6B LLM** (trained via Multi-task ListNet + RankNet + BCE loss, reaching **95.39% Fold-3 Recall@5**). Query calibration shifts are eliminated using **Per-Query Logit Centering**, and predictions are enriched with **Gold Document Propensity** and **Gold Centroid Head** priors, advancing clean Out-of-Fold (OOF) Recall@5 to **0.9505** under a strict **1.73B / 4.0B active parameter budget** to prevent meta-overfitting, locking **Top 7 on the Private Leaderboard**.

```text
                                  [User Legal Query]
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
   [BM25Okapi Lexical]                                     [3x Bi-Encoders GPU Search]
 (Word segmentation & tuning)                             (AITeamVN + VNLegal-LAL + E5)
            │                                                           │
            └─────────────────────────────┬─────────────────────────────┘
                                          ▼
                   [Hybrid Fusion & 179.7K Region Screening]
                   ├── Candidate Acquisition Oracle: 99.32% Recall@100
                   └── Pre-rerank Selector Top-30 Order: 93.58% OOF R@5
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
  [XLM-R Cross-Encoders]                                     [Qwen3-Reranker-0.6B]
 (AIT / BGE with LoRA on Top-30)                           (Multi-task ListNet/RankNet/BCE)
            │                                              (95.39% Fold-3 Recall@5)
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
                   Active Parameters: 1.732B / 4.0B Budget Limit
                        Top 7 Private Leaderboard Solution
```

---

## 🚀 The Engineering Story: Two-Phase Evolution

Our competitive advantage stemmed from strict **engineering hygiene and rigorous forensic auditing**, splitting the project into two distinct phases:

```text
[Phase 1: Exploratory R&D (LegalIR)]
├── Parent-Scope Chunking & GBDT Learning-to-Rank (72D -> 145D Handcrafted Features)
├── Kinship Graphs, Law Authority & Citation Network Modeling -> 95.50% OOF R@5 (Exploratory)
└── Forensic Discovery: Uncovered 53 contaminated rules with article parity mismatch
              │
              ▼ (Disciplined Engineering Decision: Freeze Phase 1 & Rebuild Cleanroom)
[Phase 2: SOTA Cleanroom Architecture (Endgame) - Winning Top 7 System]
├── 8,512 Clean Corpus Universe & 179,737 Moderate Legal Regions
├── Multi-Bi-Encoder Ensembling (99.32% Recall@100, 93.58% Initial OOF R@5)
├── XLM-R Cross-Encoders with LoRA & Centered Qwen3-0.6B Reranker (95.39% Fold-3 R@5)
└── Bayesian Priors: Gold Document Propensity (0.9505 Clean OOF R@5 -> Top 7 Private LB)
```

- **Phase 1 Archive ([`research_archive/legalir_public_phase/`](research_archive/legalir_public_phase/)):** Contains exploratory work with 145-dimensional GBDT features (LightGBM/XGBoost achieving **95.50% OOF Recall@5** on early exploratory splits) and the detailed forensic audit logs identifying deployment parity failure across 53 rules.
- **Phase 2 Production ([`src/`](src/)):** Cleanroom codebase with zero data leakage, group-isolated 5-fold cross-validation, and production reranking modules delivering the unbiased **0.9505 OOF Recall@5** and **Top 7 Private Leaderboard**.

---

## 📊 Empirical Results & Ablation Study

All metrics below are evaluated under the frozen **6,991 primary evaluable queries** across 5 grouped folds (Seed 276) matching the official competition evaluation protocol.

| Pipeline Stage | Model / Strategy | Metric Score | Δ vs Prev | Description / Key Insight |
|---|---|:---:|:---:|---|
| **Phase 1 Baseline** | 145D LightGBM / XGBoost LTR | `0.9550` OOF R@5 | — | Exploratory R&D phase with 145 features across 5 families before forensic audit |
| **Stage 01 Raw Lexical** | BM25Okapi (Passage-level) | `~0.7600` OOF R@5 | — | Lexical search baseline over raw parent articles |
| **Stage 02B Retriever** | VietLegal-E5 Bi-Encoder | `0.8520` OOF R@5 | +9.20% | Dense representations on moderate regions |
| **Stage 02B Retriever** | AITeamVN Bi-Encoder | `0.8710` OOF R@5 | +1.90% | State-of-the-art Vietnamese sentence embeddings |
| **Stage 02B Multi-Retrieval** | 3 Bi-Encoders + BM25 | **`0.9932` Recall@100** | — | **Candidate Acquisition Oracle**: 99.32% gold recall at depth 100 |
| **Stage 03B0 Shortlist Order**| Pre-Rerank Selector Order | **`0.9358` OOF R@5** | — | Initial candidate ranking of the fused Top-30 shortlist |
| **Stage 03B1 Cross-Encoder** | AIT XLM-R Reranker (CE-LR 45D) | `0.9426` OOF R@5 | +0.68% | Full cross-attention interaction over top-30 shortlist |
| **Stage 06B / 07L LoRA** | XLM-R Cross-Encoders + LoRA | `0.9438` OOF R@5 | +0.12% | BGE-Reranker-v2-m3 & AITeamVN adapted with PEFT/LoRA |
| **Stage 07B LLM Rerank** | Qwen3-Reranker-0.6B (Raw) | `0.9452` OOF R@5 | +0.14% | Fine-tuned with ListNet (0.55) + RankNet (0.25) + BCE (0.20) |
| **Stage 07B Fold Validation**| Qwen3-Reranker-0.6B (Fold-3) | **`0.9539` Fold-3 R@5** | — | In-memory evaluation on held fold 3 (`0.95386`) |
| **Stage 07N Calibration** | Qwen3-0.6B + Logit Centering | `0.9465` OOF R@5 | +0.13% | Subtracts query-mean logit to eliminate query scale bias |
| **Stage 07O Semantic Prior** | Gold Centroid Head | `0.9474` OOF R@5 | +0.09% | Distance projection to empirical gold embedding centroid |
| **Stage 07R Statistical Prior**| Gold Document Propensity | `0.9489` OOF R@5 | +0.15% | Bayesian prior addressing long-tail statutory frequency |
| **Stage 07Y Final Clean Fusion**| **AIT + Qwen + Propensity ($w=0.75$)** | **`0.9505` OOF R@5** | **+0.16%** | **Final Winning Submission (`GOLD_PROP_AIT_W75_K5.zip`) — Top 7** |

---

## 🔬 Core Architectural Innovations

### 1. Structure-Aware Moderate Region Segmentation (Stage 00C & 02B)
Vietnamese statutory texts exhibit extreme structural variance—ranging from concise definitions to sprawling 40-clause articles.
- Raw documents were cleaned from 8,532 to **8,512 retrievable documents** (discarding 20 empty-passage artifacts).
- Structured segmentation produced **179,737 moderate regions** (Điều, Khoản, Phụ lục) preserving hierarchical context without exceeding transformer context windows.

### 2. Multi-Bi-Encoder Complementary Screening (Stage 02B)
To guarantee that the reranker receives the correct article in its candidate window:
- Integrated 3 Bi-Encoders: `AITeamVN/Vietnamese_Bi-Encoder` (568M), `vnlegal-lal` (596M), and `vietlegal-e5`.
- Fused with lexical BM25 using GPU-accelerated matrix multiplication for cosine similarity search.
- Achieved **99.32% Recall@100** oracle acquisition coverage and a pre-rerank shortlist Recall@5 of **93.58%**.

### 3. XLM-R Cross-Encoders & Qwen3-Reranker-0.6B with LoRA (Stage 06B & 07B)
- **XLM-RoBERTa Cross-Encoders:** Deployed `AITeamVN/Vietnamese_Reranker` and `BAAI/bge-reranker-v2-m3` (both built on XLM-RoBERTa architecture), fine-tuned with Parameter-Efficient Fine-Tuning (PEFT/LoRA) over the Top-30 candidates.
- **Instruction-Tuned Qwen3-0.6B:** Trained using a multi-task objective balancing listwise, pairwise, and classification dynamics:
  $$\mathcal{L} = 0.55 \cdot \mathcal{L}_{\text{ListNet}} + 0.25 \cdot \mathcal{L}_{\text{RankNet/Margin}} + 0.20 \cdot \mathcal{L}_{\text{BCE}}$$
  Reaching **95.39% Recall@5** on held Fold 3 validation.
- **Per-Query Logit Centering:** 
  $$\hat{s}(q, d_i) = s(q, d_i) - \frac{1}{|K|}\sum_{j \in K} s(q, d_j)$$
  Zero-centers the reranker output per candidate slate, eliminating inter-query distribution shifts and ensuring cross-query comparability.

### 4. Bayesian Priors: Gold Document Propensity & Centroid Head (Stage 07O, 07R & 07Y)
- **Gold Document Propensity:** Formulated a regularized empirical prior representing statutory citation frequency across the legal corpus, penalizing spurious one-off lexical matches.
- **Gold Centroid Head:** Projected query-candidate pairs against an aggregated centroid of confirmed gold document vectors.
- **Robust Clean Fusion ($w_{prop}=0.75$):** Linearly combined cross-encoder logits with the propensity prior, hitting **0.9505 OOF Recall@5**.

### 5. Strict Parameter Budget Governance (1.73B / 4.0B)
Under official UIT DSC 2026 competition regulations, submission pipelines are bounded by an **active parameter budget of 4.0B**.
- Our final production pipeline operates with **1.732B active parameters**:
  - `AITeamVN Embedding`: 568M
  - `VNLegal-LAL Retriever`: 596M
  - `AITeamVN Reranker`: 568M
  - **Total Active Parameters:** $\mathbf{1.732B \ll 4.0B}$
- This lightweight parameter footprint prevented meta-overfitting while ensuring blazing-fast inference speeds.

---

## 📁 Repository Structure

```text
├── src/                                   # SOTA Cleanroom Production Pipeline (Top 7)
│   ├── common/                            # Metric definitions & evaluation protocols
│   ├── stage00_audit/                     # Corpus sanitization (8,512 docs universe)
│   ├── stage01_query_analysis/            # Query length, intent & duplicate analysis
│   ├── stage02_candidate_generation/      # 179.7K region indexer & multi-bi-encoders
│   ├── stage03_rerank/                    # Cross-encoder inference & shortlist builder
│   ├── stage04_private/                   # Private leaderboard inference runners
│   ├── stage05_analysis/                  # Per-fold diagnostics & error breakdown
│   ├── stage06_evidence/                  # Sub-clause evidence routing & verification
│   ├── stage06_reranker/                  # BGE / XLM-R Cross-Encoder LoRA training
│   ├── stage07_breakthrough/              # Qwen3-0.6B multi-task fine-tuning & centering
│   └── stage07_local/                     # Centroid generation & local CV replay
│
├── reports/                               # Immutable evaluation artifacts & audits
│   ├── stage02a_parent_anchor/            # Baseline retriever reports & token audits
│   ├── stage02b5_acquisition_oracle/      # ACQUISITION_ORACLE.json (99.32% Recall@100 evidence)
│   ├── stage03b1_aiteam_reranker/         # RERANK_OOF.json (93.58% pre-rerank & 94.26% CE)
│   ├── stage07n_centered_qwen_private/    # CANDIDATE_REPORT.json (Logit centering audit)
│   ├── stage07r_gold_doc_propensity/      # Propensity audit logs & validation splits
│   └── stage07y_final_clean_fusion/       # FUSION_AUDIT.json (Frozen 0.9505 winning evidence)
│
├── configs/                               # Stage configurations and hyperparameters
├── reproduce_gold_local_cv.py             # Preflight check & local-CV replay script
├── requirements-gold-local-lock.txt       # Locked dependencies for exact reproduction
│
└── research_archive/                      # PHASE 1 RESEARCH ARCHIVE (Historical R&D)
    └── legalir_public_phase/
        ├── README.md                      # Phase 1 technical summary & audit retrospective
        ├── docs/gemini/                   # 145D GBDT audit (95.50% OOF) & 53-rule parity discovery
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
