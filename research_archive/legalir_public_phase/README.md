# 🔬 LegalIR: Phase 1 Exploratory R&D & Forensic Audit Archive

> **Archive Status:** Historical Phase 1 Research Codebase (Public Test Evaluation ~1.5 months).  
> **Production Successor:** The production, winning Top-7 pipeline lives in the root directory under [`src/`](../../src/).

---

## 📖 Overview

This directory preserves the complete experimentation history of **Phase 1 (LegalIR)** during the UIT Data Science Challenge 2026. Over 1.5 months of R&D, this phase explored:
- Advanced structural chunking (`structural_chunker_v3.py`, `parent_child_consensus_scope_chunks.py`).
- Dense + Sparse retrieval (`VietLegal-E5`, `BGE-M3`, `BKAI-BiEncoder`, `BM25Okapi`).
- Feature engineering & Learning-to-Rank (GBDT) with up to **145 hand-crafted features** capturing document hierarchy, citations, kinship relationships, and law authority.
- Extensive diagnostic evaluations and the **Forensic Audit** that prompted the cleanroom reconstruction of the system in `endgame`.

---

## 🧪 Key Experimental Milestones

### 1. Chunking & Retrieval Experiments
- **Parent-Scope Consensus**: Chunking legal articles while attaching chapter titles and Article 1 scope definitions to combat isolated clause ambiguity.
- **Hybrid Fusion**: Benchmarking BM25 against dense representations (`VietLegal-E5` with `passage: ` / `query: ` prefixes), evaluated using Reciprocal Rank Fusion (RRF).

### 2. Learning-to-Rank (GBDT 145D) & Kinship Graphs
- Extracted comprehensive ranking feature sets:
  - Lexical overlap (BM25 raw, title match, clause exact matches).
  - Graph & Kinship features (`src/gemini/kinship.py`): Document citation networks, amending/guiding law relationships.
  - Legal Authority (`src/gemini/authority.py`): Hierarchy scores based on legal document type (Luật > Nghị định > Thông tư).
- Trained LightGBM / CatBoost rankers over shortlisted candidate pools.

### 3. The Turning Point: The Forensic Audit
During deployment verification for public holdouts, deep diagnostics revealed:
- **Article Parity Failure**: A subtle alignment shift between training queries and official test evaluation schemas across 53 legal articles.
- **The Engineering Pivot**: Rather than patching corrupted features, the decision was made to freeze `LegalIR` as historical evidence and engineer a ground-up **Cleanroom Pipeline (`endgame`)** with strict grouped-fold cross-validation, moderate region segmentation, and LLM-centric reranking.

Detailed audit reports are preserved in:
- [`docs/gemini/forensic_audit_h61_deployment_parity_and_leakage.md`](docs/gemini/forensic_audit_h61_deployment_parity_and_leakage.md)
- [`docs/gemini/forensic_audit_v5_breakthrough.md`](docs/gemini/forensic_audit_v5_breakthrough.md)
- [`docs/gemini/complexity_generalization_audit.md`](docs/gemini/complexity_generalization_audit.md)
- [`docs/gemini/research_log.md`](docs/gemini/research_log.md)

---

## 📂 Directory Structure

```text
research_archive/legalir_public_phase/
├── docs/
│   ├── gemini/            # Forensic audit logs, research logs, and breakthrough reports
│   └── exp*.md            # Experiment runbooks and plans (EXP-109 through EXP-112)
├── scripts/               # Training, evaluation, and benchmark scripts
├── src/
│   ├── exp_final/         # Final late-stage experiments, contracts, and set rankers
│   ├── gemini/            # Kinship graph, authority scores, and 145D feature extractors
│   ├── structural_chunker_v3.py
│   └── ...                # Retrieval, reranking, and evaluation modules
├── tests/                 # Unit and integration test suites
├── requirements.txt       # Dependencies for Phase 1
└── scoring.py             # Official evaluation scoring script
```
