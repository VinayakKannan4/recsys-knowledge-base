# Blogs Index

**Summary**: Catalog of all engineering blog posts in raw/blogs/, with company, date, key takeaway, and links to source files.

**Tags**: #index #blogs #meta #pinterest #tiktok #retrieval #ranking #feature-interaction #serving #sequence-modeling

**Sources**: raw/blogs/\*\*/\*.md

**Last Updated**: 2026-04-07

---

## Overview

11 blog posts currently ingested, spanning Meta, Pinterest, TikTok, and third-party analysis. Pinterest is the most heavily represented company with 6 posts covering their full ads stack evolution. Topics range from feature interaction and multi-task learning to serving infrastructure, online-offline discrepancy, and next-generation GPU-based ranking.

---

## Meta

| Title | Date | Key Takeaway | Raw File |
|-------|------|-------------|----------|
| **Sequence learning: A paradigm shift for personalized ads recommendations** | Nov 2024 | Meta replaced DLRM-style human-engineered sparse features with event-based sequence learning (EBFs + transformer attention), preserving sequential and temporal order; achieved 2–4% more conversions on select segments. | [sequence-learning-ads-2024.md](../../raw/blogs/meta/sequence-learning-ads-2024.md) |

---

## Pinterest

| Title | Date | Key Takeaway | Raw File |
|-------|------|-------------|----------|
| **Contextual relevance in ads ranking** | Apr 2020 | Pinterest moved beyond engagement-only ranking to explicit contextual relevance scoring using human-labeled data, calibrated regression models, and relevance filtering early in the retrieval funnel. | [contextual-relevance-ads-ranking.md](../../raw/blogs/pinterest/contextual-relevance-ads-ranking.md) |
| **Unpacking How Ad Ranking Works at Pinterest** | Mar 2024 | End-to-end overview of Pinterest's ads delivery funnel (retrieval → ranking → auction), covering two-tower EBR, the evolution from GBDT+LR to DNN/Transformer, PinnerFormer user sequences, and MLOps/monitoring practices. | [ad-ranking-infoq-2024.md](../../raw/blogs/pinterest/ad-ranking-infoq-2024.md) |
| **Evolution of Ads Conversion Optimization Models at Pinterest** | Jan 2024 | Documents Pinterest's conversion model journey from GBDT+LR (2018) → AutoML → multi-task ensemble of DCNv2 + Transformer + MaskNet (DHEN framework) with GPU serving; highlights unique challenges of sparse, delayed conversion labels. | [ads-conversion-optimization-evolution-2024.md](../../raw/blogs/pinterest/ads-conversion-optimization-evolution-2024.md) |
| **Handling Online-Offline Discrepancy in Pinterest Ads Ranking System** | Jan 2024 | Catalogs five root causes for offline AUC gains not translating to online CPA improvement (metric misalignment, feature delays, ensemble dilution, training-serving skew) and Pinterest's tooling investments to diagnose them quickly. | [online-offline-discrepancy-2024.md](../../raw/blogs/pinterest/online-offline-discrepancy-2024.md) |
| **Establishing a Large Scale Learned Retrieval System at Pinterest** | Jan 2025 | How Pinterest replaced heuristic (graph/interest-based) homefeed candidate generators with a production two-tower embedding retrieval system, including popularity-bias correction and auto-retraining with version-synchronized ANN indexing. | [large-scale-learned-retrieval-2025.md](../../raw/blogs/pinterest/large-scale-learned-retrieval-2025.md) |
| **Beyond Two Towers: Re-architecting the Serving Stack for Next-Gen Ads Lightweight Ranking Models (Part 1)** | Feb 2026 | Pinterest replaced its Two-Tower lightweight ranker with a GPU-based general-purpose model by embedding high-value ad features as PyTorch registered buffers and moving business logic into the model; reduced p90 latency from 4000 ms to 20 ms. | [beyond-two-towers-2026.md](../../raw/blogs/pinterest/beyond-two-towers-2026.md) |

---

## TikTok

| Title | Date | Key Takeaway | Raw File |
|-------|------|-------------|----------|
| **The Secret Sauce of TikTok's Recommendations** | Feb 2023 | Analysis of TikTok's Monolith recommendation system: how a collisionless cuckoo-hash embedding table and a Kafka-based online training architecture enable real-time model updates from user feedback at production scale. | [tiktok-secret-sauce-shaped.md](../../raw/blogs/tiktok/tiktok-secret-sauce-shaped.md) |

---

## Other / Third-Party Analysis

| Title | Source | Date | Key Takeaway | Raw File |
|-------|--------|------|-------------|----------|
| **Aman's AI Journal: Recommendation Systems — Popular Architectures** | aman.ai | — | Comprehensive reference primer on RecSys architectures from Wide & Deep (2016) through DHEN (2022), covering FM, DeepFM, NCF, DCN, AutoInt, DLRM, DCN V2, and GNN-based models with detailed architectural explanations. | [aman-ai-recsys-architectures.md](../../raw/blogs/other/aman-ai-recsys-architectures.md) |
| **The Rise of Deep and Cross Networks in Recommender Systems** | ML Frontiers (Substack) | Feb 2024 | Traces the DCN family evolution (DCN → DCN-V2 → DCN-Mix → GDCN), explaining how replacing weight vectors with matrices, adding low-rank MoE, and gating noisy crosses each improved Criteo benchmark performance. | [rise-of-dcn-ml-frontiers.md](../../raw/blogs/other/rise-of-dcn-ml-frontiers.md) |
| **The Two-Tower Model for Recommendation Systems: A Deep Dive** | Shaped | May 2025 | Deep dive on two-tower architecture covering training (in-batch negatives, sampling-bias correction) and serving (ANN/HNSW), with comparisons to MF, FMs, and GNNs and guidance on where ranking handoff is needed. | [two-tower-deep-dive-shaped.md](../../raw/blogs/other/two-tower-deep-dive-shaped.md) |

---

## Coverage by Topic

| Topic | Blog Posts |
|-------|-----------|
| Two-tower retrieval / EBR | Pinterest (large-scale-learned-retrieval-2025, ad-ranking-infoq-2024, beyond-two-towers-2026), Other (two-tower-deep-dive-shaped) |
| Feature interaction (DCN, Transformer, MaskNet) | Pinterest (ads-conversion-optimization-evolution-2024), Other (aman-ai-recsys-architectures, rise-of-dcn-ml-frontiers) |
| Sequence modeling | Meta (sequence-learning-ads-2024), Pinterest (ad-ranking-infoq-2024, ads-conversion-optimization-evolution-2024) |
| Multi-task learning | Pinterest (ads-conversion-optimization-evolution-2024) |
| Conversion optimization | Pinterest (ads-conversion-optimization-evolution-2024, online-offline-discrepancy-2024) |
| Online-offline discrepancy | Pinterest (online-offline-discrepancy-2024) |
| GPU serving / inference | Pinterest (ads-conversion-optimization-evolution-2024, beyond-two-towers-2026) |
| Online training / real-time updates | TikTok (tiktok-secret-sauce-shaped), Meta (sequence-learning-ads-2024) |
| Ads relevance / contextual ranking | Pinterest (contextual-relevance-ads-ranking) |

---

## Related Pages

- [[master-index]] — Master table of contents across all source types
- [[papers-index]] — Academic papers with key contributions
