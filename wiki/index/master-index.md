# Master Index

**Summary**: Master table of contents for the RecSys knowledge base — all wiki pages organized by category, with one-line summaries and links to raw sources.

**Tags**: #index #master

**Last Updated**: 2026-04-07

---

## Overview

Entry point for the knowledge base. The wiki covers recommendation systems for social and ad platforms, with emphasis on Meta, Pinterest, Google/YouTube, and TikTok. Raw sources live in `raw/papers/` and `raw/blogs/`; compiled wiki pages are in `wiki/`.

**Current state**: 10 papers · 11 blog posts · 7 architecture pages · 7 concept pages · 4 company pages · 4 comparison pages · 3 index pages

---

## Architecture Pages

| Page | One-line Summary |
|------|-----------------|
| [[dlrm]] | Meta's 2019 dot-product interaction model: sparse embeddings + pairwise dot products + MLP; canonical production ranking architecture |
| [[dcn-v2]] | Google's 2020 matrix cross-layer model: bit-wise feature interactions of arbitrary polynomial degree; current dominant ranking family |
| [[two-tower-model]] | Dual-encoder retrieval architecture: separate user/item towers scored by dot product, enabling ANN search over billion-item corpora |
| [[monolith]] | TikTok's real-time system: collisionless cuckoo-hashmap embedding table + streaming Kafka/Flink online training with minutes update latency |
| [[din-dien]] | Alibaba's target-aware attention over user history: adaptive user representation per candidate ad, with DIEN adding GRU-based interest evolution |
| [[wide-and-deep]] | Google's 2016 joint memorization + generalization model: manual cross features (wide) + MLP (deep); established cross features as a first-class primitive |
| [[deepfm]] | Huawei's 2017 FM-based CTR model: shared-input FM layer for automatic pairwise interactions + deep MLP; no manual feature engineering |

---

## Concept Pages

| Page | One-line Summary |
|------|-----------------|
| [[feature-crosses]] | Second- (and higher-) order feature products: from Wide & Deep manual crosses (2016) → FM dot products → DCN polynomial → DCN-V2 matrix → GDCN gated |
| [[embedding-tables]] | Learned sparse-ID-to-dense-vector lookup tables: the dominant memory consumer in production recommendation models; covers hashing, collisions, and Monolith's cuckoo hashmap |
| [[multi-task-learning]] | Joint training for CTR + CVR + engagement + quality: shared-bottom → MMoE (task-specific expert gating) → PLE (private + shared experts) |
| [[negative-sampling]] | Constructing training negatives from implicit feedback: random vs. in-batch vs. hard negatives; Yi et al. sampling-bias correction; Meta's hard negative mining |
| [[click-through-rate-prediction]] | Estimating P(click \| user, item, context): calibration for auction correctness, position-bias correction, and the full CTR model architecture timeline |
| [[online-learning]] | Continuous streaming model updates vs. batch retraining: concept drift latency, Monolith's Worker-PS architecture, and Meta's accelerated batch approach |
| [[two-stage-pipeline]] | Universal retrieval → ranking funnel: ANN retrieval for recall over billions, expensive rankers for precision over hundreds; latency budget breakdown |

---

## Company Pages

| Page | One-line Summary |
|------|-----------------|
| [[meta-ads-ranking]] | Meta's ads stack: EBR retrieval → DLRM (2019) → DHEN (2022) → EBF + Transformer sequence learning (2024); batch training, Big Basin hardware co-design |
| [[google-youtube-ranking]] | Google's pipeline: YouTube DNN two-stage template (2016) → Wide & Deep → sampling-bias-corrected two-tower (2019) → DCN-V2 cross-product deployment (2020) |
| [[tiktok-bytedance-ranking]] | TikTok's For You feed: multi-strategy retrieval → Monolith online-trained ranker; completion rate + multi-task objectives; real-time within-minutes model updates |
| [[pinterest-ads-ranking]] | Pinterest's evolving stack: GBDT+LR (2018) → DNN+MTL → DCNv2+Transformer+MaskNet ensemble (2023) → GPU "Beyond Two Towers" pre-ranking (2026) |

---

## Comparison Pages

| Page | One-line Summary |
|------|-----------------|
| [[dlrm-vs-dcn-v2]] | DLRM dot-product interactions vs. DCN-V2 matrix cross layers: expressiveness ceiling, hardware cost, and when to choose each |
| [[two-tower-vs-late-fusion]] | Two-tower retrieval (pre-computed, recall-optimized) vs. late fusion ranking (joint user×item interactions, precision-optimized): where each belongs and the emerging middle ground |
| [[batch-vs-online-training]] | Meta's accelerated batch cycles + rich serving-time features vs. TikTok's Monolith streaming training: content velocity, failure modes, and infrastructure complexity |
| [[company-pipeline-comparison]] | Side-by-side Meta vs. Google vs. TikTok vs. Pinterest: full retrieval → ranking → auction pipeline diagrams, architecture choices, training paradigms, and key differentiators |

---

## Index Pages

| Page | Description |
|------|-------------|
| [[master-index]] | This page — top-level table of contents |
| [[papers-index]] | All 10 papers with authors, years, key contributions, and arXiv IDs |
| [[blogs-index]] | All 11 blog posts with company, date, key takeaway, and coverage-by-topic matrix |
| [[link-graph]] | Full [[backlink]] relationship graph — every page's outgoing and incoming links |

---

## Raw Sources

### Papers (10)

| Paper | Company | Year | Topic |
|-------|---------|------|-------|
| [Wide & Deep](../../raw/papers/wide-and-deep/wide-and-deep-meta.md) | Google | 2016 | Feature interaction, ranking |
| [YouTube DNN](../../raw/papers/youtube-dnn-recommendations/youtube-dnn-recommendations-meta.md) | Google/YouTube | 2016 | Retrieval, two-tower, ranking |
| [DeepFM](../../raw/papers/deepfm/deepfm-meta.md) | Huawei | 2017 | Feature interaction, CTR |
| [DIN](../../raw/papers/din-deep-interest-network/din-deep-interest-network-meta.md) | Alibaba | 2018 | Sequence modeling, CTR |
| [MMoE](../../raw/papers/mmoe-multi-task/mmoe-multi-task-meta.md) | Google | 2018 | Multi-task learning |
| [DLRM](../../raw/papers/dlrm/dlrm-meta.md) | Meta | 2019 | Architecture, infrastructure |
| [Sampling-Bias-Corrected Two-Tower](../../raw/papers/two-tower-sampling-bias/two-tower-sampling-bias-meta.md) | Google | 2019 | Retrieval, two-tower |
| [EBR](../../raw/papers/facebook-ebr/facebook-ebr-meta.md) | Meta | 2020 | Retrieval, two-tower, ANN |
| [DCN V2](../../raw/papers/dcn-v2/dcn-v2-meta.md) | Google | 2020 | Feature interaction, ranking |
| [Monolith](../../raw/papers/monolith-tiktok/monolith-tiktok-meta.md) | TikTok/ByteDance | 2022 | Serving, online training |

### Blog Posts (11)

| Post | Company | Date | Topic |
|------|---------|------|-------|
| [Sequence learning: EBF + Transformer](../../raw/blogs/meta/sequence-learning-ads-2024.md) | Meta | Nov 2024 | Sequence modeling, event-based features |
| [Contextual relevance in ads ranking](../../raw/blogs/pinterest/contextual-relevance-ads-ranking.md) | Pinterest | Apr 2020 | Relevance modeling |
| [Evolution of Ads Conversion Optimization](../../raw/blogs/pinterest/ads-conversion-optimization-evolution-2024.md) | Pinterest | Jan 2024 | MTL, DCNv2+Transformer+MaskNet, GPU serving |
| [Online-Offline Discrepancy](../../raw/blogs/pinterest/online-offline-discrepancy-2024.md) | Pinterest | Jan 2024 | Train-serve skew |
| [How Ad Ranking Works at Pinterest](../../raw/blogs/pinterest/ad-ranking-infoq-2024.md) | Pinterest | Mar 2024 | End-to-end pipeline overview |
| [Large Scale Learned Retrieval at Pinterest](../../raw/blogs/pinterest/large-scale-learned-retrieval-2025.md) | Pinterest | Jan 2025 | Two-tower retrieval, ANN |
| [Beyond Two Towers](../../raw/blogs/pinterest/beyond-two-towers-2026.md) | Pinterest | Feb 2026 | GPU pre-ranking, 4000ms → 20ms |
| [TikTok's Secret Sauce](../../raw/blogs/tiktok/tiktok-secret-sauce-shaped.md) | TikTok | Feb 2023 | Monolith, online training |
| [Aman's AI: RecSys Architectures](../../raw/blogs/other/aman-ai-recsys-architectures.md) | aman.ai | — | Architecture survey, Wide&Deep through DHEN |
| [Rise of Deep and Cross Networks](../../raw/blogs/other/rise-of-dcn-ml-frontiers.md) | ML Frontiers | Feb 2024 | DCN family, Criteo leaderboard |
| [Two-Tower Model: A Deep Dive](../../raw/blogs/other/two-tower-deep-dive-shaped.md) | Shaped | May 2025 | Two-tower training and serving |

---

## Related Pages

- [[papers-index]] — academic paper details
- [[blogs-index]] — blog post details
- [[link-graph]] — full backlink relationship map
