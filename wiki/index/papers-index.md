# Papers Index

**Summary**: Catalog of all academic papers in raw/papers/, with title, authors, year, company, key contribution, and links to source files.

**Tags**: #index #papers #ranking #retrieval #multi-task #feature-interaction #infrastructure #sequence-modeling

**Sources**: raw/papers/*/\*-meta.md

**Last Updated**: 2026-04-07

---

## Overview

This index covers 10 papers spanning the core architectural families in recommendation systems: feature interaction models (Wide & Deep, DeepFM, DCN V2, DLRM), retrieval models (YouTube DNN, Facebook EBR, Sampling-Bias-Corrected Two-Tower), user behavior modeling (DIN), multi-task learning (MMoE), and serving infrastructure (Monolith). Together they trace the industry evolution from shallow models to deep learning at scale across Google, Meta, Alibaba, Huawei, and TikTok.

---

## Feature Interaction & Ranking

| Title | Authors | Year | Company | Key Contribution | Raw File |
|-------|---------|------|---------|-----------------|----------|
| **Wide & Deep Learning for Recommender Systems** | Cheng, Koc, Harmsen et al. | 2016 | Google | Jointly trains a wide linear model (memorization via cross-product feature transforms) and a deep neural network (generalization via embeddings), deployed on Google Play with significant app acquisition gains. | [wide-and-deep-meta.md](../../raw/papers/wide-and-deep/wide-and-deep-meta.md) |
| **DeepFM: A Factorization-Machine based Neural Network for CTR Prediction** | Guo, Tang, Ye, Li, He | 2017 | Huawei | Replaces Wide & Deep's hand-crafted wide component with a factorization machine sharing inputs with the deep network, enabling end-to-end CTR prediction with no manual feature engineering. | [deepfm-meta.md](../../raw/papers/deepfm/deepfm-meta.md) |
| **DCN V2: Improved Deep & Cross Network** | Wang, Shivanna, Cheng et al. | 2020 | Google | Upgrades DCN's cross-network by replacing the weight vector with a full weight matrix for richer bit-wise feature interactions; adds optional low-rank mixture-of-experts decomposition for efficiency at web scale. | [dcn-v2-meta.md](../../raw/papers/dcn-v2/dcn-v2-meta.md) |
| **Deep Learning Recommendation Model for Personalization and Recommendation Systems (DLRM)** | Naumov, Mudigere, Shi et al. | 2019 | Meta | Production-scale recommendation model combining sparse embedding tables with dense MLPs, with model parallelism on embeddings and data parallelism on fully-connected layers to handle memory constraints at scale. | [dlrm-meta.md](../../raw/papers/dlrm/dlrm-meta.md) |

---

## Sequence Modeling

| Title | Authors | Year | Company | Key Contribution | Raw File |
|-------|---------|------|---------|-----------------|----------|
| **Deep Interest Network for Click-Through Rate Prediction (DIN)** | Zhou, Song, Zhu et al. | 2018 | Alibaba | Introduces a local activation unit that adaptively computes user interest representations from historical behavior relative to each candidate ad, replacing a single fixed-length user vector that loses diversity of interests. | [din-deep-interest-network-meta.md](../../raw/papers/din-deep-interest-network/din-deep-interest-network-meta.md) |

---

## Retrieval / Candidate Generation

| Title | Authors | Year | Company | Key Contribution | Raw File |
|-------|---------|------|---------|-----------------|----------|
| **Deep Neural Networks for YouTube Recommendations** | Covington, Adams, Sargin | 2016 | Google/YouTube | Two-stage DNN pipeline (candidate generation via in-batch softmax + ranking via regression) that scaled to hundreds of millions of users; establishes the modern retrieval-then-rank paradigm. | [youtube-dnn-recommendations-meta.md](../../raw/papers/youtube-dnn-recommendations/youtube-dnn-recommendations-meta.md) |
| **Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations** | Yi, Yang, Hong et al. | 2019 | Google | Corrects popularity bias introduced by in-batch negative sampling in two-tower models via a streaming frequency estimator that adjusts logits, improving retrieval quality over large item corpora. | [two-tower-sampling-bias-meta.md](../../raw/papers/two-tower-sampling-bias/two-tower-sampling-bias-meta.md) |
| **Embedding-based Retrieval in Facebook Search (EBR)** | Huang, Sharma, Sun et al. | 2020 | Meta | Unified two-tower embedding framework for personalized semantic retrieval in Facebook Search, integrating social graph context; covers end-to-end ANN optimization and full-stack production deployment lessons. | [facebook-ebr-meta.md](../../raw/papers/facebook-ebr/facebook-ebr-meta.md) |

---

## Multi-Task Learning

| Title | Authors | Year | Company | Key Contribution | Raw File |
|-------|---------|------|---------|-----------------|----------|
| **Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts (MMoE)** | Ma, Zhao, Yi, Chen, Hong, Chi | 2018 | Google | Multi-gate MoE architecture with task-specific gating networks that selectively combine shared expert modules, outperforming hard and soft parameter sharing baselines on tasks with varying relationships. | [mmoe-multi-task-meta.md](../../raw/papers/mmoe-multi-task/mmoe-multi-task-meta.md) |

---

## Serving Infrastructure

| Title | Authors | Year | Company | Key Contribution | Raw File |
|-------|---------|------|---------|-----------------|----------|
| **Monolith: Real Time Recommendation System With Collisionless Embedding Table** | Liu, Zou, Zou et al. | 2022 | TikTok/ByteDance | Collisionless cuckoo-hash embedding table with expirable embeddings and frequency filtering, combined with a fault-tolerant online training architecture that continuously updates model weights from real-time user feedback. | [monolith-tiktok-meta.md](../../raw/papers/monolith-tiktok/monolith-tiktok-meta.md) |

---

## arXiv Quick Reference

| Paper | arXiv ID |
|-------|----------|
| Wide & Deep | 1606.07792 |
| DeepFM | 1703.04247 |
| DIN | 1706.06978 |
| DLRM | 1906.00091 |
| Facebook EBR | 2006.11632 |
| DCN V2 | 2008.13535 |
| Monolith | 2209.07663 |
| YouTube DNN | ACM DL (no arXiv) |
| Sampling-Bias-Corrected Two-Tower | ACM DL (no arXiv) |
| MMoE | ACM DL (no arXiv) |

---

## Related Pages

- [[master-index]] — Master table of contents across all source types
- [[blogs-index]] — Engineering blog posts with practical deployment details
