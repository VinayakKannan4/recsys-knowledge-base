# Meta Ads Ranking

**Summary**: Meta's ads and recommendations ranking stack, anchored by DLRM (2019) and evolving toward Transformer-based sequence learning over event-based features (2024) — spanning candidate retrieval via embedding-based retrieval (EBR) through heavy ranking, multi-task prediction, and auction.

**Tags**: #meta #facebook #instagram #ads #ranking #dlrm #sequence-learning #production

**Sources**:
- raw/papers/dlrm/dlrm-meta.md
- raw/papers/facebook-ebr/facebook-ebr-meta.md
- raw/blogs/meta/sequence-learning-ads-2024.md
- raw/blogs/other/aman-ai-recsys-architectures.md

**Last Updated**: 2026-04-07

---

## Overview

Meta operates one of the largest ads recommendation systems in the world, serving personalized ads across Facebook, Instagram, Messenger, and Audience Network to over 3 billion monthly active users. The system spans the full recommendation pipeline: retrieval, ranking, auction, and serving — each optimized for Meta's specific constraints: enormous feature vocabularies (billions of user/item IDs), extremely high QPS, and strict per-request latency budgets of hundreds of milliseconds.

Meta's ranking system is characterized by three major architectural eras:

1. **DLRM era (2019–~2022)**: Embedding tables + pairwise dot-product interactions + MLP. Open-sourced as a benchmark for hardware co-design. The canonical production architecture that defined recommendation model infrastructure at Meta.
2. **DHEN era (~2022–2023)**: Hierarchical ensemble of interaction modules (dot products, self-attention, convolution, DCN-style crossing) built on top of DLRM's feature processing. Demonstrated +0.27% NE improvement over DLRM at internal scale.
3. **Sequence learning era (2024–present)**: Replacement of human-engineered sparse features with event-based features (EBFs) and Transformer-based sequence models. Demonstrated 2–4% conversion improvement on select segments.

---

## System Architecture

Meta's ads serving pipeline follows the standard multi-stage funnel, with each stage applying progressively more expensive models to progressively fewer candidates:

```
[Full ads inventory: billions of ads]
         ↓
[Retrieval / Candidate Generation]
  - Embedding-Based Retrieval (EBR) for Search
  - Multi-stage targeting and retrieval for Feed/Reels
  - Two-tower style: user embedding ANN over item embeddings
  → O(thousands) candidates
         ↓
[Light Ranking / Pre-ranking]
  - Lightweight models to score and filter
  → O(hundreds) candidates
         ↓
[Heavy Ranking]
  - DLRM / DHEN / Sequence model
  - Multi-task: pCTR, pCVR, pEngagement
  - Full feature materialization per candidate
  → O(tens) candidates
         ↓
[Auction]
  - expected_value = pCTR × bid (second-price or VCG variant)
  - Ad quality score incorporated
  → Winning ads
         ↓
[User Feed]
```

**Key infrastructure facts**:
- Embedding tables: terabyte-scale; model-parallel sharding across GPUs
- Dense layers (MLP): data-parallel across devices
- Serving: custom Big Basin AI platform optimized for recommendation workloads
- Training: PyTorch + Caffe2; DLRM open-sourced in both frameworks

---

## Model Evolution Timeline

### Pre-2019: Matrix Factorization and Early Deep Learning
- Collaborative filtering and matrix factorization baselines
- Early deep learning experiments with standard MLP ranking models
- Manual feature engineering for cross features

### 2019: DLRM — Deep Learning Recommendation Model
Naumov et al. published and open-sourced DLRM, which became the canonical Meta production architecture:

- **Architecture**: dense features → Bottom MLP → projected dense embedding; sparse features → embedding table lookup; all embedding vectors → pairwise dot product interactions; concatenated interactions → Top MLP → CTR prediction
- **Key design choices**: treat each embedding vector as an atomic unit (unlike DCN which operates element-wise); limit interactions to second-order (pairwise dot products)
- **Hardware innovation**: hybrid model parallelism (embedding tables) + data parallelism (MLPs); established the template for recommendation hardware co-design
- **Performance**: beats DCN on Criteo; published as a benchmark for system co-design

### ~2020: Embedding-Based Retrieval (EBR) for Facebook Search
Huang et al. described the deployment of embedding-based retrieval in Facebook Search — the first time Meta moved away from Boolean matching to neural retrieval:

- **Context**: Facebook Search posed unique challenges vs. web search — social graph context (searcher's friends, interests, social connections) is a critical signal
- **Architecture**: two-tower model with user context tower incorporating social graph features; ANN index served via FAISS
- **Training**: hard negative mining was central — after initial training on easy in-batch negatives, near-miss negatives (items ranked just outside top-K) were mined and used for retraining
- **Deployment**: served via inverted index infrastructure adapted to work with embedding lookup; full-stack optimization including ANN parameter tuning

### ~2022: DHEN — Deep Hierarchical Ensemble Network
Zhang et al. at Meta introduced DHEN to address DLRM's limitation: second-order interactions miss higher-order signals (e.g., user × movie × director triples):

- **Architecture**: replaces DLRM's dot product interactions with a hierarchy of interaction modules including dot products, self-attention (AutoInt-style), convolution, linear transforms, and DCN-style crossing
- **Result**: +0.27% NE (Normalized Entropy) improvement over DLRM on internal CTR data
- **Infrastructure**: required a new distributed training paradigm — Hybrid Sharded Data Parallel — achieving 1.2× throughput improvement over prior distributed learning
- **Context**: even a 0.27% NE improvement is significant at Meta's scale, justifying the substantial increase in model complexity

### 2024: Sequence Learning — Event-Based Features (EBF) + Transformer
Sri Reddy et al. described a paradigm shift from DLRM's aggregated sparse features to direct sequence learning over raw event streams:

**The problem with DLRM-era sparse features**:
- Sparse features were hand-engineered aggregations: "ads user clicked in the last N days", "pages visited in past M days with visit counts"
- These aggregations lost sequential information (order of events), fine-grained colocation (which attributes appeared together in the same event), and required human intuition to design
- Redundant overlapping aggregations increased compute/storage and made feature management cumbersome

**Event-Based Features (EBFs)**:
EBFs replace aggregated sparse features with structured representations of individual events:
1. **Event stream**: the sequence of recent events of a given type (e.g., recent ad engagements)
2. **Sequence length**: how many recent events to include (tuned by stream importance)
3. **Event information**: semantic/contextual attributes of each event (ad category, timestamp)

An EBF is a coherent object capturing all key information about an event, akin to a "word" in an NLP model. The vocabulary is orders of magnitude larger than NLP (millions of entities vs. tens of thousands of words).

**Sequence model architecture**:
- Event model: synthesizes embeddings from EBF attributes; timestamp encoding for recency/order
- Sequence model: Transformer with multi-head attention pooling to summarize event sequence into query-keyed embeddings (reduces complexity from O(N²) to O(M×N) via multi-head attention pooling where M is tunable)
- The sequence model acts as a "person-level event summarization model" — the attention is keyed by the ad to be ranked, making the user representation target-aware

**Infrastructure adaptations**:
- Jagged tensors: different users have different event history lengths; requires native PyTorch jagged tensor support, kernel-level GPU optimization, and custom Jagged Flash Attention
- Multi-precision quantization and value-based sampling to scale to longer sequences
- Vectorized quantization for richer semantic signals in each event embedding

**Results**: 2–4% improvement in advertiser conversions on select segments.

---

## Key Technical Contributions

### DLRM Parallelization Scheme

DLRM introduced the first systematic treatment of parallelism in recommendation models:

```
Embedding table E_1 ── Device 0
Embedding table E_2 ── Device 1        ← model parallelism
Embedding table E_3 ── Device 2        (memory-bound, large tables)
...

Bottom MLP replicated ── all devices   ← data parallelism
Top MLP replicated    ── all devices     (compute-bound, dense layers)
```

All-to-all communication is required to route embedding lookup results to the correct devices for the interaction layer. This hybrid parallelism became the template for recommendation hardware design across the industry.

### Hard Negative Mining (EBR)

Meta's EBR paper established the practice of staged hard negative mining for retrieval models:
1. Train on in-batch negatives (easy)
2. Run model; collect near-miss negatives (items ranked 101–200, not clicked)
3. Retrain with mixture of easy + hard negatives
4. Repeat

This practice is now standard for production two-tower training across the industry.

### Online-Offline Correlation

Meta operates on a batch training paradigm (multiple-times-daily updates) rather than streaming online training. The 2024 sequence learning work implicitly addresses some staleness by using richer features that capture more recent context at serving time, rather than modifying the training frequency.

---

## Industry Impact

| Innovation | Impact |
|-----------|--------|
| DLRM open-source | Established the standard recommendation model benchmark; influenced hardware design across the industry |
| EBR hard negative mining | Became standard practice for two-tower retrieval training |
| DHEN hierarchical interactions | Demonstrated value of higher-order interactions beyond DLRM's second-order dot products |
| EBF + Sequence learning | Established the "event-based features + Transformer" paradigm as the next-generation replacement for DLRM-era aggregated features |

---

## Strengths and Limitations of Meta's Approach

**Strengths**
- Open publication culture: DLRM, EBR, DHEN all published in detail, enabling industry-wide learning
- Strong hardware co-design: DLRM parallelism scheme maps cleanly to custom AI accelerators
- Vertical integration: model architecture, training infrastructure, and serving hardware are co-designed

**Limitations**
- Batch training paradigm: unlike TikTok's Monolith, Meta does not publish evidence of minutes-level model update latency; this may limit responsiveness to sudden trends
- DLRM's second-order limitation: needed DHEN to address; the transition from DLRM to sequence learning was a multi-year effort
- Scale creates friction: the cost of architecture changes at Meta's scale (billions of users, terabyte embedding tables) means iteration cycles are long

---

## Related Pages

- [[dlrm]] — Meta's canonical production ranking architecture (DLRM)
- [[two-tower-model]] — EBR deployed two-tower retrieval for Facebook Search
- [[embedding-tables]] — Meta's embedding tables are terabyte-scale; DLRM's parallelism scheme is designed around them
- [[feature-crosses]] — DHEN hierarchical interactions represent the state of the art in feature cross learning for ads
- [[multi-task-learning]] — production ranking predicts CTR, CVR, engagement jointly
- [[negative-sampling]] — EBR paper introduced hard negative mining; sampling-bias correction used in retrieval
- [[two-stage-pipeline]] — Meta's full retrieval → ranking → auction pipeline
- [[online-learning]] — Meta uses accelerated batch cycles rather than streaming online training (contrast with TikTok Monolith)
- [[click-through-rate-prediction]] — pCTR is the primary ranking signal feeding the auction
- [[batch-vs-online-training]] — compares Meta's batch approach vs. TikTok's Monolith streaming training
- [[company-pipeline-comparison]] — side-by-side Meta vs. Google vs. TikTok vs. Pinterest pipeline

## Open Questions

- How does Meta's batch training approach (multiple-times-daily) compare to TikTok's minute-level online training for capturing sudden trends?
- What is the optimal sequence length for EBFs, and how does the Jagged Flash Attention approach scale to longer sequences?
- With the transition from DLRM to sequence learning, does the embedding table size change or does the new architecture rely more on content features?
