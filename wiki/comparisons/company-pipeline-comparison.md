# Company Pipeline Comparison: Meta vs. Pinterest vs. Google vs. TikTok

**Summary**: A side-by-side comparison of the full ads/feed ranking pipelines at Meta, Pinterest, Google/YouTube, and TikTok — covering retrieval, pre-ranking, heavy ranking, training approach, model architecture choices, key innovations, and the distinct engineering priorities that differentiate each company's system.

**Tags**: #meta #pinterest #google #youtube #tiktok #pipeline #comparison #production #ranking #retrieval

**Sources**:
- raw/papers/dlrm/dlrm-meta.md
- raw/papers/facebook-ebr/facebook-ebr-meta.md
- raw/blogs/meta/sequence-learning-ads-2024.md
- raw/papers/youtube-dnn-recommendations/youtube-dnn-recommendations-meta.md
- raw/papers/two-tower-sampling-bias/two-tower-sampling-bias-meta.md
- raw/papers/dcn-v2/dcn-v2-meta.md
- raw/papers/monolith-tiktok/monolith-tiktok-meta.md
- raw/blogs/tiktok/tiktok-secret-sauce-shaped.md
- raw/blogs/pinterest/ads-conversion-optimization-evolution-2024.md
- raw/blogs/pinterest/beyond-two-towers-2026.md
- raw/blogs/pinterest/large-scale-learned-retrieval-2025.md
- raw/blogs/pinterest/online-offline-discrepancy-2024.md
- raw/blogs/other/aman-ai-recsys-architectures.md
- raw/blogs/other/rise-of-dcn-ml-frontiers.md

**Last Updated**: 2026-04-07

---

## Quick Reference: Pipeline Summary

| Stage | Meta | Google / YouTube | TikTok | Pinterest |
|-------|------|-----------------|--------|-----------|
| **Retrieval** | EBR (two-tower + FAISS); hard negative mining | Two-tower + ANN (ScaNN); sampling-bias correction | Multi-strategy: two-tower + collaborative filtering + trending | Two-tower + Manas (HNSW); PinnerSage + real-time Transformer |
| **Pre-ranking** | Lightweight model; O(hundreds) → O(tens) | Optional lightweight model | Implicit (internal filtering) | Light-Weight Scoring (LWS): legacy two-tower dot product → "Beyond Two Towers" GPU model (2026) |
| **Heavy Ranking** | DLRM (2019) → DHEN (2022) → EBF + Transformer (2024) | DCN-V2 / DCN-Mix across multiple LTR systems | Monolith: online-trained model with cuckoo hashmap embeddings | DCNv2 + Transformer + MaskNet ensemble; DHEN + user sequence Transformer |
| **Training approach** | Batch (multiple times daily); EBF adds real-time context at serving | Batch (periodic) + streaming embedding updates for recent items | **Online streaming** (Kafka + Flink + Worker-PS); minutes latency | Batch (periodic); explicit online-offline discrepancy tracking |
| **Multi-task objectives** | pCTR, pCVR, pEngagement | Watch time (YouTube); app installs (Play); conversion (Ads) | Completion rate, likes, follows, shares, comments, not-interested | pCTR, pCVR, pEngagement, pRelevance (separate model) |
| **Embedding infrastructure** | Terabyte-scale model-parallel tables; PyTorch + Caffe2 | Variable embedding sizes per feature vocabulary | Collisionless cuckoo hashmap (dynamic vocab, expiry, frequency filter) | Unified Feature Representation (MLEnv); GPU serving with CUDA graphs |
| **Key differentiator** | DLRM open-source + sequence learning (EBF + Transformer) | DCN-V2 cross-product deployment; sampling-bias correction | Real-time online training (minutes update latency) | Ensemble interaction modules; "Beyond Two Towers" GPU pre-ranking |
| **Publication culture** | High — DLRM, EBR, DHEN, sequence learning all published | High — Wide & Deep, DCN, DCN-V2, sampling-bias correction | Medium — Monolith published; serving details sparse | Very high — multiple detailed engineering blog posts per year |

---

## Full Pipeline Diagrams

### Meta (Facebook, Instagram, Ads)

```
[Full ads inventory: billions of ads]
         ↓
[Retrieval: Embedding-Based Retrieval (EBR)]
  - Two-tower model: user embedding + social graph context
  - ANN via FAISS over pre-indexed ad embeddings
  - Training: in-batch negatives + staged hard negative mining
  - Hard negatives: items ranked 101–200, not clicked
  → O(thousands) candidates

         ↓
[Light Ranking / Pre-ranking]
  - Lightweight model; scores and filters
  → O(hundreds) candidates

         ↓
[Heavy Ranking]
  - Era 1 (2019–~2022): DLRM
      Sparse → Embedding tables (model-parallel, terabyte-scale)
      Dense → Bottom MLP → projected dense
      All embeddings → Pairwise dot products (2nd-order)
      Dot product outputs → Top MLP → pCTR
  - Era 2 (~2022–2023): DHEN
      Replaces dot-product interaction with hierarchical ensemble:
      self-attention + convolution + DCN-style crossing + dot products
      +0.27% NE improvement over DLRM
  - Era 3 (2024–present): Sequence Learning (EBF + Transformer)
      Event-Based Features replace aggregated sparse features
      Transformer with multi-head attention pooling over event sequences
      Target-aware: user representation keyed by the ad being ranked
      +2–4% advertiser conversion improvement
  → O(tens) candidates

         ↓
[Auction]
  expected_value = pCTR × bid   (second-price or VCG)
  Ad quality incorporated
  → Winning ads

         ↓
[User Feed / Ad Slot]
```

**Training**: Batch, multiple times daily. Warm-starting from previous model weights. Custom hybrid parallelism (embedding model-parallel, MLP data-parallel). Big Basin AI platform.

---

### Google / YouTube

```
[Video corpus: hundreds of millions of videos]
         ↓
[Stage 1: Candidate Generation (Two-Tower)]
  - User tower: watch history (averaged video embeddings) + search tokens
               + demographics + geographic info + device
  - Item tower: video ID embedding
  - Sampling-bias correction (Yi et al. 2019):
      score(u,v) = u·v − log p(v)
      p(v) estimated from streaming item frequency counts
  - Serving: ANN search (ScaNN / custom) over pre-computed video embeddings
  - "Example age" feature: corrects temporal bias in training data
  → O(hundreds) candidates

         ↓
[Stage 2: Ranking]
  - Features: per-candidate features not available at retrieval scale
              (impressions, engagement history with similar videos, freshness)
  - Model: Wide & Deep (2016 original) → DCN (2017) → DCN-V2 (2020)
      DCN-V2 cross layer: x_{l+1} = x_0 ⊙ (W_l x_l + b_l) + x_l
      W_l ∈ R^(d×d) — matrix cross (vs. DCN's vector cross)
      Stacked or parallel structure; 4+ cross layers beat 2 (unlike original DCN)
      DCN-Mix: low-rank factorization + MoE for production-scale efficiency
  - Objective (YouTube): predicted watch time via weighted logistic regression
      positive examples weighted by watch time (not uniform weighting)
      aligns model with user satisfaction, not just click behavior
  → O(tens) ranked videos

         ↓
[Presentation Layer]
  - A/B testing: thumbnails, display positions, UI
  - Final ordered list shown to user
```

**Training**: Batch (periodic). Wide & Deep used warm-starting from previous model's embeddings. DCN-V2 deployed across many Google systems simultaneously (Search, YouTube, Ads, Play Store). Variable embedding sizes per feature vocabulary.

**GDCN (2023)**: adds information gate to DCN-V2 cross layers: `x_{l+1} = x_0 ⊙ (W_l x_l + b_l) ⊙ G_l(x_l) + x_l`. Current Criteo SOTA at 0.8161 AUC.

---

### TikTok / ByteDance (For You Feed)

```
[Content corpus: billions of short videos, created continuously]
         ↓
[Candidate Generation / Retrieval]
  - Multi-strategy: user interest embedding retrieval, collaborative filtering,
                    trending content boosting, creator-based retrieval
  - Two-tower style models for some candidate generators
  - ANN search over pre-indexed video embeddings
  → O(thousands) candidates

         ↓
[Monolith Ranker: online-trained model]
  - Embedding table: custom cuckoo hashmap
      Collisionless: two hash tables T₀, T₁ with independent hash functions
      Frequency filter: IDs below threshold not inserted (long-tail pruning)
      Expiry timer: inactive IDs deleted automatically (memory management)
      Dynamic: new video IDs inserted immediately without table rebuild
  - Neural network on top: MLP / Transformer (architecture details not published)
  - Multi-task prediction:
      completion rate (primary signal: did user watch the full video?)
      likes, follows, shares, comments
      explicit "not interested" negative signal
  - Training: CONTINUOUS ONLINE LEARNING
      [Action log: Kafka] + [Feature log: Kafka]
               ↓
      [Flink Joiner: join on request ID]
      (in-memory cache for fast pairs; KV-store for high-latency arrivals)
               ↓
      [Kafka: training_examples]
      ↙                          ↘
[HDFS: batch training]   [Online Training Workers: consume stream]
                                  ↓
                         [Training Parameter Server]
                                  ↓ (sync every ~minutes)
                         [Serving Parameter Server]
                                  ↓
                          [Live Model: minutes latency]
  → O(tens–hundreds) ranked videos

         ↓
[Re-ranking / Diversity Enforcement]
  - Duration diversity (short/long mix)
  - Interest diversification
  - Creator diversity
  → Final ordered For You feed
```

**Training**: Online streaming (Monolith). Batch training used only for architecture changes and initial model setup. Sync interval between Training PS and Serving PS: empirically optimal at ~10 minutes (diminishing returns beyond that). Model updates reflect user behavior from minutes ago.

---

### Pinterest (Homefeed + Ads)

```
[Full Pins inventory + Ads: billions of Pins]
         ↓
[Candidate Generation / Retrieval]
  - 20+ candidate generators for homefeed (mix of heuristic + learned)
  - Learned retrieval (2025):
      User tower: PinnerSage (long-term interest embedding)
                + real-time user sequence Transformer (short-term intent)
                + user profile + context
      Item tower: Pin embeddings
      Training: in-batch negatives + sampling-bias correction (score − log p(item))
      ANN: Manas (in-house HNSW-based serving system)
      Model version sync: metadata attached to ANN index to prevent embedding space mismatch
  → O(thousands–tens of thousands) candidates

         ↓
[Light-Weight Scoring (LWS) / Pre-ranking]
  - Legacy: two-tower dot product (ANN over pre-computed embeddings)
  - 2026 "Beyond Two Towers": GPU-based general model inference
      Replaces dot product with interaction-capable model
      Engineered to 20ms latency over O(10K–100K) candidates via:
        Feature bundling (O(1M) high-value item embeddings on GPU HBM)
        Business logic inside model (top-K selection on-GPU, reduces D2H transfer)
        Multi-stream CUDA (overlaps H2D, compute, D2H transfers)
        Kernel fusion via Triton (Linear + Activation patterns)
        BF16 arithmetic
      Result: ~20% reduction in offline model loss at LWS stage
  → O(1,000) candidates

         ↓
[Heavy Ranking]
  - 2023 ensemble: DCNv2 + Transformer + MaskNet
      Individual training losses per backbone (not joint loss)
      Empirical score fusion formula
      Shared bottom feature processing (cost optimization)
  - Current: DHEN framework + user sequence Transformer
      Long lookback windows capture sparse conversion signals
      Sequence Transformer encodes temporal user history
      DHEN: dot products + self-attention + convolution + cross + MaskNet hierarchy
  - Multi-task prediction: pCTR, pCVR, pEngagement, pRelevance
  - GPU serving: CUDA Graphs + FP16 mixed precision
  → O(tens–hundreds) candidates

         ↓
[Auction]
  utility = f(pCTR, pCVR, bid, relevance_score)
  Separate contextual relevance model (human-labeled 5-point scale)
  → Winning ads

         ↓
[Allocation + Feed Assembly]
  → User feed / notifications
```

**Training**: Batch (periodic). MLEnv platform (unified PyTorch). Unified Feature Representation eliminates C++ UDF train-serve skew. Detailed tracking of online-offline discrepancy (10/15 iterations had consistent directional movement, but magnitude correlation was poor).

---

## Architectural Choice Breakdown

### Retrieval Architecture

| Company | Retrieval Model | ANN System | Training Distinguisher |
|---------|----------------|------------|------------------------|
| Meta | Two-tower + social graph features | FAISS | Hard negative mining (staged: easy → near-miss) |
| Google | Two-tower + YouTube watch history | ScaNN / custom | Sampling-bias correction (`score − log p(item)`) |
| TikTok | Multi-strategy (two-tower + CF + trending) | Internal ANN | Batch-trained (online training is for ranking only) |
| Pinterest | Two-tower (PinnerSage + real-time Transformer) | Manas (HNSW) | Sampling-bias correction + model version sync |

### Heavy Ranking Model

| Company | Architecture | Interaction Type | Key Innovation |
|---------|-------------|-----------------|----------------|
| Meta | DLRM → DHEN → EBF+Transformer | Dot products → hierarchical ensemble → target-aware attention | Sequence learning on event-based features |
| Google | DCN-V2 / DCN-Mix | Matrix cross layers (bit-wise interactions) | Low-rank factorization + MoE for production scale |
| TikTok | Monolith (MLP/Transformer on cuckoo embeddings) | Not published | Collisionless embeddings + online training |
| Pinterest | DCNv2 + Transformer + MaskNet ensemble | Three interaction paradigms ensembled | No single module dominates; ensemble outperforms any individual |

### Training Paradigm

| Company | Training Mode | Model Update Latency | Staleness Mitigation |
|---------|-------------|---------------------|----------------------|
| Meta | Batch (multiple times daily) | Hours | EBF sequence features carry recent context to serving time |
| Google | Batch (periodic) | Hours | Streaming embedding updates for recently active items (partial) |
| TikTok | Online streaming (Kafka + Flink) | **Minutes** | Continuous parameter updates; collisionless embeddings handle new IDs |
| Pinterest | Batch (periodic) | Hours | Real-time features at serving; explicit discrepancy tracking |

### Multi-Task Objective Design

| Company | Primary Signal | Secondary Signals | Notes |
|---------|--------------|------------------|-------|
| Meta | pCTR → Auction | pCVR, pEngagement | Sequence learning improves CVR specifically |
| Google (YouTube) | Watch time (weighted logistic regression) | pCTR (auxiliary) | Watch time chosen over CTR to fight clickbait |
| TikTok | Completion rate | Likes, follows, shares, comments, not-interested | Completion rate is a clean binary signal; short videos make it practical |
| Pinterest | pCTR + pCVR | pEngagement, pRelevance | Separate relevance model as auction guardrail against pure CTR optimization |

---

## Key Technical Differentiators

### Meta: Hardware-Model Co-Design
Meta's DLRM was designed simultaneously as a production architecture and a hardware benchmark. The hybrid parallelism scheme (embedding model-parallel, MLP data-parallel) was co-designed with the Big Basin AI platform. This vertical integration — from model architecture to serving chip — means Meta's systems can be extremely efficient on their own hardware while being harder to port elsewhere. The 2024 EBF sequence work required new infrastructure (jagged tensors, Jagged Flash Attention, vectorized quantization) that was co-developed with model changes.

**Source**: `raw/papers/dlrm/dlrm-meta.md`, `raw/blogs/meta/sequence-learning-ads-2024.md`

### Google: Cross-Product Architecture Deployment
DCN-V2 was deployed across *many* Google learning-to-rank systems simultaneously (Search, YouTube, Ads, Play Store). This breadth of deployment makes DCN-V2 among the most production-validated ranking architectures in existence — not just for one surface, but across diverse query types, content types, and user populations. The sampling-bias correction (Yi et al. 2019) was similarly deployed across all Google two-tower retrieval systems. Google's publication strategy documents these cross-system deployments explicitly.

**Source**: `raw/papers/dcn-v2/dcn-v2-meta.md`, `raw/papers/two-tower-sampling-bias/two-tower-sampling-bias-meta.md`

### TikTok: Treating Infrastructure as Competitive Moat
TikTok's Monolith paper is notable for what it does *not* describe: the actual ranking model architecture (which MLP structure, Transformer variant, attention mechanism). What Monolith details is the infrastructure — the cuckoo hashmap, the Kafka/Flink pipeline, the Worker-PS training architecture. ByteDance treats the online training infrastructure itself as the competitive differentiator, not the model architecture. The existence of **BytePlus Recommend** (Monolith externalized as a commercial product) confirms this view — the infrastructure is the product.

**Source**: `raw/papers/monolith-tiktok/monolith-tiktok-meta.md`, `raw/blogs/tiktok/tiktok-secret-sauce-shaped.md`

### Pinterest: Engineering Honesty and Iterative Refinement
Pinterest publishes more detail about their *failures and trade-offs* than any other company in this group. Their 2024 analysis of online-offline discrepancy across 15 production model iterations (only 10 showed consistent directional movement; only 8 were statistically significant online) is a rare public treatment of the gap between offline AUC and online CPA. Their "Beyond Two Towers" work documents the full engineering journey from 4000ms to 20ms latency in granular detail. This transparency makes Pinterest's publications uniquely valuable for practitioners.

**Source**: `raw/blogs/pinterest/online-offline-discrepancy-2024.md`, `raw/blogs/pinterest/beyond-two-towers-2026.md`

---

## Scale Context

| Company | Monthly Active Users | Content Corpus Size | Key Serving Constraint |
|---------|---------------------|--------------------|-----------------------|
| Meta | 3B+ | Billions of ads; dynamic daily creation | QPS × latency at 3B user scale |
| Google / YouTube | YouTube: 2B+ | Hundreds of millions of videos | 8.5B Search queries/day; multi-surface |
| TikTok | 1.5B+ | Billions of short videos created daily | Short-video velocity; hours-scale trend lifecycle |
| Pinterest | 500M+ | Billions of Pins; mostly evergreen | Multi-surface (homefeed, search, related); ads + organic |

---

## Shared Patterns Across All Four

Despite their differences, all four pipelines share a core set of patterns:

1. **Two-stage (or multi-stage) funnel**: retrieval → ranking → auction is universal. Pinterest extends to 4 stages. All companies separate recall-optimized retrieval from precision-optimized ranking.

2. **Two-tower retrieval**: all four use some form of two-tower + ANN for candidate generation. The variation is in what goes inside each tower and how training negatives are handled.

3. **Embedding tables as the dominant infrastructure challenge**: billions of user/item IDs create terabyte-scale embedding tables that cannot fit in a single GPU. All four have developed approaches to table sharding, parallelism, or dynamic management.

4. **Multi-task prediction**: all four simultaneously predict multiple signals (CTR, CVR, engagement, watch time) rather than a single objective. The signals are weighted in downstream auction or ranking logic.

5. **Serving latency as a hard constraint**: all four operate under 100–300ms total end-to-end budgets. This constraint — not model accuracy — drives many architectural decisions (number of stages, model size, feature computation cost).

---

## Evolution Trajectory

| Company | Current Direction | Next Likely Step |
|---------|-----------------|-----------------|
| Meta | Sequence learning on event-based features; Transformer over user history | Longer sequences, richer event semantics, possibly online training for embedding freshness |
| Google | GDCN gated cross layers; DCN-Mix MoE at scale | More MoE specialists; possible Transformer-based feature interaction |
| TikTok | Online training operational; architectural details undisclosed | Possibly online training for retrieval stage (not just ranking) |
| Pinterest | GPU pre-ranking replacing two-tower dot product; Beyond Two Towers | Richer interaction features at pre-ranking stage; possibly reducing to 3 stages |

---

## Related Pages

- [[meta-ads-ranking]] — full Meta pipeline details
- [[google-youtube-ranking]] — full Google/YouTube pipeline details
- [[tiktok-bytedance-ranking]] — full TikTok pipeline details
- [[pinterest-ads-ranking]] — full Pinterest pipeline details
- [[two-stage-pipeline]] — the canonical funnel structure shared by all four
- [[two-tower-model]] — the retrieval architecture used across all four at Stage 1
- [[dlrm]] — Meta's canonical ranking architecture
- [[dcn-v2]] — Google's (and Pinterest's) canonical ranking architecture
- [[monolith]] — TikTok's online training infrastructure
- [[multi-task-learning]] — all four companies use multi-task prediction
- [[online-learning]] — TikTok's streaming training vs. others' batch approaches
- [[negative-sampling]] — Meta's hard negative mining vs. Google's sampling-bias correction

## Open Questions

- As GPU memory grows, will the "Beyond Two Towers" pattern (interaction models at pre-ranking) become universal, collapsing the pre-ranking and heavy ranking stages?
- Do all four companies converge on Transformer-based sequence modeling as the primary ranking signal, or do architectural differences persist based on content type?
- How does TikTok's decision not to detail their ranking model architecture (while fully detailing their infrastructure) reflect on where the actual competitive moat lies?
