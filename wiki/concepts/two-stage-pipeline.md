# Two-Stage Recommendation Pipeline

**Summary**: The canonical production architecture that splits recommendation into a fast retrieval stage (candidate generation, optimizing recall over millions/billions of items) followed by a slower ranking stage (optimizing precision over hundreds–thousands of candidates) — balancing latency and accuracy requirements that no single model can satisfy simultaneously.

**Tags**: #pipeline #retrieval #ranking #latency #infrastructure #production

**Sources**:
- raw/papers/youtube-dnn-recommendations/youtube-dnn-recommendations-meta.md
- raw/papers/two-tower-sampling-bias/two-tower-sampling-bias-meta.md
- raw/blogs/other/two-tower-deep-dive-shaped.md
- raw/blogs/other/aman-ai-recsys-architectures.md
- raw/blogs/pinterest/ad-ranking-infoq-2024.md
- raw/blogs/pinterest/contextual-relevance-ads-ranking.md

**Last Updated**: 2026-04-07

---

## Overview

A platform with a billion items and a hundred millisecond latency budget cannot use a complex ranking model to score every item for every user request. At 1ms per item, scoring 1 billion items takes ~277 hours. Even at 1µs per item (an unrealistic lower bound for a neural network forward pass), it takes 16 minutes. The two-stage pipeline is the industry-universal solution to this fundamental tension.

The two-stage pipeline was formally described in Google/YouTube's 2016 deep learning recommendations paper and has since become the default architecture at virtually every major recommendation platform: YouTube, Facebook, Pinterest, TikTok, Twitter, LinkedIn, Snap, and others. The paper explicitly named the stages: **candidate generation** (retrieval) and **ranking**, each optimized for different objectives.

More recently, platforms have extended to three or four stages (retrieval → pre-ranking → ranking → re-ranking), adding intermediate stages to progressively apply more expensive models to smaller candidate sets. But the core two-stage logic — fast broad retrieval, slow precise ranking — is universal.

---

## Key Concepts

### The Fundamental Trade-off

Any recommendation model involves a trade-off between:
- **Expressiveness**: models with user×item feature interactions (cross features, attention) are more accurate but require full feature materialization for each user-item pair at query time
- **Scalability**: a model whose user and item computations can be separated allows pre-computation of item representations, enabling fast approximate nearest neighbor search

These two properties are in direct tension:
- The two-tower model is maximally scalable (pre-compute item embeddings) but minimally expressive (only dot product at the end, no user×item crosses)
- DLRM/DCN-V2 are highly expressive (cross features, attention) but require per-candidate computation

The two-stage pipeline resolves this by using the scalable model for a broad initial pass and the expressive model for a focused final pass.

### Recall vs. Precision Objectives

The stages optimize for different metrics:

**Retrieval stage** optimizes **recall**: "Did we include the truly relevant items in the candidate set?" If the ranker never sees a relevant item, it cannot rank it highly. Missing recall at retrieval is unrecoverable.

**Ranking stage** optimizes **precision**: "Among the retrieved candidates, are the truly relevant items ranked highest?" The ranker can distinguish between candidates using richer features.

This is why retrieval models are typically evaluated on recall@K (what fraction of relevant items appear in the top K candidates) and ranking models on NDCG or AUC (how well they order the retrieved set).

---

## How It Works

### Stage 1: Retrieval (Candidate Generation)

**Model**: Two-tower neural network (or collaborative filtering, or embedding-based retrieval)
**Input**: Full item corpus (millions to billions)
**Output**: Top-K candidates (typically 100–10,000)
**Mechanism**: Approximate nearest neighbor (ANN) search over pre-computed item embeddings

```
User request arrives
    ↓
[User Tower: forward pass on user features]    → user embedding u ∈ R^d
                                                         ↓
[Pre-computed item embeddings stored in ANN index]
                                                         ↓
[ANN Search: find top-K items with highest u · v]  → K candidates
```

Item embeddings are computed offline (batch) and refreshed on a schedule (hourly, daily, or triggered by content changes). The ANN search (FAISS, ScaNN, HNSW) runs in milliseconds.

The YouTube DNN paper (2016) described an early version of this: a neural network trained on user watch history to produce a softmax over all videos, approximated via nearest neighbor lookup at serving time.

**Optimization target**: maximize recall — retrieve as many of the truly relevant items as possible within the K-slot budget.

### Stage 2: Ranking

**Model**: DLRM, DCN-V2, DeepFM, Wide & Deep, or similar
**Input**: K candidates from retrieval
**Output**: Ranked list of top-N items (typically 10–100 shown to user)
**Mechanism**: Per-candidate forward pass with full user×item feature materialization

```
K candidates from retrieval
    ↓
[Feature materialization: user features × item features for each candidate]
    ↓
[Ranking model: DCN-V2 / DLRM / DeepFM]
    ↓
[Score: P(click | user, item)]
    ↓
[Sort candidates by score → top-N]
```

The ranking model can use any features — user ID, item ID, cross features between them, user history, context — because it only needs to process K items rather than the entire corpus.

**Optimization target**: maximize precision — given the retrieved candidates, order them so the most relevant appear first.

### Extended Pipelines (3–4 Stages)

Many platforms add intermediate stages:

```
[Full corpus: 1B items]
         ↓ (Two-Tower + ANN)
[Stage 1 retrieval: ~10,000 candidates]
         ↓ (Light ranker: linear model or small NN)
[Pre-ranking: ~1,000 candidates]
         ↓ (Heavy ranker: DLRM / DCN-V2 / Transformer)
[Ranking: ~100 candidates]
         ↓ (Re-ranking: diversity, freshness, business rules)
[Final feed: ~10–50 items]
```

Pinterest explicitly describes this four-stage funnel: candidate retrieval → heavyweight ranking → auction → allocation.

---

## Industry Implementations

### Google / YouTube (2016)

The canonical two-stage system. Stage 1 uses a softmax over all videos approximated by nearest neighbor lookup. Stage 2 uses a DNN that scores each candidate on watch time prediction. Key innovation: treating recommendation as "extreme multiclass classification" for training, then using approximate NN at serving. Also introduced "example age" as a feature to correct for temporal bias in training data (older examples are over-represented).

### Google (Sampling-Bias-Corrected, 2019)

Yi et al. refine two-tower retrieval with bias correction: popular items appear more frequently in training, causing in-batch negatives to over-represent them. Correction: subtract `log p(item)` from the dot product score before softmax, where `p(item)` is estimated from streaming item frequency counts. This produces retrieval that better approximates a uniform-negative baseline.

### Meta / Facebook (EBR, 2020)

Facebook Search adopted embedding-based retrieval (EBR) for the first time in 2020, transitioning from Boolean matching. Key challenge: social graph context (who is the searcher, who are their friends) must be incorporated into the query embedding. Hard negative mining was central: after initial training, run the model, find near-miss examples (items ranked just outside top-K), and use these as harder negatives for re-training.

### Pinterest (2023)

Pinterest's ads pipeline: retrieval → heavyweight ranking → auction → allocation. Retrieval uses two-tower models; ranking uses a multi-model ensemble including DCN-V2-inspired interaction layers, Transformer sequence models, and MaskNet. Multiple objectives are scored simultaneously (click, conversion, relevance). Latency budget: ~200ms end-to-end.

### TikTok / ByteDance (Monolith)

TikTok uses a retrieval stage to generate candidates for the For You feed, with Monolith's online-trained ranker scoring retrieved candidates in real time. The key innovation is in the ranker (online training with Kafka/Flink streaming) rather than the retrieval stage itself.

---

## Latency Budget Example

A rough latency breakdown for a production ads system (approximate):

| Stage | Model | Candidates In | Candidates Out | Latency |
|-------|-------|--------------|----------------|---------|
| Retrieval | Two-tower + ANN | 1B+ | ~5,000 | ~10ms |
| Pre-ranking | Light NN | ~5,000 | ~500 | ~20ms |
| Ranking | DCN-V2 / DLRM | ~500 | ~50 | ~50ms |
| Auction | Rule-based | ~50 | ~10 | <5ms |
| **Total** | | | | **~85ms** |

Numbers vary widely by platform, hardware, and model size. The total end-to-end latency budget for ads systems is typically 100–300ms.

---

## Strengths and Limitations

**Strengths**
- Solves the fundamental scale problem: billions of items, millisecond latency
- Modular: retrieval and ranking teams can iterate independently
- Retrieval provides a "safety net": if recall is high, ranking can focus on ordering
- Each stage can be optimized for its specific objective (recall vs. precision)

**Limitations**
- **Recall ceiling**: errors in retrieval are unrecoverable; items not retrieved cannot be ranked
- **Distribution shift**: the ranker is trained on retrieved candidates, not the full item distribution; poor retrieval can bias the ranker
- **Cascade feedback**: ranking signals (clicks) are only observed for items that passed through retrieval, creating selection bias in training data
- **Complexity**: operating two (or more) models with different training pipelines, feature stores, and serving infrastructure multiplies operational overhead

---

## Related Pages

- [[two-tower-model]] — the dominant retrieval architecture in stage 1
- [[dlrm]] — the dominant ranking architecture in stage 2 (Meta)
- [[dcn-v2]] — alternative ranking architecture in stage 2 (Google, Pinterest)
- [[deepfm]] — commonly used ranking model in stage 2
- [[negative-sampling]] — how retrieval models are trained; in-batch negatives and hard negatives
- [[click-through-rate-prediction]] — the primary signal optimized by the ranking stage
- [[multi-task-learning]] — production rankers simultaneously predict multiple signals (CTR, CVR, engagement)
- [[monolith]] — TikTok's online-trained ranker that operates in stage 2

## Open Questions

- What is the optimal retrieval-to-ranking candidate ratio, and how does it depend on corpus size and query volume?
- Can a single unified model replace the two-stage pipeline for catalogs up to a certain size (e.g., 10M items)?
- How does retrieval recall affect downstream ranking quality, and is there a floor below which ranking performance degrades rapidly?
