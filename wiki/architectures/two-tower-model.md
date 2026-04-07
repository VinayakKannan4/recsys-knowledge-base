# Two-Tower Model

**Summary**: A dual-encoder architecture that encodes users and items into separate embedding vectors and scores their affinity via dot product, enabling fast approximate-nearest-neighbor retrieval over billion-item corpora at serving time.

**Tags**: #retrieval #candidate-generation #embeddings #two-tower #google #youtube #scalability

**Sources**:
- raw/papers/youtube-dnn-recommendations/youtube-dnn-recommendations-meta.md
- raw/papers/two-tower-sampling-bias/two-tower-sampling-bias-meta.md
- raw/blogs/other/two-tower-deep-dive-shaped.md

**Last Updated**: 2026-04-07

---

## Overview

The two-tower model (also called dual encoder or bi-encoder) is the dominant architecture for **candidate generation** in large-scale recommendation systems. Its defining property is that user features and item features are encoded by *separate, non-interacting* neural networks (towers), and their affinity is computed as a simple dot product (or cosine similarity) of the resulting vectors.

This separation is what makes the architecture scalable: item embeddings can be pre-computed offline and stored in a vector index. At serving time, only a single forward pass through the user tower is needed, followed by an approximate nearest neighbor (ANN) query (e.g., FAISS, ScaNN) to retrieve the top-K candidates in milliseconds — regardless of whether the item catalog contains millions or billions of entries.

The architecture was formally popularized in Google's 2016 YouTube paper (Covington et al.) and refined for bias correction in Google's 2019 ACM RecSys paper (Yi et al.) on sampling-bias-corrected neural modeling. It has since become standard at Google, Meta, Pinterest, LinkedIn, TikTok, Twitter, and virtually every major platform with a large-scale recommendation problem.

---

## Key Concepts

### User and Item Towers

Each tower is an independent neural network:

```
u = Tower_User(user_features)    ∈ R^d
v = Tower_Item(item_features)    ∈ R^d
```

**User features** can include: user ID (learned embedding), demographics, device, historical interactions (as a bag-of-embeddings or via RNN/Transformer), recent queries, context.

**Item features** can include: item ID, category, content embeddings (text, image via pretrained encoders), popularity statistics.

The towers can use any architecture internally: MLPs, RNNs (LSTM/GRU for user history), Transformers (BERT4Rec, SASRec), or GNNs (PinSage). The only constraint is that the final output is a fixed-dimensional vector.

### Dot Product Scoring

```
score(u, v) = u · v = Σ_k u[k] * v[k]
```

Alternatively, cosine similarity normalizes by vector magnitudes:
```
score(u, v) = (u · v) / (||u|| * ||v||)
```

Dot product is more common in production (it allows ANN libraries like FAISS inner product search).

### Training vs. Serving Asymmetry

**Training**: both towers are trained jointly end-to-end. The loss function (cross-entropy, BPR, InfoNCE contrastive loss) pushes user embeddings toward their positive items and away from negatives.

**Serving**:
1. Pre-compute `v` for every item offline → store in an ANN index
2. At query time: compute `u` (one forward pass, ~milliseconds) → ANN search → return top-K items

This asymmetry is the key scalability insight.

---

## How It Works

### Training with In-Batch Negatives

The most efficient training strategy uses **in-batch negative sampling**: within a mini-batch of (user, positive_item) pairs, all other items in the batch serve as negatives for each user. For a batch of size B:

- B positive (user, item) pairs
- B×(B-1) implicit negatives (all other items in batch)

This is computationally free since item embeddings are already computed. The softmax loss is:

```
L = -Σ_i log [ exp(u_i · v_i) / Σ_j exp(u_i · v_j) ]
```

### Sampling Bias Correction (Yi et al. 2019)

In-batch negatives over-represent popular items (which appear frequently in training data). The 2019 Google paper by Yi et al. introduces a correction: subtract the log-probability of sampling each item:

```
corrected_score(u, v) = u · v - log(p(v))
```

where `p(v)` is the estimated sampling probability (proportional to item frequency). This de-biases the softmax denominator, resulting in a loss that better approximates the true retrieval objective over the full item corpus.

### YouTube Architecture (Covington et al. 2016)

The YouTube DNN paper described an early large-scale two-stage recommendation system:

**Candidate generation** (two-tower-like):
- User tower: watch history (averaged video embeddings), search history, demographic features → MLP → user vector
- Item tower: video ID embedding → item vector
- Trained with cross-entropy on implicit feedback (watch events)
- Served via approximate nearest neighbor

**Ranking** (second stage):
- Takes ~hundreds of candidates from retrieval
- Uses richer features including explicit cross-features (watch time, freshness)
- Outputs a calibrated watch-time prediction

The paper introduced the now-standard two-stage (retrieval → ranking) pipeline and the "example age" feature for correcting temporal bias in training data.

---

## Pipeline Position

**Two-Tower is a retrieval (candidate generation) model.** It is the first stage of the recommendation funnel.

```
[Full Item Corpus: billions of items]
         ↓ (ANN search on pre-computed item embeddings)
[Two-Tower Retrieval: top ~1000 candidates]
         ↓
[Ranking Model: DCN-V2, DeepFM, DLRM]
         ↓
[Auction / Final Ranking: top ~100]
         ↓
[User Feed]
```

The trade-off: two-tower retrieval optimizes **recall** (not missing relevant items) at the cost of **precision** (not distinguishing between retrieved candidates). The ranking stage handles precision.

---

## Industry Usage

| Company | Surface | Notes |
|---------|---------|-------|
| Google / YouTube | YouTube Watch Next, Search | Covington et al. 2016 paper; Yi et al. 2019 for bias correction |
| Meta | Facebook Search, Instagram Reels | EBR (Embedding-Based Retrieval) paper 2020 |
| Pinterest | Organic + Ads retrieval | PinSage uses GNN inside item tower |
| LinkedIn | Job recommendations | Multiple papers on two-tower for job retrieval |
| TikTok | For You feed retrieval | Used as first stage before Monolith-style ranker |
| Twitter/X | Who to follow, timeline | Widely reported in engineering blogs |

---

## Strengths and Limitations

**Strengths**
- **Massive scalability**: ANN search over pre-computed item embeddings is O(log N) not O(N); feasible for billion-item catalogs
- **Low online latency**: single user tower forward pass + ANN query, typically <50ms end-to-end
- **Flexible feature integration**: any feature type (text, image, tabular) can be incorporated within a tower
- **Modular**: user and item towers can be updated independently; item embeddings only need recomputing when items change
- **Cold start**: item tower can embed new items immediately from content features, even without interaction history

**Limitations**
- **No user×item feature interactions**: dot product at the end cannot capture conditional preferences (e.g., user likes brand X but only in category Y). Ranking models handle this
- **Simple scoring function**: dot product may be insufficient for complex multi-objective affinity; interaction-based rankers are strictly more expressive
- **Popularity bias**: in-batch negatives over-represent popular items unless corrected (Yi et al. 2019)
- **Stale item embeddings**: pre-computed embeddings go out of date; dynamic content (live videos, trending items) requires frequent re-indexing

---

## Related Pages

- [[dlrm]] — downstream ranking model; takes two-tower candidates and re-scores with pairwise dot-product interactions
- [[dcn-v2]] — downstream ranking model; uses cross-matrix layers for richer feature interactions on two-tower candidates
- [[deepfm]] — downstream ranking model; FM-based cross interactions on candidates
- [[wide-and-deep]] — downstream ranking model; the original Google production ranker that followed YouTube-style retrieval
- [[monolith]] — TikTok's online-training system that operates on candidates from a retrieval stage
- [[two-tower-vs-late-fusion]] — comparison of two-tower retrieval vs. interaction-based ranking: when each is appropriate and the emerging middle ground

## Open Questions

- How much recall is lost by using dot product vs. a learned interaction function at retrieval time?
- What is the practical trade-off between two-tower model size and ANN index refresh latency?
- Can distillation from a ranker into the two-tower retriever close the recall-precision gap without sacrificing latency?
