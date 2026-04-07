# Two-Tower Retrieval vs. Late Fusion Ranking: When to Use Each

**Summary**: Two-tower models and late fusion (interaction-based) rankers are not competing architectures — they occupy complementary positions in the recommendation funnel. But the boundary between them is shifting: Pinterest's "Beyond Two Towers" work (2026) and other industry trends show that interaction-based models are being pushed earlier in the funnel as serving infrastructure improves. This page compares the two approaches across efficiency, expressiveness, and deployment context.

**Tags**: #retrieval #ranking #two-tower #late-fusion #feature-interaction #scalability #comparison

**Sources**:
- raw/papers/youtube-dnn-recommendations/youtube-dnn-recommendations-meta.md
- raw/papers/two-tower-sampling-bias/two-tower-sampling-bias-meta.md
- raw/blogs/other/two-tower-deep-dive-shaped.md
- raw/blogs/pinterest/beyond-two-towers-2026.md
- raw/blogs/pinterest/large-scale-learned-retrieval-2025.md
- raw/blogs/other/aman-ai-recsys-architectures.md

**Last Updated**: 2026-04-07

---

## Defining the Terms

**Two-tower model** (also: dual encoder, bi-encoder): user and item features are encoded by *separate, non-interacting* networks. Affinity is computed as a dot product of the two output vectors. Item embeddings are pre-computed and stored in an ANN index; serving requires only one user tower forward pass + ANN search. See [[two-tower-model]].

**Late fusion / interaction-based ranker** (also: cross-attention ranker, interaction model, heavy ranker): user and item features are processed *jointly*, either by concatenating them into a shared representation or by computing attention/cross layers between them. User×item interactions (e.g., `user_category × item_category`) are explicitly modeled. Cannot pre-compute item representations; requires a full forward pass per candidate at serving time. Examples: [[dlrm]], [[dcn-v2]], [[deepfm]], [[din-dien]].

---

## The Fundamental Trade-off

```
Two-Tower                                    Late Fusion
─────────────────────────────────────────────────────────────────────
Item embeddings pre-computed offline      Per-candidate forward pass at serving
ANN search: O(log N) over any corpus      O(K) forward passes (K = candidate count)
No user×item interactions                 Full user×item feature interactions
Score = u · v (simple dot product)        Score = f(user, item) (arbitrary function)
Optimizes RECALL (retrieve broadly)       Optimizes PRECISION (rank accurately)
```

These trade-offs are not incidental — they are structural consequences of the architectural constraint:
- **Two-tower**: separating user and item computation enables pre-computation, enabling ANN retrieval over billions of items, but eliminates the possibility of modeling how a specific user's features interact with a specific item's features.
- **Late fusion**: joining user and item computation enables any interaction, but requires materializing features for every candidate at query time, making it infeasible over large corpora.

---

## Side-by-Side Comparison

| Dimension | Two-Tower | Late Fusion / Interaction Ranker |
|-----------|-----------|----------------------------------|
| **Scoring function** | `u · v` (dot product) | `f(concat(user_feats, item_feats))` |
| **User×item interactions** | None (only aggregate: whole-vector dot product) | Full (cross features, attention, FM-style) |
| **Serving computation** | 1 user forward pass + ANN query | K forward passes (one per candidate) |
| **Serving latency** | ~10ms (user tower + ANN) | ~50–200ms (scales with K and model size) |
| **Item pre-computation** | Yes — embeddings cached in ANN index | No — per-request feature materialization |
| **Max corpus size** | Billions (ANN is sub-linear) | Thousands (linear in K) |
| **Optimization target** | Recall (don't miss relevant items) | Precision (rank retrieved items correctly) |
| **Cold start (new items)** | Good — encode via content features immediately | Dependent on retrieved candidates |
| **User history encoding** | Typically averaged or pooled | Can be target-aware (DIN: attention over history keyed by candidate) |
| **Feature interaction order** | 0 (no cross features) | 2nd+ order (FM/DLRM) or arbitrary (DCN-V2/Transformer) |
| **Canonical deployment** | Candidate generation / Stage 1 | Ranking / Stage 2 |

---

## Why Two-Tower Cannot Replace a Ranker

The limitation of two-tower is structural: the dot product `u · v` can only measure alignment between two fixed vectors. It cannot model conditional preferences:

- A user who likes brand X only when the price is below $50 (user × item × price interaction)
- A user whose interest in cooking content depends on who they've followed recently (user history × item category × recency)
- An ad that performs well for mobile users but poorly for desktop users (user device × item creative format)

These require seeing user and item features *together* during scoring. The dot product at the end of a two-tower model collapses all of this into a single inner product — which is why the industry consistently observes that two-tower retrieval models have lower precision than rankers trained on the same data, even when the retrieved set is identical.

The Google sampling-bias-corrected two-tower paper (Yi et al., 2019 — `raw/papers/two-tower-sampling-bias`) acknowledges this explicitly: the model is optimized for *retrieval recall*, and its performance as a ranker (when applied to a fixed candidate set) is measurably worse than a jointly-trained interaction model.

---

## Why Late Fusion Cannot Replace Retrieval

The limitation of late fusion is computational: scoring `K` candidates requires `K` forward passes. In production:

- A typical ranking model (e.g., DCN-V2 with 10M parameters) takes ~1–5ms per candidate on GPU
- Scoring 1,000 candidates: ~1–5 seconds — unacceptable for a real-time feed
- Scoring 1,000,000 candidates: ~1,000–5,000 seconds — physically impossible

Even with aggressive optimization (batching, quantization, GPU parallelism), the linear scaling in K is insurmountable for corpora larger than ~10,000 items within a 100ms serving budget.

Pinterest's "Beyond Two Towers" work (2026, `raw/blogs/pinterest/beyond-two-towers-2026.md`) demonstrates this concretely: their initial attempt to run a general GPU inference model over 100,000 candidates produced **4,000ms latency** before extensive optimization (feature bundling, multi-stream CUDA, kernel fusion, BF16). After engineering investment, they achieved ~20ms over their candidate set — but this required materializing all O(100K) candidates in GPU memory, not scoring billions.

---

## The Emerging Middle Ground: Lightweight Interaction Models

The strict binary (two-tower retrieval OR heavy interaction ranker) is increasingly challenged by intermediate approaches:

### Pinterest: Beyond Two Towers (2026)
Pinterest replaced their two-tower dot-product scoring at the lightweight ranking stage (LWS) with a GPU-based general model that can use interaction features, achieving ~20ms latency over O(10K–100K) candidates via:
1. Feature bundling (high-value item embeddings registered as GPU buffer — zero network overhead)
2. Business logic inside the model (top-K selection happens on-GPU, reducing D2H transfer)
3. Multi-stream CUDA, Triton kernel fusion, BF16

**Source**: `raw/blogs/pinterest/beyond-two-towers-2026.md`

The result: ~20% reduction in offline model loss at the lightweight ranking stage. The key insight is that GPU memory capacity has grown enough to hold O(100K) candidate features simultaneously, making on-GPU interaction feasible at an intermediate stage.

### DIN / Target-Aware Attention at Retrieval
Some platforms apply target-aware attention (à la DIN — [[din-dien]]) inside the user tower to produce candidate-specific user representations. This partially closes the expressiveness gap: instead of one fixed user vector, the user tower produces a different representation for each candidate (attending over the user's history with the candidate item as the query key). This violates the strict pre-computation assumption but can be made efficient by caching user history encodings and computing attention at serving time.

### Learned Score Distillation
A ranker trained with full interactions can be used to generate soft labels for a two-tower model via knowledge distillation. The two-tower learns to approximate the ranker's predictions — gaining some of the ranker's expressiveness without its latency cost. This is becoming standard practice at platforms that want stronger retrieval without adding a pre-ranking stage.

---

## Where Each Belongs in the Funnel

```
[Full Corpus: billions of items]
          ↓ ← Two-Tower is here: ANN search over pre-computed embeddings
[Stage 1 Retrieval: ~5,000–10,000 candidates]    ← optimize for RECALL
          ↓ ← Lightweight interaction model (emerging, à la Pinterest "Beyond Two Towers")
[Pre-ranking: ~500–1,000 candidates]
          ↓ ← Late fusion / interaction ranker is here
[Heavy Ranking: ~50–200 candidates]              ← optimize for PRECISION
          ↓
[Re-ranking + Auction]
          ↓
[User Feed: ~10–50 items]
```

---

## Training Comparison

| Aspect | Two-Tower | Late Fusion |
|--------|-----------|-------------|
| **Positives** | (user, item) co-interaction events | (user, item) co-interaction events |
| **Negatives** | In-batch negatives (other items in batch); hard negatives | Same items not clicked/engaged; hard negatives |
| **Negative bias** | Popularity bias: popular items over-represented in batch negatives → corrected via `score - log p(item)` (Yi et al.) | Less affected: each negative is an actual impression from a specific user |
| **Training objective** | Recall-oriented: maximize P(correct item | user) across full corpus | Precision-oriented: maximize P(click | user, item) for a fixed candidate set |
| **Label type** | Implicit (any positive interaction) | Implicit or explicit (clicks, conversions, watch time) |
| **Feature cost** | Low — user and item features computed independently | High — user×item feature pairs materialized per training example |

---

## Decision Guide

**Use two-tower (retrieval) when:**
- Corpus size > 100K items — ANN is the only scalable option
- Latency budget < 20ms for the retrieval stage
- Building the first stage of a multi-stage pipeline
- New items enter the catalog frequently — content-feature tower enables immediate scoring
- Interaction features are not yet available or too expensive at retrieval scale

**Use late fusion (interaction ranker) when:**
- Operating on a pre-filtered candidate set of < 10K items
- User×item interaction features are predictive and available
- Precision matters more than recall (ranking a known good set vs. finding the set)
- Target-aware user history (DIN-style attention keyed by the candidate) is important
- Latency budget > 50ms allows per-candidate inference

**Use an intermediate interaction model (à la Pinterest "Beyond Two Towers") when:**
- GPU infrastructure is available for the pre-ranking stage
- 20–100ms latency is acceptable at the intermediate stage
- The expressiveness gain of interaction features at pre-ranking is measurable in A/B tests
- Engineering investment in GPU memory optimization is feasible

---

## Related Pages

- [[two-tower-model]] — full breakdown of the two-tower architecture, training, and serving
- [[dlrm]] — canonical late-fusion ranking model (Meta)
- [[dcn-v2]] — canonical late-fusion ranking model (Google, Pinterest)
- [[din-dien]] — target-aware attention for user history; pushes expressiveness into user tower
- [[two-stage-pipeline]] — the funnel architecture that gives each model its position
- [[negative-sampling]] — how both models handle negative examples differently during training
- [[pinterest-ads-ranking]] — "Beyond Two Towers" work pushing interaction models into pre-ranking
- [[google-youtube-ranking]] — YouTube DNN that defined the two-stage paradigm; Yi et al. bias correction for two-tower
- [[feature-crosses]] — why interaction models outperform two-tower for ranking

## Open Questions

- At what corpus size does a well-optimized interaction model with GPU batching become feasible at the retrieval stage?
- Can knowledge distillation from a ranker into a two-tower model close the recall-precision gap at retrieval time?
- Does adding target-aware attention inside the user tower (DIN-style) capture most of the interaction signal at retrieval, or is full late fusion still required?
