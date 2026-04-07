# Pinterest Ads Ranking

**Summary**: Pinterest's ads and organic recommendation stack, evolving from GBDT + logistic regression (2018) through DNN + MTL (2020) to a DCNv2 + Transformer + MaskNet ensemble (2023) with GPU serving, a two-tower learned retrieval system (2025), and a GPU-based "beyond two towers" lightweight ranking replacement (2026) — all while managing substantial online-offline discrepancy challenges.

**Tags**: #pinterest #ads #ranking #retrieval #dcnv2 #multi-task-learning #gpu-serving #production

**Sources**:
- raw/blogs/pinterest/ads-conversion-optimization-evolution-2024.md
- raw/blogs/pinterest/beyond-two-towers-2026.md
- raw/blogs/pinterest/large-scale-learned-retrieval-2025.md
- raw/blogs/pinterest/online-offline-discrepancy-2024.md
- raw/blogs/pinterest/contextual-relevance-ads-ranking.md
- raw/blogs/pinterest/ad-ranking-infoq-2024.md

**Last Updated**: 2026-04-07

---

## Overview

Pinterest operates a two-sided marketplace connecting 500+ million monthly active users with advertisers. The recommendation system must balance three parties: user experience (showing relevant, inspiring content), advertiser performance (driving conversions and ROI), and platform revenue. This multi-objective tension is deeply embedded in Pinterest's technical choices — from multi-task learning that co-trains click, engagement, and conversion objectives, to explicit relevance models that serve as guardrails against pure CTR optimization.

Pinterest's engineering blog is among the most transparent in the industry about their ML systems, documenting the full evolution from 2018 through 2026 across retrieval, ranking, serving infrastructure, and the endemic challenge of online-offline metric discrepancy.

The system serves across multiple surfaces — home feed, search, related pins, and notifications — each with different context signals and candidate generation strategies. Ads and organic content flow through related but distinct pipelines, with the ads pipeline having additional auction and conversion prediction stages.

---

## System Architecture

Pinterest's recommendation funnel comprises multiple stages:

```
[Full ads inventory + organic content: billions of Pins]
         ↓
[Candidate Generation / Retrieval]
  - Legacy heuristic: Pin-Board graphs, user-followed interests
  - Learned retrieval (2025): two-tower + ANN (Manas HNSW system)
  - Multiple candidate generators (20+ in production for homefeed)
  → O(thousands–tens of thousands) candidates
         ↓
[Light-Weight Scoring (LWS) / Pre-ranking]
  - Two-tower dot product (legacy)
  - GPU-based general model inference (2026, "Beyond Two Towers")
  → O(1,000) candidates
         ↓
[Heavy Ranking]
  - DCNv2 + Transformer + MaskNet ensemble (2023)
  - DHEN framework with sequence modeling (current)
  - Multi-task: pCTR, pCVR, pEngagement, pRelevance
  - GPU serving with CUDA graphs + mixed precision
  → O(tens–hundreds) candidates
         ↓
[Auction]
  - Utility = f(pCTR, pCVR, bid, relevance score)
  - Separate relevance model scored in dedicated service
  → Winning ads
         ↓
[Allocation + Feed Assembly]
         ↓
[User Feed / Notification]
```

**Key infrastructure facts**:
- ML platform: MLEnv (unified PyTorch-based framework, launched 2021–2022)
- Feature representation: Unified Feature Representation (flattened feature definitions to reduce train-serve skew)
- Serving: GPU clusters with CUDA graphs and FP16 mixed precision
- ANN system: in-house Manas HNSW-based serving

---

## Model Evolution Timeline

### 2018: GBDT + Logistic Regression
Pinterest's first conversion optimization model was a **hybrid GBDT + logistic regression** system:
- GBDT used as feature translators (embedding/transforming raw features)
- Logistic regression as the prediction head
- Feature hashing trick for ID space generalization across advertisers
- Motivation: performant for data volumes and ML stack of the time; controlled cross-advertiser label impacts

**Limitation**: Not scalable as data volumes grew; GBDT feature extraction became a bottleneck.

### 2020: DNN + MTL + AutoML
Transition to deep learning with two key changes:

**Single DNN model**: replaced the GBDT+LR stack with a unified deep neural network, enabling end-to-end learning.

**Multi-Task Learning**: co-trained multiple objectives simultaneously:
- Clicks, good clicks, checkout, add-to-cart conversions
- Sharing click signal helped conversion prediction (click labels are 100× denser than conversion labels)

**AutoML**: shifted from manual feature engineering to automatic feature interaction learning. Engineers define raw features; AutoML discovers useful crosses. This eliminated hand-crafted feature interactions as a bottleneck.

**By 2020**, Pinterest transitioned to AutoML with MTL — a significant advance in iteration speed.

### 2021–2022: MLEnv Overhaul
Complete rebuild of the ML platform under "MLEnv" (unified ML engine):
- Standardized ML lifecycle across all use cases
- Flattened Unified Feature Representation: eliminated C++ UDF mismatches between training and serving
- Enabled faster model architecture iteration
- Adopted sampling near the model trainer for more efficient training data pipelines

This was necessary technical debt reduction that unblocked the architectural innovations that followed.

### 2023: Feature Interaction Module Evolution

With MLEnv as a stable backbone, Pinterest experimented with and deployed three modern feature interaction modules:

**DCNv2**:
- Cross network with low-dimensional cross matrix for explicit feature interactions
- Better than MLP alone; higher infrastructure cost due to cross network
- Captures both explicit (cross network) and implicit (deep network) interactions

**Transformer**:
- Self-attention applied to feature crossing: maps input to Q/K/V; Q×K captures interaction, weighted into V
- Significant model performance improvement
- High memory usage during training; higher latency

**MaskNet**:
- Instance-guided mask: element-wise products in both embedding and feedforward layers
- Global contextual mask dynamically highlights important features
- MaskBlock = instance-guided mask + feedforward layer + normalization
- High performance; popular in production recommendation systems

**Ensemble architecture (the key innovation)**:
Pinterest found that no single module dominated — each had unique strengths. Rather than choosing, they ensembled:

```
Shared Feature Processing (bottom)
    ↓
[DCNv2 backbone]  [Transformer backbone]
    ↓                    ↓
[Individual training losses and gradients]
    ↓                    ↓
[Empirical score fusion formula]
    ↓
Final prediction
```

Key detail: individual training loss and gradient descent per backbone, not a joint loss — allowing each backbone to learn independently before their predictions are combined. This diversity of learning drives ensemble benefits.

**Cost optimization**: the ensemble was expensive. Pinterest decoupled feature interaction modules from feature processing, using a shared bottom for feature processing while keeping the top architecture separated — maintaining performance while reducing infra cost.

### 2023: DHEN + User Sequence Modeling
Pinterest adopted Meta's DHEN (Deep Hierarchical Ensemble Network) framework as a principled approach to in-model ensembling, combining it with **user sequence modeling**:

**User sequence modeling benefits for conversion prediction**:
- Long lookback windows capture sparse user-advertiser interaction signals
- Direct training on user sequence data learns interest representation from activities
- Temporal information enables learning of seasonal, news-level, and lifetime interest shifts

**Architecture**: DHEN interaction hierarchy + sequence Transformer for user history:

```
User sequence (temporal events)
      ↓ [Sequence Transformer]
User sequence embedding
      ↓
[DHEN: dot product + self-attention + convolution + DCN + MaskNet]
      ↓
Multi-task prediction heads (CTR, CVR, engagement)
```

This combination achieved the best offline and online performance of any architecture in Pinterest's conversion model history.

### 2023: GPU Serving
Complex ensemble models required GPU serving to meet latency requirements:
- **CUDA Graphs**: reduces kernel launch overhead; requires static tensor shapes (solved via zero padding for ragged tensors)
- **Mixed Precision (FP16)**: applied to high-compute modules (feature crossing, ensemble, projection, sequence layers); reduces memory and increases throughput without accuracy loss
- **Result**: GPU serving enabled model capacity that would be infeasible on CPU clusters

### 2025: Learned Retrieval for Homefeed
Pinterest deployed a two-tower embedding-based retrieval system for organic homefeed candidate generation:

**Architecture**:
- User tower: PinnerSage (long-term user representation) + real-time user sequence Transformer (short-term intent) + user profile + context
- Item tower: Pin embeddings
- Training: in-batch negatives with sampling-bias correction (`score = u·v - log P(item in batch)`)
- ANN serving: Manas (in-house HNSW-based system)

**Model version synchronization challenge**: two-tower models are split into separate user and item artifacts deployed to different services. If the ANN index (item embeddings) and the user model are from different training runs, the embedding spaces don't align. Solution: model version metadata attached to each ANN service host; homefeed backend reads version metadata before computing user embeddings.

**Impact**: learned retrieval CG became the top coverage and top-3 save rate candidate generator in homefeed; enabled deprecation of two heuristic-based candidate generators with overall engagement wins.

### 2026: Beyond Two Towers — GPU-Based Lightweight Ranking
Pinterest's most recent published work replaces the two-tower dot product at the lightweight ranking stage with a GPU-based general-purpose model inference engine:

**Motivation**: two-tower dot product cannot use interaction features (how a specific user interacts with a specific item) or target attention. These cross-features require seeing user and item data simultaneously — which the decoupled two-tower design structurally prevents.

**Challenge**: the serving stack was highly optimized for dot products and ANN search, not general GPU inference at O(10K–100K) candidates/request. Naively adding a GPU model would have added 4000ms latency (initial benchmark).

**Key engineering optimizations to reach 20ms**:
1. **Feature bundling**: high-value O(1M) candidates have features embedded directly in the PyTorch model file as registered buffers (on GPU HBM) — zero network overhead for these
2. **Business logic inside the model**: utility calculation, diversity rules, and top-K selection moved inside PyTorch model itself; model outputs O(1K) winners rather than O(100K) raw scores, reducing D2H transfer
3. **Multi-stream CUDA**: overlaps H2D transfers, compute kernels, and D2H transfers using different CUDA streams per worker
4. **Worker alignment**: threads pinned to physical CPU cores to avoid context switching
5. **Kernel fusion via Triton**: fuse Linear + Activation patterns to reduce memory bandwidth pressure
6. **BF16**: brain float 16 for faster math and lower memory footprint
7. **Column-wise retrieval protocol**: retrieval engine returns IDs+Bids only (not full metadata) for O(100K) candidates; metadata fetched lazily for top O(1K) winners only

**Distribution shift issue**: the new GPU inference performs global ranking (true top-K from all candidates) vs. the legacy local ranking (root-leaf architecture: local top-K per shard then aggregate). Global ranking is theoretically superior but changed the candidate distribution, causing unexpected metric shifts during A/B testing. Required significant analysis and tuning.

**Result**: ~20% reduction in offline model loss at the lightweight ranking stage; sets the foundation for next-generation modeling innovations at Pinterest.

---

## Online-Offline Discrepancy

Pinterest has one of the most detailed public treatments of the offline→online metric correlation challenge. Key observations from their 2024 analysis of 15 major conversion model iterations in 2023:

- 10 out of 15 iterations showed consistent directional movement between offline AUC and online CPA (cost per acquisition)
- Only 8 showed statistically significant online movements
- Even with consistent direction, the magnitude correlation was poor: a given AUC improvement could not reliably predict the CPA improvement

**Root causes identified**:

| Cause | Description |
|-------|-------------|
| Metric misalignment | AUC measures ranking quality; CPA involves bidding logic, attribution, and revenue — not directly derivable from AUC |
| Control model contamination | Control model learns from treatment traffic in A/B tests; minimizes anticipated gain |
| Downstream logic dilution | Ensemble scores feed utility calculation; independent optimization of each model misaligns with joint optimization |
| Feature freshness mismatch | Offline backfilling uses fully available labels; online serving uses features computed on a schedule (e.g., 7-day aggregates computed at 3am, stale until then) |
| Diminishing marginal returns | As models improve, expected marginal gains from each iteration shrink below detectable A/B thresholds |

**Infrastructure investments to reduce discrepancy**:
- Unified Feature Representation (prevents C++ UDF train-serve skew)
- Feature Stats Validation checks in training pipelines
- Batch inference capability to replay traffic and compare train vs. serve feature values
- Monitoring for serving-logging discrepancies (features scored vs. features logged may differ due to async logging)

---

## Contextual Relevance Model

Pinterest operates a **dedicated contextual relevance model** alongside engagement prediction:
- Separates relevance from pure click optimization (CTR models can promote clickbait)
- Human-labeled data: 5-point scale, crowdsourced teams, <10% row-level disagreement rate
- Three main relevance surfaces: user personalization, search queries, related pins closeups
- Separate model serving infrastructure optimized for large models and fewer candidates
- Added category-match feature for shopping ads → +4.25% fraction of relevant shopping ads, +1.45% click rate, smoother calibration curve

This explicit relevance model is a guardrail for auction — preventing pure revenue optimization from degrading user experience.

---

## Key Technical Insights

1. **No single interaction module dominates**: DCNv2, Transformer, and MaskNet each have unique strengths; ensembling all three outperforms any single module
2. **User sequence modeling is especially valuable for conversion prediction**: the sparsity and delay of conversion signals is partially compensated by richer user behavioral signals
3. **Model version synchronization is a critical operational challenge for two-tower systems**: mismatched user and item embedding spaces silently degrade retrieval quality
4. **Local vs. global ranking**: legacy root-leaf retrieval architectures create local ranking bias that new global GPU inference exposes; requires careful distribution shift analysis

---

## Related Pages

- [[two-tower-model]] — Pinterest deployed learned retrieval (2025) and is moving beyond it (2026)
- [[dcn-v2]] — DCNv2 is a core component of Pinterest's heavy ranking ensemble
- [[multi-task-learning]] — Pinterest co-trains click, conversion, engagement, and relevance jointly
- [[feature-crosses]] — DCNv2, Transformer, and MaskNet interaction modules all address feature crossing
- [[negative-sampling]] — learned retrieval uses in-batch negatives with sampling-bias correction
- [[two-stage-pipeline]] — Pinterest's full multi-stage funnel (CG → LWS → heavy ranking → auction → allocation)
- [[click-through-rate-prediction]] — pCTR + pCVR + relevance combined in utility function
- [[online-learning]] — Pinterest uses batch training; online-offline discrepancy is a major engineering focus
- [[embedding-tables]] — feature hashing trick (2018), transition to Unified Feature Representation (2021)
- [[two-tower-vs-late-fusion]] — Pinterest's "Beyond Two Towers" work is a key example of the emerging middle ground
- [[company-pipeline-comparison]] — side-by-side Pinterest vs. Meta vs. Google vs. TikTok pipeline

## Open Questions

- Does the "Beyond Two Towers" GPU-based lightweight ranking approach improve recall (by enabling global ranking) or primarily improve precision (via interaction features)?
- What is the optimal way to combine pCTR, pCVR, and relevance scores in the utility function — does this require its own optimization loop?
- How does Pinterest handle the freshness-accuracy trade-off for ANN index refreshes, given that learned retrieval embedding spaces change with retraining?
