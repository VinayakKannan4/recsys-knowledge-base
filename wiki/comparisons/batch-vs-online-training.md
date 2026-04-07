# Batch vs. Online Training: Meta's Approach vs. TikTok's Monolith

**Summary**: Two philosophies for keeping recommendation models fresh: Meta's accelerated batch training (multiple-times-daily retraining cycles with rich features capturing recent context at serving time) vs. TikTok's streaming online training (Monolith — Kafka/Flink pipeline that updates the live model within minutes of user interactions). The choice reflects different content velocities, infrastructure priorities, and tolerance for engineering complexity.

**Tags**: #training #online-learning #batch-training #meta #tiktok #monolith #infrastructure #concept-drift #comparison

**Sources**:
- raw/papers/monolith-tiktok/monolith-tiktok-meta.md
- raw/blogs/tiktok/tiktok-secret-sauce-shaped.md
- raw/blogs/meta/sequence-learning-ads-2024.md
- raw/papers/dlrm/dlrm-meta.md

**Last Updated**: 2026-04-07

---

## The Core Problem Both Approaches Solve

Every recommendation model faces **model staleness**: the model was trained on historical data, but users interact with the platform right now. New content is created, trends emerge, and user interests shift — but a model trained yesterday reflects yesterday's patterns.

The question is not whether to address staleness (everyone does), but *how much staleness is acceptable* given the content type, infrastructure investment, and engineering complexity tolerance.

---

## Overview Comparison

| Dimension | Meta (Batch) | TikTok Monolith (Online) |
|-----------|-------------|--------------------------|
| **Training paradigm** | Batch retraining, multiple times daily | Continuous streaming training |
| **Data source** | HDFS (historical logs) | Kafka queue (real-time event stream) |
| **Model update latency** | Hours (train cycle + validation + deploy) | Minutes (configurable PS sync interval) |
| **Feature freshness at serving** | Recent features computed at serving time; model weights updated on batch schedule | Model weights updated continuously; feature log feeds training pipeline directly |
| **Embedding table type** | Standard TensorFlow/PyTorch static tables (model-parallel across GPUs) | Custom cuckoo hashmap — collisionless, dynamic vocab, with expiry + frequency filter |
| **Infrastructure complexity** | Moderate (standard distributed training) | High (Kafka, Flink, dual PS, cuckoo hashmap) |
| **Fault tolerance** | High (batch jobs are retryable from checkpoint) | Moderate (stream can replay; in-flight gradient updates lost on crash) |
| **Reproducibility** | Full — fixed dataset → deterministic training | Difficult — model state depends on exact stream order |
| **Vocabulary dynamics** | Periodic full table rebuilds | Continuous: new IDs inserted automatically; expired IDs deleted |
| **Catastrophic forgetting risk** | Low — training on large historical windows | Higher — continuous updates can underweight historical patterns |
| **Architecture change support** | Easy — retrain from scratch on any new architecture | Requires batch training phase first; online training mode is for parameter updates only |
| **Primary innovation source** | Model architecture (DLRM → DHEN → Sequence learning) | Training infrastructure (online training + collisionless embeddings) |

---

## Meta's Batch Training Approach

### Architecture Era Context

Meta's training pipeline has evolved alongside its model architecture:

| Era | Model | Training Frequency | Key Staleness Mitigation |
|-----|-------|-------------------|--------------------------|
| 2019–~2022 | DLRM | Multiple times daily | Frequent batch cycles; embedding warm-starting |
| ~2022–2023 | DHEN | Multiple times daily | Same; richer interaction modules, not faster updates |
| 2024–present | Sequence learning (EBF + Transformer) | Multiple times daily | Event-based features capture recent context *at serving time* — staleness partially bypassed by feature design |

Source: `raw/papers/dlrm/dlrm-meta.md`, `raw/blogs/meta/sequence-learning-ads-2024.md`

### How Meta Partially Compensates for Batch Staleness

Rather than moving to streaming training (with its engineering complexity), Meta's 2024 sequence learning work takes a different approach: **make serving-time features carry more recent context**.

Instead of aggregated sparse features ("ads user clicked in the last N days" — computed in batch), Event-Based Features (EBFs) represent individual recent events:
- A sequence of recent ad engagement events, each with timestamps and semantic attributes
- A Transformer sequence model produces target-aware user representations at serving time
- The sequence includes events from *right up to the request* — not yesterday's batch

This means even with a batch-trained model backbone, the user representation at serving time reflects recent behavior. The model weights are stale, but the features are fresh. Result: **2–4% improvement in advertiser conversions on select segments**, from feature design alone — without changing training frequency.

### Meta's Warm-Starting and Infrastructure

- **Embedding warm-starting**: new model training initializes from the previous model's embedding weights, dramatically reducing the training steps needed to converge. Enables ~500B example training cycles without full cold-start.
- **Hybrid model parallelism**: embedding tables are model-parallel across GPUs (each shard on a different device); MLP layers are data-parallel. The embedding lookup → all-to-all communication → interaction computation pipeline is optimized for the Big Basin AI platform.
- **Reliability priority**: Meta's infrastructure emphasizes determinism, reproducibility, and fault isolation. A failed batch training job restarts from checkpoint; a bad model version rolls back. This operational safety comes at the cost of real-time responsiveness.

---

## TikTok's Monolith: Online Training Architecture

### Why Batch Training Failed for TikTok

Before Monolith, TikTok ran on standard batch training. The problems were acute:

1. **Trend insensitivity**: short-video trends emerge and fade within hours. A model trained at midnight has zero signal about a video that goes viral at 2pm. By the time the next batch cycle captures it, the trend may have peaked.
2. **Session context blindness**: a user who watches 5 cooking videos in a row is clearly in a cooking session — but a model trained yesterday on yesterday's data has no way to respond to this within-session signal until the next batch cycle.
3. **New content gap**: millions of new videos are created daily. Videos created after the last training cutoff have no learned collaborative signal — they can only be scored via content features until the model is retrained.
4. **Hash collisions**: at TikTok's vocabulary scale (billions of user and video IDs), standard hash-based embedding tables have significant collision rates, degrading embedding quality for distinct IDs that share the same slot.

Source: `raw/papers/monolith-tiktok/monolith-tiktok-meta.md`

### Monolith's Two Core Innovations

**1. Collisionless Cuckoo Hashmap Embedding Table**

```
Standard approach:
  slot = hash(feature_id) % table_size   → collisions guaranteed at scale

Monolith:
  Two hash tables T₀, T₁ with independent hash functions h₀, h₁
  Insert(id): try h₀(id) in T₀; if occupied, evict and try h₁ in T₁; repeat
              → zero collisions guaranteed
  Frequency filter: IDs appearing < threshold times not inserted (long-tail pruning)
  Expiry timer: inactive IDs automatically deleted (reclaims memory; removes stale embeddings)
```

The cuckoo hashmap enables dynamic vocabulary: new video IDs are inserted immediately as they appear in the event stream, without rebuilding the entire table. This is a prerequisite for online training — you cannot stream train on new IDs if those IDs have no embedding slot.

**2. Worker-ParameterServer Streaming Training**

```
[User action: watch, like, follow, skip]
       ↓
[Kafka: action_log]        [Kafka: feature_log (model server outputs)]
       ↓                          ↓
       └──── [Apache Flink Joiner] ────┘
              (join on request ID; in-memory cache + KV-store for async arrival)
                          ↓
              [Kafka: training_examples]
                    ↙               ↘
     [HDFS: batch storage]    [Online Training Workers]
                                     ↓
                          [Training Parameter Server]
                                     ↓ (sync every ~minutes)
                          [Serving Parameter Server]
                                     ↓
                             [Live Model → Users]
```

The configurable sync interval between Training PS and Serving PS is the primary knob for the latency-accuracy trade-off.

### Empirical Impact of Sync Interval

The Monolith paper directly measured recommendation quality as a function of sync interval:

| Sync Interval | Quality vs. Batch Baseline |
|--------------|---------------------------|
| Daily batch | Baseline (0%) |
| Hourly sync | Significant improvement |
| ~10 minutes | Further material improvement |
| ~1 minute | Marginal additional improvement |

**Key finding**: most of the gain comes from moving off daily batch cycles. The law of diminishing returns sets in after ~10 minutes — capturing within-day trends matters far more than within-minute updates. This implies that *the primary value of online training is daily-trend responsiveness*, not second-by-second adaptation.

---

## The Two-Stage Training Approach (Monolith)

Monolith doesn't abandon batch training entirely — it uses a two-stage approach that reveals the practical limits of pure online training:

**Stage 1: Batch training** — used when:
- Initializing a new model from scratch
- Making architecture changes (adding features, modifying network structure)
- Evaluating the impact of major modifications in a reproducible setting

**Stage 2: Online training** — the continuous production mode:
- Model architecture is fixed; only parameters evolve
- Training workers consume the Kafka stream continuously
- Seamless transition between modes without restarting

**The implication**: online training is not a replacement for batch training. Architecture changes and initial training still require the reproducibility and stability of batch training. Online training extends the lifecycle of a deployed model, keeping its parameters fresh between major architecture updates.

---

## Infrastructure Complexity Comparison

| Component | Meta (Batch) | TikTok Monolith (Online) |
|-----------|-------------|--------------------------|
| Training data pipeline | HDFS batch ETL jobs | Kafka queues + Apache Flink joiner |
| Embedding storage | Standard TF/PyTorch model-parallel tables | Custom cuckoo hashmap library |
| Training orchestration | Scheduled distributed training jobs (PyTorch + Caffe2) | Continuous TensorFlow Worker-PS |
| Model deployment | Periodic model swap (train → validate → deploy) | Continuous PS sync (Training PS → Serving PS) |
| Fault recovery | Checkpoint & restart | Stream replay; gradient loss is acceptable |
| Feature store | Precomputed batch features + real-time lookup at serving | Real-time features flow through Flink joiner into training |
| Monitoring | Offline metrics (AUC, NE) + online A/B metrics | All of the above + drift detection, stream lag monitoring |

---

## Failure Modes

| Failure | Batch Training | Online Training (Monolith) |
|---------|---------------|---------------------------|
| Worker crash | Job fails → restart from HDFS checkpoint | In-flight mini-batch lost; stream resumes automatically |
| Parameter server crash | Job fails → reload from checkpoint | Parameters rolled back to last checkpoint; stream replay to recover |
| Bad training data (e.g., logging bug) | Affects next batch cycle; discovered at validation | Immediately contaminates live model; requires real-time detection |
| Adversarial feedback (manipulation) | Discovered in post-batch analysis | Amplified into live model before detection |
| Vocabulary explosion (new ID flood) | Table size fixed per batch cycle | Frequency filter and expiry timers contain growth |

**Online training's worst failure mode**: a logging bug or adversarial feedback enters the Kafka stream and immediately degrades the live model — before any offline validation catches it. Monolith requires real-time monitoring of model behavior to detect and halt this quickly. Batch training's validation step provides a natural checkpoint that catches this before deployment.

---

## Content Velocity: The Underlying Driver

The fundamental reason TikTok invested in online training and Meta did not is **content velocity** — how quickly new content rises, trends, and falls relative to the model update cycle.

| Platform | Content type | Trend lifecycle | Batch cycle tolerance |
|----------|-------------|-----------------|----------------------|
| TikTok | Short videos (15s–3min) | Hours to days | Low — trends peak before next batch |
| Facebook/Instagram | Mixed (posts, photos, ads) | Days to weeks | High — trends persist across batch cycles |
| YouTube | Long-form video (10–30min) | Days to weeks | High — trending content has longer windows |
| Pinterest | Evergreen content (home decor, recipes) | Weeks to months | Very high — content is not time-sensitive |

TikTok's For You feed operates in an environment where the half-life of a trending video can be measured in hours. A model trained at midnight and deployed at 8am is already 8 hours behind before it serves a single request. The batch cycle latency (train + validate + deploy) makes this worse. Online training reduces the effective lag to minutes — which, for hourly trends, is the difference between surfacing viral content at its peak vs. after it has faded.

Meta's ads system, by contrast, operates on a longer timescale. An advertiser's campaign runs for days or weeks; audience behavior shifts on weekly cycles (not hourly ones). The 2–4% improvement Meta achieved from EBF sequence features came from richer modeling, not faster training.

---

## Which Approach Is Right for a New System?

**Start with batch training if:**
- Content half-life is days or longer (evergreen, seasonal, or campaign-based)
- Engineering team is < 20–30 ML engineers (online training stack requires dedicated infra work)
- Reproducibility and debuggability are high priorities
- Model architecture is actively changing (online training requires stable architecture)

**Invest in online training if:**
- Content half-life is hours (short-form video, breaking news, live events)
- Within-session user behavior is predictive and currently unexploited
- Engineering team can staff dedicated streaming infra work
- Batch training staleness is measurably hurting key metrics in A/B tests

**A practical middle path**: move from daily to multiple-times-daily batch retraining; add real-time feature computation at serving time (compute recent user behavior statistics at request time, even if model weights are batch-trained). This captures most of the benefit of online training (within-day trend responsiveness) at a fraction of the infrastructure cost.

---

## Related Pages

- [[monolith]] — full technical breakdown of TikTok's online training system
- [[online-learning]] — the concept and infrastructure requirements for continuous training
- [[dlrm]] — Meta's batch-trained ranking model
- [[tiktok-bytedance-ranking]] — TikTok's full production pipeline including Monolith
- [[meta-ads-ranking]] — Meta's full pipeline including batch training and sequence features
- [[embedding-tables]] — Monolith's cuckoo hashmap vs. standard static tables
- [[click-through-rate-prediction]] — CTR model staleness is a primary motivation for online training

## Open Questions

- Is there a principled formula for the optimal training frequency as a function of content velocity and user drift rate?
- Can Meta's EBF + Transformer approach (fresh features, batch model) close most of the quality gap with Monolith's fully online approach?
- How does catastrophic forgetting manifest in practice in Monolith's continuous training, and how is it detected and mitigated?
- Could Monolith's online training approach be adapted for the retrieval stage (two-tower), where ANN index freshness is an additional challenge?
