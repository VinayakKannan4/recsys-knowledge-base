# Online Learning (Incremental Training)

**Summary**: The shift from periodic batch retraining (daily/hourly cycles) to continuous model updates incorporating real-time user feedback — reducing model staleness, improving sensitivity to trending content and evolving user interests, at the cost of significant infrastructure complexity.

**Tags**: #training #online-learning #real-time #infrastructure #concept-drift #tiktok #meta

**Sources**:
- raw/papers/monolith-tiktok/monolith-tiktok-meta.md
- raw/blogs/tiktok/tiktok-secret-sauce-shaped.md
- raw/blogs/meta/sequence-learning-ads-2024.md

**Last Updated**: 2026-04-07

---

## Overview

Standard deep learning training assumes a fixed dataset: sample a mini-batch, compute gradients, update parameters, repeat. In recommendation systems, this batch training paradigm creates a fundamental problem: **model staleness**. The world changes constantly — new content is created, trends emerge, user interests evolve — but a model trained on yesterday's data reflects yesterday's patterns.

For platforms like TikTok, where short-video trends can rise and fall within hours, a model trained with a 24-hour batch cycle is perpetually behind. A video that goes viral at 2pm is invisible to a model trained at midnight. The gap between when user behavior changes and when the model reflects that change is **concept drift latency**.

Online learning (also called incremental or continuous training) addresses this by making the training loop continuous: as users interact with the platform, their actions flow into the training pipeline, which updates model parameters in real time. The model deployed to users reflects events from minutes ago, not the previous day.

TikTok's Monolith system is the most detailed public description of a production online learning architecture for recommendations, demonstrating empirically that more frequent model updates consistently improve recommendation quality.

---

## Key Concepts

### Batch Training vs. Online Training

**Batch training** (traditional):
```
T=0:  Collect data from [T-24h, T-0h]
T=6h: Train model on collected data
T=8h: Validate and deploy model
T=8h–T+16h: Model serves users (reflecting data up to T-0h)
```

Total lag between user action and model reflecting it: up to 40+ hours.

**Online training** (Monolith/streaming approach):
```
T=0:  User takes action (click, watch, skip)
T=0:  Action logged to Kafka queue
T+seconds: Features joined with action log
T+minutes: Training worker consumes example, computes gradients
T+minutes: Parameters pushed to Training PS
T+minutes: Training PS syncs to Serving PS
T+minutes: Live model updated
```

Total lag: minutes rather than hours/days.

### Why Staleness Matters

1. **New content**: fresh videos/ads created after the last training cutoff are invisible to the model's learned parameters (though they can still be scored via content features, their collaborative signal is missing)
2. **Trending content**: items that suddenly become popular get more clicks, but a stale model has no signal about this trend
3. **User interest drift**: a user's interests evolve — someone who was interested in fitness last week may now be following a cooking phase; a daily batch model lags this shift
4. **Session context**: within a single browsing session, a user's implicit intent is clear from their recent interactions; a model trained offline cannot capture real-time within-session patterns

### Concept Drift

Concept drift describes the statistical change in the relationship between inputs and labels over time. In recommendation:
- **Sudden drift**: a major event causes rapid shift in user behavior (e.g., breaking news, a viral meme)
- **Gradual drift**: user interests evolve slowly over weeks/months (seasonal content, life events)
- **Recurring drift**: periodic patterns (morning vs. evening content preferences, weekly cycles)

Batch training handles gradual and recurring drift reasonably well with frequent enough retraining. It handles sudden drift poorly. Online learning is most valuable precisely where sudden drift occurs.

---

## How It Works

### Monolith's Architecture (TikTok)

Monolith's online training system uses a Worker-ParameterServer architecture with a streaming data pipeline:

**Data pipeline**:
```
User actions (clicks, watches, skips)  → [Kafka: action_log]
Model server feature outputs           → [Kafka: feature_log]
                                              ↓
                               [Flink Joiner: join on request ID]
                                              ↓
                               [Kafka: training_examples]
                             ↙                          ↘
              [HDFS for batch training]      [Training Workers: consume stream]
```

The Flink joiner merges action logs with feature logs (which are produced asynchronously by the serving infrastructure), using an in-memory cache for low-latency pairs and a KV-store on disk for pairs with high arrival-time differences.

**Training loop**:
```
Training Worker:
  1. Consume mini-batch from Kafka stream
  2. Look up embeddings from Training Parameter Server (PS)
  3. Forward pass
  4. Backward pass → gradients
  5. Push gradient updates to Training PS

Training PS → (sync every few minutes) → Serving PS → Live model
```

**Collisionless embedding table**: critical for online learning because the vocabulary grows continuously (new videos created every second). Standard hash-based tables would require periodic rebuilds. Monolith's cuckoo hashmap dynamically inserts new IDs with no collisions, with automatic expiry of stale embeddings and frequency filtering of noise IDs.

**Fault tolerance**: online training introduces new failure modes. If a training worker crashes, it loses the in-flight mini-batch and resumes from the Kafka stream. Parameters are not lost (they live on the PS). The paper demonstrates that moderate reliability degradation is an acceptable trade-off for real-time learning.

### Meta's Approach: Faster Batch Cycles

Meta does not use fully online training for its primary DLRM-based ads system. Instead, it pursues **accelerated batch training**:
- Move from daily to multiple-times-daily retraining cycles
- Real-time feature computation (recent clicks, engagement history) fed into the model at serving time even with a batch-trained backbone
- 2024 sequence learning work (EBF + Transformer) adds session-level context that partially compensates for batch staleness

Meta's infrastructure prioritizes reliability and reproducibility over real-time updates. The 2024 sequence learning paper reports 2–4% improvement in advertiser conversions from moving to sequence-based features — this gain comes from richer feature engineering rather than faster training cycles.

### Google / YouTube: Streaming + Periodic Retraining

YouTube uses a hybrid: embedding tables are updated more frequently (or in a streaming fashion for recently active items) while the full model retraining is periodic. The specific architecture is not publicly detailed but the general approach is known from engineering blog posts and conference talks.

---

## Empirical Results from Monolith

The Monolith paper directly measures the effect of sync interval (time between Training PS and Serving PS synchronization) on recommendation quality:

| Sync Interval | Recommendation Quality |
|--------------|----------------------|
| Daily batch | Baseline |
| Hourly | Significant improvement |
| ~10 minutes | Further improvement |
| ~1 minute | Marginal further improvement |

This demonstrates a clear law of diminishing returns: most of the gain from online training comes from moving from daily to hourly updates; further reduction in sync interval provides smaller incremental gains.

---

## Infrastructure Requirements

Online learning is significantly more complex to build and operate than batch training:

| Requirement | Batch Training | Online Training |
|-------------|----------------|-----------------|
| Data pipeline | HDFS batch jobs | Kafka + Flink streaming |
| Embedding storage | Static tables | Dynamic cuckoo hashmap with expiry |
| Training loop | Scheduled jobs | Continuous workers consuming stream |
| Parameter serving | Periodic model swap | Continuous PS sync |
| Fault tolerance | Checkpoint & resume | Stream replay, PS checkpoint |
| Monitoring | Offline metrics (AUC, logloss) | Online metrics + concept drift detection |

---

## Strengths and Limitations

**Strengths**
- **Reduced model staleness**: model reflects user behavior from minutes ago, not yesterday
- **Better trend responsiveness**: trending content and shifting interests are captured quickly
- **Within-session learning**: in fully streaming systems, the model can adapt within a single user session
- **Eliminates cold-start for new content**: new items enter the embedding table immediately and start receiving gradient updates

**Limitations**
- **Engineering complexity**: Kafka, Flink, distributed PS, cuckoo hashmaps — a dramatically more complex stack than batch pipelines
- **Reproducibility**: online trained models are harder to reproduce or debug; the model at time T depends on the exact stream order
- **Catastrophic forgetting**: continuous updates on the most recent data can cause the model to "forget" patterns from historical data that are less frequently seen in current streams
- **Feature store latency**: real-time features (e.g., a user's clickstream from 30 seconds ago) must be available in the feature store before training — requires real-time feature computation infrastructure
- **Amplification of errors**: if a bad recommendation is served, users react negatively; those negative signals enter training; the model learns from them; requires careful feedback loop monitoring

---

## Related Pages

- [[monolith]] — TikTok's production online training system; the most detailed public description of this architecture
- [[two-stage-pipeline]] — online training applies primarily to the ranking stage; retrieval models are typically updated on batch schedules
- [[embedding-tables]] — dynamic embedding tables (Monolith's cuckoo hashmap) are a prerequisite for production online learning
- [[click-through-rate-prediction]] — CTR model staleness is one of the primary motivations for online learning; fresh CTR estimates improve auction efficiency
- [[dlrm]] — Meta's batch-trained DLRM; contrast with Monolith's online training approach

## Open Questions

- Is there a principled way to determine the optimal training frequency as a function of content velocity and user interest drift rate?
- How should online learning systems handle adversarial feedback (e.g., coordinated manipulation of the training stream)?
- Can the Monolith online training approach be adapted for the retrieval stage (two-tower), where ANN index freshness is an additional challenge?
