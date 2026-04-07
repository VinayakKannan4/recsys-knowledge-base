# Multi-Task Learning

**Summary**: Training a single model to predict multiple objectives simultaneously (click, conversion, engagement, watch time, quality), sharing representations across tasks to improve generalization and reduce serving cost — critical for production recommendation systems that optimize for more than just clicks.

**Tags**: #multi-task-learning #mtl #mmoe #ple #ranking #objectives #google #youtube

**Sources**:
- raw/papers/mmoe-multi-task/mmoe-multi-task-meta.md
- raw/blogs/other/aman-ai-recsys-architectures.md
- raw/blogs/pinterest/ad-ranking-infoq-2024.md
- raw/blogs/pinterest/contextual-relevance-ads-ranking.md

**Last Updated**: 2026-04-07

---

## Overview

Production recommendation systems never optimize for a single signal. A naive CTR-maximizing model promotes clickbait. A watch-time-only model might recommend addictive but low-quality content. A pure conversion model ignores the cost to user experience of irrelevant ads. Real platforms must simultaneously optimize for clicks, conversions, engagement depth, content quality, and business revenue — all at once.

Multi-task learning (MTL) addresses this by training a single model with multiple prediction heads, one per objective. Shared layers learn representations that are useful across tasks, providing implicit regularization: tasks with sparse labels (e.g., post-click conversions) benefit from the signal in tasks with abundant labels (e.g., clicks). A single shared model also costs less to serve than running K separate models.

The central challenge of MTL is **negative transfer**: when two tasks have conflicting gradients, sharing representations can hurt both relative to task-specific models. The evolution from shared-bottom to MMoE to PLE is driven by progressively better solutions to this challenge.

---

## Key Concepts

### Why Production Recsys Needs MTL

1. **Sparse labels for important objectives**: conversions (purchases, sign-ups) are rare — often <1% of clicks. Training a standalone conversion model on this data leads to high variance. Sharing with a click-prediction task provides a rich auxiliary signal.

2. **Multi-objective ranking function**: production platforms combine multiple scores into a final ranking score:
   ```
   rank_score = α·pCTR + β·pCVR + γ·pEngagement + δ·pQuality
   ```
   All of these need to be predicted per candidate. Running separate models for each would multiply serving cost and feature lookup overhead.

3. **Cross-objective regularization**: a model that knows both "did the user click?" and "did the user convert after clicking?" learns richer user intent representations than a model trained on either alone.

4. **Entire space MTL (ESMM from Alibaba)**: CVR is typically modeled as `P(conversion | click)`, trained only on clicked items. This introduces selection bias: the training population (clicked items) differs from the scoring population (all retrieved items). ESMM instead decomposes: `P(CTR×CVR) = P(CTR) × P(CVR|CTR)` and trains both jointly on all impressions.

### Tasks Typically Predicted Jointly

| Task | Signal | Sparsity |
|------|--------|---------|
| CTR | Click | Dense (~1–10% of impressions) |
| CVR | Post-click conversion | Sparse (~0.1–2% of clicks) |
| Engagement | Long click / video completion / save | Medium |
| Dwell time | Time spent with content | Continuous |
| Dislike / hide | Negative feedback | Rare |
| Quality / relevance | Human-labeled or proxy | Periodic |

---

## How It Works: MTL Architectures

### Shared Bottom

The simplest MTL architecture: all tasks share a common set of MLP layers (the "bottom"), then branch into task-specific towers (the "heads"):

```
Input features
     ↓
[Shared Bottom: MLP layers]
    ↙     ↓     ↘
[Head 1] [Head 2] [Head 3]    ← task-specific MLPs
  CTR     CVR    Watch time
```

**Problem**: the shared bottom must serve all tasks equally. If tasks have conflicting gradient directions (e.g., CTR and CVR can be anti-correlated for certain ad types), the shared layers receive contradictory update signals — **negative transfer**. Tasks that are dissimilar in nature (low label correlation) hurt each other more than if trained separately.

### MMoE: Multi-gate Mixture of Experts (Google, 2018)

Introduced by Ma et al. at Google (KDD 2018), MMoE replaces the single shared bottom with a set of **expert networks** (E₁, E₂, ..., E_K) and per-task **gating networks** (G₁, G₂, ..., G_T):

```
Input features
     ↓
[Expert 1] [Expert 2] ... [Expert K]    ← K shared expert MLPs

Gate_1(input) → softmax weights over K experts
Gate_2(input) → softmax weights over K experts
...

Task 1 input = Σ_k Gate_1(x)[k] · Expert_k(x)
Task 2 input = Σ_k Gate_2(x)[k] · Expert_k(x)
    ↓              ↓
[Task 1 Tower]  [Task 2 Tower]
```

**Key formulation**:
```
f_i(x) = Σ_{k=1}^{K} g_i^k(x) · e_k(x)

where  g_i^k(x) = softmax(W_{g_i} x)[k]   (gate for task i, expert k)
       e_k(x)   = ReLU(W_{e_k} x)          (expert k's output)
```

**Why this works**: each task can choose which combination of experts to use via its gate. Tasks with conflicting objectives learn to emphasize different experts. The experts can specialize: some may learn CTR-relevant features, others CVR-relevant patterns. Tasks with similar gradients share the same experts naturally.

MMoE was deployed at YouTube and demonstrated significant improvements in multiple engagement metrics simultaneously.

### PLE: Progressive Layered Extraction (Tencent, 2020)

MMoE uses fully shared experts, meaning even the "private" information for each task flows through shared components. PLE (introduced by Tencent) separates task-specific and shared experts explicitly:

```
Layer 1:
  [Task 1 Experts: E1_1, E1_2]   (private to task 1)
  [Shared Experts: ES_1, ES_2]    (shared across tasks)
  [Task 2 Experts: E2_1, E2_2]   (private to task 2)

Extraction Network for Task 1 at Layer 1:
  gate_1^(1) over [task 1 experts + shared experts]
  → output h_1^(1)

Extraction Network for Task 2 at Layer 1:
  gate_2^(1) over [task 2 experts + shared experts]
  → output h_2^(1)

Layer 2:
  [Task 1 Experts on h_1^(1)]
  [Shared Experts on h_1^(1) + h_2^(1)]
  [Task 2 Experts on h_2^(1)]
  ...
```

PLE is "progressive" because the layered extraction allows increasingly task-specific representations to build up as depth increases. This further reduces interference between dissimilar tasks. Tencent reported consistent improvements over MMoE on video recommendation tasks with low task correlation.

### CGC vs. PLE

PLE includes both a **Customized Gate Control (CGC)** (single-layer PLE) and the multi-layer progressive variant. CGC alone — which is just the first layer of PLE with separate experts — often provides most of the benefit over MMoE at lower cost.

### ESMM: Entire Space Multi-Task Model (Alibaba, 2018)

ESMM addresses selection bias in CVR prediction:

```
P(impression → conversion) = P(click|impression) × P(conversion|click)
         ↑ CTCVR                   ↑ CTR                   ↑ CVR
```

Both towers (CTR and CVR) share an embedding layer and are trained jointly:
- CTR tower trained on all impressions with click/no-click labels
- CVR tower trained on all impressions with conversion/no-conversion labels (including non-clicked impressions as negatives)

This eliminates the sample selection bias inherent in training CVR only on clicked examples.

---

## Industry Usage

| Company | System | Approach | Tasks |
|---------|--------|----------|-------|
| Google / YouTube | Recommendations | MMoE | Watch time, engagement, satisfaction |
| Tencent | Video recommendations | PLE | CTR, VTR (video completion), sharing |
| Alibaba | Display ads | ESMM | CTR, CVR (entire space) |
| Pinterest | Ads ranking | Multi-head ensemble | CTR, CVR, relevance, engagement |
| Meta | News Feed, Ads | Multi-task DLRM-based | CTR, CVR, video completion, emoji reactions |
| TikTok | For You Feed | Multi-head | Watch completion, follow, like, comment |

---

## Strengths and Limitations

**Strengths**
- **Label efficiency**: tasks with sparse labels (CVR) benefit from dense-label auxiliary tasks (CTR)
- **Serving efficiency**: one model forward pass produces all predictions, vs. K separate model calls
- **Better embeddings**: shared representations trained on multiple signals are more general
- **Constraint satisfaction**: multi-objective optimization directly encodes business priorities

**Limitations**
- **Negative transfer**: dissimilar tasks hurt each other; requires careful task selection and architecture (MMoE/PLE mitigate but don't eliminate this)
- **Training complexity**: conflicting gradients between tasks require careful loss weighting (GradNorm, uncertainty weighting); wrong weights can harm all tasks
- **Evaluation complexity**: improvements on one task metric may mask degradations on others; requires multi-dimensional evaluation
- **Task gradient conflicts at scale**: even with MMoE, at very large scale with many dissimilar tasks, negative transfer remains a challenge

---

## Related Pages

- [[two-stage-pipeline]] — multi-task ranking models sit in the ranking stage, producing multiple scores that feed the auction/allocation
- [[click-through-rate-prediction]] — CTR is the primary task; MTL adds complementary tasks alongside it
- [[dlrm]] — DLRM-based architectures at Meta are extended with multi-task heads for production use
- [[two-tower-model]] — retrieval models can also be trained with multi-task objectives (engagement + diversity)
- [[wide-and-deep]] — foundational ranking architecture on which multi-task heads are added

## Open Questions

- What is the right metric for detecting negative transfer early in training before it degrades final model quality?
- Does the number of experts in MMoE need to scale with the number of tasks, and what is the optimal expert-to-task ratio?
- Can Transformer-based architectures (with multi-head attention) naturally subsume the role of MMoE's expert networks?
