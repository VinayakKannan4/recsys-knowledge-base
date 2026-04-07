# DIN and DIEN

**Summary**: Alibaba's Deep Interest Network (DIN, 2017) and Deep Interest Evolution Network (DIEN, 2019) extend standard Embedding&MLP ranking models with target-aware attention over user behavior history — DIN with a local activation unit per candidate ad, DIEN with a GRU-based model of how user interests evolve over time.

**Tags**: #ranking #user-behavior #attention #sequence-modeling #alibaba #ctr

**Sources**:
- raw/papers/din-deep-interest-network/din-deep-interest-network-meta.md

**Last Updated**: 2026-04-07

---

## Overview

Standard CTR models (Wide & Deep, DeepFM, DLRM) compress a user's full interaction history into a single fixed-length vector — typically by summing or averaging item embeddings from historical interactions. This compression is target-agnostic: the same user representation is used regardless of which candidate ad is being scored. This is the bottleneck DIN and DIEN target.

**DIN** (Deep Interest Network, Zhou et al., Alibaba, 2017) introduces a **local activation unit**: instead of a fixed user representation, it computes attention weights over the user's historical behaviors with respect to the specific candidate ad being scored. The user representation adapts per candidate — users who have browsed cosmetics get a cosmetics-focused representation when scoring a lipstick ad, and a sports-focused representation when scoring a running shoe ad.

**DIEN** (Deep Interest Evolution Network, 2019) extends DIN by modeling the *temporal evolution* of user interests using a GRU-based sequence model. Rather than independently attending to each history item, DIEN captures how interest state transitions over time via a two-stage GRU: a base GRU for interest extraction and an AUGRU (Attention Update Gate GRU) for target-aware interest evolution.

Both models were deployed in Alibaba's online display advertising system and demonstrated substantial improvements over the prior Embedding&MLP baseline on data with over 2 billion samples.

---

## Key Concepts

### The Fixed-Length Bottleneck

In standard Embedding&MLP architectures:

```
User history: {item_1, item_2, ..., item_N}
         ↓ (sum/average of embeddings)
User vector: e_u  (fixed, same for all candidate ads)
         ↓ concat with candidate ad embedding e_a
[MLP] → CTR prediction
```

The problem: user interests are **diverse**. A user who has interacted with clothes, sports gear, and electronics has interests in all three domains. Compressing this into one vector loses the target-dependent structure.

### DIN: Local Activation Unit

DIN replaces the fixed sum/average with a **weighted sum**, where weights are computed via an attention score between each history item and the candidate ad:

```
attention_score_i = a(e_{u_i}, e_a)
user_representation = Σ_i attention_score_i * e_{u_i}
```

Where:
- `e_{u_i}`: embedding of history item `i`
- `e_a`: embedding of the candidate ad
- `a(·, ·)`: a learned activation function (small MLP)

The user representation now **varies by candidate ad**, enabling the model to focus on the most relevant part of the user's history for each specific item being scored.

### DIN Attention Formulation

The local activation function `a` takes both embeddings and their element-wise product as input:

```
attention_score_i = σ(W · [e_{u_i}, e_a, e_{u_i} ⊙ e_a] + b)
```

The element-wise product `e_{u_i} ⊙ e_a` explicitly captures interaction between the history item and the candidate — items similar to the candidate receive higher weight. The attention scores are **not softmax-normalized**; they use sigmoid activation, preserving the magnitude of interest (a user with many relevant history interactions should have a stronger signal than one with few).

---

## How It Works

### DIN Architecture

```
User History: [item_1, ..., item_N] → [Embedding] → e_{u_1}, ..., e_{u_N}
Candidate Ad:                        → [Embedding] → e_a

Local Activation Unit:
  attention_i = MLP(e_{u_i}, e_a, e_{u_i} ⊙ e_a)   ← target-dependent weights
  user_vec = Σ_i attention_i * e_{u_i}               ← adaptive user representation

Additional user features (demographics, context) → Embedding

Concat(user_vec, e_a, other_features) → [MLP] → CTR prediction
```

### DIN Training Techniques

The paper also introduces two training innovations for industrial-scale deep networks:

1. **Mini-batch aware regularization**: standard L2 regularization penalizes all parameters uniformly, but in sparse models (with millions of embedding parameters), most embeddings are not updated in any given mini-batch. DIN applies regularization only to parameters that appear in the current batch, drastically reducing wasted computation.

2. **Data Adaptive Activation Function (Dice)**: a generalization of PReLU that adapts its rectification point based on the data distribution of each layer, improving convergence in sparse, high-dimensional settings.

### DIEN: Modeling Interest Evolution

DIEN adds temporal structure to DIN's attention mechanism. User interest doesn't just depend on what they've clicked — it evolves over time. A user who was interested in baby products last month and is now browsing strollers is showing a temporal pattern that DIN's bag-of-clicked-items representation misses.

**Stage 1 — Interest Extractor Layer** (base GRU):
```
h_t = GRU(e_{u_t}, h_{t-1})
```
A standard GRU processes the user history sequence, with each hidden state `h_t` representing the user's interest state at timestep `t`.

**Stage 2 — Interest Evolving Layer** (AUGRU — Attention Update Gate GRU):

AUGRU modifies the GRU's update gate with target-aware attention:

```
ũ_t = a_t * u_t     ← attention score gates the update

h'_t = (1 - ũ_t) ⊙ h'_{t-1} + ũ_t ⊙ h̃'_t
```

Where:
- `a_t`: attention score between history item `t` and the target item (same as DIN's local activation)
- `u_t`: standard GRU update gate
- `ũ_t`: attention-weighted update gate

The intuition: if history item `t` is irrelevant to the candidate ad (low `a_t`), its hidden state update is suppressed, so the model retains the interest state from the previous step. If relevant (high `a_t`), the hidden state evolves to reflect that interest.

The final hidden state `h'_T` of the AUGRU serves as the user's evolved interest representation for the candidate.

---

## Pipeline Position

**DIN and DIEN are ranking models.** They operate on pre-retrieved candidates and score each with a target-aware user representation.

```
[Two-Tower Retrieval] → [DIN / DIEN Ranker] → [Auction / Allocation]
```

The attention mechanism over user history requires materializing `N` history item embeddings per candidate per request — computationally feasible for hundreds of candidates but not for full-corpus scoring.

---

## Industry Usage

| Company | Surface | Notes |
|---------|---------|-------|
| Alibaba | Display advertising (Taobao) | Original deployment; DIN deployed in 2017, DIEN in 2019; evaluated on 2B+ sample dataset |
| Various e-commerce | Product recommendations | Architecture widely adopted in ad/recommendation contexts |
| Meta | News Feed, Ads (influenced) | Meta's BST (Behavior Sequence Transformer) follows similar target-aware behavior modeling |

---

## Strengths and Limitations

**Strengths**
- **Target-aware user modeling**: user representation adapts to each candidate, capturing diverse multi-domain user interests
- **Directly interpretable**: attention weights over history items are inspectable — the model effectively tells you which past behaviors are relevant to each prediction
- **DIEN adds temporal context**: interest evolution modeling captures trajectory, not just a bag of past items
- **Proven at Alibaba scale**: 2B+ training samples, significant AUC improvements over Embedding&MLP baselines

**Limitations**
- **Quadratic in history length**: for a user with N history items and M candidates, the attention computation is O(N·M) — expensive for users with very long histories
- **History length truncation**: in practice, history is capped (e.g., last 50 items), which may lose distant but relevant signals
- **No explicit cross-features between user and item attributes**: DIN attends over history but doesn't learn feature-level interactions (as in DeepFM or DCN); these are complementary approaches often combined in practice
- **DIEN adds training complexity**: the two-stage GRU with auxiliary supervision requires careful training with an auxiliary loss for the extractor layer

---

## Related Pages

- [[wide-and-deep]] — predecessor paradigm; DIN extends the deep component with target-aware attention
- [[deepfm]] — complementary: DeepFM handles feature-level interactions; DIN handles user behavior sequence modeling; Alibaba's practice combines both
- [[dcn-v2]] — complementary: cross-matrix feature interactions; DIN-style attention over history can be layered on top
- [[dlrm]] — DLRM ignores user history sequencing; DIN/DIEN address this gap
- [[two-tower-model]] — upstream retrieval; DIN/DIEN rank the candidates that two-tower retrieval produces
- [[monolith]] — TikTok's production system; attention over user history (DIN-like) is a natural addition to Monolith's embedding layer

## Open Questions

- What is the optimal history length to attend over, and does retrieval over user history (semantic search into past interactions) outperform fixed-window truncation?
- How does DIEN compare to Transformer-based sequence models (BST, SASRec) for interest evolution modeling?
- Can DIN's local activation unit be combined with DCN-V2's cross layers in a single model without prohibitive computational cost?
