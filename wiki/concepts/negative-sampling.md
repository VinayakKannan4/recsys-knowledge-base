# Negative Sampling

**Summary**: The techniques used to construct negative training examples for recommendation models trained on implicit feedback (clicks, watches, purchases) — where only positive interactions are observed — covering random, in-batch, hard, and importance-sampling-corrected strategies and their specific effects on two-tower retrieval quality.

**Tags**: #training #negative-sampling #two-tower #retrieval #implicit-feedback #bias

**Sources**:
- raw/papers/two-tower-sampling-bias/two-tower-sampling-bias-meta.md
- raw/papers/facebook-ebr/facebook-ebr-meta.md
- raw/blogs/other/two-tower-deep-dive-shaped.md

**Last Updated**: 2026-04-07

---

## Overview

Recommendation datasets are almost exclusively **implicit feedback**: we observe what users clicked, watched, or purchased, but we don't observe explicit "I did not like this item" signals. Every training example in the log is a positive (user interacted with item). To train a model that assigns high scores to relevant items and low scores to irrelevant ones, we need to construct negatives — examples of (user, item) pairs that the user did *not* interact with.

The choice of which negatives to use is not merely a training detail — it is one of the most consequential modeling decisions in building a retrieval system. The negative distribution determines what the model learns: the implicit task is "score this item higher than the negatives you saw during training." If all negatives are random, the model learns a different function than if negatives are sampled from near-misses. And if the negative distribution is biased (e.g., over-represents popular items), the model's scoring behavior will reflect that bias in ways that may not match the deployment distribution.

---

## Key Concepts

### Why Negatives Are Necessary

Without negatives, a model trained to maximize the score of positive pairs has a trivial solution: assign every item the same maximum score. The model must be trained to *contrast* positives against negatives. In the two-tower setting, the contrastive objective is:

```
L = -log [ exp(u · v+) / (exp(u · v+) + Σ_j exp(u · v_j-)) ]
```

Where `v+` is the positive item embedding and `{v_j-}` is the set of negative item embeddings. Maximizing this pushes `u · v+` above `u · v_j-` for all negatives.

### The Negative Distribution Matters

The implicit question the model answers during training is: "Is this item more relevant to this user than a typical item from the negative distribution?" If the negative distribution is very different from the serving distribution (e.g., random items when serving requires discriminating between near-similar items), the model may score well on training but poorly in production.

---

## Types of Negatives

### 1. Random Negatives

Sample negatives uniformly at random from the item corpus.

**Properties**:
- Most negative items will be clearly irrelevant (random items rarely match user interests)
- Easy to implement; requires no mining
- Training signal is too "easy" — the model can learn to separate positives from random negatives with a relatively coarse representation

**Use case**: appropriate for initial training or when the item corpus is small and relatively homogeneous.

**Limitation**: once trained, the model may fail at the hard part of ranking — discriminating between somewhat-relevant but non-clicked items and truly relevant items.

### 2. In-Batch Negatives

Within a mini-batch of B (user, positive_item) pairs, treat all other items in the batch as negatives for each user. For a batch of size B, each example gets B−1 negatives.

```
Batch: [(u_1, v_1), (u_2, v_2), ..., (u_B, v_B)]

For user u_i: negatives are {v_j : j ≠ i}
```

**Properties**:
- Computationally free: item embeddings are already computed for the batch
- B−1 negatives per example without extra forward passes
- Scales naturally with batch size: larger batches → more negatives per example

**Implementation** (matrix form):
```
scores = U @ V.T    # (B × B) similarity matrix
loss = cross_entropy(scores, eye(B))   # diagonal is positives
```

**Problem — popularity bias**: items that appear frequently in training data appear more often as negatives. This means popular items receive many gradient updates pushing them *down*, causing the model to systematically under-rank popular-but-relevant items. This is the **sampling bias** problem described in Yi et al. (2019).

### 3. Sampling-Bias-Corrected Negatives (Yi et al. 2019)

Google's paper on two-tower training identifies that in-batch negatives create a biased sampler: the probability of an item appearing as a negative is proportional to its frequency in training data (i.e., its popularity). This distorts the learned embedding space.

**Correction**: subtract the log-probability of sampling each negative:

```
corrected_score(u, v) = u · v - log p(v)
```

Where `p(v)` is the estimated sampling probability for item `v` (proportional to its frequency in the training stream, estimated via streaming frequency counting).

The corrected loss:
```
L = -log [ exp(u · v+ - log p(v+)) / Σ_j exp(u · v_j - log p(v_j)) ]
```

**Effect**: this correction is equivalent to de-biasing the softmax denominator, making the implicit training distribution closer to a uniform negative sampler. Items are ranked relative to the uniform corpus distribution rather than the popularity-weighted one. This directly addresses the over-ranking of unpopular items and under-ranking of popular-but-relevant ones.

### 4. Hard Negatives

Hard negatives are items that the model currently scores highly for a given user but the user did not actually interact with — near-misses in the embedding space.

**Construction**:
1. Train an initial model on easy negatives
2. Run the model on all (user, item) pairs in validation set
3. For each user, find items ranked just outside the top-K (e.g., rank 101–200) that the user did not interact with
4. Use these as additional negatives for re-training

**Properties**:
- Provides a strong training signal for discriminating between similar items
- Prevents the model from learning a coarse "good items are here, junk is there" partition
- Drives learning of subtle, fine-grained distinctions in the embedding space

**Risk**: if negatives are too hard (accidentally include relevant items the user didn't get a chance to interact with), the model can degrade — known as **false negatives**. Hard negative mining requires careful calibration of the difficulty threshold.

Facebook's EBR paper (2020) identified hard negative mining as a critical component for improving search quality: after initial training, they constructed hard negatives from near-misses and retrained, significantly improving recall@K in Facebook Search.

### 5. Mixed Negatives

In practice, effective systems use a mix:
- **Easy negatives**: provide stable, broad signal across the full score range
- **Hard negatives**: provide fine-grained discrimination near the decision boundary

A common recipe: for each positive, sample M easy (in-batch or random) negatives + N hard negatives, with M >> N (e.g., 99 easy + 5 hard per positive).

---

## Effect on Two-Tower Model Quality

The two-tower training loop is particularly sensitive to negative sampling because the dot-product scoring function has a limited number of degrees of freedom to separate items. Poor negatives lead to:

1. **Embedding collapse**: all item embeddings cluster in a small region; the model doesn't learn to spread the embedding space
2. **Popularity bias**: popular items are scored too low (pushed down by over-abundant negative updates)
3. **Poor recall@K**: the model retrieves the wrong items because it learned to contrast against the wrong distribution

The sampling-bias correction from Yi et al. is now considered standard practice for production two-tower training at Google and influenced practices at other platforms.

---

## Industry Practice Summary

| Company | Strategy | Key Contribution |
|---------|----------|-----------------|
| Google / YouTube | In-batch negatives + bias correction | Yi et al. 2019: subtract log p(item) from scores |
| Meta / Facebook | In-batch + staged hard negative mining | EBR paper 2020: hard negatives from near-misses after initial training |
| Pinterest | Mixed negatives + popularity downsampling | Combine in-batch with sampled hard negatives from retrieval near-misses |
| General practice | In-batch as default; hard negatives for refinement | Most platforms follow this two-stage training recipe |

---

## Strengths and Limitations

**Strengths** (of in-batch negatives specifically)
- Zero extra compute cost: piggybacks on the batch already being processed
- Scales with batch size: more negatives for free by increasing B
- Simple to implement correctly

**Limitations**
- Popularity bias is systematic and significant: requires explicit correction
- In-batch negatives are correlated (they come from the same mini-batch distribution, not the full item distribution)
- Hard negative mining requires an additional training pass and introduces false negative risk
- No universally optimal strategy: the right mix depends on corpus size, task, and data distribution

---

## Related Pages

- [[two-tower-model]] — primary consumer of negative sampling techniques; the entire training procedure depends on negative strategy
- [[two-stage-pipeline]] — retrieval stage where negative sampling quality directly affects recall@K
- [[click-through-rate-prediction]] — ranking stage uses impression-level negatives (items shown but not clicked), which is a different negative distribution than retrieval
- [[embedding-tables]] — the embedding quality learned by the two-tower model is directly shaped by negative sampling

## Open Questions

- What is the optimal hard-to-easy negative ratio as a function of corpus size and task difficulty?
- Does the bias correction from Yi et al. generalize to non-popularity-based biases (e.g., recency bias, positional bias in the training corpus)?
- Can self-supervised contrastive learning objectives (e.g., SimCLR-style augmentation on item features) supplement or replace behavioral negatives in cold-start settings?
