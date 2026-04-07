# Click-Through Rate (CTR) Prediction

**Summary**: The binary classification task of estimating the probability that a user will click on a shown item — the central ranking signal in both organic recommendation and advertising, requiring well-calibrated outputs, position-bias correction, and integration with auction mechanisms that multiply pCTR by advertiser bid to determine expected revenue.

**Tags**: #ctr #ranking #calibration #position-bias #auction #advertising #binary-classification

**Sources**:
- raw/papers/wide-and-deep/wide-and-deep-meta.md
- raw/papers/deepfm/deepfm-meta.md
- raw/papers/dcn-v2/dcn-v2-meta.md
- raw/papers/youtube-dnn-recommendations/youtube-dnn-recommendations-meta.md
- raw/blogs/other/aman-ai-recsys-architectures.md
- raw/blogs/pinterest/ad-ranking-infoq-2024.md
- raw/blogs/pinterest/contextual-relevance-ads-ranking.md

**Last Updated**: 2026-04-07

---

## Overview

Click-through rate prediction is the task every recommendation and advertising system must solve: given a user and a candidate item (ad, video, product, pin), estimate the probability that the user will click. This probability — written pCTR — is used to rank candidates, allocate ads inventory, and compute expected revenue in auctions.

CTR prediction sits at the intersection of three challenges:
1. **Modeling**: building a model expressive enough to capture complex user-item interactions (the focus of DeepFM, DCN-V2, DLRM, DIN, and the entire ranking architecture literature)
2. **Calibration**: ensuring predicted probabilities reflect true empirical click rates (a model can rank well without being calibrated, but auctions require calibrated probabilities)
3. **Bias correction**: training data is collected under the existing ranking policy, which shows popular or high-CTR items more often — creating feedback loops that must be corrected

---

## Key Concepts

### Formal Problem Formulation

CTR prediction is a binary classification problem:

```
Input:  (user u, item i, context c)    ← user features, item features, contextual features
Output: p̂(click | u, i, c) ∈ [0, 1]  ← estimated click probability

Label:  y = 1 if user u clicked item i, 0 otherwise
Loss:   L = -Σ [y · log(p̂) + (1-y) · log(1-p̂)]   ← binary cross-entropy
```

In practice, "context" includes: page position, time of day, device type, request context (search query vs. feed vs. related items).

### Connection to Ranking

For organic recommendations, candidates are ranked directly by pCTR (or a weighted combination of multiple predicted signals). The final feed presented to the user is the top-N items by predicted relevance/engagement.

For ads, CTR prediction integrates with the auction:

```
expected_value_i = pCTR_i × bid_i
```

Where `bid_i` is the advertiser's willingness to pay per click. The auction ranks ads by expected value; the winner pays the second-highest bid (Vickrey/second-price auction) or a variant. This means:
- **An ad with high predicted CTR wins the auction without needing to bid highest** — good CTR models benefit advertisers with relevant ads
- **Miscalibrated pCTR breaks the auction**: if pCTR is systematically too high, advertisers are charged too much; if too low, relevant ads lose to less relevant but higher-bidding ads

### Calibration

A CTR model is **well-calibrated** if predicted probabilities match empirical click rates across the scoring distribution:

```
E[y | p̂ = p] ≈ p    for all p ∈ [0, 1]
```

In practice: if the model predicts pCTR = 0.05 for 1000 impressions, approximately 50 of those should result in clicks.

**Why calibration is hard**:
- Training data is collected under the existing ranking/auction policy, not uniformly
- Different content types (shopping, video, regular ads) have different baseline CTRs
- Temporal drift: CTR distributions shift as user behavior evolves, but model calibration is fit at training time

Pinterest explicitly describes calibration as a first-class concern: miscalibrated models cause irregular calibration curves and require post-hoc correction layers or dedicated calibration models. They found that missing features (e.g., missing category-match features for shopping ads) cause characteristic calibration failures.

**Post-hoc calibration methods**:
- Platt scaling: fit a logistic regression on model outputs vs. empirical click rates on a held-out set
- Isotonic regression: non-parametric monotone transform to match empirical rates
- Temperature scaling: divide logits by a learned temperature T before sigmoid

---

## How It Works

### Feature Categories for CTR Models

| Category | Examples | Modeling |
|----------|---------|---------|
| User identity | User ID, device | Embedding lookup |
| User history | Recent clicks, watches, interactions | Sum pool, DIN attention, sequence model |
| Item attributes | Item ID, category, content features | Embedding lookup + content encoders |
| Context | Time of day, page, position | Normalized numerics, embeddings |
| Cross features | User×item interactions | Explicit (DeepFM/DCN) or implicit (MLP) |

### Training Data Construction

Training examples are impressions: items that were shown to users. The label is whether the user clicked.

**Key biases in training data**:

1. **Positivity bias**: the label is 1 only if the user clicked. Non-click does not mean irrelevant — the user may simply have missed the item or not been in a clicking mood.

2. **Exposure bias**: items that the existing model ranked highly are shown more often. A new model trained on this data learns partly from a distribution shaped by the old model's decisions — a feedback loop.

3. **Position bias**: items shown in prominent positions (top of feed, first ad slot) get more clicks regardless of quality. A user is more likely to click position 1 than position 5, holding item quality fixed.

### Position Bias Correction

Position bias is a major confound in training data: the click label reflects both item relevance and position. Naively training on click data teaches the model that position-1 items are better, regardless of their actual quality.

**Correction methods**:

1. **Inverse propensity weighting (IPW)**: weight each training example by the inverse probability that the item was shown at its position:
   ```
   L_corrected = Σ_i (y_i / p(position_i)) · log(p̂_i)
   ```

2. **Shallow position tower**: use a two-tower structure where a separate "shallow" tower models position effects, and the main tower models relevance. YouTube's "Watch Next" paper and others describe additive or multiplicative composition of these towers.

3. **Position as feature only at training**: include position as an input feature during training; at serving, set position to a neutral/unknown value. The model learns to separate relevance from position during training but scores without position bias at inference.

4. **Counterfactual logging**: if the platform can randomly swap item positions in some fraction of impressions, this provides unbiased data.

### Model Architecture Evolution

The history of CTR model architectures is essentially the history of the ranking architecture literature:

```
2016: Wide & Deep     — manual cross features + MLP; deployed on Google Play
2017: DeepFM          — FM-style automatic crosses; Huawei
2017: DIN             — target-aware attention over user history; Alibaba
2019: DLRM            — dot-product pairwise interactions + MLP; Meta
2020: DCN-V2          — matrix cross layers; Google
2019: MMoE            — multi-task CTR + CVR + engagement; YouTube
2023: GDCN            — gated cross layers; current Criteo SOTA
2024: Sequence models — Transformer over event history (EBF); Meta
```

Each step improves AUC on benchmark datasets (Criteo, Avazu) and/or on proprietary production data, with the gains translating to measurable improvements in click volume or advertiser ROI.

### Evaluation Metrics

**Offline metrics**:
- **AUC** (Area Under ROC Curve): measures ranking quality; how well the model separates clickers from non-clickers. Independent of calibration.
- **Logloss** (binary cross-entropy): measures both ranking and calibration quality. Standard benchmark on Criteo dataset.
- **NE** (Normalized Entropy): Logloss normalized by the baseline entropy; measures improvement over predicting the empirical average CTR.

**Online metrics**:
- **CTR lift**: percentage increase in click-through rate in A/B test
- **Revenue lift**: change in advertiser revenue per thousand impressions (RPM)
- **Conversion rate**: downstream conversions per click (used in multi-task settings)

The offline-online correlation is imperfect: a model can improve AUC offline but show no gain online (or vice versa). Position bias, feedback loops, and explore/exploit dynamics create gaps.

---

## Industry Usage

| Platform | CTR Use Case | Key Challenges |
|---------|-------------|----------------|
| Google Ads / Search | Ad ranking, auction | Calibration for auction correctness; advertiser quality |
| Meta Ads | News Feed, Instagram ads | Multi-task (CTR + CVR + engagement); sequence learning |
| YouTube | Video recommendation | CTR + watch time; position bias from autoplay |
| Pinterest | Ads + organic | Multi-surface calibration; relevance vs. click tension |
| TikTok | For You Feed | CTR + completion rate + share rate; real-time online training |

At Google scale, a 0.1% improvement in CTR prediction quality (measured as logloss or NE) can translate to hundreds of millions of dollars in annual revenue — this is why the Criteo leaderboard is competitive and industry teams invest heavily in CTR architecture research.

---

## Strengths and Limitations

**Strengths**
- Well-defined objective with clear evaluation metrics and benchmark datasets
- Directly connects to business value (clicks → revenue in ads context)
- Binary cross-entropy loss is well-behaved and widely understood
- Large body of public research (Criteo, Avazu benchmarks enable apples-to-apples comparison)

**Limitations**
- **Misaligned with user value**: maximizing pCTR promotes clickbait; engagement depth and satisfaction are better proxies for user value but harder to predict
- **Feedback loops**: models trained on CTR data shape what users see, which shapes the CTR data used to retrain models — a closed loop that can amplify biases
- **Position bias**: click labels conflate relevance and position; requires active correction
- **Calibration maintenance**: CTR distributions drift over time; calibration requires continuous monitoring and recalibration
- **Surrogate for complex behavior**: a click is a noisy proxy for relevance; user hide/skip signals, engagement time, and post-click behavior are richer but sparser

---

## Related Pages

- [[feature-crosses]] — the primary mechanism for improving CTR model expressiveness; progression from Wide&Deep to DCN-V2
- [[two-stage-pipeline]] — CTR prediction is the core signal in the ranking stage; retrieval is typically optimized for recall, not CTR
- [[multi-task-learning]] — production CTR models predict CTR jointly with CVR, engagement, and other signals
- [[negative-sampling]] — CTR training requires constructing negative examples from non-clicked impressions
- [[dlrm]] — Meta's production CTR prediction architecture
- [[dcn-v2]] — Google's production CTR prediction architecture with cross layers
- [[deepfm]] — Huawei's CTR model; FM-style pairwise crosses for automated cross feature learning
- [[din-dien]] — Alibaba's CTR model with target-aware attention over user behavior history

## Open Questions

- Is CTR the right primary optimization target, or does it systematically diverge from user satisfaction in ways that matter long-term?
- Can offline logloss/AUC improvements reliably predict online CTR/revenue gains, and what causes the cases where they diverge?
- How should platforms balance CTR optimization with diversity and fairness objectives that are harder to quantify?
