# Feature Crosses

**Summary**: Second- (and higher-) order features formed by combining two or more input features, capturing interaction effects that linear models miss — progressing from manual engineering in Wide & Deep (2016) to automatic polynomial crossing in DCN (2017), FM-style dot products in DeepFM (2017), and matrix-based bit-wise crossing in DCN-V2 (2020).

**Tags**: #feature-interaction #cross-features #ranking #ctr #memorization

**Sources**:
- raw/papers/wide-and-deep/wide-and-deep-meta.md
- raw/papers/dcn-v2/dcn-v2-meta.md
- raw/papers/deepfm/deepfm-meta.md
- raw/blogs/other/rise-of-dcn-ml-frontiers.md
- raw/blogs/other/aman-ai-recsys-architectures.md

**Last Updated**: 2026-04-07

---

## Overview

Feature crosses are among the most practically important concepts in ranking model design. A linear model operating on raw categorical features (user installed Netflix, impression app is Hulu) cannot detect that a user who installed a streaming video app is more likely to click on another streaming video app — this conditional dependency requires a *product* of the two features. Feature crosses provide these products explicitly.

The evolution of feature cross methodology tracks closely with the evolution of ranking architectures over the past decade:

1. **Manual crosses (Wide & Deep, 2016)**: domain experts hand-pick which feature pairs to cross; effective but labor-intensive and non-scaling
2. **Dot-product crosses (DeepFM, 2017)**: factorization machine automatically computes all pairwise dot products; no engineering, second-order only
3. **Polynomial crossing (DCN, 2017)**: cross neural network generates arbitrary-order polynomial feature crosses; vector-based, limited expressiveness
4. **Matrix crossing (DCN-V2, 2020)**: replaces weight vector with weight matrix, enabling bit-wise interactions and much higher expressiveness
5. **Gated crossing (GDCN, 2023)**: adds information gates to filter noisy crosses, enabling still deeper crossing without overfitting

---

## Key Concepts

### What Is a Feature Cross?

A feature cross is a product of two or more feature values. For binary (one-hot) features, this is an AND:

```
cross(f_a, f_b) = 1  iff  f_a = v_a AND f_b = v_b
```

Example from Google Play (Wide & Deep paper):
```
AND(user_installed_app='netflix', impression_app='hulu')    → 1 if true
AND(user_installed_category='video', impression_category='video')
```

For real-valued features or embeddings, a cross is a product of their values (element-wise or via dot product).

### Why MLPs Cannot Efficiently Learn Feature Crosses

Standard feedforward MLPs are universal function approximators, but empirical results (Wang et al. 2020, Beutel et al. 2018) show they **cannot efficiently approximate even 2nd or 3rd-order feature interactions** from sparse data without explicit guidance. The reason: in sparse settings, a feature pair may co-occur in only a small fraction of training examples. An MLP distributes its capacity evenly; it cannot efficiently allocate capacity to model specific sparse interactions. Explicit cross features provide a shortcut.

### Memorization vs. Generalization

Feature crosses serve two distinct purposes:
- **Fine-grained crosses → memorization**: `AND(user_installed='netflix', impression='hulu')` captures a specific user-item pattern. The model memorizes a rule.
- **Coarse-grained crosses → generalization**: `AND(user_category='video', impression_category='video')` generalizes across all video apps. The model learns a broader principle.

This memorization–generalization duality was the original motivation for Wide & Deep's combined architecture.

---

## How It Works: The Progression of Cross Architectures

### 1. Manual Crosses: Wide & Deep (2016)

The wide component is a generalized linear model:
```
y_wide = W · [x, φ(x)] + b
```
where `φ(x)` is a manually engineered set of cross-product features. For Google Play, engineers built crosses like `(user_installed_app, impression_app)` pairs.

**Problem**: requires domain expertise to enumerate useful pairs. At web scale (millions of features), this is infeasible. Every new product surface requires a new round of feature engineering.

### 2. FM-Style Dot Products: DeepFM (2017)

The FM layer automatically computes pairwise interactions via embedding dot products:
```
ŷ_FM = w_0 + Σ_i w_i x_i + Σ_i Σ_{j>i} <v_i, v_j> x_i x_j
```

Key efficiency: all pairwise dot products computed in O(kn) via:
```
Σ_i Σ_{j>i} <v_i, v_j> x_i x_j = ½ [(Σ_i v_i x_i)² - Σ_i (v_i x_i)²]
```

**Limitation**: all pairwise crosses are computed uniformly — including noisy ones. Bounded to second-order.

### 3. Polynomial Crossing: DCN (2017)

The cross network generates polynomial feature interactions of arbitrary order by stacking cross layers. Each DCN cross layer:
```
x_{l+1} = x_0 · x_l^T · w_l + b_l + x_l
```

With L layers, DCN can represent features of polynomial degree up to L+1. However, the weight **vector** `w_l ∈ R^d` limits expressiveness — the outer product `x_0 · x_l^T` is collapsed by `w_l` into a single weighted combination.

**Observation**: DCN performance plateaus after ~2 cross layers, suggesting the weight vector is a bottleneck.

### 4. Matrix Crossing: DCN-V2 (2020) — Key Innovation

DCN-V2 replaces the weight vector with a weight **matrix**:
```
x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l
```

Where:
- `W_l ∈ R^(d×d)`: a full weight matrix (vs. vector `w_l` in DCN)
- `⊙`: element-wise (Hadamard) product

**Effect**: `W_l` enables **bit-wise interactions** — each dimension of `x_l` can interact with every dimension of `x_0` independently, via `d²` degrees of freedom (vs. `d` in DCN). DCN-V2 continues improving up to 4+ cross layers, confirming the original vector was the expressiveness bottleneck.

**DCN-Mix** (same paper) adds low-rank factorization `W_l ≈ U_l V_l^T` (reducing cost from O(d²) to O(d·r)) and Mixture-of-Experts (K expert matrices, combined via a gate), preserving expressiveness at reduced computation.

### 5. Gated Crossing: GDCN (2023)

Brute-force crossing generates many noisy interactions that can cause overfitting when stacking many layers. GDCN adds an **information gate** per cross layer:
```
x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) ⊙ G_l(x_l) + x_l
```

The gate `G_l` dynamically filters cross interactions by importance, making the crossing sparse rather than dense. Visualization of the gate values shows that most feature cross interactions are low-weight (noise), validating the gating hypothesis. GDCN achieves 0.8161 AUC on Criteo (current SOTA as of 2023).

---

## Comparison Table

| Architecture | Cross Type | Order | Manual Effort | Expressiveness |
|-------------|-----------|-------|---------------|----------------|
| Wide & Deep (2016) | Manual AND/product | 2nd | High | Low (fixed pairs) |
| DeepFM (2017) | FM dot product | 2nd | None | Medium (all pairs, uniform) |
| DCN (2017) | Cross layer (vector) | Up to L+1 | None | Medium (vector bottleneck) |
| DCN-V2 (2020) | Cross layer (matrix) | Up to L+1 | None | High (bit-wise) |
| DCN-Mix (2020) | Low-rank MoE cross | Up to L+1 | None | High + efficient |
| GDCN (2023) | Gated cross | Up to L+1 | None | Highest (sparse, filtered) |

---

## Pipeline Position

Feature cross learning is a **ranking-stage** technique. The computation of cross interactions requires materializing both user and item features simultaneously, which is only feasible for a limited candidate set (hundreds–thousands), not a full item corpus.

```
[Two-Tower Retrieval (no cross features)]
         ↓
[Ranking Model with cross features: DCN-V2, DeepFM, DLRM]
         ↓
[Auction]
```

Note: the two-tower retrieval stage deliberately *avoids* user×item cross features (the towers compute user and item representations independently). This is the fundamental trade-off: two-tower gains scalability by sacrificing cross-feature expressiveness. Ranking recovers the expressiveness at the cost of evaluating fewer candidates.

---

## Industry Usage

| Company | Architecture | Cross Mechanism |
|---------|-------------|-----------------|
| Google | DCN-V2, Wide & Deep (legacy) | Matrix cross layers; previous manual crosses in wide component |
| Meta | DLRM, DHEN | Pairwise dot products (FM-style); DHEN adds hierarchical higher-order crosses |
| Huawei | DeepFM | FM dot products; no manual engineering |
| Pinterest | DCN-V2-inspired, Transformer ensemble | Cross layers + attention-based feature interaction |
| Alibaba | DIN on top of DeepFM | FM crosses + target-aware attention over user history |

---

## Related Pages

- [[wide-and-deep]] — introduced the memorization/generalization framing and the first production deployment of cross features
- [[deepfm]] — automating cross features via FM dot products, eliminating manual engineering
- [[dcn-v2]] — the current dominant approach: matrix cross layers for bit-wise interactions, with DCN-Mix and GDCN extensions
- [[dlrm]] — pairwise dot product interactions between all embedding pairs (similar to FM component)
- [[embedding-tables]] — cross features operate on the dense vectors produced by embedding tables
- [[click-through-rate-prediction]] — cross features are the primary driver of CTR model quality improvements

## Open Questions

- Is there a theoretical limit to how much cross feature learning can improve CTR prediction, beyond which the gains become architecture-agnostic?
- How do DCN-V2 cross layers compare to self-attention (AutoInt) when controlling for parameter count — are they learning qualitatively different interactions?
- Do cross features generalize across domains (a model trained on news feed clicks tested on ad clicks), or do they memorize domain-specific patterns?
