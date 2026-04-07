# Embedding Tables

**Summary**: Learned lookup tables that map high-cardinality sparse categorical feature IDs (user IDs, item IDs, content categories) to dense low-dimensional vectors, enabling neural networks to operate on discrete entities — the single largest memory consumer in production recommendation models.

**Tags**: #embeddings #sparse-features #infrastructure #memory #representation-learning

**Sources**:
- raw/papers/dlrm/dlrm-meta.md
- raw/papers/monolith-tiktok/monolith-tiktok-meta.md
- raw/blogs/tiktok/tiktok-secret-sauce-shaped.md
- raw/blogs/other/aman-ai-recsys-architectures.md
- raw/blogs/meta/sequence-learning-ads-2024.md

**Last Updated**: 2026-04-07

---

## Overview

Recommendation systems are fundamentally different from vision and NLP models in one critical way: the dominant features are **sparse and categorical**, not dense and continuous. A user ID from a platform with 3 billion users is meaningless as an integer — what matters is the *pattern of behavior* associated with that ID across billions of training examples. Embedding tables solve this: they learn a dense vector representation `e ∈ R^d` for each unique ID, such that IDs with similar behavioral patterns end up close in the embedding space.

In a production recommender, the embedding tables collectively dwarf every other part of the model. At Meta, embedding tables for the full DLRM system occupy terabytes of memory — far more than the MLP weights. This makes embedding tables the primary bottleneck for both memory capacity and memory bandwidth, directly shaping hardware choices, parallelization strategies, and infrastructure design. The way embeddings are stored, indexed, and updated is as architecturally important as the neural network sitting on top of them.

---

## Key Concepts

### Sparse Categorical Features

Recommendation inputs can have cardinalities ranging from thousands (content categories) to billions (user IDs, item IDs, ad IDs). These cannot be one-hot encoded — a 3-billion-dimensional input vector is impossible to process. Instead, each categorical feature is mapped to a compact dense vector via a lookup:

```
Feature ID (integer) → [Embedding Table] → e ∈ R^d
```

The embedding dimension `d` is typically 16–256, depending on the feature's cardinality and importance. At serving time, embedding lookup is an O(1) memory read.

### Training via Backpropagation

Embedding vectors are initialized randomly (or with a small normal distribution) and trained end-to-end via backpropagation. Because the lookup is a differentiable operation (the gradient flows back to the retrieved embedding vector directly), embeddings learn representations that minimize the final task loss.

A key property: **only embeddings for IDs that appear in a mini-batch receive gradient updates**. For long-tail distributed features (most IDs appear rarely), most embedding rows are updated infrequently. This sparsity of updates is a fundamental challenge for optimization.

### Pooling: Bag-of-Embeddings

Users interact with many items. A user's recent history might be a sequence of 50–1000 item IDs. Standard approaches pool these into a single vector:

- **Sum pooling**: `e_user = Σ_i e_{item_i}` — simple, fast, ignores order
- **Mean pooling**: `e_user = (1/N) Σ_i e_{item_i}` — same, normalizes by count
- **Attention pooling (DIN-style)**: `e_user = Σ_i a_i(candidate) * e_{item_i}` — target-aware weights

Sum/mean pooling loses sequential and target-conditional information. DIN and DIEN replace this with attention-based pooling that adapts the user representation per candidate. Meta's 2024 sequence learning work (EBF + Transformer) goes further, treating the raw event sequence as input to a Transformer rather than pooling at all.

---

## How It Works

### Memory Layout

An embedding table for a feature with vocabulary size V and embedding dimension d is a matrix `E ∈ R^(V × d)`. For V = 1 billion, d = 64:

```
Table size = 1e9 × 64 × 4 bytes (float32) = 256 GB
```

This is why a single production recommendation model can require terabytes of storage. Multiple tables exist (one per categorical feature), each with different vocabulary sizes.

### The Hashing Trick

For features with unbounded or very large vocabularies, direct indexing is impractical. The **hashing trick** maps each ID to a table slot using a hash function:

```
slot_index = hash(feature_id) % table_size
```

This caps table size but introduces **hash collisions**: two different IDs map to the same embedding vector and share representations. At small collision rates this is tolerable, but at large scale (billions of IDs, modestly sized tables) it meaningfully degrades model quality.

All major frameworks (TensorFlow, PyTorch) use the hashing trick by default. TensorFlow's `tf.nn.embedding_lookup_sparse` and PyTorch's `nn.EmbeddingBag` both rely on it.

### Collisionless Embedding: Monolith's Cuckoo Hashmap

TikTok's Monolith system identified hash collisions as a significant quality problem and built a **collisionless embedding table** using a cuckoo hash map:

- Two hash tables T₀ and T₁ with independent hash functions h₀ and h₁
- To insert ID `x`: try slot `h₀(x)` in T₀; if occupied, evict the existing entry and insert it at `h₁(evicted)` in T₁; repeat until all entries placed or rehash triggers
- Guaranteed unique slot per ID — zero collisions

Additional memory controls to prevent unbounded table growth:
- **Frequency filter**: IDs appearing fewer than a threshold count are not inserted (long-tail IDs don't contribute signal worth the memory cost)
- **Expiry timer**: IDs inactive for a configurable duration (e.g., stale video IDs) are automatically deleted

This allows TikTok to maintain a fresh, collision-free table over a continuously changing vocabulary without unbounded memory growth.

### DLRM's Parallelization Scheme

At Meta's scale, embedding tables are too large to fit on a single device. DLRM introduced a hybrid parallelism strategy:

- **Model parallelism for embedding tables**: each table (or table shard) is pinned to a specific GPU/device. Embedding lookups across devices require all-to-all communication
- **Data parallelism for MLP layers**: dense layers are replicated across all devices; each device processes a shard of the batch

This split reflects the fundamental nature of the problem: embedding table operations are memory-bandwidth-bound, while MLP operations are compute-bound. Modern recommendation hardware (e.g., Meta's ZION platform) is designed around this split.

---

## Pipeline Position

Embedding tables are present in every stage of the recommendation pipeline:

```
[Two-Tower Retrieval]      → user/item embedding tables → dense tower inputs
[Ranking Model (DLRM/DCN)] → all feature embedding tables → interaction layer inputs
[DIN/DIEN]                 → user history embedding lookups → attention over history
```

They are the first operation in every forward pass and the primary memory consumer at every stage.

---

## Industry Usage

| Company | Scale | Approach |
|---------|-------|----------|
| Meta | Terabyte-scale tables; billions of user/item IDs | DLRM model parallelism; embedding sharding across GPUs; 2024 shift to sequence learning (EBF) |
| TikTok/ByteDance | Continuously growing vocab (new videos, users daily) | Monolith cuckoo hashmap; frequency filtering; expiry timers |
| Google | Multiple tables per feature; diverse vocabulary sizes | DCN-V2 supports variable embedding sizes; distributed PS training |
| Pinterest | User, item, visual, graph embeddings | PinSage GNN embeddings in item tower; separate visual embedding model |

---

## Strengths and Limitations

**Strengths**
- Enable neural networks to process arbitrarily large categorical vocabularies
- Learned representations capture semantic similarity (similar items end up close)
- Efficient O(1) lookup at inference time
- Sharable across multiple model components (shared embedding in DeepFM, DIN)

**Limitations**
- **Dominant memory consumer**: tables dwarf all other model parameters in production
- **Sparse gradient updates**: long-tail IDs are updated infrequently, making their embeddings noisy
- **Vocabulary drift**: new IDs appear continuously (new users, new content); tables must grow or replace stale entries
- **Hash collisions** (with hashing trick): degrade model quality at scale; collisionless alternatives add engineering complexity
- **Cold start**: new IDs have random embeddings until enough training examples accumulate; content-based initialization (from item features) partially addresses this

---

## Related Pages

- [[dlrm]] — embedding tables are DLRM's primary memory component; hybrid model/data parallelism is driven by table size
- [[two-tower-model]] — each tower's first layer is an embedding lookup; item embeddings are pre-computed and stored in ANN index
- [[monolith]] — introduces collisionless cuckoo hashmap embedding table with expiry and frequency filtering
- [[din-dien]] — applies attention-weighted pooling over user history embeddings instead of sum pooling
- [[feature-crosses]] — cross feature learning operates on the dense embeddings produced by the embedding tables

## Open Questions

- What is the optimal embedding dimension as a function of vocabulary size and training data volume (the "embedding dimension law")?
- How much quality is lost from hash collisions at various table compression ratios?
- Can pre-trained foundation model embeddings (e.g., from LLMs) replace ID-based embeddings for cold-start, and at what accuracy/latency cost?
