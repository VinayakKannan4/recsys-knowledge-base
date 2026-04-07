# RecSys Knowledge Base — Agent Schema

## Purpose
This is a personal knowledge base focused on **recommendation systems** for social media platforms and advertising. It covers ranking, retrieval, feature interaction, multi-task learning, and serving infrastructure at companies like Meta, Pinterest, Google/YouTube, TikTok/ByteDance, LinkedIn, and Snap.

## Architecture

### Three Layers (NEVER mix them)

1. **`raw/`** — Source of truth. Immutable. The agent reads from here but NEVER modifies these files.
   - `raw/papers/` — Academic papers as markdown or PDF
   - `raw/blogs/{company}/` — Engineering blog posts clipped as markdown
   - `raw/talks/` — Conference talk notes
   - `raw/code/` — Key repo READMEs, architecture docs
   - `raw/datasets/` — Dataset descriptions, benchmark info

2. **`wiki/`** — Agent-generated and agent-maintained. The human reads, the agent writes.
   - `wiki/index/` — Master index files (entry points for queries)
   - `wiki/concepts/` — Concept pages (e.g., embedding-tables.md, feature-crosses.md)
   - `wiki/architectures/` — Architecture pages (e.g., dlrm.md, dcn-v2.md, two-tower.md)
   - `wiki/companies/` — Company-specific pages (e.g., meta-ads-ranking.md)
   - `wiki/systems/` — System pages (e.g., retrieval-pipeline.md, ranking-pipeline.md)
   - `wiki/comparisons/` — Comparison pages (e.g., dlrm-vs-dcn-v2.md)

3. **`outputs/`** — Derived artifacts. Slides, charts, reports generated from the wiki.

### File Format

Every wiki page must follow this template:

# [Page Title]

**Summary**: One-sentence description of this concept/architecture/system.

**Tags**: #tag1 #tag2 #tag3

**Sources**: List of raw/ files this page was compiled from.

**Last Updated**: YYYY-MM-DD

---

## Overview
[2-3 paragraph overview]

## Key Concepts
[Detailed content]

## How It Works
[Technical details, equations if relevant]

## Industry Usage
[Which companies use this, how, scale]

## Strengths and Limitations
[Honest assessment]

## Related Pages
- [[page-name]] — brief description of relationship
- [[page-name]] — brief description of relationship

## Open Questions
[Things that are unclear or need more research]


### Naming Conventions
- All filenames: lowercase, hyphens for spaces (e.g., `two-tower-model.md`)
- All wiki links: use `[[filename]]` format (Obsidian-compatible)
- Tags: use `#category` format

### Index Files
- `wiki/index/master-index.md` — Lists ALL wiki pages with one-line summaries
- `wiki/index/papers-index.md` — Lists all papers in raw/ with key takeaways
- `wiki/index/blogs-index.md` — Lists all blog posts in raw/ with key takeaways
- `wiki/index/concepts-index.md` — Lists all concept pages
- `wiki/index/architectures-index.md` — Lists all architecture pages

### Agent Behavior Rules

1. **Compilation**: When given a new raw source, read it fully, then:
   - Add it to the relevant index file
   - Create or update concept/architecture pages it relates to
   - Add backlinks from existing pages to the new content
   - Flag any contradictions with existing wiki content

2. **Querying**: When asked a question:
   - First read the master-index.md to find relevant pages
   - Read those pages, then synthesize an answer
   - If the answer adds new insight, file it back into the wiki

3. **Linting**: When asked to lint/health-check:
   - Scan all wiki pages for inconsistencies
   - Check that all sources in raw/ are represented in index files
   - Find concepts mentioned but without their own page
   - Suggest new comparison pages or cross-references
   - Flag stale information

4. **Never**:
   - Never modify files in raw/
   - Never delete wiki pages without being asked
   - Never fabricate sources — only cite what's in raw/
   - Never remove backlinks when updating a page

## Topic Taxonomy

The knowledge base covers these main areas:

### Retrieval / Candidate Generation
- Two-tower models, embedding-based retrieval
- Deep Retrieval, approximate nearest neighbors (ANN)
- FAISS, ScaNN

### Ranking / CTR-CVR Prediction
- DLRM, DCN/DCN-V2, DeepFM, Wide & Deep
- Feature interaction learning
- Multi-task learning (MMoE, PLE, ESMM)
- Sequence modeling for user behavior (DIN, DIEN)

### Ads-Specific
- Auction mechanisms, bid optimization
- Conversion optimization, attribution
- Ad relevance and quality scoring
- Budget pacing and allocation

### Infrastructure / Serving
- Online vs batch training
- Real-time feature stores
- GPU serving for ranking
- Model compression and quantization
- Online-offline discrepancy

### Companies
- Meta (Facebook/Instagram ads)
- Pinterest (ads + organic)
- Google/YouTube
- TikTok/ByteDance
- LinkedIn
- Snap