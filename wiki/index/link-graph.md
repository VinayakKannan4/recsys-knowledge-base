# Link Graph

**Summary**: Every [[backlink]] relationship in the wiki — outgoing links (what each page references) and incoming links (what links to each page). Use this to see which pages are most central, find orphaned pages, and navigate the knowledge graph.

**Tags**: #index #graph #backlinks #navigation

**Last Updated**: 2026-04-07

---

## How to Read This

**Outgoing links** (A → B, C): page A contains `[[B]]` and `[[C]]`
**Incoming links** (X ← A, B): pages A and B contain `[[X]]`

Pages with many incoming links are **hub pages** — core concepts referenced everywhere.
Pages with few incoming links may be **entry points** or **leaves** — detailed but less cross-referenced.

---

## Outgoing Links by Page

### Architecture Pages

**[[dlrm]]** → [[deepfm]], [[dcn-v2]], [[wide-and-deep]], [[two-tower-model]], [[monolith]]

**[[dcn-v2]]** → [[wide-and-deep]], [[deepfm]], [[dlrm]], [[din-dien]], [[two-tower-model]]

**[[two-tower-model]]** → [[dlrm]], [[dcn-v2]], [[deepfm]], [[wide-and-deep]], [[monolith]]

**[[monolith]]** → [[two-tower-model]], [[dlrm]], [[din-dien]]

**[[din-dien]]** → [[wide-and-deep]], [[deepfm]], [[dcn-v2]], [[dlrm]], [[two-tower-model]], [[monolith]]

**[[wide-and-deep]]** → [[deepfm]], [[dcn-v2]], [[dlrm]], [[din-dien]], [[two-tower-model]]

**[[deepfm]]** → [[wide-and-deep]], [[dcn-v2]], [[dlrm]], [[din-dien]], [[two-tower-model]]

### Concept Pages

**[[feature-crosses]]** → [[wide-and-deep]], [[deepfm]], [[dcn-v2]], [[dlrm]], [[embedding-tables]], [[click-through-rate-prediction]]

**[[embedding-tables]]** → [[dlrm]], [[two-tower-model]], [[monolith]], [[din-dien]], [[feature-crosses]]

**[[multi-task-learning]]** → [[two-stage-pipeline]], [[click-through-rate-prediction]], [[dlrm]], [[two-tower-model]], [[wide-and-deep]]

**[[negative-sampling]]** → [[two-tower-model]], [[two-stage-pipeline]], [[click-through-rate-prediction]], [[embedding-tables]]

**[[click-through-rate-prediction]]** → [[feature-crosses]], [[two-stage-pipeline]], [[multi-task-learning]], [[negative-sampling]], [[dlrm]], [[dcn-v2]], [[deepfm]], [[din-dien]]

**[[online-learning]]** → [[monolith]], [[two-stage-pipeline]], [[embedding-tables]], [[click-through-rate-prediction]], [[dlrm]]

**[[two-stage-pipeline]]** → [[two-tower-model]], [[dlrm]], [[dcn-v2]], [[deepfm]], [[negative-sampling]], [[click-through-rate-prediction]], [[multi-task-learning]], [[monolith]]

### Company Pages

**[[meta-ads-ranking]]** → [[dlrm]], [[two-tower-model]], [[embedding-tables]], [[feature-crosses]], [[multi-task-learning]], [[negative-sampling]], [[two-stage-pipeline]], [[online-learning]], [[click-through-rate-prediction]]

**[[google-youtube-ranking]]** → [[wide-and-deep]], [[dcn-v2]], [[two-tower-model]], [[feature-crosses]], [[negative-sampling]], [[two-stage-pipeline]], [[click-through-rate-prediction]], [[multi-task-learning]], [[embedding-tables]]

**[[tiktok-bytedance-ranking]]** → [[monolith]], [[online-learning]], [[embedding-tables]], [[two-tower-model]], [[two-stage-pipeline]], [[negative-sampling]], [[multi-task-learning]], [[click-through-rate-prediction]]

**[[pinterest-ads-ranking]]** → [[two-tower-model]], [[dcn-v2]], [[multi-task-learning]], [[feature-crosses]], [[negative-sampling]], [[two-stage-pipeline]], [[click-through-rate-prediction]], [[online-learning]], [[embedding-tables]]

### Comparison Pages

**[[dlrm-vs-dcn-v2]]** → [[dlrm]], [[dcn-v2]], [[deepfm]], [[wide-and-deep]], [[feature-crosses]], [[meta-ads-ranking]], [[google-youtube-ranking]], [[pinterest-ads-ranking]]

**[[two-tower-vs-late-fusion]]** → [[two-tower-model]], [[dlrm]], [[dcn-v2]], [[din-dien]], [[two-stage-pipeline]], [[negative-sampling]], [[pinterest-ads-ranking]], [[google-youtube-ranking]], [[feature-crosses]]

**[[batch-vs-online-training]]** → [[monolith]], [[online-learning]], [[dlrm]], [[tiktok-bytedance-ranking]], [[meta-ads-ranking]], [[embedding-tables]], [[click-through-rate-prediction]]

**[[company-pipeline-comparison]]** → [[meta-ads-ranking]], [[google-youtube-ranking]], [[tiktok-bytedance-ranking]], [[pinterest-ads-ranking]], [[two-stage-pipeline]], [[two-tower-model]], [[dlrm]], [[dcn-v2]], [[monolith]], [[multi-task-learning]], [[online-learning]], [[negative-sampling]]

### Index Pages

**[[master-index]]** → [[papers-index]], [[blogs-index]], [[link-graph]], (all wiki pages)

**[[papers-index]]** → [[master-index]], [[blogs-index]]

**[[blogs-index]]** → [[master-index]], [[papers-index]]

---

## Incoming Links by Page (Backlinks)

### Architecture Pages

**[[dlrm]]** ← [[dcn-v2]], [[two-tower-model]], [[monolith]], [[din-dien]], [[wide-and-deep]], [[deepfm]], [[feature-crosses]], [[embedding-tables]], [[multi-task-learning]], [[click-through-rate-prediction]], [[online-learning]], [[two-stage-pipeline]], [[meta-ads-ranking]], [[tiktok-bytedance-ranking]] *(influenced)*, [[dlrm-vs-dcn-v2]], [[two-tower-vs-late-fusion]], [[batch-vs-online-training]], [[company-pipeline-comparison]]
*Incoming count: 18*

**[[dcn-v2]]** ← [[dlrm]], [[two-tower-model]], [[din-dien]], [[wide-and-deep]], [[deepfm]], [[feature-crosses]], [[click-through-rate-prediction]], [[two-stage-pipeline]], [[google-youtube-ranking]], [[pinterest-ads-ranking]], [[dlrm-vs-dcn-v2]], [[two-tower-vs-late-fusion]], [[company-pipeline-comparison]]
*Incoming count: 13*

**[[two-tower-model]]** ← [[dlrm]], [[dcn-v2]], [[monolith]], [[din-dien]], [[wide-and-deep]], [[deepfm]], [[embedding-tables]], [[multi-task-learning]], [[negative-sampling]], [[two-stage-pipeline]], [[meta-ads-ranking]], [[google-youtube-ranking]], [[tiktok-bytedance-ranking]], [[pinterest-ads-ranking]], [[two-tower-vs-late-fusion]], [[company-pipeline-comparison]]
*Incoming count: 16*

**[[monolith]]** ← [[dlrm]], [[two-tower-model]], [[din-dien]], [[embedding-tables]], [[online-learning]], [[two-stage-pipeline]], [[tiktok-bytedance-ranking]], [[batch-vs-online-training]], [[company-pipeline-comparison]]
*Incoming count: 9*

**[[din-dien]]** ← [[dcn-v2]], [[wide-and-deep]], [[deepfm]], [[dlrm]], [[embedding-tables]], [[click-through-rate-prediction]], [[two-tower-vs-late-fusion]]
*Incoming count: 7*

**[[wide-and-deep]]** ← [[dlrm]], [[dcn-v2]], [[din-dien]], [[deepfm]], [[feature-crosses]], [[multi-task-learning]], [[google-youtube-ranking]], [[dlrm-vs-dcn-v2]]
*Incoming count: 8*

**[[deepfm]]** ← [[dlrm]], [[dcn-v2]], [[two-tower-model]], [[din-dien]], [[wide-and-deep]], [[feature-crosses]], [[click-through-rate-prediction]], [[two-stage-pipeline]], [[dlrm-vs-dcn-v2]]
*Incoming count: 9*

### Concept Pages

**[[feature-crosses]]** ← [[embedding-tables]], [[click-through-rate-prediction]], [[meta-ads-ranking]], [[google-youtube-ranking]], [[pinterest-ads-ranking]], [[dlrm-vs-dcn-v2]], [[two-tower-vs-late-fusion]]
*Incoming count: 7*

**[[embedding-tables]]** ← [[feature-crosses]], [[negative-sampling]], [[online-learning]], [[meta-ads-ranking]], [[google-youtube-ranking]], [[tiktok-bytedance-ranking]], [[pinterest-ads-ranking]], [[batch-vs-online-training]]
*Incoming count: 8*

**[[multi-task-learning]]** ← [[click-through-rate-prediction]], [[two-stage-pipeline]], [[meta-ads-ranking]], [[google-youtube-ranking]], [[tiktok-bytedance-ranking]], [[pinterest-ads-ranking]], [[company-pipeline-comparison]]
*Incoming count: 7*

**[[negative-sampling]]** ← [[click-through-rate-prediction]], [[two-stage-pipeline]], [[meta-ads-ranking]], [[google-youtube-ranking]], [[tiktok-bytedance-ranking]], [[pinterest-ads-ranking]], [[two-tower-vs-late-fusion]], [[company-pipeline-comparison]]
*Incoming count: 8*

**[[click-through-rate-prediction]]** ← [[feature-crosses]], [[negative-sampling]], [[online-learning]], [[two-stage-pipeline]], [[meta-ads-ranking]], [[google-youtube-ranking]], [[tiktok-bytedance-ranking]], [[pinterest-ads-ranking]], [[batch-vs-online-training]]
*Incoming count: 9*

**[[online-learning]]** ← [[meta-ads-ranking]], [[tiktok-bytedance-ranking]], [[pinterest-ads-ranking]], [[batch-vs-online-training]], [[company-pipeline-comparison]]
*Incoming count: 5*

**[[two-stage-pipeline]]** ← [[multi-task-learning]], [[negative-sampling]], [[online-learning]], [[meta-ads-ranking]], [[google-youtube-ranking]], [[tiktok-bytedance-ranking]], [[pinterest-ads-ranking]], [[two-tower-vs-late-fusion]], [[company-pipeline-comparison]]
*Incoming count: 9*

### Company Pages

**[[meta-ads-ranking]]** ← [[dlrm-vs-dcn-v2]], [[batch-vs-online-training]], [[company-pipeline-comparison]]
*Incoming count: 3*

**[[google-youtube-ranking]]** ← [[dlrm-vs-dcn-v2]], [[two-tower-vs-late-fusion]], [[company-pipeline-comparison]]
*Incoming count: 3*

**[[tiktok-bytedance-ranking]]** ← [[batch-vs-online-training]], [[company-pipeline-comparison]]
*Incoming count: 2*

**[[pinterest-ads-ranking]]** ← [[dlrm-vs-dcn-v2]], [[two-tower-vs-late-fusion]], [[company-pipeline-comparison]]
*Incoming count: 3*

### Comparison Pages

**[[dlrm-vs-dcn-v2]]** ← [[master-index]], [[dlrm]], [[dcn-v2]]
*Incoming count: 3*

**[[two-tower-vs-late-fusion]]** ← [[master-index]], [[two-tower-model]], [[pinterest-ads-ranking]]
*Incoming count: 3*

**[[batch-vs-online-training]]** ← [[master-index]], [[monolith]], [[meta-ads-ranking]], [[tiktok-bytedance-ranking]]
*Incoming count: 4*

**[[company-pipeline-comparison]]** ← [[master-index]], [[meta-ads-ranking]], [[google-youtube-ranking]], [[tiktok-bytedance-ranking]], [[pinterest-ads-ranking]]
*Incoming count: 5*

### Index Pages

**[[papers-index]]** ← [[master-index]], [[blogs-index]]

**[[blogs-index]]** ← [[master-index]], [[papers-index]]

**[[link-graph]]** ← [[master-index]]

---

## Hub Pages (Most Incoming Links)

Ranked by incoming link count — these are the most central concepts in the knowledge base:

| Rank | Page | Incoming Links | Category |
|------|------|---------------|----------|
| 1 | [[dlrm]] | 18 | Architecture |
| 2 | [[two-tower-model]] | 16 | Architecture |
| 3 | [[dcn-v2]] | 13 | Architecture |
| 4 | [[click-through-rate-prediction]] | 9 | Concept |
| 4 | [[two-stage-pipeline]] | 9 | Concept |
| 4 | [[monolith]] | 9 | Concept |
| 4 | [[deepfm]] | 9 | Architecture |
| 8 | [[embedding-tables]] | 8 | Concept |
| 8 | [[negative-sampling]] | 8 | Concept |
| 8 | [[wide-and-deep]] | 8 | Architecture |
| 11 | [[feature-crosses]] | 7 | Concept |
| 11 | [[din-dien]] | 7 | Architecture |
| 11 | [[multi-task-learning]] | 7 | Concept |
| 14 | [[online-learning]] | 5 | Concept |
| 15 | [[meta-ads-ranking]] | 3 | Company |
| 15 | [[google-youtube-ranking]] | 3 | Company |
| 15 | [[pinterest-ads-ranking]] | 3 | Company |
| 18 | [[tiktok-bytedance-ranking]] | 2 | Company |
| 19 | [[company-pipeline-comparison]] | 5 | Comparison |
| 20 | [[batch-vs-online-training]] | 4 | Comparison |
| 21 | [[dlrm-vs-dcn-v2]] | 3 | Comparison |
| 21 | [[two-tower-vs-late-fusion]] | 3 | Comparison |

**Observations**:
- `dlrm`, `two-tower-model`, and `dcn-v2` are the most-referenced architecture pages — every other architecture and concept links back to them
- `click-through-rate-prediction` and `two-stage-pipeline` are the most-referenced concept pages — they appear in every company and comparison page
- Company pages (`meta-ads-ranking`, etc.) are relatively leaf nodes — they receive links primarily from comparison pages, not from concept or architecture pages
- Comparison pages have only 1 incoming link each (from master-index) — they are designed as entry points, not cross-referenced hubs

---

## Pages with No Incoming Links (Potential Orphans)

No true orphans exist in the current wiki. All comparison pages are reachable from both [[master-index]] and their subject architecture/company pages. The lowest-linked pages are:

- [[tiktok-bytedance-ranking]] — 2 incoming (comparison pages only); consider adding a reference from [[monolith]] body text
- [[online-learning]] — 5 incoming; reasonably connected but could be referenced from [[two-stage-pipeline]] body text

---

## Related Pages

- [[master-index]] — full table of contents
- [[papers-index]] — academic paper catalog
- [[blogs-index]] — blog post catalog
