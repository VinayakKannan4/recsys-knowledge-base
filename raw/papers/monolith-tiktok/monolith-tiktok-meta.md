---
source_url: https://arxiv.org/abs/2209.07663
pdf_path: raw/papers/monolith-tiktok/monolith-tiktok.pdf
arxiv_id: 2209.07663
title: "Monolith: Real Time Recommendation System With Collisionless Embedding Table"
authors: "Zhuoran Liu, Leqi Zou, Xuan Zou, Caihua Wang, Biao Zhang, Da Tang, Bolin Zhu, Yijie Zhu, Peng Wu, Ke Wang, Youlong Cheng"
downloaded_date: 2026-04-07
type: paper
---

# Monolith: Real Time Recommendation System With Collisionless Embedding Table

**Authors**: Zhuoran Liu, Leqi Zou, Xuan Zou, Caihua Wang, Biao Zhang, Da Tang, Bolin Zhu, Yijie Zhu, Peng Wu, Ke Wang, Youlong Cheng

**arXiv**: [2209.07663](https://arxiv.org/abs/2209.07663)

## Abstract

Building a scalable and real-time recommendation system is vital for many businesses driven by time-sensitive customer feedback, such as short-videos ranking or online ads. Despite the ubiquitous adoption of production-scale deep learning frameworks like TensorFlow or PyTorch, these general-purpose frameworks fall short of business demands in recommendation scenarios for various reasons: on one hand, tweaking systems based on static parameters and dense computations for recommendation with dynamic and sparse features is detrimental to model quality; on the other hand, such frameworks are designed with batch-training stage and serving stage completely separated, preventing the model from interacting with customer feedback in real-time. These issues led us to reexamine traditional approaches and explore radically different design choices. In this paper, we present Monolith, a system tailored for online training. Our design has been driven by observations of our application workloads and production environment that reflects a marked departure from other recommendations systems. Our contributions are manifold: first, we crafted a collisionless embedding table with optimizations such as expirable embeddings and frequency filtering to reduce its memory footprint; second, we provide an production-ready online training architecture with high fault-tolerance; finally, we proved that system reliability could be traded-off for real-time learning. Monolith has successfully landed in the BytePlus Recommend product.

## Key Takeaways

*(To be filled by the LLM agent during compilation)*

## Notes

*(To be filled by the LLM agent during compilation)*
