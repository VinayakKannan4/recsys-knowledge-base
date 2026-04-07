---
source_url: https://medium.com/pinterest-engineering/evolution-of-ads-conversion-optimization-models-at-pinterest-84b244043d51
title: "Evolution of Ads Conversion Optimization Models at Pinterest | by Pinterest Engineering | Pinterest Engineering Blog | Medium"
clipped_date: 2026-04-07
type: blog
---

# Evolution of Ads Conversion Optimization Models at Pinterest | by Pinterest Engineering | Pinterest Engineering Blog | Medium

# Evolution of Ads Conversion Optimization Models at Pinterest

[Pinterest Engineering](/@Pinterest_Engineering?source=post_page---byline--84b244043d51---------------------------------------)

11 min read

·

Jan 9, 2024

--

Listen

Share

A Journey from GBDT to Multi-Task Ensemble DNN

Aayush Mudgal | Staff Machine Learning Engineer, Ads Ranking Conversion Modeling; Han Sun | Staff Machine Learning Engineer, Ads Ranking Conversion Modeling; Ke Xu | Senior Machine Learning Engineer, Ads Ranking Conversion Modeling;Matt Meng (He Him) | Senior Machine Learning Engineer, Ads Ranking Conversion Modeling; Runze Su | Senior Machine Learning Engineer, Ads Ranking Conversion Modeling Jinfeng Zhuang (He Him) | Senior Staff Machine Learning Engineer, Ads Ranking Conversion Modeling

In this blog post, we will share how we improved Pinterest’s conversion optimization performance by leveraging Deep Neural Networks (DNN), Multi-Task Learning (MTL), state-of-the-art feature interaction modules, in-model ensemble techniques, and user sequence modeling. We will also cover our transition to GPU serving, which is indispensable for large-scale complex models.

## Background

People often come to Pinterest for inspiration on their next life or shopping ideas. In fact, over half of Pinterest users think of Pinterest as a place to shop (Source: Pinterest Internal Data, June 2023). Ads and other content from businesses are an important part of the consumer journey because they add value to the discovery process. **Conversion Optimization** optimizes **Promoted Pins** for specific consumer conversion actions, rather than just clicks. Advertisers can choose conversions as a campaign objective and inspire people to take specific actions like online checkouts, increased signups, or stronger leads. In this blog we will deep dive into some of our recent advancements in machine learning modeling to connect pinners with the most relevant ads.

The Ads Ranking system at Pinterest has many standard pieces of a recommendation system involving targeting, retrieval, ranking, auction, and allocation, as it needs to be responsive to 498 million monthly active users actions and performance and billions of pieces of content on the platform. The ranking layer focuses on finding the relevant Pins given the user context, so improving this part of the system has a significant impact on the user experiences. This blog post will focus on our recent improvements on improving the models that predict offsite conversions given a user request on Pinterest. The following is a very high level overview of how the conversion optimization system typically works.

Press enter or click to view image in full size

Figure 1: A high level overview of conversion optimization pipeline (For illustrative purposes only)

Before diving deeper into the model evolution, let’s also look into the unique characteristics of Ads conversion prediction, which is very different from typical engagement ranking problems.

1. **Label quality**: Since conversion events happen on advertiser platforms, the quality of the label is dependent on the advertisers, so we often deal with inaccurate and abnormal conversion volumes. Besides, the user match and attribution process is probabilistic which further introduces noise in the labels.
2. **Data volume and label sparsity**: Conversion is a more downstream event, and is thus inherently much sparser than click events on the platform. This significantly limits the complexity of the model we can build. Since the label is sparser, it also requires higher traffic percentages or longer running A/B experiments to detect significant changes in metrics, slowing down the iteration speed.
3. **Delayed feedback**: Conversions are observed with a long-tail delay distribution after the onsite engagements (clicks or views) have happened, resulting in false negative labels during training. This brings challenges on the model training strategy, e.g., the model’s update frequency, and complicates calibration estimations of the learned models.

## Classic Machine Learning Model Period

In the beginning (2018), our ads ranking model was a hybrid of Gradient Boosted Decision Trees (GBDT), embedded as feature translators, and a logistic regression model. This design choice enabled us to build performant models quickly for the scale of data and machine learning stack of that time. Ad space, comprising advertisers, ad groups, and ad campaigns have very different label characteristics both in terms of quality and volume, hence generalization across the sparse ID space was top consideration. To enable this we utilized the [feature hashing trick](https://en.wikipedia.org/wiki/Feature_hashing) into the logistic regression layer to learn the ID space information and control cross-advertiser label impacts.

As the product **grew rapidly, fueled by increasing amounts of data**, it required better scalability in model training and serving, as the previous design became the bottleneck for fast system growth. The next set of iterations happened by transitioning from this GBDT + logistic regression structure to a deep learning based single model, and also unlocking the ability to do Multi-Task Learning (MTL) by co-training multiple objectives together like clicks, good clicks, checkout, and add-to-cart conversions. Moving to MTL and leveraging onsite actions helped increase model robustness to sparser tasks and unleashed the power to learn from the holistic onsite and offsite experience. By 2020, we transitioned to AutoML ([link](/pinterest-engineering/how-we-use-automl-multi-task-learning-and-multi-tower-models-for-pinterest-ads-db966c3dc99e)), which was a shift from previous traditional machine learning approaches by introducing an automatic way of feature engineering. The engineers could work on developing raw features instead of hand crafting the complex feature interactions. However, over this period of time, we encountered significant technical debt as we were moving fast to bring technical advancements by patching over existing systems to bring the fastest results. Between 2021–2022 we accomplished a complete overhaul of the entire ML Platform ([link](/pinterest-engineering/mlenv-standardizing-ml-at-pinterest-under-one-ml-engine-to-accelerate-innovation-e2b30b2f6768)), also leveraging flattened feature structures and adopting more sampling near the model trainer. This infrastructure migration enabled us to have a much more robust backbone for a substantially wider range of modeling ideas. Next we will discuss some of the recent architectural advancements and the lessons that help power these models today.

## Modern Model Architectural Advancement

With the transition to AutoML (auto feature interaction) and the MLEnv (Unified ML Engine) framework, we are unblocked with **much faster model architecture iteration speed**. Since then, we have experimented, iterated, and launched a number of state-of-the-art modern model architectures. Here is our learning summary of the architecture evolution.

### Feature Interaction Modules

A very important component in modern recommendation systems is the feature interaction learning modules. Starting from MLP, our model advancement iterates along the following three modules, as elaborated below:

Figure 2: Different feature interaction modules

**DCNv2:**

[DCNv2](https://arxiv.org/abs/2008.13535), also known as Deep & Cross Network version 2, is a step forward from the original DCN model. It introduces a cross network and multiple deep layers to capture both explicit and implicit feature interactions. With DCNv2, explicit low-ordered feature interactions are captured through a low-dimensional cross network with a feature crossing matrix, and other implicit feature interactions are captured in a deep network, which is similar to the MLP layers. It outperforms the MLP layers while there’s a larger infrastructure cost due to the introduction of the cross network.

**Transformer:**

[Transformer](https://arxiv.org/abs/1706.03762) is originally proposed for natural language processing tasks, but it also shows great potential in learning feature crossing in the advertising field. The self-attention mechanism in the transformer encoder maps the input into Q(Query), K(Key) and V(Value), where Q and K capture the feature interaction and cast the interaction into V. Transformer has demonstrated its performance by significantly improving the model performance, but it also resulted in a high memory usage during training and relatively higher latency.

**MaskNet:**

[MaskNet](https://arxiv.org/abs/2102.07619) proposes an instance-guided mask method which uses element-wise products in both the feature embedding layer and the feedforward layer in DNN. The instance-guided mask contains global contextual information that is dynamically incorporated into the feature embedding layer and the feedforward layer to highlight the important features.

MaskBlock is the module for feature interaction learning. It consists of three parts: the instance-guided mask, the feedforward layer, and the normalization layer. With this structure, the standard DNN is extended to contain interactive feature structures that can be added and multiplied. MaskNet is one of the most popular modules in advertising and recommendation systems and is of high performance in feature crossing modules.

### Modern Model Multi-tasking and Ensemble Learning

**Multi-Task Learning (MTL) and in-model ensembling frameworks** have been pushing recommendation model performance limits in recent years. With many powerful state-of-the-art feature interaction algorithms’ (above discussed, e.g., transformer feature interaction, masknet feature interaction, etc). We embraced all their powers together and explored the following directions

We introduced **MTL** to combine multiple conversion objectives into a unified model by leveraging abundant onsite actions like clicks on the platform to enhance the training of sparse conversion objectives. This not only benefited each conversion objective but also largely reduced model serving and maintenance cost, as well as improved experimentation iteration velocity.

Figure 3: Architectural evolution from single task to a shared multi-task model architecture

In the next iteration, we made the model more robust and introduced in-model ensemble techniques, where we **ensembled two model backbones for feature crossing: DCNv2 and transformer respectively.** We curated an empirical formula for their output scores at inference to compute the final model prediction, while we maintained individual training loss and gradient descent for each ensembled model. This curated architecture was able to benefit from each model backbone’s diverse learning.

Figure 4: Ensemble model structure with DCNv2 and Transformer as feature interaction modules

The above architecture significantly increased the serving infrastructure cost. Next we decoupled feature interaction modules from the feature processing modules and utilized a **shared bottom architecture for feature processing** while maintaining the top architecture separated. This evolution significantly reduced infra cost of the model while maintaining same level of performance after parameter tunings

Figure 5: Model architecture that leverages shared feature processing layer before the ensemble feature interaction learning

Realizing the power of learning in-model ensemble, we wanted to next move to a principled way of doing the same. We adopted the [**Deep and Hierarchical Ensemble Network**](https://arxiv.org/abs/2203.11014) **(DHEN) framework with user sequence and MaskNet feature interaction**. This modeling framework is scalable, optimizable, and modularizable and achieved the best offline and online model performance compared to each of the previous architectural choices.

Figure 6: Model architecture with Sequence Transformer addition

### User Sequence Modeling

The capability to model user activity sequence is a powerful modern modeling technique. We have explored extensively in the area of user sequence modeling and launched it in our conversion ranking model with significant offline and online performance gains. There are several advantage of including user sequences into conversion models:

* With a **long lookback window**, user sequence is extremely effective to capture information from sparse data space — in our particular case, user and Ads interactions. It is even more powerful for Ads user conversion optimizations, given its sparsity and delaying nature compared to user engagement optimizations.
* A **direct model training with user sequence data** in conversion ranking models unleashes its power in learning user interest from user activities — from user offsite events that are directly related to offsite conversions, to user onsite engagements that represents user intends.
* The essence part of user sequence data is its **temporal information**. The incorporation of temporal dimensionality to ranking models allow the rich feature interaction modules to learn news-level, seasonal, and life-time user interest shifts and patterns. This information is vitally important serving as the backbone of user representation in the ranking models.

## Evaluation Results

Together with other improvements, such as additional features to feed the advanced model architectures, our above iterations have achieved a significant amount of both offline and online metrics gains and we summarized the relative performance as below.

Source: Pinterest Internal Data, Global, December 2023

## GPU Serving

Our model ensemble and feature interaction components have made significant advancements; however, serving these models has become increasingly complex. We initially used CPU clusters to serve these ranking models, but this restricted the model capacity we could use whilst still maintaining low latency. To serve these big models with minimal cost increments, we also transitioned to using GPUs for online serving.

### CUDA Graphs

With the migration of the ranking models to [Pinterest’s MLEnv and PyTorch based framework](/pinterest-engineering/mlenv-standardizing-ml-at-pinterest-under-one-ml-engine-to-accelerate-innovation-e2b30b2f6768) , we were able to easily leverage [CUDA graphs](https://pytorch.org/docs/master/notes/cuda.html#constraints) for online serving. The traditional launch of CPU kernels creates gaps and inadvertent overhead between launches. CUDA graphs greatly reduce these gaps, optimizing subsequent executions and minimizing delays. Using CUDA graphs does present certain constraints. For example, the network must be graph-safe. A crucial condition of using CUDA graphs is the need for tensor shapes in memory to be static — dynamic inputs with varying tensor dimensions cause numerical errors in CUDA graphs. We used zero padding to ensure constant feature dimensions for ragged and other dynamically-shaped tensors.

### Mixed Precision Serving

To further cut down on the infrastructure costs of model serving, we implemented the technique of mixed precision serving. This technique is prevalent in large model training and inferencing; it merges the use of 32-bit floating point (FP32) and 16-bit floating point (FP16) data types. The end result is a reduction in memory usage and a boost in the speed of model development and inference, all without compromising the accuracy of the model’s predictions. For an in-depth understanding, refer to Nvidia’s [blog post](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html). We applied half-precision to modules with high computational demands, particularly those that included feature crossing layers, the ensemble layer, feature projection layers, and sequential processing layers.

Press enter or click to view image in full size

Figure 7: Integration of Mixed Precision Wrapper with the serving model artifact

Based on our internal testing, utilizing mixed precision serving resulted in an increase in throughput and decrease in forward computation time.

Conclusion:

Optimizing for conversion brings its unique challenges and are non-trivial to handle. Our work around incorporating Multi-Task Learning, moving to in-model ensemble techniques, and leveraging real time user action signals has greatly improved ads recommendation systems. Since there is a tradeoff between model capacity and infrastructure costs, requiring GPU serving and optimization indispensable for large scale, complex models.

## Acknowledgements

This work represents a result of collaboration of the conversion modeling team members and across multiple teams at Pinterest.

Engineering Teams:

Ads Ranking: Kungang Li, Meng Qi, Meng Mei, Yinrui Li, Zhixuan Shao, Zhifang Liu, Liangzhe Chen, Yulin Lei Kaili Zhang Qifei Shen

Advanced Technology Group: Yi-Ping Hsu, Pong Eksombatchai, Xiangyi Chen

Ads ML Infra: Shantam Shorewala, Kartik Kapur, Matthew Jin, Haoyu He, Nuo Dou, Yiran Zhao, Joey Wang, Haoyang Li

User Sequence Support: Kangnan Li, Zefan Fu, Kimmie Hua

Leadership: Ling Leng, Shu Zhang, Jiajing Xu, Xiaofang Chen, Behnam Rezaei

## References

1. Wang, Zhiqiang, Qingyun She, and Junlin Zhang. “MaskNet: Introducing feature-wise multiplication to CTR ranking models by instance-guided mask.” arXiv preprint arXiv:2102.07619 (2021).
2. Zhang, Buyun, et al. “DHEN: A deep and hierarchical ensemble network for large-scale click-through rate prediction.” arXiv preprint arXiv:2203.11014 (2022).
3. Wang, Ruoxi, et al. “Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems.” Proceedings of the web conference 2021. 2021.
4. MLEnv: Standardizing ML at Pinterest Under One ML Engine to Accelerate Innovation [[Link](/pinterest-engineering/mlenv-standardizing-ml-at-pinterest-under-one-ml-engine-to-accelerate-innovation-e2b30b2f6768)]
5. How we use AutoML, Multi-task learning and Multi-tower models for Pinterest Ads [[Link](/pinterest-engineering/how-we-use-automl-multi-task-learning-and-multi-tower-models-for-pinterest-ads-db966c3dc99e)]
6. Feature Hashing Trick: [[Link](https://en.wikipedia.org/wiki/Feature_hashing)]
7. CUDA Semantics: [[Link](https://pytorch.org/docs/master/notes/cuda.html#constraints)]
8. Mixed Precision Training [[Link](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)]

*To learn more about engineering at Pinterest, check out the rest of our* [*Engineering Blog*](https://medium.com/pinterest-engineering) *and visit our* [*Pinterest Labs site*](https://www.pinterestlabs.com/?utm_source=Medium&utm_campaign=engineering-Q12024_Mudgal&utm_medium=blogarticle)*. To explore and apply to open roles, visit our* [*Careers*](https://www.pinterestcareers.com/?utm_source=Medium&utm_campaign=engineering-Q12024_Mudgal&utm_medium=blogarticle) *page.*