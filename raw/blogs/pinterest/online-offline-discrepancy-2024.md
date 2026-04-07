---
source_url: https://medium.com/pinterest-engineering/handling-online-offline-discrepancy-in-pinterest-ads-ranking-system-8fd662da4c2d
title: "Handling Online-Offline Discrepancy in Pinterest Ads Ranking System | by Pinterest Engineering | Pinterest Engineering Blog | Medium"
clipped_date: 2026-04-07
type: blog
---

# Handling Online-Offline Discrepancy in Pinterest Ads Ranking System | by Pinterest Engineering | Pinterest Engineering Blog | Medium

# Handling Online-Offline Discrepancy in Pinterest Ads Ranking System

[Pinterest Engineering](/@Pinterest_Engineering?source=post_page---byline--8fd662da4c2d---------------------------------------)

13 min read

·

Jan 18, 2024

--

2

Listen

Share

Cathy Qian | Senior Machine Learning Engineer, Ads Ranking Conversion Modeling; Aayush Mudgal| Staff Machine Learning Engineer, Ads Ranking Conversion Modeling; Yinrui Li | Machine Learning II, Ads Ranking Conversion Modeling; Jinfeng Zhuang | Senior Staff Machine Learning Engineer, Ads Ranking Conversion Modeling; Shantam Shorewala | Software Engineer II, Ads ML Serving Infra; Yiran Zhao| Staff Software Engineer, Ads ML Serving Infra; Harshal Dahake, Software Engineer, Ads ML Training Infra

Press enter or click to view image in full size

Image from [https://unsplash.com/photos/w7ZyuGYNpRQ](https://unsplash.com/photos/man-in-white-long-sleeve-shirt-holding-black-dslr-camera-JxoWb7wHqnA?utm_content=creditShareLink&utm_medium=referral&utm_source=unsplash)

## Introduction

At Pinterest, our mission is to bring everyone the inspiration to create a life they love. People often come to Pinterest when they are considering what to do or buy next. Understanding this evolving user journey while balancing across multiple objectives is crucial to bring the best experience to Pinterest users and is supported by multiple recommendation models, with each providing real-time inference within a few hundreds of milliseconds for every request. In particular, our machine learning powered ads ranking systems are trying to understand users’ engagement and conversion intent and promote the right ads to the right user at the right time. Our engineers are constantly discovering new algorithms and new signals to improve the performance of our machine learning models. A typical development cycle involves offline model training to realize offline model metric gains and then online A/B experiments to quantify online metric movements. However, it is not uncommon that offline metric gains do not translate into online business metric wins. In this blog, we will focus on some online and offline discrepancies and development cycle learnings we have observed in Pinterest ads conversion models, as well as some of the key platform investments Pinterest has made to minimize such discrepancies.

## What, Why, and How

During our machine learning model iteration, we usually implement and test model improvement offline and then use a candidate model to serve online traffic in A/B experiments to measure its impact on business metrics. However, we observed that offline model performance improvement does not always translate directly into online metric gains. Specifically, such discrepancies unfold into the following scenarios:

* **Bug-free scenario**: Our ads ranking system is working bug-free. We see both offline and online metric movements, but the correlation between these movements is not clear. This is where meticulous metric design is needed.
* **Buggy scenario**: We see diminished gain or neutral results, or even online loss for models where we see offline gains, and we suspect something is not working properly in our ads ranking system. This is where investment in ML tooling helps to narrow down the problem quickly.

## Bug-free Scenario

In bug-free scenarios, we notice that it’s challenging to translate offline model performance metric gains to online business metric movements quantitatively. Figure 1 plotted the relative online business metric movement and offline model metric movement for 15 major conversion model iterations in 2023. Ten out of all the 15 iterations showed consistent directional movement between area under the ROC curve (ROC-AUC) and cost-per-acquisition (CPA). Eight of the experiments show statistically-significant movements. Quantitatively, if we make a linear plot out of all the stat-sig data points, we can see a downward trend with higher AUC increase indicating higher CPA reduction in general. However, given a specific AUC change, the predicted CPA from the linear plot would suffer from large variance and thus a lack of confidence. For example, when we see a material ROC-AUC change in our offline model iteration, we are not 100% confident that we would see statistically significant CPA reduction in online experiments, let alone the amount of CPA reduction. Such online-offline discrepancy can impact our model development velocity.

Press enter or click to view image in full size

Figure 1. Relative online business metric ( cost per acquisition, aka CPA) movement versus offline model performance metric (area under curve, aka ROC-AUC) movement between treatment models and control models. Gray triangles represent non-statistically-significant data and green spheres represent statistically-significant data.

There are a few hypotheses that can explain the observed online-offline discrepancy, as elaborated below:

A. The offline model evaluation metrics are not well aligned with the online business metrics.

ROC-AUC is one of the main metrics used for offline model evaluation. A higher ROC-AUC value indicates a higher probability of a true positive event ranked higher than a true negative event. The higher ROC-AUC, the better the binary classifier is at separating positive and negative events. However, the predicted conversion probabilities are directly used for ads ranking and bidding, whereas they are not directly related to ROC-AUC; i.e., we can have decent AUC but poor probability scores and vice versa.

On the other hand, CPA is the guardrail business metric used for online model performance evaluation, and it’s defined as:

CPA = (total revenue) / (total number of conversions)

Here, the total number of conversions Pinterest has is just an approximation. Besides, revenue calculation involves bidding and many other business logics and thus is also not straightforward. Consequently, the relationship between online CPA reduction and offline AUC gain can hardly be derived using simple mathematical formulas.

Besides, AUC and CPA are compound metrics, and their fluctuations across various traffic segments can differ, thereby increasing the difficulty in predicting their correlation. For instance, let’s suppose we have two distinct traffic segments: A and B. If the AUC declines slightly in segment A but increases a lot in segment B, and concurrently the CPA rises in segment A but declines in segment B at a similar scale, the combined effect could potentially result in an overall increase in AUC but a neutral or increasing CPA.

B. During online experiments, the control model learns from the treatment model traffic and thus minimizes the anticipated online performance gain.

When we do online experiments, the control model is usually the production model that has been serving real traffic for months if not weeks. When we do offline experiments, both the control and treatment model are trained and evaluated on the traffic that is served by the control model, aka the control traffic. During online experiments, the treatment model is serving real traffic, aka the treatment traffic. An associated concern is the potential for the control model to begin learning from the treated traffic, thereby minimizing the anticipated online benefits from the treatment model. This poses a significant challenge and calls for a precise partitioning of training and feature pipeline to accurately capture the effect. However, at Pinterest, due to the complexity of this process and the relatively minimal benefit of such bias reduction, we currently don’t take specific steps regarding this concern.

C. Complex downstream logic may dilute online gains. At Pinterest, ensemble model scores are used to calculate the downstream utility scores for bidding purposes while each model is optimized independently offline. As a result, the optimization dynamics between individual models and the utility scores may not fully align, causing a dilution of the overall online gains. We’re actively working on optimizing the downstream efficiency.

D. Certain unavoidable feature delays in online settings may result in diminished online performance gain. In feature engineering iterations, we use existing historical data to backfill new features. In online serving, these features are populated on a predetermined schedule. However, our offline backfilling timeline may not be fully aligned with the online feature population timeline.

For example, 7-day aggregated conversion counts are calculated using the past seven day conversion labels, which is fully available when we do backfilling. At serving time, assuming today’s date is dt, we would use the aggregated conversion counts between dt-7 and dt to generate this feature. However, the feature aggregation pipeline may finish at dt 3 am UTC, making this updated feature only available for serving afterwards. Serving between dt 12 am UTC and 3 am UTC would use a stale version of this feature. Such delay at an hourly granular level is unavoidable because our feature aggregation pipeline needs time to get their jobs done. Sensitivity of the treatment model to such feature freshness may result in reduced online performance gains than expected.

E. We can’t drive business metrics infinitely by optimizing machine learning models. As our models get better over time, we expect the marginal improvement in business metrics, and at some point, such gains too small to be detected in online experiments.

These are all hard challenges without easy mitigations. Fortunately, we are not the only ones to observe such online-offline discrepancies in large-scale recommendation systems. Peer efforts towards this challenge can be summarized into the two following directions:

1. Leave it as it is, but iterate fast and use offline results as a healthy check [ref 2]
2. Use multiple offline metrics instead of a single one and do compound analysis [ref 3]

Currently, the ads ranking team at Pinterest are actively investigating online-offline discrepancies in the bug-free scenario.

## Buggy Scenario

Here we have summarized general ads model flow, both involving the offline model training flow and the online serving flow.

Press enter or click to view image in full size

Figure 2: High Level Overview of Online Serving and Offline Model Training Flow

When we don’t see any online gain or even online loss for models where we see offline gain, the first thing we usually do is to check if something goes wrong in our ads ranking system. Practically, anything that may break and will break at a certain point either in the online and offline flow or both. Here we will go over some of the common failure patterns that we have observed and some of the safety measures that we have to debug and alert better.

## Data Issues

Data, including features and labels, is crucial for training machine learning models. However, various factors can corrupt this data in large-scale recommendation systems. For example: feature distribution may shift over time due to either changes in upstream pipelines or business trend, feature coverage may suddenly drop significantly due to upstream pipeline delays or failures, or labels may be delayed unexpectedly resulting in false negatives in training data.

To detect such irregularities, we have incorporated Feature Stats Validation checks into our model training pipelines as well as monitoring dashboards to capture feature coverage, freshness, and distribution over time using Apache Superset. Given that we have thousands of features and not all features are relevant to every model, we’ve also developed feature validation processes that focus on core features to specific models. This tailored approach reduces unnecessary noise in alerts and enables more efficient response and action.

Besides, the same feature may have different values during training and serving time due to asynchronous logging setup. During online serving, the Ads server makes two types of requests to the ML inference server (ranking service in Figure 2). The first batch of requests is to rank all the ad candidates using different models and a single second request to log features for certain candidates such as auction winners. In order to fulfill both these requests, the ML server fetches features from either the local cache (if available) or fetches the pre-computed value from the feature stores. This could lead to discrepancy between what was used for scoring versus what feature value is logged. The team is working on two solutions to overcome this serving-logging discrepancies. The first is to build a service that ingests and compares feature values used during the serving and logging request made to the ML server.This service can emit and highlight any value discrepancies, enabling faster detection of inconsistent data. Further, the team is working on removing the inconsistency by unifying the serving-logging path such that all score and logging requests for each candidate used the same cached value for any given feature. In addition to this, the logic behind feature backfilling could be faulty, leading to potential data leakage issues in offline experiments. Furthermore, there’s a possibility that the features extracted from the cache during the serve time might not be up-to-date.

At Pinterest, our efforts to ensure high data consistency between training and serving, unfold into the following directions:

1. In earlier model versions, each modeling use-case defines custom user defined functions (UDFs) in C++ for feature extraction. This could create a mismatch between the training UDF binary and serving binary, causing discrepancies in the served model. We’ve since transitioned to a Unified Feature Representation (flattened feature definitions), making ML features a priority in the ML lifecycle. This shift allows most of the feature pre-processing logic to either be pushed to the feature store or controlled by the model trainer, thereby reducing such discrepancies.
2. We do confidence checks on a few days of forward logged data and backfilled data to make sure they are consistent before proceeding to backfilling a few months of data.
3. We maintain a small log of features on the model server to compare with our asynchronous logging pipeline. This allows us to compare serving features with logged features and easily detect any discrepancies due to failed cache or logging logic.

Figure 3: Evolution towards Unified Feature Representation for standardization across differrent ML use-cases

During training, previous model checkpoints are retrieved from S3, and signals are fetched and processed before feeding into the algorithm for continuous model training. When the training job finishes, the trained model checkpoint is saved to the S3 bucket and later fetched during model serving. Possible issues we may encounter in the above process include:

1. Model checkpoint corruption
2. S3 access issues due to network failure or other issues
3. Model training getting stuck due to infrastructure issues, e.g. insufficient GPUs
4. Model staleness due to delayed or broken model training pipelines
5. Serving failure due to high traffic load or high serving latency

Similar to solutions for diagnosing data issues, sufficient monitoring and alerting systems are needed to allow for quick failure diagnosis and detection in model training and serving pipelines. At Pinterest, we have built easily-scalable tools in our new MLEnv training framework [ref 4] that can now be applied easily across different verticals during model offline training and online serving. Specifically, we have developed batch inference capabilities to take any model checkpoint and make inference against logged features. This allows us not only to replay traffic and check for discrepancy, but also simulate different scenarios, e.g.,missing or corrupted features, for model stability check.

Beyond the system-related data complications described above, the quality of the labels in conversion prediction contexts is often reliant on the advertisers given that conversion events take place on their platforms. We have seen cases of inaccurate conversion volumes, atypical conversion trends in data, and conversion label loss due to increasing privacy concerns. To mitigate these issues, we emphasize the detection and exclusion of outlier advertisers from both model training and evaluation. We also have a dedicated team working on improving conversion data quality; for example, optimizing the view-through checkout attribution time window for better conversion model training. We are also heavily investing in model training that is resilient against noisy and missing labels.

## Case Study

Here, we would like to share a case where we discovered significant treatment model performance deterioration during an online experiment and how we figured out the root cause.

First, we examined the model training pipeline and realized it was running smoothly. Then we evaluated the performance of the control and the treatment model on recent testing data and realized the relative difference between these metrics aligned well with our expectation.

Next, we investigated whether the treatment model generated the same predictions online and offline for the same input. At Pinterest, data used for offline model training are stored on AWS S3, while data used for online model serving are fetched from cache or feature stores. So it’s possible that discrepancies exist in these two data sources due to S3 failure or stale cache. However, our analysis showed that offline and online predictions from the treatment model were consistent on an aggregated level and thus this possibility was ruled out. Note that Pinterest relies on an asynchronous call to log features used for model serving and minor online-offline feature discrepancies may occur due to continuous feature updates but not significant ones.

Subsequently, we opted to delve into the patterns prevalent in abnormal occurrences by assessing the timeline associated with these issues. Our analysis divulged that these problems typically synchronized with bouts of peak traffic, impacting our experimental model for a duration of 2–3 minutes. In creating simulations of the features, we unearthed that the root of the issue lay in the elevated query per second (QPS) during peak traffic periods. This led our feature servers to provide ‘Null’ values for requests that could not be completed in time. This fact underscores the crucial need for production models to be sufficiently robust to resist unpredictable fluctuations and ensure that predictions don’t go awry in such outlier scenarios.

Additionally, we undertook initiatives to explore and refine techniques to evaluate and augment the robustness of novel model architectures. We gauged their performance in situations characterized by missing features and spearheaded new strategies to preclude the possibility of a data explosion.

## Conclusion

In summary, we have explored the what, why, and how of online-offline discrepancies existing within Pinterest’s large-scale ad ranking systems, both in relation to bug-free and buggy scenarios. We proposed several hypotheses/learnings that have helped us unravel the occurrence of discrepancies in a bug-free scenario, with the most important one being the discord between offline model evaluation metrics and online business metrics. We’ve additionally pinpointed potential issues that could arise in such large-scale machine learning systems and shared methods to diagnose such failure effectively. In order to provide tangible context, we’ve provided a detailed case study on a real-life Pinterest issue and demonstrated how we resolved the incident sequentially. It’s our hope that our work contributes some valuable insights towards the resolution of online-offline discrepancies in sizable machine learning applications which, in turn, could expedite the evolution of future machine learning solutions.

## Acknowledgements

This work represents a result of collaboration of the ads ranking conversion modeling team members and across multiple teams at Pinterest.

Engineering Teams:

* Ads Ranking: Han Sun, Hongda Shen, Ke Xu, Kungang Li, Matt Meng, Meng Mei, Meng Qi, Qifei Shen, Runze Su, Yiming Wang
* Ads ML Infra: Haoyang Li, Haoyu He, Joey Wang, Kartik Kapur, Matthew Jin
* Ads Data Science: Adriaan ten Kate, Lily Liu

Leadership: Behnam Rezaei, Ling Leng, Shu Zhang, Zhifang Liu, Durmus Karatay

## References

1. Learning and Evaluating Classifiers under Sample Selection Bias [Link](https://www.math.kth.se/matstat/gru/sf2935/zadrozny.pdf)
2. [150 Successful Machine Learning Models | Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining](https://dl.acm.org/doi/10.1145/3292500.3330744)
3. [Predictive model performance: offline and online evaluations](https://dl.acm.org/doi/abs/10.1145/2487575.2488215)
4. [MLEnv: Standardizing ML at Pinterest Under One ML Engine to Accelerate Innovation](/pinterest-engineering/mlenv-standardizing-ml-at-pinterest-under-one-ml-engine-to-accelerate-innovation-e2b30b2f6768)