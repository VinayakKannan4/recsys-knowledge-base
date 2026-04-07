---
source_url: https://medium.com/pinterest-engineering/contextual-relevance-in-ads-ranking-63c2ff215aa2
title: "Contextual relevance in ads ranking | by Pinterest Engineering | Pinterest Engineering Blog | Medium"
clipped_date: 2026-04-07
type: blog
---

# Contextual relevance in ads ranking | by Pinterest Engineering | Pinterest Engineering Blog | Medium

# Contextual relevance in ads ranking

[Pinterest Engineering](/@Pinterest_Engineering?source=post_page---byline--63c2ff215aa2---------------------------------------)

5 min read

·

Apr 9, 2020

--

1

Listen

Share

Mihai Roman | Software Engineer, Ads Relevance

Pinterest runs an ads marketplace that balances value for Pinners, partners, and the business.

We initially used engagement prediction as our main method to balance the marketplace before eventually adding contextual relevance to increase Pinner value. We have three main surfaces for determining contextual relevance: user personalization, search queries, and closeups recommendations.

We removed a series of bottlenecks over the last year to bring relevance modeling to a robust production stage. We’ll go over those improvements below.

## Measurements

Measuring contextual relevance for long-term trends and experimentation was our first hurdle. We often observed high variance in relative results when comparing two treatments (e.g., three consecutive measurements resulted in relative changes of -5%, +5%, and +15%, respectively). While we generally need both [precise and accurate](https://en.wikipedia.org/wiki/Accuracy_and_precision) measurements, we settled for precision to get started since it’s the prerequisite for running A/B experiments.

To improve the user experience, we only needed confidence in relative measurements for production vs. holdout and control vs. experiment treatment. We achieved this by controlling for platform variance and running extensive A/A tests for various query sets. We run trend metrics on randomly sampled queries, scraped daily with fixed user contexts. We use metrics like fraction of relevant as well as ranking metrics derived from [nDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain).

We measure realized relevance by scraping production and holdout settings, which allows us to observe seasonal effects and impact from new launches. We keep a stable ~20% gap on both search and related pins surfaces:

Press enter or click to view image in full size

## Guidelines

To define relevance, we use a human evaluation template with a 5-point scale and explicit guidelines. Labels are collected with dedicated crowdsourcing teams. For quality control, we track metrics such as row-level disagreement rate, which hovers below 10% for regular dataset jobs.

We rely on user research and dissatisfaction reports to catch blind spots and improve predictive models, and we analyze ads inventory for seasonality effects.

\* mocks for Search and Related Pins human labeling templates

## Models

Legacy prototype models were trained on separate content types (shopping, video, regular ads, etc.), which posed a maintenance issue and limited our ability to design a sophisticated marketplace. Even when building single models, calibrating them on content type was challenging because of differences in impression ratios and feature coverage.

For example, on shopping content, we were missing a category matching feature that did a good job separating the 3-point scale labels:

Press enter or click to view image in full size

By adding it, we increased the fraction of relevant shopping ads by +4.25% and click rate by 1.45%. It also had the effect of smoothing out the calibration curve:

Press enter or click to view image in full size

Old model on the left w/ irregular calibration curve corresponding to missing features

When training on ads impressions, bias is introduced from both the production relevance model and the marketplace dynamics. For example, content type and relevance distributions change dramatically from the candidates level to the impressions level in the serving funnel. For engagement prediction models, it’s harder to collect counterfactual labels, build dedicated calibration models that learn from their own impressions, or implement reinforcement learning techniques. For relevance, since we’re training the models on human labels and don’t use many interaction features, the bottlenecks are the labeling budgets and the candidate-level sampling infrastructure.

The differences across content types (e.g. videos, carousels, or shopping) can cause miscalibration even when feature coverage is similar. For example, two shopping ads, one with a lifestyle image and one with a product image, will fare differently on an image similarity feature despite pointing to the same product. You can read more about signals we use, such as [visual embeddings](/pinterest-engineering/unifying-visual-embeddings-for-visual-search-at-pinterest-74ea7ea103f0) and [graph embeddings](/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48).

As for how we use the models, we first apply a classification model in retrieval that implicitly controls for ad load and trims the most irrelevant ads. In auction, we originally used a [learn to rank](https://en.wikipedia.org/wiki/Learning_to_rank) model with good properties on the relative distance between candidates. We’re replacing them with regression models, tuned to maintain similar [ranking metrics](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) yet provide better interpretability. It’s important to provide calibrated, interpretable scores so we can ship new models without tuning the ranking function that combines expected revenue, engagement, and relevance.

## Infrastructure

Our stack consists of three main services: retrieval, model scoring, and auction. The first models were trained on an imperfect sampling heuristic based on scraping post-auction ads from zero-relevance holdout settings. To correct the inherent bias, we invested in sampling from candidates throughout the retrieval funnel.

In the lower funnel, the relevance model evaluated in the retrieval service trims irrelevant ads. Traditionally, we don’t log these candidates because of costs, but since for relevance we’re only limited by labeling budget, we log a 1% candidates sample that we further sample down for continuous labeling. We keep track of the stage a candidate is dropped in the funnel so we can train/test/validate models at each stage.

As we go up the stack, the ads ranking function changes the candidate distribution w.r.t. relevance. As such, we score a dedicated model in a separate service that is optimized for large models and fewer candidates.

The chart below shows the fraction of relevant ads through the different stages in the funnel across different content categories. Since relevance is positively correlated with other ranking components like revenue estimation and engagement, we can see how it increases before auction, then decreases in winning candidates as a result of marketplace competition.

Press enter or click to view image in full size

relevance score through the funnel, for different categories of ads

In production, we monitor metrics such as feature coverage, score distributions, and trim rates by surface, content type and model:

Press enter or click to view image in full size

## Conclusion

Building a robust machine learning relevance stack has allowed us to drastically improve our user experience. Our team is growing, so check our [careers page](https://www.pinterestcareers.com/homepage) if you’d like to contribute to efforts like this!

## Acknowledgements

**Engineering Team**: Josh Kelle, Benjamin Weitz, Tarun Kumar, Satyajit Gupte, Mihai Roman

**Product Management**: Sayantan Mukhopadhyay

**Data Science**: Holly Capell, Ashim Datta

**Leadership**: Jiajing Xu, Hari Venkatesan, Roelof van Zwol