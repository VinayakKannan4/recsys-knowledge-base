---
source_url: https://www.infoq.com/articles/pinterest-ad-ranking-ai/
title: "Unpacking How Ad Ranking Works at Pinterest - InfoQ"
clipped_date: 2026-04-07
type: blog
---

# Unpacking How Ad Ranking Works at Pinterest - InfoQ

[InfoQ Homepage](/ "InfoQ Homepage")
[Articles](/articles "Articles")
Unpacking How Ad Ranking Works at Pinterest

[AI, ML & Data Engineering](/ai-ml-data-eng/ "AI, ML & Data Engineering")

[Designing Data Layers for Agentic AI: Patterns for State, Memory, and Coordination at Scale (Webinar May 12th)](https://www.infoq.com/url/t/95243370-17c6-4296-86ff-b50308bd7ada/?label=YugabyteDB-EventPromoBox )

# Unpacking How Ad Ranking Works at Pinterest

This item in
[japanese](/jp/articles/pinterest-ad-ranking-ai)

Mar 26, 2024
18
min read

by

* [Anthony Alford](/profile/Anthony-Alford/)

#### Write for InfoQ

**Feed your curiosity.**
Help 550k+ global   
senior developers   
each month stay ahead.[Get in touch](https://www.infoq.com/write-for-infoq/)

Like

* [Reading list](/showbookmarks.action)

### Key Takeaways

* Deep learning based machine learning algorithms are leveraged for responsive and personalized ad recommendations.
* The advertising platform objective is to maximize value for users, advertisers, and the platform in the long term
* Ads Delivery Funnel comprises candidate retrieval, heavyweight ranking, auction, and allocation to ensure low latency serving at high QPS
* Over time, Pinterest has evolved its machine learning models from traditional approaches to more complex ones, such as deep neural networks (DNNs) and transformer architectures, which boost personalization
* Robust MLOps practices such as including continuous integration and deployment (CI/CD), model versioning, testing, and monitoring, are crucial to iterating fast and effectively.

Aayush Mudgal, Staff Machine Learning Engineer at Pinterest, presented at QCon San Francisco 2023 a session on [Unpacking how Ads Ranking Works at Pinterest](https://qconsf.com/presentation/oct2023/unpacking-how-ads-ranking-works-pinterest). In it he walked through how Pinterest uses deep learning and big data to tailor relevant advertisements to their users.

As with most online platforms, personalized experience is at the heart of [Pinterest](https://www.pinterest.com/). This personalized experience is powered through a variety of different machine learning (ML) applications. Each of them is trying to learn complex web patterns from large-scale data collected by the platform.

In his talk, Mudgal focused on one part of the experience: serving advertisements. He discussed in detail how Machine Learning is used to serve ads at large scale. He then went over ads marketplaces and the ad delivery funnel and talked about the typical parts of the ad serving architecture, and went into two of the main problems: ads retrieval and ranking. Finally, he discussed how to monitor the system health during model training and wrapped up with some of the challenges and solutions for large model serving.

## Content Recommendation

#### Related Sponsors

* ##### [The missing layer in the agentic AI stack: Why AI applications need durable sessions](/url/f/b53053d5-b897-4c93-b36b-4c0ab4acdaaf/)

#### Related Sponsor

**Drop in Ably AI Transport.**  
Purpose-built infrastructure for the entire agent-to-user experience. **[Start building](/url/f/cc7950b0-9ce8-48cd-a235-0a77f40b217d/).**

Mudgal first presented the characteristics of a content recommendation system. Every social media platform has millions or billions of content items that it could potentially show to users. The goal is to find items that are relevant to a particular user, but since the content catalog and user base are so large, a platform like Pinterest cannot precompute the relevance probability of each content item for each user.

Instead, the platform needs a system that can predict this probability quickly: within hundreds of milliseconds. It must also handle high queries-per-second (QPS). Finally, it needs to be responsive to users’ changing interests over time. To capture all of these nuances, platforms need to make sure that the recommendation system solves a multi-objective optimization problem.

When a user interacts with a particular element on a platform, they are often presented with a variety of similar content. This is a crucial moment where targeted advertisements can come into play. These ads aim to bridge the gap between users' and advertisers’ content within the platform. The goal is to engage users with relevant content which can potentially lead them from the platform to the advertiser's website.

This is a two-sided marketplace. Advertising platforms like Pinterest, Meta, Google help to connect users with advertisers and the relevant content. Users visit the platform to engage with the content. Advertisers pay these advertising platforms so that they can show their content so that users engage with it. Platforms want to maximize the value for the users, the advertisers, and the platform.

## Advertising Marketplaces

Advertisers want to have their content shown to the users. It could be as simple as creating an awareness for that brand, or driving more clicks on-site on the platform. When they do this, the advertisers can also choose how much they value a particular ad shown on the platform.

Advertisers have the option to select from two main bidding strategies. One approach allows advertisers to pay a predetermined amount for each impression or interaction generated via the platform. Alternatively, they can set a defined budget and rely on the platform's algorithms to distribute it optimally through automated bidding processes.

Next, the advertisers also choose their *creative* or image content. Before serving the creative, the advertising platform needs to define what's a good probability score for deciding to serve  this particular content to a user. This could be defined as a click prediction: given a user and the journey they are taking on the platform, what's the probability that this user is going to click on the content?

However, maximizing clicks might not give the best relevance on the platform: it might promote spammy content. Platforms sometimes also have *shadow predictions* such as "good" clicks, hides, saves, or reposts that are trying to capture the user journey in a holistic way. On some platforms, there may be more advertising objectives like conversion optimization, which is trying to drive more sales on the advertiser's website; this is challenging to capture, as conversion happens off the platform.

Also, suppose the platform wants to expand the system to more content types, like videos and collections. Not only do they need to make these predictions that are shown here, but they also need to understand what a good video view is on the platform.

Finally, the different platform *surfaces* also have different contexts. This could be a user’s home feed, where the platform doesn't have any context or relevance information at that particular time, or a search query where the user has an intent behind it.

Given this complexity, as the platform scales it needs to make sure that it is able to make all these predictions in a performant way. Some of the design decisions that are taken here also cater to support scaling and product growth.

## Ads Serving Infrastructure

Mudgal then presented a high-level overview of the ads serving infrastructure at Pinterest. When a user interacts with the platform, the platform needs to fetch content that it wants to show to the user. The user’s request is passed in via a load balancer to an app server. This is then passed to an ad server which returns ads that are inserted into the user's feed.

**Figure 1: Ads Serving Infrastructure High Level Overview**

The ad server needs to do this in a very low latency manner, around hundreds of milliseconds, end-to-end. The input to the ad server is typically rather sparse: a user ID, the user’s ip address, and time of the day, for example.

The first task is to retrieve features for this user. This could be things like the user’s location from their IP address, or how this user has interacted on the platform in the past. These are usually retrieved from a key-value store where the key is the user ID and the values are the features.

Once this system has enriched the feature space, these are then passed into a candidate retrieval phase, which is trying to sift through billions of content items trying to find the best set of candidates to find hundreds or thousands of candidates which could be shown to the user. Then these are passed into a ranking service, which uses *heavyweight models* to determine the user’s probability of interaction with the content across multiple objectives (click, good clicks, save, reposts, hides).

This ranking service also typically has access to feature extraction, since the system cannot transmit all the content features in a candidate ranking request performantly. Typically, hundreds to thousands of candidates are sent into the ranking service, and sending all of those features together would bloat the request.

Instead, these features are fetched through a local in-memory cache (which could be something like [leveldb](https://github.com/google/leveldb)), and to ensure maximized cache hits, could utilize an external routing layer. Finally, the ranking service then sends the ads back to the ad server.

In most traditional machine learning systems, the values of the features that are used to show that ad through a particular time are very important to train machine learning models. In addition to the synchronous request to fetch these features, there is also an asynchronous request that's sent to a feature logging service which logs them. Also, to make the system more performant, there are fallback candidates: if any part of the system fails, or is unable to retrieve candidates, fallback candidates can be shown to the user so that the user always sees some content on the platform.

From the ad server, ad content is returned and inserted into the feed for the user. As the user now interacts with the feed, there is an event logging service, which could use [Apache Kafka](https://kafka.apache.org/) to log all of these events in real-time. This event logging service is very important, since advertisers are billed if the user interacts or clicks on an ad.

Furthermore, advertisers must be billed in real-time, because they define the maximum budget they can spend in a day. If the logging pipeline does not have real-time performance, the platform might overshoot an advertiser’s budget or deliver free impressions back to the advertisers.

The event logging pipeline also feeds into a reporting system, which includes hourly or daily monitoring systems. This reporting system also has a linkage to the features that are logged, because platforms want to show advertisers data about ad performance with respect to different features like country, age, or other features that might be on the platform. Finally, the event logging service and the feature logger together combine the training data for all of Pinterest’s machine learning models.

## Ads Delivery Funnel

Mudgal then showed the ads delivery funnel in more detail. This is broken down into three steps: retrieval, ranking, and auction. In the retrieval step, there are millions of parallel-running candidate generators: given a request, their motivation is to get the best set of ad candidates. This could be based on several criteria, like fresh content, a user’s recent interactions, or embedding-based generators. The candidates are then passed into a ranking model, which is trying to predict for different engagement predictions discussed earlier.

**Figure 2: Ads Delivery Funnel**

Given those predictions, the auction step determines the value of serving a particular ad to the user in the overall context. Depending on the value of that ad, the platform can decide to show it to the user or not. Also, different business logic and constraints around allocation can be handled at this time: for example, should the platform keep two ads together in the feed, or separate them?

## Ads Retrieval

The main motivation of ad retrieval is to select the best ads candidate with the best efficiency. This process uses very lightweight ML models, which can run at a very low computing cost. The quality metric for the models is *recall*.

Remember that the input to this system is the user’s ID and the content ID and request-level features. Retrieval requires *signal enrichment*, which uses several graph-based expanders to fetch extra features from key-value feature stores. For example: a user ID maps features such as age, location, gender, prior engagement rates. Similarly, the content ID maps to content features that are precomputed in the pipeline to reduce computation and improve online latency.

**Figure 3: Signal Enrichment**

Retrieval is a scatter-gather approach, invoking several components. The first is a lightweight scoring and targeting filter. Scoring estimates how valuable the content is using very simple models. Targeting restricts the ads to certain subsets of users, based on criteria selected by advertisers: for example, targeting ads based on the user’s location.

**Figure 4: Standard Query during Retrieval - Scatter Gather**

The next steps are around budgeting and pacing. If an ad has finished all its budget, it should not be retrieved. Pacing is a related concept: it is a means of spreading the ad spend across time. For example, if an ad has a $100 budget, the advertiser doesn't want to spend this $100 in the first hour, because that might not result in the best value. Advertising platforms tend to match pacing to daily patterns of traffic on their platform.

To ensure that there's a diversity of ads, deduping limits the number of ads that an advertiser can contribute: the platform shouldn’t overwhelm a feed with ads only from a single advertiser. For example, only the top-K candidates per advertiser are allowed to be passed into the next stage. Finally, since this is a scatter-gather approach, there may be different retrieval sources whose results must be blended together before sending to further down in the funnel.

The next step is candidate selection and  recent advancements in this field. Traditionally, candidate retrievers could be as simple as matching keywords or ad copy text. As the system grows more complex, this becomes harder to maintain and makes it harder to iterate.

In 2016, YouTube had a seminal paper which changed the way these retrieval systems worked, by introducing [Two-Tower Deep Neural Networks](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/). The idea is to learn latent representations of users and content based on their features. These representations and features are kept separate from each other in the model. In the end, however, if the user has engaged with a content item, those representations should be very close to each other, and so that’s the training objective of the model.

**Figure 5: Two-Tower DNN [[P Covington et. al, 2016](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)]**

The benefit of this model is that ad embeddings can be precomputed, cached, and indexed offline. The ads database builds an index by passing each ad through the ad "tower" of the model to generate its embedding. Once the ad is indexed during the serving time, the retrieval server needs to only call the user part of the model, then utilize approximate nearest neighbor search algorithms like [HNSW](https://arxiv.org/abs/1603.09320),  to find relevant ads in the ad database index.

**Figure 6: Two Tower Model Deployment**

## The Ranking Model

Next up is  the ranking model. Beginning in 2014, the models were simplistic ones like logistic regression. The next step that happened in this evolution, to make the models more expressive, Pinterest moved from simplistic solutions to more complex models such as [GBDT](https://dl.acm.org/doi/abs/10.1145/2648584.2648589) plus logistic regression solutions.

There are four types of features that a model can have: user features; content features; interaction between the user and the content in the history; and finally, events happening during this impression time. Models should learn some nonlinear interactions between these features, and GBDTs are good at it. Also, the model retains a logistic regression framework, which is a linear model which captures high cardinality features. Note that GBDTs aren’t good with such kinds of features.

**Figure 7: GBDT + Logistic Regression Ensemble Models**

Very soon, Pinterest had around 60 models in production. These models were growing and the product was growing. It became complex to maintain all these models, leading to long cycles in feature adoption and deletion, which lead to suboptimal systems.

Also, around this time, machine learning systems did not easily support model serving. Pinterest was using a different language or framework for training models vs. serving them. For example, Pinterest used to train using XGBoost, then translate that into a TensorFlow model, then translate that into C++, Pinterest’s serving language. These kinds of hops in the system lead to suboptimality and longer cycles to develop new features.

Finally, new ad groups were constantly being created or deleted: ads might have maybe a one or two month time window when they're alive. Pinterest needed the models to be responsive, so that they could be trained more incrementally on new data distributions that are coming in. However, the GBDT models are static: there is no way to train them incrementally. Deep neural networks (DNN), on the other hand, have incremental power to be trained.

The [next iteration](https://medium.com/pinterest-engineering/how-we-use-automl-multi-task-learning-and-multi-tower-models-for-pinterest-ads-db966c3dc99e) was replacing the GBDT with DNN approaches. A DNN brings many benefits, but they are more complex models. One of the things that changed here is that previous traditional machine learning algorithms relied more on feature engineering by hand, where engineers would define what two features might be related. The models inherently couldn't learn feature interactions by themselves. In DNNs architectures, the model can learn these interactions.

Most recommendation models in the industry share a similar multi-layer architecture. The first is the representation layer, which is where platforms define features and how the model can understand those features. In this particular scenario, for DNNs, feature processing is very important. If the feature scale is different across different features, the model can break, so the layer includes logic for squashing or clipping the values, or doing some normalization to the features.

**Figure 8: Pinterest’s AutoML Architecture**

Next, if two features are related to each other, the model can summarize them together and learn a common embedding. After that are multiplicative cross-layers, which learn feature interactions, followed by fully-connected layers.

Another benefit of DNNs is multitask learning across different objectives. Network weights are shared across different objectives like clicks, repins, or whatever other metrics there may be on the platform, removing the need to train different models for different objectives.

The next iteration of the model utilizes the sequence of activities that the user is doing on the platform. Let's assume that a user could have interacted with multiple pins on the platform or multiple images on the platform, which could be food related pins, home decor, or travel related pins. The idea is, can the platform use this representation of what the user is doing to define what the user might do next?

To implement this, Pinterest turned to the Transformer DNN architecture. Transformers can encode very powerful information about feature interactions. A key parameter of the model is the maximum sequence length. As sequence length increases, model size grows quadratically, which impacts serving capacities.

Increasing  the sequence length to, say, 100 events, makes the complex features discussed above impractical. Instead, the model uses simple features like: what's that action? Did the user click or not? Very simple features, but a longer sequence enables the model to have better capacities.

The latest model architecture for offline user representation at Pinterest is based on a Transformer encoder called PinnerFormer. This component takes inputs from user engagements in the past: say from yesterday to a year back. All of these engagements are encoded in an offline manner to learn an embedding for each user, which can then be used as a feature input into a downstream DNN model.

**Figure 9: [PinnerFormer: Sequence Modeling for User Representation at Pinterest](https://arxiv.org/pdf/2205.04507.pdf)**

Another input to that model is the real-time sequence coming from current user engagements. A combination of these two can learn what the user is doing on the platform. Utilizing these sequences, which is taking inspiration from the NLP domain, is what's powering Pinterest’s recommendation system.

**Figure 10: Combining Long Sequences**

## MLOps at Pinterest

In the recommendation system overall, and how it’s deployed and operated in production, machine learning is a very small piece of it. There are many other things to consider, such as: how to make sure that dev teams can iterate faster? How to make sure the serving infrastructure can support the models? How to manage resources? How to store and verify the data? Having all those kinds of checks are very important.

In the past, every team at Pinterest used to have many different pipelines: everyone was rearchitecting the same wheel. Pinterest needed to do this in a more scalable manner. That's where most of the iterations happened in the last year. Pinterest built a unified, Pytorch-based ML framework ([MLEnv](https://medium.com/pinterest-engineering/mlenv-standardizing-ml-at-pinterest-under-one-ml-engine-to-accelerate-innovation-e2b30b2f6768)) that provides Docker images and traditional CI/CD services. The part where the user writes the code is very small, and integration between various MLOps components is done seamlessly through API based solutions, which allows teams to iterate faster.

The standard model deployment process uses [MLflow](https://mlflow.org/), which is an open source solution. As these models are moved into the production pipeline, they are versioned, so that teams can do rollbacks easily. Also, models are reproducible: MLflow has a UI where users can see what parameters went into training. If a team needs to retrain and reevaluate the training process, that's easy to do.

## Testing and Monitoring

The first testing step is integration testing. When a code change is written, Pinterest can test it in the production environment through shadow traffic to see what would happen if the change were deployed. Automatically capturing metrics ensures that nothing is missed during the testing process. There is also  a debugging system that can replay how a particular request would look based on the serving of a particular model version.

The next step is around how the code is released once it's merged into the system. Pinterest follows the standard process of canary, staging, and production pipelines. Each of these stages monitors real-time metrics that the business cares about. If there is a deviation day-over-day, or there's a deviation between production and another environment, the deployment would be stopped and rolled back in a seamless manner.

Finally, despite all these safeguard, bugs could still escape. Also, advertisers might have different behaviors. So Pinterest has real-time monitoring which is capturing the day-over-day and week-over-week patterns into the system along different dimensions, which could be revenue, insertion rates, and QPS.

## ML Workflow Validation and Monitoring

Besides monitoring the production metrics, ML workflows have additional monitoring requirements. The first step is looking at the training datasets that are being fed into the model, and defining coverage and alerting on top of that. For example, monitoring features and how they change over time, and ensuring that features are fresh.

The next set of testing that happens is around the offline model evaluation. Once there is a trained model, developers need to check whether this model will make the right predictions. Pinterest captures model metrics like AUC, but they also capture predictions, to see if there are spikes into the predictions. If there are, these can halt the model validation processes. They also monitor for prediction spikes in production.

To be able to debug the system, Pinterest has developed several different tools. One key is to have visibility into the ad delivery funnel: retrieval, budgeting, indexing, and advertiser. Pinterest’s tools help them locate where the ad is being removed from the funnel.

For example, suppose an ad is not being shown very often. If it's not shown on the serving side, it might be that the ad is very low quality, or this ad is not competitive in the auction. Another scenario might be an advertiser only wants to show the ad to particular users; this is a very tight retrieval scenario, so that's why the ad might not be showing.

## Serving Large Models

Another goal is to make sure that the serving infrastructure has low latency, which enables Pinterest to score more ads. One way to improve latency is to move to GPU serving if the models are more complex. If that's not an option, there are optimization techniques, such as quantizing models or knowledge distillation, to improve latency, usually at the cost of inference accuracy.

## Conclusion

Mudgal presented an overview of Pinterest’s ad serving system and how they use ML at scale in production. He also discussed how Pinterest monitors and tests their models before and after they are deployed to production. Mudgal provided several insights that the audience can apply to their systems to overcome similar challenges.

## About the Author

#### **Anthony Alford**

Show moreShow less











### Rate this Article

Adoption

Style

Author Contacted

#### This content is in the [AI, ML & Data Engineering](/ai-ml-data-eng/) topic

##### Related Topics:

* [AI, ML & Data Engineering](/ai-ml-data-eng/)
* [QCon San Francisco 2023](/qcon-san-francisco-2023/)
* [Neural Networks](/Neural-Networks/)
* [Deep Learning](/Deep+Learning/)
* [QCon Software Development Conference](/qcon/)
* [Machine Learning](/MachineLearning/)
* [Rankings](/rankings/)
* [Advertising](/Advertising/)

* #### Related Editorial
* #### Popular across InfoQ

  + ##### [Anthropic Designs Three-Agent Harness Supports Long-Running Full-Stack AI Development](/news/2026/04/anthropic-three-agent-harness-ai/)
  + ##### [Dynamic Languages Faster and Cheaper in 13-Language Claude Code Benchmark](/news/2026/04/ai-coding-language-benchmark/)
  + ##### [Beyond RAG: Architecting Context-Aware AI Systems with Spring Boot](/articles/beyond-rag-context-aware/)
  + ##### [TigerFS Mounts PostgreSQL Databases as a Filesystem for Developers and AI Agents](/news/2026/04/tigerfs-postgresql-filesystem/)
  + ##### [Pinterest Deploys Production-Scale Model Context Protocol Ecosystem for AI Agent Workflows](/news/2026/04/pinterest-mcp-ecosystem/)
  + ##### [Context Engineering with Adi Polak](/podcasts/context-engineering-large-language-models/)

### **The InfoQ** Newsletter

A round-up of last week’s content on InfoQ sent out every Tuesday. Join a community of over 250,000 senior developers.
[View an example](https://assets.infoq.com/newsletter/regular/en/newsletter_sample/newsletter_sample.html)

Enter your e-mail address

Select your country

Select a country


I consent to InfoQ.com handling my data as explained in this [Privacy Notice](https://www.infoq.com/privacy-notice).

[We protect your privacy.](/privacy-notice/)