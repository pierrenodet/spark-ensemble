---
id: boosting
title: Boosting
---

The old Boosting, à la papa, from Freund and Schapire [[1](#references)].

For classification, SAMME (Multi-class AdaBoost) and SAMME.R [[2](#references)] from Ji Zhu are implemented.

For regression, R2 (Improving Regressors using Boosting Techniques) [[3](#references)] from H. Drucker has been chosen.

`BoostingRegressionModel` proposes two different voting strategies to aggregate predictions from base models, one using the weighted median as described in [[3](#references)], the other one using the weighted mean.

## Parameters

The parameters available for Boosting are related to the loss function and the algorithm for weight computation.

```scala
import org.apache.spark.ml.classification.{BoostingClassifier, DecisionTreeClassifier}
import org.apache.spark.ml.regression.{BoostingRegressor, DecisionTreeRegressor}

new BoostingClassifier()
        .setBaseLearner(new DecisionTreeClassifier()) //Base learner used by the meta-estimator.
        .setNumBaseLearners(10) //Number of base learners.
        .setAlgorithm("real") //SAMME or SAMME.R algorithm.      

new BoostingRegressor()
        .setBaseLearner(new DecisionTreeClassifier()) //Base learner used by the meta-estimator.
        .setNumBaseLearners(10) //Number of base learners.
        .setLoss("squared") //Loss function.    
        .setVotingStrategy("median") //Voting strategy.         
```

## References

 * [[1](https://www.sciencedirect.com/science/article/pii/S002200009791504X?ref=pdf_download&fr=RR-2&rr=757a8834fa1bd397)] Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences, 55(1), 119-139.
 * [[2](https://web.stanford.edu/~hastie/Papers/samme.pdf)] Hastie, T., Rosset, S., Zhu, J., & Zou, H. (2009). Multi-class adaboost. Statistics and its Interface, 2(3), 349-360.
 * [[3](https://pdfs.semanticscholar.org/8d49/e2dedb817f2c3330e74b63c5fc86d2399ce3.pdf)] Drucker, H. (1997, July). Improving regressors using boosting techniques. In ICML (Vol. 97, pp. 107-115).