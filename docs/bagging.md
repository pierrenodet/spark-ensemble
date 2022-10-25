---
id: bagging
title: Bagging
---

Bagging (Bootstrap aggregating) is a meta-algorithm introduced by Breiman [[1](#references)] that generates multiple versions of a predictor and uses these to get an aggregated predictor.

The Random Subspace Method is another meta-algorithm proposed by Ho [[2](#references)] that performs the same transformations as Bagging but on the feature space.

Combining these two methods is called SubBag and is designed by Pance Panov and Saso Dzeroski [[3](#references)].

Here the `BaggingClassifier` and the `BaggingRegressor` implement the SubBag meta-estimator.

For classification, `BaggingClassificationModel` uses a majority vote of the base model predictions.
It can be either `soft` or `hard`, using the predicted classes or the predicted probabilities of each base model.

For regression, `BaggingRegressionModel` uses the average of the base model predictions.

## Parameters

The parameters available for Bagging are related to the number of base learners and the randomness of the subbag method.

```scala
import org.apache.spark.ml.classification.{BaggingClassifier, DecisionTreeClassifier}
import org.apache.spark.ml.regression.{BaggingRegressor, DecisionTreeRegressor}

new BaggingClassifier()
        .setBaseLearner(new DecisionTreeClassifier()) //Base learner used by the meta-estimator.
        .setNumBaseLearners(10) //Number of base learners.
        .setSubsampleRatio(0.8) //Ratio sampling of examples.
        .setReplacement(true) //Exemples drawn with replacement or not.
        .setSubspaceRatio(0.8) //Ratio sampling of features.
        .setVotingStrategy("soft") //Soft or Hard majority vote.
        .setParallelism(4) //Number of base learners trained simultaneously.

new BaggingRegressor()
        .setBaseLearner(new DecisionTreeRegressor()) //Base learner used by the meta-estimator.
        .setNumBaseLearners(10) //Number of base learners.
        .setSubsampleRatio(0.8) //Sampling ratio of examples.
        .setReplacement(true) //Exemples drawn with replacement or not.
        .setSubspaceRatio(0.8) //Sampling ratio of features.
        .setParallelism(4) //Number of base learners trained simultaneously.
```

## References

 * [[1](https://www.stat.berkeley.edu/~breiman/bagging.pdf)] Leo Breiman (1994) Bagging Predictors
 * [[2](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=709601)] Tin Kam Ho (1998) The Random Subspace Method for Constructing Decision Forests
 * [[3](http://kt.ijs.si/panovp/Default_files/Panov08Combining.pdf)] Panov P., Džeroski S. (2007) Combining Bagging and Random Subspaces to Create Better Ensembles
