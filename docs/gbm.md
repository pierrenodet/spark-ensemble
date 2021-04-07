---
id: gbm
title: GBM
---

God Jerome H. Friedman enlightened mankind with GBM (Gradient Boosting Machines) in the beginning of the third millennium.

The first of his ten commandments was named : Greedy Function Approximation: A Gradient Boosting Machine [[1](#references)], introducing a meta algorithm that was aimed to do gradient descent in function space. In the end you were kinda doing it in error space, but the heuristic was god sent.

The second commandment was : Stochastic Gradient Boosting [[2](#references)] introducing randomness in each iteration by using SubBags.

Beware of this evil trick from Satan : GBM only works with Regressors as base learners.

PS : It works for multi-class Classification.
PPS : Early stop is implemented with a N Round variant.

## Parameters

The parameters available for GBM are related to the base framework, the stochastic version and early stop.

```scala
import org.apache.spark.ml.classification.GBMRegressor
import org.apache.spark.ml.classification.GBMClassifier
import org.apache.spark.ml.regression.DecisionTreeRegressor

new GBMRegressor()
        .setBaseLearner(new DecisionTreeRegressor()) //Base learner used by the meta-estimator.
        .setNumBaseLearners(10) //Number of base learners.
        .setLearningRate(0.1) //Shrinkage parameter.
        .setSampleRatio(0.8) //Ratio sampling of exemples.
        .setReplacement(true) //Exemples drawn with replacement or not.
        .setSubspaceRatio(0.8) //Ratio sampling of features.
        .setOptimizedWeights(true) //Line search the best step size or use 1.0 instead.
        .setLoss("squared") //Loss function used for residuals and optimized step size.
        .setAlpha(0.5) //Extra parameter for certain loss functions as quantile or huber.
        .setValidationIndicatorCol("val") //Column name that contains true or false for the early stop data set.
        .setTol(1E-3) //Tolerance for optimized step size and gain in loss on early stop set.
        .setNumRound(8) //Number of rounds to wait for the loss on early stop set to decrease.

new GBMClassifier()
        .setBaseLearner(new DecisionTreeRegressor()) //Base learner used by the meta-estimator.
        .setNumBaseLearners(10) //Number of base learners.
        .setLearningRate(0.1) //Shrinkage parameter.
        .setSampleRatio(0.8) //Ratio sampling of exemples.
        .setReplacement(true) //Exemples drawn with replacement or not.
        .setSubspaceRatio(0.8) //Ratio sampling of features.
        .setOptimizedWeights(true) //Line search the best step size or use 1.0 instead.
        .setLoss("deviance") //Loss function used for residuals and optimized step size.
        .setValidationIndicatorCol("val") //Column name that contains true or false for the early stop data set.
        .setTol(1E-3) //Tolerance for optimized step size and gain in loss on early stop set.
        .setNumRound(8) //Number of rounds to wait for the loss on early stop set to decrease.
        .setParallelism(4) //Number of base learners trained at the same time. Should be at most the number of classes.
        .setInstanceTrimmingRatio(1.0) //Quantile of highest instance weights kept each round.
```

## References

 * [[1](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)] Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of statistics, 1189-1232.
 * [[2](https://astro.temple.edu/~msobel/courses_files/StochasticBoosting(gradient).pdf)] Friedman, J. H. (2002). Stochastic gradient boosting. Computational statistics & data analysis, 38(4), 367-378.
