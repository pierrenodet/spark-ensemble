---
id: gbm
title: GBM
---

God Jerome H. Friedman enlightened humankind with GBM (Gradient Boosting Machines) at the beginning of the third millennium.

The first of his ten commandments was named: Greedy Function Approximation: A Gradient Boosting Machine [[1](#references)], introducing a meta-algorithm that was aimed to do gradient descent in function space. In the end, you are kinda doing it in error space, but the heuristic was god sent.

The second commandment was: Stochastic Gradient Boosting [[2](#references)], introducing randomness in each iteration by using SubBags.

Beware of this evil trick from Satan: GBM only works with Regressors as base learners.

PS : It works for multi-class Classification.
PPS : Early-stop is implemented with an N-Round variant.
PPPS : Newton Boosting is available for losses that admit a non-constant hessian.

## Parameters

The parameters available for GBM are related to the base framework, the stochastic version, and the early-stop.

```scala
import org.apache.spark.ml.classification.GBMRegressor
import org.apache.spark.ml.classification.GBMClassifier
import org.apache.spark.ml.regression.DecisionTreeRegressor

new GBMRegressor()
        .setBaseLearner(new DecisionTreeRegressor()) //Base learner used by the meta-estimator.
        .setNumBaseLearners(10) //Number of base learners.
        .setInitStrategy("base") //Strategy for the initialization of predictions.
        .setLearningRate(0.1) //Shrinkage parameter.
        .setSubsampleRatio(0.8) //Ratio sampling of examples.
        .setReplacement(true) //Exemples drawn with replacement or not.
        .setSubspaceRatio(0.8) //Ratio sampling of features.
        .setLoss("squared") //Loss function used for residuals and optimized step size.
        .setAlpha(0.5) //Quantile parameter for quantile or huber losses.
        .setUpdates("newton") //Newton or Gradient boosting.
        .setOptimizedWeights(true) //Line search the best step size or use 1.0 instead.
        .setMaxIter(100) //Optimizer maximum number of iterations for optimized step size.
        .setTol(1E-3) //Optimizer tolerance for optimized step size.
        .setValidationIndicatorCol("val") //Column name that contains true or false for the early stop data set.
        .setValidationTol(1E-2) //Tolerance for gain in loss on early stop set.
        .setNumRounds(8) //Number of rounds to wait for the loss on early stop set to decrease.

new GBMClassifier()
        .setBaseLearner(new DecisionTreeRegressor()) //Base learner used by the meta-estimator.
        .setNumBaseLearners(10) //Number of base learners.
        .setInitStrategy("prior") //Strategy for the initialization of predictions.
        .setLearningRate(0.1) //Shrinkage parameter.
        .setSubsampleRatio(0.8) //Ratio sampling of examples.
        .setReplacement(true) //Exemples drawn with replacement or not.
        .setSubspaceRatio(0.8) //Ratio sampling of features.
        .setLoss("logloss") //Loss function used for residuals and optimized step size.
        .setUpdates("newton") //Newton or Gradient boosting.
        .setOptimizedWeights(true) //Line search the best step size or use 1.0 instead.
        .setMaxIter(100) //Optimizer maximum number of iterations for optimized step size.
        .setTol(1E-3) //Optimizer tolerance for optimized step size.
        .setValidationIndicatorCol("val") //Column name that contains true or false for the early stop data set.
        .setValidationTol(1E-2) //Tolerance for gain in loss on early stop set.
        .setNumRounds(8) //Number of rounds to wait for the loss on early stop set to decrease.
        .setParallelism(4) //Number of base learners trained simultaneously. Should be at most the number of classes.
```

## References

 * [[1](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)] Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of statistics, 1189-1232.
 * [[2](https://astro.temple.edu/~msobel/courses_files/StochasticBoosting(gradient).pdf)] Friedman, J. H. (2002). Stochastic gradient boosting. Computational statistics & data analysis, 38(4), 367-378.