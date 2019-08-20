# Bagging

Bagging (Bootstrap aggregating) is a meta-algorithm introduced by Breiman [[1](#references)] that generates multiple versions of a predictor and uses these to get an aggregated predictor.

The Random Subspace Method is another meta-algorithm proposed by Ho [[2](#references)] that performs the same transformations as Bagging but on the feature space.

Combining this two methods is called SubBag and is designed by Pance Panov and Saso Dzeroski [[3](#references)].

Here the `BaggingClassifier` and the `BaggingRegressor` implement the SubBag meta-estimator.

For prediction, `BaggingClassificationModel` uses a majority vote and `BaggingRegressionModel` uses the average.

## Parameters

The parameters available for Bagging are related to the number of base learners and the randomness of the subbag method.

```scala
import org.apache.spark.ml.classification.{BaggingClassifier, DecisionTreeClassifier}

new BaggingClassifier()
        .setBaseLearner(new DecisionTreeClassifier()) //Base learner used by the meta-estimator.
        .setNumBaseLearners(10) //Number of base learners.
        .setSampleRatio(0.8) //Ratio sampling of exemples.
        .setReplacement(true) //Exemples drawn with replacement or not.
        .setSubspaceRatio(0.8) //Ratio sampling of features.
        .setParallelism(4) //Number of base learners trained at the same time.
```

## References

 * [[1](https://www.stat.berkeley.edu/~breiman/bagging.pdf)] Leo Breiman (1994) Bagging Predictors
 * [[2](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=709601)] Tin Kam Ho (1998) The Random Subspace Method for Constructing Decision Forests
 * [[3](http://kt.ijs.si/panovp/Default_files/Panov08Combining.pdf)] Panov P., Džeroski S. (2007) Combining Bagging and Random Subspaces to Create Better Ensembles
