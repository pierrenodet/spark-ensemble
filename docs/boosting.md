# Boosting


## Parameters

The parameters available for Boosting are related to the number of iterations and early stop criteria.

```scala
import org.apache.spark.ml.classification.{BoostingClassifier, DecisionTreeClassifier}

new BoostingClassifier()
        .setBaseLearner(new DecisionTreeClassifier()) //Weak learner used by the meta-estimator.
        .setNumBaseLearners(10) //Number of weak learners.
        .setLoss("exponential") //Loss function used to caculate the weights of the exemples to use in the next iteration.
        .setValidationIndicatorCol("val") //Name of the column that indicates the validation set in the training set.
        .setTol(0.1) //Tolerance for the increase in performance in the validation set.
        .setNumRound(10) //Number of iterations to wait for an increase in performance on the validation set.
```

## References

 * [[1](https://www.stat.berkeley.edu/~breiman/bagging.pdf)] Leo Breiman (1994) Bagging Predictors.
 * [[2](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=709601)] Tin Kam Ho (1998) The Random Subspace Method for Constructing Decision Forests
 * [[3](http://kt.ijs.si/panovp/Default_files/Panov08Combining.pdf)] Panov P., Džeroski S. (2007) Combining Bagging and Random Subspaces to Create Better Ensembles
