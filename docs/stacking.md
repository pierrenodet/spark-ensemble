---
id: stacking
title: Stacking
---

Stacking (Stacked Generalization) is a meta-algorithm introduced by David H. Wolpert [[1](#references)] that involves training a learning algorithm to combine the predictions of several other learning algorithms.

## Parameters

The parameters available for Stacking are related to the base and stack learner.

```scala
import org.apache.spark.ml.classification.{StackingClassifier, RandomForestClassifier, DecisionTreeClassifier}

new StackingClassifier()
        .setBaseLearners(Array(new DecisionTreeClassifier(), new RandomForestClassifier())) //Base learners used by the meta-estimator.
        .setStackMethod("proba") //Methods called for each base learner, only for classification.
        .setStacker(new DecisionTreeClassifier()) //Learner that will combine the predictions of base learners.
        .setParallelism(4) //Number of base learners trained simultaneously.
```

## References

 * [[1](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.133.8090&rep=rep1&type=pdf)] David H. Wolpert (1992) Stacked Generalization