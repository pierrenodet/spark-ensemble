# Overview 

## Why ?

Want to switch Decision Trees to Neural Net in Random Forest to make avNNet ?

There is no curse in elvish, entish or the tongues of men for this treachery.

But it's easy now !

## Implemented Meta-Algorithms

Le classico of meta-algorithms have been implemented in this library as closely as described in the papers of the original authors.

Here is the current list :

 * Stacking
 * Bagging
 * Boosting
 * GBM
 
They all work with Multiclass Classification and Regression.

## Quick Exemple

**Building Base Learner**

```scala
import org.apache.spark.ml.classification.DecisionTreeClassifier

val baseClassifier = new DecisionTreeClassifier()
.setMaxDepth(20)
```

**Building Meta Estimator**

```scala
import org.apache.spark.ml.classification.BaggingClassifier

val baggingClassifier = new BaggingClassifier()
.setBaseLearner(baseClassifier)
.setNumBaseLearners(10)
.setParallelism(4)
```

**Building Param Grid**

```scala
import org.apache.spark.ml.tuning.ParamGridBuilder

val paramGrid = new ParamGridBuilder()
        .addGrid(baggingClassifier.numBaseLearners, Array(10,20))
        .addGrid(baseClassifier.maxDepth, Array(10,20))
        .build()
```

**Grid Search with Cross Validation**

```scala
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.classification.BaggingClassificationModel

val cv = new CrossValidator()
        .setEstimator(baggingClassifier)
        .setEvaluator(new MulticlassClassificationEvaluator())
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(5)
        .setParallelism(4)

val cvModel = cv.fit(data)

val bestModel = cvModel.bestModel.asInstanceOf[BaggingClassificationModel]

bestModel
```

## Save and Load

You can for sure save and load all this meta models, at the condition that the base models are saveable and loadable too.

```scala
bestModel.write.overwrite().save("/tmp/kek")
val loaded = BaggingClassificationModel.load("/tmp/kek")
```