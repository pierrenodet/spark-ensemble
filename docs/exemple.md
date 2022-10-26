---
id: exemple
title: Exemple
---

## Building Base Learner

```scala
import org.apache.spark.ml.classification.DecisionTreeClassifier

val baseClassifier = new DecisionTreeClassifier()
.setMaxDepth(20)
```

## Building Meta Estimator

```scala
import org.apache.spark.ml.classification.BaggingClassifier

val baggingClassifier = new BaggingClassifier()
.setBaseLearner(baseClassifier)
.setNumBaseLearners(10)
.setParallelism(4)
```

## Building Param Grid

```scala
import org.apache.spark.ml.tuning.ParamGridBuilder

val paramGrid = new ParamGridBuilder()
        .addGrid(baggingClassifier.numBaseLearners, Array(10,20))
        .addGrid(baseClassifier.maxDepth, Array(10,20))
        .build()
```

## Grid Search with Cross Validation

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

```scala
bestModel.write.overwrite().save("/tmp/model")
val loaded = BaggingClassificationModel.load("/tmp/model")
```