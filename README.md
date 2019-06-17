# Ensemble Estimators for Apache Spark
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/pierrenodet/spark-ensemble/blob/master/LICENSE)
[![Build Status](https://travis-ci.org/pierrenodet/spark-ensemble.svg?branch=master)](https://travis-ci.org/pierrenodet/spark-ensemble)
[![codecov](https://codecov.io/gh/pierrenodet/spark-ensemble/branch/master/graph/badge.svg)](https://codecov.io/gh/pierrenodet/spark-ensemble)
[![Maven Central](https://img.shields.io/maven-central/v/com.github.pierrenodet/spark-ensemble_2.12.svg?label=maven-central&colorB=blue)](https://search.maven.org/search?q=g:%22com.github.pierrenodet%22%20AND%20a:%22spark-ensemble_2.12%22)

Library of Meta-Estimators à la scikit-learn for Ensemble Learning for Apache Spark ML

## Setup

Download the dependency from Maven Central

**SBT**

```scala
libraryDependencies += "com.github.pierrenodet" % "spark-ensemble_2.12" % "0.5.4"
```

**Maven**

```maven-pom
<dependency>
  <groupId>com.github.pierrenodet</groupId>
  <artifactId>spark-ensemble_2.12</artifactId>
  <version>0.5.4</version>
</dependency>
```

## What's inside

This Spark ML library contains the following algorithms for ensemble learning :

 * [Bagging and Random Subspaces](https://pdfs.semanticscholar.org/d38f/979ad85d59fc93058279010efc73a24a712c.pdf)
 * [Boosting R2](https://pdfs.semanticscholar.org/8d49/e2dedb817f2c3330e74b63c5fc86d2399ce3.pdf)
 * [Boosting SAMME](https://web.stanford.edu/~hastie/Papers/samme.pdf)
 * [Stacking](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking)
 * [GBM](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)

## How to use

**Loading Features**

```scala
val raw = spark.read.option("header", "true").option("inferSchema", "true").csv("src/test/resources/data/iris/train.csv")

val vectorAssembler = new VectorAssembler()
.setInputCols(raw.columns.filter(x => !x.equals("class"))).
setOutputCol("features")

val stringIndexer = new StringIndexer()
.setInputCol("class")
.setOutputCol("label")
    
val data = stringIndexer.fit(raw).transform(vectorAssembler.transform(raw))
```

**Base Learner Settings**

```scala
val baseClassifier = new DecisionTreeClassifier()
.setMaxDepth(20)
.setMaxBin(30)
```

**Meta Estimator Settings**

```scala
val baggingClassifier = new BaggingClassifier()
.setBaseLearner(baseClassifier)
.setMaxIter(10)
.setParallelism(4)
```

**Train and Test**

```scala
val Array(train, test) = data.randomSplit(Array(0.7, 0.3))

val model = baggingClassifier.fit(train)

model.models.map(_.asInstanceOf[DecisionTreeClassificationModel])

val predicted = model.transform(test)
predicted.show()

val re = new MulticlassClassificationEvaluator()
println(re.evaluate(predicted))
```

**Cross Validation**

```scala
val paramGrid = new ParamGridBuilder()
        .addGrid(baggingClassifier.sampleRatioFeatures, Array(0.7,1))
        .addGrid(baggingClassifier.replacementFeatures, Array(x = false))
        .addGrid(baggingClassifier.replacement, Array(x = true))
        .addGrid(baggingClassifier.sampleRatio, Array(0.7, 1))
        .addGrid(baseClassifier.maxDepth, Array(1,10))
        .addGrid(baseClassifier.maxBins, Array(30,40))
        .build()

val cv = new CrossValidator()
        .setEstimator(br)
        .setEvaluator(new MulticlassClassificationEvaluator())
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(5)
        .setParallelism(4)

val cvModel = cv.fit(data)

cvModel.bestModel.asInstanceOf[BaggingClassificationModel]
```

## Contributing

Feel free to open an issue or make a pull request to contribute to the repository.

## Authors

* **Pierre Nodet** - *Main developer* - [GitHub Profile](https://github.com/pierrenodet)

See also the list of [contributors](https://github.com/pierrenodet/spark-bagging/graphs/contributors) who participated in this project.

## License

This project is licensed under the Apache License Version 2.0 - see the [LICENSE](LICENSE) file for details.
