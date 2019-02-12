# Ensemble Estimators for Apache Spark
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/pierrenodet/spark-ensemble/blob/master/LICENSE)
[![Build Status](https://travis-ci.org/pierrenodet/spark-ensemble.svg?branch=master)](https://travis-ci.org/pierrenodet/spark-ensemble)
[![codecov](https://codecov.io/gh/pierrenodet/spark-ensemble/branch/master/graph/badge.svg)](https://codecov.io/gh/pierrenodet/spark-ensemble)
[![Maven Central](https://img.shields.io/maven-central/v/com.github.pierrenodet/spark-ensemble_2.11.svg?label=maven-central&colorB=blue)](https://search.maven.org/search?q=g:%22com.github.pierrenodet%22%20AND%20a:%22spark-ensemble_2.11%22)

Library of Meta-Estimators à la scikit-learn for Ensemble Learning for Apache Spark ML

## Setup

Download the dependency from Sonatype

**SBT**

```scala
libraryDependencies += "com.github.pierrenodet" % "spark-ensemble_2.11" % "0.3.0"
```

**Maven**

```maven-pom
<dependency>
  <groupId>com.github.pierrenodet</groupId>
  <artifactId>spark-ensemble_2.11</artifactId>
  <version>0.3.0</version>
</dependency>
```

## How to use

**Bagging**

```scala
val data = spark.read.option("header", "true").option("inferSchema", "true").csv("src/test/resources/data/bostonhousing/train.csv")

val vectorAssembler = new VectorAssembler().setInputCols(train.columns.filter(x => !x.equals("ID") && !x.equals("medv")))).setOutputCol("features")

val baseRegressor = new DecisionTreeRegressor()
val baggingRegressor = new BaggingRegressor().setBaseLearner(baseRegressor).setFeaturesCol("features").setLabelCol("medv").setMaxIter(100).setParallelism(4)

val formatted = vectorAssembler.transform(data)
val Array(train, validation) = formatted.randomSplit(Array(0.7, 0.3))

val brModel = baggingRegressor.fit(train)

brModel.getModels

val brPredicted = brModel.transform(validation)
brPredicted.show()

val re = new RegressionEvaluator().setLabelCol("medv").setMetricName("rmse")
println(re.evaluate(brPredicted))
```

**Boosting**

```scala
val data = spark.read.option("header", "true").option("inferSchema", "true").csv("src/test/resources/data/bostonhousing/train.csv")

val vectorAssembler = new VectorAssembler().setInputCols(train.columns.filter(x => !x.equals("ID") && !x.equals("medv")))).setOutputCol("features")

val baseRegressor = new DecisionTreeRegressor()
val boostingRegressor = new BoostingRegressor().setBaseLearner(baseRegressor).setFeaturesCol("features").setLabelCol("medv").setMaxIter(10).setLearningRate(0.4)

val formatted = vectorAssembler.transform(data)
val Array(train, validation) = formatted.randomSplit(Array(0.7, 0.3))

val brModel = boostingRegressor.fit(train)

val brPredicted = brModel.transform(validation)
brPredicted.show()

val re = new RegressionEvaluator().setLabelCol("medv").setMetricName("rmse")
println(re.evaluate(brPredicted))
```

**Stacking**

```scala
val data = spark.read.option("header", "true").option("inferSchema", "true").csv("src/test/resources/data/bostonhousing/train.csv")

val vectorAssembler = new VectorAssembler().setInputCols(train.columns.filter(x => !x.equals("ID") && !x.equals("medv")))).setOutputCol("features")

val regressors = Array(new RandomForestRegressor(),new DecisionTreeRegressor())
val stacker = new DecisionTreeRegressor()
val stackingRegressor = new StackingRegressor().setStacker(stacker).setLearners(regressors).setFeaturesCol("features").setLabelCol("medv").setParallelism(4)

val formatted = vectorAssembler.transform(data)
val Array(train, validation) = formatted.randomSplit(Array(0.7, 0.3))

val srModel = stackingRegressor.fit(train)

val srPredicted = srModel.transform(validation)
srPredicted.show()

val re = new RegressionEvaluator().setLabelCol("medv").setMetricName("rmse")
println(re.evaluate(srPredicted))
```

## Built With

* [Scala](https://www.scala-lang.org/) - Programming Language
* [Spark](https://spark.apache.org/) - Big Data Framework
* [SBT](https://www.scala-sbt.org/) - Build Tool

## Contributing

Feel free to open an issue or make a pull request to contribute to the repository.

## Authors

* **Pierre Nodet** - *Main developer* - [GitHub Profile](https://github.com/pierrenodet)

See also the list of [contributors](https://github.com/pierrenodet/spark-bagging/graphs/contributors) who participated in this project.

## License

This project is licensed under the Apache License Version 2.0 - see the [LICENSE](LICENSE) file for details
