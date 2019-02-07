# Bagging Estimator for Apache Spark
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/pierrenodet/spark-bagging/blob/master/LICENSE)
[![Build Status](https://travis-ci.org/pierrenodet/spark-bagging.svg?branch=master)](https://travis-ci.org/pierrenodet/spark-bagging)
[![codecov](https://codecov.io/gh/pierrenodet/spark-bagging/branch/master/graph/badge.svg)](https://codecov.io/gh/pierrenodet/spark-bagging)
[![Release Artifacts](https://img.shields.io/nexus/releases/https/oss.sonatype.org/com.github.pierrenodet/spark-bagging_2.11.svg?colorB=blue)](https://oss.sonatype.org/content/repositories/releases/com/github/pierrenodet/spark-bagging_2.11)

Repository of an implementation of the Bagging Meta-Estimator à la scikit-learn for Apache Spark ML

## Setup

Download the dependency from Sonatype

**SBT**

```scala
libraryDependencies += "com.github.pierrenodet" % "spark-bagging" % "0.0.1"
```

**Maven**

```maven-pom
<dependency>
  <groupId>com.github.pierrenodet</groupId>
  <artifactId>spark-bagging_2.11</artifactId>
  <version>0.0.1</version>
</dependency>
```

## How to use

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

## Built With

* [Scala](https://www.scala-lang.org/) - Programming Language
* [Spark](https://spark.apache.org/) - Big Data Framework
* [SBT](https://www.scala-sbt.org/) - Build Tool

## Contributing

Feel free to open an issue or make a pull request to contribute to the repository.

## Authors

* **Nodet Pierre** - *Main developer* - [GitHub Profile](https://github.com/pierrenodet)

See also the list of [contributors](https://github.com/pierrenodet/spark-bagging/graphs/contributors) who participated in this project.

## License

This project is licensed under the Apache License Version 2.0 - see the [LICENSE](LICENSE) file for details
