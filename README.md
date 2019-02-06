# Bagging Estimator for Apache Spark ML
[![Build Status](https://travis-ci.org/pierrenodet/spark-bagging.svg?branch=master)](https://travis-ci.org/pierrenodet/spark-bagging)
[![codecov](https://codecov.io/gh/pierrenodet/spark-bagging/branch/master/graph/badge.svg)](https://codecov.io/gh/pierrenodet/spark-bagging)

Repository of an implementation of the Bagging Meta-Estimator à la SKLearn for Apache Spark ML

## How to use

```scala
val data = spark.read.option("header", "true").option("inferSchema", "true").csv("src/test/resources/data/bostonhousing/train.csv")
val Array(train, validation) = data.randomSplit(Array(0.7, 0.3))

val vectorAssembler = new VectorAssembler().setInputCols(train.columns.filter(x => !(x.equals("ID") && x.equals("medv")))).setOutputCol("features")

val baseRegressor = new DecisionTreeRegressor().setFeaturesCol("features").setLabelCol("medv")
val baggingRegressor = new BaggingRegressor().setBaseLearner(baseRegressor).setFeaturesCol("features").setLabelCol("medv").setMaxIter(100).setParallelism(4)

val brPipeline = new Pipeline().setStages((vectorAssembler :: br :: Nil).toArray)

val brModel = brPipeline.fit(train)

brModel.stages(1).asInstanceOf[BaggingRegressionModel].getModels

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

Feel free to make open an issue or make a pull request to contribute to the repository.

## Authors

* **Nodet Pierre** - *Main developer* - [GitHub Profile](https://github.com/pierrenodet)

See also the list of [contributors](https://github.com/pierrenodet/spark-bagging/graphs/contributors) who participated in this project.

## License

This project is licensed under the Apache License Version 2.0 - see the [LICENSE](LICENSE) file for details
