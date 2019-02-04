package org.apache.spark.ml.regression

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.scalatest.FunSuite

class BaggingRegressorSuite extends FunSuite with DatasetSuiteBase {

  test("simple test") {

    val data = spark.read.option("header", "true").option("inferSchema", "true").csv("src/test/resources/data/bostonhousing/train.csv")
    val Array(train, validation) = data.randomSplit(Array(0.7, 0.3))
    val test = spark.read.option("header", "true").option("inferSchema", "true").csv("src/test/resources/data/bostonhousing/test.csv")

    val vectorAssembler = new VectorAssembler().setInputCols(train.columns.filter(x => !(x.equals("ID") && x.equals("medv")))).setOutputCol("features")
    val br = new BaggingRegressor().setBaseLearner(new DecisionTreeRegressor().setFeaturesCol("features").setLabelCol("medv")).setFeaturesCol("features").setLabelCol("medv").setMaxIter(100).setParallelism(4)
    val rf = new RandomForestRegressor().setFeaturesCol("features").setLabelCol("medv")

    val brPipeline = new Pipeline().setStages((vectorAssembler :: br :: Nil).toArray)
    val rfPipeline = new Pipeline().setStages((vectorAssembler :: rf :: Nil).toArray)

    val brModel = brPipeline.fit(train)
    val rfModel = rfPipeline.fit(train)

    val brPredicted = brModel.transform(validation)
    val rfPredicted = rfModel.transform(validation)

    time {
      brPredicted.show()
    }
    time {
      rfPredicted.show()
    }

    val re = new RegressionEvaluator().setLabelCol("medv").setMetricName("rmse")

    println(re.evaluate(brPredicted))
    println(re.evaluate(rfPredicted))
  }

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")
    result
  }

}
