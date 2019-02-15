package org.apache.spark.ml.regression

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.scalatest.FunSuite

class BoostingRegressorSuite extends FunSuite with DatasetSuiteBase {

  test("benchmark") {

    val raw = spark.read.option("header", "true").option("inferSchema", "true").csv("src/test/resources/data/bostonhousing/train.csv")
    val test = spark.read.option("header", "true").option("inferSchema", "true").csv("src/test/resources/data/bostonhousing/test.csv")

    val vectorAssembler = new VectorAssembler().setInputCols(raw.columns.filter(x => !x.equals("ID") && !x.equals("medv"))).setOutputCol("features")
    val br = new BoostingRegressor().setBaseLearner(new DecisionTreeRegressor()).setFeaturesCol("features").setLabelCol("medv").setMaxIter(10)
    val gbt = new GBTRegressor().setFeaturesCol("features").setLabelCol("medv").setMaxIter(10)

    val data = vectorAssembler.transform(raw)
    data.cache()

    time {
      val brParamGrid = new ParamGridBuilder()
        .addGrid(br.learningRate, Array(0.01,0.001,0.05,0.005))
        .build()

      val brCV = new CrossValidator()
        .setEstimator(br)
        .setEvaluator(new RegressionEvaluator().setLabelCol(br.getLabelCol).setPredictionCol(br.getPredictionCol).setMetricName("rmse"))
        .setEstimatorParamMaps(brParamGrid)
        .setNumFolds(5)
        .setParallelism(4)

      val brCVModel = brCV.fit(data)

      println(brCVModel.avgMetrics.mkString(","))
      println(brCVModel.bestModel.asInstanceOf[BoostingRegressionModel].getLearningRate)
      println(brCVModel.avgMetrics.min)

    }

    time {
      val paramGrid = new ParamGridBuilder()
        .addGrid(gbt.featureSubsetStrategy, Array("auto"))
        .addGrid(gbt.subsamplingRate, Array(0.3,0.5,0.7))
        .build()

      val cv = new CrossValidator()
        .setEstimator(gbt)
        .setEvaluator(new RegressionEvaluator().setLabelCol(gbt.getLabelCol).setPredictionCol(gbt.getPredictionCol).setMetricName("rmse"))
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(5)
        .setParallelism(4)

      val cvModel = cv.fit(data)

      println(cvModel.avgMetrics.mkString(","))
      println(cvModel.bestModel.asInstanceOf[GBTRegressionModel].getSubsamplingRate)
      println(cvModel.avgMetrics.min)
    }
  }

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")
    result
  }

}
