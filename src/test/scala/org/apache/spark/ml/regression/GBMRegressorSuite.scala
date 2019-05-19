package org.apache.spark.ml.regression

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.scalatest.FunSuite

class GBMRegressorSuite extends FunSuite with DatasetSuiteBase {

  test("benchmark") {

    val raw = spark.read.option("header", "true").option("inferSchema", "true").csv("src/test/resources/data/bostonhousing/train.csv")

    val vectorAssembler = new VectorAssembler().setInputCols(raw.columns.filter(x => !x.equals("ID") && !x.equals("medv"))).setOutputCol("features")
    val gmbr = new GBMRegressor().setBaseLearner(new DecisionTreeRegressor()).setFeaturesCol("features").setLabelCol("medv").setMaxIter(10).setTol(1E-3)
    val gbt = new GBTRegressor().setFeaturesCol("features").setLabelCol("medv").setMaxIter(10)
    val tree = new DecisionTreeRegressor().setFeaturesCol("features").setLabelCol("medv")

    val data = vectorAssembler.transform(raw)
    data.cache()

    time {
      val gbmrParamGrid = new ParamGridBuilder()
        .addGrid(gmbr.learningRate, Array(0.1,0.01,0.05,1))
        .addGrid(gmbr.loss, Array("ls","lad"))
        .build()

      val gmbrCV = new CrossValidator()
        .setEstimator(gmbr)
        .setEvaluator(new RegressionEvaluator().setLabelCol(gmbr.getLabelCol).setPredictionCol(gmbr.getPredictionCol).setMetricName("rmse"))
        .setEstimatorParamMaps(gbmrParamGrid)
        .setNumFolds(5)
        .setParallelism(4)

      val gbmrCVModel = gmbrCV.fit(data)

      println(gbmrParamGrid.mkString(","))
      println(gbmrCVModel.avgMetrics.mkString(","))
      println(gbmrCVModel.bestModel.asInstanceOf[GBMRegressionModel].getLearningRate)
      println(gbmrCVModel.bestModel.asInstanceOf[GBMRegressionModel].getLoss)
      println(gbmrCVModel.bestModel.asInstanceOf[GBMRegressionModel].weights.mkString(","))
      println(gbmrCVModel.avgMetrics.min)

    }

    time {
      val paramGrid = new ParamGridBuilder()
        .addGrid(gbt.featureSubsetStrategy, Array("auto"))
        .addGrid(gbt.subsamplingRate, Array(0.3,0.5,0.7,1))
        .addGrid(gbt.lossType, Array("squared","absolute"))
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

    time {
      val paramGrid = new ParamGridBuilder()
        .build()

      val cv = new CrossValidator()
        .setEstimator(tree)
        .setEvaluator(new RegressionEvaluator().setLabelCol(gbt.getLabelCol).setPredictionCol(gbt.getPredictionCol).setMetricName("rmse"))
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(5)
        .setParallelism(4)

      val cvModel = cv.fit(data)

      println(cvModel.avgMetrics.mkString(","))
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
