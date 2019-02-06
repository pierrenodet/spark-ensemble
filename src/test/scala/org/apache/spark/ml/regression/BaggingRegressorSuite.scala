package org.apache.spark.ml.regression

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.scalatest.FunSuite

class BaggingRegressorSuite extends FunSuite with DatasetSuiteBase {

  test("benchmark") {

    val raw = spark.read.option("header", "true").option("inferSchema", "true").csv("src/test/resources/data/bostonhousing/train.csv")
    val test = spark.read.option("header", "true").option("inferSchema", "true").csv("src/test/resources/data/bostonhousing/test.csv")

    val vectorAssembler = new VectorAssembler().setInputCols(raw.columns.filter(x => !x.equals("ID") && !x.equals("medv"))).setOutputCol("features")
    val br = new BaggingRegressor().setBaseLearner(new DecisionTreeRegressor()).setFeaturesCol("features").setLabelCol("medv").setMaxIter(10).setParallelism(4)
    val rf = new RandomForestRegressor().setFeaturesCol("features").setLabelCol("medv").setNumTrees(10)

    val data = vectorAssembler.transform(raw)
    data.count()

    time {
      val brParamGrid = new ParamGridBuilder()
        .addGrid(br.sampleRatioFeatures, Array(0.3,0.7,1))
        .addGrid(br.replacementFeatures, Array(x = false))
        .addGrid(br.replacement, Array(true, false))
        .addGrid(br.sampleRatio, Array(0.3, 0.7, 1))
        .build()

      val brCV = new CrossValidator()
        .setEstimator(br)
        .setEvaluator(new RegressionEvaluator().setLabelCol(br.getLabelCol).setPredictionCol(br.getPredictionCol).setMetricName("rmse"))
        .setEstimatorParamMaps(brParamGrid)
        .setNumFolds(5)
        .setParallelism(4)

      val brCVModel = brCV.fit(data)

      println(brCVModel.avgMetrics.mkString(","))
      print(brCVModel.bestModel.asInstanceOf[BaggingRegressionModel].getReplacement + ",")
      print(brCVModel.bestModel.asInstanceOf[BaggingRegressionModel].getSampleRatio + ",")
      print(brCVModel.bestModel.asInstanceOf[BaggingRegressionModel].getReplacementFeatures + ",")
      println(brCVModel.bestModel.asInstanceOf[BaggingRegressionModel].getSampleRatioFeatures)
      println(brCVModel.avgMetrics.min)

    }

    time {
      val paramGrid = new ParamGridBuilder()
        .addGrid(rf.featureSubsetStrategy, Array("auto"))
        .addGrid(rf.numTrees, Array(10))
        .addGrid(rf.subsamplingRate, Array(0.3, 0.7, 1))
        .addGrid(rf.maxDepth, Array(1, 10, 20))
        .build()

      val cv = new CrossValidator()
        .setEstimator(rf)
        .setEvaluator(new RegressionEvaluator().setLabelCol(rf.getLabelCol).setPredictionCol(rf.getPredictionCol).setMetricName("rmse"))
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(5)
        .setParallelism(4)

      val cvModel = cv.fit(data)

      println(cvModel.avgMetrics.mkString(","))
      print(cvModel.bestModel.asInstanceOf[RandomForestRegressionModel].getSubsamplingRate + ",")
      println(cvModel.bestModel.asInstanceOf[RandomForestRegressionModel].getMaxDepth)
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
