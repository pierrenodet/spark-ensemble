package org.apache.spark.ml.regression

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.scalatest.FunSuite

class BaggingRegressorSuite extends FunSuite with DatasetSuiteBase {

  test("benchmark") {

    val raw = spark.read.format("libsvm").load("src/test/resources/data/cpusmall/cpusmall.svm")

    val bl = new DecisionTreeRegressor()
    val br = new BaggingRegressor()
      .setBaseLearner(bl)
      .setParallelism(4)
    val rf =
      new RandomForestRegressor().setNumTrees(10)

    val re = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val data = raw
    data.cache().first()

    time {
      val brParamGrid = new ParamGridBuilder()
        .addGrid(br.subspaceRatio, Array(0.7, 1))
        .addGrid(br.numBaseLearners, Array(10))
        .addGrid(br.replacement, Array(x = true))
        .addGrid(br.sampleRatio, Array(0.7, 1))
        .addGrid(bl.maxDepth, Array(10))
        .addGrid(bl.maxBins, Array(30, 40))
        .build()

      val brCV = new CrossValidator()
        .setEstimator(br)
        .setEvaluator(re)
        .setEstimatorParamMaps(brParamGrid)
        .setNumFolds(3)
        .setParallelism(4)

      val brCVModel = brCV.fit(data)

      println(brCVModel.avgMetrics.mkString(","))
      print(brCVModel.bestModel.asInstanceOf[BaggingRegressionModel].getReplacement + ",")
      print(brCVModel.bestModel.asInstanceOf[BaggingRegressionModel].getSampleRatio + ",")
      print(brCVModel.bestModel.asInstanceOf[BaggingRegressionModel].getSubspaceRatio + ",")
      print(
        brCVModel.bestModel
          .asInstanceOf[BaggingRegressionModel]
          .models(0)
          .asInstanceOf[DecisionTreeRegressionModel]
          .getMaxDepth + ",")
      println(
        brCVModel.bestModel
          .asInstanceOf[BaggingRegressionModel]
          .models(0)
          .asInstanceOf[DecisionTreeRegressionModel]
          .getMaxBins)
      println(brCVModel.avgMetrics.min)

      val bm = brCVModel.bestModel.asInstanceOf[BaggingRegressionModel]
      bm.write.overwrite().save("/tmp/bonjour")
      val loaded = BaggingRegressionModel.load("/tmp/bonjour")
      assert(re.evaluate(loaded.transform(data)) == re.evaluate(bm.transform(data)))

    }

    time {
      val paramGrid = new ParamGridBuilder()
        .addGrid(rf.featureSubsetStrategy, Array("auto"))
        .addGrid(rf.numTrees, Array(10))
        .addGrid(rf.subsamplingRate, Array(0.3, 0.7, 1))
        .build()

      val cv = new CrossValidator()
        .setEstimator(rf)
        .setEvaluator(re)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(3)
        .setParallelism(4)

      val cvModel = cv.fit(data)

      println(cvModel.avgMetrics.mkString(","))
      println(cvModel.bestModel.asInstanceOf[RandomForestRegressionModel].getSubsamplingRate)
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
