package org.apache.spark.ml.regression

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.scalatest.FunSuite

class StackingRegressorSuite extends FunSuite with DatasetSuiteBase {

  test("benchmark") {

    val raw = spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm")

    val sr = new StackingRegressor()
      .setStacker(new DecisionTreeRegressor())
      .setBaseLearners(Array(new DecisionTreeRegressor(), new RandomForestRegressor()))
      .setParallelism(2)
    val rf = new RandomForestRegressor()

    val re = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val data = raw
    data.cache().first()

    time {
      val srParamGrid = new ParamGridBuilder()
        .build()

      val srCV = new CrossValidator()
        .setEstimator(sr)
        .setEvaluator(re)
        .setEstimatorParamMaps(srParamGrid)
        .setNumFolds(3)
        .setParallelism(4)

      val srCVModel = srCV.fit(data)

      println(srCVModel.avgMetrics.min)

      val bm = srCVModel.bestModel.asInstanceOf[StackingRegressionModel]
      bm.write.overwrite().save("/tmp/bonjour")
      val loaded = StackingRegressionModel.load("/tmp/bonjour")
      assert(re.evaluate(loaded.transform(data)) == re.evaluate(bm.transform(data)))


    }

    time {
      val paramGrid = new ParamGridBuilder()
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
