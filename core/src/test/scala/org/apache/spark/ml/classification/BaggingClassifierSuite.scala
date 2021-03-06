package org.apache.spark.ml.classification

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.scalatest.FunSuite

class BaggingClassifierSuite extends FunSuite with DatasetSuiteBase {

  test("benchmark") {

    val raw = spark.read.format("libsvm").load("data/vehicle/vehicle.svm")

    val br = new BaggingClassifier()
      .setBaseLearner(new DecisionTreeClassifier())
      .setNumBaseLearners(10)
      .setParallelism(4)
    val rf = new RandomForestClassifier().setNumTrees(10)

    val mce = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val data = raw
    data.cache()

    time {
      val brParamGrid = new ParamGridBuilder()
        .addGrid(br.subspaceRatio, Array(0.7, 1))
        .addGrid(br.replacement, Array(true, false))
        .addGrid(br.sampleRatio, Array(0.7, 1))
        .build()

      val brCV = new CrossValidator()
        .setEstimator(br)
        .setEvaluator(mce)
        .setEstimatorParamMaps(brParamGrid)
        .setNumFolds(5)
        .setParallelism(4)

      val brCVModel = brCV.fit(data)

      println(brCVModel.avgMetrics.mkString(","))
      print(brCVModel.bestModel.asInstanceOf[BaggingClassificationModel].getReplacement + ",")
      print(brCVModel.bestModel.asInstanceOf[BaggingClassificationModel].getSampleRatio + ",")
      println(brCVModel.bestModel.asInstanceOf[BaggingClassificationModel].getSubspaceRatio)
      println(brCVModel.avgMetrics.max)

      val bm = brCVModel.bestModel.asInstanceOf[BaggingClassificationModel]
      bm.write.overwrite().save("/tmp/bonjour")
      val loaded = BaggingClassificationModel.load("/tmp/bonjour")
      assert(mce.evaluate(loaded.transform(data)) == mce.evaluate(bm.transform(data)))

    }

    time {
      val paramGrid = new ParamGridBuilder()
        .addGrid(rf.subsamplingRate, Array(0.3, 0.7, 1))
        .build()

      val cv = new CrossValidator()
        .setEstimator(rf)
        .setEvaluator(mce)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(5)
        .setParallelism(4)

      val cvModel = cv.fit(data)

      println(cvModel.avgMetrics.mkString(","))
      println(cvModel.bestModel.asInstanceOf[RandomForestClassificationModel].getSubsamplingRate)
      println(cvModel.avgMetrics.max)
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
