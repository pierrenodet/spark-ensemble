package org.apache.spark.ml.classification

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions.{col, lit, rand, when}
import org.scalatest.FunSuite

class BoostingClassifierSuite extends FunSuite with DatasetSuiteBase {

  test("benchmark") {

    val raw = spark.read.format("libsvm").load("data/vehicle/vehicle.svm")

    val dc = new DecisionTreeClassifier()
    val bc =
      new BoostingClassifier().setBaseLearner(dc)
    val rf = new RandomForestClassifier().setNumTrees(30)

    val mce = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val data = raw
      .withColumn("label", col("label").minus(lit(1.0)))
      .withColumn("val", when(rand() > 0.9, true).otherwise(false))
    data.cache()

    time {
      val bcParamGrid = new ParamGridBuilder()
        .addGrid(bc.numBaseLearners, Array(50))
        .addGrid(bc.loss, Array("exponential"))
        .addGrid(bc.tol, Array(1E-9))
        .addGrid(bc.numRound, Array(3, 5, 8))
        .addGrid(bc.validationIndicatorCol, Array("val"))
        .addGrid(dc.maxDepth, Array(10, 20, 30))
        .build()

      val brCV = new CrossValidator()
        .setEstimator(bc)
        .setEvaluator(mce)
        .setEstimatorParamMaps(bcParamGrid)
        .setNumFolds(3)
        .setParallelism(4)

      val brCVModel = brCV.fit(data)

      println(brCVModel.avgMetrics.mkString(","))
      println(brCVModel.bestModel.asInstanceOf[BoostingClassificationModel].getLoss)
      println(brCVModel.bestModel.asInstanceOf[BoostingClassificationModel].numBaseModels)
      println(brCVModel.bestModel.asInstanceOf[BoostingClassificationModel].getNumRound)
      println(
        brCVModel.bestModel
          .asInstanceOf[BoostingClassificationModel]
          .models(0)
          .asInstanceOf[DecisionTreeClassificationModel]
          .getMaxDepth)
      println(brCVModel.bestModel.asInstanceOf[BoostingClassificationModel].weights.mkString(","))
      println(brCVModel.avgMetrics.max)

      val bm = brCVModel.bestModel.asInstanceOf[BoostingClassificationModel]
      bm.write.overwrite().save("/tmp/bonjour")
      val loaded = BoostingClassificationModel.load("/tmp/bonjour")
      assert(mce.evaluate(loaded.transform(data)) == mce.evaluate(bm.transform(data)))

    }

    time {
      val paramGrid = new ParamGridBuilder()
        .addGrid(rf.featureSubsetStrategy, Array("auto"))
        .addGrid(rf.subsamplingRate, Array(0.3, 0.7, 1))
        .addGrid(rf.maxDepth, Array(10, 20, 30))
        .build()

      val cv = new CrossValidator()
        .setEstimator(rf)
        .setEvaluator(mce)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(3)
        .setParallelism(4)

      val cvModel = cv.fit(data)

      println(cvModel.avgMetrics.mkString(","))
      println(cvModel.bestModel.asInstanceOf[RandomForestClassificationModel].getSubsamplingRate)
      println(cvModel.bestModel.asInstanceOf[RandomForestClassificationModel].getNumTrees)
      println(cvModel.bestModel.asInstanceOf[RandomForestClassificationModel].getMaxDepth)
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
