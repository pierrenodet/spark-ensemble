package org.apache.spark.ml.classification

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions._
import org.scalatest.FunSuite

class GBMClassifierSuite extends FunSuite with DatasetSuiteBase {

  test("benchmark") {

    val raw = spark.read.format("libsvm").load("data/vehicle/vehicle.svm")
    val data = raw
      .withColumn("label", col("label").minus(lit(1.0)))
      .withColumn("val", when(rand() > 0.8, true).otherwise(false))

    val dr = new DecisionTreeRegressor()
    val gbmc = new GBMClassifier().setBaseLearner(dr).setNumBaseLearners(10)
    val rf = new RandomForestClassifier().setNumTrees(10)
    val dc = new DecisionTreeClassifier()

    val mce = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    data.cache()

    time {
      val gbmcParamGrid = new ParamGridBuilder()
        .addGrid(gbmc.learningRate, Array(0.1))
        .addGrid(gbmc.tol, Array(1E-9))
        .addGrid(gbmc.numRound, Array(3))
        .addGrid(gbmc.validationIndicatorCol, Array("val"))
        .addGrid(gbmc.sampleRatio, Array(0.8))
        .addGrid(gbmc.replacement, Array(true))
        .addGrid(gbmc.subspaceRatio, Array(1.0))
        .addGrid(gbmc.optimizedWeights, Array(false))
        .addGrid(gbmc.loss, Array("divergence"))
        .addGrid(dr.maxDepth, Array(10))
        .build()

      val gbmcCV = new CrossValidator()
        .setEstimator(gbmc)
        .setEvaluator(mce)
        .setEstimatorParamMaps(gbmcParamGrid)
        .setNumFolds(3)
        .setParallelism(4)

      val gbmcCVModel = gbmcCV.fit(data)

      println(gbmcCVModel.avgMetrics.mkString(","))
      println(gbmcCVModel.bestModel.asInstanceOf[GBMClassificationModel].getLearningRate)
      println(gbmcCVModel.bestModel.asInstanceOf[GBMClassificationModel].getNumBaseLearners)
      println(gbmcCVModel.bestModel.asInstanceOf[GBMClassificationModel].getLoss)
      println(gbmcCVModel.bestModel.asInstanceOf[GBMClassificationModel].getOptimizedWeights)
      println(gbmcCVModel.bestModel.asInstanceOf[GBMClassificationModel].getSampleRatio)
      println(gbmcCVModel.bestModel.asInstanceOf[GBMClassificationModel].getReplacement)
      println(gbmcCVModel.bestModel.asInstanceOf[GBMClassificationModel].getSubspaceRatio)
      println(
        "weights : " + gbmcCVModel.bestModel
          .asInstanceOf[GBMClassificationModel]
          .weights
          .mkString(","))
      println(gbmcCVModel.avgMetrics.max)

      val bm = gbmcCVModel.bestModel.asInstanceOf[GBMClassificationModel]
      bm.write.overwrite().save("/tmp/bonjour")
      val loaded = GBMClassificationModel.load("/tmp/bonjour")
      assert(mce.evaluate(loaded.transform(data)) == mce.evaluate(bm.transform(data)))

    }

    time {
      val paramGrid = new ParamGridBuilder()
        .addGrid(rf.featureSubsetStrategy, Array("auto"))
        .addGrid(rf.subsamplingRate, Array(0.3, 0.7, 1))
        .addGrid(rf.maxDepth, Array(10))
        .build()

      val cv = new CrossValidator()
        .setEstimator(rf)
        .setEvaluator(
          new MulticlassClassificationEvaluator()
            .setMetricName("accuracy")
            .setLabelCol(rf.getLabelCol)
            .setPredictionCol(rf.getPredictionCol))
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(3)
        .setParallelism(4)

      val cvModel = cv.fit(data)

      println(cvModel.avgMetrics.mkString(","))
      print(
        cvModel.bestModel.asInstanceOf[RandomForestClassificationModel].getSubsamplingRate + ",")
      println(cvModel.bestModel.asInstanceOf[RandomForestClassificationModel].getMaxDepth)
      println(cvModel.avgMetrics.max)
    }

    time {
      val paramGrid = new ParamGridBuilder()
        .addGrid(dc.maxDepth, Array(1, 10, 20))
        .build()

      val cv = new CrossValidator()
        .setEstimator(dc)
        .setEvaluator(mce)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(3)
        .setParallelism(4)

      val cvModel = cv.fit(data)

      println(cvModel.avgMetrics.mkString(","))
      println(cvModel.bestModel.asInstanceOf[DecisionTreeClassificationModel].getMaxDepth)
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
