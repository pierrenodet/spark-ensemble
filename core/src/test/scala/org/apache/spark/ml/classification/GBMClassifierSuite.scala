package org.apache.spark.ml.classification

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.functions._
import org.scalatest.FunSuite

class GBMClassifierSuite extends FunSuite with DatasetSuiteBase {

  test("benchmark") {

    val raw = spark.read.format("libsvm").load("data/vehicle/vehicle.svm")
    val data = raw
      .withColumn("label", col("label").minus(lit(1.0)))
      .withColumn("val", when(rand() > 0.9, true).otherwise(false))

    val dr = new DecisionTreeRegressor()
    val gbmc = new GBMClassifier().setBaseLearner(dr).setNumBaseLearners(100)
    val rf = new RandomForestClassifier().setNumTrees(100)
    val dc = new DecisionTreeClassifier()

    val mce = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    data.cache()

    time {
      val gbmcParamGrid = new ParamGridBuilder()
        .addGrid(gbmc.learningRate, Array(0.1))
        // .addGrid(gbmc.numRound, Array(4))
        // .addGrid(gbmc.validationIndicatorCol, Array("val"))
        .addGrid(gbmc.instanceTrimmingRatio, Array(0.2))
        .addGrid(gbmc.sampleRatio, Array(0.8))
        .addGrid(gbmc.replacement, Array(true))
        .addGrid(gbmc.subspaceRatio, Array(0.8))
        .addGrid(gbmc.optimizedWeights, Array(true))
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
      println(gbmcCVModel.bestModel.asInstanceOf[GBMClassificationModel].consts.mkString(","))
      println(
        "weights : " + gbmcCVModel.bestModel
          .asInstanceOf[GBMClassificationModel]
          .weights
          .map(_.mkString(","))
          .mkString(";"))
      println(
        gbmcCVModel.bestModel.asInstanceOf[GBMClassificationModel].weights.size,
        gbmcCVModel.bestModel
          .asInstanceOf[GBMClassificationModel]
          .weights
          .map(_.size)
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
        .addGrid(rf.subsamplingRate, Array(0.7, 1))
        .addGrid(rf.maxDepth, Array(10))
        .build()

      val cv = new CrossValidator()
        .setEstimator(rf)
        .setEvaluator(mce)
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
        .addGrid(dc.maxDepth, Array(10))
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

  test("trivial taks") {
    val dr = new DecisionTreeRegressor()
    val br = new GBMClassifier()
      .setBaseLearner(dr)
      .setNumBaseLearners(20)
    val x = Seq.fill(100)(Vectors.dense(Array(1.0, 1.0))) ++ Seq.fill(100)(
      Vectors.dense(Array(0.0, 0.0)))
    val y = Seq.fill(100)(1.0) ++ Seq.fill(100)(0.0)
    import spark.implicits._
    val data = spark.sparkContext.parallelize(x.zip(y)).toDF("features", "label")
    val learned = br.fit(data)
    learned.transform(data).show()
  }

}
