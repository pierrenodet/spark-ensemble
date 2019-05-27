package org.apache.spark.ml.classification

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.scalatest.FunSuite

class GBMClassifierSuite extends FunSuite with DatasetSuiteBase {

  test("benchmark") {

    val raw = spark.read.format("libsvm").load("src/test/resources/data/vehicle/train.scale")

    val br = new GBMClassifier().setBaseLearner(new DecisionTreeRegressor()).setMaxIter(10)
    val rf = new RandomForestClassifier().setNumTrees(10)
    val dc = new DecisionTreeClassifier()

    val data = raw
    data.cache()

    time {
      val brParamGrid = new ParamGridBuilder()
        .addGrid(br.learningRate, Array(0.1, 1.0))
        .addGrid(br.maxIter, Array(1, 10))
        .build()

      val brCV = new CrossValidator()
    .setEstimator(br)
    .setEvaluator(
      new MulticlassClassificationEvaluator().setMetricName("accuracy").setLabelCol(br.getLabelCol).setPredictionCol(br.getPredictionCol)
    )
    .setEstimatorParamMaps(brParamGrid)
    .setNumFolds(5)
    .setParallelism(4)

      val brCVModel = brCV.fit(data)

      println(brCVModel.avgMetrics.mkString(","))
      println(brCVModel.bestModel.asInstanceOf[GBMClassificationModel].getLearningRate)
      println(brCVModel.bestModel.asInstanceOf[GBMClassificationModel].weights.map(_.mkString(",")).mkString(","))
      println(brCVModel.avgMetrics.max)
    }

    time {
      val paramGrid = new ParamGridBuilder()
        .addGrid(rf.featureSubsetStrategy, Array("auto"))
        .addGrid(rf.subsamplingRate, Array(0.3, 0.7, 1))
        .addGrid(rf.maxDepth, Array(1, 10, 20))
        .build()

      val cv = new CrossValidator()
        .setEstimator(rf)
        .setEvaluator(new MulticlassClassificationEvaluator().setMetricName("accuracy").setLabelCol(rf.getLabelCol).setPredictionCol(rf.getPredictionCol))
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(5)
        .setParallelism(4)

      val cvModel = cv.fit(data)

      println(cvModel.avgMetrics.mkString(","))
      print(cvModel.bestModel.asInstanceOf[RandomForestClassificationModel].getSubsamplingRate + ",")
      println(cvModel.bestModel.asInstanceOf[RandomForestClassificationModel].getMaxDepth)
      println(cvModel.avgMetrics.max)
    }


    time {
      val paramGrid = new ParamGridBuilder()
        .addGrid(dc.maxDepth, Array(1, 10, 20))
        .build()

      val cv = new CrossValidator()
        .setEstimator(dc)
        .setEvaluator(new MulticlassClassificationEvaluator().setMetricName("accuracy").setLabelCol(dc.getLabelCol).setPredictionCol(dc.getPredictionCol))
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(5)
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
