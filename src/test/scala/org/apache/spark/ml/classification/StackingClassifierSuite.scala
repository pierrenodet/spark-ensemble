package org.apache.spark.ml.classification

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.scalatest.FunSuite

class StackingClassifierSuite extends FunSuite with DatasetSuiteBase {

  test("benchmark") {

    val raw = spark.read.format("libsvm").load("src/test/resources/data/vehicle/train.scale")

    val sr = new StackingClassifier().setStacker(new DecisionTreeClassifier()).setBaseLearners(Array(new DecisionTreeClassifier(),new RandomForestClassifier())).setParallelism(4)
    val rf = new RandomForestClassifier().setNumTrees(10)

    val data = raw
    data.cache()

    time {
      val srParamGrid = new ParamGridBuilder()
        .build()

      val srCV = new CrossValidator()
        .setEstimator(sr)
        .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol(sr.getLabelCol).setPredictionCol(sr.getPredictionCol))
        .setEstimatorParamMaps(srParamGrid)
        .setNumFolds(5)
        .setParallelism(4)

      val srCVModel = srCV.fit(data)

      println(srCVModel.avgMetrics.mkString(","))
      println(srCVModel.avgMetrics.max)

    }

    time {
      val paramGrid = new ParamGridBuilder()
        .build()

      val cv = new CrossValidator()
        .setEstimator(rf)
        .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol(rf.getLabelCol).setPredictionCol(rf.getPredictionCol))
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
