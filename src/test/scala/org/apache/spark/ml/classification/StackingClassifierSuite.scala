package org.apache.spark.ml.classification

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.scalatest.FunSuite

class StackingClassifierSuite extends FunSuite with DatasetSuiteBase {

  test("benchmark") {

    val raw = spark.read.option("header", "true").option("inferSchema", "true").csv("src/test/resources/data/iris/train.csv")

    val vectorAssembler = new VectorAssembler().setInputCols(raw.columns.filter(x => !x.equals("class"))).setOutputCol("features")
    val stringIndexer = new StringIndexer().setInputCol("class").setOutputCol("label")
    val sr = new StackingClassifier().setStacker(new DecisionTreeClassifier()).setLearners(Array(new DecisionTreeClassifier(),new RandomForestClassifier())).setParallelism(4)
    val rf = new RandomForestClassifier().setNumTrees(10)

    val data = stringIndexer.fit(raw).transform(vectorAssembler.transform(raw))
    data.count()

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
      println(srCVModel.avgMetrics.min)

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
